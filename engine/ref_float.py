"""Token-at-a-time float reference of the trained model, written from scratch in numpy. Its only job is to
prove the structure is understood: it must reproduce the training code's scores."""
import numpy as np
M32 = 0xFFFFFFFF


def mix32(h):
    h &= M32
    h = ((h ^ (h >> 15)) * 0x2C1B3C6D) & M32
    h = ((h ^ (h >> 12)) * 0x297A2D39) & M32
    return h ^ (h >> 15)


def inj_index(spec, tok, prev, prev2):
    h = tok * 0x9E3779B1 + prev * 0x85EBCA77 + spec["salt"] * 0x27D4EB2F
    if spec["kind"] == "trigram":
        h += prev2 * 0xC2B2AE3D
    return mix32(h) % spec["rows"]


STATS = None


def S(name, a):
    """Record value ranges when STATS is a dict (used once, to choose fixed-point formats)."""
    if STATS is not None:
        m = STATS.setdefault(name, [0.0, 0.0, 0])
        m[0] = max(m[0], float(np.abs(a).max())); m[1] += float((np.asarray(a) ** 2).mean()); m[2] += 1
    return a


def rms(x, g):
    S("stream", x)
    return S("normed", g * x / np.sqrt((x * x).mean() + 1e-5))


class FloatRef:
    def __init__(self, P):
        self.P = P
        c = P["cfg"]
        self.c = c
        hk = c["DK"] // c["NH"]
        self.freq = 1.0 / (10000 ** (np.arange(0, hk, 2) / hk))
        self.reset()

    def reset(self):
        c = self.c
        self.toks = []
        self.keys = {l: [] for l in self.P["att_layers"]}       # per position: (NH, hk) of +-1
        self.vals = {l: [] for l in self.P["att_layers"]}       # per position: (NH, DV/NH)
        self.hist = {l: [] for l in range(c["L"]) if l not in self.P["att_layers"]}   # last 3 normalised inputs
        self.ema = {l: np.zeros((3, c["D"])) for l in range(c["L"]) if l not in self.P["att_layers"]}

    def rope(self, x, pos):
        ang = pos * self.freq
        co, si = np.cos(ang), np.sin(ang)
        a, b = x[..., 0::2], x[..., 1::2]
        return np.stack([a * co - b * si, a * si + b * co], -1).reshape(x.shape)

    def step(self, tok):
        P, c = self.P, self.c
        t = len(self.toks)
        prev = self.toks[-1] if t >= 1 else 0
        prev2 = self.toks[-2] if t >= 2 else 0
        self.toks.append(tok)
        x = P["emb"][tok].copy()
        NH, hk, hv = c["NH"], c["DK"] // c["NH"], c["DV"] // c["NH"]
        a_last = None
        for l, Ly in enumerate(P["layers"]):
            for spec in P["inj"]:
                if spec["layer"] == l:
                    i = inj_index(spec, tok, prev, prev2)
                    x = x + spec["t"][i] * spec["step"][i]
            xn = rms(x, Ly["n1"])
            if l in P["att_layers"]:
                S("q_sum", Ly["q"] @ xn); S("k_sum", Ly["k"] @ xn); S("v_sum", Ly["v"] @ xn)
                q = self.rope((Ly["q"] @ xn * Ly["q_s"]).reshape(NH, hk), t)
                q = q / np.sqrt((q * q).mean(-1, keepdims=True) + 1e-5)
                k = self.rope((Ly["k"] @ xn * Ly["k_s"] * Ly["k_g"]).reshape(NH, hk), t)
                self.keys[l].append(np.where(k >= 0, 1.0, -1.0))
                self.vals[l].append(S("v", Ly["v"] @ xn * Ly["v_s"]).reshape(NH, hv))
                if len(self.keys[l]) > c["T"]:                               # ring: only the last T positions are kept
                    self.keys[l].pop(0); self.vals[l].pop(0)
                K = np.stack(self.keys[l])                                   # (kept, NH, hk)
                Vv = np.stack(self.vals[l])
                n = len(K); first = t - (n - 1)                              # absolute position of the oldest kept one
                out = np.zeros((NH, hv)); a_last = None
                for h in range(NH):
                    s = K[:, h] @ q[h] * np.exp(Ly["temp"][h]) / np.sqrt(hk) + Ly["dbias"][h, (n - 1) - np.arange(n)]
                    S("score", s); S("score_spread_top8", np.sort(s)[-1] - np.sort(s)[-min(8, len(s))])
                    if n > c["topk"]:
                        kth = np.sort(s)[-c["topk"]]
                        s = np.where(s < kth, -np.inf, s)
                    a = np.exp(s - s.max()); a /= a.sum()
                    out[h] = a @ Vv[:, h]
                    if h == 0:
                        a_last = (first, a)
                S("att_mix", out); x = x + S("att_out", Ly["o"] @ out.reshape(-1) * Ly["o_s"])
            else:
                H = self.hist[l]
                y = np.zeros(c["D"])
                for i in range(3):
                    if len(H) > i:
                        y += Ly["taps"][i] * H[-1 - i]
                r = np.array([0.5, 0.8, 0.95])[:, None]
                self.ema[l] = r * self.ema[l] + (1 - r) * xn
                y += (Ly["avg"] * self.ema[l]).sum(0)
                H.append(xn)
                x = x + y
            xn2 = rms(x, Ly["n2"])
            e = Ly["route"][tok]
            S("w1_sum", Ly["w1"][e] @ xn2); S("pre", Ly["w1"][e] @ xn2 * Ly["w1_s"][e])
            hid = S("hid", np.maximum(Ly["w1"][e] @ xn2 * Ly["w1_s"][e], 0) ** 2)
            S("w2_sum", Ly["w2"][e] @ hid); x = x + S("mlp_out", Ly["w2"][e] @ hid * Ly["w2_s"][e])
        h = rms(x, P["nf"])
        S("head_sum", P["head"] @ h); logits = S("logits", P["head"] @ h * P["head_s"] * P["head_g"])
        first, a = a_last
        for i, w in enumerate(a):                                            # attended earlier positions vote for their successor
            if w != 0 and first + i < t:
                logits[self.toks[first + i + 1]] += P["copy_gain"] * w
        return logits
