"""Integer-only reference of Koko, token at a time. Every operation here is one the GBA will do:
adds, 32x32->64 multiplies, shifts, table lookups, one integer square root and a few divisions.
No floats after quantize(). The C engine must reproduce these numbers exactly.

Formats: stream, normalised stream, projections, hidden units, scores and output scores are Q12 in int32.
"""
import numpy as np
from ref_float import inj_index

Q = 12
ONE = 1 << Q
EXP_BITS = 7                      # softmax table resolution: 1/128 in the exponent
EXP_N = 16 << EXP_BITS            # covers score gaps up to 16.0; beyond that the weight is 0
TOPN = 40                         # sampling pool
EMA_K = (512, 205, 51)            # running averages: ema += rs((x - ema) * k, 10), i.e. keep 0.5, 0.8, 0.95


def rs(x, n):
    """Rounding arithmetic shift right (works on Python ints and numpy int64)."""
    return (x + (1 << (n - 1))) >> n


def isqrt(v):
    v = int(v); r = 0; b = 1 << 62
    while b > v: b >>= 2
    while b:
        if v >= r + b: v -= r + b; r = (r >> 1) + b
        else: r >>= 1
        b >>= 2
    return r


def fits32(a, what):
    a = np.asarray(a)
    assert a.size == 0 or (int(a.max()) < 2 ** 31 and int(a.min()) >= -2 ** 31), f"{what} overflows int32"
    return a


def quantize(P, emb_bits=6):
    """Float parameters -> the integers that go on the cartridge."""
    c = P["cfg"]
    NH, hk = c["NH"], c["DK"] // c["NH"]
    I = {"cfg": dict(c), "att_layers": list(P["att_layers"]), "layers": [], "inj": []}
    e = P["emb"]
    lev = (1 << (emb_bits - 1)) - 1                                            # 127 at 8 bits, 31 at 6
    step = np.abs(e).max(1) / lev
    I["emb"] = np.round(e / step[:, None]).astype(np.int64)                    # signed, emb_bits per value
    I["emb_step"] = np.round(step * 65536).astype(np.int64)                    # Q16, fits uint16
    freq = 1.0 / (10000 ** (np.arange(0, hk, 2) / hk))
    I["rope_inc"] = np.round(freq / (2 * np.pi) * 2 ** 32).astype(np.int64)      # turns per position, as a 32-bit fraction
    I["sin"] = np.round(np.sin(2 * np.pi * np.arange(4096) / 4096) * 16384).astype(np.int64)   # one turn, Q14
    I["exp"] = np.round(np.exp(-np.arange(EXP_N) / (1 << EXP_BITS)) * 65535).astype(np.int64)   # uint16
    for l, Ly in enumerate(P["layers"]):
        J = {"n1": np.round(Ly["n1"] * ONE).astype(np.int64), "n2": np.round(Ly["n2"] * ONE).astype(np.int64)}
        if l in P["att_layers"]:
            for nm in "qkvo":
                J[nm] = Ly[nm].astype(np.int64)
            J["k_g"] = np.round(Ly["k_g"] * ONE).astype(np.int64)             # the key's matrix scale cannot change a sign
            J["v_m"] = int(round(float(Ly["v_s"]) * 65536)); J["o_m"] = int(round(float(Ly["o_s"]) * 65536))
            J["c"] = np.round(np.exp(Ly["temp"]) / np.sqrt(hk) * 65536).astype(np.int64)
            J["dbias"] = np.round(Ly["dbias"] * ONE).astype(np.int64)
        else:
            J["taps"] = np.round(Ly["taps"] * ONE).astype(np.int64); J["avg"] = np.round(Ly["avg"] * ONE).astype(np.int64)
        J["w1"], J["w2"] = Ly["w1"].astype(np.int64), Ly["w2"].astype(np.int64)
        J["w1_m"] = np.round(Ly["w1_s"] * 65536).astype(np.int64); J["w2_m"] = np.round(Ly["w2_s"] * 65536).astype(np.int64)
        J["route"] = Ly["route"]
        I["layers"].append(J)
    for sp in P["inj"]:
        I["inj"].append({**{k: sp[k] for k in ("layer", "kind", "rows", "salt")}, "t": sp["t"].astype(np.int64),
                         "step": np.round(sp["step"] * ONE).astype(np.int64)})
    I["nf"] = np.round(P["nf"] * ONE).astype(np.int64)
    I["head"] = P["head"].astype(np.int64)
    I["head_m"] = np.round(P["head_s"] * P["head_g"] * 65536).astype(np.int64)  # per-row, fits uint16
    I["copy_gain"] = int(round(P["copy_gain"] * ONE))
    return I


class IntRef:
    def __init__(self, I):
        self.I, self.c = I, I["cfg"]
        self.reset()

    def reset(self):
        c, I = self.c, self.I
        self.toks = []
        self.keys = {l: {} for l in I["att_layers"]}       # ring slot -> (NH, hk) of +-1
        self.vals = {l: {} for l in I["att_layers"]}       # ring slot -> (int8 mantissas (DV,), shift)
        mix = [l for l in range(c["L"]) if l not in I["att_layers"]]
        self.hist = {l: [] for l in mix}
        self.ema = {l: np.zeros((3, c["D"]), dtype=np.int64) for l in mix}
        self.a_last = None
        self.nvalid = 0                                    # how many of the most recent positions are usable
        self._mark = None

    def mark(self):
        """Remember this point so the tokens fed after it can be taken back (the user edits their message)."""
        self._mark = (len(self.toks), self.nvalid, {l: [h.copy() for h in H] for l, H in self.hist.items()},
                      {l: e.copy() for l, e in self.ema.items()}, self.a_last)

    def rewind(self):
        """Back to the mark. Ring slots written since then are lost, so the usable window shrinks by that many
        (it grows again as tokens arrive)."""
        n0, v0, hist, ema, a = self._mark
        k = len(self.toks) - n0
        self.toks = self.toks[:n0]
        self.nvalid = max(0, min(v0, self.c["T"] - k))
        self.hist = {l: [h.copy() for h in H] for l, H in hist.items()}
        self.ema = {l: e.copy() for l, e in ema.items()}
        self.a_last = a

    def norm(self, x, g):
        """Q12 of g * x / rms(x)."""
        D = len(x)
        ss = int((x * x).sum())
        r = max(isqrt(ss // D + 168), 16)                  # 168 = 1e-5 in Q24
        inv = (1 << 40) // r
        t = fits32(rs(x * inv, 28), "norm t")
        return fits32(rs(t * g, Q), "normed")

    def rope(self, v, t):
        """Rotate pairs by the position's phase; positions count up without limit (the phase wraps)."""
        ph = (t * self.I["rope_inc"]) & 0xFFFFFFFF
        idx = (((ph + (1 << 19)) & 0xFFFFFFFF) >> 20) & 4095
        si, co = self.I["sin"][idx], self.I["sin"][(idx + 1024) & 4095]
        a, b = v[..., 0::2], v[..., 1::2]
        return np.stack([rs(a * co - b * si, 14), rs(a * si + b * co, 14)], -1).reshape(v.shape)

    def step(self, tok):
        I, c = self.I, self.c
        t = len(self.toks)
        T = c["T"]; slot = t % T; n = min(t + 1, T)                    # ring: the last T positions are kept
        self.nvalid = nv = min(self.nvalid + 1, T)
        prev = self.toks[-1] if t >= 1 else 0
        prev2 = self.toks[-2] if t >= 2 else 0
        self.toks.append(tok)
        NH, hk, hv = c["NH"], c["DK"] // c["NH"], c["DV"] // c["NH"]
        x = rs(I["emb"][tok] * int(I["emb_step"][tok]), 16 - Q)
        for l, J in enumerate(I["layers"]):
            for sp in I["inj"]:
                if sp["layer"] == l:
                    i = inj_index(sp, tok, prev, prev2)
                    x = x + sp["t"][i] * int(sp["step"][i])
            xn = self.norm(x, J["n1"])
            if l in I["att_layers"]:
                qr = self.rope(fits32(J["q"] @ xn, "q sum").reshape(NH, hk), t)
                kr = self.rope(rs(fits32(J["k"] @ xn, "k sum") * J["k_g"], Q).reshape(NH, hk), t)
                self.keys[l][slot] = np.where(kr >= 0, 1, -1)
                v = rs(fits32(J["v"] @ xn, "v sum") * J["v_m"], 16)
                sh = 0
                while (int(np.abs(v).max()) >> sh) > 127: sh += 1
                m8 = rs(v, sh) if sh else v                            # int8 mantissas and one shift per position
                assert np.abs(m8).max() <= 127 + (1 if sh else 0)
                self.vals[l][slot] = (np.clip(m8, -127, 127), sh)
                K = np.stack([self.keys[l][s_] for s_ in range(n)])
                dist = (slot - np.arange(n)) % T
                mix = np.zeros((NH, hv), dtype=np.int64)
                for h in range(NH):
                    r = max(isqrt(int((qr[h] * qr[h]).sum()) // hk + 168), 1)
                    m = (int(J["c"][h]) << 20) // r                      # Q24 of sharpness / rms
                    qs = fits32(rs(qr[h] * m, 24), "scaled query")
                    s = K[:, h] @ qs + J["dbias"][h, dist]
                    top = []                                             # up to 8 best, lower ring slot wins a tie
                    for j in range(n):
                        if dist[j] >= nv: continue                       # slot lost to a rewind
                        if len(top) == c["topk"] and s[j] <= s[top[-1]]: continue
                        p = len(top) if len(top) < c["topk"] else c["topk"] - 1
                        if len(top) < c["topk"]: top.append(j)
                        while p > 0 and s[top[p - 1]] < s[j]: top[p] = top[p - 1]; p -= 1
                        top[p] = j
                    w = []
                    for j in top:
                        d = int(s[top[0]] - s[j]) >> (Q - EXP_BITS)
                        w.append(int(I["exp"][d]) if d < EXP_N else 0)
                    R = (1 << 31) // sum(w)
                    pj = [(wi * R) >> 16 for wi in w]                     # Q15 weights
                    acc = np.zeros(hv, dtype=np.int64)
                    for j, p in zip(top, pj):
                        mant, shj = self.vals[l][j]
                        acc += (p * mant[h * hv:(h + 1) * hv]) << shj
                    mix[h] = rs(acc, 15)
                    if h == 0:
                        self.a_last = list(zip(top, pj))
                x = x + rs(fits32(J["o"] @ mix.reshape(-1), "o sum") * J["o_m"], 16)
            else:
                H = self.hist[l]
                y = np.zeros(c["D"], dtype=np.int64)
                for i in range(3):
                    if len(H) > i:
                        y += J["taps"][i] * H[-1 - i]
                for i in range(3):
                    self.ema[l][i] = self.ema[l][i] + rs((xn - self.ema[l][i]) * EMA_K[i], 10)
                    y += J["avg"][i] * self.ema[l][i]
                H.append(xn)
                if len(H) > 3: H.pop(0)
                x = x + rs(y, Q)
            xn2 = self.norm(x, J["n2"])
            e = int(J["route"][tok])
            pre = rs(fits32(J["w1"][e] @ xn2, "w1 sum") * int(J["w1_m"][e]), 16)
            pre = np.maximum(pre, 0)
            hid = fits32(rs(pre * pre, Q), "hidden")
            x = fits32(x + rs(fits32(J["w2"][e] @ hid, "w2 sum") * int(J["w2_m"][e]), 16), "stream")
        self.h = self.norm(x, I["nf"])
        return self.h

    def tok_at(self, s):
        """Token in ring slot s (the most recent position that landed there)."""
        t = len(self.toks) - 1
        return self.toks[t - ((t % self.c["T"] - s) % self.c["T"])]

    def scores(self, rows=None):
        """Q12 output scores for the given token rows (all rows if None), including the copy votes."""
        I = self.I
        rows = np.arange(self.c["V"]) if rows is None else np.asarray(rows)
        lg = rs(fits32(I["head"][rows] @ self.h, "head sum") * I["head_m"][rows], 16)
        t = len(self.toks) - 1
        T = self.c["T"]; slot = t % T
        pos = {int(r): i for i, r in enumerate(rows)}
        for j, p in self.a_last:
            if j != slot and p:
                tk = self.tok_at((j + 1) % T)
                if tk in pos:
                    lg[pos[tk]] += rs(I["copy_gain"] * p, 15)
        return lg

    # ---------------------------------------------------------------- fast output step
    def candidates(self, O, mprobe=16):
        """Token ids worth scoring, in the order the engine visits them (no repeats)."""
        seen, out = set(), []
        def add(ids):
            for i in ids:
                i = int(i)
                if i not in seen: seen.add(i); out.append(i)
        add(O["short"]); add(O["succ"][self.toks[-1]])
        T = self.c["T"]; last = (len(self.toks) - 1) % T
        add(self.tok_at(s_) for s_ in range(min(len(self.toks), T)) if (last - s_) % T < self.nvalid)
        rsc = rs((O["router"].astype(np.int64) @ self.h) * O["router_m"], 16)
        sc = [int(v) for v in rsc]
        for _ in range(mprobe):
            g = max(range(len(sc)), key=lambda i: (sc[i], -i))            # best score, lowest index on ties
            sc[g] = -(1 << 62)
            add(O["group_members"][O["group_start"][g]:O["group_start"][g + 1]])
        return out

    def topn(self, O, n=TOPN):
        """The n best-scoring candidates, best first; an earlier candidate wins a tie."""
        ids = self.candidates(O)
        sc = self.scores(ids)
        top = []
        for j in range(len(ids)):
            if len(top) == n and sc[j] <= sc[top[-1]]: continue
            p = len(top) if len(top) < n else n - 1
            if len(top) < n: top.append(j)
            while p > 0 and sc[top[p - 1]] < sc[j]: top[p] = top[p - 1]; p -= 1
            top[p] = j
        return [(ids[j], int(sc[j])) for j in top], len(ids)

    def banned(self, reply_start):
        """Tokens that would repeat a 4-token phrase already written in this reply."""
        n = len(self.toks)
        if reply_start is None or n - reply_start < 3: return set()
        last = self.toks[n - 3:]
        return {self.toks[j] for j in range(reply_start + 3, n) if self.toks[j - 3:j] == last}

    def pick(self, O, rng, inv_temp_q8, reply_start=None):
        """Sample the next token. rng is a 32-bit xorshift state; returns (token, new state).
        reply_start: position where the current reply began (enables the repeated-phrase rule)."""
        top, _ = self.topn(O)
        ban = self.banned(reply_start)
        w = []
        for _, s_ in top:
            d = ((top[0][1] - s_) * inv_temp_q8) >> (8 + Q - EXP_BITS)
            w.append(int(self.I["exp"][d]) if d < EXP_N else 0)
        wb = [0 if tk in ban else wi for (tk, _), wi in zip(top, w)]
        rng ^= (rng << 13) & 0xFFFFFFFF; rng ^= rng >> 17; rng ^= (rng << 5) & 0xFFFFFFFF
        if ban and not sum(wb):                              # every weighted candidate is banned: best allowed one
            for tk, _ in top:
                if tk not in ban: return tk, rng
        if sum(wb): w = wb
        r = rng % sum(w)
        for (tok, _), wi in zip(top, w):
            if r < wi: return tok, rng
            r -= wi
        raise AssertionError("unreachable")
