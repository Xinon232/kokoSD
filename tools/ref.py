"""Float reference for stories260K (llama2.c layout) + tokenizer.

Used to (a) calibrate the fixed-point formats the cartridge uses and
(b) score the integer engine against the original float model.
"""
import struct
import numpy as np


class Tok:
    def __init__(self, path):
        t = open(path, 'rb').read()
        self.maxlen = struct.unpack('i', t[:4])[0]
        o = 4
        self.vocab, self.scores = [], []
        while o < len(t):
            s, l = struct.unpack('fi', t[o:o + 8]); o += 8
            self.vocab.append(t[o:o + l]); self.scores.append(s); o += l
        # byte tokens "<0xNN>" stand for the raw byte
        self.piece = []
        for i, v in enumerate(self.vocab):
            if 3 <= i < 259 and v.startswith(b'<0x'):
                self.piece.append(bytes([int(v[3:5], 16)]))
            else:
                self.piece.append(v)
        self.lookup = {}
        # like llama2.c: look pieces up by their literal string; a character
        # with no piece of its own falls back to its byte token (byte + 3)
        for i, v in enumerate(self.vocab):
            if i >= 259:
                self.lookup.setdefault(v, i)
        for b in range(256):
            self.lookup.setdefault(bytes([b]), b + 3)

    def encode(self, text, bos=True):
        b = text.encode('utf-8') if isinstance(text, str) else text
        ids = [1] if bos else []
        if b:
            ids.append(self.lookup[b' '])
        for c in b:
            ids.append(self.lookup[bytes([c])])
        while True:
            best, bi, bid = -1e10, -1, -1
            for i in range(len(ids) - 1):
                if ids[i] < 259 or ids[i + 1] < 259:
                    continue
                m = self.lookup.get(self.piece[ids[i]] + self.piece[ids[i + 1]])
                if m is not None and self.scores[m] > best:
                    best, bi, bid = self.scores[m], i, m
            if bi < 0:
                break
            ids[bi:bi + 2] = [bid]
        return ids

    def decode(self, ids):
        out = b''
        prev = 1
        for t in ids:
            p = self.piece[t] if t >= 3 else b''
            if prev == 1 and p[:1] == b' ':
                p = p[1:]
            out += p
            prev = t
        return out.decode('utf-8', 'replace')


class Model:
    def __init__(self, path):
        d = open(path, 'rb').read()
        (self.dim, self.hidden, self.L, self.H, self.KVH, v, self.seq) = struct.unpack('7i', d[:28])
        self.shared = v > 0
        self.V = abs(v)
        w = np.frombuffer(d, dtype=np.float32, offset=28)
        self.hs = self.dim // self.H
        self.kvd = self.KVH * self.hs
        o = 0

        def take(*shape):
            nonlocal o
            n = int(np.prod(shape))
            a = w[o:o + n].reshape(shape).astype(np.float64); o += n
            return a
        D, Hd, L = self.dim, self.hidden, self.L
        self.emb = take(self.V, D)
        self.g_att = take(L, D)
        self.wq = take(L, D, D)
        self.wk = take(L, self.kvd, D)
        self.wv = take(L, self.kvd, D)
        self.wo = take(L, D, D)
        self.g_ffn = take(L, D)
        self.w1 = take(L, Hd, D)
        self.w2 = take(L, D, Hd)
        self.w3 = take(L, Hd, D)
        self.g_fin = take(D)
        take(self.seq * self.hs // 2); take(self.seq * self.hs // 2)
        self.wcls = self.emb if self.shared else take(self.V, D)
        self.freq = 1.0 / 10000 ** (np.arange(0, self.hs, 2) / self.hs)
        self.reset()

    def reset(self):
        self.kc = [[] for _ in range(self.L)]
        self.vc = [[] for _ in range(self.L)]
        self.pos = 0
        self.stats = {}

    def _stat(self, name, a):
        m = float(np.max(np.abs(a)))
        if m > self.stats.get(name, 0):
            self.stats[name] = m

    @staticmethod
    def _norm(x, g):
        return g * x / np.sqrt(np.mean(x * x) + 1e-5)

    def _rope(self, v, pos):
        v = v.reshape(-1, self.hs // 2, 2)
        a = pos * self.freq
        c, s = np.cos(a)[None, :], np.sin(a)[None, :]
        r = np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1)
        return r.reshape(-1)

    def forward(self, tok, window=None):
        """One token in, logits out. window: keep only the last N cached positions."""
        x = self.emb[tok].copy()
        pos = self.pos
        mul = self.H // self.KVH
        for l in range(self.L):
            xb = self._norm(x, self.g_att[l]); self._stat(f'n1_{l}', xb)
            q = self._rope(self.wq[l] @ xb, pos)
            k = self._rope(self.wk[l] @ xb, pos)
            v = self.wv[l] @ xb
            self._stat(f'q_{l}', q); self._stat(f'k_{l}', k); self._stat(f'v_{l}', v)
            self.kc[l].append(k); self.vc[l].append(v)
            if window and len(self.kc[l]) > window:
                self.kc[l].pop(0); self.vc[l].pop(0)
            K = np.array(self.kc[l]).reshape(-1, self.KVH, self.hs)
            Vv = np.array(self.vc[l]).reshape(-1, self.KVH, self.hs)
            out = np.zeros(self.dim)
            for h in range(self.H):
                qh = q[h * self.hs:(h + 1) * self.hs]
                sc = K[:, h // mul, :] @ qh / np.sqrt(self.hs)
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                out[h * self.hs:(h + 1) * self.hs] = sc @ Vv[:, h // mul, :]
            self._stat(f'att_{l}', out)
            x = x + self.wo[l] @ out
            xb = self._norm(x, self.g_ffn[l]); self._stat(f'n2_{l}', xb)
            h1 = self.w1[l] @ xb
            h3 = self.w3[l] @ xb
            self._stat(f'h1_{l}', h1); self._stat(f'h3_{l}', h3)
            hb = h1 / (1 + np.exp(-h1)) * h3; self._stat(f'hb_{l}', hb)
            x = x + self.w2[l] @ hb
            self._stat(f'x_{l}', x)
        x = self._norm(x, self.g_fin); self._stat('nf', x)
        self.pos += 1
        return self.wcls @ x


def load_stories(path, n=None):
    txt = open(path, encoding='utf-8').read()
    parts = [s.strip() for s in txt.split('<|endoftext|>')]
    parts = [s for s in parts[1:-1] if s]  # first/last may be cut by the range download
    return parts[:n] if n else parts


if __name__ == '__main__':
    import sys
    tok = Tok('dl/tok512.bin'); m = Model('dl/stories260K.bin')
    stories = load_stories('dl/eval/valid.txt', 40)
    nll = 0; n = 0; chars = 0
    for s in stories:
        ids = tok.encode(s)[:m.seq]
        m_stats = m.stats; m.reset(); m.stats = m_stats
        for a, b in zip(ids[:-1], ids[1:]):
            lg = m.forward(a)
            lg = lg - lg.max(); lp = lg - np.log(np.exp(lg).sum())
            nll -= lp[b]; n += 1
        chars += len(tok.decode(ids))
    print(f'{n} tokens, loss {nll / n:.4f} nats/token, {nll / np.log(2) / chars:.3f} bits/char')
    for k in sorted(m.stats):
        print(k, round(m.stats[k], 2))
    # greedy sample
    m.reset(); ids = tok.encode('Once upon a time')
    for t in ids[:-1]: m.forward(t)
    t = ids[-1]
    for _ in range(120):
        t = int(np.argmax(m.forward(t)))
        if t == 1: break
        ids.append(t)
    print(tok.decode(ids))
