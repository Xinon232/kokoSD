"""Does a rolling window stay healthy far past the trained positions? Loss by position band over a 2,048-token stream."""
import sys, pickle, numpy as np
from ref_float import FloatRef
sys.path.insert(0, '/home/user/kokosd/study')
from common import val_seqs
P = pickle.load(open("params_i_kd.pkl", "rb"))
F = FloatRef(P)
nll = lambda lg, y: np.log(np.exp(lg - lg.max()).sum()) + lg.max() - lg[y]
N = 2048
for d in ("chat", "koko"):
    s = val_seqs(d, 3, N - 1)[1].tolist()
    F.reset(); cont = []
    for t in range(N - 1): cont.append(nll(F.step(s[t]), s[t + 1]))
    fresh = [None] * (N - 1)
    for start in range(128, N - 129, 128):               # each token also scored in a fresh window with 128..255 tokens of context
        F.reset()
        for t in range(start, min(start + 256, N - 1)):
            lg = F.step(s[t])
            if t - start >= 128: fresh[t] = nll(lg, s[t + 1])
    for lo, hi in ((256, 512), (512, 1024), (1024, 1536), (1536, 2047)):
        idx = [t for t in range(lo, hi) if fresh[t] is not None]
        print(f"{d:5s} positions {lo:4d}-{hi:4d}: continuing {np.mean([cont[t] for t in idx]):.4f}   fresh windows {np.mean([fresh[t] for t in idx]):.4f}", flush=True)
