"""Can the conversation simply keep going past 256 tokens, with attention over the last 256 and positions that
keep counting? Compare the loss on tokens 256..511 with the same tokens scored inside fresh 256-token windows."""
import sys, pickle, numpy as np
from ref_float import FloatRef
sys.path.insert(0, '/home/user/kokosd/study')
from common import val_seqs, DOMAINS, REF
P = pickle.load(open("params_i_kd.pkl", "rb"))
F = FloatRef(P)
nll = lambda lg, y: np.log(np.exp(lg - lg.max()).sum()) + lg.max() - lg[y]
res = {}
for d in DOMAINS:
    s = val_seqs(d, 3, 511)[1].tolist()                  # 512 tokens
    F.reset(); ring = []
    for t in range(511):
        lg = F.step(s[t])
        if t >= 256: ring.append(nll(lg, s[t + 1]))
    fresh = {}
    for start in (128, 256):                             # fresh windows; score their second halves
        F.reset()
        for t in range(start, min(start + 256, 511)):
            lg = F.step(s[t])
            if t - start >= 128 and t >= 256: fresh[t] = nll(lg, s[t + 1])
    idx = sorted(fresh)
    r = np.array([ring[t - 256] for t in idx]); f = np.array([fresh[t] for t in idx])
    res[d] = (r.mean(), f.mean(), len(idx))
    print(f"{d:8s} continuing {r.mean():.4f}   fresh window with 128-255 tokens of context {f.mean():.4f}   ({len(idx)} tokens)", flush=True)
print("overall  continuing %.4f   fresh %.4f" % tuple(sum(REF[d] * res[d][i] for d in DOMAINS) for i in (0, 1)))
