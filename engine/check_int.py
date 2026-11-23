"""Integer reference against the float reference: score differences, same top token, and loss on held-out text."""
import sys, pickle, numpy as np
from ref_float import FloatRef
from ref_int import IntRef, quantize, ONE
sys.path.insert(0, '/home/user/kokosd/study')
from common import val_seqs, DOMAINS, REF
P = pickle.load(open("params_i_kd.pkl", "rb"))
import os
I = quantize(P, int(os.environ.get("EMB_BITS", 8)))
nseq, T = int(sys.argv[1]) if len(sys.argv) > 1 else 1, int(sys.argv[2]) if len(sys.argv) > 2 else 256
F, R = FloatRef(P), IntRef(I)
tot = {}
for d in DOMAINS:
    s = val_seqs(d, max(nseq, 2), T)
    nf = ni = same = n = 0; worst = 0
    for b in range(nseq):
        F.reset(); R.reset()
        for t in range(T):
            lf = F.step(int(s[b, t])); R.step(int(s[b, t])); li = R.scores() / ONE
            y = int(s[b, t + 1])
            nf += np.log(np.exp(lf - lf.max()).sum()) + lf.max() - lf[y]
            ni += np.log(np.exp(li - li.max()).sum()) + li.max() - li[y]
            same += lf.argmax() == li.argmax(); n += 1; worst = max(worst, np.abs(lf - li).max())
    tot[d] = (nf / n, ni / n)
    print(f"{d:8s} float {nf / n:.4f}  integer {ni / n:.4f}  same top token {same / n:.1%}  worst score gap {worst:.2f}", flush=True)
print("overall  float %.4f  integer %.4f" % tuple(sum(REF[d] * tot[d][i] for d in DOMAINS) for i in (0, 1)))
