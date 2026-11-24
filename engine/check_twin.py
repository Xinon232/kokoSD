"""C twin against the integer reference: every state value and every output score must be identical."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import sys, pickle, subprocess, numpy as np
from ref_int import IntRef, quantize
sys.path.insert(0, ROOT + '/study')
from common import val_seqs, DOMAINS
I = quantize(pickle.load(open(ROOT + "/engine/params_i_kd.pkl", "rb")))
O = pickle.load(open(ROOT + "/engine/outstep_i_kd.pkl", "rb"))
T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
R = IntRef(I)
V, D = I["cfg"]["V"], I["cfg"]["D"]
bad = 0
for d in (sys.argv[2].split(",") if len(sys.argv) > 2 else DOMAINS):
    s = val_seqs(d, 2, max(T, 256))[1][:T].tolist()
    raw = subprocess.run([ROOT + "/build/koko_twin", ROOT + "/build/koko.bin"],
                         input=" ".join(map(str, s)).encode(), capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int32).reshape(T, D + V + 83)
    R.reset(); rng = 12345; rows = []
    for t in range(T):
        h = R.step(s[t]); lg = R.scores()
        top, nsc = R.topn(O); tk, rng = R.pick(O, rng, 512); rec = a[t, D + V:]; rows.append(nsc)
        okt = (rec[0] == len(top) and rec[1] == nsc and rec[2] == tk and rec[3:3 + len(top)].tolist() == [i for i, _ in top]
               and rec[43:43 + len(top)].tolist() == [s_ for _, s_ in top])
        if not (np.array_equal(h, a[t, :D]) and np.array_equal(lg, a[t, D:D + V]) and okt):
            bad += 1
            if bad <= 3: print(d, "first mismatch at", t, "state diff", int(np.abs(h - a[t, :D]).max()), "score diff", int(np.abs(lg - a[t, D:D + V]).max()), "output step ok", okt, rec[:3], len(top), nsc, tk)
            break
    else:
        print(d, "identical for", T, "tokens (state, all scores, top-40 list, sampled token); rows scored avg", round(float(np.mean(rows))), flush=True)
print("FAILED" if bad else "TWIN MATCHES REFERENCE EXACTLY")
