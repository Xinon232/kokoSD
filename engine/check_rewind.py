"""Mark / rewind: reference and twin must agree through a script of marks, edits and rewinds, across the ring wrap."""
import sys, pickle, subprocess, random, numpy as np
from ref_int import IntRef, quantize
sys.path.insert(0, '/home/user/kokosd/study')
from common import val_seqs
I = quantize(pickle.load(open("/home/user/kokosd/engine/params_i_kd.pkl", "rb")))
O = pickle.load(open("/home/user/kokosd/engine/outstep_i_kd.pkl", "rb"))
V, D = I["cfg"]["V"], I["cfg"]["D"]
rng = random.Random(5)
s = val_seqs("chat", 2, 400)[1].tolist()
ops, i = [], 0
start = int(sys.argv[1]) if len(sys.argv) > 1 else 244        # get near the wrap, then edit
while i < start: ops.append(s[i]); i += 1
for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
    ops.append(-2)                                            # mark, feed a few, take them back, feed others
    k = rng.randint(1, 6)
    ops += [rng.randrange(3, V) for _ in range(k)]
    ops.append(-3)
    m = rng.randint(2, 7)
    ops += s[i:i + m]; i += m
raw = subprocess.run(["/home/user/kokosd/build/koko_twin", "/home/user/kokosd/build/koko.bin"],
                     input=" ".join(map(str, ops)).encode(), capture_output=True).stdout
nstep = sum(1 for o in ops if o >= 0)
a = np.frombuffer(raw, dtype=np.int32).reshape(nstep, D + V + 83)
R = IntRef(I); R.reset(); r = 0; rs_ = 12345; bad = 0
for o in ops:
    if o == -2: R.mark(); continue
    if o == -3: R.rewind(); continue
    h = R.step(o)
    if r >= start - 2:                                       # the long warm-up is covered by check_twin.py
        lg = R.scores(); top, nsc = R.topn(O); rec = a[r, D + V:]
        ok = (np.array_equal(h, a[r, :D]) and np.array_equal(lg, a[r, D:D + V]) and rec[1] == nsc
              and rec[3:3 + len(top)].tolist() == [t for t, _ in top])
        if not ok:
            bad += 1; print("mismatch at step", r, "nvalid", R.nvalid); break
    r += 1
print("steps compared:", nstep - (start - 2), "| rewinds:", ops.count(-3), "|", "FAILED" if bad else "REWIND MATCHES REFERENCE EXACTLY")
