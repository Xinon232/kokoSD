import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import sys, numpy as np, torch
from extract import extract
from ref_float import FloatRef
sys.path.insert(0, ROOT + '/study')
from common import val_seqs
P, model, cfg = extract("i_kd")
s = val_seqs("koko", 2, cfg.T)[1]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
with torch.no_grad():
    ref = model(s[None, :n])[0][0].double().numpy()
R = FloatRef(P)
worst = 0
for t in range(n):
    lg = R.step(int(s[t]))
    d = np.abs(lg - ref[t]).max()
    worst = max(worst, d)
    if t % 16 == 0 or d > 1e-2: print(t, "max |diff|", f"{d:.2e}", "argmax same", lg.argmax() == ref[t].argmax(), flush=True)
print("worst", worst)
