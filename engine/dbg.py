import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import sys, numpy as np, torch
from extract import extract
import ref_float
sys.path.insert(0, ROOT + '/study')
from common import val_seqs
P, model, cfg = extract("i_kd")
s = val_seqs("koko", 2, cfg.T)[1]
cap = {}
for l in range(cfg.L):
    model.n1[l].register_forward_hook(lambda m, i, o, l=l: cap.setdefault(("n1in", l), []).append(i[0][0, 0].double().numpy()))
    model.n2[l].register_forward_hook(lambda m, i, o, l=l: cap.setdefault(("n2in", l), []).append(i[0][0, 0].double().numpy()))
model.nf.register_forward_hook(lambda m, i, o: cap.setdefault(("nfin", 0), []).append(i[0][0, 0].double().numpy()))
with torch.no_grad():
    model(s[None, :1])
# replay the reference with taps
R = ref_float.FloatRef(P)
orig = ref_float.rms
seen = []
def spy(x, g):
    seen.append(x.copy()); return orig(x, g)
ref_float.rms = spy
R.step(int(s[0]))
names = []
for l in range(cfg.L):
    names += [("n1in", l), ("n2in", l)]
names.append(("nfin", 0))
for nm, x in zip(names, seen):
    t = cap[nm][-1]
    print(nm, "max diff", f"{np.abs(x - t).max():.3e}", "scale", f"{np.abs(t).max():.2f}", "calls", len(cap[nm]))
print("---- attention layer 1 pieces at t=0")
A = model.att[1]; Ly = P["layers"][1]
xin = torch.tensor(seen[2]).float()[None, None]           # n1 input of layer 1
with torch.no_grad():
    xn = model.n1[1](xin)
    tv = A.v(xn)[0, 0].double().numpy(); tq = A.q(xn)[0, 0].double().numpy(); tk = A.k(xn)[0, 0].double().numpy()
    y, a, kraw = A(xn)
    to = A.o(A.v(xn))[0, 0].double().numpy()
xnn = xn[0, 0].double().numpy()
mv = Ly["v"] @ xnn * Ly["v_s"]
print("v diff", np.abs(mv - tv).max(), "q diff", np.abs(Ly["q"] @ xnn * Ly["q_s"] - tq).max(), "k diff", np.abs(Ly["k"] @ xnn * Ly["k_s"] * Ly["k_g"] - tk).max())
mo = Ly["o"] @ mv * Ly["o_s"]
print("o(v) diff", np.abs(mo - to).max(), "att out vs o(v)", np.abs(y[0, 0].double().numpy() - to).max(), "a", a.flatten()[:4])
with torch.no_grad():
    for nm in "qkvo":
        m = getattr(A, nm)
        qw = m.qweight(m.weight).double().numpy()
        mine = Ly[nm] * Ly[nm + "_s"]
        print(nm, "weights diff", np.abs(qw - mine).max(), "trit mismatch", int((np.round(qw / Ly[nm + "_s"]) != Ly[nm]).sum()), "of", qw.size, "s", float(Ly[nm + "_s"]), "latent absmax", float(m.weight.abs().max()), "latent mean abs", float(m.weight.abs().mean()))
