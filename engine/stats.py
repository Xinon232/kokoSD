import sys, pickle, numpy as np
from extract import extract
import ref_float
sys.path.insert(0, '/home/user/kokosd/study')
from common import val_seqs, DOMAINS
P, model, cfg = extract("i_kd")
pickle.dump(P, open("params_i_kd.pkl", "wb"))
ref_float.STATS = {}
R = ref_float.FloatRef(P)
for d in DOMAINS:
    s = val_seqs(d, 3, cfg.T)
    for b in range(2):
        R.reset()
        for t in range(cfg.T): R.step(int(s[b, t]))
for k, (mx, ss, n) in ref_float.STATS.items(): print(f"{k:20s} absmax {mx:12.3f}  rms {np.sqrt(ss / n):10.3f}")
print("emb absmax", np.abs(P["emb"]).max(), "inj step max", max(sp["step"].max() for sp in P["inj"]))
for nm in ("n1", "n2"): print(nm, "gain absmax", max(np.abs(L[nm]).max() for L in P["layers"]))
print("nf", np.abs(P["nf"]).max(), "head_g", np.abs(P["head_g"]).max(), P["head_g"].min(), "head_s", P["head_s"], "copy_gain", P["copy_gain"])
print("k_g", max(np.abs(L["k_g"]).max() for L in P["layers"] if "k_g" in L), "temp", [np.round(L["temp"], 2).tolist() for L in P["layers"] if "temp" in L])
print("dbias absmax", max(np.abs(L["dbias"]).max() for L in P["layers"] if "dbias" in L), "taps", max(np.abs(L["taps"]).max() for L in P["layers"] if "taps" in L), "avg", max(np.abs(L["avg"]).max() for L in P["layers"] if "avg" in L))
print("w1_s range", min(L["w1_s"].min() for L in P["layers"]), max(L["w1_s"].max() for L in P["layers"]), "w2_s", min(L["w2_s"].min() for L in P["layers"]), max(L["w2_s"].max() for L in P["layers"]))
