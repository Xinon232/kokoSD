"""Successor lists: for every token, its most frequent next tokens in the training text (Koko's data counted double)."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import numpy as np, sys
A = ROOT + '/artifacts/mix1x/mix1/'
V, K = 16384, 256
cnt = np.zeros(V * V, dtype=np.int32)
for d, w in (("koko", 2), ("chat", 1), ("facts", 1), ("stories", 1)):
    ids = np.fromfile(A + f'{d}_train_16384.bin', dtype=np.uint16).astype(np.int64)
    for s in range(0, len(ids) - 1, 50_000_000):
        seg = ids[s:s + 50_000_001]
        cnt += (w * np.bincount(seg[:-1] * V + seg[1:], minlength=V * V)).astype(np.int32)
    print(d, len(ids), flush=True)
cnt = cnt.reshape(V, V)
top = np.argsort(-cnt, axis=1, kind="stable")[:, :K].astype(np.uint16)
np.save(ROOT + '/study/bigram_top256.npy', top)
nz = (np.take_along_axis(cnt, top.astype(np.int64), 1) > 0).sum(1)
print("rows with fewer than 64 real successors:", int((nz < 64).sum()), "median real successors (capped 256):", int(np.median(nz)))
