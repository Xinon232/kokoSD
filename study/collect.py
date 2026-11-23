"""Run the frozen model over text and keep, per sampled position: the head input, the 64 best-scoring
tokens with their scores, the true next token, the previous token and the window (for context tokens)."""
import sys, numpy as np, torch
from common import load, val_seqs, DOMAINS, REF, A
torch.set_num_threads(4)
run, split, nseq = sys.argv[1], sys.argv[2], int(sys.argv[3])
model, cfg, counts, res = load(run)
grab = {}
model.nf.register_forward_hook(lambda m, i, o: grab.__setitem__("h", o))
rng = np.random.default_rng(1 if split == "train" else 2)
H, TI, TV, Y, PREV, WIN, POS, DOM, LSE = [], [], [], [], [], [], [], [], []
sel = np.arange(32, cfg.T, 3)
with torch.no_grad():
    for di, d in enumerate(DOMAINS):
        n = int(round(nseq * REF[d])) if split == "train" else nseq // 4
        if split == "train":
            ids = np.fromfile(f'{A}mix1x/mix1/{d}_train_16384.bin', dtype=np.uint16)
            st = rng.integers(0, len(ids) - cfg.T - 2, size=n)
            s = torch.from_numpy(np.stack([ids[a:a + cfg.T + 1].astype(np.int64) for a in st]))
        else:
            s = val_seqs(d, n, cfg.T)
        for b in range(0, n, 16):
            x, y = s[b:b + 16, :-1], s[b:b + 16, 1:]
            out = model(x)[0][:, sel]
            v, i = out.topk(64)
            H.append(grab["h"][:, sel].flatten(0, 1).half()); TI.append(i.flatten(0, 1).to(torch.int16)); TV.append(v.flatten(0, 1).half())
            LSE.append(out.logsumexp(-1).flatten().half())
            Y.append(y[:, sel].flatten().to(torch.int16)); PREV.append(x[:, sel].flatten().to(torch.int16))
            WIN.append(x.to(torch.int16)); POS.append(torch.from_numpy(np.tile(sel, x.shape[0])).to(torch.int16))
            DOM.append(torch.full((x.shape[0] * len(sel),), di, dtype=torch.int8))
        print(d, n, flush=True)
torch.save({"H": torch.cat(H), "TI": torch.cat(TI), "TV": torch.cat(TV), "LSE": torch.cat(LSE), "Y": torch.cat(Y),
            "PREV": torch.cat(PREV), "WIN": torch.cat(WIN), "POS": torch.cat(POS), "DOM": torch.cat(DOM), "nsel": len(sel)},
           f"cap_{run}_{split}.pt")
