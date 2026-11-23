"""Second attempt: candidates = global short list + the previous token's usual successors + tokens in context."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from common import load, val_seqs, DOMAINS, REF
torch.set_num_threads(4)
run = sys.argv[1] if len(sys.argv) > 1 else "h_koko_L8"
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 24
model, cfg, counts, res = load(run)
FL, Y, PREV, CTX, DOM = [], [], [], [], []
with torch.no_grad():
    for d in DOMAINS:
        s = val_seqs(d, NSEQ, cfg.T)
        for b in range(0, NSEQ, 8):
            x, y = s[b:b + 8, :-1], s[b:b + 8, 1:]
            out = model(x)[0]
            sel = slice(32, None, 3)
            FL.append(out[:, sel].flatten(0, 1).half()); Y.append(y[:, sel].flatten()); PREV.append(x[:, sel].flatten())
            for bi in range(x.shape[0]):
                for t in range(x.shape[1])[sel]:
                    CTX.append(x[bi, :t + 1])
            DOM += [d] * FL[-1].shape[0]
FL, Y, PREV = torch.cat(FL).float(), torch.cat(Y), torch.cat(PREV)
N = FL.shape[0]
base = F.cross_entropy(FL, Y, reduction="none")
dom = np.array(DOM)
ctxmask = torch.zeros(N, cfg.V, dtype=torch.bool)
for i, c in enumerate(CTX):
    ctxmask[i, c] = True
P = FL.softmax(-1)
top1 = FL.argmax(-1)
v40, i40 = (FL / 0.7).topk(40)
p40 = v40.softmax(-1)
order = counts.argsort(descending=True)
big = torch.from_numpy(np.load('bigram_top256.npy').astype(np.int64))
ar = torch.arange(N)
out = []
for S in (0, 256, 512, 1024, 2048):
    short = torch.zeros(cfg.V, dtype=torch.bool); short[order[:S]] = True
    for K in (0, 32, 64, 128, 256):
        for use_ctx in (True,):
            cand = short[None].expand(N, -1).clone()
            if K:
                cand.scatter_(1, big[PREV][:, :K], True)
            if use_ctx:
                cand |= ctxmask
            nrows = cand.sum(-1).float()
            mass = (P * cand).sum(-1)
            in_set = cand[ar, Y]
            # loss if the scored rows are renormalised and a missed token is charged at 1e-4 probability
            nll = torch.where(in_set, base + mass.clamp_min(1e-9).log(), torch.full_like(base, 9.21))
            r = {"S": S, "K": K, "rows": round(nrows.mean().item()), "rows_p95": round(nrows.quantile(0.95).item()),
                 "true_in_set": round(in_set.float().mean().item(), 4), "top1_in_set": round(cand[ar, top1].float().mean().item(), 4),
                 "mass": round(mass.mean().item(), 4), "sample_mass": round((p40 * cand.gather(1, i40)).sum(-1).mean().item(), 4),
                 "top1_by_domain": {d: round(cand[ar, top1][torch.from_numpy(dom == d)].float().mean().item(), 4) for d in DOMAINS}}
            out.append(r); print(json.dumps(r), flush=True)
json.dump({"run": run, "positions": N, "flat_nats": base.mean().item(), "rows": out}, open(f"head_study2_{run}.json", "w"), indent=1)
