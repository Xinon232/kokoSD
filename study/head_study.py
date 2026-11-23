"""How much does a short-list + cluster search over the output rows lose, applied after training?
Captures the head input and the full scores, then asks: if only some rows are scored, is the token
the full head would have produced still among them?"""
import sys, json, math, torch, torch.nn.functional as F
from common import load, val_seqs, DOMAINS, REF
torch.set_num_threads(4)
run = sys.argv[1] if len(sys.argv) > 1 else "h_koko_L8"
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 24
model, cfg, counts, res = load(run)
grab = {}
model.nf.register_forward_hook(lambda m, i, o: grab.__setitem__("h", o))
W = model.head.qweight(model.head.weight) * model.head.g[:, None]          # effective rows (V, D)
H, FL, Y, CTX, DOM = [], [], [], [], []
with torch.no_grad():
    for d in DOMAINS:
        s = val_seqs(d, NSEQ, cfg.T)
        for b in range(0, NSEQ, 8):
            x, y = s[b:b + 8, :-1], s[b:b + 8, 1:]
            out = model(x)[0]
            h = grab["h"]
            assert torch.allclose(h @ W.T, model.head(h), atol=1e-3)
            sel = slice(32, None, 3)                                          # skip the cold start, thin positions
            H.append(h[:, sel].flatten(0, 1)); FL.append(out[:, sel].flatten(0, 1).half()); Y.append(y[:, sel].flatten())
            # tokens already in the context of each kept position (copy-head candidates)
            T = x.shape[1]
            for bi in range(x.shape[0]):
                for t in range(T)[sel]:
                    CTX.append(x[bi, :t + 1])
            DOM += [d] * (H[-1].shape[0])
H, FL, Y = torch.cat(H), torch.cat(FL).float(), torch.cat(Y)
N = H.shape[0]
base = F.cross_entropy(FL, Y, reduction="none")
print(f"{run}: {N} positions, flat nats {base.mean():.4f}", flush=True)
ctxmask = torch.zeros(N, cfg.V, dtype=torch.bool)
for i, c in enumerate(CTX):
    ctxmask[i, c] = True
P = FL.softmax(-1)
top1 = FL.argmax(-1)
# the distribution generation actually samples from: temperature 0.7, top 40
v40, i40 = (FL / 0.7).topk(40)
p40 = v40.softmax(-1)
order = counts.argsort(descending=True)


def kmeans(X, C, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xn = F.normalize(X, dim=-1)
    cen = Xn[torch.randperm(len(Xn), generator=g)[:C]].clone()
    for _ in range(iters):
        a = (Xn @ cen.T).argmax(-1)
        for c in range(C):
            m = a == c
            cen[c] = F.normalize(Xn[m].mean(0), dim=0) if m.any() else Xn[torch.randint(len(Xn), (1,), generator=g)][0]
    return (Xn @ cen.T).argmax(-1)


rows_out = []
for S in (256, 512, 1024, 2048):
    short = torch.zeros(cfg.V, dtype=torch.bool); short[order[:S]] = True
    tail = (~short).nonzero().flatten()
    for C in (64, 128, 256):
        a = kmeans(W[tail], C)
        cen = torch.stack([W[tail][a == c].mean(0) if (a == c).any() else torch.zeros(cfg.D) for c in range(C)])
        size = torch.bincount(a, minlength=C).float()
        member = torch.zeros(C, cfg.V, dtype=torch.bool); member[a, tail] = True
        cs = H @ cen.T                                                       # centroid scores (N, C)
        rank = cs.argsort(-1, descending=True)
        for m in (1, 2, 4, 8):
            cand = short[None] | ctxmask
            nrows = torch.full((N,), float(S + C)) + (ctxmask & ~short[None]).sum(-1)
            for j in range(m):
                cand = cand | member[rank[:, j]]
                nrows += size[rank[:, j]]
            mass = (P * cand).sum(-1)
            in_set = cand[torch.arange(N), Y]
            t1 = cand[torch.arange(N), top1]
            m40 = (p40 * cand.gather(1, i40)).sum(-1)
            # nats if the model's own probabilities are renormalised over the scored rows; a miss costs the flat price plus log(1/mass)... counted separately
            nll = torch.where(in_set, base + mass.log(), torch.full_like(base, float("nan")))
            r = {"S": S, "C": C, "m": m, "rows": round(nrows.mean().item()), "rows_p95": round(nrows.quantile(0.95).item()),
                 "true_in_set": round(in_set.float().mean().item(), 4), "top1_in_set": round(t1.float().mean().item(), 4),
                 "mass": round(mass.mean().item(), 4), "sample_mass_t07_k40": round(m40.mean().item(), 4)}
            rows_out.append(r); print(json.dumps(r), flush=True)
json.dump({"run": run, "positions": N, "flat_nats": base.mean().item(), "rows": rows_out},
          open(f"/home/user/kokosd/study/head_study_{run}.json", "w"), indent=1)
