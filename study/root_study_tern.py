"""A router trained on the frozen model: which groups of rare tokens are worth scoring at this position?
Candidates = short list + previous token's successors + context tokens + members of the router's top groups."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from common import load, DOMAINS
torch.set_num_threads(4)
run = sys.argv[1] if len(sys.argv) > 1 else "h_koko_L8"
model, cfg, counts, res = load(run)
W = (model.head.qweight(model.head.weight) * model.head.g[:, None]).detach()
tr, va = torch.load(f"cap_{run}_train.pt"), torch.load(f"cap_{run}_val.pt")
big = torch.from_numpy(np.load('bigram_top256.npy').astype(np.int64))
order = counts.argsort(descending=True)
V = cfg.V


def prep(c):
    H = c["H"].float(); TI = c["TI"].long() & 0xFFFF; P = (c["TV"].float() - c["LSE"].float()[:, None]).exp()
    return H, TI, P, c["Y"].long() & 0xFFFF, c["PREV"].long() & 0xFFFF


Htr, TItr, Ptr, Ytr, PREVtr = prep(tr)
Hva, TIva, Pva, Yva, PREVva = prep(va)
N = Hva.shape[0]
# context tokens for validation positions
nsel = va["nsel"]; WIN = va["WIN"].long() & 0xFFFF; POS = va["POS"].long()
ctx = torch.zeros(N, V, dtype=torch.bool)
for i in range(N):
    ctx[i, WIN[i // nsel, :POS[i] + 1]] = True
v40 = (va["TV"].float()[:, :40] / 0.7).softmax(-1)
ar = torch.arange(N)
dom = va["DOM"].long()


def kmeans(X, C, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xn = F.normalize(X, dim=-1)
    cen = Xn[torch.randperm(len(Xn), generator=g)[:C]].clone()
    for _ in range(iters):
        a = (Xn @ cen.T).argmax(-1)
        for c in range(C):
            m = a == c
            cen[c] = F.normalize(Xn[m].mean(0), dim=0) if m.any() else Xn[torch.randint(len(Xn), (1,), generator=g)][0]
    return (Xn @ cen.T).argmax(-1)


def report(tag, cand, extra=None):
    r = {"cfg": tag, "rows": round(cand.sum(-1).float().mean().item() + (extra or 0)),
         "true_in_set": round(cand[ar, Yva].float().mean().item(), 4),
         "top1_in_set": round(cand[ar, TIva[:, 0]].float().mean().item(), 4),
         "top64_mass": round((Pva * cand.gather(1, TIva)).sum(-1).mean().item() / Pva.sum(-1).mean().item(), 4),
         "sample_mass": round((v40 * cand.gather(1, TIva[:, :40])).sum(-1).mean().item(), 4),
         "top1_facts": round(cand[ar, TIva[:, 0]][dom == 2].float().mean().item(), 4)}
    print(json.dumps(r), flush=True)
    return r


out = []
feats = {"rows": W, "behaviour": W @ Htr[torch.randperm(len(Htr), generator=torch.Generator().manual_seed(0))[:768]].T}
for S, K in ((256, 0), (256, 32), (256, 64), (512, 64)):
    short = torch.zeros(V, dtype=torch.bool); short[order[:S]] = True
    tail = (~short).nonzero().flatten()
    basec = short[None].expand(N, -1).clone(); (basec.scatter_(1, big[PREVva][:, :K], True) if K else None); basec |= ctx
    out.append(report(f"S{S} K{K} no router", basec))
    for fname in ("behaviour",):
        for C in (256,):
            a = kmeans(feats[fname][tail], C)
            cl = torch.full((V,), C, dtype=torch.long); cl[tail] = a                       # short-list tokens -> dummy group C
            size = torch.bincount(a, minlength=C).float()
            # training target: where the model's own probability mass in the tail sits
            tgt = torch.zeros(len(Htr), C + 1).scatter_add_(1, cl[TItr], Ptr)[:, :C]
            wgt = tgt.sum(-1)
            from train.model import QLinear
            root = QLinear(cfg, model.st, cfg.D, C, gain=True)
            opt = torch.optim.Adam([{"params": [root.weight], "lr": 2e-2}, {"params": [root.g], "lr": 3e-3}])
            for ep in range(8):
                perm = torch.randperm(len(Htr))
                for b in range(0, len(perm), 512):
                    ix = perm[b:b + 512]
                    loss = -(tgt[ix] * root(Htr[ix]).log_softmax(-1)).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                rank = root(Hva).argsort(-1, descending=True)
            member = torch.zeros(C, V, dtype=torch.bool); member[a, tail] = True
            for m in (8, 12, 16, 24):
                cand = basec.clone()
                for j in range(m):
                    cand |= member[rank[:, j]]
                r = report(f"S{S} K{K} {fname} C{C} m{m}", cand, extra=C)
                r["max_group"] = int(size.max()); out.append(r)
json.dump(out, open(f"root_study_tern2_{run}.json", "w"), indent=1)
