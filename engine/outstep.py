"""Build the data for the fast output step and measure it with the engine's integer arithmetic.

Candidates = SHORT most frequent tokens + the previous token's KSUCC usual successors + tokens in the window
+ members of the MPROBE best of NGROUP groups chosen by a 2-bit router trained on the frozen model."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, ROOT + '/study'); sys.path.insert(0, ROOT)
from common import load
from train.model import QLinear
from train import quant
SHORT, KSUCC, NGROUP, MPROBE = 256, 32, 256, 16
S = ROOT + '/study/'


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


def build(run="i_kd"):
    torch.manual_seed(0); torch.set_num_threads(4)
    model, cfg, counts, _ = load(run)
    V = cfg.V
    W = (model.head.qweight(model.head.weight) * model.head.g[:, None]).detach()
    tr, va = torch.load(S + f"cap_{run}_train.pt"), torch.load(S + f"cap_{run}_val.pt")
    def prep(c):
        return c["H"].float(), c["TI"].long() & 0xFFFF, (c["TV"].float() - c["LSE"].float()[:, None]).exp(), c["Y"].long() & 0xFFFF, c["PREV"].long() & 0xFFFF
    Htr, TItr, Ptr, _, _ = prep(tr)
    Hva, TIva, Pva, Yva, PREVva = prep(va)
    order = counts.argsort(descending=True, stable=True)
    short = torch.zeros(V, dtype=torch.bool); short[order[:SHORT]] = True
    tail = (~short).nonzero().flatten()
    feats = W @ Htr[torch.randperm(len(Htr), generator=torch.Generator().manual_seed(0))[:768]].T
    a = kmeans(feats[tail], NGROUP)
    cl = torch.full((V,), NGROUP, dtype=torch.long); cl[tail] = a
    tgt = torch.zeros(len(Htr), NGROUP + 1).scatter_add_(1, cl[TItr], Ptr)[:, :NGROUP]
    root = QLinear(cfg, model.st, cfg.D, NGROUP, gain=True)
    opt = torch.optim.Adam([{"params": [root.weight], "lr": 2e-2}, {"params": [root.g], "lr": 3e-3}])
    for ep in range(8):
        perm = torch.randperm(len(Htr))
        for b in range(0, len(perm), 512):
            ix = perm[b:b + 512]
            loss = -(tgt[ix] * root(Htr[ix]).log_softmax(-1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    wq = quant.tern_values(root.weight.detach())
    rt = torch.sign(wq).to(torch.int8).numpy()
    rm = np.round((wq.abs().max().double() * root.g.detach().double()).numpy() * 65536).astype(np.int64)
    assert rm.min() > 0, "a negative router gain would need a signed multiplier"
    succ = np.load(S + 'bigram_top256.npy')[:, :KSUCC].astype(np.int64)
    members = [tail[a == c].numpy() for c in range(NGROUP)]
    O = {"short": order[:SHORT].numpy(), "succ": succ, "router": rt, "router_m": rm,
         "group_start": np.cumsum([0] + [len(m) for m in members]), "group_members": np.concatenate(members)}
    # ---- measure with integer router arithmetic on held-out positions
    N = Hva.shape[0]
    hq = np.round(Hva.double().numpy() * 4096).astype(np.int64)
    rs_ = lambda x, n: (x + (1 << (n - 1))) >> n
    rsc = rs_((hq @ rt.T.astype(np.int64)) * rm[None], 16)
    rank = np.argsort(-rsc, axis=1, kind="stable")[:, :MPROBE]
    nsel = va["nsel"]; WIN = va["WIN"].long() & 0xFFFF; POS = va["POS"].long()
    cand = np.zeros((N, V), dtype=bool)
    cand[:, O["short"]] = True
    np.put_along_axis(cand, succ[PREVva.numpy()], True, axis=1)
    for i in range(N):
        cand[i, WIN[i // nsel, :POS[i] + 1].numpy()] = True
        for gI in rank[i]:
            cand[i, members[gI]] = True
    ar = np.arange(N)
    v40 = (va["TV"].float()[:, :40] / 0.5).softmax(-1).numpy()
    rep = {"rows": float(cand.sum(1).mean()) + NGROUP, "rows_max": int(cand.sum(1).max()) + NGROUP,
           "top1_kept": float(cand[ar, TIva[:, 0].numpy()].mean()), "true_in_set": float(cand[ar, Yva.numpy()].mean()),
           "sample_mass_t05_k40": float((v40 * np.take_along_axis(cand, TIva[:, :40].numpy(), 1)).sum(1).mean()),
           "largest_group": int(max(len(m) for m in members))}
    return O, rep


if __name__ == "__main__":
    import pickle
    O, rep = build()
    pickle.dump(O, open(ROOT + "/engine/outstep_i_kd.pkl", "wb"))
    print(rep)
