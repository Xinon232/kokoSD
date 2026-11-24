import json, math
from pathlib import Path
import numpy as np
import pytest, torch
from train import quant
from train.model import Config, Model, MoEMLP, QState, balanced_token_routes, _mix32
from train.clusters import build_assign
from train.train import load_config, run
from train.cost import estimate

CFG = Path(__file__).resolve().parents[1] / "configs"
ARMS = sorted(p for p in CFG.glob("[ps]*.json"))
SMALL = dict(V=64, D=32, L=2, T=24, NH=2, DK=16, DV=16, HD=32, data="synthetic")


def small(path, **over):
    c = load_config(path)
    d = {**c.__dict__, **SMALL, **over}
    d["DK"] = 32 if c.DK > 64 else 16
    d["E"] = min(c.E, 8)
    d["rescore_m"], d["head_short"], d["facts_n"] = 12, min(c.head_short, 8), 0
    d["hashmem_bits"] = 6
    d["hashmem_layers"] = [l for l in c.hashmem_layers if l < 2] or ([1] if c.hashmem_heads else [])
    if c.att_layers is not None:
        d["att_layers"] = [1]
    d["inject"] = [{**t, "rows": min(t["rows"], 128), "layers": sorted({min(l, 1) for l in t["layers"]})} for t in c.inject]
    return Config(**d)


def wake(m):
    """Zero-initialised outputs would hide a leak: give every all-zero parameter random values."""
    for p in m.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    return m


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_causal(path):
    """Changing later tokens must not change earlier outputs, in every arm."""
    m = wake(Model(small(path))).eval()
    m.set_progress(1.0)
    a = torch.randint(0, 64, (2, 24))
    b = a.clone(); b[:, 15:] = torch.randint(0, 64, (2, 9))
    ya, yb = m(a)[0], m(b)[0]
    assert torch.allclose(ya[:, :15], yb[:, :15], atol=1e-5)
    assert not torch.allclose(ya[:, 15:], yb[:, 15:], atol=1e-5)


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_trains(path):
    """Every arm takes gradient steps without error and every trainable parameter gets a gradient."""
    c = small(path)
    m = Model(c)
    m.set_progress(0.5)
    x = torch.randint(0, 64, (4, 24))
    out, aux, _ = m(x)
    (torch.nn.functional.cross_entropy(out.flatten(0, 1), x.flatten()) + aux).backward()
    dead = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    # zero-initialised output paths block the gradient to what feeds them on the very first step
    # a layer with neither attention nor a mixer never uses its first norm
    unused = tuple(f"n1.{l}.g" for l in range(c.L) if not c.has_att(l) and c.mixer == "none" and not c.inject)
    assert not [n for n in dead if n not in unused], dead
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_sherry_pattern():
    w = torch.randn(3, 5, 16)
    t = quant.sherry_pattern(w).reshape(3, 5, 4, 4)
    assert ((t == 0).sum(-1) == 1).all() and set(t.unique().tolist()) == {-1.0, 0.0, 1.0}
    v = quant.sherry_values(w)
    assert torch.allclose(v.abs().amax(-1), v.abs().sum(-1) / 12)


def test_tern_and_grad():
    w = torch.randn(4, 8, requires_grad=True)
    q = quant.quantize_weight(w, "tern")
    assert len(q.detach().abs().unique()) <= 2
    q.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_arenas_off_at_eval_and_end():
    m = Model(Config(**{**SMALL, "wq": "sherry", "arenas": True}))
    x = torch.randint(0, 64, (2, 24))
    m.set_progress(0.3); assert 0 < m.st.lam < 1
    m.set_progress(0.7); assert m.st.lam == 0.0
    m.set_progress(0.3); m.eval()
    y1 = m(x)[0]; m.st.lam = 0.0; y2 = m(x)[0]
    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("keyq", ["none", "sign", "pq16", "rescore"])
def test_topk_limits_mixing(keyq):
    c = Config(**{**SMALL, "topk": 3, "keyq": keyq, "rescore_m": 8, "att_bias": True})
    att = wake(Model(c)).eval().att[0]
    _, a, _ = att(torch.randn(2, 24, 32))
    assert ((a > 0).sum(-1) <= 3).all() and torch.allclose(a.sum(-1), torch.ones(2, 2, 24), atol=1e-5)
    assert (a.triu(1) == 0).all()


def test_pq16_starts_as_sign_keys():
    c = Config(**{**SMALL, "keyq": "pq16"})
    att = Model(c).att[0]
    q, k = torch.randn(1, 2, 24, 8), torch.randn(1, 2, 24, 8)
    _, kh = att._keys(q, k)
    assert torch.allclose(kh, quant.hard_sign(k))


@pytest.mark.parametrize("wq", ["fp", "tern"])
def test_batched_experts_match_loop(wq):
    c = Config(**{**SMALL, "mlp": "moe", "E": 8, "wq": wq})
    torch.manual_seed(0)
    mlp = wake(MoEMLP(c, QState(), 0))
    x, idx = torch.randn(200, 32), torch.randint(0, 8, (200,))
    fast = mlp._run(x, idx)
    slow = torch.zeros_like(x)
    for e in range(8):
        m = idx == e
        slow[m] = mlp.w2(torch.relu(mlp.w1(x[m], e)).square(), e)
    assert torch.allclose(fast, slow, atol=1e-5)
    lop = torch.zeros(200, dtype=torch.long)                 # everything to one expert: fallback path
    assert torch.allclose(mlp._run(x, lop), mlp.w2(torch.relu(mlp.w1(x, 0)).square(), 0), atol=1e-5)


def test_token_routes_balance_mass_and_differ_by_layer():
    counts = (1.0 / torch.arange(1, 4097)) * 1e6
    p = (counts / counts.sum()).double()
    r3, r4 = balanced_token_routes(counts, 16, seed=3), balanced_token_routes(counts, 16, seed=4)
    load = torch.zeros(16, dtype=torch.float64).index_add_(0, r3, counts.double())
    assert load.max() <= max(counts.max().item(), load.sum().item() / 16 * 1.3)
    assert p[r3 == r4].sum() < 0.2            # by token mass, two layers' maps agree no more than chance-ish


def test_override_parsing_and_guards():
    c = load_config(CFG / "base.json", ["wq=tern", "router=token", "name=test1", "steps=7", "topk=8", "inject=[]"])
    assert (c.wq, c.router, c.name, c.steps, c.topk, c.inject) == ("tern", "token", "test1", 7, 8, [])
    with pytest.raises(AssertionError):
        load_config(CFG / "base.json", ["no_such_key=1"])
    with pytest.raises(AssertionError):
        Config(**{**SMALL, "row_gain": True}).validate()      # float weights: the gain would test nothing


def test_optimiser_groups():
    """Every trainable parameter is in exactly one group; only weight matrices are decayed or given to Muon."""
    from train.train import run
    import train.train as tt
    seen = {}
    real = torch.optim.AdamW
    c = Config(**{**SMALL, "steps": 2, "batch": 4, "warmup": 1, "eval_every": 5, "val_seqs": 8, "wq": "tern", "opt": "muon",
                  "keyq": "pq16", "att_bias": True, "row_gain": True, "mlp": "moe", "E": 4, "table_lr_mult": 5,
                  "inject": [{"kind": "bigram", "rows": 64, "layers": [0], "gate": False}], "name": "g"})
    m = Model(c)
    names = {id(p): n for n, p in m.named_parameters()}
    import tempfile
    captured = []
    class Spy(real):
        def __init__(self, groups, **kw):
            captured.append(("adam", groups)); super().__init__(groups, **kw)
    import train.optim as to
    realm = to.Muon
    class SpyM(realm):
        def __init__(self, groups, **kw):
            captured.append(("muon", groups)); super().__init__(groups, **kw)
    torch.optim.AdamW, to.Muon = Spy, SpyM
    try:
        run(c, "cpu", out_dir=tempfile.mkdtemp())
    finally:
        torch.optim.AdamW, to.Muon = real, realm
    kinds = {}
    for opt, groups in captured:
        for g in groups:
            for p in g["params"]:
                kinds.setdefault(id(p), []).append((opt, g["kind"], g["mult"], tuple(p.shape)))
    assert all(len(v) == 1 for v in kinds.values())
    muon_shapes = [v[0][3] for v in kinds.values() if v[0][0] == "muon"]
    assert muon_shapes and all(len(s) >= 2 and s[-1] > 4 for s in muon_shapes)
    assert (4, 24) not in muon_shapes and not any(s[-2:] == (16, 4) for s in muon_shapes)   # no distance bias, no codebook
    assert sum(1 for v in kinds.values() if v[0][2] == 5) == 1                               # only the n-gram rows get x5


@pytest.mark.parametrize("mode,short", [("random", 0), ("freq", 8), ("random", 8)])
def test_two_level_head_is_a_distribution(mode, short):
    c = Config(**{**SMALL, "head": "two_level", "head_clusters": mode, "head_short": short, "wq": "tern", "row_gain": True})
    m = wake(Model(c)).eval()
    lp = m(torch.randint(0, 64, (2, 24)))[0]
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 24), atol=1e-4)
    h = m.head                                               # brute force for one position
    x = torch.randn(1, 32)
    r = h.root(x).log_softmax(-1)[0]; z = h.mem(x)[0]
    want = torch.empty(64)
    for t in range(64):
        s = int(h.slot[t])
        if s < short:
            want[t] = r[s]
        else:
            mem = (h.slot == s).nonzero().flatten()
            want[t] = r[s] + z[t] - torch.logsumexp(z[mem], 0)
    assert torch.allclose(h(x)[0], want, atol=1e-4)


def test_assign_shapes_and_kmeans():
    counts = torch.arange(200, 0, -1).float()
    for mode in ("random", "freq", "kmeans"):
        feats = torch.randn(200, 6) if mode == "kmeans" else None
        a = build_assign(counts, feats, 20, 12, mode, seed=1)
        assert sorted(a[:20].tolist()) == list(range(20))            # the most frequent 20 are short-listed
        sizes = torch.bincount(a[20:] - 20, minlength=12)
        assert sizes.sum() == 180 and sizes.max() <= 15 and a.max() == 31


def test_hash_spreads_and_is_stable():
    t = torch.arange(0, 4096)
    h = _mix32(t * 0x9E3779B1 + 7 * 0x85EBCA77) % 16384
    assert h.unique().numel() > 3500 and torch.equal(h, _mix32(t * 0x9E3779B1 + 7 * 0x85EBCA77) % 16384)


def test_each_arm_is_distinct_and_valid():
    seen = set()
    for p in ARMS:
        d = json.loads(p.read_text()); d.pop("base")
        key = json.dumps(d, sort_keys=True)
        assert key not in seen, p.stem
        seen.add(key)
        assert all(k in Config.__dataclass_fields__ for k in d), p.stem
        e = estimate(load_config(p))
        assert e["cycles_per_token"] > 0 and e["rom_mb"] > 0


def test_cost_matches_benchmark():
    """The benchmarked shape used sign keys and a two-step head: 3.28M cycles of which 0.18M was the router."""
    c = Config(V=16384, D=256, L=6, T=256, NH=4, DK=64, DV=64, HD=256, wq="sherry", keyq="sign",
               head="two_level", head_C=128)
    e = estimate(c)
    assert abs(e["cycles_per_token"] - 3_100_000) / 3_100_000 < 0.10, e
    flat = estimate(Config(V=16384, D=256, L=6, wq="sherry", keyq="sign"))
    assert flat["cycles_per_token"] > 2 * e["cycles_per_token"] and flat["gba_tokens_per_s_two_step_head"] > 4


@pytest.mark.parametrize("over", [
    {"wq": "sherry", "arenas": True, "mlp": "moe", "E": 4},
    {"wq": "tern", "opt": "muon", "wd_off_frac": 0.5, "mlp": "moe", "E": 4, "router": "token"},
    {"head": "two_level", "head_clusters": "kmeans", "head_short": 8, "x0_mix": True, "table_lr_mult": 5,
     "inject": [{"kind": "bigram", "rows": 64, "layers": [0], "gate": False, "bits": 4}]},
])
def test_smoke_training(tmp_path, over):
    c = Config(**{**SMALL, "steps": 60, "batch": 8, "warmup": 10, "eval_every": 30, "val_seqs": 16, "name": "smoke", **over})
    r = run(c, "cpu", out_dir=tmp_path)
    assert r["train_loss_last"] < r["train_loss_first"] and math.isfinite(r["val_nats_per_token"])
    assert (tmp_path / "result.json").exists()


def test_facts_probe(tmp_path):
    pytest.importorskip("tokenizers")
    if not (CFG.parent / "data" / "tok_4096.json").exists():
        pytest.skip("tokenizer not prepared")
    from train.data import load_tokenizer
    from train.facts import Facts, REL, HELD_OUT
    tok = load_tokenizer(4096)
    f = Facts(40, tok, seed=0, marker=True)
    rows = f.rows(np.random.default_rng(0), 3, 40)
    assert rows.shape == (3, 41) and rows.max() < 4096 and 0 not in rows
    assert f.sentence(0, "color", 0).startswith(" @ ") and "{" not in f.sentence(0, "job", 4, cut=True)
    assert len(f.sent_ids) == 40 * 6 * 4 + 20 * 6 and f.seen5.sum() == 20
    assert f.exposures_per_fact(100, 8, 40) > 0

    class Oracle(torch.nn.Module):
        """Knows every fact: puts all probability on the true continuation. Must score 100%."""
        def __init__(self): super().__init__(); self.truth = {}
        def forward(self, x):
            out = torch.full((*x.shape, 4096), -20.0)
            for b, row in enumerate(x.tolist()):
                for t in range(len(row)):
                    nxt = self.truth.get(tuple(row[:t + 1]))
                    if nxt is not None:
                        out[b, t, nxt] = 20.0
            return out, None, None
    o = Oracle()
    for p in range(40):
        for r in REL:
            for w in range(HELD_OUT + 1):
                ids = tok.encode(f.sentence(p, r, w)).ids
                for t in range(1, len(ids)):
                    o.truth[tuple(ids[:t])] = ids[t]
    res = f.probe(o, "cpu", n_people=40)
    assert res["trained_wording"]["recall_among_values"] == 1.0 and res["trained_wording"]["next_token_top1"] == 1.0
    assert res["wording_unseen_for_this_person"]["recall_among_values"] == 1.0
    assert abs(res["trained_wording"]["bits_per_fact"] - res["bits_available_per_fact"]) < 0.05
    m = Model(Config(V=4096, D=32, L=1, T=64, NH=2, DK=16, DV=16, HD=32)).eval()       # knows nothing: near chance
    res = f.probe(m, "cpu", n_people=40)
    assert res["trained_wording"]["recall_among_values"] < 0.25 and res["trained_wording"]["bits_per_fact"] < 0.5


def test_mixed_corpus_batches_and_scores(tmp_path):
    if not (CFG.parent / "data" / "mixq" / "meta_4096.json").exists():
        pytest.skip("quick mixed corpus not prepared (python3 -m train.corpus --name mixq --quick)")
    from train.data import TokenData
    d = TokenData("mixq", 4096, mix={"chat": 0.5, "facts": 0.25, "stories": 0.25})
    x, y = d.batch(np.random.default_rng(0), 8, 32)
    assert x.shape == (8, 32) and torch.equal(x[:, 1:], y[:, :-1])
    chat_only = TokenData("mixq", 4096, mix={"chat": 1.0})
    xc, _ = chat_only.batch(np.random.default_rng(0), 64, 64)
    xs, _ = TokenData("mixq", 4096, mix={"stories": 1.0}).batch(np.random.default_rng(0), 64, 64)
    assert ((xc == 1) | (xc == 2)).float().mean() > 20 * ((xs == 1) | (xs == 2)).float().mean() + 0.01   # role tokens
    c = Config(V=4096, D=32, L=1, T=64, NH=2, DK=16, DV=16, HD=32, data="mixq", steps=30, batch=8, warmup=5,
               eval_every=100, val_seqs=16, facts_n=20, name="mixsmoke")
    r = run(c, "cpu", out_dir=tmp_path)
    assert set(r["val_by_domain"]) == {"chat", "facts", "stories"} and r["facts"]["facts"] == 120
    w = sum(r["val_by_domain"][k]["nats_per_token"] * v for k, v in {"chat": .5, "facts": .25, "stories": .25}.items())
    assert abs(w - r["val_nats_per_token"]) < 2e-3
