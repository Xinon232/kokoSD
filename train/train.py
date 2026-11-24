"""Train one config and write its results.

    python3 -m train.train --config train/configs/p00_fp.json [--set key=value ...] [--device cuda]

A config JSON may name a "base" file (same directory) whose values it overrides.
Outputs go to train/runs/<name>/: log.jsonl, result.json, model.pt.
"""
from __future__ import annotations

import argparse, os
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cost import estimate
from .data import TokenData
from .model import Config, Model, QLinear, config_dict

RUNS = Path(__file__).resolve().parent / "runs"


def load_config(path, overrides=()):
    path = Path(path)
    d = json.loads(path.read_text())
    if "base" in d:
        base = json.loads((path.parent / d.pop("base")).read_text())
        base.update(d)
        d = base
    d.setdefault("name", path.stem)
    for o in overrides:
        k, v = o.split("=", 1)
        assert k in Config.__dataclass_fields__, f"unknown config key {k!r}"
        try:
            d[k] = json.loads(v)
        except ValueError:
            d[k] = v
    return Config(**d)


REF_MIX = {"chat": 0.5, "facts": 0.25, "stories": 0.25}   # headline score weights, fixed whatever the training mix
REF_MIX_KOKO = {"koko": 0.4, "chat": 0.2, "facts": 0.2, "stories": 0.2}   # for corpora with Koko's own data


@torch.no_grad()
def evaluate_all(model, data, cfg, device):
    """(headline loss, {domain: loss}). For a mixed corpus the headline is the fixed-weight average."""
    if data.domains is None:
        return evaluate(model, data, cfg, device), {}
    by = {k: evaluate(model, data, cfg, device, k) for k in data.domains}
    ref = REF_MIX_KOKO if "koko" in by else REF_MIX
    w = {k: ref.get(k, 0.0) for k in by}
    return sum(by[k] * w[k] for k in by) / sum(w.values()), by


@torch.no_grad()
def evaluate(model, data, cfg, device, domain=None):
    model.eval()
    lam, model.st.lam = model.st.lam, 0.0
    tot, n = 0.0, 0
    for x, y in data.val_batches(cfg.val_seqs, cfg.batch, cfg.T, domain):
        logits, _, _ = model(x.to(device))
        tot += F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten(), reduction="sum").item()
        n += y.numel()
    model.st.lam = lam
    model.train()
    return tot / n


def run(cfg: Config, device="cpu", max_steps=None, out_dir=None):
    steps = max_steps or cfg.steps
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = TokenData(cfg.data, cfg.V, mix=cfg.mix)
    aux = {"counts": torch.from_numpy(np.asarray(data.token_counts))}
    if cfg.head == "two_level":
        from .clusters import build_assign, cooccurrence_features
        feats = cooccurrence_features(data.train, cfg.V) if cfg.head_clusters == "kmeans" else None
        aux["assign"] = build_assign(aux["counts"], feats, cfg.head_short, cfg.head_C, cfg.head_clusters, cfg.seed)
    facts = None
    if cfg.facts_n:
        from .data import load_tokenizer
        from .facts import Facts
        facts = Facts(cfg.facts_n, load_tokenizer(cfg.V, data.dir), cfg.seed, cfg.facts_marker)
    model = Model(cfg, aux).to(device)
    if cfg.init:                                         # fine-tuning: start from trained weights
        model.load_state_dict(torch.load(cfg.init, map_location=device))
    teacher = None
    if cfg.teacher and cfg.kd_w > 0:                     # a frozen full-precision model of the same kind as coach
        import dataclasses
        tr = json.loads((Path(cfg.teacher) / "result.json").read_text())["config"]
        tcfg = Config(**{k: v for k, v in tr.items() if k in {f.name for f in dataclasses.fields(Config)}})
        assert tcfg.V == cfg.V and tcfg.T >= cfg.T
        teacher = Model(tcfg, aux).to(device)
        teacher.load_state_dict(torch.load(Path(cfg.teacher) / "model.pt", map_location=device))
        teacher.set_progress(1.0); teacher.eval()
        for p_ in teacher.parameters(): p_.requires_grad_(False)
    # three groups: weight matrices (decayed; Adam or Muon), lookup rows (own rate), everything else
    tables = {id(p) for p in model.table_params()}
    mats = [m.weight for m in model.modules() if isinstance(m, QLinear) and m.weight.requires_grad]
    mat_ids = {id(p) for p in mats}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in tables and id(p) not in mat_ids]
    emb = list(model.emb.parameters())
    rows = [p for p in model.table_params() if all(p is not e for e in emb)]
    adam_groups = [{"params": emb, "weight_decay": 0.0, "mult": 1.0, "kind": "table"},
                   {"params": other, "weight_decay": 0.0, "mult": 1.0, "kind": "other"}]
    if rows:
        adam_groups.append({"params": rows, "weight_decay": 0.0, "mult": cfg.table_lr_mult, "kind": "table"})
    if cfg.opt == "muon":
        from .optim import Muon
        # as in the published recipe, the output head stays on Adam; Muon takes the hidden matrices
        head_w = {id(m.weight) for m in ([model.head] if isinstance(model.head, QLinear) else
                                         [model.head.root, model.head.mem] if model.head is not None else [])}
        hidden = [p for p in mats if id(p) not in head_w]
        head_mats = [p for p in mats if id(p) in head_w]
        if head_mats:
            adam_groups.append({"params": head_mats, "weight_decay": cfg.wd, "mult": 1.0, "kind": "mat"})
        opts = [Muon([{"params": hidden, "mult": cfg.muon_lr / cfg.lr, "kind": "mat"}], weight_decay=cfg.wd),
                torch.optim.AdamW(adam_groups, lr=cfg.lr, betas=(0.9, 0.95))]
    else:
        opts = [torch.optim.AdamW([{"params": mats, "weight_decay": cfg.wd, "mult": 1.0, "kind": "mat"}] + adam_groups,
                                  lr=cfg.lr, betas=(0.9, 0.95))]
    # fact rows are ADDED to the batch and drawn from their own generator, so a facts arm trains on
    # exactly the same ordinary text as its twin without facts
    frng = np.random.default_rng(cfg.seed + 1000)
    n_fact = round(cfg.batch * cfg.facts_frac) if facts else 0
    out = Path(out_dir) if out_dir else RUNS / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    ck, start, best, done_s = out / "ckpt.pt", 0, float("inf"), 0.0
    if cfg.ckpt_every and ck.exists():
        c = torch.load(ck, map_location=device, weights_only=False)
        assert c["steps"] == steps, "a checkpoint resumes only the schedule it was saved under"
        model.load_state_dict(c["model"])
        for o, sd in zip(opts, c["opts"]):
            o.load_state_dict(sd)
        rng.bit_generator.state, frng.bit_generator.state = c["rng"], c["frng"]
        torch.set_rng_state(c["torch_rng"])
        start, best, done_s = c["step"] + 1, c["best"], c["elapsed"]
    log = open(out / "log.jsonl", "a" if start else "w")
    t0, ent, first = time.time() - done_s, None, (c["first"] if start else None)
    for step in range(start, steps):
        p = step / steps
        model.set_progress(p)
        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * p))
        qmult = cfg.wq_lr_mult if (model.st.on and cfg.wq != "fp") else 1.0
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr * g["mult"] * (qmult if g["kind"] == "mat" else 1.0)
                if g["kind"] == "mat":
                    g["weight_decay"] = cfg.wd if p < cfg.wd_off_frac else 0.0
                    g["wd_lr"] = lr * qmult
        x, y = data.batch(rng, cfg.batch, cfg.T)
        if n_fact:
            fr = torch.from_numpy(facts.rows(frng, n_fact, cfg.T))
            x, y = torch.cat([x, fr[:, :-1]]), torch.cat([y, fr[:, 1:]])
        logits, aux, e = model(x.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())
        ce = loss
        if teacher is not None:
            with torch.no_grad():
                tl = teacher(x.to(device))[0]
            T_ = cfg.kd_t
            kd = F.kl_div((logits / T_).log_softmax(-1).flatten(0, 1), (tl / T_).softmax(-1).flatten(0, 1), reduction="batchmean") * T_ * T_
            loss = (1 - cfg.kd_w) * ce + cfg.kd_w * kd
        for o in opts:
            o.zero_grad(set_to_none=True)
        (loss + cfg.moe_aux * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        ent = e.item() if e is not None else None
        first = loss.item() if first is None else first
        if step % 50 == 0 or step == steps - 1:
            rec = {"step": step, "loss": round(ce.item(), 4), **({"kd": round(kd.item(), 4)} if teacher is not None else {}), "lr": lr, "lam": model.st.lam,
                   "quant_on": model.st.on, "elapsed": round(time.time() - t0, 1)}
            if (step and step % cfg.eval_every == 0) or step == steps - 1:
                rec["val"] = round(evaluate_all(model, data, cfg, device)[0], 4)
                if cfg.ckpt_every and rec["val"] < best:
                    best = rec["val"]
                    torch.save(model.state_dict(), out / "best.pt")
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if cfg.ckpt_every and step and step % cfg.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts], "step": step,
                        "steps": steps, "rng": rng.bit_generator.state, "frng": frng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "best": best, "first": first,
                        "elapsed": time.time() - t0}, out / "ckpt.tmp")
            os.replace(out / "ckpt.tmp", ck)
    model.set_progress(1.0)                      # quantisation on, residual gone: what would ship
    val, by = evaluate_all(model, data, cfg, device)
    wall = time.time() - t0
    by_domain = {k: {"nats_per_token": round(v, 4),
                     "bits_per_char": round(v / math.log(2) / data.domains[k][2], 4)} for k, v in by.items()}
    res = {"name": cfg.name, "val_nats_per_token": round(val, 4),
           "val_bits_per_char": round(val / math.log(2) / data.chars_per_token, 4),
           "val_by_domain": by_domain,
           "train_loss_first": round(first, 4), "train_loss_last": round(ce.item(), 4),
           "steps": steps, "train_tokens": steps * cfg.batch * cfg.T,
           "train_tokens_facts": steps * n_fact * cfg.T, "wall_s": round(wall, 1),
           "train_tokens_per_s": round(steps * cfg.batch * cfg.T / wall),
           "params": model.counts(), "expert_usage_entropy": ent, "gba": estimate(cfg),
           "facts": (model.eval(), facts.probe(model, device), model.train())[1] if facts else None,
           "fact_exposures": round(facts.exposures_per_fact(steps, n_fact, cfg.T), 1) if facts else None,
           "device": device, "config": config_dict(cfg)}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), out / "model.pt")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    r = run(load_config(a.config, a.set), a.device, a.max_steps)
    print(json.dumps({k: r[k] for k in ("name", "val_nats_per_token", "val_bits_per_char",
                                        "train_tokens_per_s", "val_by_domain", "wall_s", "expert_usage_entropy", "facts", "gba")}))
