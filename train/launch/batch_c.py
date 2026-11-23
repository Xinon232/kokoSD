"""Chat-first pilot batch: reference arms, mix comparison, then every s-arm, all on the mixed corpus.
Runs four at a time, skips anything already finished, and prints a table at the end."""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T2 = [{"kind": "bigram", "rows": 16384, "layers": [0, 2], "gate": False},
      {"kind": "trigram", "rows": 16384, "layers": [0, 2], "gate": False}]
COMMON = {"steps": 15000, "batch": 128, "warmup": 500, "eval_every": 3000, "val_seqs": 512, "data": "mix0"}
RUNS = [  # (name, config, overrides) in priority order
    ("c_ref_fp", "p00_fp", {}),
    ("c_ref_tern_lr10", "p02_tern", {"wq_lr_mult": 10}),
    ("c_ref_topk8", "p06_topk8", {}),
    ("c_ref_moe16", "p08_moe16", {}),
    ("c_ref_tables", "s54_tab_float_ref", {}),
    ("c_ref_moe16_tables", "p08_moe16", {"inject": T2}),
    ("c_ref_pkg_tern", "p00_fp", {"wq": "tern", "wq_lr_mult": 10, "topk": 8, "mlp": "moe", "E": 16, "inject": T2}),
    ("c_mix_story_heavy", "p00_fp", {"mix": {"chat": 0.2, "facts": 0.1, "stories": 0.7}}),
    ("c_mix_even", "p00_fp", {"mix": {"chat": 0.34, "facts": 0.33, "stories": 0.33}}),
    ("c_mix_chat_heavy", "p00_fp", {"mix": {"chat": 0.7, "facts": 0.2, "stories": 0.1}}),
    ("c_ref_fp_seed2", "p00_fp", {"seed": 2}),
]
RUNS += [("c_" + os.path.basename(f)[:-5], os.path.basename(f)[:-5], {})
         for f in sorted(glob.glob("train/configs/s*.json")) if "s54_tab_float_ref" not in f]
only = sys.argv[1:]


def go(r):
    name, cfg, over = r
    if only and not any(name.startswith(o) for o in only):
        return name, "skipped"
    if os.path.exists(f"train/runs/{name}/result.json"):
        return name, "already done"
    sets = [f"{k}={json.dumps(v)}" for k, v in {**COMMON, **over, "name": name}.items()]
    p = subprocess.run(["python3", "-m", "train.train", "--config", f"train/configs/{cfg}.json", "--set", *sets],
                       stdout=open(f"train/runs_{name}.out", "w"), stderr=subprocess.STDOUT)
    return name, f"exit {p.returncode}"


os.makedirs("train/runs", exist_ok=True)
with ThreadPoolExecutor(4) as ex:
    for name, status in ex.map(go, RUNS):
        print("finished", name, status, flush=True)
for f in sorted(glob.glob("train/runs/c_*/result.json")):
    r = json.load(open(f))
    d = r["val_by_domain"]
    fa = r.get("facts") or {}
    print(r["name"], r["val_nats_per_token"], *(d[k]["nats_per_token"] for k in ("chat", "facts", "stories")),
          (fa.get("trained_wording") or {}).get("recall_among_values"), round(r["wall_s"] / 60, 1),
          r["gba"]["gba_tokens_per_s_two_step_head"], r["gba"]["rom_mb"])
