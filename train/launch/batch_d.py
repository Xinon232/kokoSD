"""Follow-up to batch C: the best options combined (float and ternary, three widths), fact recall
for the ternary packages, and a diagnostic of which part of the model ternary weights hurt most.
Runs three at a time, skips anything already finished, prints a table at the end."""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T2 = [{"kind": "bigram", "rows": 16384, "layers": [0, 2], "gate": False},
      {"kind": "trigram", "rows": 16384, "layers": [0, 2], "gate": False}]
T4 = [{"kind": k, "rows": 32768, "layers": [0, 2], "gate": False, "bits": 4} for k in ("bigram", "trigram")]
COMMON = {"steps": 15000, "batch": 128, "warmup": 500, "eval_every": 3000, "val_seqs": 512, "data": "mix0"}
PK = {"mlp": "moe", "E": 64, "router": "token", "inject": T4, "table_lr_mult": 5, "copy_head": True, "topk": 8}
TERN = {"wq": "tern", "wq_lr_mult": 10, "row_gain": True}
W192 = {"D": 192, "DK": 96, "DV": 96, "HD": 384}
W256 = {"D": 256, "DK": 128, "DV": 128, "HD": 512}
OLD = {"mlp": "moe", "E": 16, "inject": T2, "topk": 8, "wq": "tern", "wq_lr_mult": 10, "facts_n": 5000}
F = {"facts_n": 5000}
RUNS = [
    ("d_pkg_tern_256", {**PK, **TERN, **W256}),
    ("d_pkg_tern_256_facts", {**PK, **TERN, **W256, **F}),
    ("d_pkg_tern_192", {**PK, **TERN, **W192}),
    ("d_pkg_tern_192_facts", {**PK, **TERN, **W192, **F}),
    ("d_pkg_tern_192_sign", {**PK, **TERN, **W192, "keyq": "sign", "att_bias": True, "DK": 192}),
    ("d_pkg_fp", PK),
    ("d_pkg_tern", {**PK, **TERN}),
    ("d_pkg_tern_seed2", {**PK, **TERN, "seed": 2}),
    ("d_pkg_tern_facts", {**PK, **TERN, **F}),
    ("d_pkg_fp_facts", {**PK, **F}),
    ("d_tern_int8_head", {"wq": "tern", "wq_lr_mult": 10, "int8_parts": ["head"]}),
    ("d_tern_int8_att", {"wq": "tern", "wq_lr_mult": 10, "int8_parts": ["att"]}),
    ("d_tern_int8_mlp", {"wq": "tern", "wq_lr_mult": 10, "int8_parts": ["mlp"]}),
    ("d_facts_old_pkg_int8_head", {**OLD, "int8_parts": ["head"]}),
    ("d_facts_old_pkg_int8_mlp", {**OLD, "int8_parts": ["mlp"]}),
    ("d_facts_old_pkg_marker", {**OLD, "facts_marker": True}),
]
only = sys.argv[1:]


def go(r):
    name, over = r
    if only and not any(name.startswith(o) for o in only):
        return name, "skipped"
    if os.path.exists(f"train/runs/{name}/result.json"):
        return name, "already done"
    sets = [f"{k}={json.dumps(v)}" for k, v in {**COMMON, **over, "name": name}.items()]
    p = subprocess.run(["python3", "-m", "train.train", "--config", "train/configs/base.json", "--set", *sets],
                       stdout=open(f"train/runs_{name}.out", "w"), stderr=subprocess.STDOUT)
    return name, f"exit {p.returncode}"


with ThreadPoolExecutor(int(os.environ.get("PAR", "3"))) as ex:
    for name, status in ex.map(go, RUNS):
        print("finished", name, status, flush=True)
for f in sorted(glob.glob("train/runs/d_*/result.json")):
    r = json.load(open(f))
    d = r["val_by_domain"]
    fa = r.get("facts") or {}
    print(r["name"], r["val_nats_per_token"], *(d[k]["nats_per_token"] for k in ("chat", "facts", "stories")),
          (fa.get("trained_wording") or {}).get("recall_among_values"), round(r["wall_s"] / 60, 1),
          r["gba"]["gba_tokens_per_s_two_step_head"], r["gba"]["rom_mb"])
