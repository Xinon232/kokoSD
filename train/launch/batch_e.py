"""Shape check before the full-size run: 256 wide with 1-bit keys, attention in half the layers, 6 layers."""
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
SIGN = {"keyq": "sign", "att_bias": True}
HALF = {"mixer": "diag"}
RUNS = [
    ("e_256_sign", {**PK, **TERN, **W256, **SIGN, "DK": 256}),
    ("e_256_sign_half", {**PK, **TERN, **W256, **SIGN, "DK": 256, **HALF, "att_layers": [2, 3]}),
    ("e_256_L6_sign_alt", {**PK, **TERN, **W256, **SIGN, "DK": 256, **HALF, "L": 6, "att_layers": [1, 3, 5]}),
    ("e_192_L6_sign", {**PK, **TERN, **W192, **SIGN, "DK": 192, "L": 6}),
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


with ThreadPoolExecutor(int(os.environ.get("PAR", "2"))) as ex:
    for name, status in ex.map(go, RUNS):
        print("finished", name, status, flush=True)
for f in sorted(glob.glob("train/runs/e_*/result.json")):
    r = json.load(open(f))
    d = r["val_by_domain"]
    fa = r.get("facts") or {}
    print(r["name"], r["val_nats_per_token"], *(d[k]["nats_per_token"] for k in ("chat", "facts", "stories")),
          (fa.get("trained_wording") or {}).get("recall_among_values"), round(r["wall_s"] / 60, 1),
          r["gba"]["gba_tokens_per_s_two_step_head"], r["gba"]["rom_mb"])
