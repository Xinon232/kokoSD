"""Fine-tuning runs (mix1, 16,384 tokens). One run per GPU (GPUS=0,1,2,3)."""
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
BASE = {**PK, **TERN, **W256, **SIGN, "DK": 256, "mixer": "diag", "att_layers": [2, 3]}   # = e_256_sign_half, 1.951


def tab(rows, layers):
    return [{"kind": k, "rows": rows, "layers": layers, "gate": False, "bits": 4} for k in ("bigram", "trigram")]


MIX = {"koko": 0.4, "chat": 0.25, "facts": 0.1, "stories": 0.25}
G = {**BASE, "V": 16384, "data": "mix1", "mix": MIX, "inject": tab(32768, [0]), "steps": 30000}
# Fine-tuning Koko (h_koko_L8) with its full-precision twin as coach, on mix2 (mix1 with corrected capitals).
L8 = {**G, "L": 8, "att_layers": [1, 3, 5, 7], "E": 128, "HD": 128}
FT = {**L8, "data": "mix2", "steps": 45000, "eval_every": 5000, "ckpt_every": 2000, "val_seqs": 1024, "warmup": 500,
      "lr": 0.0005, "init": "train/runs/h_koko_L8/model.pt"}
# The coach and its targets need about 6 GB more than a 32 GB card has at batch 128, so coached runs use
# batch 96 for 36,000 steps (shortened to fit the time available: 60% of the control's training tokens).
KD = {"teacher": "train/runs/h_koko_L8_fp", "kd_w": 0.5, "kd_t": 1.0, "batch": 96, "steps": 36000, "eval_every": 6000}
RUNS = [
    ("i_kd", {**FT, **KD}),                                  # coached
    ("i_plain", FT),                                         # control: the same extra training without the coach
    ("i_kd_t2", {**FT, **KD, "kd_w": 0.7, "kd_t": 2.0}),     # softer targets, more weight on the coach
    ("i_kd_lr1", {**FT, **KD, "lr": 0.001}),                 # a bolder learning rate
]
only = sys.argv[1:]


import queue
GPUS = queue.Queue()
for g in os.environ.get("GPUS", "0").split(","):
    GPUS.put(g)


def go(r):
    name, over = r
    g = GPUS.get()
    try:
        return _go(name, over, g)
    finally:
        GPUS.put(g)


def _go(name, over, g):
    if only and not any(name.startswith(o) for o in only):
        return name, "skipped"
    if os.path.exists(f"train/runs/{name}/result.json"):
        return name, "already done"
    sets = [f"{k}={json.dumps(v)}" for k, v in {**COMMON, **over, "name": name}.items()]
    p = subprocess.run(["python3", "-m", "train.train", "--config", "train/configs/base.json", "--set", *sets],
                       stdout=open(f"train/runs_{name}.out", "w"), stderr=subprocess.STDOUT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
    return name, f"exit {p.returncode}"


with ThreadPoolExecutor(GPUS.qsize()) as ex:
    for name, status in ex.map(go, RUNS):
        print("finished", name, status, flush=True)
for f in sorted(glob.glob("train/runs/i_*/result.json")):
    r = json.load(open(f))
    d = r["val_by_domain"]
    fa = r.get("facts") or {}
    print(r["name"], r["val_nats_per_token"], *(d[k]["nats_per_token"] for k in ("koko", "chat", "facts", "stories")),
          (fa.get("trained_wording") or {}).get("recall_among_values"), round(r["wall_s"] / 60, 1),
          r["gba"]["gba_tokens_per_s_two_step_head"], r["gba"]["rom_mb"])
