"""Closed-book questions, greedy answers, checked by pattern. Chat models are asked in their chat format; plain
language models are given "Question: ...\\nAnswer:" and their first line is taken."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import sys, re, json
sys.path.insert(0, ROOT + '/scoreboard')
import models as M
from qa_sets import CAPITALS, GENERAL, arithmetic
SETS = {"capitals": [(f"What is the capital of {c}?", p) for c, p in CAPITALS.items()], "general": GENERAL,
        "arithmetic": [(q, v) for q, v in arithmetic()]}
COLS = ["Koko (cartridge, model only)", "Koko (cartridge, with fact table and calculator)", "SmolLM2-135M-Instruct", "SmolLM2-135M", "GPT-2 small", "TinyStories-33M"]


def ask(col, q):
    if col.startswith("Koko"):
        return M.koko_reply([("user", q)], temp=0.05, maxn=40, helpers="with" in col)
    if col == "SmolLM2-135M-Instruct":
        return M.hf_generate(col, q, 40, chat=True)
    out = M.hf_generate(col, f"Question: {q}\nAnswer:", 24).strip()
    return out.split("\n")[0]


def right(kind, ans, key):
    a = ans.lower()
    if kind == "arithmetic":
        nums = re.findall(r"-?\d+", a)
        return bool(nums) and nums[-1] == str(key) if "=" in a or " is " in a else str(key) in nums[:1]
    return re.search(key, a) is not None


res, log = {}, {}
for kind, items in SETS.items():
    for col in COLS:
        ok = 0
        for q, key in items:
            ans = ask(col, q); r = right(kind, ans, key); ok += r
            log.setdefault(kind, {}).setdefault(q, {})[col] = [ans[:160], bool(r)]
        res.setdefault(col, {})[kind] = [ok, len(items)]
        print(f"{kind:10s} {col:50s} {ok}/{len(items)}", flush=True)
json.dump({"results": res, "answers": log}, open(ROOT + '/scoreboard/qa.json', 'w'), indent=1)
