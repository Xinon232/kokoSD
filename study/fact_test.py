"""Drilled-fact recall: ask each fact once in its plainest wording, greedy reply, is the table value in it?"""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import sys, json, random, re, torch
sys.path.insert(0, ROOT)
from common import load, tokenizer
from gen import drills as D
torch.set_num_threads(4)
run = sys.argv[1] if len(sys.argv) > 1 else "h_koko_L8"
model, cfg, counts, res = load(run)
tok = tokenizer()
countries = D.load_countries(ROOT + '/artifacts/kd/countries.json')
rng = random.Random(7)


def reply(q, n=28):
    ids = tok.encode(f"<|user|>{q}<|koko|>").ids
    out = []
    with torch.no_grad():
        for _ in range(n):
            nxt = model(torch.tensor([ids]))[0][0, -1].argmax().item()
            ids.append(nxt); out.append(nxt)
            s = tok.decode(out, skip_special_tokens=False)
            if "<|" in s:
                return s.split("<|")[0].strip()
    return tok.decode(out, skip_special_tokens=False).strip()


tests = []
for f in countries:
    for rel, q in (("capital", "What is the capital of {s}?"), ("continent", "What continent is {s} in?"),
                   ("language", "What language do people speak in {s}?"), ("currency", "What currency does {s} use?")):
        if rel in f:
            tests.append((rel, q.format(s=f["s"]), str(f[rel])))
for s, v in D.STATES:
    tests.append(("state capital", f"What is the capital of {s}?", v))
for op, name, lo in (("+", "plus", 50), ("-", "minus", 60), ("x", "times", 12)):
    for _ in range(80):
        a, b = rng.randint(0, lo), rng.randint(0, lo)
        if op == "-": a, b = max(a, b), min(a, b)
        v = a + b if op == "+" else a - b if op == "-" else a * b
        tests.append((f"arithmetic {name}", f"What is {a} {name} {b}?", str(v)))
for i in range(12):
    tests.append(("calendar", f"What month comes after {D.MONTHS[i]}?", D.MONTHS[(i + 1) % 12]))
for i in range(7):
    tests.append(("calendar", f"What day comes after {D.DAYS[i]}?", D.DAYS[(i + 1) % 7]))
for i in range(8):
    tests.append(("calendar", f"What is the {D.ORD[i]} planet from the Sun?", D.PLANETS[i]))
if len(sys.argv) > 2:
    rng.shuffle(tests); tests = tests[:int(sys.argv[2])]
score, wrong = {}, {}
for k, (rel, q, v) in enumerate(tests):
    r = reply(q)
    if rel.startswith("arithmetic"):
        ok = v in re.findall(r"-?\d+", r.replace(q, ""))[-1:]            # the last number in the reply is the answer
    else:
        ok = v.lower() in r.lower()
    a = score.setdefault(rel, [0, 0]); a[0] += ok; a[1] += 1
    if not ok and len(wrong.setdefault(rel, [])) < 6:
        wrong[rel].append((q, v, r))
    if k % 100 == 0: print(k, len(tests), flush=True)
print(json.dumps({k: f"{a}/{b} = {a / b:.1%}" for k, (a, b) in score.items()}, indent=1))
for k, w in wrong.items():
    for q, v, r in w[:4]: print(f"  [{k}] {q}  want {v!r}  got {r!r}")
json.dump({"run": run, "score": score, "wrong": wrong}, open(f"fact_test_{run}.json", "w"), indent=1)
