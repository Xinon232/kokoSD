"""Raw generator shards -> one clean training document per line.

    python -m gen.assemble gen_out [--opener-cap 40]

Parses every conversation (gen/parse.py), drops exact duplicates, and caps how often one opening
of Koko's first reply may appear per 10,000 kept documents. Writes <out>/koko_docs.txt and
<out>/assemble.json (counts and drop reasons).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json

from .parse import opener_key, parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--opener-cap", type=int, default=40, help="max documents per opening per 10,000 kept")
    a = ap.parse_args()
    seen, openers = set(), collections.Counter()
    kept, total, why, chars = collections.Counter(), collections.Counter(), collections.Counter(), 0
    files = sorted(glob.glob(f"{a.out}/raw_*.jsonl"))
    with open(f"{a.out}/koko_docs.txt", "w") as f:
        for path in files:
            for line in open(path):
                r = json.loads(line)
                total[r["kind"]] += 1
                doc, reason = parse(r["raw"], r["kind"], r["meta"]) if r["finish"] == "stop" else (None, "cut_off")
                if doc is None:
                    why[reason] += 1
                    continue
                h = hashlib.md5(doc.encode()).digest()
                if h in seen:
                    why["duplicate"] += 1
                    continue
                key, n = opener_key(doc), sum(kept.values())
                if r["kind"] != "identity" and openers[key] >= max(a.opener_cap, a.opener_cap * n // 10000):
                    why["opening_repeated"] += 1
                    continue
                seen.add(h)
                openers[key] += 1
                kept[r["kind"]] += 1
                chars += len(doc)
                f.write(doc + "\n")
    info = {"shards": len(files), "total": dict(total), "kept": dict(kept), "kept_all": sum(kept.values()),
            "chars": chars, "dropped": dict(why.most_common()), "top_openings": openers.most_common(15)}
    json.dump(info, open(f"{a.out}/assemble.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
