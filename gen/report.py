"""Quality report on raw shards: keep rate per kind, drop reasons, repeated openings, samples.

    python -m gen.report gen_out [--samples 3]
"""
from __future__ import annotations

import collections
import glob
import json
import sys

from .parse import opener_key, parse


def main(out, n_samples=3):
    kept, total, why = collections.Counter(), collections.Counter(), collections.Counter()
    openers, samples, chars = collections.Counter(), collections.defaultdict(list), collections.Counter()
    bad = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{out}/raw_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            total[r["kind"]] += 1
            doc, reason = parse(r["raw"], r["kind"], r["meta"]) if r["finish"] == "stop" else (None, "cut_off")
            if doc is None:
                why[(r["kind"], reason)] += 1
                if len(bad[reason]) < 2:
                    bad[reason].append(r["raw"][:300])
                continue
            kept[r["kind"]] += 1
            chars[r["kind"]] += len(doc)
            openers[opener_key(doc)] += 1
            if len(samples[r["kind"]]) < n_samples:
                samples[r["kind"]].append(doc)
    n = sum(kept.values())
    print(f"kept {n} of {sum(total.values())}")
    for k in total:
        print(f"  {k}: {kept[k]}/{total[k]} = {kept[k] / total[k]:.1%}, mean {chars[k] / max(kept[k], 1):.0f} chars")
    print("drop reasons:", sorted(why.items(), key=lambda x: -x[1])[:25])
    print("top openings:", [(o, c, f"{c / max(n, 1):.1%}") for o, c in openers.most_common(12)])
    print("distinct openings per 10K:", round(len(openers) / max(n, 1) * 10000))
    for k, docs in samples.items():
        for d in docs:
            print(f"\n[{k}] {d}")
    for reason, raws in bad.items():
        for r in raws:
            print(f"\n[DROPPED {reason}] {r!r}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
