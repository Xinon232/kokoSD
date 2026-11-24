"""Generate raw conversations with an open model through vLLM, one shard file at a time.

    python -m gen.run --model <local copy of Qwen2.5-7B-Instruct> --out gen_out --shards 0 100 [--per 20000] [--only chat]

Each shard is deterministic in its number, so the run can be stopped and resumed; finished shards
are skipped. Output: <out>/raw_<shard>.jsonl with kind, meta and the raw text.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def load_pool(data_raw: Path):
    """Encyclopedia paragraphs and a list of common words, both from Simple English Wikipedia."""
    import collections
    import re
    import pyarrow.parquet as pq
    cache = data_raw / "gen_pool.json"
    if cache.exists():
        return json.loads(cache.read_text())
    t = pq.read_table(data_raw / "simplewiki.parquet", columns=["title", "text"]).to_pydict()
    wiki, counts = [], collections.Counter()
    for title, text in zip(t["title"], t["text"]):
        paras = [p.strip() for p in text.split("\n") if len(p) > 120 and not p.startswith(("Category:", "=="))]
        for p in paras[:3]:
            if len(p) <= 900 and all(ord(c) < 127 for c in p) and all(ord(c) < 127 for c in title):
                wiki.append([title, p])
        counts.update(re.findall(r"\b[a-z]{3,12}\b", " ".join(paras[:3])))
    words = [w for w, _ in counts.most_common(9000)][150:]
    pool = {"wiki": wiki, "words": words}
    cache.write_text(json.dumps(pool))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="train/data/raw")
    ap.add_argument("--shards", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--per", type=int, default=20000)
    ap.add_argument("--only")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from .prompts import build
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(Path(a.raw))
    print("pool:", len(pool["wiki"]), "paragraphs,", len(pool["words"]), "words", flush=True)
    llm = LLM(a.model, max_model_len=2048, gpu_memory_utilization=float(os.environ.get("GPU_FRAC", "0.9")))
    tok = llm.get_tokenizer()
    for shard in range(*a.shards):
        path = out / f"raw_{shard:05d}.jsonl"
        if path.exists():
            continue
        try:                                   # several workers share the shard list: first to claim wins
            os.close(os.open(path.with_suffix(".claim"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        t0 = time.time()
        ps = build(a.per, 1_000_003 * shard + 17, pool, a.only)
        texts = [tok.apply_chat_template([{"role": "system", "content": p["system"]},
                                          {"role": "user", "content": p["user"]}],
                                         tokenize=False, add_generation_prompt=True) + p["prefill"] for p in ps]
        params = [SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=p["max_new"]) for p in ps]
        res = llm.generate(texts, params)
        n_tok = sum(len(r.outputs[0].token_ids) for r in res)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for p, r in zip(ps, res):
                f.write(json.dumps({"kind": p["kind"], "meta": p["meta"], "raw": p["prefill"] + r.outputs[0].text,
                                    "finish": r.outputs[0].finish_reason}) + "\n")
        tmp.rename(path)
        dt = time.time() - t0
        print(f"shard {shard}: {len(ps)} conversations, {n_tok} tokens, {dt:.0f}s, {n_tok / dt:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
