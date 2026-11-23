"""Build the mixed chat / facts / stories corpus for the pilots.

    python3 -m train.corpus --name mix0 --vocab 4096 [--quick]

Three domains, each tokenised to its own train and validation stream so the training mix is a
setting (Config.mix) and every run is scored on all three:

  chat     SODA (everyday two-person dialogues, CC BY 4.0), TinyDialogues (simple adult-child
           dialogues, MIT), everyday-conversations (short assistant chats, Apache 2.0)
  facts    Simple English Wikipedia, opening paragraphs (CC BY-SA)
  stories  TinyStories V2 (CDLA-Sharing)

Chat is written as  <|user|>text<|koko|>text...  with the second speaker as Koko. This is public
data for design experiments; Koko's own conversations and fact drills come later.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from .data import DATA_DIR, EOT, TRAIN_URL, download, read_stories, tokenise

USER, KOKO = "<|user|>", "<|koko|>"
HF = "https://huggingface.co/datasets/"
SOURCES = {
    "soda": HF + "allenai/soda/resolve/main/train.parquet",
    "soda_quick": HF + "allenai/soda/resolve/main/valid.parquet",
    "everyday": HF + "HuggingFaceTB/everyday-conversations-llama3.1-2k/resolve/main/data/train_sft-00000-of-00001.parquet",
    "tinydialogues": HF + "styfeng/TinyDialogues/resolve/main/tinydialogue_train_ordered.txt",
    "simplewiki": HF + "wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet",
}
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                       "…": "...", " ": " "})


def clean(s: str) -> str | None:
    """Fold typographic punctuation to ASCII, squeeze spaces; None if the text is not plain English."""
    s = re.sub(r"[ \t]+", " ", s.translate(_FOLD)).strip()
    if not s or sum(ord(c) > 126 for c in s) > 0.01 * len(s):
        return None
    return s.encode("ascii", "ignore").decode()


def chat_doc(turns: list[tuple[int, str]], max_turn=240, max_chars=1200) -> str | None:
    """turns: (speaker 0/1, text). Speaker 0 is the user. Consecutive turns by one speaker are joined."""
    merged: list[list] = []
    for who, text in turns:
        text = clean(text)
        if text is None:
            return None
        if merged and merged[-1][0] == who:
            merged[-1][1] += " " + text
        else:
            merged.append([who, text])
    if len(merged) < 2 or merged[0][0] != 0 or any(len(t) > max_turn for _, t in merged):
        return None
    out = ""
    for who, text in merged:
        piece = (USER if who == 0 else KOKO) + text
        if len(out) + len(piece) > max_chars:
            break
        out += piece
    return out if out.count(KOKO) >= 1 else None


def soda_docs(path: Path, limit=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    n = 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["dialogue", "speakers"]).to_pydict()
        for dia, spk in zip(t["dialogue"], t["speakers"]):
            names = list(dict.fromkeys(spk))
            if len(names) != 2 or len(dia) != len(spk):
                continue
            d = chat_doc([(names.index(s), u) for s, u in zip(spk, dia)])
            if d:
                yield d
                n += 1
                if limit and n >= limit:
                    return


def everyday_docs(path: Path):
    import pyarrow.parquet as pq
    for msgs in pq.read_table(path, columns=["messages"]).to_pydict()["messages"]:
        d = chat_doc([(0 if m["role"] == "user" else 1, m["content"]) for m in msgs if m["role"] != "system"])
        if d:
            yield d


_TD_TURN = re.compile(r"\*\*([^*]+)\*\*(?:\s*\([^)]*\))?:\s*(.*?)(?=\s*\\n\s*\\n\s*\*\*|\s*$)", re.S)


def tinydialogue_docs(path: Path, limit=None):
    """One dialogue per line: **Speaker**: text \\n\\n **Speaker**: text ... The child is the user."""
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            turns = _TD_TURN.findall(line.strip().replace("<|endoftext|>", ""))
            names = list(dict.fromkeys(s.strip() for s, _ in turns))
            if len(names) != 2:
                continue
            child = next((i for i, s in enumerate(names) if "child" in s.lower()), 0)
            d = chat_doc([(0 if names.index(s.strip()) == child else 1, t.replace("\\n", " ")) for s, t in turns])
            if d:
                yield d
                n += 1
                if limit and n >= limit:
                    return


def wiki_docs(path: Path, max_chars=1500, limit=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    n = 0
    for g in range(pf.num_row_groups):
        for text in pf.read_row_group(g, columns=["text"]).to_pydict()["text"]:
            paras = [p for p in text.split("\n") if len(p) > 80 and not p.startswith(("Category:", "=="))]
            out = ""
            for p in paras[:4]:
                p = clean(p)
                if p is None or len(out) + len(p) > max_chars:
                    break
                out += (" " if out else "") + p
            if len(out) > 200:
                yield out
                n += 1
                if limit and n >= limit:
                    return


def train_tokenizer(samples: list[str], vocab: int, path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT, USER, KOKO], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(samples, trainer=trainer, length=len(samples))
    assert tok.get_vocab_size() == vocab and tok.token_to_id(USER) == 1 and tok.token_to_id(KOKO) == 2
    tok.save(str(path))


def prepare(name="mix0", vocab=4096, quick=False, data_dir: Path = DATA_DIR, koko_dir=None) -> None:
    out = data_dir / name
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / f"meta_{vocab}.json"
    if meta_path.exists():
        print(f"{meta_path} exists, nothing to do")
        return
    raw = data_dir / "raw"
    t0 = time.time()
    lim = 20000 if quick else None
    soda = raw / ("soda_valid.parquet" if quick else "soda_train.parquet")
    download(SOURCES["soda_quick" if quick else "soda"], soda)
    download(SOURCES["everyday"], raw / "everyday.parquet")
    download(SOURCES["tinydialogues"], raw / "tinydialogues.txt", 8_000_000 if quick else None)
    download(SOURCES["simplewiki"], raw / "simplewiki.parquet")
    download(TRAIN_URL, raw / "train.txt", 1_000_000_000 if not quick else 250_000_000)
    rng = np.random.default_rng(0)
    everyday = list(everyday_docs(raw / "everyday.parquet"))
    chat = list(soda_docs(soda, lim)) + list(tinydialogue_docs(raw / "tinydialogues.txt", lim)) + everyday * (1 if quick else 20)
    facts = list(wiki_docs(raw / "simplewiki.parquet", limit=lim))
    stories = [st.translate(_FOLD) for st in read_stories(raw / "train.txt")]
    if quick:
        stories = stories[:20000]
    domains = {"chat": chat, "facts": facts, "stories": stories}
    if koko_dir:                                         # Koko's own conversations and exact fact drills
        koko = []
        for fn in ("koko_docs.txt", "drills.txt"):
            with open(Path(koko_dir) / fn, encoding="utf-8") as f:
                koko += [line.rstrip("\n") for line in f if line.strip()]
        domains = {"koko": koko, **domains}
    print({k: len(v) for k, v in domains.items()}, f"({time.time() - t0:.0f}s)", flush=True)
    for docs in domains.values():                        # a fixed shuffle, then the tail is the validation set
        rng.shuffle(docs)
    n_val = {k: min(max(len(v) // 50, 200), 6000) for k, v in domains.items()}
    tok_path = out / f"tok_{vocab}.json"
    if not tok_path.exists():
        per = 60000 if not quick else 6000
        sample = [d for v in domains.values() for d in v[:per]]
        train_tokenizer(sample, vocab, tok_path)
        print(f"trained {tok_path} ({time.time() - t0:.0f}s)", flush=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    meta = {"vocab": vocab, "name": name, "eot_id": tok.token_to_id(EOT), "user_id": tok.token_to_id(USER),
            "koko_id": tok.token_to_id(KOKO), "domains": {}}
    counts = np.zeros(vocab, dtype=np.int64)
    for k, docs in domains.items():
        tr, va = docs[:-n_val[k]], docs[-n_val[k]:]
        meta["domains"][k] = {"train": tokenise(tr, tok, out / f"{k}_train_{vocab}.bin"),
                              "val": tokenise(va, tok, out / f"{k}_val_{vocab}.bin")}
        counts += np.bincount(np.fromfile(out / f"{k}_train_{vocab}.bin", dtype=np.uint16), minlength=vocab)
        print(k, meta["domains"][k]["train"]["tokens"], "train tokens", f"({time.time() - t0:.0f}s)", flush=True)
    np.save(out / f"counts_{vocab}.npy", counts)
    (out / "samples.txt").write_text("\n\n".join(f"[{k}] {d}" for k, v in domains.items() for d in v[:8]))
    meta_path.write_text(json.dumps(meta, indent=1))        # written last: marks the set as complete
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="mix0")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--quick", action="store_true", help="small downloads, for testing the pipeline")
    ap.add_argument("--koko", help="directory with koko_docs.txt and drills.txt (adds a 'koko' domain)")
    a = ap.parse_args()
    prepare(a.name, a.vocab, a.quick, koko_dir=a.koko)
