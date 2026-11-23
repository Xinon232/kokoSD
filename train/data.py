"""Data for the Koko training harness.

Preparation (run once):   python3 -m train.data [--vocab 4096] [--train-bytes 250000000]

  1. download  the first ~250 MB of TinyStoriesV2-GPT4-train.txt (HTTP range request) and the
               whole TinyStoriesV2-GPT4-valid.txt into train/data/raw/
  2. tokenizer byte-level BPE (Hugging Face `tokenizers`), <|endoftext|> as the only special token
  3. tokenise  every story separately, followed by <|endoftext|>, into one uint16 stream per split
               (train_V.bin, val_V.bin) plus meta_V.json (sizes, characters per token) and
               counts_V.npy (training-token frequencies, used by the two-level head)

At training time `TokenData` serves random crops of the training stream and a fixed validation set.
`data="synthetic"` in the config replaces the files by a small seeded Markov stream (tests, smoke runs).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_URL = HF + "TinyStoriesV2-GPT4-train.txt"
VALID_URL = HF + "TinyStoriesV2-GPT4-valid.txt"
EOT = "<|endoftext|>"
DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------- download

def download(url: str, dest: Path, n_bytes: int | None = None) -> None:
    """Fetch `url` into `dest`; with n_bytes only bytes [0, n_bytes) (range request). Resumable."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.with_suffix(dest.suffix + ".done")
    if done.exists():
        return
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if n_bytes is not None and have > n_bytes:
        raise RuntimeError(f"{part} is larger than the requested {n_bytes} bytes; delete it")
    if n_bytes is None or have < n_bytes:
        last = "" if n_bytes is None else str(n_bytes - 1)
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{last}",
                                                   "User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
            if r.status != 206:
                raise RuntimeError(f"expected HTTP 206 (partial content) from {url}, got {r.status}")
            t0 = time.time()
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                if time.time() - t0 > 10:
                    print(f"  {dest.name}: {have / 1e6:.0f} MB", flush=True)
                    t0 = time.time()
    if n_bytes is not None and have != n_bytes:
        raise RuntimeError(f"{url}: got {have} bytes, wanted {n_bytes} (re-run to resume)")
    os.replace(part, dest)
    done.write_text(json.dumps({"url": url, "bytes": have}))
    print(f"downloaded {dest} ({have / 1e6:.1f} MB)")


def read_stories(path: Path) -> list[str]:
    """Split a TinyStories file into stories. A trailing story cut off by the range request is dropped."""
    text = path.read_bytes().decode("utf-8", errors="ignore")
    parts = text.split(EOT)
    complete = parts[:-1]            # whatever follows the last separator is empty or truncated
    return [s.strip() for s in complete if s.strip()]


# --------------------------------------------------------------------------- tokenizer

def train_tokenizer(stories: list[str], vocab: int, path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(stories, trainer=trainer, length=len(stories))
    if tok.get_vocab_size() != vocab:
        raise RuntimeError(f"tokenizer has {tok.get_vocab_size()} tokens, wanted {vocab}")
    tok.save(str(path))


def load_tokenizer(vocab: int, data_dir: Path = DATA_DIR):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(data_dir / f"tok_{vocab}.json"))


def tokenise(stories: list[str], tok, out: Path, chunk: int = 20000) -> dict:
    """Write EOT + (story tokens + EOT)* as uint16. Returns sizes for the bits-per-character metric."""
    eot = tok.token_to_id(EOT)
    assert eot is not None and tok.get_vocab_size() <= 65536
    pieces = [np.array([eot], dtype=np.uint16)]
    for i in range(0, len(stories), chunk):
        for enc in tok.encode_batch(stories[i:i + chunk]):
            pieces.append(np.array(enc.ids + [eot], dtype=np.uint16))
    ids = np.concatenate(pieces)
    ids.tofile(out)
    # The separator counts as one character; the leading EOT is context only (never a target).
    n_chars = sum(len(s) for s in stories) + len(stories)
    n_bytes = sum(len(s.encode("utf-8")) for s in stories) + len(stories)
    n_targets = len(ids) - 1
    return {"tokens": int(len(ids)), "stories": len(stories), "chars": int(n_chars), "bytes": int(n_bytes),
            "chars_per_token": n_chars / n_targets, "bytes_per_token": n_bytes / n_targets}


def prepare(vocab: int = 4096, train_bytes: int = 250_000_000, data_dir: Path = DATA_DIR) -> None:
    raw = data_dir / "raw"
    download(TRAIN_URL, raw / "train.txt", train_bytes)
    download(VALID_URL, raw / "valid.txt", None)
    meta_path = data_dir / f"meta_{vocab}.json"
    if meta_path.exists():
        print(f"{meta_path} exists, nothing to do")
        return
    t0 = time.time()
    train, valid = read_stories(raw / "train.txt"), read_stories(raw / "valid.txt")
    print(f"{len(train)} training stories, {len(valid)} validation stories ({time.time() - t0:.0f}s)")
    tok_path = data_dir / f"tok_{vocab}.json"
    if not tok_path.exists():
        train_tokenizer(train, vocab, tok_path)
        print(f"trained {tok_path} ({time.time() - t0:.0f}s)")
    tok = load_tokenizer(vocab, data_dir)
    meta = {"vocab": vocab, "eot_id": tok.token_to_id(EOT), "train_bytes_requested": train_bytes,
            "train": tokenise(train, tok, data_dir / f"train_{vocab}.bin"),
            "val": tokenise(valid, tok, data_dir / f"val_{vocab}.bin")}
    ids = np.fromfile(data_dir / f"train_{vocab}.bin", dtype=np.uint16)
    np.save(data_dir / f"counts_{vocab}.npy", np.bincount(ids, minlength=vocab).astype(np.int64))
    meta_path.write_text(json.dumps(meta, indent=1))        # written last: marks the set as complete
    print(json.dumps(meta, indent=1), f"\n({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- serving

def synthetic_stream(n: int, vocab: int, seed: int) -> np.ndarray:
    """A learnable toy stream: Zipf-distributed tokens, 60% of which are followed by a fixed successor."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1)
    p /= p.sum()
    succ = rng.permutation(vocab)
    draws = rng.choice(vocab, size=n, p=p)
    follow = rng.random(n) < 0.6
    out = np.empty(n, dtype=np.uint16)
    out[0] = draws[0]
    for i in range(1, n):
        out[i] = succ[out[i - 1]] if follow[i] else draws[i]
    return out


class TokenData:
    """Token streams plus the two things the model and metrics need from them."""

    def __init__(self, source: str, vocab: int, data_dir: Path | str | None = None, mix: dict | None = None):
        self.vocab = vocab
        self.domains = None                 # mixed corpora: {domain: (train stream, val stream, chars per token)}
        self.dir = Path(data_dir) if data_dir else DATA_DIR
        if source.startswith("mix"):
            self.dir = self.dir / source
            meta_path = self.dir / f"meta_{vocab}.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"{meta_path} missing: run `python3 -m train.corpus --name {source} --vocab {vocab}`")
            meta = json.loads(meta_path.read_text())
            self.domains = {}
            for k, m in meta["domains"].items():
                tr = np.memmap(self.dir / f"{k}_train_{vocab}.bin", dtype=np.uint16, mode="r")
                va = np.memmap(self.dir / f"{k}_val_{vocab}.bin", dtype=np.uint16, mode="r")
                assert len(tr) == m["train"]["tokens"] and len(va) == m["val"]["tokens"]
                self.domains[k] = (tr, va, m["val"]["chars_per_token"])
            mix = mix or {k: 1.0 for k in self.domains}
            assert set(mix) <= set(self.domains) and sum(mix.values()) > 0, mix
            self.mix = {k: v / sum(mix.values()) for k, v in mix.items() if v > 0}
            self.token_counts = np.load(self.dir / f"counts_{vocab}.npy")
            # a sample of every domain, for statistics such as token co-occurrence
            self.train = np.concatenate([np.asarray(t[:8_000_000]) for t, _, _ in self.domains.values()])
            self.val = None
            self.chars_per_token = float(np.mean([c for _, _, c in self.domains.values()]))
            self.bytes_per_token = self.chars_per_token
            return
        if source == "synthetic":
            self.train = synthetic_stream(60_000, vocab, seed=1234)
            self.val = synthetic_stream(12_000, vocab, seed=4321)
            self.token_counts = np.bincount(self.train, minlength=vocab).astype(np.int64)
            self.chars_per_token = self.bytes_per_token = 1.0   # so bits/char == bits/token
        elif source == "tinystories":
            d = Path(data_dir) if data_dir else DATA_DIR
            meta_path = d / f"meta_{vocab}.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"{meta_path} missing: run `python3 -m train.data --vocab {vocab}`")
            meta = json.loads(meta_path.read_text())
            self.train = np.memmap(d / f"train_{vocab}.bin", dtype=np.uint16, mode="r")
            self.val = np.memmap(d / f"val_{vocab}.bin", dtype=np.uint16, mode="r")
            assert len(self.train) == meta["train"]["tokens"] and len(self.val) == meta["val"]["tokens"]
            self.token_counts = np.load(d / f"counts_{vocab}.npy")
            self.chars_per_token = meta["val"]["chars_per_token"]
            self.bytes_per_token = meta["val"]["bytes_per_token"]
        else:
            raise ValueError(f"unknown data source {source!r} (use 'tinystories' or 'synthetic')")
        assert int(self.train.max() if source == "synthetic" else vocab - 1) < vocab

    def token_rank(self) -> np.ndarray:
        """rank[token] = 0 for the most frequent training token (ties broken by token id)."""
        order = np.lexsort((np.arange(self.vocab), -self.token_counts))
        rank = np.empty(self.vocab, dtype=np.int64)
        rank[order] = np.arange(self.vocab)
        return rank

    @staticmethod
    def _crops(stream: np.ndarray, starts: np.ndarray, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        win = stream[starts[:, None] + np.arange(T + 1)[None, :]].astype(np.int64)
        win = torch.from_numpy(win)
        return win[:, :-1], win[:, 1:]

    def batch(self, rng: np.random.Generator, B: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """B random crops of T tokens (inputs) and their next tokens (targets). For a mixed corpus the
        rows are split between domains by the configured shares (same split every batch)."""
        if self.domains is None:
            starts = rng.integers(0, len(self.train) - T - 1, size=B)
            return self._crops(self.train, starts, T)
        exact = {k: v * B for k, v in self.mix.items()}
        n = {k: int(v) for k, v in exact.items()}
        for k in sorted(exact, key=lambda k: exact[k] - n[k], reverse=True)[:B - sum(n.values())]:
            n[k] += 1
        xs, ys = [], []
        for k in self.mix:
            if n[k]:
                tr = self.domains[k][0]
                x, y = self._crops(tr, rng.integers(0, len(tr) - T - 1, size=n[k]), T)
                xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def val_batches(self, n_seqs: int, B: int, T: int, domain: str | None = None):
        """The same n_seqs evenly spaced validation windows every time, in batches of B. A mixed corpus
        needs a domain; if it holds fewer disjoint windows than asked, all of them are used."""
        val = self.val if domain is None else self.domains[domain][1]
        n_max = (len(val) - 1) // T
        if n_seqs > n_max:
            if domain is None:
                raise ValueError(f"validation stream holds only {n_max} disjoint windows of {T}, asked for {n_seqs}")
            n_seqs = n_max
        starts = np.linspace(0, len(val) - T - 1, n_seqs).astype(np.int64)
        for i in range(0, n_seqs, B):
            yield self._crops(val, starts[i:i + B], T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--train-bytes", type=int, default=250_000_000)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    a = ap.parse_args()
    prepare(a.vocab, a.train_bytes, Path(a.data_dir))
