"""Tokenizer tables for the cartridge: byte -> token, merge table sorted for binary search, token text."""
import json, numpy as np
A = '/home/user/kokosd/artifacts/mix1x/mix1/tok_16384.json'


def tables():
    t = json.load(open(A))
    vocab = t["model"]["vocab"]
    inv = sorted(vocab, key=vocab.get)
    byte_id = np.zeros(128, dtype=np.int64)
    for b in range(32, 127):
        byte_id[b] = vocab["Ġ" if b == 32 else chr(b)]
    keys, vals = [], []
    for rank, (a, b) in enumerate(t["model"]["merges"]):
        keys.append(vocab[a] << 14 | vocab[b]); vals.append(rank << 16 | vocab[a + b])
    o = np.argsort(keys)
    text, off = bytearray(), [0]
    for s in inv:
        if s.startswith("<|") and s.endswith("|>") and vocab[s] < 3:
            raw = b""                                                  # markers print nothing
        else:
            raw = bytes(32 if ch == "Ġ" else (ord(ch) if 33 <= ord(ch) < 127 else 63) for ch in s)   # non-ASCII -> '?'
        text += raw; off.append(len(text))
    return {"byte": byte_id, "mkey": np.array(keys)[o], "mval": np.array(vals)[o], "off": np.array(off), "text": np.frombuffer(bytes(text), dtype=np.uint8)}
