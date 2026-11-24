"""Which tokens are whole words? Used for typing suggestions (a suggestion must be a word, not a fragment).
SP: starts with a space, letters only, and is usually followed by a space / punctuation / marker.
START: no leading space, letters only, usually right after a marker (the first word of a message)."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import json, numpy as np
A = ROOT + '/artifacts/mix1x/mix1/'


def flags():
    t = json.load(open(A + 'tok_16384.json'))
    vocab = t["model"]["vocab"]; V = len(vocab)
    inv = sorted(vocab, key=vocab.get)
    sp = np.array([s.startswith("Ġ") for s in inv])
    alpha_sp = np.array([s.startswith("Ġ") and len(s) > 1 and all(c.isascii() and (c.isalpha() or c == "'") for c in s[1:]) and s[1].isalpha() for s in inv])
    alpha_st = np.array([all(c.isascii() and c.isalpha() for c in s) and len(s) > 0 for s in inv])
    ends = np.array([s.startswith("Ġ") or not (s[:1].isascii() and (s[:1].isalpha() or s[:1] == "'")) for s in inv])   # a token that ends the word before it
    ends[:3] = True
    cnt = np.zeros(V); fol = np.zeros(V); pre = np.zeros(V)
    for d in ("koko", "chat", "stories", "facts"):
        ids = np.fromfile(A + f'{d}_train_16384.bin', dtype=np.uint16)[:60_000_000].astype(np.int64)
        cnt += np.bincount(ids[1:-1], minlength=V)
        fol += np.bincount(ids[1:-1], weights=ends[ids[2:]], minlength=V)
        pre += np.bincount(ids[1:-1], weights=(ids[:-2] < 3), minlength=V)
    f = fol / np.maximum(cnt, 1); p = pre / np.maximum(cnt, 1)
    word_sp = alpha_sp & (cnt >= 50) & (f >= 0.6)
    word_st = alpha_st & (cnt >= 20) & (f >= 0.6) & (p >= 0.5)
    return word_sp, word_st, inv


if __name__ == "__main__":
    a, b, inv = flags()
    print("whole words after a space:", int(a.sum()), "| first words of a message:", int(b.sum()))
    print([inv[i] for i in np.nonzero(a)[0][:40]]); print([inv[i] for i in np.nonzero(b)[0][:60]])
