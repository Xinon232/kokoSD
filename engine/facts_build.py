"""The cartridge's fact table: the opening sentence of the longest Simple English Wikipedia articles, stored as
token ids, looked up by a hash of the article title. Text: Simple English Wikipedia, CC BY-SA.

    table  = sorted uint32 title hashes | uint32 byte offsets | sentences (1 length byte + 14-bit token ids)
"""
import re, sys, json, numpy as np
sys.path.insert(0, '/home/user/kokosd/study')
RAW = '/home/user/kokosd/train/data/raw/simplewiki.parquet'
BUDGET = 1_890_000                      # bytes on the cartridge for sentences and their keys (aliases add a little)
# words people use in chat that also happen to be article titles: never answered from the table
STOP = set("""you me i we they it he she this that these those what who whom why how where when which your my mine our us them
his her up down there here now then today tomorrow yesterday new wrong happening going on name your name my name time love life
fun game games good bad best favorite favourite matter problem answer question thing things stuff nothing something anything
everything one two three koko ok okay yes no hi hello hey thanks thank bye goodbye lol well so and or but joke story poem song
weather day night morning""".split()) | {"going on", "your name", "my name", "the matter", "the time", "the weather", "the answer"}


def norm(s):
    """Title or question subject -> lookup key: lowercase letters, digits and single spaces; no leading article."""
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r" +", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def fnv(s):
    h = 0x811C9DC5
    for c in s.encode():
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h


def ascii_fold(s):
    """Accented letters to plain ones (Pokemon, Hindustan); anything else non-ASCII is left for the caller to reject."""
    import unicodedata
    s = s.replace("\u0142", "l").replace("\u0141", "L").replace("\u00f8", "o").replace("\u00d8", "O").replace("\u00df", "ss").replace("\u00e6", "ae")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


SENT_END = r"(?<=[a-z0-9\)\"'])\.(?= +[A-Z\"]|$)"


def first_sentence(text, title=""):
    for p in text.split("\n"):
        p = p.strip()
        if len(p) < 40 or p.startswith(("Category:", "==", "*", "|", "{", "[[")):
            continue
        p = p.replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace(" ", " ")
        p = re.sub(r"\s*\((?:[^()]|\([^()]*\))*[^\x00-\x7f](?:[^()]|\([^()]*\))*\)", "", ascii_fold(p) if False else p)   # drop (pronunciation, native script)
        p = ascii_fold(p)
        p = re.sub(r"\[\d+\]", "", p)
        p = re.sub(r"\s*\((?:[\s,;:]|\b(?:or|and|listen)\b)*\)", "", p)                      # what is left of stripped templates
        p = re.sub(r"\s*\([A-Z][A-Za-z ]{1,20}:\s*\)", "", p)                                  # "(Hindi: )" with the script gone
        if re.match(r"(is|are|was|were) ", p) and title:        # the subject was a template that did not survive
            p = title + " " + p
        m = re.search(SENT_END, p)
        s = (p[:m.end()] if m else p).strip()
        if m and len(s) < 60:                                   # "Snakes are reptiles." is true but thin: take the next sentence too
            m2 = re.search(SENT_END, p[m.end():])
            if m2 and m.end() + m2.end() <= 200:
                s = p[:m.end() + m2.end()].strip()
        s = re.sub(r" +", " ", s)
        if not (20 <= len(s) <= 235) or any(ord(c) > 126 or ord(c) < 32 for c in s) or not s.endswith("."):
            return None
        if re.search(r"\b(may refer to|can mean|could mean|might mean|can refer to|disambiguation)\b", s) or s.endswith(":."):
            return None
        if re.match(r"(This article|This page|The correct title|For other|This is a list|See also|Not to be confused|The following)", s):
            return None                                       # a note about the article, not about its subject
        return s
    return None


def build():
    import pyarrow.parquet as pq
    from common import tokenizer
    tok = tokenizer()
    t = pq.read_table(RAW, columns=["title", "text"]).to_pydict()
    common = set(json.load(open('/home/user/kokosd/train/data/raw/gen_pool.json'))["words"]) if __import__("os").path.exists('/home/user/kokosd/train/data/raw/gen_pool.json') else set()
    cands = {}
    for title, text in zip(t["title"], t["text"]):
        title = ascii_fold(title)
        if "(" in title or ":" in title or any(ord(c) > 126 for c in title) or re.fullmatch(r"[\d ]+|.*\b(List of|Lists of)\b.*", title):
            continue
        key = norm(title)
        if not key or key in STOP or len(key) < 2 or key.isdigit():
            continue
        s = first_sentence(text, title)
        if s is None:
            continue
        # longest articles first, as a rough measure of importance; everyday words count five times
        score = len(text) * (5 if key in common or key.rstrip("s") in common else 1)
        if key not in cands or score > cands[key][0]:
            cands[key] = (score, title, s, len(text))
    order = sorted(cands, key=lambda k: -cands[k][0])
    enc = tok.encode_batch([cands[k][2] for k in order[:60000]])
    used, rows, seen = 0, [], set()
    for k, e in zip(order, enc):
        ids = e.ids
        if len(ids) > 60 or tok.decode(ids) != cands[k][2]:
            continue
        h = fnv(k)
        size = 8 + 1 + (len(ids) * 14 + 7) // 8
        if h in seen or used + size > BUDGET:
            if used + size > BUDGET: break
            continue
        seen.add(h); rows.append((h, k, ids)); used += size
    # short names for well-known people: "lincoln", "mozart", "martin luther king" (never replacing a real title)
    have = {k for _, k, _ in rows}
    people = sorted((r for r in rows if re.search(r"\(.*\b1?\d{3}\b.*\)", cands[r[1]][2][:90]) and cands[r[1]][3] > 5000), key=lambda r: -cands[r[1]][3])
    alias = {}
    for h, k, ids in people:
        w = k.split()
        for a in ([w[-1]] if len(w) > 1 and len(w[-1]) >= 5 else []) + ([" ".join(w[:-1])] if len(w) > 2 and w[-1] in ("jr", "sr", "ii", "iii") else []):
            if a not in have and a not in STOP and a not in common and a not in alias and a not in cands:
                alias[a] = (k, ids)
    for a, (k, ids) in alias.items():
        if fnv(a) not in seen and used + 8 <= BUDGET + 60000:
            seen.add(fnv(a)); rows.append((fnv(a), a, None)); used += 8
            cands[a] = cands[k]; alias[a] = k
    rows.sort()
    blob, offs, at = bytearray(), [], {}
    for h, k, ids in rows:
        if ids is None: continue                                  # aliases share their person's sentence
        at[k] = len(blob); blob.append(len(ids))
        acc = nb = 0
        for i in ids:
            acc |= i << nb; nb += 14
            while nb >= 8: blob.append(acc & 255); acc >>= 8; nb -= 8
        if nb: blob.append(acc & 255)
    offs = [at[k] if ids is not None else at[alias[k]] for h, k, ids in rows]
    T = {"hash": np.array([r[0] for r in rows], dtype=np.uint32), "off": np.array(offs, dtype=np.uint32),
         "text": np.frombuffer(bytes(blob), dtype=np.uint8), "keys": [r[1] for r in rows], "sent": {r[1]: cands[r[1]][2] for r in rows}}
    return T


if __name__ == "__main__":
    import pickle
    T = build()
    pickle.dump(T, open('/home/user/kokosd/engine/facts_table.pkl', 'wb'))
    n = len(T["hash"]); size = n * 8 + len(T["text"])
    print(n, "facts,", size, "bytes,", round(size / n, 1), "bytes each")
    for k in ("george washington", "moon", "dog", "france", "albert einstein", "volcano", "siberian husky", "game boy advance", "pizza", "dinosaur", "london", "photosynthesis"):
        print(f"  {k:18s} {T['sent'].get(k)}")
