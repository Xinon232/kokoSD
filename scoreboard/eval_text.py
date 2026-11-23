"""Text prediction on identical held-out text: bits per character (lower is better).
Stories = TinyStories validation documents, facts = Simple English Wikipedia validation documents (both from
Koko's held-out split; no model here was trained on these exact documents by us, but the public models may
have seen the public sources). Every model gets exactly the same characters, each document on its own."""
import sys, json, math, numpy as np
sys.path.insert(0, '/home/user/kokosd/scoreboard')
import models as M
A = '/home/user/kokosd/artifacts/mix1x/mix1/'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
tok = M.koko_tok()


def docs(domain, n):
    ids = np.fromfile(A + f'{domain}_val_16384.bin', dtype=np.uint16)
    cuts = np.nonzero(ids == 0)[0]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        d = ids[a + 1:b][:255].tolist()                 # at most Koko's window
        text = tok.decode(d, skip_special_tokens=False)
        if len(d) >= 40 and tok.encode(text).ids == d:
            out.append(text)
        if len(out) == n: break
    return out


res = {}
sets = {"stories": docs("stories", N), "facts": docs("facts", N)}
for dom, items in sets.items():
    chars = sum(len(t) for t in items)
    print(dom, len(items), "documents,", chars, "characters", flush=True)
    for name in ["Koko (cartridge, 2-bit)", "Koko full-precision twin"] + list(M.BASELINES):
        tot = ntok = 0
        for t in items:
            if name.startswith("Koko"):
                a, b = M.koko_nats("i_kd" if "cartridge" in name else "h_koko_L8_fp", t)
            else:
                a, b = M.hf_nats(name, t)
            tot += a; ntok += b
        bpc = tot / math.log(2) / chars
        res.setdefault(name, {})[dom] = {"bits_per_char": round(bpc, 4), "tokens": ntok, "chars": chars}
        print(f"  {name:26s} {bpc:.3f} bits/char  ({ntok / len(items):.0f} tokens per document)", flush=True)
json.dump({"n": N, "results": res}, open('/home/user/kokosd/scoreboard/text.json', 'w'), indent=1)
