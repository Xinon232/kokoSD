"""Read replies under a few sampling settings."""
import sys, torch
from common import load, tokenizer
torch.set_num_threads(4)
model, cfg, counts, res = load(sys.argv[1] if len(sys.argv) > 1 else "h_koko_L8")
tok = tokenizer()
PROMPTS = ["hey koko", "what is your favourite food", "tell me a story about a lost kitten", "what is the moon",
           "i had a bad day at school", "can you tell me a joke", "who was george washington", "what should i do when i am bored",
           "what is a husky", "why do cats purr"]
SETTINGS = [("greedy", 0, 1, 1.0), ("t0.5 k40", 0.5, 40, 1.0), ("t0.7 k40", 0.7, 40, 1.0), ("t0.7 k40 rep1.15", 0.7, 40, 1.15),
            ("t0.9 k40", 0.9, 40, 1.0)]


def reply(q, temp, k, rep, seed, n=70):
    g = torch.Generator().manual_seed(seed)
    ids = tok.encode(f"<|user|>{q}<|koko|>").ids
    start = len(ids)
    with torch.no_grad():
        for _ in range(n):
            lg = model(torch.tensor([ids[-cfg.T:]]))[0][0, -1]
            if rep != 1.0:
                seen = torch.tensor(sorted(set(ids[start:])), dtype=torch.long)
                if len(seen):
                    lg[seen] = torch.where(lg[seen] > 0, lg[seen] / rep, lg[seen] * rep)
            if temp == 0:
                nxt = lg.argmax().item()
            else:
                v, i = (lg / temp).topk(k)
                nxt = i[torch.multinomial(v.softmax(-1), 1, generator=g)].item()
            ids.append(nxt)
            s = tok.decode(ids[start:], skip_special_tokens=False)
            if "<|" in s:
                return s.split("<|")[0].strip()
    return tok.decode(ids[start:], skip_special_tokens=False).strip() + " [cut]"


for q in PROMPTS:
    print(f"\n### {q}")
    for name, t, k, rep in SETTINGS:
        print(f"  [{name}] {reply(q, t, k, rep, 1)}", flush=True)
