"""Talk to the cartridge engine (native build): exactly the integers the GBA computes."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import subprocess, sys
sys.path.insert(0, ROOT + '/study')
from common import tokenizer
tok = tokenizer()
U, K, E = (tok.token_to_id(t) for t in ("<|user|>", "<|koko|>", "<|endoftext|>"))


def reply(history, seed=1, temp=0.5, maxn=80):
    ids = []
    for who, text in history:
        ids += [U if who == "user" else K] + tok.encode(text).ids
    ids += [K]
    out = subprocess.run([ROOT + "/build/koko_gen", ROOT + "/build/koko.bin", str(seed), str(round(256 / temp)),
                          str(maxn), str(U), str(E if E is not None else U)], input=" ".join(map(str, ids)).encode(), capture_output=True).stdout
    return tok.decode([int(x) for x in out.split()], skip_special_tokens=False).strip()


if __name__ == "__main__":
    print("special ids", U, K, E)
    for q in sys.argv[1:]:
        print("USER:", q); print("KOKO:", reply([("user", q)])); print()
