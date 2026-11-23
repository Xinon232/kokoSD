"""Repeated-phrase rule: near-greedy generation with and without the rule (native engine), and reference == engine with it."""
import sys, pickle, subprocess
from ref_int import IntRef, quantize
sys.path.insert(0, '/home/user/kokosd/study')
from common import tokenizer
tok = tokenizer()
I = quantize(pickle.load(open("/home/user/kokosd/engine/params_i_kd.pkl", "rb")))
O = pickle.load(open("/home/user/kokosd/engine/outstep_i_kd.pkl", "rb"))
INV = 5120                                    # temperature 0.05: almost always the top token, which is what loops
def gen(prompt, rule, n=60, seed=7):
    out = subprocess.run(["/home/user/kokosd/build/koko_gen", "/home/user/kokosd/build/koko.bin", str(seed), str(INV), str(n), "1", "0", "1" if rule else "0"],
                         input=" ".join(map(str, prompt)).encode(), capture_output=True).stdout.split()
    return [int(x) for x in out]
def longest_repeat(ids):
    return max([0] + [k for k in range(1, 12) for i in range(len(ids) - k) if any(ids[i:i + k] == ids[j:j + k] for j in range(i + 1, len(ids) - k + 1))])
for q in sys.argv[1:] or ["can you tell me a joke", "sing me a song", "count to twenty", "say woof ten times"]:
    p = [1] + tok.encode(q).ids + [2]
    a, b = gen(p, 0), gen(p, 1)
    print(f"> {q}\n   without: longest repeated run {longest_repeat(a)} tokens: {tok.decode(a)[:150]!r}\n   with:    longest repeated run {longest_repeat(b)} tokens: {tok.decode(b)[:150]!r}")
# reference vs engine, rule on
p = [1] + tok.encode("can you tell me a joke").ids + [2]
R = IntRef(I); R.reset()
for t in p: R.step(t)
rs_, ref, start = 7, [], len(p)
for _ in range(40):
    t, rs_ = R.pick(O, rs_, INV, start)
    if t in (0, 1): break
    ref.append(t); R.step(t)
eng = gen(p, 1, 40)
print("reference == engine with the rule on:", ref == eng, f"({len(ref)} tokens)")
