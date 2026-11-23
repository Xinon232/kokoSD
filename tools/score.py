"""Score the integer twin against the float model on held-out TinyStories.

Both are teacher-forced on the same token sequences; the twin's int32 logits
come from build/twin (the same C that runs on the cartridge).
"""
import os
import subprocess
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ref import Model, Tok, load_stories

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MAXLEN = 256  # one attention window, so the float model sees the same context

tok = Tok('dl/tok512.bin'); m = Model('dl/stories260K.bin')
seqs = [tok.encode(s)[:MAXLEN] for s in load_stories('dl/eval/valid.txt', N)]

# encoder check: the C tokenizer must agree with the Python one
bad = 0
for s in load_stories('dl/eval/valid.txt', N):
    line = s.split('\n')[0][:200]
    if not line.isascii():
        continue
    c = subprocess.run(['build/twin', 'encode', line], capture_output=True, text=True).stdout.split()
    bad += [int(t) for t in c] != tok.encode(line)
print(f'tokenizer: {bad} mismatching lines')

inp = '\n'.join(' '.join(map(str, s)) for s in seqs) + '\n'
raw = subprocess.run(['build/twin', 'score'], input=inp.encode(), capture_output=True).stdout
il = np.frombuffer(raw, dtype=np.int32).reshape(-1, m.V) / 65536.0


def logp(lg):
    lg = lg - lg.max()
    return lg - np.log(np.exp(lg).sum())


nf = ni = 0.0; n = 0; agree = 0; chars = 0; maxerr = 0.0
for s in seqs:
    m.reset()
    for a, b in zip(s[:-1], s[1:]):
        fl = m.forward(a)
        q = il[n]
        nf -= logp(fl)[b]; ni -= logp(q)[b]
        agree += int(np.argmax(fl) == np.argmax(q))
        maxerr = max(maxerr, float(np.abs(fl - q).max()))
        n += 1
    chars += len(tok.decode(s))
assert n == len(il)
b = np.log(2)
print(f'{n} tokens from {len(seqs)} stories')
print(f'float : {nf / n:.4f} nats/token  {nf / b / chars:.4f} bits/char')
print(f'int8  : {ni / n:.4f} nats/token  {ni / b / chars:.4f} bits/char')
print(f'top-1 agreement {100 * agree / n:.2f}%   max |logit error| {maxerr:.3f}')
