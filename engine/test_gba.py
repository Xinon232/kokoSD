"""Build the test ROM, run it in headless mGBA, and require the same numbers as the native twin."""
import subprocess, struct, sys, os, numpy as np
os.chdir('/home/user/kokosd')
sys.path.insert(0, 'study')
from common import val_seqs
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 12
FULL = "--full" in sys.argv
toks = val_seqs("koko", 2, max(NTOK, 256))[1][:NTOK].tolist()
open('gen/test_tokens.h', 'w').write(f"#define NTOK {NTOK}\nstatic const unsigned short TEST_TOKENS[NTOK] = {{{','.join(map(str, toks))}}};\n")
CF = '-mcpu=arm7tdmi -mtune=arm7tdmi -marm -O2 -ffreestanding -fno-builtin -Wall -Igen -Iengine -DKOKO_PROF'.split() + (['-DFULL_SCORES'] if FULL else [])
def sh(*a): subprocess.run(a, check=True)
sh('arm-none-eabi-gcc', *CF, '-nostdlib', '-Wl,-T,src/gba.ld', '-Wl,-Map,build/koko_test.map', '-o', 'build/koko_test.elf',
   'src/crt0.s', 'engine/gba_test.c', 'engine/koko.c', 'bench/tern.s', 'engine/kernels.s', 'engine/blob.s', '-lgcc')
sh('arm-none-eabi-objcopy', '-O', 'binary', 'build/koko_test.elf', 'build/koko_test.gba')
sh('python3', 'tools/fixrom.py', 'build/koko_test.gba')
sh('arm-none-eabi-size', '-A', 'build/koko_test.elf')
KP = ['embed+inject', 'norm', 'table build', 'query proj', 'key+value proj', 'key scan', 'top-8 mix', 'attn out', 'mixer', 'expert in', 'expert out', 'output rows']
n = 3 + 6 * NTOK + len(KP)
sh('build/runner', 'build/koko_test.gba', 'until:0x02000004:1:200000', f'dump:0x02000000:{4 * n}:build/koko_dbg.bin')
v = struct.unpack(f'{n}I', open('build/koko_dbg.bin', 'rb').read())
assert v[0] == 0x4B4F4B4F and v[2] == NTOK, v[:3]
cs_h, cs_s, cyc, prof = v[3:3 + NTOK], v[3 + NTOK:3 + 2 * NTOK], v[3 + 2 * NTOK:3 + 3 * NTOK], v[3 + 3 * NTOK:3 + 3 * NTOK + len(KP)]
b0 = 3 + 3 * NTOK + len(KP); cs_t, cyc_out, nrows = v[b0:b0 + NTOK], v[b0 + NTOK:b0 + 2 * NTOK], v[b0 + 2 * NTOK:b0 + 3 * NTOK]
raw = subprocess.run(['build/koko_twin', 'build/koko.bin'], input=' '.join(map(str, toks)).encode(), capture_output=True).stdout
a = np.frombuffer(raw, dtype=np.int32).reshape(NTOK, -1)
V = 16384
def cs(arr):
    c = 0
    for x in arr.tolist(): c = (c * 31 + (x & 0xFFFFFFFF)) & 0xFFFFFFFF
    return c
ok_h = all(cs(a[t, :256]) == cs_h[t] for t in range(NTOK))
ok_s = all(cs(a[t, 256:256 + V]) == cs_s[t] for t in range(NTOK)) if FULL else None
def cst(rec):
    n = int(rec[0]); c = n
    for k in range(n): c = ((c * 31 + int(rec[3 + k])) * 31 + (int(rec[43 + k]) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return (c * 31 + int(rec[2])) & 0xFFFFFFFF
ok_t = all(cst(a[t, 256 + V:]) == cs_t[t] for t in range(NTOK))
print("fast output step identical to twin (top-40 ids, scores, sampled token):", ok_t)
print(f"output step: {np.mean(cyc_out):,.0f} cycles for {np.mean(nrows):,.0f} rows + 256 router rows = {np.mean(cyc_out) / (np.mean(nrows) + 256):,.0f} per row")
print(f"END TO END: {np.mean(cyc) + np.mean(cyc_out):,.0f} cycles = {16777216 / (np.mean(cyc) + np.mean(cyc_out)):.2f} tokens/s (short context, unoptimised)")
print("state identical to twin:", ok_h, "| all output scores identical:", ok_s)
print("cycles per token (model body):", [f"{c:,}" for c in cyc[:4]], "... last", f"{cyc[-1]:,}")
tot = sum(prof)
for nm, p in zip(KP, prof): print(f"  {nm:16s} {p / NTOK:12,.0f}  {100 * p / tot:5.1f}%")
if FULL: prof = prof[:-1]; KP = KP[:-1]
print(f"body: {np.mean(cyc):,.0f} cycles = {16777216 / np.mean(cyc):.2f} tokens/s before the output step")
