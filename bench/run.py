"""Build a benchmark ROM for each candidate shape, run it in headless mGBA, and
report cycles per token (checked against a native build of the same code)."""
import os
import struct
import subprocess
import sys

NTOK = 6
NAMES = ['embed', 'norm', 'table build', 'q/k/v proj', 'key scan', 'top-k mix', 'attn out',
         'router', 'expert in', 'expert out', 'output head', 'key-value memory']
CF = '-mcpu=arm7tdmi -mtune=arm7tdmi -marm -O2 -ffreestanding -fno-builtin -Wall'.split()
BASE = dict(D=256, L=6, DK=64, DV=64, NH=4, T=256, E=64, KACT=1, HD=256, NCLS=128, NMEM=128)
SHAPES = {
    'A  d256 x6, 64 experts':                  dict(NE=32),
    'B  + memory 1 head x128 half-keys':       dict(NE=32, NPK=1, NHK=128),
    'C  + memory 1 head x256 half-keys':       dict(NE=32, NPK=1, NHK=256),
    'D  no experts, memory 2x256':             dict(NE=32, KACT=0, E=0, NPK=2, NHK=256),
    'E  small MLP (128) + memory 1x256':       dict(NE=32, E=1, HD=128, NPK=1, NHK=256),
    'F  x8, small MLP (128) + memory 1x256':   dict(NE=32, E=1, HD=128, NPK=1, NHK=256, L=8),
    'G  d384 x6, small MLP (192) + mem 1x256': dict(NE=32, E=1, HD=192, NPK=1, NHK=256, D=384),
}


def sh(*a, **k):
    subprocess.run(a, check=True, **k)


def bench(i, label, over):
    p = dict(BASE, **over)
    d = f'build/bench/s{i}'
    os.makedirs(d, exist_ok=True)
    sh('python3', 'bench/gen.py', f's{i}', *[f'{k}={v}' for k, v in p.items()])
    inc = [f'-I{d}', f'-DNTOK={NTOK}']
    sh('arm-none-eabi-gcc', *CF, *inc, '-nostdlib', '-Wl,-T,src/gba.ld', '-o', f'{d}/b.elf',
       'src/crt0.s', 'bench/core.c', 'bench/tern.s', f'{d}/data.c', '-lgcc')
    sh('arm-none-eabi-objcopy', '-O', 'binary', f'{d}/b.elf', f'{d}/b.gba')
    sh('python3', 'tools/fixrom.py', f'{d}/b.gba', stdout=subprocess.DEVNULL)
    sh('gcc', '-O2', *inc, '-o', f'{d}/twin', 'bench/core.c', f'{d}/data.c')
    want = int(subprocess.run([f'{d}/twin'], capture_output=True, text=True, check=True).stdout)
    sh('build/runner', f'{d}/b.gba', 'until:0x02000004:1:20000', f'dump:0x02000000:{16 + 48}:{d}/dbg.bin')
    v = struct.unpack('16I', open(f'{d}/dbg.bin', 'rb').read())
    assert v[0] == 0xBE7C4000 and v[2] == NTOK
    cyc = [c / NTOK for c in v[4:]]
    tot = sum(cyc)
    active = p['L'] * ((2 * p['DK'] + p['DV']) * p['D'] + p['DV'] * p['D'] + p['E'] * p['D']
                       + p['KACT'] * 2 * p['HD'] * p['D']) + (p['NCLS'] + p['NMEM']) * p['D']
    stored = p['L'] * ((2 * p['DK'] + p['DV']) * p['D'] + p['DV'] * p['D'] + p['E'] * p['D']
                       + p['E'] * 2 * p['HD'] * p['D']) + (p['NCLS'] + p['NCLS'] * p['NMEM']) * p['D']
    npk, nhk = p.get('NPK', 0), p.get('NHK', 0)
    active += p['L'] * npk * (2 * nhk * p['D'] // 2 + 8 * p['D'])
    stored += p['L'] * npk * (2 * nhk * p['D'] // 2 + nhk * nhk * p['D'])
    return dict(label=label, ok=v[3] == want, cyc=cyc, tot=tot, active=active, stored=stored, p=p)


res = [bench(i, k, v) for i, (k, v) in enumerate(SHAPES.items())]
print(f'{"shape":42s} {"exact":5s} {"cycles/token":>12s} {"tokens/s":>8s} {"weights used/token":>18s} {"weights stored":>14s}')
for r in res:
    print(f'{r["label"]:42s} {"yes" if r["ok"] else "NO":5s} {r["tot"]:12,.0f} {16777216 / r["tot"]:8.2f} '
          f'{r["active"]:18,d} {r["stored"]:14,d}')
for r in (res[2], res[4]):
  print(f'\nwhere the time goes, shape {r["label"]}:')
  for n, c in zip(NAMES, r['cyc']):
    print(f'  {n:12s} {c:10,.0f}  {100 * c / r["tot"]:5.1f}%')
