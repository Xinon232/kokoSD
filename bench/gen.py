"""Random weights for one candidate model shape (timing only, nothing is trained).

Ternary matrices are stored the way the cartridge kernel reads them: 4 weights
per byte, and each byte is not an index but a pointer step (see tern.s).
"""
import sys
import numpy as np

name = sys.argv[1]
P = dict(a.split('=') for a in sys.argv[2:])
D, L, DK, DV, NH, T = (int(P[k]) for k in ('D', 'L', 'DK', 'DV', 'NH', 'T'))
E, KACT, HD, NCLS, NMEM = (int(P[k]) for k in ('E', 'KACT', 'HD', 'NCLS', 'NMEM'))
NPK, NHK = int(P.get('NPK', 0)), int(P.get('NHK', 128))   # product-key memory: heads, half-keys
NE = int(P.get('NE', 81))     # table entries per 4 inputs: 81 = free ternary, 32 = exactly one zero per 4
rng = np.random.default_rng(1)
C, H = [], []


def arr(ctype, nm, a, per=24):
    a = np.asarray(a).reshape(-1)
    H.append(f'extern const {ctype} {nm}[{len(a)}];')
    C.append(f'const {ctype} {nm}[{len(a)}] __attribute__((aligned(4))) = {{\n' +
             ',\n'.join(','.join(str(int(v)) for v in a[i:i + per]) for i in range(0, len(a), per)) + '\n};')


def tern(nm, rows, n_in, gain):
    if NE == 81:
        t = rng.integers(0, 3, size=(rows, n_in // 4, 4))        # trit codes: 0, +1, -1
        idx = t[..., 0] + 3 * t[..., 1] + 9 * t[..., 2] + 27 * t[..., 3]
    else:
        idx = rng.integers(0, 32, size=(rows, n_in // 4))        # zero position * 8 + three sign bits
    step = np.empty_like(idx)
    step[:, 0] = idx[:, 0] + 1
    step[:, 1:] = NE + idx[:, 1:] - idx[:, :-1]
    arr('uint8_t', 'S_' + nm, step)
    arr('int32_t', 'M_' + nm, np.full(rows, int(65536 * gain / np.sqrt(n_in * 2 / 3))), 8)


tern('qkv', DK + DK + DV, D, 1.0)
tern('wo', D, DV, 0.3)
tern('rt', max(E, 1), D, 1.0)
tern('w1', HD, D, 1.0)
tern('w2', D, HD, 0.3)
if NPK:
    tern('pk', NPK * 2 * NHK, D // 2, 1.0)        # half-keys, each reading half of the stream
    arr('int8_t', 'V_pk', rng.integers(-100, 100, size=(64, D)))
tern('cls', NCLS, D, 1.0)
tern('mem', NMEM, D, 1.0)
arr('int8_t', 'EMB', rng.integers(-100, 100, size=(64, D)))
arr('uint16_t', 'T_exp', np.minimum(np.rint(np.exp(-np.arange(1024) / 64.0) * 65536), 65535))
H.insert(0, f'''#ifndef DATA_H
#define DATA_H
#include <stdint.h>
#define D {D}
#define L {L}
#define DK {DK}
#define DV {DV}
#define NH {NH}
#define T {T}
#define E {E}
#define KACT {KACT}
#define HD {HD}
#define NCLS {NCLS}
#define NMEM {NMEM}
#define NE {NE}
#define NPK {NPK}
#define NHK {NHK}''')
H.append('#endif')
open(f'build/bench/{name}/data.h', 'w').write('\n'.join(H) + '\n')
open(f'build/bench/{name}/data.c', 'w').write('#include "data.h"\n' + '\n'.join(C) + '\n')
