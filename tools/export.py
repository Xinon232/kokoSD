"""Quantize stories260K to int8 and emit gen/model_data.c + gen/model_data.h.

Every matrix row is int8 with its own scale. The scale, the fixed-point format
of the vector being multiplied and the Q16 output format are folded into one
int32 multiplier per row, so the cartridge does  out = (acc * mult) >> MSH.
Weights are stored as unsigned bytes, weight + 128: the assembly kernel then
pulls a byte out of a word in one instruction instead of two, and subtracts
128 * sum(input) once per row.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ref import Model, Tok

# fixed-point formats (fraction bits) of the int16 vectors fed to each matvec
Q_N1, Q_N2, Q_NF = 10, 11, 10   # outputs of the three kinds of RMSNorm
Q_ATT, Q_HB = 12, 10            # attention output (into wo), SwiGLU output (into w2)
Q_QK, Q_V = 8, 11               # query/key and value formats in the attention cache
Q_G = 12                        # norm gains
MSH = 20                        # shift after acc * mult
WINDOW = 256                    # attention ring size (positions kept)

m = Model('dl/stories260K.bin')
tok = Tok('dl/tok512.bin')
os.makedirs('gen', exist_ok=True)
C = []   # C source chunks
H = []   # header chunks


def arr(ctype, name, a, per=16):
    a = np.asarray(a).reshape(-1)
    H.append(f'extern const {ctype} {name}[{len(a)}];')
    lines = [','.join(str(int(v)) for v in a[i:i + per]) for i in range(0, len(a), per)]
    C.append(f'const {ctype} {name}[{len(a)}] __attribute__((aligned(4))) = {{\n' + ',\n'.join(lines) + '\n};')


def quant(W, q_in):
    """W: (..., rows, cols) float -> int8 rows + int32 multipliers."""
    s = np.abs(W).max(axis=-1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    w8 = np.rint(W / s).astype(np.int64)
    assert np.abs(w8).max() <= 127
    mult = np.rint(s[..., 0] * 2.0 ** (16 - q_in + MSH)).astype(np.int64)
    assert mult.max() < 2 ** 31
    return w8, mult, s[..., 0]


for name, W, q in [('wq', m.wq, Q_N1), ('wk', m.wk, Q_N1), ('wv', m.wv, Q_N1), ('wo', m.wo, Q_ATT),
                   ('w1', m.w1, Q_N2), ('w3', m.w3, Q_N2), ('w2', m.w2, Q_HB)]:
    w8, mult, _ = quant(W, q)
    arr('uint8_t', f'W_{name}', w8 + 128, 32)      # stored +128: see matvec
    arr('int32_t', f'M_{name}', mult, 8)

# classifier == embedding (tied). One int8 table, two multiplier sets.
w8, mult, s = quant(m.emb, Q_NF)
arr('uint8_t', 'W_emb', w8 + 128, 32)
arr('int32_t', 'M_cls', mult, 8)
arr('int32_t', 'M_emb', np.rint(s * 2.0 ** 24), 8)      # x_q16 = (w8 * M_emb + 128) >> 8

for name, g in [('G_att', m.g_att), ('G_ffn', m.g_ffn), ('G_fin', m.g_fin)]:
    arr('int32_t', name, np.rint(g * 2.0 ** Q_G), 8)

# lookup tables
arr('int16_t', 'T_sin', np.rint(np.sin(np.arange(4097) * 2 * np.pi / 4096) * 16384))
arr('uint16_t', 'T_exp', np.minimum(np.rint(np.exp(-np.arange(4096) / 256.0) * 65536), 65535))
arr('uint16_t', 'T_sig', np.rint(32768 / (1 + np.exp(-(np.arange(4097) - 2048) / 256.0))))
arr('uint32_t', 'ROPE_STEP', np.rint(m.freq / (2 * np.pi) * 2.0 ** 32), 4)

# ---- tokenizer ----
assert all(tok.scores[i] == 259 - i for i in range(259, m.V)), 'merge priority must equal id order'
blob = b''
offs = []
for i, v in enumerate(tok.vocab):
    p = b'' if i < 3 else (tok.piece[i] if i < 259 else v)
    offs.append(len(blob)); blob += p
offs.append(len(blob))
arr('uint8_t', 'TOK_bytes', np.frombuffer(blob, dtype=np.uint8), 32)
arr('uint16_t', 'TOK_off', offs)
# ids >= 259 sorted by their string, for the encoder's binary search
order = sorted(range(259, m.V), key=lambda i: tok.vocab[i])
arr('uint16_t', 'TOK_sorted', order)
# single ASCII characters -> their own piece (or the byte-fallback token)
arr('uint16_t', 'TOK_char', [tok.lookup[bytes([c])] for c in range(128)])

H.insert(0, f'''#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include <stdint.h>
#define DIM {m.dim}
#define HID {m.hidden}
#define NLAYER {m.L}
#define NHEAD {m.H}
#define NKVHEAD {m.KVH}
#define HEADSZ {m.hs}
#define KVDIM {m.kvd}
#define VOCAB {m.V}
#define WINDOW {WINDOW}
#define Q_N1 {Q_N1}
#define Q_N2 {Q_N2}
#define Q_NF {Q_NF}
#define Q_ATT {Q_ATT}
#define Q_HB {Q_HB}
#define Q_QK {Q_QK}
#define Q_V {Q_V}
#define Q_G {Q_G}
#define MSH {MSH}
#define NSORTED {len(order)}''')
H.append('#endif')
open('gen/model_data.h', 'w').write('\n'.join(H) + '\n')
open('gen/model_data.c', 'w').write('#include "model_data.h"\n' + '\n'.join(C) + '\n')
n8 = sum(a.size for a in (m.wq, m.wk, m.wv, m.wo, m.w1, m.w2, m.w3, m.emb))
print(f'exported: {n8} int8 weights, vocab blob {len(blob)} bytes')
