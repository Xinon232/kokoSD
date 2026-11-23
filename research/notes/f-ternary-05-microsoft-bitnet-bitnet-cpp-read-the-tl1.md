# microsoft/BitNet (bitnet.cpp): TL1/TL2 lookup-table kernels  (https://github.com/microsoft/BitNet, 2024-2026, Microsoft Research; papers arXiv 2410.16144 and 2502.11880, Wang et al.)

Tags: [R] repo at commit 0b341e5 (27 Jul 2026), read: `utils/codegen_tl1.py`, `codegen_tl2.py`, `convert-hf-to-gguf-bitnet.py`, `src/ggml-bitnet-lut.cpp`, `docs/codegen.md`, `src/README.md`, README. [P] the two papers' HTML via a fetch-and-extract tool. [D] my derivation.

## What it is
The official CPU inference runtime for BitNet b1.58 models, a fork of llama.cpp with three ternary kernels. I2_S stores 2 bits per weight and uses SIMD multiply-add. TL1 and TL2 are the lookup kernels: per input vector they build small tables of partial sums and each packed weight index selects one. All are bit-exact against the float model with 8-bit activations. It is our kernel's closest published relative, built for SIMD rather than a scalar CPU.

## Mechanism
- Activations [R]: `s = 127 / max|x|` over the whole vector, `x_q = round(x*s)` to int8. Output `y = acc / s * w_scale`, one float scale per matrix.
- TL1 [R]: weights taken in pairs along the input axis; `idx = 3*w0 + w1 + 4` (0..8) in 4 bits, two indices per byte, 2 bits/weight. Per pair a 16-slot table of int16 sums (9 used), stored as two 16-byte planes (high bytes, low bytes) = 32 bytes per pair. Lookup = two `vqtbl1q_s8` (one per plane) zipped to int16, so nothing is rounded.
- TL2 [R]: weights in triples; `v = 9*w0 + 3*w1 + w2` in -13..13; store `|v|` (0..13) in 4 bits and the sign as 1 bit in a separate stream: 5 bits per 3 weights = 1.67 bits/weight. Table holds only the 14 non-negative sums; sign applied as `(x + s) ^ s` with s = 0 or -1. Columns that do not fill a block of triples fall back to TL1.
- Why g stops at 3 [P]: a table must fit one 16-entry byte shuffle; 3^2 = 9 and 3^3/2 = 13.5 fit, 3^4 does not.
- Tiling [R]: M outputs cut into BM blocks, K inputs into BK blocks; inside a (BM, BK) tile `bm` = 32 or 64 outputs are accumulated in int16 over the whole BK span, then widened into int32. Shipped BK: 64-128 (TL1), 96-192 (TL2). Worst case 128 pairs x 254 = 32,512, just under int16 [D].
- ROM order [R]: the converter permutes weights so that, per tile and per table, the indices of 16 consecutive output rows are contiguous: the kernel reads strictly sequentially.

## Numbers
- [P, 2410.16144 Tables 4-5] dummy models, tokens/s, bitnet.cpp vs llama.cpp fp16. Unlimited threads: 125M 593 vs 434 (1.37x, M2 Ultra), 389 vs 164 (2.37x, i7-13700H); 700M 1.70x / 3.88x; 13B 3.84x / 6.17x; 70B 5.07x (M2).
- [P, 2410.16144 Table 7] exact match with fp32 output on 1,000 prompts, 700M model: I2_S, TL1, TL2 100%; llama.cpp TQ1_0/TQ2_0 1.4%.
- [P, 2502.11880 Table 2] WikiText2 PPL: fp16 11.29; lossless TL1_1/TL2_1/I2_S 11.29; int8-rounded-table variants TL1_0/TL2_0 11.30; Q4_0 11.57.
- [P, 2502.11880 Table 7] i7, 700M: TL1 75.6, TL2 127.0, I2_S 125.4, T-MAC 76.3 tokens/s. Lookup is not faster than multiply-add on SIMD CPUs; appendix C.2: MAD 3.77 ns, TBL+ADD+CVT 6.20 ns (i5-13400F).
- [R, src/README.md] embedding precision on the 2B4T model, WikiText PPL: F32 17.109; Q8_0 17.120; Q6_K 17.149; Q4_0 17.353; ternary I2_S embeddings: model fails.

## Smallest scale tested, and whether the result held there
125M (dummy weights, speed only): the gain shrinks to 1.37x on ARM, because float embedding/head and non-matmul work take a larger share. Exactness was checked at 700M. Nothing below 125M, nothing scalar, no microcontroller.

## Training recipe and gotchas
Inference only. For the trainer: exactly ternary weights with one scale per matrix; per-vector absmax int8 activations; kernels are generated per matrix shape with divisibility rules (M % BM, K % BK). Post-hoc ternary embeddings break the model; 8-bit ones are free.

## Mapping onto the GBA design
- Our kernel is TL1 with g = 4 and a byte index; the 16-entry limit is a SIMD limit that we do not have [D]. Table entries int16 (max 4 x 127 = 508), row accumulator 32-bit in a register.
- g = 5 [D]: 3^5 = 243 fits one byte: 1.6 bits/weight (20% less ROM), one lookup per 5 weights. Build is 243 entries per group against 81. With about 4 cycles per entry built and 6.4 per lookup, 20 inputs cost 5 x (324 + 6.4M) at g = 4 and 4 x (972 + 6.4M) at g = 5: break-even near M = 350 output rows; a win for the 640-row expert up-projection, a loss for 256-row matrices. Cycle figures are assumptions.
- Mirror trick [D]: g = 4 with 41 entries + sign bit halves table RAM and build time (640 inputs: 26 KB to 13 KB) for one conditional negate per lookup (about +2 cycles on 6.4). Worth it only where M is small.
- Tiling fixes the RAM problem [D]: process BK = 128 inputs at a time; 32 live tables x 162 bytes = 5.2 KB in fast RAM; partial 32-bit accumulators (M x 4 bytes) parked between tiles. ROM laid out tile-by-tile, row-major inside a tile, so reads stay sequential. Cost: one load+store per row per tile, about 1 in 32 lookups.
- Build tables incrementally: `T[i + 3^k*t] = T[i] +/- x_k`, one add per entry.

## Verdict
Adopt (mechanism, not the speed numbers). It confirms the kernel family is exact and gives three concrete upgrades: K-tiling, the 3^5 byte format, and tile-ordered ROM packing. Settling experiment: emulator cycle counts for a 320x640 and a 640x320 matrix with g = 4 vs g = 5 vs mirrored g = 4, tiled vs untiled.

## Could not verify
- Paper tables came through a fetch tool, not re-checked by eye; README speed graphs (Jan 2026 update, 1.15x-2.1x claimed) are images.
- No scalar measurements exist in the source; all GBA cycle estimates are mine.
- TL1 is ARM-only and TL2 x86-only in this repo.

## Links worth following
- T-MAC (bit-wise LUT, mirror consolidation, table quantization): https://github.com/microsoft/T-MAC
- Bitnet.cpp ELUT paper: https://arxiv.org/abs/2502.11880
- 1-bit AI Infra Part 1.1: https://arxiv.org/abs/2410.16144
