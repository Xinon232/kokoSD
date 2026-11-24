# T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge
https://arxiv.org/abs/2407.00088 (v2; EuroSys 2025). Wei, Cao, Cao, Ma, Wang, Zhang, Yang (Microsoft Research / USTC / UCAS). Code: github.com/microsoft/T-MAC.
Tags: [P] paper HTML v2, read through a fetch-and-extract tool (figure bars not readable); [R] repository README; [B] follow-up paper Bitnet.cpp (arXiv 2502.11880), same access; [I] my inference.

## What it is
An inference kernel library for weight-quantized LLMs on CPUs that replaces multiply-accumulate with table lookups and never dequantizes weights. An n-bit weight matrix is split into n one-bit matrices; for each group of g=4 activations a 16-entry table of signed partial sums is built at run time, and packed weight nibbles index it. It is the published ancestor of our kernel: same idea, but bit-plane tables (2 lookups per 4 ternary weights) where ours enumerates ternary states (1 lookup). It is inference-only; no training method.

## Mechanism
[P] Algorithm 1, activations A[N,K], weights W[M,K], b bits:
```
offline: for each bit i, pack bit i of every g consecutive weights into one index W_i[m, k/g]
online:  for each group of g activations: LUT[group][idx] = sum_j (bit j of idx ? +A_j : -A_j)   # 2^g entries
         R_i[n,m] = sum over groups LUT[group][W_i[m,group]]
         R = sum_i alpha_i * R_i + A . B^T
```
- [P] Bits are mapped 0 -> -1, 1 -> +1 (chosen to minimise the table's value range); W = sum_i alpha_i 2^i W'_i + B, with B a constant matrix, so the correction is one scalar times sum(A). [I] For this mapping alpha = 1/2 and B = (2^b - 1)/2.
- [P] "Ternary weights in 1.58bit BitNet are interpreted as 2-bit and decomposed into two 1-bit matrices": two lookups per 4 weights, 2 bits/weight stored. The bit mapping is not given.
- [P] Mirror consolidation: an index and its complement give negated sums, so only 2^(g-1) entries are stored (lossless).
- [P] Table quantization: table entries go from fp16 to int8 with a dynamic per-table scale, so a g=4 table fits one 128-bit register (NEON `vqtbl1q_u8`, AVX2 `_mm256_shuffle_epi8`). g=5 needs two registers and slower instructions.
- [P] Layout: loop over K (groups) outermost so one small table stays in registers while many output rows consume it; example index tile [K=4, M=32]. Weights are permuted and interleaved offline so the packed stream is read strictly sequentially.
- [P] Fast aggregation (optional): sums kept in 8 bits with averaging instructions, bias-corrected; lossy.

## Numbers
- [P] Kernel, single-thread GEMV, Llama-2-7B/13B shapes, max speedup over llama.cpp: 11.2x (1-bit), 5.8x (2-bit), 4.7x (3-bit), 3.1x (4-bit) (Sec. 5.2, Fig. 6). Cost scales linearly with bit width.
- [P] End to end, Raspberry Pi 5, single thread: 2.8x (Llama-2-7B 4-bit), 6.7x (2-bit), 5.8x (BitNet-3B); peak 11 tokens/s BitNet-b1.58-3B on Pi 5, 71 tokens/s on M2-Ultra (8 cores), 30 on one core (Sec. 5.3, abstract).
- [P] Ablation, M2-Ultra (Fig. 10): plain lookup without the layout work is up to 17% slower than llama.cpp; table quantization brings parity; then tiling up to 1.45x, permutation up to 1.39x, interleaving 1.42x, fast aggregation up to 1.29x.
- [P] Error, Table 3, 4096x4096 GEMV NMSE: llama.cpp 3.33e-3, T-MAC 3.35e-3, with fast aggregation 8.09e-3. Table 4, Llama-2-7B 4-bit: WikiText2 PPL 5.96 / 5.96 / 6.38; WinoGrande 70.8 / 70.8 / 67.8.
- [P] Table 5, Jetson AGX Orin, 7B 2-bit: 15.62 tokens/s at 10.4 W vs llama.cpp CPU 7.08 at 15.0 W.
- [B] Table 7, Intel i7-13700H, tokens/s, T-MAC vs element-wise TL2 (16-entry table + sign bit per 3 ternary weights, 1.67 bits/weight): 700M 76.29 vs 126.99; 3.8B 18.12 vs 35.43; 100B 0.73 vs 1.69 (2.32x). On M2 Ultra they are near equal at 700M (220.22 vs 229.21). [B] T-MAC's int8 table requantization is not lossless for BitNet; int16 sums are (Table 2: PPL 11.30 vs 11.29).

## Smallest scale tested, and whether the result held there
Kernels at 4096-wide Llama shapes; models from 3B (BitNet) to 13B [P]; the follow-up goes down to 700M, where T-MAC still beats 4-bit llama.cpp (76 vs 68 tokens/s, Intel) [B]. Nothing below that, nothing without SIMD, nothing on a microcontroller.

## Training recipe and gotchas
None; models come from GPTQ, BitDistiller, OneBit and BitNet [P]. Gotchas for us:
- Naive lookup was slower than multiply code until memory layout was fixed: order of ROM bytes matters as much as the table trick.
- 8-bit sums are lossy; keep 16-bit tables and accumulators. [I] Worst case with 8-bit signed inputs: 4 x 127 x (K/4) = 32,512 at K=256 (fits int16), 48,768 at K=384 (overflows): use 32-bit accumulators or 7-bit inputs.
- Train with the exact integer activation format the kernel uses [B].

## Mapping onto the GBA design
[I] throughout; cycle figures use the project's 1.6 cycles/weight (6.4 per lookup+add) and an assumed, unmeasured 4 cycles per table entry built.
- Bit-plane vs our kernel, 256x256 ternary matrix: T-MAC needs 2 x 64 x 256 = 32,768 lookups (about 210k cycles) and 1,024 table entries; ours needs 16,384 lookups (105k) and 5,184 entries (21k). Ours is about 1.7x cheaper. T-MAC's form wins only for matrices with fewer than roughly 40 output rows, where table build dominates.
- Adopt the layout: iterate groups outermost with weights stored column-group-major in ROM. Then only one 81-entry table (162 bytes) is live, plus M accumulators (1 KB at M=256, 32-bit), instead of 64 tables = 10.4 KB of the 32 KB fast RAM. ROM reads stay sequential.
- Mirror consolidation: 81 -> 41 entries, build cost halves (about 10k cycles saved per 256x256), but each lookup needs a conditional negate; likely a net loss when rows >= 64. Measure.
- No register limit on a scalar CPU, so g=5 is open: 3^5 = 243 states fit one byte (1.6 bits/weight, 20% fewer lookups, 20% less ROM); build is 243 entries per 5 inputs, which pays off only for about 500+ rows per table.
- Build each table with one add per entry (entry = smaller entry +/- one input), not g adds as in Algorithm 1.
- Skip table quantization and fast aggregation: SIMD-register tricks with no scalar benefit.
- No multiplies in the kernel; one scale per row can fold into the requantisation step.

## Verdict
**Adopt the layout principles; skip the bit-plane arithmetic.** For ternary weights element-wise tables beat bit-planes (confirmed at 700M-100B by the follow-up), and our kernel is already element-wise. Settling experiment: in the emulator, time a 256x256 and a 384x1536 ternary matrix in (a) row-major with all tables in RAM, (b) group-major with one live table, (c) g=5 byte codes, (d) mirrored 41-entry tables.

## Could not verify
- Figures 6-11 are charts; only numbers quoted in the text were readable. The abstract's "60-70% energy reduction" does not match all Fig. 9 text values (20.6%, 61.2%, 51.3%).
- Not stated in the paper: table-build share of run time, tuned tile sizes, exact int8 scale formula, the ternary-to-bit-plane mapping.
- Kernel source files were not read; only the README.
- All GBA cycle estimates are mine, not measured.

## Links worth following
- https://arxiv.org/abs/2502.11880 (Bitnet.cpp: element-wise ternary tables TL1/TL2, lossless variants)
- https://github.com/microsoft/T-MAC and https://github.com/microsoft/BitNet (kernel code generators)
