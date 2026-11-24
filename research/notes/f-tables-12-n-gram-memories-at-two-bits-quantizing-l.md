# N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models

https://arxiv.org/abs/2610.21810 (v1 15 Oct 2026, v3 21 Oct 2026). Han, Jung, Kwak, Lee, Lee. The paper is real; read as https://arxiv.org/html/2610.21810v3 through a fetch tool that summarises, so numbers below are as extracted, not eyeballed. Tags: [P] this paper, [Q] QTIP paper (arXiv 2406.11235), [I] my inference.

## What it is
Post-training compression of the hashed n-gram embedding tables ("conditional memory", Engram-style) in three production LLMs. Each row is stored independently as a 2-bit-per-coordinate trellis code plus one 16-bit scale, so a single row can be decoded without touching neighbours. The decoder ("Conditional Program", CP) is one small lookup table shared by the whole table: no per-table codebook fitting, no calibration data, no retraining. It is essentially QTIP's HYB trellis applied row-wise, without rotation, with a norm-restoring row scale [P].

## Mechanism
Encode (offline) [P, Sec. 3.3-3.4, Eq. 6-7]:
1. `z = sqrt(d) * e / ||e||` (row normalised to unit variance per coordinate).
2. `c = argmin_c ||z - f(c)||^2` by Viterbi over a tail-biting bitshift trellis (two Viterbi passes).
3. `alpha = ||e|| / ||f(c)||` (norm restoration, not least squares); store as bf16.
4. Decode: `e_hat = alpha * f(c)`. Rate = b + 16/d bits per parameter.
A fixed shrink factor multiplies the decoded code: 0.95 at 2-3 bits, 0.90 at 3.5, 0.85 at 4.

Decoder f, from QTIP [Q, Alg. 3]: state L=16 bits, V=2 values per step, k=2 bits per value, so each step shifts in 4 new bits. For step t take the 16-bit window `x` of the row bitstream starting at bit 4t (wrapping at the row end: tail-biting), then
```
x = (x*x + x) mod 2^32
idx = (x >> (15-Q)) & (2^Q - 1)      # Q = 9
(v0, v1) = LUT[idx]                  # 512 x 2 entries, 2 KiB
flip sign according to bit 15 of x   # QTIP text and algorithm disagree on whether one or both values flip
```
LUT = 512-point k-means on a 2-D standard Gaussian. Steps are independent, so decode is parallel and random-access.

## Numbers
All [P], all on tables with d=256 or 160 and 31-197B table parameters.
- Tables (Table 6): DeepSeek-V4.1-Flash 768M rows x 256, 202.7 GB FP8; Qwen3.8-Flash-Next 320M x 160, 102.4 GB; LongCat-Flash-Lite 122.7M x 256, 62.8 GB. Rows read per token: 48 (a footnote says 24) / 16 / 12.
- Output KL, geometric mean over six domains (Table 13), DeepSeek / Qwen / LongCat:
  - table removed 0.773 / 0.308 / 2.89
  - NVFP4 4.5 bpp: .0199 / .0166 / .0111
  - RTN 3-bit (3.4-3.5 bpp): .0293 / .0256 / .0277
  - RTN 2-bit (2.4-2.5 bpp): .0688 / .0588 / .117
  - CP 2-bit (2.06-2.10 bpp): .0401 / .0327 / .0426
  - CP 1.5-bit (1.56-1.60 bpp): .0593 / .0477 / .0692
  - CP2/RTN2 ratio 0.585 / 0.556 / 0.366 (the "41-63% lower"); CP2/QTIP-with-rotation 0.97 / 0.94 / 0.91.
- Task scores (Table 1, overall of 7 benchmarks): DeepSeek 79.8 original, 73.5 without table, 79.9 at 2 bits, 79.5 at 1.5 bits. LongCat 68.2 original, 30.5 without table, 67.3-67.4 at 2 bits, 66.7 for RTN 2-bit; HumanEval drops 68.9 -> 63.4.
- WikiText-2 perplexity (Table 14): LongCat 13.316 -> 13.538 at 2 bits (103.5 with table removed); Qwen 4.671 -> 4.830.
- Matched 2.10 bpp on Qwen, FineWeb-Edu KL (Table 3): shared trellis .0198; fitted VQ 256x4 .0238; fitted VQ 65,536x8 .0226; mean+trellis .0193. Relative row MSE: trellis ~.072, VQ .091-.099.
- Norm restoration vs row least squares (Table 4): .0327 vs .0349 Qwen, .0426 vs .0462 LongCat. Rotation gives no resolved gain.
- Gaussian 2-bit MSE [Q, Table 1]: scalar Lloyd-Max 0.118, HYB 0.071, rate-distortion bound 0.063.
- Token-indexed Gemma per-layer embeddings (App. H, Table 26): 2 bpp worst-domain KL .106, falling to .027 when the BOS row and top-100 rows are kept in bf16.

## Smallest scale tested, and whether the result held there
Smallest is LongCat-Flash-Lite's 31.4B-parameter table (and Gemma E2B per-layer tables in App. H). Nothing near our scale; no small models, no rows narrower than 160. Not tested with ternary backbones or integer activations.

## Training recipe and gotchas
- Pure post-training: no QAT, no fine-tuning, no calibration. Encoding took ~3.1 h on 4 H200 for 197B parameters; our tables are about 1000x smaller.
- Works because trained hashed rows look near-Gaussian per coordinate; but 16 fitted directions hold 10-33% of energy on hot rows versus 6-10% Gaussian (Table 2), so there is structure the code ignores.
- Frequency-based bit allocation was tried and rejected (overfits the web read distribution).
- Token-indexed tables have a few very sensitive rows; keep them at high precision.
- 16/d scale overhead: 2.06 bpp at d=256, but 2.25 at d=64, 2.5 at d=32 [I]. An 8-bit log scale would halve that [I].

## Mapping onto the GBA design
- ROM: a d=256 row is 64 + 2 = 66 bytes versus 128+ at 4 bits: 32 MB holds ~508K rows instead of ~260K [I].
- Decode cost per row [I, estimate]: 128 steps, each one window extract, one 32-bit multiply, shift, mask, a 2-entry LUT read, sign fix: roughly 25-35 cycles per step, so 3-5K cycles per d=256 row. Eight rows per token is about 30-40K cycles, about 1% of budget. Applying alpha costs one multiply per coordinate (~1-2K cycles per row) or can be folded into the gate scalar.
- RAM: LUT is 512 x 2 int16 = 2 KB, could sit in fast RAM or ROM. No other state.
- Caveat for us: the decoded row is dense 8/16-bit, not ternary, which is fine because rows are added to the residual, not multiplied through the ternary kernel.
- Alternative we already have: ternary rows at 1.6 bits trained with QAT. This paper offers no comparison with QAT-trained low-bit tables.

## Verdict
Test. It is cheap evidence that 2-bit (even 1.5-bit) rows keep nearly all the value of a hashed memory, and the decoder is integer-friendly and nearly free for us. But all evidence is post-training on huge models; we train from scratch and can use QAT. Settling experiment: train the small model with float n-gram tables at fixed ROM bytes, then compare (a) 4-bit rows, half the row count, (b) 2-bit CP rows post-hoc, (c) ternary QAT rows, (d) 2-bit CP then brief fine-tune of the backbone, on held-out loss.

## Could not verify
- The fetch tool truncated/summarised: 1.5-bit trellis construction, exact decoder details in this paper (taken from QTIP instead), and code release were not found.
- Table 1 as extracted lists CP only at 1.5 bits and two unlabeled "QTIP 2" rows; I assume one is CP 2-bit. Extracted LongCat perplexity at 1.5 bits (13.484) is lower than at 2 bits (13.538), which looks like an extraction error or noise.
- Model names (DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next) taken as given; not cross-checked.
- All GBA cycle counts are my estimates.

## Links worth following
- QTIP: https://arxiv.org/abs/2406.11235 (trellis, HYB/1MAD/3INST codes; 1MAD/3INST need no LUT)
- Engram: https://arxiv.org/abs/2601.07372 (the memory being compressed)
