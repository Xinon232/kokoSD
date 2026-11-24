# Compression Scaling Laws: Unifying Sparsity and Quantization  (https://arxiv.org/abs/2502.16440, v1, Feb 2025, Frantar, Evci, Park, Houlsby, Alistarh; Google DeepMind / ISTA)

Read directly: full arXiv HTML v1 (https://arxiv.org/html/2502.16440v1), all five tables, and the two figure images the HTML actually renders (Fig. 3 fit, Fig. 5 fit). Figures 1, 2, 4, 6, 7 did not render in HTML and were not seen. It is a short technical report with no appendix and no code.

## What it is
A from-scratch pretraining study that asks how much one compressed weight is "worth" relative to one BF16 weight. It claims each compression format C (k-bit weights, k-bit weights+activations, x% sparsity) acts as a constant multiplier eff(C) on parameter count inside a Chinchilla-style loss law, independent of model size and data. It fits that multiplier (EPM) for 1/2/4-bit weight-only QAT, 1/2/4/8-bit full QAT, and 50/75% sparsity. No ternary point is measured.

## Mechanism
- Law (Eq. 1/2): `L(N,D,C) = a/(N*eff(C))^b + c/D^d + e`. Fitted a,b,c,d,e are NOT reported.
- Fitting eff(C): sweep model size at fixed tokens-per-parameter (Chinchilla ratio), for dense and for each C, same hyperparameters; eff(C) is the horizontal shift that puts the compressed curve on the dense one.
- Weight QAT: "roughly follow BitNet": vanilla STE, forward pass with quantized weights, scale computed dynamically from weight statistics, not learned (no LSQ/PACT). Exact scale statistic and granularity are not stated.
- 2-bit grid: symmetric, zero-point 1.5, i.e. levels {-1.5,-0.5,+0.5,+1.5} x scale (no zero), beat centered zero-point 2, i.e. {-2,-1,0,1}.
- Activation QAT: only the inputs of linear non-embedding layers are quantized; attention itself is not.
- Sparsity: STE with top-k magnitude mask recomputed before every forward pass; no gradual schedule. Per-row top-k used for the sweeps.
- "Size gain" = (16/bits) x EPM, or (1/density) x EPM (I checked this reproduces every table entry).

## Numbers
- Table 2, weight-only: EPM 0.923 (4-bit), 0.702 (2-bit), 0.466 (1-bit); size gain 3.69x / 5.62x / 7.46x.
- Table 3, weights+activations: EPM 0.857 (8-bit), 0.747 (4-bit), 0.289 (2-bit), 0.067 (1-bit).
- Table 5, sparsity: EPM 0.871 (50%), 0.622 (75%); size gain 1.74x / 2.49x. About 10% better than Frantar et al. 2024.
- Table 1 (Llama 100M, 10B tokens, 2-bit): val loss 3.250 symmetric vs 3.268 centered.
- Table 4 (Llama 100M, 10B tokens, 50% sparse): 2:4 = 3.274; 64:128 = 3.223; per-output 3.223; per-tensor 3.224.
- Fig. 3, my reading of plotted points (approximate): four sizes, about 0.2B, 0.44B, 0.8B, 1.6B non-embedding parameters. Val loss BF16 2.90/2.70/2.56/2.42; 4-bit 2.91/2.72/2.57/2.43; 2-bit 2.98/2.79/2.63/2.49; 1-bit 3.11/2.89/2.73/2.58. So at 0.2B the 1-bit penalty is about 0.21 nats and 2-bit about 0.08. Curves are straight and parallel on log-log axes.
- Sec. 2.5: Pareto lines of 2W4A and 2W2A overlap, as do 4W2A and 4W4A; authors guess halving activation bits is like halving model size (stated as unverified).
- Fig. 2 (text only): multipliers fitted at 1x Chinchilla predict the 2x-Chinchilla runs "virtually identically", two settings only.
- Fig. 1 (text only): LR sweep ordering identical for BF16 and INT3; they could not reproduce "quantized models tolerate higher LR".

## Smallest scale tested, and whether the result held there
Sweeps: about 200M non-embedding parameters (figure reading); tuning runs: "Llama 100M" on 10B tokens. Parallel lines hold across the 8x range shown. Nothing below 100M is tested, so our few-million-weights-touched regime is an extrapolation of roughly two orders of magnitude; EPM is stated to depend on architecture.

## Training recipe and gotchas
- Llama-type decoder, "a variant of C4", Chinchilla tokens-per-parameter. Optimizer, LR, batch, widths/depths, tokenizer not given.
- Use the dense-tuned hyperparameters unchanged for QAT; do not raise LR.
- Prefer a symmetric no-zero grid at 2 bits.
- Small-block N:M sparsity is measurably worse than per-row top-k (2:4 costs 0.05 nats at 100M); large blocks are free.
- Keep activations at 4 bits or more; quality collapses below.
- Authors say compression benefits grow with longer training (asserted from prior work, no new data here).

## Mapping onto the GBA design
The paper prices bits; we pay per lookup. Derived by me, not in the paper:
- Ternary (1.585 bits) EPM by linear-in-bits interpolation between 0.466 and 0.702 is about 0.60. Then ternary/binary is about 1.3x and 4-level/ternary about 1.16x.
- Per lookup+add, for a 256x256 matrix including table builds: ternary 4/byte (81-entry tables): 5,184 + 16,384 ops for about 39K effective weights (1.8 per op). Binary 8/byte (256-entry tables, 16 KB RAM at 16 bit): 8,192 + 8,192 ops for about 30.5K (1.9 per op). 4-level 4/byte (256-entry): 16,384 + 16,384 ops for 46K (1.4 per op). So 2-bit loses; ternary and packed binary tie, and binary's larger tables hurt in fast RAM and when few rows are evaluated (hard-routed experts).
- The "one zero per 4 weights" 32-entry format is a small-block N:M constraint; Table 4 warns such constraints cost quality (measured for 2:4, not 3:4).
- 8-bit activations into linears look affordable (8W8A 0.857); no change to ROM/RAM needed.

## Verdict
Test, do not adopt as a planning constant. It supports "stay ternary, widen instead of adding bits", and warns about the 3-of-4 format, but has no ternary point and no small-model point. Settling experiment: at our real width, train float, binary, ternary, 3-of-4 ternary and 4-level at 3 widths each with identical hyperparameters; fit the horizontal shift; compare per lookup.

## Could not verify
- Ternary EPM (interpolated by me). Fitted law constants. Exact scale rule and granularity. Model shapes and optimizer settings. Figures 1, 2, 4, 6, 7. The 75%-plus third sparsity level mentioned in text has no table entry. Combined sparsity+quantization is explicitly not studied. Nothing here comes from secondary sources.

## Links worth following
- Scaling Laws for Precision (Kumar et al.): https://arxiv.org/abs/2411.04330 (competing law, claims a barrier near 8 bits)
- Scaling Laws for Sparsely-Connected Foundation Models (Frantar et al., ICLR 2024): cited without URL in the paper; source of the law's form
- Spectra (ternary vs float size sweeps): https://arxiv.org/abs/2407.12327
