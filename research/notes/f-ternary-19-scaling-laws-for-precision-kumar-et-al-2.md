# Scaling Laws for Precision  (https://arxiv.org/abs/2411.04330, v1 7 Nov 2024, v2 30 Nov 2024; Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan)

Read directly: arXiv abstract page and arXiv HTML (https://arxiv.org/html/2411.04330), main text and appendices A-O, through a text extraction tool (figures not seen; equations and Table 2 transcribed from that extraction).

## What it is
A pretraining study (465 runs, 30M-220M non-embedding parameters, 1.5B-26B tokens of Dolma, checks up to 1.7B) that adds bit precision to a Chinchilla-style loss law. Training with low-precision weights, activations and KV cache is modelled as shrinking an "effective parameter count" by a saturating-exponential factor per component. It also models post-training quantization (PTQ) damage, which grows with training data. Headline claims: compute-optimal training precision is about 7-8 bits, and going below 4 bits needs a disproportionately (more than 4x) larger model. **No run uses fewer than 3 bits; binary and ternary are cited, never trained.**

## Mechanism
- Law (Eq. 1): `L = A*N_eff^-a + B*D^-b + E + d_PTQ`.
- Eq. 4: `N_eff = N * f_w(P_w) * f_a(P_a) * f_kv(P_kv)`, `f_x(P) = 1 - exp(-P/g_x)`; the three factors are independent and multiplicative (Finding 2).
- Appendix K says the exponentials are "really computed with offset" `n_x`. Table 2: g_w 2.6745, n_w 0.3037; g_i (activations) 2.2102, n_i 1.4072; g_kv 0.9578, n_kv 2.4185; A 4.299e3, a = b = 0.4965 (tied), B 1.806e4, E 2.7648. PTQ: C_T 0.0598, g_D 0.5068, g_N 0.3439.
- PTQ (Eq. 2): `d_PTQ = C_T * D^g_D / N^g_N * exp(-P_post/g_post)`.
- Compute model (Eq. 5): `C = (6/16)*N*D*P`, cost linear in bits.
- Quantizer: integer, scale = Qp/max|x|, round-to-nearest, clamp; weights per-channel, activations per-tensor; forward pass only ("fake quantization"); the gradient rule (STE or other) is not stated.
- Architecture: Transformer++ (SwiGLU, RoPE, RMSNorm), seq 1024, batch 256, Adam (0.9, 0.95), cosine LR, 10% warmup, weight decay 0.1. Bits swept 3-12 plus BF16.

## Numbers
- Sec. 4.3: joint optimum over N, D, P is about 7-8 bits (integer fit), independent of compute budget; at fixed N, optimal P grows as log(D/N).
- Sec. 4.3: below 4 bits (all components) model size must grow more than 4x.
- Validation R^2 of the N_eff form: 0.82 (Table 1); PTQ fit 0.97 (Fig. 2); unified law 0.90 over 1000+ points (Fig. 7).
- Appendix L: runs with P_a <= 3 or P_kv <= 3 "often diverge" and are excluded from fits; P_w = 3 is "well behaved".
- Appendix M: KV cache is insensitive at high bits, becomes sensitive around 4-5 bits. Appendix I: activations are more sensitive than weights.
- Floating-point weights (Appendix N): g_w 2.8111, offset 0.1240.
- **My arithmetic, not in the paper** (assuming `f = 1 - exp(-P/g + n)`; the sign is my inference, the only one consistent with the 4-5 bit KV statement and the ">4x" claim): f_w = 0.93 (8 bit), 0.70 (4), 0.56 (3), 0.36 (2), 0.25 (1.585, extrapolated), 0.07 (1). f_a = 0.89 (8 bit), 0.33 (4). f_kv = 0.996 (8), 0.83 (4), 0.51 (3). All-4-bit product 0.19, i.e. 5x, matching the paper's ">4x".
- With a = 0.4965, quartering N_eff doubles the reducible term: about +0.83 nats at N = 30M by these constants (my arithmetic).

## Smallest scale tested, and whether the result held there
30M non-embedding parameters (the main figure model), 1.5B tokens. The law is fitted across 30-220M and holds there. Our 1.3-2M weights touched per token is 15x below, and our bit-widths (1.585) are below the tested 3-bit floor: double extrapolation. Authors state the constants "are unlikely to be useful" outside their setup.

## Training recipe and gotchas
- Max-abs scaling with no clipping search, no learned step, no mean-abs scale, no architecture change: a naive quantizer. Frantar et al. (BitNet-style scale, measured at 1/2/4 bits) get EPM 0.466/0.702/0.923 where this fit implies 0.07/0.36/0.70. The disagreement is mostly quantizer quality, and the authors concede (Sec. 6) that architectural tweaks close much of the gap.
- Naive low-bit activations and attention diverge at 3 bits or fewer; weights do not.
- PTQ damage rises with tokens/parameter: never plan on float training followed by PTQ to ternary. Low-precision training makes models more PTQ-robust (Finding 4).

## Mapping onto the GBA design
Nothing to implement at run time; this is a planning law.
- Their cost model (linear in bits) happens to match ours: 8-bit MAC 7.5 cycles vs ternary 1.6 = 4.7x for 5x bits. Effective weights per cycle under their (extrapolated) fit: ternary 0.25/1.6 = 0.16; 8-bit 0.93/7.5 = 0.12; under Frantar's numbers ternary is about 0.60/1.6 = 0.37. So ternary wins per cycle under both, but by 1.3x here versus 3x there; 2-bit or 4-bit table kernels could beat ternary under this law if they cost under about 2.3 or 4.5 cycles per weight.
- 8-bit activations: f_a about 0.89, acceptable; do not go below about 6 bits for residual-stream activations without a better quantizer.
- Sign/ternary attention keys sit far below their divergence point (P_kv <= 3). Their law says nothing usable there; rely on the dedicated binary-key papers and test early.
- ROM is free for us, so "bits are cheap, cycles are not": the law's memory-side argument does not apply.

## Verdict
**Skip as a planning constant, keep as a warning.** It does not overturn "widening beats bits": it has no data below 3 bits and uses a weak quantizer, so its low-bit penalty is an upper bound. It does show the conclusion is sensitive to the assumed law (1.3x to 3x). Settling experiment (same as the Frantar note): at our real width, train float, ternary, 2-bit and 4-bit weights at 3 widths each with 8-bit activations, fit the horizontal shift, divide by measured cycles per weight.

## Could not verify
- Meaning and sign of the offsets n_x (not defined in the text I saw); all f values above depend on it.
- g_post, C_w, the gradient rule, model width/depth tables, and all figures (not rendered).
- Exact N_eff/N at any bit-width is never stated numerically in the paper.
- Nothing here is from secondary sources.

## Links worth following
- Compression Scaling Laws (Frantar et al. 2025), already noted: https://arxiv.org/abs/2502.16440
- Spectra (measured ternary vs float size sweeps): https://arxiv.org/abs/2407.12327
