# Binarized Neural Machine Translation (https://arxiv.org/abs/2302.04907, NeurIPS 2023; Zhang, Garg, Cao, Lew, Ghorbani, Zhang, Firat — Google / Cornell)

Read in primary form (arXiv HTML full text, incl. Table 1, ablations, Appendix A). Figures were not readable as text; figure-only values are marked.

## What it is
A recipe for training encoder-decoder translation Transformers with 1-bit weights (and optionally 1-bit activations) end to end, not post-hoc. The core finding is that binarized dot products have variance proportional to the fan-in D, which breaks training; dividing by s ~ sqrt(D), or better a LayerNorm after every binarized linear layer, fixes it. A shortcut around the attention output projection is added. Weight-only binarization matches float on WMT17 En-De; binarizing attention activations does not. A scaling study (6-48 layers, 3B sentence pairs) shows 1-bit-weight models follow nearly the same power law as float, with a constant loss gap.

## Mechanism
Binarizer (from PokeBNN): `x_b = (floor(clip(x/B, -1+eps, 1-eps)) + 0.5) * B`, so x_b is in {-B/2, +B/2}. STE: gradient 1 for x in [-B, B], else 0.

Bounds, one per dot product: `B_W = max|W|` over the contraction (input) axis, i.e. one scale per output channel, static at inference; `B_A = max|A|` over d_model, per token, dynamic.

Variance argument: Var(A_b.W_b) = (B^4/16) * D, versus sigma_a^2 for Xavier float. With B in [1,3], D = 4096: at least 256x inflation.

Fix 1: `Dense(A_b) = A_b.W_b / s`, s proportional to sqrt(D) (s = 64 for D = 4096).
Fix 2 (adopted): LayerNorm with learnable gamma, beta in place of s.

1-bit FFN: `FFN(A) = LN( bin( LN( ReLU(A_b W1_b + b1) ) ) W2_b + b2 )`. The inner LN re-centres the all-positive ReLU output so bipolar {-1,+1} binarization is possible (binarized BERTs used {0,1} instead).

Attention: LN after each of the Q, K, V and output projections; extra shortcut around the output projection (Bi-Real-Net style). Not needed around QKV.

## Numbers
Table 1, WMT17 En-De, 6L6L, d=1024, 16 heads, FFN 4096, with KD (val loss / BLEU, beam 4):
- Row 1 float (bfloat16): 1.39 / 26.35
- Row 2 all dense weights 1-bit, activations float: 1.38 / 25.93
- Row 3 FFN weights + activations 1-bit, rest float: 1.40 / 25.44
- Row 4 all weights + FFN activations: 1.51 / 24.11
- Row 5 row 4 + QKV input activations: 1.72 / 21.55
- Row 6 row 4 + output-projection input activations: 1.60 / 21.06
- Row 7 row 4 + both: 1.89 / 17.87
- Row 8 all weights + QK einsum binarized (FFN activations float): 1.76 / 18.27
- Row 9 row 8 + score-V einsum binarized: 2.81 / 9.42

Scaling (Fig. 2; d=512, FFN 2048, 8 heads, weight-only binarization, no KD): exponents in-domain float p_e=0.18, p_d=0.31 vs binary 0.16, 0.28; OOD Wikipedia 0.13, 0.25 for both. A constant loss gap exists here (size only in the figure). 6L42L binary (195M params = 195M bits) equals the loss of 6L8L float (52M params, 832M bits): 4.3x fewer bits, but 3.75x more parameters, i.e. more compute. Binary is 2-3 BLEU below float at equal depth (Fig. 4a); 42L6L binary equals 10L6L float BLEU.

Ablations (Sec. 5): s <= 8 gives near-random translation; loss saturates at s = 64, slightly above 1.39, still improving slowly above 64 (Fig. 5). Attention output projection: fixed s gains about 0.2 loss over none; LN is better and recovers faster; LN + shortcut reaches about float (Fig. 6, no numbers in text).

## Smallest scale tested, and whether the result held there
Smallest: 6L6L at d=512/FFN 2048 (about 50M params; in-house data) and 6L6L at d=1024 (WMT). "Matches float" holds only for weight-only binarization on WMT with KD. On the large in-house corpus without KD there is a visible gap at every size including the smallest. Nothing at our 256-384 width, nothing decoder-only, nothing below tens of millions of parameters.

## Training recipe and gotchas
- Adam, beta = (0.9, 0.98), no weight decay, base LR 1e-3, batch 1024, dropout 0.1.
- Three stages, each a full cosine decay to zero: float (50,000 steps) -> weights 1-bit (88,339 steps) -> weights + activations 1-bit (88,339 steps). Binary runs used 9 epochs versus 5 for float.
- KD = replace labels by teacher softmax; used on WMT, not in the scaling study.
- Label smoothing hurt on large data.
- Much of the binary model's loss reduction comes in the last steps at vanishing LR (down to 2.5e-15); replacing the tail with a constant tiny LR made it worse. Decay fully to zero, and do not judge a QAT run before the end of its schedule.
- Activation binarization without the scale fix does not converge at all.

## Mapping onto the GBA design
Our plan (ternary weights, 8-16 bit activations) corresponds to row 2, the one configuration that matched float; it supports the from-scratch, staged-QAT approach. The per-output-channel weight scale B_W is a float in the paper: on GBA fold it into the following norm's gamma (ROM, one 8x16 multiply per channel, about 7.5 cycles) or constrain it to a shift.

The rule "normalise after every low-bit matmul" costs, per integer LayerNorm at d=256: 256 adds (mean), 256 square-accumulates, one rsqrt table lookup, 256 scale multiplies: roughly 4-6K cycles. Six norms per layer over 6 layers is about 150-200K cycles, about 4-5% of a 4M budget; no extra RAM beyond the vector. Cheaper: a fixed right shift s = sqrt(D) (shift 4 at D=256), which the paper shows converges but lands worse than LN. The extra shortcut costs 256 adds.

Warnings for our attention: 1-bit Q.K (row 8) cost +0.38 loss and about 8 BLEU; binarizing V/score mixing was catastrophic. This supports keeping values 8-bit. Our sign/ternary keys with multi-bit queries are an asymmetric case the paper did not test.

Our routed MLP experts: row 3 says fully 1-bit FFNs (weights and activations) are nearly free in quality when the rest is float (+0.01 loss), and the authors name MoE as the use case. A 1-bit expert input turns the 4-input partial-sum table into a 16-entry table of sign patterns, or pure XNOR-popcount.

## Verdict
Test. Adopt directly: staged float -> weight-quantized schedule with full cosine per stage, a norm (or sqrt(D) shift) after each ternary matmul, shortcut around the attention output projection. Settling experiment: at d=256-384 decoder-only on our data, compare (a) LN after every ternary projection, (b) fixed shift, (c) baseline pre-norm only; then (d) sign-binarized expert-MLP input activations versus 8-bit. Measure validation loss at equal steps.

## Could not verify
- Figure-only values: size of the in-house loss gap, Fig. 5/6 curves, BLEURT numbers.
- Step counts as printed (50,000 / 88,339) and "final 10 steps" wording look odd for a 3B-pair corpus; reproduced as written.
- No code link in the paper text I read; the paper says JAX/Flax. Whether it was released (possibly in Google's AQT library) was not checked.
- No inference-speed or energy measurements are given.
- The sqrt in "s proportional to sqrt(D)" was lost in the HTML-to-text conversion; inferred from s = 64 at D = 4096 and the attention sqrt(d_k) analogy.

## Links worth following
- PokeBNN (Zhang et al., 2022): source of the binarizer and training recipe.
- Bi-Real Net (Liu et al., 2018): the extra-shortcut argument.
- BiT, "Robustly binarized multi-distilled transformer" (Liu et al., 2022): {0,1} activations and incremental bit reduction.
- Ghorbani et al., 2021, "Scaling laws for neural machine translation": the fit form used.
