# RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs

https://arxiv.org/abs/2602.05367 (v1 Feb 2026, v3 Jun 2026, ICML 2026). You, Lee, Choi, Kim, Chong, Kim, Kim, Kim — Samsung Research. Code: https://github.com/SamsungLabs/RaBiT (Apache-2.0).
Read in primary form: arXiv HTML v3 (via a summarising fetch tool, so numbers are primary but not character-checked) plus `quantization/functions/binary.py` and `quantization/modules/rabit.py` from the repo.

## What it is
A QAT method for "2-bit" LLMs built as the sum of two 1-bit matrices, each with its own per-row and per-column float scales. The claim: if the two binary planes are trained as independent latent weights, they co-adapt and learn redundant features ("inter-path adaptation"). RaBiT keeps ONE latent full-precision weight and re-derives both planes from it on every forward pass, so plane 2 is always the sign of plane 1's residual. It is a compress-a-pretrained-model method (distillation from the FP teacher), not from-scratch training.

## Mechanism
Per linear layer, latent `W` (out x in); per path i: `g_i` (out,1), `h_i` (1,in), all learnable floats.

```
R = W
for i in 1..k:                      # k=2 -> ~2.02 bits/weight incl. scales
    B_i = sign(R); B_i[B_i==0] = +1
    if i < k: R = R - g_i * B_i * h_i      # scales NOT detached; no autograd through this anyway
W_eff = sum_i g_i * B_i * h_i
y = x @ W_eff.T (+ bias, bias kept)
```
Backward (hand-written autograd Function in the repo):
- `dW = dY^T x` — the gradient w.r.t. `W_eff` copied straight onto `W` (unclipped STE; the dependence of the residual on W and on the scales is ignored).
- `dg_i = sum_n D_n * (B_i (h_i * x_n))`, `dh_i = sum_n (B_i^T (D_n * g_i)) * x_n`, B treated as constant.

Initialisation (two parts):
1. I/O importance preconditioning: `W' = s_out^a_out * W * s_in^a_in`, with s_in = max-abs input activation, s_out from output gradients on a calibration set; scales are divided back out afterwards. Grid-searched a_in 0.75-0.95, a_out 0.45-0.70.
2. Iterative residual SVID (Gauss-Seidel, T=20): for each path, `R_i = W' - sum of the other paths' current estimates`; `(B_i,g_i,h_i) = SVID(R_i)` (sign + rank-1 SVD of |R_i|, as in OneBit).

Loss: KL to teacher logits + gamma * intermediate-layer MSE (gamma=100 Llama, 0 for Gemma3 because MSE was unstable).
Inference: `y = g1*(B1(h1*x)) + g2*(B2(h2*x))`; W discarded.

## Numbers
All 2-bit, pretrained-then-QAT, eval context 4096 (paper Table 2 unless noted).
- Ablation, Llama2-7B Wiki2 ppl: independent-latent QAT 6.55 (6.18 with both init tricks); coupled 5.84 (5.78 with both). FP16 5.12. So coupling is worth ~0.4-0.7 ppl; the init tricks only ~0.06 once coupled.
- Llama2-7B: RaBiT 5.78 / C4 7.64 / QA avg 61.51; QTIP 5.86 / 7.73 / 58.97; DBF 6.10 / 8.05 / 58.42; MBOK 6.99; DB-LLM 7.23.
- Llama3-8B: FP 5.75, RaBiT 7.34, QTIP 7.52, DBF 7.78. Llama2-13B: RaBiT 5.15 vs QTIP 5.11 (loses).
- Gemma3-1B: RaBiT 11.27 / 15.54 / QA 53.18; DBF 13.28; QTIP 13.14; FP16 QA 57.82.
- Qwen3-1.7B: RaBiT 10.19, DBF 11.89, QTIP 11.19.
- 3-bit (k=3) Wiki2: 5.36 (L2-7B), 4.84 (13B), 6.58 (L3-8B).
- Training cost (Table 12, Llama2-7B, 4xH100, per epoch): single-path 2-bit QAT 5.67 h; independent two-latent 10.33 h; RaBiT 6.50 h (+14.7%). Half the optimizer state of two latents.
- Total: 6 epochs over 200M tokens (WikiText-2 + C4), Gemma3-1B 8 h on 4xH100, Llama2-7B 39 h.
- Seeds: 5.78 +/- 0.005 over five seeds; AdamW matches Muon (5.78).
- Speed (RTX 4090, Table 7): 291.88 tok/s vs 64.96 FP16 (4.49x); 3-bit 191.63 tok/s.
- Table 1 path statistics (Llama2-7B): Corr(y1,y2) about -0.08..-0.13 independent vs -0.34..-0.50 RaBiT. The text elsewhere calls the independent case "strongly positive" — an inconsistency I could not resolve.

## Smallest scale tested, and whether the result held there
Gemma3-1B (pretrained, fine-tuned 8 h). The advantage over DBF/QTIP is largest there (11.27 vs ~13.2), so the effect grows as models shrink — but 1B is ~1000x our dense parameter count, and nothing is trained from scratch. No sub-100M result, no ternary or BitNet baseline, no activation quantisation.

## Training recipe and gotchas
- Muon (AdamW equivalent), cosine decay, LR 5e-6..1.2e-5, warmup 0.03, batch 4/GPU, 6 epochs. These tiny LRs are fine-tuning values; a from-scratch latent would need BitNet-style LRs.
- Sign(0) forced to +1.
- Scale gradients are hand-derived; the STE is identity with no clipping.
- Repo oddities: `pack_weights` state-dict filter `startswith('w')` would also drop `weight`; README example checkpoints are placeholders (no confirmed released weights).
- Intermediate MSE distillation diverged on Gemma3 — do not assume it helps small models.

## Mapping onto the GBA design
Kernel cost for a 2-plane binary layer (my derivation, not in the paper):
- Column scales `h_i`: fold into the inputs before building tables. Cost: d_in multiplies per path per layer (~7.5 cycles each; 384 x 2 = ~6K cycles, <3% of a 384x384 matrix).
- Each path needs its own table per 4 inputs (16 entries, 8 with sign symmetry — cheaper to build than the 81-entry ternary table), and one ROM byte holds both 4-bit indices. So a row costs TWO lookups+adds per 4 weights (plus nibble extraction), roughly 2x our 1.6 cycles/weight, for 2 bits/weight of ROM.
- Row scales `g_i`: if g1 and g2 differ per row, two accumulators and two multiplies per row (cheap: 2 x d_out multiplies). If `g2 = rho * g1` with a per-layer rho, rho folds into table 2 and one accumulator suffices. Either way the second lookup is unavoidable; tying only saves the extra accumulator/multiply. A single merged 256-entry table (byte indexes directly, weights in {+-1 +-rho}) is one lookup but 256 table entries per 4 inputs — only worthwhile when d_out >> 256.
- Transfer to our ternary format: `T1 = tern(W)`, `T2 = tern(W - a1*T1)`: 3.17 bits, two 81-entry lookups per 4 weights.
- Free by-product: per-row/per-column scales on a SINGLE ternary matrix cost only d_in + d_out multiplies per layer. That is an expressivity gain at ~2-3% compute and is the most directly usable idea here.
- RAM: second table set (16-81 entries x 16-bit) — trivial. ROM: 2x per matrix, irrelevant at our budget.
- The catch: compute, not ROM, is our scarce resource. A two-plane layer costs about the same cycles as a ternary layer with twice the weights, and this paper gives no evidence on that trade.

## Verdict
TEST (low-medium priority). Adopt the coupling rule unconditionally IF we ever use multi-plane weights — it is free at training time (+15%) and independent planes are measurably worse. But whether two planes beat one ternary plane of 2x the parameters at equal cycles is unproven. Settling experiment: at our scale, from scratch, same cycle budget: (a) ternary d=384; (b) coupled residual-binary or residual-ternary at d~272 (half the weights); (c) ternary d=384 plus learned row/column scales. Compare val loss. Likely useful spots: the small high-leverage matrices (attention Q/K/V projections, class head) rather than routed MLP experts.

## Could not verify
- Which layers are binarised (embeddings/head), and training sequence length — not found.
- FP16 perplexity for Gemma3-1B/Qwen3 not captured.
- FP16 QA average for Llama2-7B appeared as 62.26 in one table reading and 62.40 in another.
- No from-scratch, ternary, BitNet or <1B results exist in the paper.
- All GBA cost figures are my estimates.

## Links worth following
- Code: https://github.com/SamsungLabs/RaBiT (`quantization/functions/binary.py`)
- OneBit (SVID init) and DBF (Double Binary Factorization) — cited baselines; URLs not fetched here.
