# ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization  (https://arxiv.org/abs/2502.02631, v2, 2025, Zechun Liu, Changsheng Zhao, ... Vikas Chandra, Meta; code https://github.com/facebookresearch/ParetoQ)

Tags: [P] paper HTML (v2, read in full); [R] repo source/README on `main`; [D] my derivation.

## What it is
A unified quantization-aware fine-tuning recipe for 1, 1.58 (ternary), 2, 3 and 4-bit weights, applied to MobileLLM 125M-1.5B and LLaMA-3 1B/3B/8B [P]. Two contributions: a training-budget study (full-precision pretraining first, then a short QAT phase) and a per-bit-width quantizer with a learned per-row scale, "Stretched Elastic Quant" (SEQ) for ternary and 2-bit [P]. It concludes ternary, 2-bit and 3-bit tie on accuracy per stored bit and beat 4-bit and binary [P]. Only weights are quantized; activations stay 16-bit float and embeddings/output head are untouched in the main tables [P].

## Mechanism
Per linear layer, latent float W (out x in), learned scale alpha of shape (out, 1), i.e. one per output row [R].
```
# ternary (repo w_bits=0): n=1.5, shift=0     2-bit: n=2, shift=0.5
u   = clamp(W/alpha, -0.99, 0.99)             # paper says clip to [-1,1]; repo uses 1-1e-2
q   = (round(u*n - shift) + shift) / n        # ternary: {-2/3, 0, +2/3}; 2-bit: {+-1/4, +-3/4}
W_q = alpha * q                               # ternary zero iff |w| < alpha/3
dL/dW     = g * 1[|W/alpha| < 0.99]           # STE
dL/dalpha = sum_row g * gs * (q - W/alpha inside | -+Qp outside)
gs = 1/sqrt(numel(W) * Qp)                    # LSQ gradient scale (Qp taken from the integer range) [R]
init: alpha = max|W| per row (ternary, 2-bit); mean|W| (1-bit); max|W|/(2^(b-1)-1) (3/4-bit LSQ)
```
- Ternary init puts the zero threshold at max|w|/3 per row, so most weights start at zero for bell-shaped rows [D]; alpha then learns down.
- Fig. 6 [P]: learned scale beats statistics-based ranges (min-max, clipped) at every bit-width; SEQ beats LSQ at ternary and 2-bit; LSQ slightly better at 3/4-bit. Only figures, no table.

## Numbers
Avg = mean of 8 zero-shot tasks; Wiki2 = perplexity. Tables 2-5 [P], 1-bit rows from repo README [R].

| MobileLLM | FP | 1-bit | Ternary | 2-bit | 3-bit | 4-bit |
|---|---|---|---|---|---|---|
| 125M Avg | 48.5 | 43.7 | 46.5 | 47.1 | 47.9 | 48.7 |
| 350M Avg | 54.8 | 47.6 | 51.4 | 52.8 | 54.7 | 55.0 |
| 600M Avg | 58.2 | 50.3 | 55.5 | 56.6 | 58.5 | 59.2 |
| 1.5B Avg | 62.7 | 55.9 | 60.5 | 61.8 | 62.9 | 63.6 |

- Ternary Wiki2 vs FP: 19.8 vs 14.9 (125M), 14.4 vs 10.4, 11.4 vs 9.0, 10.0 vs 8.0, 9.0 vs 7.8 (1.5B). Ratio 1.33, 1.38, 1.27, 1.25, 1.15 [D].
- Parameters to match FP [D from tables]: ternary 350M (51.4) > FP 125M (48.5); ternary 600M (55.5) ~ FP 350M (54.8); ternary 1B (58.5; row omitted above) ~ FP 600M (58.2). Roughly 1.7-2.8x parameters. Prior ternary: 1-bit-era 700M 48.0 / 17.3; LLM-QAT 125M 39.8 / 87.3 (Table 2) [P].
- Budget (Fig. 2, 125M, 100B tokens total, ratios 0...1): accuracy peaks near 90% FP + 10% QAT for ternary/2/3/4-bit; drops sharply beyond 90%; QAT from scratch and PTQ both worse [P]. No numeric table.
- QAT saturation (Fig. 3): ~10B tokens for 3/4-bit, ~30B for 1/1.58/2-bit [P].
- Weight movement (Fig. 4): L1 change 10-20% at 3/4-bit, ~40% at 2-bit and below ("reconstruction") [P].
- Embeddings (A.6, Fig. 11): 4-bit embeddings, or embeddings at the weight bit-width, are on the size-accuracy Pareto front; 8/16-bit are not [P, figure only].

## Smallest scale tested, and whether the result held there
MobileLLM-125M (30 layers, width 576 per the MobileLLM paper, not restated here). The recipe held: ternary loses 2.0 Avg points and 33% perplexity. The gap is largest at the small end and at 350M, so expect worse below 125M. Nothing under 125M, nothing from-scratch at our size.

## Training recipe and gotchas
- QAT: AdamW, weight decay 0, lr 2e-5 cosine to 0, no warm-up, 120k iterations at 16 GPUs x batch 8 for 1/1.58/2-bit; lr 1e-5 and 40k iterations for 3/4-bit [P][R]. The budget study instead used QAT lr 1e-4 with linear decay (FP: 2.5e-3, wd 0.1) [P A.5]. No distillation, no activation quantization. At 2048 context that is ~31B tokens [D].
- Inconsistencies: text says the ternary 600M scores 58.7, Table 2 says 55.5. Paper Wiki2 for ParetoQ 2-bit and 4-bit 125M (25.1, 20.4) disagree with the released checkpoints (18.2, 14.1) [P vs R]; ternary agrees (19.8 vs 20).

## Mapping onto the GBA design
- Inference cost: none beyond the existing ternary kernel. ROM holds the trits; the per-row alpha*2/3 is one multiplier per output row (384 hardware multiplies per matrix-vector product, ~3K cycles, under 0.1% of budget) or folds into the next requantization multiplier [D].
- SEQ's zero fraction is free and learned; the 32-entry "one zero per 4" format fixes it at 25%. Either keep the 81-entry table for SEQ, or use SEQ's learned per-row alpha only as the output scale on top of the 3:4 pattern (untested combination).
- Schedule maps directly: spend ~90% of GPU-hours on the float model (which is also the distillation teacher), ~10% on ternary QAT at ~100x lower lr.
- Hashed n-gram and per-layer embedding tables: A.6 supports storing them at 4 bits or less, doubling entries per ROM byte versus 8-bit.

## Verdict
Adopt the schedule (FP first, short low-lr QAT, wd 0) and the per-row learned scale with LSQ gradient scaling; test SEQ against absmean/median and against 3:4. Settling experiment: at width 256-384, fixed token budget, compare (a) QAT from scratch, (b) 90/10, (c) 70/30 with Arenas-style residual; and SEQ vs absmean, each with 8-bit activations on.

## Could not verify
- Fig. 2, 3, 4, 6, 7, 11 values exist only as plots; I report captions and text, not read-off numbers.
- Training corpus is not named. Whether SEQ helps with quantized activations or below 125M is not shown.
- Whether the 90/10 optimum survives a total budget far below 100B tokens: not tested.

## Links worth following
- https://github.com/facebookresearch/ParetoQ (models/utils_quant.py, train.py)
- https://huggingface.co/facebook/MobileLLM-ParetoQ-125M-1.58-bit
- MobileLLM: https://arxiv.org/abs/2402.14905

## Independent verification
Second reader, 2026-11-18. Read directly: arXiv HTML v2 (https://arxiv.org/html/2502.02631v2; its title reads "ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization") and `models/utils_quant.py` on the repo `main` branch. Figures were not viewed as images; only captions and body text.

1. **90% FP + 10% QAT beats QAT-from-scratch and PTQ (Fig. 2) - confirmed, with wording softer than the claim.** Fig. 2 caption, Sec. 2.1 and A.5: MobileLLM-125M, 100B tokens total, ratios 0 to 1, four bit-widths (1.58/2/3/4; binary not in this study). The text says the peak is at ~90% "for almost every bit-width choice" and that optimal performance is "nearly achieved" with ~10% QAT, so it is not claimed to be the exact optimum for each bit-width, and which bit-width is the exception is not stated. No numbers, plot only. The sweep used its own settings (QAT lr 1e-4, linear decay, A.5), different from the main-table recipe (2e-5, cosine). Only one model size and one total budget were tested.
2. **Ternary costs 2.0 points and 1.33x perplexity at 125M - confirmed for the numbers, partly for the attribution.** Table 2: ParetoQ ternary 125M Avg 46.5, Wiki2 19.8. Table 3 (the 2-bit table carries the FP row): FP 125M Avg 48.5, Wiki2 14.9. 19.8/14.9 = 1.33. "Per-row scale" is not stated in the paper text; it comes from the repo (`weight_clip_val` has shape `(weight.shape[0], 1)`). "Activations unquantized" is not stated explicitly either; it is inferred from the quantizer touching weights only and from the latency section's W2A16/W4A16 kernels. Embedding and output layers are also left unquantized in these tables (Sec. 5.1), so the 2.0-point gap excludes any loss from quantizing those. The ternary result is fine-tuned from an already fully pretrained MobileLLM with 120k iterations (the paper calls this 30B tokens), i.e. the QAT phase is extra budget, not 10% of a fixed one.
3. **Saturation at ~30B vs ~10B tokens; ~40% vs 10-20% L1 change - partly.** The numbers are as stated (Fig. 3 caption, Sec. 2.2 items 2-3, Fig. 4 caption). What differs: (a) Fig. 3 is said to span "7 architectures and 5 bit levels" without naming them in the text, so "125M-8B" is an assumption (the paper uses eight models overall); (b) the text does not say which model(s) Fig. 4 was measured on; (c) the L1 figure is relative change of latent float weights from the FP init, described for binary/ternary/2-bit together, not ternary specifically; (d) "typically saturates" is the paper's wording, plot only.
4. **4-bit or weight-matched embeddings are on the Pareto front, 8/16-bit are not - confirmed as a statement, thinly supported.** A.6 is one sentence plus Fig. 11 (plot only). The text does not say which models, which embedding quantizer, or whether the output head is tied/quantized together with it; "MobileLLM family" is not stated in the text I read. The only described embedding quantization elsewhere is 8-bit channel-wise in the CPU latency test (A.4).

**Overall.** The table numbers are transcribed correctly, but everything about schedule, saturation and embeddings rests on plots and one-line statements at 125M parameters and above, with float activations, a float embedding/output layer, a fully pretrained starting point and a 30B-token QAT phase. Our model touches roughly 100x fewer weights per token, uses 8-bit activations, hard routing and hashed lookup tables, and will see far fewer training tokens, so treat the 90/10 split and the 2-point/1.33x gap as a direction (FP first, then QAT) and a lower bound on the loss, not as numbers to plan with. Missed caveat: the paper's own trend is that the ternary gap widens as models shrink (and it argues 2-bit is often the more practical deployment choice), and the 10% figure is tied to a 100B-token budget in which 10% is already 10B tokens; at a budget of a few billion tokens, 10% may be far below the ~30B the same paper says ternary needs to saturate.
