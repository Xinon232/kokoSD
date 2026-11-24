# Hamming Attention Distillation: Binarizing Keys and Queries for Efficient Long-Context Transformers

https://arxiv.org/abs/2502.01770 (v1, Feb 2025). Mark Horton, Tergel Molom-Ochir, Peter Liu, Bhavna Gopal, Chiyue Wei, Cong Guo, Brady Taylor, Deliang Fan, Shan X. Wang, Hai Li, Yiran Chen.

Read in primary form (arXiv HTML) through a fetch tool that condenses the page; figures were not visible. Items marked "derived" are my arithmetic.

## What it is

A fine-tuning recipe (HAD) that turns a pretrained transformer's query and key *activations* into +/-1 vectors, so attention logits become a Hamming distance, and keeps only the top-N logits per query before the softmax. Everything else (values, weights, other activations) stays full precision. The student starts as a copy of the teacher and is distilled on the teacher's attention logits and output logits. Tested on BERT-base, DeiT-B/T and T5-base only; no decoder LM.

## Mechanism

Inference (Eqs. 4-8):

```
Q = sign(Qc); K = sign(Kc)          # Qc = X W_Q, continuous
Al = Q K^T                          # integer in [-dk, dk], step 2
At = topn(Al, N)                    # keep N largest per query
A  = softmax(At / sqrt(dk)); out = A V
```

Scaling and softmax come after top-N. No learned temperature.

Training-time binariser, per layer, with sigma_Q, sigma_K = mean per-minibatch std of Qc, Kc over 100 random minibatches of 16:

- Stages 1-2: Q = c * sigma_Q * tanh(Qc / (c * sigma_Q)); same for K.
- Stages 3-4: Q = sigma_Q * STE(Qc / sigma_Q); forward sign, gradient 1 on [-1, 1], else 0.

Losses (Eqs. 9-11), unweighted sum:

- L_att = mean over heads/layers and queries of sum_j exp(A_t[i,j]) * (A_t[i,j] - A_s[i,j]), on attention logits.
- L_out = sum_i exp(z_t[i]) * (z_t[i] - z_s[i]) on output logits.

As written these use exp(logit), not a normalised softmax; I assume log-softmax is intended.

Schedule (Adam, batch 16, grad clip 0.5, c multiplied by 0.9998 per minibatch):

| Stage | Binariser | LR | Length | Loss |
|---|---|---|---|---|
| 1 | tanh, c 5.0 -> 1.0 | 1e-5 | ~8,000 steps (derived) | att + out |
| 2 | tanh, c 1.0 -> 0.05 | 1e-5 | ~15,000 steps (derived) | att + out |
| 3 | STE | 1e-5 | 10,000 | att + out |
| 4 | STE | 1e-6 | 10,000 | out only |

Top-N masking is on in stages 1-3 (stage 4 not stated explicitly). N = 30 for GLUE (context 256) and ImageNet (context 197); for longer contexts N scales linearly (15 at 128 up to 120 at 1024).

## Numbers

Table 1, GLUE, BERT-base, context 256, N = 30 (average of 8 tasks):

| Baseline | HAD | BiT | SAB variant | no att. distill | no tanh |
|---|---|---|---|---|---|
| 82.59 | 80.81 | 73.51 | 57.67 | 80.13 | 80.19 |

Loss 1.78 points for HAD against 9.08 for BiT (which binarises the whole model, so not like-for-like). Worst tasks: RTE 72.56 -> 65.70, MRPC 87.75 -> 83.82, MNLI 84.58 -> 82.45. CoLA rises 53.39 -> 55.47 (no explanation given).

Table 2, ImageNet top-1, N = 30 of 197:

| | Baseline | HAD | BiViT | SAB | no att. distill | no tanh |
|---|---|---|---|---|---|---|
| DeiT-B | 81.74 | 79.24 | 69.6 | 6.36 | 79.29 | 79.52 |
| DeiT-T | 72.01 | 66.59 | 37.9 | 4.32 | 66.42 | 66.78 |

On ImageNet both ablations match or beat the full recipe; the stages matter only on GLUE, by 0.6-0.7 points.

QuALITY, T5-base, contexts 128-1024: "within 3%" of baseline; values are in Figure 5 only.

Table 3, synthesised attention head (context 256, N = 30) against BF16: area 31.795 -> 6.724 mm2, power 25.491 -> 3.301 W. Node and latency not stated. Irrelevant to a CPU.

## Smallest scale tested, and whether the result held there

DeiT-Tiny (about 5M parameters, 192 wide; size from general knowledge, not this paper). It held least well there: 5.42 points lost against 2.50 for DeiT-B. The authors suggest low-capacity attention is hit disproportionately. That is the regime we are in.

## Training recipe and gotchas

- Fine-tuning from a converged full-precision teacher, not from scratch; about 43K steps at batch 16 in total.
- sigma is estimated once before training; T5 needed it because its Q/K std is far from 1.
- The top-N sweep on DeiT-T reportedly improves from N = 100 down to about 30, then falls. Nothing near our ratio (8 of 256, 3%) is reported in text.
- Integer logits tie often; tie handling in top-N is not stated.
- Only Q/K activations are binarised. Combining with ternary weights and 8-bit values is untested here.
- Authors note decoder-only LLMs "are very sensitive to small loss increases"; left as future work.

## Mapping onto the GBA design

- Storage: dk sign bits per cached key. At dk = 64, 8 bytes per position per head; 256 positions = 2 KB per head, in slow RAM.
- Scoring: XOR query with key and popcount. No popcount instruction, so use a 256-byte table in fast RAM: 8 lookup+adds per position, about 2,000 per head per token, zero multiplies. A few tens of thousands of cycles per head (my estimate), small against 3-5M.
- Top-N: scores take only dk+1 values, so a 65-bin histogram gives an exact threshold in one pass; no sort.
- Softmax: exp(score/sqrt(dk)) has dk+1 possible inputs, so it is a 65-entry ROM table per layer, with sigma_Q*sigma_K folded in. Normalising 8 weights needs one reciprocal (table).
- Value mix: 8 positions x width MACs at 7.5 cycles; this dominates and the paper does nothing for it.
- Changes needed: causal masking (early positions have fewer than N candidates), N = 8, training from scratch or from our own float teacher.

## Verdict

Test. The inference form matches our design and confirms sign Q/K plus top-N is viable for encoders at about 2 points cost. The elaborate schedule is weakly supported: plain STE with output distillation gets nearly all of it. Settling experiment: on our float teacher, compare (a) STE + sigma normalisation + output KL, (b) plus attention-logit KL, (c) plus tanh stages, at N in {8, 16, 30}, measuring validation nats per token.

## Could not verify

- Per-N accuracies (Figure 3) and QuALITY accuracies (Figure 5): figure-only.
- Stage 1-2 step counts, ImageNet iteration counts, whether top-N is applied to the teacher or inside the attention loss.
- Whether sigma scales the inference logits (Eqs. 4-8 omit it; training equations include it).
- No code release found in the paper. No perplexity results of any kind.

## Links worth following

- BiT (Liu et al., 2022), the GLUE comparison baseline: https://arxiv.org/abs/2205.13016
- BiViT (source of SAB): cited in the paper; URL not checked.
