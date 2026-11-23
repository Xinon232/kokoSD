# BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers

https://arxiv.org/abs/2603.09582 (v1, March 2026; CVPR 2026 per the repo README). Chaodong Xiao, Zhengqiang Zhang, Lei Zhang (HK PolyU / OPPO Research). Code: https://github.com/EdwardChasel/BinaryAttention

Read in primary form: arXiv HTML (main text and appendices A-D, through a summarising fetcher, so table cells are as it reported them) and the repo files `README.md`, `models.py`, `utils.py`, `losses.py`, `main.py` (raw text, read directly). Sibling note `f-ternary-11-...` covers the quantisation angle; this one covers the position scheme and recipe.

## What it is

Attention in which queries and keys are replaced by their signs, so the score is XNOR+popcount, with softmax output and values kept at 8 bits. Lost magnitude information is put back by a per-head mean-magnitude scale and an additive learnable relative-position bias on the score. Models are fine-tuned from float checkpoints with STE-based QAT and distillation from the float model. Weights and MLPs stay float. Evaluated on DeiT (ImageNet, COCO, ADE20K) and DiT/SiT; no language experiments.

## Mechanism

Per head, head dim d, N tokens (from `models.py`, `Attention.forward`):

```
mu(x) = mean over tokens and channels of |x|     # per sample, per head, recomputed each forward, gradient flows
q' = mu(q) * sign(q);  k' = mu(k) * sign(k)      # sign(0) = +1
S  = (q' @ k'^T) / sqrt(d) + B[h][rel_index(i,j)]
P  = softmax(S);  P8 = clamp(round(255*P), 0, 255) / 255
V8 = symmetric 8-bit, scale = 127 / max|v| over tokens (per channel)
y  = P8 @ V8
```

- Bias B: Swin/BEiT table, `(2H-1)(2W-1)+3` rows x `num_heads` columns (732 x heads for a 14x14 grid; 3 extra rows for the class token), one table per layer, trunc-normal init std 0.02, excluded from weight decay.
- The absolute learned position embedding added at the input is kept (`x + self.pos_embed`). So the bias is an addition to a model that already has position information, not a replacement.
- STE: sign passes gradient where |x| <= 1, zero elsewhere (`STESign`). Rounding uses `z + (round(z) - z).detach()`. V quantiser zeroes gradient outside [-2, 2].
- Theorem 1: for zero-mean Gaussian q, k, E[sign(q) sign(k)^T] = (2/pi) arcsin(C). Sign similarity is a monotone function of cosine similarity.

## Numbers

Table 5, ImageNet-1K top-1, DeiT-T (6M) / S (22M) / B (87M); float baseline 72.2 / 79.8 / 81.8:

| scale | bias | distill | T | S | B |
|---|---|---|---|---|---|
| no | no | no | 71.95 | 79.59 | 81.10 |
| yes | no | no | 72.42 | 79.81 | 81.33 |
| yes | no | yes | 72.44 | 79.97 | 81.99 |
| yes | yes | yes | 72.88 | 80.24 | 82.04 |

Bias gain: +0.44 / +0.27 / +0.05. Scale gain: +0.47 / +0.22 / +0.23. Distillation: +0.02 / +0.16 / +0.66.

- Table 7 (DeiT-T): dense bias 72.88, memory 29904M at 1024 px; "decomposable" bias 72.97 at 5307M (FlashAttention2 5304M).
- Table 9 (no bias): 100 epochs 71.98 / 79.44 / 81.80; 300 epochs 72.44 / 79.97 / 81.99.
- Table 6 (DeiT-B): cosine similarity to float attention maps 0.9186 at layer 0, 0.8740 at layer 6.
- Table 4, FID: DiT-XL/2 cfg 1.25: 3.22 float vs 4.26 binary (worse); cfg 1.50: 2.27 vs 2.19.

## Smallest scale tested, and whether the result held there

DeiT-Tiny: 6M parameters, width 192, 3 heads of 64 dims, 12 layers, 197 tokens. It held, and the bias mattered most there. This is bidirectional vision, float weights, float-initialised.

## Training recipe and gotchas

- Fine-tune from the float checkpoint: AdamW, 300 epochs, batch 1024, lr 1e-4 to 1e-5 cosine, 5 warm-up epochs, weight decay 0.02, drop path 0.1 (Appendix C; README shows lr 5e-5 at 8x128, which DeiT's lr scaling turns into the same 1e-4).
- Distillation (`losses.py`, unmodified DeiT): loss = 0.5 * task loss + 0.5 * distillation, where `hard` is cross-entropy against the teacher's argmax and `soft` is KL at tau = 1.0. The README command shows `hard` and says "defaults to tiny disable, small and base enable", which matches distillation being worth only +0.02 on Tiny.
- Seed fixed at 0 in `main.py`; no repeated runs, so +0.44 is suggestive only (my judgement).
- No ablation of bias without distillation, of bias without the absolute embedding, or of training from scratch. No RoPE or other position scheme compared.
- mu is a per-input statistic over all tokens: in a causal LM it would leak a scalar from future tokens during training and is not computable when streaming.

## Mapping onto the GBA design

Cost figures are my estimates.

- Bias table: causal 1-D distance, 256 entries x heads x layers of int8; 4 heads x 6 layers = 6 KB ROM. Inference: one byte load and one add per scored position, a few thousand cycles per layer at most.
- Fold the scale into the table: top-8 selection needs only ranking, so store b' = round(b / c_h) in popcount units, with c_h a learned per-head constant replacing mu_q*mu_k/sqrt(d). c_h is then needed only for the 8 survivors' softmax, via an exp lookup table.
- Free extension: index the bias by (distance bucket, token class or hash). The paper names "context-aware" bias but does not test it.
- No extra RAM. Not covered: ternary weights, top-k truncation, causal masking, ternary keys.

## Verdict

Test, leaning adopt. An additive distance-indexed bias is the cheapest position scheme available to a sign-key scorer, and it is the only one with small-model evidence of surviving sign keys. The evidence is one seed, vision only, on top of an absolute embedding. Settling experiment on the pilot LM with sign keys and top-8: {no position, absolute embedding only, bias only, both, RoPE applied before sign} x {fixed scale, none}, validation loss at equal steps, 2-3 seeds.

## Could not verify

- The formula for the "decomposable" bias (only its Table 7 row).
- Whether Tiny's headline 72.88 used distillation: Table 5 says yes, the README default says no.
- Any language, causal, from-scratch or RoPE result: none exists in the paper.

## Links worth following

- https://github.com/EdwardChasel/BinaryAttention (`models.py` lines 60-160)
- Swin Transformer, source of the bias table: https://arxiv.org/abs/2103.14030 (ID from memory, not re-checked)
