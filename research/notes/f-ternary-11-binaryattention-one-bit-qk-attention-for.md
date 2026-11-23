# BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers

https://arxiv.org/abs/2603.09582 (v1, 10 Mar 2026). Chaodong Xiao, Zhengqiang Zhang, Lei Zhang (HK PolyU / OPPO Research). Code: https://github.com/EdwardChasel/BinaryAttention (Apache-2.0). The repo README says CVPR 2026; the arXiv HTML page lists no venue.

Read in primary form: arXiv HTML (main text + appendices A-D), repo README, `models.py`, `utils.py`, `main.py`. All fetches went through a summarising reader, so table cells are as that reader reported them, not eyeballed by me.

## What it is

Attention where queries and keys are reduced to their signs, so Q.K^T is XNOR+popcount, while the softmax output P and values V are quantized to 8 bits. Two corrections make it work: a per-head mean-magnitude scale on each side, and a learnable real-valued bias added to the score (relative-position bias in the experiments). Models are fine-tuned from full-precision checkpoints with QAT (STE) and optional self-distillation from the float model. Only attention is quantized; MLPs and all weights stay float. Tested on DeiT (ImageNet, COCO, ADE20K) and DiT/SiT; no language model.

## Mechanism

Per head, with q_i, k_j in R^d (d = head dim):

- s_i = mu_q * sign(q_i), t_j = mu_k * sign(k_j)  (Eq. 5). Code: `mu = x.abs().mean(dim=-2).mean(dim=-1)`, i.e. mean |x| over all tokens and channels of that head, recomputed per input (not a stored parameter). sign(0) = +1.
- Score S_ij = mu_q * mu_k * sign(q_i).sign(k_j) / sqrt(d) + b_ij  (Eq. 6).
- b_ij: Swin/BEiT-style relative-position table, `nn.Parameter((2H-1)(2W-1)+3, num_heads)`, trunc-normal init std 0.02, gathered by a fixed index buffer and added before softmax. Paper also mentions dense and "decomposable" bias variants; I did not find the decomposable formula.
- P = softmax(S); P~ = round(255*P) in [0,255]; V~ = 8-bit symmetric, scale from max |v| along the token axis (per channel); y_i = (delta_v/255) * sum_j P~_ij V~_j  (Eq. 7).
- STE: sign backward passes gradient where |x| <= 1, zero outside. Rounding uses `z + (round(z)-z).detach()`. V quantizer zeroes gradients outside [-2, 2].
- Theory (Theorem 1): for zero-mean Gaussian q, k, E[sign(q) sign(k)^T] = (2/pi) arcsin(C), C the correlation matrix. So the sign dot product is a monotone function of cosine similarity; what is lost is magnitude, which the scale and bias are meant to restore.

## Numbers

ImageNet-1K top-1, 224 px (Table 1), float DeiT vs BinaryAttention:
- Tiny (6M): 72.2 vs 72.88
- Small (22M): 79.8 vs 80.24
- Base (87M): 81.8 vs 82.04; Base 384 px: 83.1 vs 83.64

Ablation, top-1 T / S / B (Table 5):
- sign only (no scale, bias, distillation): 71.95 / 79.59 / 81.10
- + scale: 72.42 / 79.81 / 81.33
- + scale + distillation: 72.44 / 79.97 / 81.99
- + scale + bias + distillation: 72.88 / 80.24 / 82.04

So on Tiny: scale +0.47, distillation +0.02, bias +0.44. On Base distillation is the big term (+0.66) and bias small (+0.05). Bias without distillation is not isolated.

Schedule (Table 9, no bias): 100 epochs 71.98 / 79.44 / 81.80; 300 epochs 72.44 / 79.97 / 81.99. Short fine-tuning leaves Tiny/Small below float.

Attention fidelity on DeiT-B (Table 6): cosine similarity to float attention 0.919 at layer 0, 0.874 at layer 6; the maps are similar, not identical.

Diffusion FID (Table 4): DiT-S/2 cfg 1.5: 43.87 float (400K steps) vs 38.96 binary (pretrained + 200K more, so not an equal-compute comparison). DiT-XL/2 cfg 1.25: 3.22 vs 4.26 (binary worse); cfg 1.5: 2.27 vs 2.19.

Dense tasks (Tables 2, 3): within about +-0.3 AP / mIoU of DeiT, except Base ADE20K 46.86 -> 47.76.

Speed (Table 8, A100 kernel): 175.3 ms FlashAttention2, 124.6 SageAttention, 88.2 BinaryAttention. Irrelevant to ARM7.

## Smallest scale tested, and whether the result held there

DeiT-Tiny: 6M parameters, width 192, 3 heads (head dim 64), 12 layers, 197 tokens. It held: +0.68 over float with the full recipe, and even bare sign attention lost only 0.25. Width and sequence length are close to ours (256-384 wide, 256 positions). But it is bidirectional vision with float weights and float MLPs, initialised from a converged float model.

## Training recipe and gotchas

- Fine-tune from full-precision weights; AdamW, 300 epochs, batch 1024, lr 1e-4 -> 1e-5 cosine, 5 warm-up epochs, weight decay 0.02, drop path 0.1 (Appendix C). That is 10x lower lr than DeiT from scratch.
- Distillation: teacher is the same architecture in float. `main.py` default is `none`; the README comment says "tiny disable, small and base enable" and the command shows `hard`, which sits oddly with Table 5 having a distillation row for Tiny. Loss form not stated in the paper.
- From-scratch training with sign Q/K is not reported.
- mu_q, mu_k are batch statistics over all tokens of the input. In a causal, streaming LM this is not available (and would leak future tokens in training).
- STE zeroes gradient for |q| > 1, so pre-sign activations must stay in a sane range (pre-norm helps).

## Mapping onto the GBA design

(Cost figures here are my estimates, not from the paper.)

- Score: our keys are already sign/ternary. Per head, score = popcount-based integer in [-d, d].
- Scale: replace the data-dependent mu_q*mu_k/sqrt(d) by one learned constant c_h per head and layer (train with EMA or a learnable scalar). Top-8 selection only needs the ranking, so fold the bias into popcount units: b'[h][dist] = round(b[h][dist] / c_h), an int8.
- Bias: 1-D relative distance = (write pointer - slot) mod 256. Table is 256 bytes per head per layer; 4 heads x 6 layers = about 6 KB of ROM. Cost: one byte load + one add per scored position per head, roughly 256 x heads x ~4 cycles, i.e. a few thousand cycles per layer against a 3-5M budget.
- Extension that stays free: index the bias by (distance bucket, query token class) or by a hash, since lookups cost nothing; the paper calls this "context-aware" bias but does not test it.
- Softmax over the 8 survivors: exp LUT on (score - max) * c_h, normalise to 8-bit weights summing to about 255, then 8 x d multiply-accumulates with 8-bit values. This is exactly Eq. 7 and is what the paper validates (8-bit P, 8-bit V).
- RAM: nothing new beyond the key bit-planes already in the ring buffer.
- Not covered by the paper: top-k truncation, ternary weights, causal masking, text.

## Verdict

Test (cheap, likely adopt). The bias costs almost nothing at inference and was worth +0.44 top-1 at our width in the only small-scale ablation; the scale was worth +0.47. Evidence is vision-only and from float-initialised fine-tuning, so it does not prove the gain for a from-scratch ternary causal LM. Settling experiment: on the pilot LM, 2x2 of {fixed per-head scale, none} x {relative-position bias table, none} with sign keys and top-8, compared on validation loss against float-QK attention at equal steps; add a fifth arm with token-class-conditioned bias.

## Could not verify

- Formula for the "decomposable" bias (only its memory figure, Table 7).
- Distillation loss and which model sizes actually used it for the headline numbers.
- An ablation row with bias but no distillation, or any from-scratch result.
- Any language-model result: none exists in the paper.
- CVPR 2026 acceptance comes from the repo README only.
- `losses.py` not read.

## Links worth following

- https://github.com/EdwardChasel/BinaryAttention (`models.py` Attention class, `utils.py` STESign)
- SageAttention (8-bit attention baseline it builds on): https://arxiv.org/abs/2410.02367 (ID from memory, not re-checked)
- BiViT (cited for the 300-epoch low-bit schedule); not located by me.
