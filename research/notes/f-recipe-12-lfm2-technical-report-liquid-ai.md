# LFM2 Technical Report  (https://arxiv.org/abs/2511.23404, v1 28 Nov 2025, Liquid AI Team)

Read in primary form: full arXiv HTML (Sections 1-4, 8-9, Appendix A), plus the released `LiquidAI/LFM2-350M` `config.json` and the Hugging Face `transformers` `modeling_lfm2.py` (both primary, used for details the paper omits).

## What it is
A family of on-device LLMs (350M to 2.6B dense, 8.3B MoE) whose backbone was chosen by a hardware-in-the-loop search on phone and laptop CPUs. The search kept a "minimal hybrid": most layers use a gated depthwise short convolution (kernel 3) as the sequence mixer and a minority use grouped-query attention; SSM and linear-attention operators were dropped. The report also gives a top-K distillation loss that is safe with truncated teacher logits. Operator comparisons are stated in prose only; no ablation table exists.

## Mechanism
Gated short conv block (Sec 2.2, verbatim structure; h is L x d):

    (B, C, h~) = Linear(h)        # d -> 3d, split in three
    y = B * h~                    # elementwise
    z = Conv_k(y)                 # depthwise causal 1D conv, k = 3
    o = Linear_out(C * z)         # d -> d

From the code: gates B and C are raw linear outputs (no sigmoid); no biases anywhere (`conv_bias: false`); the conv is causal (pad k-1, state = last k-1 = 2 vectors of y per layer during decode).

Layer (code): `x += Op(RMSNorm(x)); x += SwiGLU(RMSNorm(x))`, Op = conv or attention. SwiGLU: `w2(silu(w1 x) * w3 x)`. Attention: GQA, RoPE (theta 1e6), RMSNorm on q and k (QK-norm). Final RMSNorm, tied input/output embeddings, vocab 65,536 byte-level BPE.

LFM2-350M (Table 1 + config): 16 layers, d=1024, FF 4608 (4.5x), 16 heads / 8 KV groups / head size 64, attention at layer indices [2, 5, 8, 10, 12, 14]; the other 10 layers are conv. So the stack starts with two conv layers and attention gets denser toward the top. Parameters per mixer: conv = 4d^2 + 3d; GQA attention here = 3d^2. The conv block is not smaller than attention; it saves the KV cache and the per-position scoring.

Decoupled top-K distillation (Sec 3.3, App. A), teacher top-K = 32 logits stored per token, T = top-K set:

    L = KL(Bern(P_T(T)) || Bern(P_S(T)))  +  P_T(T) * tau^2 * KL(P_T(.|T)^(tau) || P_S(.|T)^(tau))

Temperature only inside the conditional term; added to hard-label cross-entropy.

## Numbers
- Decode tokens/s, llama.cpp Q4_0, batch 1 (Tables 2, 3). Galaxy S25: LFM2-350M 194.1 (1K prefix) / 143.8 (4K) vs Granite-4.0-350M 132.9 / 70.7 and Granite-4.0-H-350M 150.7 / 119.3. Ryzen HX 370: LFM2-350M 207.7 / 170.5 vs Granite-4.0-350M 225.0 / 166.2 (LFM2 is not faster here at 1K).
- Prefill tokens/s, S25: LFM2-350M 1,067 / 657 vs Granite-4.0-350M 528 / 210.
- Quality, 350M (Table 6, their harness): MMLU 43.43, IFEval 65.12, GSM8K 30.10 vs Qwen3-0.6B 44.93 / 64.24 / 36.47. Trained on 11T tokens (Qwen3: 36T).
- Pre-training: 10T tokens at 4,096 context, then 1T at 32,768; teacher LFM1-7B.
- MoE variant: 32 experts, top-4, first two layers dense "for stability".
- Operator search outcome (Sec 2.1): adding linear-attention, SSM or extra conv operators "does not improve aggregate quality ... and typically worsens device metrics". No numbers.

## Smallest scale tested, and whether the result held there
350M parameters, d=1024, 16 layers. Nothing below that; search-time training scale is only described as "large enough to surface meaningful quality differences". No evidence for 256-384 width, ternary weights, or a 256-token context.

## Training recipe and gotchas
- SFT stage (Sec 4.2.2): AdamW beta 0.9/0.95, wd 0.1, clip 1.0, cosine 3e-5 to 1e-7, 500 warm-up steps, 10% dropout on input embeddings only, 3 epochs. Pre-training optimizer settings are not given.
- tau, K-vs-quality, and the KD/CE weight are not stated.
- They report that perplexity and cache-size proxies (their earlier STAR search) did not transfer to task scores or device latency.
- Speedup depends on context length and device: large at 4K on a phone, absent at 1K on the laptop.

## Mapping onto the GBA design
Per conv layer per token at width d: ternary matrices 3d^2 (in) + d^2 (out) = 4d^2 weights; 2d elementwise 8/16-bit multiplies; 3d conv taps; RAM = 2d state values.
- d=256: 262K ternary weights (about 420K cycles at 1.6 c/weight) + 512 multiplies + 768 tap MACs (about 10K cycles). d=384: 590K weights.
- That is the same matrix cost as our attention projections, so the win is not multiplies. The win is RAM and scoring: an attention layer needs 256 x d value bytes (64 KB at d=256) plus keys per layer and a 256-position scan; a conv layer needs 0.5-1 KB. With 256 KB slow RAM, 4-6 attention layers do not fit comfortably; 2 attention + rest conv does. (Our arithmetic, not the paper's.)
- 4d^2 per mixer is heavy against a 1.3-2M weight budget. Changes to try (ours): shrink the in-projection (shared or grouped B/C/h~), or ternary conv taps.
- ROM precompute (ours): if layer 0 is a conv layer fed only by token embeddings, y = B*h~ and C are functions of the token id alone, so store y[token] and C[token] in ROM (16K x d each); layer 0 then costs 3 lookups, 3d adds, d multiplies and one d^2 out-projection. LFM2 does start with two conv layers.
- The product of two linear terms doubles dynamic range: needs a rescale shift after each gate under integer inference.
- Top-K=32 distillation targets fit our storage-limited teacher pipeline directly.

## Verdict
Test. The conv-for-most-layers layout is the best-supported way to cut attention RAM, but the evidence is qualitative and three orders of magnitude above our scale. Experiment: at d=256, 6 layers, ternary QAT, same data and steps, compare (a) all attention, (b) attention at 2 upper layers + gated conv elsewhere, (c) as (b) with the conv block removed (MLP only); measure held-out loss and a copy/recall probe over 256 tokens. Adopt the decoupled top-K loss now (low risk).

## Could not verify
- Any per-operator ablation numbers, the search's candidate count, or the training scale used in the search: not in the report.
- Why attention sits at [2, 5, 8, 10, 12, 14]: layout is only in the config, unexplained.
- tau and loss weights for distillation; pre-training optimizer and batch size.
- Whether Granite-4.0-350M is a fair like-for-like baseline (not checked).
- The HTML shows "Publication Date: August 24, 2026" although v1 is dated 28 Nov 2025; unexplained.
- No secondary sources were used.

## Links worth following
- https://huggingface.co/LiquidAI/LFM2-350M (config with `full_attn_idxs`)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py
- Canon layers (Allen-Zhu, 2025), cited in Sec 8.2 as a related finding on cheap local mixing; URL not retrieved.
- STAR (Thomas et al., 2024), the earlier proxy-based architecture search; URL not retrieved.
