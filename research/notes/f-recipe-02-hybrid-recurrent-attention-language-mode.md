# Attention, Dynamic Wave and Hybrid Sequence Mixers in Language Models

https://arxiv.org/abs/2610.39854 (v1, 29 Oct 2026). Dmytro Babarytskyi, Oleksandr Babarytskyi.
Read in primary form (arXiv HTML, via a fetch tool that summarises the page; numbers below are from that reading, equations transcribed by the tool, figures not seen). Tags: [P] primary, [I] my inference.

## What it is
A small empirical paper comparing three 4-block, 10.94M-parameter language models on a TinyStories subset: all attention, all "Dynamic Wave" (a diagonal, input-gated linear recurrence with a cosine-modulated input), and a hybrid with two Wave blocks followed by two attention blocks. The hybrid has the best held-out perplexity (5.95 vs 6.93 transformer, 7.26 pure Wave). Most of the paper is about GPU memory and latency, and its own conclusion is that the efficiency case for the recurrence vanishes against fused attention, so hybridisation should be justified on quality only. Evidence is one seed, one epoch, one scale.

## Mechanism
[P] Per block (pre-norm): `u = x + Mixer(LN1(x)); x' = u + W2 GELU(W1 LN2(u))`.
Wave mixer, input x_t in R^d (after LN), state h_t in R^d, h_0 = 0:
```
alpha_t = softplus(x_t W_a + b_a)      # per-channel forget rate, > 0
omega_t = x_t W_w + b_w                # "frequency"
phi_t   = x_t W_p + b_p                # phase
h_t = exp(-alpha_t) * h_{t-1} + cos(omega_t + phi_t) * x_t     (elementwise)
y_t = W_o h_t + b_o
```
- Parameters: 4d^2 + 4d per mixer (three d x d gate projections, one output map).
- Training-time parallel form: prefix scan over pairs (a,b) with (a_i,b_i)∙(a_j,b_j) = (a_i a_j, a_j b_i + b_j).
- Config: d = 384, FFN 1536, 4 blocks, 6 heads (attention blocks), tied embedding/head, BPE vocab 10,000. Hybrid order: Wave, Wave, Attention, Attention.
- [I] omega and phi only ever appear summed, so they are one projection in effect; the "wave" is just an input gate in [-1, 1] that is not tied to the forget gate. Nothing in the recurrence depends on h_{t-1} except the decay, so it is in the minGRU / HGRN / LRU family (the paper cites LRU, Mamba, RWKV, Griffin, Jamba; not minGRU or HGRN).

## Numbers
All at 10.94M parameters, TinyStories 500K-story subset, 2,000 held-out 512-token chunks. [P]
- Table IV (full held-out, 63 batches): CE / PPL: Transformer 1.9357 / 6.93; Pure Wave 1.9818 / 7.26; Hybrid 1.7831 / 5.95. Hybrid is 0.153 nats (about 14% PPL) better than the transformer; pure Wave is 0.046 nats worse.
- Table III (quick, 50 batches): PPL 6.75 / 7.05 / 5.79 (same ranking).
- Table III latency at 512 tokens, batch 16, T4: reference 72.6 / 180.2 / 114.2 ms; after hoisting gate computation out of the loop: 63.3 / 71.2 / 67.3 ms (Wave 2.53x faster).
- Table V peak VRAM at 4096 tokens, batch 8: unfused attention 6601.6 MB vs Wave 1404.3 MB; fused attention 1403.6 MB (no saving). Hybrid tracks the transformer.
- Table VIII generation without state caching: 526 / 49.6 / 80.3 tok/s (bounded) — an artefact of re-running the recurrence from token 0.
- Table IX training wall-clock, one epoch: 18.4 / 328.3 / 175.0 min (transformer was torch.compiled, others not).
- Fig. 4 text: channel-mean alpha ranges 0.53-1.49 on one sentence, 2.19 at the final period. [I] exp(-alpha) is then about 0.22-0.59, so the state forgets within a few tokens on average: the Wave layers act as a short-range local mixer, with attention doing long range.

## Smallest scale tested, and whether the result held there
Only one scale: 10.94M parameters, d = 384, 4 blocks, about 106M training tokens ([I]: 6,472 steps x 32 x 512). This happens to be almost exactly Koko's candidate width and depth. The hybrid win held there, but with a single seed and one epoch.

## Training recipe and gotchas
[P] AdamW, weight decay 0.01, one-cycle cosine, peak LR 5e-4, 500 warm-up steps, batch 32 x 512 tokens, 6,472 steps (one epoch), FP16 autocast, grad clip 1.0, one Tesla T4.
Gotchas:
- No seeds, no error bars, no ordering ablation (attention-first or interleaved untested), no matched comparison to Mamba/RWKV/minGRU.
- Positional encoding is not described for any model. [I] If the attention baseline had weak or no positional signal, part of the hybrid's gain may simply be the recurrence supplying position; this is the main threat to the headline number.
- Initialisation of the gates not described; no stability or NaN discussion.
- The sequential loop is slow in eager PyTorch (18x the transformer's epoch time); hoisting and a scan implementation are needed to train at our scale.
- Everything is full precision; nothing is said about quantisation.

## Mapping onto the GBA design
[I] throughout.
- Per-token compute as published: 4d^2 ternary weights per Wave mixer = 590K at d = 384, about 0.94M cycles at 1.6 cycles/weight; the same order as an attention block's Q/K/V/O projections. So a Wave block is not cheaper in matrix work unless the gates are slimmed.
- Slimming: merge omega and phi into one projection (exact, saves d^2); make the forget and input gates low-rank, block-diagonal, or partly a ROM lookup keyed on token id / n-gram hash (nearly free) plus a small projection. A plausible target is about 1.2-2 d^2 per mixer.
- Elementwise step: exp(-softplus(z)) = sigmoid(-z) and cos(z) are each one 256-entry ROM LUT on an 8-bit pre-activation; then 2 multiplies per channel (a*h, g*x) = 768 multiplies, about 6K cycles at 7.5 cycles each. Negligible.
- RAM: state is d 16-bit values = 768 bytes per Wave layer, against roughly 256 x 384 = 98 KB of 8-bit values (plus keys) for one attention ring buffer. Replacing the lower half of the attention layers with recurrence roughly halves the context RAM, which matters against 256 KB, and removes the 256-position scoring and top-8 selection for those layers.
- No KV cache, O(1) per token, trivial state save/restore for predictive typing (snapshot 768 B per layer, run the user's-side continuation, restore).
- Risk: a multiplicative recurrence accumulates quantisation error. The strong measured decay (a about 0.2-0.6) limits the horizon, which helps, but a must be kept at 8+ fractional bits and h at 16 bits; the prelim notes already flag recurrent paths as more ternary-fragile.

## Verdict
Test. The quality claim is exactly at our scale and the RAM argument is strong for this hardware, but the evidence is one unseeded run with an unspecified positional scheme and no ternary result. Settling experiment: at d = 384, 4-6 layers, same data and tokens, 3 seeds, train (a) all-attention with our top-8 ring buffer, (b) recurrent-lower / attention-upper, (c) interleaved, each with an explicit positional scheme for attention, full precision and then ternary QAT with 8-bit gate LUTs; compare held-out loss at equal per-token cycle count. Adopt if (b) or (c) is within noise or better after ternarisation.

## Could not verify
- No code or data repository is given in the paper; none found.
- Positional encoding, gate initialisation, attention parameter count and why totals match exactly ("parameter-matched by construction") are not stated.
- Figures were not viewed; Fig. 3/4 values come from the paper's text. Equations were relayed by a summarising fetch tool rather than copied by eye from the page.
- Tokens seen (106M) and all cycle/RAM costs are my derivations.

## Links worth following
- Griffin (recurrence + local attention), cited as the placement principle: https://arxiv.org/abs/2402.19427
- BLaLM, mLSTM + sliding-window attention on BabyLM (from prelim notes): https://arxiv.org/abs/2511.05560
- "Were RNNs All We Needed?" (minGRU), the simplest gate form to compare against: https://arxiv.org/abs/2410.01201
