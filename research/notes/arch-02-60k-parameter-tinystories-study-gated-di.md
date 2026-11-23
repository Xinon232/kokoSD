# Baseline Shape Decides the Verdict: A Controlled Re-Examination of Ternary Language Models at 60K Parameters  (https://arxiv.org/abs/2609.29397, v1 24 Sep 2026, Gautam Veldanda, independent)

Tags: [P] read in the primary text (arXiv HTML v1, full paper incl. appendix); [S] secondary; [I] my inference or arithmetic.

## What it is
A 98-run, 3-seed, laptop-scale audit of Atome-LM's claim that a routed ternary block (conv + diagonal SSM + top-4 attention, mixed by a router) beats an FP32 transformer by 22% at 60K parameters [P]. Models are byte-level (vocab 256, context 256) on TinyStories V2, FP32 and ternary, at 16M and 130M training bytes [P]. Findings: transformer shape alone moves loss 22.6%; a plain gated diagonal-SSM block beats the routed block and every transformer; the ternary penalty looks larger for recurrent models (confounded); FP-then-ternary wins only with a 10x higher stage-2 LR [P].

## Mechanism
- Atome SSM pathway [P]: `h_t = a*h_{t-1} + b*x_t; y_t = c*h_t`, elementwise, `a = tanh(a_raw)`; a, b, c are per-channel FP32 vectors.
- Gated SSM ("ours") [P]: "a single recurrent pathway with a GELU feed-forward layer and no router"; d=44, 3 layers, d_ff=2d, 64,196 params, 1.6% kept FP (SSM scalars, LayerNorms). The paper gives no gate equation.
- Reconstructed by parameter arithmetic [I]: 64,196 = 2x256x44 (untied embed/head) + 3 layers x (3d^2 + 4d^2 + 3d + 2 LayerNorms) + final LayerNorm, exactly, no biases. So each layer has three d x d ternary matrices around the recurrence (plausibly input, gate, output) plus a d->2d->d FFN. The same accounting reproduces the routed (60,800) and 3x32 (61,888) counts.
- Quantizer [P]: `W~ = alpha*clip(round(W/alpha),-1,1)`, `alpha = mean|W|` per tensor, recomputed every forward, STE. Embedding, head and all matrices ternary.
- Transformers [P]: learned positional embeddings kept FP32 (11-22% of params), 4 heads, d_ff=4d (2d for 1x52).
- Chunked-scan trainer [P, App. A]: chunks of T=32; in-chunk lower-triangular `M_ts = a^(t-s)`; carry `H_n = a^T H_{n-1} + h_end`. Exact to 2e-6, 19x faster on CPU.

## Numbers
Validation nats/byte, mean +/- sd over 3 seeds, 60K params (Table 4) [P]:

| System | 16M FP | 16M tern | 130M FP | 130M tern | penalty @130M |
|---|---|---|---|---|---|
| Gated SSM 3x44 | 2.704+/-.041 | 3.376+/-.093 | 1.516+/-.012 | 1.942+/-.018 | +28.1% |
| Routed 4x64 | 2.808 | 3.290 | 1.668 | 1.993 | +19.5% |
| Transformer 1x52 | 2.835 | 3.426 | 2.195 | 2.311 | +5.3% |
| Transformer 2x36 | 3.080 | 3.752 | 2.169 | 2.320 | +7.0% |
| Transformer 3x32 | 3.239 | 3.870 | 2.143 | 2.328 | +8.6% |
| Transformer 4x28 / 6x24 | 3.411 / 3.477 | 4.103 / 4.162 | not run | not run | |

- Shape span at 16M: 22.6%; ordering reverses at 130M (Sec. 6.1) [P].
- Gated SSM beats routed by 9.1% in FP (2 sigma 3.2%); tie in ternary (-2.6%, inside 3.7%) (Sec. 6.2) [P].
- [I] Ternary SSM (1.942) is 16% below the best ternary transformer (2.311) and 9% below the best FP32 transformer (2.143).
- Transformer penalty shrinks with budget (+20.9% -> +5.3%); recurrent penalties do not (+24.9% -> +28.1%, +17.2% -> +19.5%) (Sec. 6.3) [P].
- FP-init 90/10 at 130M (Table 5) [P]: stage-2 LR sweep 3e-5..3e-3, routed 2.758 / 2.335 / 2.110 / 1.944 / 1.911; transformer 2.701 / 2.474 / 2.348 / 2.298 / 2.265. Three seeds: routed 1.900 (-4.7% vs scratch 1.993) at 3e-3 but +15.3% at 1e-4; transformer 1x52 2.262 (-2.1%) at 3e-3 but +7.1% at 1e-4. Sweep not saturated.
- Router weights (Sec. 6.5) [P]: SSM pathway 0.49-0.73, conv 0.09-0.25, attention 0.19-0.28.

## Smallest scale tested, and whether the result held there
60K parameters is the only size [P]. At 16M bytes (245 steps) the SSM leads the best transformer by 4.6% in FP and 1.5% in ternary [I]; the big gaps appear only at 130M bytes (1,991 steps). No subword vocab, nothing above 60K; the author says the percentages "should not be carried over" [P].

## Training recipe and gotchas
- AdamW (0.9, 0.95), wd 0.1 on all params, peak LR 3e-4 cosine to 3e-5, warmup 3.33%, clip 1.0, seq 256, batch 64x4 [P].
- From-scratch ternary reused the FP LR, so every penalty may be inflated [P].
- FP->ternary: sweep stage-2 LR upward from 10x the pretrain peak; "fine-tuning" LRs reverse the verdict [P]. No FP-init arm was run for the gated SSM [P].
- Sweep depth/width per data budget; quantize baselines equally (use RoPE) [P].
- Byte-level input favours recurrence [P]. Atome's C engine had an SSM state-carry divergence bug and now re-derives state from the full prefix each call [P, Atome HONEST_RESULTS.md]; if inference streams, train with carried state [I].

## Mapping onto the GBA design
[I] throughout. Diagonal update per layer: 3d fixed-point multiplies (d=256: ~5.8K cycles), or none if `a` is restricted to 1-2^-k shifts (untested). State RAM: d x 16-32 bit = 0.5-1 KB per layer, versus about 18 KB for an attention layer's 256-token cache. ROM: 3d bytes of 8-16 bit a, b, c (the paper keeps them FP). The cost is the three d x d matrices: 3d^2 = 197K ternary weights = ~315K cycles per layer at d=256, no cheaper than attention projections; the saving is the key scan and the RAM. Layer 1's input projection can be a ROM table keyed on token id. State needs >=16 bits when `a` is near 1.

## Verdict
**Test.** Strong signal that an O(d)-state gated recurrence is the best mixer at tiny scale, even in ternary; but one size, byte-level, narrow dense-attention baselines, single author, not peer-reviewed. **Adopt** the two process rules now (shape sweep per budget; high-LR ternary stage). Settling experiment: at 2-5M params with the 16K vocab, swap attention for gated diagonal recurrence in 0, half, all layers at equal GBA cycles, RoPE baseline, per-arm ternary LR sweep, compare after QAT.

## Could not verify
- Code: github.com/veldanda/ByteLM was unreachable (404 / no access); Zenodo holds only the PDF. The gate equation and layer wiring are unverified; the 3d^2 structure is inferred.
- No MCU latency, RAM or integer-activation results here. Atome's own figures (944K model at about 1 token/s on ESP32 [S, README via tool summary]; 411 KB peak RAM [P, HONEST_RESULTS.md]) are not from this paper.

## Links worth following
- Atome-LM: https://github.com/TilelliLab/atome-lm (C99 engine, base-3 packing)
- ParetoQ: https://arxiv.org/abs/2502.02631
- Thomassen, sub-100M QAT factorial: https://arxiv.org/abs/2605.25966
- Ternary Mamba (W1.58A16): https://arxiv.org/abs/2606.18114
