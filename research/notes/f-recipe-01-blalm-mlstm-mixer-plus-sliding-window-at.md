# BLaLM: Sample-Efficient Language Modeling with Linear Attention and Lightweight Enhancements  (https://arxiv.org/abs/2511.05560, v1 4 Nov 2025; Haller, Golde, Akbik; Humboldt-Universität zu Berlin; BabyLM 2025 workshop)

Tags: [P] primary text (arXiv HTML v1, whole paper and appendices A-H, read as raw text); [P2] xLSTM paper 2405.04517v2, partly read; [I] my inference or arithmetic.

## What it is
A BabyLM-2025 entry that swaps self-attention in a Qwen-style decoder for an mLSTM matrix-memory mixer, then tries add-ons one at a time: short convolutions, parallel sliding-window attention (SWA), a learned per-layer SWA weight ("DynMod"), Hedgehog feature maps [P]. It also compares Muon with AdamW [P]. Models are 250-270M parameters; almost every comparison is a single run [P]. No code found.

## Mechanism
Block: pre-RMSNorm, mixer, SwiGLU MLP, RoPE (used in attention and SWA only) [P]. mLSTM, per head [P, eq. 3-4]:
```
C_t = f_t*C_{t-1} + i_t * v_t k_t^T        # dh x dh matrix memory
n_t = f_t*n_{t-1} + i_t * k_t              # normaliser
h_t = o_t ⊙ (C_t q_t) / max(|n_t·q_t|, 1)
```
f, i are scalars per head; i is exponential, stabilised by `m_t = max(log f_t + m_{t-1}, log i_t)` [P2]. xLSTM's block puts a causal depthwise conv, kernel 4, before q and k [P2].
Add-ons [P, §4.1]:
- ShortConv: 1-D depthwise conv on the q and k projections (kernel not given).
- SWA: both mixers see the same input; `h = h_LA/2 + h_SWA/2`.
- DynMod: `h = h_LA + α*h_SWA`, α learned per layer; "Bounded" uses `tanh(α)`. α is larger in later layers (Fig. 3 caption).
- Hedgehog maps on q and k. "SWA with Memory" is never defined.
Config (Table 8) [P]: d 1024, intermediate 1536, 16 heads, vocab 15K; Transformer 26 layers / 250M, BLaLM 24 layers / 270M.

## Numbers
"Avg" is the mean of six columns: BLiMP, BLiMP-supplement, entity tracking, EWoK and two reading-time ΔR² values near 1 [I, recomputed]. Entity tracking swings 13-43 between runs and dominates it [P, Tables 9-13].

- Mixer (Tables 3, 10; AdamW; 3 LRs; single runs) [P]: Avg, 10M words: Transformer 32.27 vs BLaLM 35.96; 100M: 35.03 vs 35.42. BLiMP at LR 3/4/5e-4, 10M: 64.47/64.95/65.37 vs 63.07/66.93/66.72; 100M: 72.81/72.44/72.47 vs 73.40/74.60/74.49. The 10M Avg gap is all entity tracking (18.07 vs 40.93); the other three accuracies average 57.84 vs 57.77 [I].
- Optimizer (Tables 4, 11; 100M; 3 runs) [P]: validation perplexity AdamW 11.21 ± 0.11, Muon 7.95 ± 0.15; Avg 35.75 ± 1.74 vs 36.24 ± 1.16.
- Add-ons (Tables 6, 13; Muon, LR 4e-4, single runs) [P], perplexity / Avg:

| | 10M | 100M |
|---|---|---|
| base | 20.01 / 37.27 | 7.95 / 35.08 |
| ShortConv | 12.37 / 36.41 | 6.48 / 34.57 |
| SWA | 12.08 / 36.16 | 7.38 / 35.86 |
| SWA with Memory | 10.08 / 34.96 | 6.67 / 37.21 |
| SWA DynMod | 9.44 / 36.15 | 7.76 / 38.82 |
| DynMod Bounded | 8.58 / 34.41 | 6.84 / 36.21 |
| Hedgehog | 6.18 / 33.58 | 6.68 / 36.65 |
| Hedgehog + SWA | 7.27 / 36.25 | 6.63 / 34.20 |

  The abstract's "consistently improves zero-shot performance" fails at 10M: every SWA variant scores below the base [I]. Perplexity improves in every cell.
- Final (Tables 7, 14) [P]: zero-shot 29.54 (10M) and 36.49 (100M); BLiMP 67.0 / 74.7; GLUE 57.35 / 56.70.

## Smallest scale tested, and whether the result held there
250M parameters on 10M words [P]. Nothing small, nothing quantized. The mixer gap there is entity tracking only; the one repeated pattern is +1 to +2 BLiMP at 100M [I].

## Training recipe and gotchas
- Context 512, batch 64, 10 epochs, cosine with 10% warmup; best LR 7e-4 (10M), 5.5e-4 (100M); Muon on matrices, AdamW on embeddings, biases and norms [P].
- Checkpoints are chosen by the reported zero-shot average [P]: selection bias.
- Table errors [I]: Table 12's 1e-4 and 2e-4 Avg values are swapped; perplexity 16.41 appears at three LRs in Table 5; base 100M perplexity is 8.06 in Table 5, 7.95 in Table 6. The 10M base perplexity (20.01) is an outlier against neighbouring LRs (about 16), inflating add-on gains.
- The final 100M model uses Bounded DynMod (36.21), not the best variant (38.82) [P].

## Mapping onto the GBA design
All [I], d = 256.
- Our ring attention is already the "SWA" half; the paper's idea adds a matrix memory beside it for older context.
- Projections q, k, v, o: up to 4d² = 262K ternary weights, about 420K cycles; no saving over attention.
- State: d*dh entries. dh = 16: 8 KB per layer at 16 bits; update plus readout = 2*d*dh multiplies = 61K cycles. dh = 64: 32 KB, 246K cycles. Our top-8 mix costs about 2K adds, so mLSTM costs more compute but less RAM than a ring of 8-bit values (64 KB per layer).
- Decay: f is scalar per head, so keep a per-head exponent (the m_t trick) and renormalise by shifts. The divisor is one clamped scalar per head: a reciprocal table lookup.
- DynMod's α folds into the output scale in ROM: free. ShortConv with kernel 4 costs 4d multiplies (about 8K cycles) and 3d bytes per layer.

## Verdict
**Test, low priority; do not rely on its mixer claims.** Worth taking: Muon for matrices (perplexity 11.21 to 7.95, three runs) and a short causal conv before q/k. Settling experiment: at our size with ternary QAT, compare ring attention alone, ring attention plus a dh = 16 matrix memory with learned α, and both with ShortConv, at equal GBA cycles; held-out loss, BLiMP and a recall probe beyond 256 tokens, three seeds.

## Could not verify
- SWA window size, ShortConv kernel, "SWA with Memory", α values (Figure 3 is an image), Muon settings, tokenizer type, code.
- Whether BLaLM keeps xLSTM's block (conv, projection factor 2) or only the cell. The 1/sqrt(d) key scaling is from memory, not re-read.

## Links worth following
- xLSTM: https://arxiv.org/abs/2405.04517
- Hedgehog: Zhang et al. 2024, "The Hedgehog & the Porcupine" (cited, not read)
- BabyLM 2025 pipeline: https://github.com/babylm/evaluation-pipeline-2025
