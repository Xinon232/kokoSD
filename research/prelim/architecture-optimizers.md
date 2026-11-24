## Tiny LM design research (web only)

Tags: **[P]** = read in the primary source (arXiv page or README, via a fetch tool that summarises pages); **[S]** = search snippet or secondary summary only; **[I]** = my inference.

### 1. modded-nanogpt speedrun
https://github.com/KellerJordan/modded-nanogpt [P]

Current record is #92: 0.665 min on 8xH100 to 3.28 val loss, dated 08/30/26. The recipe as listed in the README:

- **Architecture, quality per parameter or step:** RoPE, QK-norm, ReLU^2, zero-init projections, embedding skip to every block plus block 3->6 skip, value embeddings with gates, MUDD skips, "smear" one-token lookback, sparse attention gate, residual-stream decay, partial key offset, paired-head attention, learnable XSA, light dynamically-composable MHA, long-short sliding-window attention with YaRN warmup, logit soft-capping, head/embedding untied at 2/3 of training, hashed bigram/trigram embedding table (84.6M rows), reduced depth with 64-dim QK and half-width V on two layers.
- **Training objective and schedule:** multi-token prediction, prefix-token auxiliary loss, sampled softmax, batch-size and sequence-length schedules, EOS-aligned batches.
- **Optimizer:** Muon with Polar Express, NorMuon, the ANVIL twin-momentum variant, cautious weight decay tied to LR, 2-step gradient accumulation and sparse row updates for embeddings, tail EMA.
- **Pure speed, ignore:** FP8 everywhere, Flash Attention 3, CUDA graphs, Triton kernels.

Transfer to 1-10M parameters [I]:
- Adopt: RoPE, QK-norm, ReLU^2 (integer-friendly), zero-init, embedding skips, value embeddings, smear, narrow QK with half-width V (shrinks your cache directly), soft-capping, EMA.
- Especially attractive for you: hashed n-gram embeddings, which cost ROM lookups and almost no compute. Shrink the table drastically.
- Skip: window schedules, FP8, sampled softmax (your two-level head replaces it).
- The README itself says speedrun methods may not all scale; none are validated at 1-10M.

### 2. Sequence mixers at tiny scale
- **minGRU/minLSTM** https://arxiv.org/html/2410.01201v3 [P]: char-level Shakespeare, 3 layers, d=384. Test loss minGRU 1.548, minLSTM 1.555, Mamba 1.575, Transformer 1.547. The transformer needed about 2.5x more steps. This is the only LM evidence in the paper and it is a toy.
- **BabyHGRN** https://arxiv.org/html/2412.15978v1 [P]: BabyLM 10M words, macro average HGRN2 57.5 > xLSTM 57.0 > Mamba 56.7 > Transformer 56.0 > LSTM 54.8. These models are 300-360M parameters, so not tiny, and RWKV is absent.
- **60K-parameter TinyStories study** https://arxiv.org/html/2609.29397v1 [P]: gated diagonal SSM 1.516 nats/byte versus best transformer 2.143 in FP32. Under ternary the SSM loses 28% against 5-9% for transformers (SSM 1.942, transformer 2.31). Single author, untuned ternary LR.
- **xLSTM** [S]: beats Mamba, Llama and RWKV-4 on SlimPajama at large scale. **RWKV-7** https://arxiv.org/abs/2503.14456 [S]: no results under 100M found.
- **Hybrids:** BLaLM https://arxiv.org/abs/2511.05560 [P, abstract]: mLSTM mixer plus sliding-window attention consistently improves BabyLM zero-shot scores. arXiv 2610.39854 [S]: at 10.9M parameters on TinyStories, hybrid perplexity 5.95 versus transformer 6.93.

Verdict [I]: no rigorous head-to-head under 30M exists. The weight of evidence favours diagonal gated recurrence (minGRU/HGRN-style: O(d) state, no cache) plus a small amount of attention. Recurrent paths are more ternary-fragile.

### 3. Output layer
Adaptive softmax https://arxiv.org/abs/1609.04309 (full text via ar5iv) [P]:
- Text8 perplexity: full softmax 144, adaptive 147, frequency-binned HSM 166, similarity-clustered HSM 155.
- Europarl: adaptive equals full (37), HSM 43.
- A frequency short-list in the root beats putting all words in leaf classes by 5-10%. Tail clusters use d/4 projections.

Implications [I]: use a frequent-token short-list at the root next to the class logits, and form tail classes by similarity or learned clusters within frequency bands rather than pure frequency bins. Sample the class first, then the member; apply temperature and top-p to the joint. I found no 2023-2026 small-LM paper using it [S: search came up empty]; the speedrun's sampled softmax is training-only.

### 4. Weight sharing
- MobileLLM https://arxiv.org/abs/2402.14905 [P, abstract]: block-wise sharing adds 0.7-0.8% accuracy at zero size cost, but compute doubles for the shared block. It trades memory, not compute.
- Recursive-transformer work (arXiv 2512.12880) [S]: sharing "collapses layer-wise expressivity".

Confirmed wrong direction for you [I]. Go the other way: untied head and embedding, per-layer value embeddings, hashed n-gram tables. All are ROM-heavy and compute-free, and all are in the speedrun recipe [P].

### 5. Tiny-model ablations
No systematic 2025-26 sub-10M ablation of heads, MLP ratio and normalisation was found. What exists:
- 2609.29397 [P]: at 60K parameters, shape choice (22.6% spread) exceeds architecture effects, and the best shape flips from 1 layer to 3 layers with more data. Sweep shape per budget.
- MobileLLM [P] and PanGu-pi Pro https://arxiv.org/abs/2402.02791 [S]: deep-thin wins, at 100M+.
- TinyStories https://arxiv.org/abs/2305.07759 [S]: depth helps coherence, width helps knowledge.

### 6. Muon
- "When Does Muon Actually Win?" https://arxiv.org/abs/2610.21555 [P, abstract]: at 17.7M and 33.6M LMs, Muon wins all 20 equal-data test-loss pairs. On wall-clock AdamW is 3-7% better with the reference implementation, where Muon takes 56-70% of step time. Muon has higher regret with limited tuning.
- arXiv 2509.02046 [S]: matrix-optimizer speedup is about 1.4x at 0.1B and shrinks with scale.
- BLaLM [P]: Muon stabilised training and lowered perplexity.
- Ternary QAT: no controlled study. Community reports of Muon with STE working are unreviewed [S]. ACL 2025 arXiv 2502.11895 [S]: 16-bit then 1.58-bit beats ternary from scratch. 2609.29397 [P]: the ternary stage needs about 10x higher LR, otherwise it is worse than from scratch.

### Recommendation [I]
- **Shape:** 6 layers, d=128, roughly 12d^2 = 0.2M ops per layer, 1.2M in total. Alternative: 4 layers at d=160.
- **Attention:** in 2-3 layers only, 2-4 heads, 32-dim ternary keys, 64-dim 8-bit values. That is about 18 KB per attention layer at 256 tokens. Use RoPE, QK-norm, value embeddings.
- **Other layers:** minGRU-style diagonal recurrence, kept at 8-bit rather than ternary.
- **MLP:** ReLU^2, ratio 2-3.
- **Normalisation:** parameter-free RMSNorm, pre-norm.
- **Extras:** embedding skips, small hashed-bigram table, untied two-level head with a frequency short-list.
- **Optimizer:** NorMuon/Muon for matrices, Adam for embeddings and gains; FP pretraining, then a ternary QAT stage with re-tuned high LR, plus EMA.

**First ablations:**
1. Mixer: all-attention versus hybrid versus all-recurrent, at equal ops and measured after ternarisation.
2. Shape: 4x160 versus 6x128 versus 8x112, with MLP ratio 2 versus 4.
3. Muon versus AdamW through the QAT stage, with an LR sweep and FP-then-ternary versus ternary from scratch.