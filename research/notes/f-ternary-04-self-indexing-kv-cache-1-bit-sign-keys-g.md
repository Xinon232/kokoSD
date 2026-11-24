# Self-Indexing KVCache: Predicting Sparse Attention from Compressed Keys

https://arxiv.org/abs/2603.14224 (v1, 15 Mar 2026). Xu Yang, Jiapeng Zhang, Dongyang Zhao, Guo Chen, Zhuo Tang (Hunan University). Code: github.com/LfieLike/selfindexingkv (not read).

Read in primary form (arXiv HTML), through a fetch tool that condenses the page. Equation numbers and table values below are from that reading; I did not see the raw per-dataset columns.

## What it is

A training-free KV-cache compression scheme for long-context LLM inference. Each attention key is stored as its sign bits (1 bit per channel) plus a 2-bit magnitude; the sign bits double as a retrieval index. For each query, the model scores all cached keys with table lookups over 4-bit sign codes, picks the top-k, and runs attention only over those. It is a post-hoc method applied to pretrained 8B and 14B models; nothing is trained.

## Mechanism

Per head, keys K in R^(L x D), D = 128 for Llama-3.1-8B.

1. Centre (Eq. 5): K' = K - mu, mu_d = mean over tokens of K[:, d]. Softmax is shift-invariant (q.mu is a per-query constant), so the output is unchanged; the purpose is to balance signs so each bit carries about 1 bit of entropy (Eq. 6).
2. Group (Eq. 1): split D into G = D/4 subvectors of 4 dims (G = 32 at D = 128; my derivation).
3. Code (Eqs. 2-3): code = 4 sign bits packed into 0..15.
4. Codebook (Eq. 4): for each group, centroid c_j = mean of the subvectors whose code is j. One pass, no k-means. 16 centroids x 4 dims per group.
5. Score (Eq. 8), per query:

```
for g in 0..G-1:
    for j in 0..15: T[g][j] = dot(q[4g:4g+4], c[g][j])     # 64 MACs per group
for each cached token i:
    score[i] = sum_g T[g][code[i][g]]                      # G lookup+adds
select top-k by score; attend over those plus the sink tokens
```

6. Magnitudes (Eqs. 9-13): |K'| scaled per channel by alpha_j = max|K[:, j]|, then asymmetric min/max quantised to 2 bits, with scale and zero point stored per token (groups of 32 elements, 16-bit each) so single tokens can be fetched at random. Values: 2 bits, same quantiser. The stored sign bit is reused as the sign of the dequantised key.
7. Sink tokens: 64 tokens chosen by SnapKV at prefill, kept in full precision, always attended.

Storage per token per head (Llama-3.1-8B): 128 sign bits + 512 bits of 2-bit K and V + 256 bits of scale/zero-point = about 768 bits, against 4096 bits at 16-bit.

## Numbers

All at 8B / 14B scale, long contexts.

- Table 1, LongBench average, budget 160 tokens (64 sinks + 96 selected). Llama-3.1-8B: full 58.7; SnapKV 49.9; Quest 56.5; DoubleSparse 54.8; ours with 16-bit K/V 58.4; ours with 2-bit K/V 58.2. Qwen2.5-14B: full 56.9; ours 16-bit 55.9; ours 2-bit 55.7.
- Table 2, RULER at 32K, 7.5% of tokens kept. Llama: full 90.8; Quest 84.5; ours 16-bit 89.4; ours 2-bit 89.2. Qwen: full 95.1; ours 95.0 / 93.9.
- Table 5, ablation on four LongBench sets (MF-en / HotpotQA / GovReport / RB-P): full method 54.6 / 55.7 / 33.1 / 54.4; sign-only retrieval (no centroid magnitudes) 52.8 / 53.2 / 31.9 / 52.1; without sink tokens 52.6 / 55.2 / 32.5 / 54.2; without sign in quantisation 52.5 / 54.9 / 32.2 / 50.9.
- Table 4 (16K tokens, batch 10): retrieval 0.039 ms against 0.166 ms for full q.K^T; sparse attention 0.116 ms against 0.776 ms full (the "6.7x"); one-pass clustering 88.85 ms against 2113.9 ms for 20-iteration k-means.
- Table 3: time to second token at 32K is 6.51 s against 6.18 s for FlashAttention2 (about 5% prefill overhead).
- Memory: "up to 78%" saving in one section, "nearly 5x" in another; the paper does not reconcile them (768/4096 bits gives 81%, or 5.3x, before fixed overheads).

## Smallest scale tested, and whether the result held there

Smallest model is Llama-3.1-8B; shortest context regime is LongBench (thousands of tokens). Nothing is tested on small models, short contexts or small head dimensions. The result is therefore an existence proof for the scoring structure, not evidence about a 256-wide model with a 256-token window.

## Training recipe and gotchas

- No training. The model was trained with full-precision attention and is only approximated at inference. Our design trains with sign keys from the start, which this paper says nothing about.
- Top-k selection uses the approximate score, but the final attention over the selected tokens uses dequantised keys (sign x 2-bit magnitude), not the lookup score.
- Centroid magnitudes matter: sign-only retrieval loses 1.2-2.5 points (Table 5).
- Always-kept sink tokens matter on some tasks (up to 2.0 points).
- Centring is essential to the 1-bit code being informative; the paper does not say how mu, alpha or the codebook are updated for tokens generated after prefill.

## Mapping onto the GBA design

The scoring loop is the same shape as our ternary kernel (a small run-time table per 4 inputs, then one lookup+add per stored nibble), so it reuses the existing inner loop.

Assume key dim 64 per head, 256-token ring:
- Key store: 64 bits x 256 = 2 KB per head per layer in RAM. With 4 layers x 1 head that is 8 KB, which fits slow RAM; fast RAM only if the ring is scored one layer at a time.
- Table build: 16 groups x 16 entries x 4 MACs = 1024 MACs, about 7.7K cycles. With pure +/-1 centroids it is adds only (about 256).
- Scoring: 256 tokens x 16 lookup+adds = 4096, about 26K cycles at 6.4 cycles per lookup+add. Pairing two nibbles into a byte with 256-entry tables (8 tables x 256 x 2 bytes = 4 KB) halves this to about 13K.
- Total about 20-35K cycles per head per layer, under 1% of the 3-5M budget. Top-8 selection is one pass with a small insertion list.

Changes needed:
- Because we train with this attention, mu and the centroid codebook can be fixed constants in ROM (16 x 4 bytes per group per head), computed over training data, instead of per-prompt statistics. Simpler still: learn a per-group 16 x 4 table directly ("asymmetric" scoring: full-precision query side, 1-bit key side).
- Use the lookup score itself as the attention logit for the top-8 softmax; skip the 2-bit magnitude store.
- The sink-token idea maps to always-attended slots (BOS, the last 1-2 tokens, the start of the user turn) outside the top-8.

## Verdict

Test (leaning adopt). It confirms that 1-bit keys with a real-valued query side and per-group magnitude tables are enough to pick the right tokens in large models, and the cost on the GBA is negligible. It does not show that a tiny model trained this way works. Settling experiment: train the 256-wide model with (a) full-precision keys with top-8, (b) sign keys with +/-1 scoring, (c) sign keys with learned 16-entry group tables, and compare validation loss; also measure overlap of the top-8 under (b)/(c) against exact scores.

## Could not verify

- Whether the codebook is per head / per layer / per prompt (not stated), and handling of decode-time tokens.
- Exact definition of "sign-only retrieval" (+/-1 or something else) and of "w/o sign in quantisation".
- Figure 4 values at sparsity ratios other than 7.5%; no top-k recall figures are reported.
- Whether the query is quantised (appears to be full precision).
- The repository was not opened. Per-dataset columns of Tables 1-2 were not transcribed.
- All GBA cycle figures are my estimates from the project's measured costs, not from the paper.

## Links worth following

- "KV cache is 1 bit per channel" (Zhang et al., 2024) and QJL (Zandieh et al., 2024): cited 1-bit key methods; no URLs taken from the paper.
- H1B-KV, https://arxiv.org/abs/2510.05529 (from the prelim report): 1-bit keys with fine-tuning, closer to our trained setting.
- PQCache and ClusterKV: product-quantised / clustered key retrieval baselines cited by the paper.
- github.com/LfieLike/selfindexingkv: LUT-GEMV kernel source.
