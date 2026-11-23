# Scone: Scaling Embedding Layers in Language Models  (https://arxiv.org/abs/2502.01637, v1 Feb 2025, v3 Oct 2025, NeurIPS 2025; Yu, Cohen, Ghazi, Huang, Kamath, Kumar, Liu, Zhang; Google)

Tags: [P] v3 full text with appendices; [F] read off a rendered paper figure (±0.1 ppl); [O] Over-Tokenized Transformer (2501.16975) text; [M] MoLE (2503.15798) text; [E] my estimate or inference.

## What it is
Scone adds a second input-embedding table keyed on frequent token n-grams ("f-grams", n = 2..K). Each position uses the row of the longest f-gram ending there instead of its token embedding. The rows are not free parameters: in training a side transformer computes them from the f-gram's tokens, and afterwards its output for every f-gram is cached, so inference is a lookup. More rows and a bigger side transformer both lower perplexity at fixed inference compute. It is the precedent for filling a ROM n-gram table with a network that is then thrown away.

## Mechanism
[P Alg. 1, 3; App. F]
1. f-gram set: count n-grams for n = 2..K, one corpus scan per n; in scan n+1 skip candidates whose n-gram prefix or suffix has count below 5; sort all by count; keep the top S. Default K = 5. Top-S by count is suffix-closed up to ties, so lookup can extend 2, 3, .., K and stop at the first miss [E].
2. Position i: j = smallest index with (σ_j..σ_i) in the set.
   - No match: e_i = T(σ_i), the ordinary token embedding.
   - Match, training: e_i = A_fgram(T(σ_j), .., T(σ_i)), the last-position output of a causal transformer with its own absolute position embedding (length K); f-grams padded per batch.
   - Match, inference: e_i = F[(σ_j..σ_i)], the cached output.
   The row replaces the token embedding. Nothing is summed or gated.
3. Main model and head are unchanged. A_fgram, main model and T train jointly from scratch.
4. A_fgram is the baseline architecture without its embedding layer, 0.5-3× the main model's non-embedding parameters.
5. Pre-scan the corpus once and store each token's longest-match length, so training does no table lookups.

## Numbers
GPT-2 tokenizer (50,257), WebText (9B tokens), 80B tokens seen; d = 1024 (128M model) or 1536.
- Rows, K=5, A_fgram 1×, 128M model (76M non-embedding, 6 layers), WebText ppl [F Fig. 4]: baseline 24.6; 512K f-grams 23.5; 1M 23.1; 2M 22.95; 10M 22.3; 20M 22.1; 50M 21.95; 100M 21.85. That is 0.046 nats at 512K, 0.119 at 100M, about 0.03 nats per 10× rows [E]. Doubled-depth 204M baseline: 20.6. WikiText-103: 35.1 → 34.75 (512K) → 31.4 (100M); 204M: 28.95.
- 419M: 19.0 → 16.85; 589M: 18.1 → 16.15 at 100M f-grams [F], matching the 759M and 1099M baselines [P].
- A_fgram size, 128M main, 100M f-grams [F Fig. 12]: 38M / 76M / 152M / 228M → 22.4 / 21.85 / 21.3 / 21.05. 419M main, WikiText [P]: 26.1 → 23.4 (170M A_fgram) → 22.1 (1020M).
- K, S = 20M, 3 seeds, baseline star matches the 589M model [F Fig. 11]: WebText 18.1 → 16.33 (K=2), 16.10 (3), 16.03 (4), 16.02 (5), 15.97 (8); mean matched length 1.96, 2.56, 2.81, 2.90, 2.97. Bigrams alone give about 84% of the K=5 gain [E].
- OLMo, 200B tokens, mean ppl over 11 corpora [P Table 3]: 1B 16.082; +10M f-grams 15.459 (0.6B A_fgram), 15.125 (1.8B); +1B f-grams 15.187, 14.581; 1.9B baseline 14.598.
- Zero-shot mean of 6 tasks [P Table 1]: OLMo-1B 53.7; +OE-12.8M (hashed, directly trained) 54.4; +10M f-grams 54.9; +1B f-grams 57.0; OLMo-1.9B 56.8.
- Bigger vocabulary instead [P App. D]: over 100M training tokens, 97.6% of a 32K vocabulary gets more than 100 updates, 7.3% of a 2M vocabulary; BPC worsens past 512K.
- Rows per training token [E]: 100M × 1024 cached values from 80B tokens is under one token per value, far below what direct training needs.
- Training FLOPs per sequence [P Table 5]: 2.73e13 (1.9B) vs 4.94e13 and 5.57e13 (Scone).

## Smallest scale tested, and whether the result held there
128M total (76M non-embedding, d = 1024, 6 layers); smallest table 512K rows. The gain held (0.119 nats, as large as at 589M), but this is the one size where Scone did not reach the doubled baseline (21.85 vs 20.6). Nothing below d = 1024, no quantisation, no small vocabulary.

## Training recipe and gotchas
- AdamW, weight decay 0.1, peak lr 2e-4, cosine, batch 512 × 1024, bfloat16 [P F.1]. No separate lr for A_fgram.
- Training costs 1.8-2× FLOPs; the paper halves training tokens to compensate (500B vs 1T).
- Error bars only on the K sweep; on WikiText they are as large as the K = 2..5 differences [F].
- §4.1.1 says perplexity "rises" from K=2 to 4; figure and caption show it falling.
- Returns from A_fgram size flatten by 3×. MoLE, the same trick for token-id-keyed experts, saturates between 4d and 16d expert width [M].
- Hashed tables need 1-gram and 2-gram rows summed and table size coprime with V [O]; exact-match Scone has no collisions.
- I found no released code.

## Mapping onto the GBA design
All [E].
- Per token: 2-4 ROM probes (hash of previous node and token id, fingerprint compare), about 300 cycles, plus one d-byte row copy, about 1K cycles. No multiplies; no RAM beyond the last 4 token ids. Under 0.1% of budget.
- ROM is the constraint: 4-6 key bytes plus the row. d = 256: int8 260 B → 64K rows per 16 MB; ternary 55 B → 300K; rank-48 int8 rows expanded by a 48×256 ternary matrix (12K weights, about 20K cycles) 52 B → 320K. The paper's smallest table is 512K rows on open web text; our corpus is more repetitive, so coverage per row should be higher, unmeasured.
- The transferable idea: any table keyed only on recent token ids can be trained as side network, then baked. That covers this input table, per-layer rows keyed on f-grams, Engram-style block-2 rows and MoLE-style expert outputs. It removes the tokens-per-table-parameter limit in the Engram notes.
- QAT: put the embedding fake-quantiser on A_fgram's output so cached rows equal inference rows (untested). MoLE's baked tables survive NF4/NF3 post-training quantisation [M].
- A_fgram can be 10-50× our main model on a 5090; the paper tested to 3×.
- The suffix trie can also store top next tokens per node for typing suggestions.

## Verdict
Test, high priority. Inference cost is nil and 0.03 nats per 10× rows agrees with Engram and OverEncoding, but there is no evidence below d = 1024, under 512K rows, on a 16K vocabulary, or with quantised rows. (1) CPU only: match rate and mean matched length against S (64K-1M) and K on our corpus. (2) d = 256 QAT model at equal ROM bytes: no table; directly trained hashed 2/3-gram rows; Scone rows with A_fgram at 1×, 4×, 16× and K in {2, 3, 5}; injected at input, block 2, or per layer. Report validation nats and typing top-3 hit rate.

## Could not verify
- Figure values are read from rendered SVGs; the paper has no tables for Figs. 4-6, 11, 12.
- The K sweep's model size is inferred from the baseline star.
- Absent: match rates, replace versus add, a directly trained table at equal S, per-position training cost, how Qwen3-8B outputs fit a 4B model (App. E.3).
- Suffix-closure, GBA costs and the per-layer extension are mine.

## Links worth following
- https://arxiv.org/abs/2501.16975 (OverEncoding; 3-gram input encoding helped a 2.4M-parameter GPT-2 on a synthetic grammar)
- https://arxiv.org/abs/2503.15798, https://github.com/JieShibo/MoLE
- https://neurips.cc/media/neurips-2025/Slides/116760.pdf (Scone slides, not read)
