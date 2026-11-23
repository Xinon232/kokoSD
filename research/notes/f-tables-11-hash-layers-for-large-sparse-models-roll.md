# Hash Layers For Large Sparse Models  (https://arxiv.org/abs/2106.04426, NeurIPS 2021; Roller, Sukhbaatar, Szlam, Weston, Facebook AI Research)

Read in primary form: arXiv HTML of v3, full text and Appendices A-C. Figures were not readable as numbers.

## What it is
A mixture-of-experts FFN whose expert is chosen by a fixed, precomputed lookup table on the current input token id, not by a learned router on the hidden state. No routing parameters, no load-balancing loss, no assignment algorithm. At 222M-4.5B parameters it matches or beats Switch Transformer and BASE Layers at equal compute. The paper ablates what to hash on and proposes MultiHash, which assembles one FFN from several independently hashed slices.

## Mechanism
- Eq. 4: `h_t = FFN_{hash(x_t)}(h_bar_t)`, `h_bar_t` = attention output, `x_t` = raw token id at position t. The FFN still consumes the hidden state; only selection uses the token.
- `hash`: table `vocab -> {0..K-1}`, fixed before training.
  - Random: random expert per token.
  - Balanced: tokens in descending training frequency, each greedily assigned to the emptiest bucket. Imperfect, since some tokens exceed 1/K of the mass (Fig. 4).
  - Clustered: k-means (K centroids) on a baseline model's token embeddings. Dispersed: each cluster's tokens spread evenly over all buckets.
  - Also: bigram `(x_{t-1}, x_t)`, previous token, position, oracle and predicted `x_{t+1}`.
- Several sparse layers: a different table per layer.
- MultiHash: N hashes `k_m = hash_m(x)`. Split `A: R^d -> R^D` into row-blocks `A_m: R^d -> R^{D/N}`, and `B` into `B_m: R^D -> R^{d/N}`. `v = relu([A_{k_1}(h),...,A_{k_N}(h)])`, `out = [B_{k_1}(v),...,B_{k_N}(v)]`. Same parameters and compute; K^N composite experts.

## Numbers
Pushshift Reddit, 8008-token BPE, 2 encoder + 11 decoder layers, d=1024, D=4096, 100k updates; valid PPL (Table 1 unless stated):
- Dense 222M: 24.90. Wider dense 755M: 23.32. Deeper dense (22 layers) 755M: 22.72.
- 1x64 (one sparse layer, decoder layer 7), 751M: Hash 23.16, Switch 23.65.
- 1x128, 1.28B: Hash 22.89, Switch 23.52.
- 1x32, 483M: Hash 23.58, Switch 23.79. 1x16, 348M: Hash 24.01, Switch 24.00 (Table 7).
- 5x16 (every other layer), 852M: Hash 23.21, Switch 23.00.
- Hash choice at 1x64 (Table 3): balanced 23.16, random 23.22, clustered 23.90, dispersed 23.17, position 25.07, bigram 24.19, previous token 24.16, predicted next token 25.02, oracle next token 1.97.
- 5x16, same table in every layer 23.74 vs different tables 23.21 (Table 3).
- MultiHash 1x32, 1/2/4/8 hashes: 23.58 / 23.48 / 23.38 / 23.28. 1x16: 24.01 / 23.88 / 23.73 / 23.83 (Table 7).
- Learned router fed a token embedding ("Token Switch"): 23.43 at 1x64 (Table 6). Switch with no balancing loss: 24.80 (Table 9).
- RoBERTa+cc100en, 51200-token vocab (Table 2): dense 266M 28.85, Switch 27.41, Hash 26.99.
- 4.5B LM vs BASE (Fig. 2, curves only): Hash better at every step, about 11% more updates/s.
- Fig. 3: later layers better; the Hash advantage over Switch grows with K.

## Smallest scale tested, and whether the result held there
Wikitext-103, 8 layers, d=512, D=512, 1x16 (Table 4; parameter count not given). 8008-token vocab: dense 12.58, Switch 11.67, Hash 11.58 - held. 267,739-word vocab: dense 33.09, Switch 31.76, Hash 32.32 - beats dense, loses to Switch. Nothing below d=512 was tested.

## Training recipe and gotchas
- Appendix B.1: Adam, max LR 0.002, 10k warmup, inverse-sqrt decay, batch 40, context 128, clip 1.0; 8 V100s for about 20 hours.
- Do not put similar tokens in one expert (23.90 vs 23.22); fine distinctions are needed within a cluster.
- Do not reuse one table across layers (0.5 PPL worse).
- Routing on older or wider context is worse at fixed K.
- Sparse capacity is weak per parameter: 3.4x parameters bought about 7% PPL at 1x64, and an equal-size deeper dense model was better (at more compute).
- My inference, not in the paper: each expert sees about 1/K of tokens, so ternary QAT experts get sparse gradients; balanced tables and longer schedules matter more for us.

## Mapping onto the GBA design
- Routing: one ROM byte read per layer per token; tables 16K x 6 layers = 96 KB ROM. No multiplies, no RAM. A learned 64 x 256 ternary router (about 16K weights, roughly 26K cycles per layer at 1.6 cycles/weight) disappears.
- The expert id is known before attention runs, so ROM pointers can be set up early.
- MultiHash fits the ternary partial-sum kernel unchanged: tables are built from the input once, and each hidden-row block reads its bytes from a different expert's ROM region. Extra cost is N byte lookups; ROM is unchanged. N=4, K=64 gives 16.7M composite experts.
- To do: build balanced tables from our unigram counts, one per layer.
- Our 16K vocabulary lies between the two sizes where Hash won (8K, 51K).
- Not tested in the paper but needed by us: every layer sparse (they did at most 5 of 11), width 256-384, ternary weights.

## Verdict
Adopt as the default router; test MultiHash (N=2-4). It removes routing compute and a tuning knob, with direct evidence of being no worse than learned routing at K>=32. Settling experiment at our real shape (d=256, 6 layers, K=64, ternary QAT), equal steps: dense; balanced token hash with per-layer tables; MultiHash N=4; Switch with load_bal 0.1; and a product of current-token and previous-token buckets.

## Could not verify
- Exact values behind Fig. 2 and Fig. 3.
- A public code release: one search found none; the paper names ParlAI and the fairseq BASE code base without a repository link.
- Behaviour under 100M parameters, with all layers sparse, or with quantized weights: absent from the paper.
- "Wider" is printed as d=2048, D=6144 in Table 1 while the text gives only D=6144; copied as printed.
- Nothing here comes from a secondary source.

## Links worth following
- BASE Layers (Lewis et al., 2021), the paper's reference [10].
- Switch Transformers (Fedus et al., 2021), reference [8].
- Large Memory Layers with Product Keys (Lample et al., 2019); already covered in capacity-02.

## Independent verification
Second reader, 2026-11-18. Checked against the primary source only (ar5iv HTML rendering of arXiv 2106.04426: Tables 1-7 and 9, sections 5.1-5.3). Figures 2-4 not readable as numbers. No code repository found or checked.

1. Hash beats Switch at 1x64 (23.16 vs 23.65, dense 24.90; 26.99 vs 27.41 on RoBERTa+cc100en) - CONFIRMED, with one scale detail. Table 1 rows 4-5 give 751M, valid 23.65 / 23.16 (test 23.73 / 23.23), baseline 222M at 24.90; d=1024, D=4096; 8008-token BPE (sec. 5.1). Table 2 gives 27.41 / 26.99 with a 51200-token GPT-2 vocabulary, but those models are 795M / 794M with a 266M dense baseline (28.85), not 751M, and Table 2 reports valid PPL only. The Switch figure is the best of five load-balance settings (Table 9), so the comparison is fair to Switch. Not stated by the first reader: a deeper dense model with the same 755M parameters scores 22.72, better than both sparse models (at more compute).
2. Balanced ~ random; clustered, previous-token, bigram, position worse - CONFIRMED. Table 3: balanced 23.16, random 23.22, clustering 23.90, dispersed 23.17, position 25.07, bigrams 24.19, previous token 24.16. Two qualifications: (a) "equivalent" is the paper's "perform similarly well" on a 0.06 PPL gap; no seeds, repeats or error bars are reported anywhere in the paper, so gaps under about 0.1 are not distinguishable from noise on the evidence given; (b) position routing (25.07) is worse than the dense baseline (24.90), i.e. it is harmful, not just less good. Table 3 itself lists no parameter counts; 751M is carried over from Table 1. The paper does not say how the bigram is mapped to the 64 experts in the passage read, so "bigram is worse" is not evidence against every two-token routing scheme.
3. Per-layer tables needed (23.21 vs 23.74 at 5x16); Switch slightly better there (23.00) - CONFIRMED. Table 3 last two rows: same hash per layer 23.74, different hash per layer 23.21. Table 1 rows 8-10: Switch 5x16 load_bal 0.1 = 23.00, load_bal 0.01 = 23.19, Hash 5x16 = 23.21, all 852M. So Switch wins or ties depending on its balancing coefficient. The same-table result is a single run at one configuration; "needs" is the reader's wording, the paper only reports the two numbers.
4. MultiHash monotone at 32 experts (23.58 / 23.48 / 23.38 / 23.28, 483M) - CONFIRMED (Table 5 in the main text and Table 7 in Appendix A; test 23.65 / 23.53 / 23.45 / 23.34). Caveat already in the notes above but not in the claim: at 16 experts (348M) the trend is not monotone, 24.01 / 23.88 / 23.73 / 23.83, so 8 hashes was worse than 4. The main text nevertheless says "steadily improving results for both 16 or 32 modules", which its own table does not fully support. Total gain at 32 experts is 0.30 PPL (about 1.3%), single runs.

Trust for our scale: the numbers are transcribed correctly, but every result comes from d=1024 encoder-decoder models of 222M-1.28B full-precision parameters, where each token touches roughly 200M weights - about two orders of magnitude more than our 1-2M ternary weights - and the sparse layer is usually one decoder layer out of 11. Missed or under-weighted caveats:
- All differences are 0.06-0.6 PPL (0.3-2.5%) from single runs with no variance estimate; the direction is consistent across two datasets, the magnitudes are not reliable.
- The paper's own trend (sec. 5.3.1, Fig. 3 left) is that Hash and Switch are level at 16-32 experts and Hash pulls ahead only as K grows; with experts in every layer the only evidence (5x16) has Switch equal or ahead. Our design (64 experts in every one of 4-6 layers) is not covered.
- The only small model tested (Wikitext-103, d=512, D=512, 1x16, Table 4) shows Hash losing to Switch with a 267K vocabulary (32.32 vs 31.76) and winning with 8K (11.58 vs 11.67); the paper concludes smaller dictionaries hash better. 16K is closer to the favourable case but untested.
- Nothing on quantized weights, and nothing below d=512.
