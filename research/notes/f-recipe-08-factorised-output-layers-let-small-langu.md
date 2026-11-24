# Factorised Output Layers Let Small Language Models Use a 131k Vocabulary at Equal Compute

https://www.alphaxiv.org/abs/2611.factorised-output-layers-small-language-models (v1, 15 Nov 2026). Jakub Michałek, independent. No arXiv ID. Code, logs, paper source: https://github.com/TG-TG-TG-TG-TG-TG/factorised-output-layers-let-small-language-models-use-a-131k-vocabulary-at-equal-compute
Tags: [P] = read in the primary text (`paper/paper.md` and scripts fetched raw from the repo); [I] = my inference. Not peer reviewed; the author states AI assistance.

## What it is
An equal-training-FLOPs comparison of three output heads on 6-block, d=384 (and d=128) decoder LMs trained on FineWeb-Edu: flat tied softmax, adaptive softmax, and a two-level class head (frequent short-list + k-means tail clusters). The claim is about training compute: a cheaper head buys more tokens, and that makes a 131k vocabulary pay. The architecture is not claimed as new. It also reports equal-token results, a design ablation, exact decoding rules and decode timings [P].

## Mechanism
- F = the K tokens most frequent as training targets; every other token is in exactly one of C' tail clusters. Top softmax over K + C' logits: u_w·h for w in F, z_c(h) for clusters. p(w) = p_top(w) if w in F, else p_top(c(w)) · softmax over members of c(w) of u_v·h. u = tied input embedding, full width [P].
- K = 8,192, C' = 4,096; mean cluster size 11 (50k vocab) or 30 (131k) [P].
- Cluster logit z_c = (mean(u_v, v in c) + R_c)·h + b_c, R zero-init. Within noise of a free cluster vector [P, Table 2].
- Clusters: balanced k-means, capacity ceil((V−K)/C'), on embedding rows of a flat d=128 proxy trained on ~50M tokens [P; `round13_classes.py`].
- Head cost per token: d(K + C' + n̄) versus dV: 0.245x at 50k, 0.094x at 131k [P].
- Exact sampling: draw a top-level entry; if it is a cluster, draw a member (one cluster expanded). Exact greedy: p(w) ≤ p(c(w)), so expand clusters in decreasing p(c) until p(c) ≤ best p(w) found [P].
- Adaptive-softmax baseline: same short-list, two frequency bands, untied tails at d/4 = 96 and d/16 = 24, or untied full width [P].

## Numbers
All FineWeb-Edu validation bits/byte (BPB), d=384, 6 blocks, budget 1x = 6.717e16 training FLOPs unless stated [P].
- Table 1, 50k: flat 1.1753 / 1.1763 (347M tokens); two-level 1.1626 / 1.1616 (606M); adaptive 1.1624 / 1.1632. Two-level minus flat −0.0137 (2-seed mean); two-level ties adaptive.
- Table 1, 131k: flat 1.2248 (177M tokens, 1 seed); two-level 1.1512 / 1.1539; adaptive reduced-width 1.1629 / 1.1628; adaptive full-width 1.1522 / 1.1536 (108.1M params vs 62.5M). Full-width tail is worth −0.0100 BPB.
- Budget sweep, 50k, seed 0: two-level minus flat −0.0226 (0.5x), −0.0127 (1x), −0.0056 (2x, 1 seed).
- Equal tokens: two-level minus flat +0.0115 at d=384, 300M tokens (1 seed, uncharged "old" clusters); +0.0101 at d=128, 1.2B tokens. The loss is in rare targets: +0.24 bits/token for ranks 4k-16k, +0.36 for ranks ≥ 16k, about zero for frequent ones (d=128, Fig. 4).
- Table 2 (d=128, equal tokens, 50k): flat 1.2835; 1,024 random clusters 1.3694; k-means −0.0572; + 4,096 singletons −0.0107; 8,192 singletons −0.0043; 4,096 vs 1,024 clusters −0.0001; exact re-scoring of top-16 clusters −0.0015.
- Smaller vocab instead: flat 8k 1.1875, flat 4k 1.2056 versus two-level 50k 1.1621. Bytes/token: 3.473 (4k), 3.933 (8k), 4.633 (GPT-2 50k), 4.944 (131k).
- Short-list coverage: top 8,192 of 131k cover 0.857 of training targets (Appendix C).
- Greedy on the trained 131k model: exact on 4,096 positions, 10.0 clusters expanded on average, fallback 0.17%.
- Batch-1 decode 0.203 vs 0.279 ms/token (two-level vs flat, 50k); training throughput 1.98x / 4.29x (50k / 131k).
- BLiMP: no difference (72.06% vs 71.68%). LAMBADA: flat 8k 16.8% vs about 14-15% for both 50k heads, a tokenizer effect.

## Smallest scale tested, and whether the result held there
d=128, 6 blocks, 50k vocabulary, up to 1.2B tokens: the ablation and the equal-token gap were measured there, and the gap (+0.0101) matches d=384. No vocabulary below 50k with a factorised head, no model under about 12M parameters, no quantisation, no dialogue data.

## Training recipe and gotchas
- AdamW (0.9, 0.95), weight decay 0.1, clip 1.0, peak LR 2e-3 for every arm, 65,536 tokens/step, WSD schedule (3M-token warmup, decay to 0.1x over 10% of steps), RMSNorm, RoPE, GELU, context 1,024, JAX on one RTX 5070 Ti [P].
- Random clusters are far worse than a flat head; clustering quality is the largest single factor [P].
- Reduced-width tails cost 0.01 BPB; keep tail rows full width and tied [P].
- A linear cluster logit cannot track the log-sum-exp of its members (mean Jensen gap 3.70 bits on tail targets at d=128); offered as a hypothesis for the equal-token loss, not established [P].
- The within-cluster loss has no gradient along the cluster mean [P].
- 1-2 seeds; both seeds share clusters; 50k and 131k tokenizers differ in domain; one learning rate [P].

## Mapping onto the GBA design
[I] throughout. Our case is the paper's unfavourable one: inference compute forces the factorisation while training can afford anything, so we pay the equal-token penalty. +0.0115 BPB x 4.633 bytes/token is about 0.053 bits/token, about 3.8% perplexity, concentrated in rare tokens.
- Cost is d(K + C' + n̄) ternary weights at 1.6 cycles. At d=256, 16K vocab: K=512, C'=250, n̄=62 gives 211K weights, about 337K cycles per sampled token; all rows reuse the partial-sum tables built from h. RAM: 762 + 62 16-bit logits, about 1.7 KB.
- Sampling expands one cluster. Greedy or top-3 typing suggestions need more: 10 clusters on average in the paper, which would be about 159K extra weights here. Use the p(c) bound and cap the expansions.
- Cluster count barely matters (1,024 vs 4,096: −0.0001), so choose K and C' purely for cycles; short-list size matters more.
- Use free ternary cluster rows in ROM (the tied-mean form buys nothing); ROM also holds the id-to-(cluster, slot) permutation and row offsets.
- Head cost is nearly independent of V, and larger vocabularies give more bytes per token at fixed tokens/s, so a 32-64K vocabulary is worth a test. LAMBADA warns that rare names copy worse with large vocabularies.
- Keep tail rows at full width; drop the d/4 projection idea from the adaptive-softmax note.

## Verdict
Adopt the structure and the clustering recipe (balanced k-means on a cheap proxy's embeddings, short-list in the root, full-width tied tail). Test the vocabulary size. Settling experiment: at our width, ternary QAT, 16K and 32K vocabularies, compare flat (upper bound), random-cluster, and k-means two-level heads at equal tokens; report bits/byte, loss by frequency rank, and clusters expanded for top-1 and top-3.

## Could not verify
- Any result with ternary weights, a vocabulary under 50k, or sampled-text quality: none exist in the source.
- The inside of `balanced_kmeans` (iterations, initialisation): only the calling script was read.
- Figures were not viewed; Fig. 4 numbers are from the text.
- Whether the flat-to-factorised gap can be closed by distilling from a flat-head teacher: not tested there.
- The selection blurb pairs "0.094x" with "50k": 0.094x is the 131k ratio; at 50k it is 0.245x.

## Links worth following
- Repo `runs/round13/round13_summary.md`, `runs/round12/perf.md`, `round9_heads.py`
- Godey et al. 2024, arXiv:2404.07647 (softmax bottleneck in small LMs)
- Tao et al. 2024, "Scaling Laws with Vocabulary" (NeurIPS 2024)
