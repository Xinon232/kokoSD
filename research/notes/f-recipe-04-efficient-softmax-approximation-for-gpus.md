# Efficient softmax approximation for GPUs (adaptive softmax)

https://arxiv.org/abs/1609.04309 (v1 Sep 2016, v3 Jun 2017, ICML 2017). Grave, Joulin, Cissé, Grangier, Jégou (Facebook AI Research). Code: https://github.com/facebookresearch/adaptive-softmax (Lua Torch, archived 2021).
Tags: [P] = read in the primary text (ar5iv full text, via a summarising fetch tool); [S] = secondary; [I] = my inference.

## What it is
A two-level factorised output layer for word-level language models. The most frequent words sit directly in the root ("short-list") next to one logit per tail cluster; rare words live in frequency-banded clusters that are evaluated only when needed and use a narrower classifier. Cluster boundaries are chosen to minimise a measured model of compute time, not to equalise probability mass. The paper is about GPU training time; inference cost is not discussed [P].

## Mechanism
- Sort the vocabulary by frequency. Partition into head V_h (k_h words) and J tail clusters V_1..V_J, contiguous in rank [P].
- Head: softmax over k_h + J outputs from hidden state h (dimension d). Tail cluster i: softmax over k_i words. P(w) = P_head(w) for head words; P(w) = P_head(cluster i) * P_i(w | h) otherwise [P].
- Tail capacity: each tail cluster first projects h down with its own matrix, "from d to d_t = d/4" [P]. PyTorch's `AdaptiveLogSoftmaxWithLoss` generalises this to floor(d / div_value^i) for cluster i, div_value = 4.0, head_bias = False [P, PyTorch docs].
- Cost model (GPU): g(k,B) = max(c + λ k0 B0, c + λ k B); flat below k0 ≈ 50 outputs, affine above. Total C = g(J + k_h, B) + Σ_i g(k_i, p_i B). In the linear regime C = (J+1)c + λB[J + k_h + Σ_i p_i k_i], with p_i the probability mass of cluster i [P, Eq. 5-10].
- Consequence: frequent words go in the smallest cluster; for fixed J the optimal sizes come from dynamic programming over contiguous splits of the sorted vocabulary [P].
- J: the model's optimum is 10-15 clusters, but gains beyond 5 are small; the authors use 2-5 [P, Fig. 4].
- Released cutoffs [P, README]: PTB 2000; Text8 (44k vocabulary) 2000, 10000; Billion Word (~800k) 4000, 40000, 200000.

## Numbers
All one-layer LSTMs, 5 epochs unless stated [P].
- Text8, 44k vocabulary, d = 512 (Table 1), perplexity / minutes: full 144 / 83; sampling 166 / 41; HSM-frequency 166 / 34; HSM-similarity (Brown clusters) 155 / 41; D-softmax 195 / 53; D-softmax with projections 147 / 54; adaptive 147 / 30.
- Europarl (Table 3), full vs adaptive: bg 50k vocabulary 37 / 58 vs 37 / 18; cs 83k 62 / 132 vs 62 / 30; da 128k 37 / 713 vs 35 / 105; de 143k 42 / 802 vs 40 / 110. HSM-frequency: 43, 78, 42, 51. HSM-similarity: 39, 68, 38, 43.
- Billion Word, ~800k vocabulary, d = 2048 (Table 2): 43.9 test perplexity, 39.8 with two layers; under three days on one GPU.
- All words in leaf clusters instead of a root short-list: "around 5-10% performance drop" [P]; stated in prose, no table, metric not specified.
- Penn Treebank: 20% of the vocabulary covers 87% of tokens [P].
- Follow-up, Baevski and Auli 2018 (arXiv 1809.10853), transformer on WikiText-103, Table 3 [P for that paper]: full softmax 24.92 (476.8M parameters), adaptive softmax 22.18 (263.1M), adaptive tied input and output 20.51 (246.9M). Dropout 0.2 on the tail projections moved validation perplexity from 24.74 to 21.23; full softmax 23.87. Tail dropout did not help on Billion Word.

## Smallest scale tested, and whether the result held there
Text8: first 100M characters of Wikipedia, 44k vocabulary, LSTM-512 with 256-wide embeddings. It held: 3 perplexity points behind full softmax at 2.8x less training time. Nothing was tested at a 16K vocabulary, with subword tokens, with models under about 10M parameters, or with quantised weights.

## Training recipe and gotchas
- Adagrad, step 0.1, batch 128, BPTT 20, gradient norm clipped to 1, weight decay 1e-6 (none on Billion Word) [P].
- Token ids must be in frequency order; the head has no bias in the reference code [P, PyTorch docs].
- Pure frequency binning for all words is the weak variant (HSM-frequency loses 5-22 points to full softmax). The gains come from the short-list plus the fact that only rare words pay the factorisation penalty [P].
- Tail projections overfit on small data; use dropout there (follow-up paper).
- Table 2's caption says "55 epochs" while the text says five; unresolved in the paper.
- [I] At a 16K vocabulary the 5090 can afford the exact full distribution log P(w) = log P(c) + log P(w|c) for every token, so distillation with a KL loss needs no sampling. Train the factorised form from the start, since it is what ships.

## Mapping onto the GBA design
[I] throughout. The GBA cost is linear with no flat region: weights touched = d(k_h + J) + Σ_i p_i (d·d_t + d_t·k_i), or p_i·d·k_i without projection. A flat 16K x 256 head is 4.1M ternary weights, about 6.5M cycles, more than the whole token budget, so a factorised head is mandatory.
- Example, d = 256: head of 512 words + 62 clusters of 250 words = 147K weights (about 235K cycles) every token. A tail cluster at full width is 64K more weights; with a d/4 projection it is 16K + 16K = 32K. That is paid only when a cluster is sampled, so expected cost scales with tail mass.
- ROM is not the constraint (16K x 256 ternary is about 1 MB), so the d/4 projection is purely a compute choice. It also forces a second set of run-time partial-sum tables for the 64-wide vector (16 groups x 81 entries); the head and full-width clusters reuse the tables built from h.
- Precompute into ROM: frequency-sorted token ids, cluster boundaries, per-cluster row offsets. RAM: head logits (574 x 16 bit) plus one cluster's logits (250 x 16 bit), about 1.6 KB.
- Changes to the current "class then member" plan: add a short-list in the root; pick boundaries by minimising the weight count above with a DP over measured token frequencies; cluster tail members by similarity inside frequency bands.
- Predictive typing: top-3 needs the head plus only those clusters whose cluster probability exceeds the third-best head word.
- Sampling: draw from the head; expand a cluster only if a cluster entry is drawn. Top-p or temperature over the joint needs care, because tail members are not all scored.

## Verdict
Adopt the structure (short-list in the root, frequency-ordered tail clusters, cost-optimal boundaries). Test the narrow tail projection. Settling experiment: on the 16K subword vocabulary, at the target width and with ternary weights, compare (a) full softmax, (b) all-leaf class/member, (c) short-list + clusters at full width, (d) as (c) with d/4 tails; report perplexity and expected weights touched per token.

## Could not verify
- Whether the gain survives subword vocabularies, whose frequency distribution is flatter than word-level. No source found.
- The actual cutoffs used for the paper's tables (only the README defaults are known), and numeric λ and c (only c_m = 0.40 ms on K40, 0.22 ms on M40).
- What metric the "5-10% drop" refers to.
- Any inference-time measurement. The paper has none.
- I read the paper through a summarising fetch tool, not the raw PDF; Table 3 columns beyond those quoted were not rechecked.

## Links worth following
- https://arxiv.org/abs/1809.10853 (adaptive inputs: tied banded input embeddings, tail dropout)
- https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html
- https://github.com/facebookresearch/adaptive-softmax

## Independent verification
Second reader, 2026-11-19. Read directly from the raw HTML of the paper (arxiv.org/html/1609.04309 and ar5iv, text extracted locally, no summarising tool): Sections 4.2, 4.3, 5, Tables 1-3. All below is [P] unless marked [I].

1. Short-list + tail clusters within 3 perplexity points of full softmax - CONFIRMED. Table 1 (Text8, 44k words, d = 512): full 144 / 83 min, ours 147 / 30 min; the text says "below by 3 points of perplexity". Table 3 (Europarl, ppl after 5 epochs): bg 50k 37 vs 37, cs 83k 62 vs 62; also da 37 vs 35, de 42 vs 40, el 100k 38 vs 36, es 87k 30 vs 29. Caveats: single runs, integer-rounded perplexity, no variance; fixed 5 epochs of Adagrad, so this is equal-epoch, not converged, comparison; one-layer word-level LSTM with 256-wide input embeddings. The number of clusters and cutoffs behind each table row are not stated in the paper (only "between 2 and 5").

2. All-leaf placement costs "around 5-10%" - PARTLY. The words are there (Section 4.2), but the figure sits inside the citation parenthesis for Mikolov et al. 2011c and Zweig & Makarychev 2013: "(Mikolov et al. 2011c; Zweig & Makarychev 2013, around 5-10% performance drop,)". So it reads as a figure taken from (or supported by) that older class-based-LM literature, not a measurement reported in this paper; the authors only say "we observe empirically" with no table, no metric, no setting. Nothing in Tables 1-3 isolates root short-list versus all-leaf with the same clusters (HSM-freq differs in binning too). Treat as an anecdote, not a number to design against.

3. Frequency-binned two-level softmax much worse; similarity clustering recovers part - CONFIRMED (numbers exact). Table 1: HSM (freq) 166, HSM (sim) 155, full 144. Table 3: cs 78 / 68 / 62, de 51 / 43 / 42 (freq / sim / full); also bg 43 / 39 / 37, da 42 / 38 / 37, el 45 / 39 / 38, es 34 / 30 / 30. Details the first note omitted: the freq baseline bins on the square root of counts (the best of the variants they tried); sim uses Brown clustering; the number of classes is not given; HSM (sim) is slower than HSM (freq) and than adaptive softmax (e.g. de 154 min vs 124 vs 110). On de and es similarity clustering recovers nearly all of the gap (43 vs 42, 30 vs 30), on Text8 and cs only about half. Same equal-epoch caveat as claim 1.

4. d/4 tail projection "without visible loss" - PARTLY. The table numbers are right (da 35 vs 37, de 40 vs 42, vocab 128k / 143k, d = 512). But the paper never ablates the projection: there is no adaptive-softmax row with full-width tails, and Section 4.2 only says the tail projection "Typically ... reduces the size from d to d_t = d/4" - it does not state the exact tail widths used for each table. The only related evidence is D-softmax (disjoint slices of h: da 46, de 56) versus D-softmax [*] (whole h through projections of different sizes: da 36, de 41), which shows that projecting the whole hidden state beats slicing it, not that d/4 is free. Beating full softmax after 5 epochs more plausibly reflects faster optimisation or regularisation of rare words than a capacity result. Here d/4 = 128 float dimensions; the GBA analogue would be 64-96 ternary dimensions, which is untested.

Overall [I]: the table numbers were transcribed correctly, but they come from float one-layer LSTM-512 word-level models with 44k-143k word vocabularies, 5 fixed epochs, single runs; the paper measures GPU training time only and has no inference, subword, small-vocabulary or quantised result, and itself notes approximations are "comparatively less interesting for small vocabularies". The structural conclusion (root short-list better than pure frequency-binned classes; similarity helps) is safe to carry over as a hypothesis; the "5-10%" figure and "d/4 is free" are not established by this source and need the settling experiment already listed in the Verdict.
