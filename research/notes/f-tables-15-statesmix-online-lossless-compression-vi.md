# StateSMix: Online Lossless Compression via Mamba State Space Models and Sparse N-gram Context Mixing

https://arxiv.org/abs/2605.02904 (v1, submitted 5 Apr 2026 per the arXiv stamp, although the ID prefix is 2605), Roberto Tacconelli, single author, 10 pages. Code: https://github.com/robtacconelli/StateSMix (pure C, Apache-2.0).

Read in primary form (arXiv HTML and repo README) through a summarising fetch tool, so the numbers below are from the paper's tables but not character-checked by eye.

## What it is

A lossless file compressor: a tiny Mamba-style state space model, trained online from random initialisation on the file being compressed, predicts the next BPE token, and nine sparse n-gram count tables add a bias to its logits before arithmetic coding. Everything is learned from the file itself; there are no pretrained weights. It beats xz on small enwik8 excerpts and loses at 100 MB, far behind real context mixers (PAQ8px, cmix).

Correction to the selection rationale: there is **no PAQ-style learned mixer** and **no per-order ablation**. Mixing is a fixed, hand-set additive logit bias; a learned mixer appears only under future work. Per-order statistics are "qualitative" estimates.

## Mechanism

Neural part: Mamba, d_model 32, 2 layers, state dim 16, inner dim 64, conv width 4; about 19.7K floats in the layers, the rest in embedding and head (32 x vocab each). Tokens: GPT-NeoX BPE (49,152 types) remapped to a per-file vocabulary of 18K-44K.

N-gram tables (Table 3). Bigram is a direct array indexed by previous token; the others are hash tables of 2^24 slots, linear probing depth 8 (insert dropped if no free slot). Keys: exact 16-bit packing for orders 2-4; for longer contexts a rolling `k = k*104729 + token` then a mix64 finaliser (constants not given). Each slot holds a small sparse list of (uint16 token id, uint16 count), typically 4-32 entries.

Bias for token j from order k (Eq. 10):

    delta_j = lambda_k * log(1 + c_j / alpha_k)   if c_j > 0, else 0

| order | bigram | 3 | 4 | 5 | 6 | 7 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| lambda | 0.15 | 0.10 | 0.08 | 0.06 | 0.05 | 0.04 | 0.03 | 0.50 | 1.00 |
| alpha | 0.10 | 0.05 | 0.03 | 0.02 | 0.015 | 0.010 | 0.008 | 0.001 | 0.001 |

Only tokens actually seen in that context are touched (unseen tokens get 0, so the cost is the list length, not the vocabulary).

Entropy gate: with H the entropy (nats) of the neural softmax,

    s = clip(0.4 + 0.6 * H / 5.5, 0.2, 2.5)

so the tables count for more when the network is unsure.

Other terms: global unigram prior `0.1 * log(count_j + 1)`; a recency bonus `0.05 * exp(-3 * age)` over the last 64 tokens (age normalised to 0..1); a "two-token LZ" predictor keyed on the last two tokens that stores the latest successor and a count c, boosting that one token by `1.5 * (1 - 1/(1 + 0.3c))`.

Final: `logit = ssm + s*unigram + s*sum_k delta_k + lz + recency`, then softmax, then a 32-bit range coder with 2^16 CDF scale.

## Numbers

All on enwik8 prefixes, CPU only.

- Table 4, bits per byte, StateSMix vs xz -9e: 1 MB 2.123 vs 2.326 (-8.7%); 3 MB 2.149 vs 2.271 (-5.4%); 10 MB 2.162 vs 2.177 (-0.7%); 100 MB 2.130 vs 1.992 (+6.9%, worse).
- Table 5, ablation at 3 MB: unigram counts only 4.191 bpb; n-grams + counts, no network 3.517; network + counts, no n-grams 2.240; full 2.149. So the nine tables buy about 4.1% on top of the network, and 16% over unigram alone.
- Table 6, 100 MB: bits per token 8.10 at start, 6.83 by 3M tokens, 6.813 final; flat after 3M tokens.
- Table 8 (10 MB, approximate): context hit rate >0.99 bigram, 0.85 trigram, 0.65 4-gram, 0.45 5-gram, 0.28 6-gram, 0.14 7-gram, 0.05 8-gram, 0.01 16-gram, <0.005 32-gram; successors per context falls from about 8 to about 1.2.
- Table 7: 1.1K tokens/s on one core, 2.0K on four; 6.1 GB peak RAM, 5.1 GB of it n-gram tables; about 75% of time is training, mostly the output head.
- README landscape for 100 MB (author's citations): PAQ8px about 1.27, cmix v21 about 1.17, NNCP v3 about 1.19 bpb.

## Smallest scale tested, and whether the result held there

1 MB of enwik8 (about 0.3M tokens) is the smallest and is where it looks best. The model itself is tiny (about 20K non-embedding parameters), which is interesting for us, but it plateaus at 6.8 bits per token, a very weak language model. The n-gram gain was only measured at 3 MB.

## Training recipe and gotchas

Online only: 32-token chunks, Adam lr 0.002, betas 0.9/0.999, gradient clip 5.0, label smoothing 0.12; 8 passes per chunk for the first 10 chunks, 4 for the next 20, then 2; truncated backpropagation through time.

Gotchas: the lambda and alpha values are hand-tuned for one file and one weak network; nothing is learned jointly. The abstract says about 120K active parameters while Table 2 gives 1.18M-2.85M totals. Count update, capping and eviction rules are not described. Single author, no independent replication, one dataset.

## Mapping onto the GBA design

What transfers is the sparse additive bias, not the system. Our tables would be built offline from the training corpus and frozen in ROM:

- Per order: hash the last k tokens (a few shifts/adds, one multiply per token if the rolling hash is kept incrementally), one ROM probe, then a list of at most 4-8 (token id, bias) pairs with the bias `lambda*log(1+c/alpha)` already computed and quantised to 4-8 bits. Cost per order: about 1 lookup plus up to 8 adds. Nine orders: well under 1,000 cycles, negligible against 3-5M.
- ROM: an entry of 8 x (16-bit id + 8-bit bias) is 24 bytes; 1M contexts is 24 MB, so we must prune hard by count and probably stop at orders 2-5, where Table 8 says hits are frequent.
- Two-level head: a bias on token j must be added both to its class logit (or to the class logit as a log-sum, which we cannot afford) and to its member logit. Simplest: store a separate class-bias list per context and apply member biases only inside the chosen class.
- Entropy gate: entropy needs logs; replace it with a cheap proxy such as the top-1 minus top-2 logit gap, mapped through a 16-entry table to a shift amount.
- The recency bonus (last 64 tokens) and the two-token "latest successor" table are RAM-cheap (64 entries; a few KB of slow RAM) and give in-conversation adaptation, e.g. repeating the user's names. Also directly useful for predictive typing.
- Better than fixed lambdas: learn one scale per order (or per order x confidence bucket) during training, which our existing hashed n-gram embeddings partly already do in vector form.

## Verdict

**Test, low priority.** The evidence is weak (one file, a 20K-parameter network at 6.8 bits per token, hand-set weights), and a 4% gain over such a weak model says little about a trained 256-wide transformer that already reads hashed n-gram embeddings. But the output-side sparse bias is nearly free on our hardware and is a different path from input-side n-gram embeddings. Settling experiment: on the trained model, add ROM-style pruned 2-5-gram bias lists with learned per-order scales and a confidence gate; measure held-out perplexity and top-3 typing accuracy against the same ROM bytes spent on more hashed embeddings.

## Could not verify

- Any PAQ-style logistic mixing or learned mixer weights: not present in the paper.
- True per-order ablations: not present; Table 8 is labelled qualitative.
- mix64 constants and count update/eviction rules: not in paper or README; would need `ssm_model.c` / `ssm_codec.c`, which I did not read.
- Baseline figures for PAQ8px, cmix, NNCP are the author's README citations, not checked against their sources.
- Text was read through a summarising tool; exact wording is not quoted.

## Links worth following

- https://github.com/robtacconelli/StateSMix (source files `ssm_codec.c`, `ssm_config.h`, `architecture.txt` for the unspecified details)
- PAQ8px / cmix / Mahoney's "Data Compression Explained" for real learned logistic mixing with context-selected weight sets, which is what this paper was expected to contain (no URL verified in this pass)
- NNCP (Bellard) for online-trained neural compression at larger scale (no URL verified in this pass)
