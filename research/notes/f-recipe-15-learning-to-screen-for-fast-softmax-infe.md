# Learning to Screen for Fast Softmax Inference on Large Vocabulary Neural Networks (L2S)

https://arxiv.org/abs/1810.12406 (ICLR 2019; Patrick H. Chen, Si Si, Sanjiv Kumar, Yang Li, Cho-Jui Hsieh; UCLA / Google Research).
Read in primary form: full text of the arXiv PDF (13 pages, including appendix tables 4-6). Figures 2-7 are plots I could not read; only their captions.

## What it is

A post hoc inference speed-up for the output layer of a frozen, already-trained model with a flat softmax. Context (final hidden) vectors are partitioned into r clusters by a learned linear classifier; each cluster owns a fixed candidate word list; at inference the exact logits are computed only for that list. Clusters and lists are learned jointly so that the list contains the full softmax's top-5 words under an average-list-size budget. It exploits structure in the contexts, where earlier methods (SVD-softmax, adaptive softmax, MIPS) exploit only the structure of the weight rows.

## Mechanism

Inference (Eq. 2): `z = argmax_t v_t·h` (r dot products), then `logit_s = w_s·h + b_s` for s in list C_z only; softmax/top-k over those; words outside get probability 0. Cost O((r + L̄)d) versus O(Ld), L̄ = average list size.

Training (Algorithm 1), inputs: context vectors h_i from the training set, frozen W, b.
1. Labels y_i = top-5 words of the full softmax for h_i (the model's own prediction, not the corpus word).
2. Initialise v_t by spherical k-means over {h_i}; all lists empty.
3. Repeat T times:
   a. Lists fixed, learn v by SGD on Eq. 8: `sum_i [ sum_{s in y_i} (1 - c_{z_i,s})^2 + λ sum_{s not in y_i} c_{z_i,s}^2 ] + γ·max(0, L̄ - B)`. z_i is a straight-through Gumbel-softmax sample (temperature 1) of P(t|h) = softmax_t(v_t·h). L̄ is a moving average over mini-batches.
   b. v fixed, rebuild the lists by greedy knapsack: each (cluster t, word s) is an item whose weight is proportional to the cluster's sample count; sort by value/weight and add until average size reaches B.

Hyperparameters: λ = 0.0003, γ = 10 everywhere; r = 100 or 200 recommended; B is the speed/accuracy knob. The budget is global, so lists differ in length per cluster.

My derivation (the paper gives no formula): with n_ts = contexts in cluster t having s in their top-5 and N_t = cluster size, the item's value is n_ts - λ(N_t - n_ts) and its weight N_t/N, so the greedy order is simply descending n_ts/N_t: "add the words most often in the top-5 within this cluster, one global threshold".

## Numbers

All models are 2-layer LSTMs; times are softmax layer only, numpy, one Xeon E5-2620 thread.

Table 1 (speed-up, P@1, P@5 against the exact softmax's top-k):

| | PTB-Small (d=200) | PTB-Large (d=1500) | IWSLT14 DE-EN (~25K vocab) |
|---|---|---|---|
| L2S | 10.6x, 0.998, 0.990 | 45.3x, 0.996, 0.982 | 20.4x, 0.989, 0.993 |
| FGD | 1.3x, 0.980, 0.989 | 6.9x, 0.975, 0.979 | 6.7x, 0.987, 0.981 |
| SVD-softmax | 0.8x, 0.987, 0.99 | 2.3x, 0.988, 0.981 | 3.4x, 0.98, 0.985 |
| Adaptive softmax | 1.9x, 0.972, 0.981 | 4.2x, 0.974, 0.937 | 3.2x, 0.982, 0.984 |

Full softmax: 0.32 / 4.32 / 4.83 ms per step.

- Table 4: spherical k-means alone (no joint learning): 4x / 0.988 / 0.992; 6.9x / 0.992 / 0.971; 13.8x / 0.991 / 0.993. So most of the gain on NMT comes from plain clustering; the learned step matters most on PTB.
- Table 2 (BLEU): DE-EN beam 5: 30.33 -> 30.19 at 13.4x; beam 1: 29.50 -> 29.46 at 14.0x. EN-VI beam 5: 25.35 -> 25.27 at 20x.
- Table 3 (r = 50/100/200/250): 0.12/0.17/0.14/0.12 ms; P@1 0.997/0.998/0.998/0.994; P@5 0.988/0.99/0.99/0.98.
- Table 5 (perplexity, out-of-list logits from a low-rank W, rank 20/200): PTB-Small 112.28 -> 115.91 at 5.69x; PTB-Large 78.32 -> 80.09 at 8.11x.

## Smallest scale tested, and whether the result held there

PTB-Small: d = 200, 2-layer LSTM, PTB vocabulary (10K by the standard preprocessing; the paper does not print it). It held: 10.6x at P@1 0.998, but the speed-up is the smallest of the three, and perplexity cost 3.2% at only 5.7x. No transformer, no quantised model, no subword vocabulary, and no sampling-quality test.

## Training recipe and gotchas

- Nothing in the LM is retrained; you need a dump of hidden states and the full-softmax top-5 for each.
- Not reported: B or L̄ for any result, T, learning rate, batch size, number of context vectors, whether P@k is on held-out text. Implied row counts cannot be recovered from numpy wall-clock ratios (the 10.6x vs 45.3x gap between d=200 and d=1500 suggests fixed overheads).
- Table 4 prints PTB-Large L2S as P@1 0.999 / P@5 0.82, contradicting Table 1 (0.996 / 0.982); probably a typo.
- Labels are top-5 only, so the lists are tuned for greedy/top-5 decoding. Sampling with a wider nucleus needs wider labels (top-k larger, or the top-p set).
- Lists have unequal lengths: worst-case per-token time is unbounded unless a per-cluster cap is added.
- No code link in the paper; I found no repository.

## Mapping onto the GBA design

A full 16K x d softmax is impossible (16,384 rows x 256 weights x 1.6 cycles = 6.7M cycles), so some screen is mandatory; the candidate design uses class-then-member (about 128 + 128 rows = 65K weights = 105K cycles at d=256).

L2S version: r = 128 ternary cluster rows + L̄ member rows read from a flat 16K x d ternary output matrix. The run-time partial-sum tables for h are built once and shared by all rows.
- Cycles (d=256): r rows 52K; L̄ = 256 / 512 rows 105K / 210K. Total 0.16-0.26M, i.e. 1.5-2.5x the two-level head, 4-8% of a 3-5M budget.
- ROM: output matrix 4.2M weights (about 1 MB at 2 bits); lists r x L̄ x 2 bytes = 131 KB at L̄ = 512. Negligible.
- RAM: one 16-bit logit per candidate (1 KB at L̄ = 512), or just a top-k heap for typing suggestions.
- No multiplies beyond what the ternary kernel already does.

What it buys over class-then-member: exact global top-k (the typing-suggestion case; a factorised head needs a beam over several classes to get a true top-3), and the model can be trained with an ordinary flat softmax, with the screen fitted afterwards and re-tunable without retraining.

GBA-specific variant (my proposal, not in the paper): step 3b works for any hard partition, so the cluster key can be something free, such as the previous token id or a hashed bigram, or a few sign bits of a small ternary projection, instead of r dot products. The list can also be the union of a token-keyed list and a cluster list.

## Verdict

Test. It is cheap to try post hoc on whatever model we train, and the k-means baseline already captured most of the gain on one of three tasks. It is not obviously better than the two-level head for sampling, because required L̄ at 16K subword vocabulary and with sampling (not top-5) is unknown. Settling experiment: dump about 5M hidden states from the trained model; at L̄ in {128, 256, 512, 1024} compare (a) global frequency list, (b) previous-token-keyed list, (c) spherical k-means r=128, (d) L2S, (e) two-level head with a top-m class beam; report P@1, P@3, captured probability mass of the top-p=0.9 set, and perplexity of the truncated distribution on held-out text.

## Could not verify

- L̄/B behind every headline number; PTB and EN-VI vocabulary sizes; exact DE-EN vocabulary ("around 25K" only).
- Figures 2-7 (precision vs speed-up curves).
- Any replication on transformers or on small/quantised models; I did not search for follow-ups beyond a code search.
- The knapsack value formula and the free-key variant are my inferences.

## Links worth following

- Shim et al. 2017, SVD-softmax (low-rank preview logits; the fallback used for perplexity here) - NeurIPS 2017.
- Zhang et al. 2018, FGD: Navigating with Graph Representations for Fast and Scalable Decoding of Neural Language Models - the strongest baseline here.
- Grave et al. 2017, adaptive softmax (already in notes: f-recipe-04).
