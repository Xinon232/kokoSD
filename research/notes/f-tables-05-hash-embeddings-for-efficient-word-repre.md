# Hash Embeddings for Efficient Word Representations  (https://arxiv.org/abs/1709.03933, Sep 2017; Dan Svenstrup, Jonas Meinertz Hansen, Ole Winther; DTU / FindZebra)

Tags: [P] full text read via the ar5iv HTML mirror, through a summarising fetch tool, so numbers are as that tool extracted them from the paper's tables; [M] my memory, not checked; [E] my estimate.

## What it is
A two-table embedding. Each token (word or n-gram) gets k candidate vectors, picked by k hash functions from a small shared pool, plus k learned scalar "importance" weights of its own; the embedding is the weighted sum. The pool is small (B rows), the per-token cost is only k scalars, so the addressable vocabulary K can be 10-100x the pool. The hashing trick (k=1, weights fixed to 1) and an ordinary embedding (B = vocabulary, identity hash) are special cases. Evidence is seven text-classification sets with bag-of-n-gram models only; no language modelling, no sequence model.

## Mechanism
[P §3]
- Pool E: B x d trainable. Importance table P: K x k trainable.
- D1: token -> {1..K}, a dictionary or a hash. D2_i: {1..K} -> {1..B}, k independent hashes.
- c_i = E[D2_i(D1(w))], i = 1..k;  p = P[D1(w)];  e_w = sum_i p_i * c_i.
- Optional: append p to e_w (width d+k). Figure 1 omits it; the text never says whether experiments used it.
- Parameters: B*d + K*k, against K*d for a standard table. One more vocabulary entry costs k scalars instead of d.
- Typical: k = 2, K > 10*B. Overhead: one 1xk by kxd product.

Pseudocode:
```
id = D1(ngram)                  # dictionary or hash, range K
p  = P[id]                      # k scalars
e  = 0
for i in 0..k-1: e += p[i] * E[hash_i(id) mod B]
```
Collision argument [P §4, Thm 4.1]: the chance a token collides with another is 1-(1-1/K)^(|T|-1) ~ 1-exp(-|T|/K). With |T| = 100M tokens and B = 1M, a token shares its full component set with another with probability ~1 at k=1 and ~0.0001 at k=2. Tokens that collide in D1 are indistinguishable. Weights driven to 0 act as learned pruning, so only important tokens compete for pool rows.

## Numbers
All test accuracy, % [P Table 2]; AG, Amazon-full, DBpedia, Yahoo, Yelp-full, Amazon-pol, Yelp-pol (120K-3.6M training documents, Table 1).
- No dictionary, bigrams only, d=20, one softmax layer [P §5.3]. Hash embedding K=10M, k=2, B=1M, 40M parameters: 92.4 / 60.0 / 98.5 / 72.3 / 63.8 / 94.4 / 95.9. Hashing trick, 10M buckets, 200M parameters: 92.0 / 58.3 / 98.6 / 72.3 / 62.6 / 94.2 / 95.5. Five times fewer parameters; better on 5, equal on 1, worse on 1 (by 0.1).
- Dictionary of the 1M most frequent n-grams (n<10), d=200, k=2, three 1000-unit ReLU layers, batch norm [P §5.4]. Hash (B cross-validated in 500-150K): 91.5 / 59.4 / 98.7 / 71.3 / 62.6 / 94.7 / 95.8. Standard embedding (vocabulary cap 10K-1M): 91.7 / 58.5 / 98.6 / 65.8 / 61.4 / 93.6 / 95.0. "3 times less parameters on average"; per-dataset B not reported.
- Ensemble of 10, B=50K, different hashes per model, ~150M parameters total [P §5.5]: 92.0 / 60.5 / 98.8 / 72.9 / 62.9 / 94.7 / 95.7.
- Parameter example [P §4]: growing K from 10^7 to 10^8 costs <200M parameters at k=2 against ~900M for a standard table.
- A separate hash for the importance row gave "a small but consistent improvement", initial findings only [P §6].
Single runs, no seeds; most gaps are 0.1-1.7 points.

## Smallest scale tested, and whether the result held there
Smallest data: AG News, 120K documents; +0.4 shallow, -0.2 deep. Smallest pool in the search grid: B = 500 rows, but which B won is not reported. Models are bag-of-embeddings; nothing autoregressive, nothing quantised.

## Training recipe and gotchas
- Adam, lr 0.001, cross-entropy, early stopping patience 10, 5% of training data as validation, Keras/TensorFlow [P §5].
- Regulariser: train on random snippets of 4-100 consecutive n-grams, test on full documents. No dropout mentioned.
- Initialisation of E and P is not stated; neither is the hash family.
- Training was "marginally faster" than standard embeddings at large vocabulary [P §3].
- Deep models underperformed the shallow one; authors suspect overfitting.
- [E] Gradients to P are sparse and scaled by the pool rows; follow Engram's recipe (Adam, raised lr, no weight decay) from capacity-04.

## Mapping onto the GBA design
All [E].
- Not needed for the 16K token embedding: 16K x 256 ternary is under 1 MB.
- Where it fits: the hashed n-gram tables. Engram-style multi-head hashing is this paper's k>1 with all weights 1. Adding P gives each n-gram id a private, static, learned gate: zero for junk n-grams, signed scale for useful ones. It is a hidden-state-free substitute for Engram's gate (which costs two ternary matrices or ~2K cycles tied).
- Cost per table at k=2, d=256: ~100 cycles hashing, 1 P read (2 bytes), 2 row reads, 512 multiply-adds at 7.5 = ~3.8K cycles; 0.1% of budget. Quantise p to {0, +-1, +-2, +-4} (3 bits, shifts) and it is ~1.5K cycles with no multiply.
- ROM: B=2^16 rows x 256 x 8 bit = 16 MB (3.3 MB ternary); K=2^22 ids x 2 bytes = 8 MB. So 4M distinguishable n-gram ids over a 64K-row pool, versus 16K-64K undistinguished slots per head for the same row bytes.
- RAM: none beyond the last 2-3 token ids.
- Export-time options: D1 as an exact dictionary of frequent n-grams (sorted table or perfect hash in ROM) so rare n-grams contribute nothing; drop ids with |p| ~ 0; fold any linear projection into E. Pre-multiplying p into rows is not possible without paying K x d ROM.
- Use power-of-two B and K with multiply-shift hashes (no divide).

## Verdict
Test. It is nearly free at inference and attacks the real weakness of small hashed n-gram tables (collisions, junk n-grams), but its evidence is 2017 classification with single runs, and ROM is not our scarce resource, so the 3-5x parameter saving alone is not the point. Settling experiment, inside the Engram ablation already planned: at fixed ROM bytes compare (a) K heads, weights 1; (b) hash embedding k=2 with 8-bit P; (c) k=2 with 3-bit shift P; (d) (b) plus a separate hash for P; sweep K/B in {1, 4, 16, 64}; report validation nll, the share of p near 0, and typing top-3 hit rate.

## Could not verify
- Venue: I recall NIPS 2017 [M]; the arXiv page and ar5iv text do not state it.
- The official code (github.com/dsv77/hashembedding): the README confirms it is code for this paper, but source files returned 404 through my tools, so initialisers, hash functions and whether p is appended are unknown.
- Per-dataset B and exact parameter counts in §5.4; any language-modelling result; any low-bit result.
- Text was read through a summarising fetch, not the raw PDF; table values were not cross-checked against the PDF.

## Links worth following
- https://github.com/dsv77/hashembedding (source files, for init and hashing)
- https://arxiv.org/abs/2601.07372 (Engram; see capacity-04 note)
- Not read: Bloom embeddings, compositional (quotient-remainder) embeddings and ROBE-Z are the usual follow-ups; I did not look them up, so no URLs given.
