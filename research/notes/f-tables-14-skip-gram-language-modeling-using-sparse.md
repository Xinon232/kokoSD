# Skip-gram Language Modeling Using Sparse Non-negative Matrix Probability Estimation

https://arxiv.org/abs/1412.1454 (v2, June 2015; Shazeer, Pelemans, Chelba, Google). Journal version: "Sparse Non-negative Matrix Language Modeling", TACL 2016, https://aclanthology.org/Q16-1024. Both PDFs read in full as extracted text. Tags: [A] arXiv v2, [T] TACL, [I] my inference.

## What it is
A pure count-and-lookup language model (SNM) that combines arbitrary overlapping context features (n-grams, gapped "skip-grams") without a back-off hierarchy. Each feature votes with its relative-frequency distribution over next words, rescaled by a learned per-(feature, target) factor. With skip-gram features it comes within 3% perplexity of the best RNN of 2013 on One Billion Word, at enormous model size.

## Mechanism
- Feature = tuple (r, s, a): r remote words, s skipped words, a adjacent words. "brown skip-2 over the lazy" is (1,2,3). Tied variant drops the value of s ("skip-*") so all gap lengths share one row [A Sec. 2].
- Prediction, F = active features of the context [A Eq. 5]:
  `P(w|ctx) = sum_{f in F} M[f,w] / sum_{f in F} rowsum[f]`, with rowsum precomputed.
- `M[f,w] = exp(A(f,w)) * C[f,w] / C[f,*]` [A Eq. 6]; `A = sum_k theta[hash(meta_k)]`.
- Meta-features: all conjunctions of {feature identity, feature type (r,s,a), log2 bucket of C[f,*], target identity, log2 bucket of C[f,w]} = 2^5-1 = 31 (up to 2^7-1 with a second, ceil bucket weighted by the fractional log2) [A Sec. 3.2; T App. A pseudocode].
- Only theta is hashed (modulo table size, collisions allowed); M and counts are stored exactly in sorted string tables [T Sec. 3.2].
- Loss: per-word Poisson instead of softmax; negatives are folded into positives so updates touch positive pairs only [A Eq. 14]: `grad_A = M[f,w] * (C[f,*]/C[f,w] - 1/y_w)`. Leave-one-out counts (subtract the current event) prevent overfitting [A Eq. 15].
- Configurations: SNM5-skip = 5-grams + (r 1..3, s 1..3, r+a <= 4) + tied (r 1..2, s >= 4, r+a <= 4). SNM10-skip = 10-grams + (s=1, r+a <= 5) + tied (r=1, s 1..10, r+a <= 4).

## Numbers
One Billion Word, 0.8B training tokens, 793,471-word vocabulary, test perplexity:
- n-gram only, orders 5/6/7/8 [A Table 1]: KN 67.6/64.3/63.2/62.9; SNM 70.8/67.0/65.4/64.8; Katz 79.9/80.5/82.2/83.5; KN+SNM 66.5/63.0/61.7/61.4.
- Skip features only, no n-grams above unigram: 69.8 (61B parameters). Plus 5-grams (SNM5-skip): 54.2 (62B). Interpolating KN5 instead of joint: 56.5; KN5 + SNM5-skip: 53.6 [A Table 2].
- SNM10-skip 52.9 (33B); RNNME-256/512/1024: 58.2/54.6/51.3 (20B each); SNM10-skip + RNNME-1024 41.3; all models 41.0 [A Table 3, T Table 3].
- Meta-feature ablation on SNM5, baseline 70.8 [A Table 4]: no feature identity 71.9; no feature type 71.4; no feature count 70.6; no feature-target count 73.2; no second bucket 70.3-70.6.

## Smallest scale tested, and whether the result held there
44M-token Gigaword subset, 56k vocabulary [T Table 4], in-domain check set: KN5 104.7, SNM5 108.3, SNM5-skip 89.5, SNM10-skip 87.5, with cross-sentence skip features 78.4; RNNME-1024 68.0. The skip gain shrinks (17% vs 23%) and the RNN wins by 13%. So the lookup model loses ground to a neural model as data shrinks.

## Training recipe and gotchas
- theta initialised to 0; AdaGrad, accumulator 1, scale 0.02; one epoch over only 30M examples (1B corpus) or 10M (44M corpus) [T Sec. 5, 6].
- theta table 200M entries for the 1B model, 10M for the small one; larger gave "no significant improvements" [T].
- Estimating theta on held-out counts "did not yield good results"; leave-one-out is required [T Sec. 3.4].
- No per-template ablation: the paper never reports which (r,s,a) shapes carry the gain, and tied vs untied is not isolated.

## Mapping onto the GBA design
All [I].
- Runtime form (exp(A) and count ratio folded into the stored weight offline; theta never ships): for each active template, hash the context to a ROM bucket holding K (token id, 8-bit weight) pairs plus a row sum. With 16 templates and K=8: 16 hashes, 128 ROM reads and adds into a sparse candidate list (about 0.5 KB RAM). Probably under 10K cycles, well below 1% of budget. No multiplies; one normalising divide, or none if it is used as a logit bias.
- Additive probability mixing does not match our logit-space head. Options: store log-weights and add as sparse logit biases (then it is a learned output-side n-gram table, not SNM); or use the union of buckets as the candidate shortlist for the class/member head and for typing suggestions.
- ROM: the paper's models are 33-62B exact entries. At 3 bytes per entry, 16 MB holds about 5M entries, so heavy pruning or hashed buckets are mandatory; neither is tested in this paper.
- Template choice we can copy directly: dense short gaps (s=1) plus one-remote-word tied long gaps up to 10.

## Verdict
Test. Strong evidence that gapped contexts are worth about as much as contiguous ones (69.8 skip-only vs 70.8 5-gram) and stack with them (54.2), and that a hashed correction table can be tiny. Weak evidence at our ROM size and data scale. Settling experiment: on our corpus and 16K vocabulary, build a pruned top-K table with and without the SNM10-skip templates at fixed ROM bytes, and measure (a) stand-alone perplexity and top-3 typing hit rate, (b) held-out loss gain when added as a sparse logit bias to the transformer.

## Could not verify
- Pruning follow-up (Pelemans, Shazeer, Chelba, Interspeech 2015) seen only as a search-result summary: claims pruned SNM n-gram models have 18% lower perplexity than Katz and 57% lower than Kneser-Ney at equal size. Not read.
- No results with subword vocabularies, hashed M rows, or quantised weights exist in either version.
- GBA cycle and byte figures are my estimates.

## Links worth following
- Pruning SNM: https://www.isca-archive.org/interspeech_2015/pelemans15_interspeech.html
- Chelba and Pereira (2016), multinomial loss on held-out data for SNM: cited in [T]; URL not found.
