# Subword Language Model for Query Auto-Completion

https://arxiv.org/abs/1909.00599 (v1, Sep 2019; EMNLP-IJCNLP 2019; Gyuwan Kim, Clova AI / NAVER). Code: https://github.com/clovaai/subword-qac (MIT).
Read in primary form: full PDF text, plus `generate.py`, `dataset.py`, `tokenization.py` from the repository. Nothing below is from a secondary summary.

## What it is

A short paper on completing a typed prefix with a subword LM when the prefix ends mid-token. "Tokenize the prefix, then beam-search" fails with deterministic BPE, because the forced segmentation of the prefix rarely matches how the full string was segmented in training. Two fixes: **retrace** (back up a few characters, force the next token to cover them) and **reranking by approximate marginalisation** (sum beam entries that spell the same string). It also defines **mean recoverable length (MRL)**, a metric needing no prefix sampling.

## Mechanism

p(q) = sum of p(t) over segmentations t with concat(t) = q. Training uses one segmentation per query (BPE: deterministic; SR: sampled, SentencePiece l = inf, alpha = 0.2).

Retrace (paper Sec. 4.1; code `gen_collate_fn`):
```
ext[s] = ids of all tokens that have s as a PROPER prefix   # precomputed from vocab
rows = [(prefix, mask=None)]                                # R0: ordinary decoding
for r in 1..L (code: suffix lengths 1..retrace+1):
    inc = last r chars of prefix
    if inc in ext: rows += [(prefix[:-r], allowed = ext[inc])]
for each row: h, logp0 = LM(tokenize(row.prefix))           # logp0 = log p of the prefix tokens
              beam search from h, start score logp0,
              first step only: logits of tokens not in `allowed` set to -inf
pool all rows' finished candidates (as strings)
```
Every beam starts from the log-probability of its own prefix tokens, so scores are comparable across rows. Each row is a separate beam search.

Marginalisation (+M): log-sum-exp the scores of identical strings, re-sort, return top N. Without +M: keep the best copy.

Beam search (code): B = 30, branching factor 30, N = 10; stop when the best live score is below the worst finished score.

MRL: RL(q) = largest l such that q is in the top-N candidates for prefix q[1:|q|-k] for every k = 1..l; MRL = mean RL. Rank is ignored. PMRR: reciprocal rank of the first candidate that equals the query or is a whole-word prefix of it.

## Numbers

All from Table 1; AOL query log, 1,317,632 test queries (670,810 seen / 646,822 unseen in training); 1-layer LSTM, hidden 600, vocab 256.

| Model | MRR all | PMRR all | MRL all | CPU QPS | Decode len |
|---|---|---|---|---|---|
| MPC (trie of past queries) | .290 | .360 | 4.10 | >100 | n/a |
| Char | .311 | .464 | 5.02 | 11.0 | 14.5 |
| BPE, no retrace | .164 | .269 | 0.51 | 24.2 | 7.1 |
| BPE+R1 | .294 | .444 | 4.64 | 15.8 | 11.8 |
| BPE+R2 | .296 | .446 | 4.74 | 15.5 | 12.2 |
| BPE+R-inf | .296 | .446 | 4.76 | 15.3 | 12.2 |
| SR | .288 | .461 | 4.48 | 20.8 | 6.8 |
| SR+R-inf+M | .291 | .458 | 4.54 | 16.2 | 7.4 |

- Seen/unseen MRR: MPC .570/.000; Char .458/.160; BPE+R-inf .431/.157.
- +M alone on SR: MRR .288 -> .289, PMRR .461 -> .455 (tiny, mixed).
- n-best prefix segmentations for SR were worse than 1-best (no numbers given).
- "Up to 2.5x faster" refers to GPU QPS of SR without retrace (40.1 vs 16.5).

## Smallest scale tested, and whether the result held there

One scale only: a single-layer LSTM (hidden 600, embedding 100, tied; roughly 1-2M parameters by my arithmetic, not stated), 256-token vocabulary, 17.5M short queries. Already tiny, and the retrace effect is very large (MRL 0.51 -> 4.64). Not tested on transformers, sentence text, or vocabularies near 16K.

## Training recipe and gotchas

- Adam, lr 5e-3, batch 1024, 30 epochs, recurrent dropout 0.25, queries truncated to 40 characters; lowercased ASCII.
- Stated without numbers: BPE accuracy "degrades fast as the vocabulary size increases"; SR stays stable. This is a warning for a 16K vocabulary.
- R2 matches R-inf only because most of the 256 subwords are 3 characters or shorter. Longer tokens need deeper retrace.
- The code's `--retrace R` appears to cover suffixes of length 1..R+1 (my reading).
- A retraced row tokenizes `prefix[:-r]` normally; that can still differ from the segmentation inside the full word. Retrace reduces the mismatch, it does not remove it.

## Mapping onto the GBA design

- Koko suggests words, so decoding stops at the next word boundary (1-3 tokens). Natural policy: retrace to the start of the word being typed, with committed context ending at the last space.
- `ext` becomes a ROM table: sort the 16K vocabulary strings; tokens extending a string form one contiguous range, found by binary search (about 14 probes) or a trie. Roughly 150-250 KB of ROM (my estimate). No multiplies.
- Each row costs forward passes for the partial word's tokens plus one constrained output step. A word with w typed characters has at most w rows; states can be cached per split point, so each keystroke adds about one row: roughly 1-3 token evaluations (3-5M cycles each) per keystroke, fine for d-pad typing.
- RAM: uncommitted positions need per-layer keys/values held outside the ring buffer; a few positions of layers x width bytes, fits in slow RAM.
- The constrained step interacts with the two-level head: allowed tokens span several classes, and p(token) = p(class) x p(member | class) needs a member normaliser per class touched. Options: visit only the top-k classes, or build classes so tokens sharing a prefix share a class. The paper used a flat 256-way softmax.
- Marginalisation merges a handful of strings; log-add can be a small table, or simply max.
- Beam 30 is unaffordable; beam 2-4 is realistic. The paper has no beam-size ablation.
- MRL (or a per-word variant) is computed offline; the paper notes it suits one-word-at-a-time suggestion.

## Verdict

**Adopt** retrace and MRL; **test** marginalisation. Retrace is cheap and table-driven, and the no-retrace failure of BPE is too large to ignore. Settling experiment (PC, trained Koko model): top-3 word-completion accuracy and per-word recoverable length with (a) no retrace, (b) retrace to word start, (c) plus merge, at beam 1, 2, 4 and class cap 1, 2, 4.

## Could not verify

- No numbers for the vocabulary-size or n-best experiments, and no beam-size ablation.
- Parameter count is not stated.
- Transfer to a 16K vocabulary, a transformer, or sentence text is untested.
- ROM, RAM and cycle figures in the mapping section are my estimates.

## Links worth following

- Park and Chiba 2017, "A Neural Language Model for Query Auto-Completion" (character baseline, PMRR).
- Kudo 2018, "Subword Regularization" (https://arxiv.org/abs/1804.10959): sampled segmentations, n-best decoding.
- Later work on token-boundary bias and "token healing" in LLMs; not read for this note.
