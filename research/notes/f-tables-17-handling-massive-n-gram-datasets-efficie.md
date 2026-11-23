# Handling Massive N-Gram Datasets Efficiently  (https://arxiv.org/abs/1806.09447, v2 2020; ACM TOIS 2019; G. E. Pibiri and R. Venturini; code https://github.com/jermp/tongrams)

Read in primary form (arXiv PDF v2; sections 3, 4, 6 in full, section 5 skimmed). Tags: [P] paper, [R] tongrams README, [mine] my estimate.

## What it is
An exact, lossless, static n-gram index: a sorted-array trie whose levels are Elias-Fano (EF) coded integer sequences, plus "context-based identifier remapping" (a word ID is replaced by its rank among the successors of its preceding k words). Also a minimal-perfect-hash (MPH) variant and a faster Kneser-Ney estimator. On 0.1-11 billion word n-grams the keys take 1.28-1.38 bytes/gram, smaller than lossy RandLM and about 2.4x smaller than KenLM's trie at similar speed.

## Mechanism
- Word -> ID by MPH; IDs assigned in decreasing order of occurrence in the gram-ID sequences [P 4.1.1].
- Level n holds `ids` (last-word ID of each n-gram, sorted within each parent's range), `pointers` (start of each node's child range in level n+1, plus a dummy), `indexes` (rank of the value in a per-order array C[n] of distinct values).
- `ids` is made monotone by adding the last value of the previous range to each range, then EF-coded: at most m*ceil(log2(u/m)) + 2m bits for m integers below u, constant-time access [P]. (Background, not in the paper: low floor(log2(u/m)) bits verbatim, high bits unary; access(i) = (select1(i) - i) << l | low[i].)
- Partitioned EF with uniform blocks: 64 integers for bigrams, 128 for higher orders; block upper bounds uncompressed; 3-7% above optimal partitioning [P 4.3.1].
- Lookup [P Fig. 4]: p = id(w1); for k = 1..n-1: (b, e) = pointers_k[p], pointers_k[p+1]; p = binary search of id(w_{k+1}) in ids_{k+1}[b, e). Return C[n][indexes_n[p]].
- Remapping, context k = 1 or 2 [P 4.1.2]: levels above k+1 store the position of w among the children of its preceding k words, found by searching the first k+1 levels. No extra structure, only extra searches.
- Values: count ranks as variable-length codewords (rank i -> i + 2 - 2^len, len = floor(log2(i+2))) with a select bitvector, or prefix sums + PEF. Probabilities and backoffs: binning quantization, 8+8 bits in the benchmark.
- MPH variant: emphf (2.61 bits/key) + 8-byte or 4-byte fingerprint + value rank.

## Numbers
Datasets [P Table 1], word 5-gram sets: Europarl 101.4M grams, YahooV2 828.2M, GoogleV2 11.13B. Xeon 2.4 GHz, one core.
- Keys + pointers, bytes/gram (Europarl / YahooV2 / GoogleV2) [P Tables 2, 4]: EF 1.97 / 2.17 / 2.13; PEF-Trie 1.87 / 1.91 / 1.52 (1.35-1.91 usec/query); PEF-RTrie 1.28 / 1.38 / 1.31 (1.64-2.30 usec).
- Competitors, keys only [P Table 4]: KenLM trie 2.99 / 3.44; BerkeleyLM compressed 1.70 / 1.69 / 1.45; RandLM (lossy, 1/256 false positives) 1.81 / 2.02 / 2.60; MPH with 8-byte fingerprints 8.33 (0.26-0.37 usec); KenLM probing 9.40-16.91.
- Counts [P Table 3]: 0.30-0.36 bytes/count Europarl, 1.30-1.46 GoogleV2. Whole GoogleV2 index = 1.31 + 1.30 = about 2.6 bytes/gram [mine: the source of the "2.6" figure].
- Kneser-Ney 5-gram, 8-bit prob + 8-bit backoff, total bytes/gram [P Table 5]: PEF-RTrie 2.91 / 3.06 (Europarl / YahooV2); KenLM trie 4.58 / 5.04; MPH 9.92 / 9.94.

## Smallest scale tested, and whether the result held there
Paper: 101M grams. README micro-benchmark on 252,550 5-gram-order n-grams: 2.40 bytes/gram EF-Trie, 1.93 with k=1, 1.75 with PEF + k=2, counts included [R]. So it holds at a quarter-million grams, about 10% worse than Europarl's 1.58 total. Nothing on subword vocabularies or 1-10M grams.

## Training recipe and gotchas
No training; offline sort and encode. Remapping requires every (k+1)-word sub-path of a stored gram to be stored too (true for sliding-window extraction, broken by arbitrary pruning). Speeds assume popcount/ctz. Timings are cache-miss dominated and do not transfer to GBA.

## Mapping onto the GBA design
- ROM [mine]: 10 MB holds about 3.3M grams with prob + backoff (3 bytes/gram) or about 5M with one 4-8-bit value. With a 16K token vocabulary the unremapped gap is already small, so remapping should gain less than for words; measure on our tokens.
- Price of exactness: 1.3-1.9 bytes/gram of key here; MPH with b-bit fingerprint 0.33 + b/8; blind hashing 0.
- The trie enumerates a context's successors (pointer pair, then sequential decode); hashes cannot. That is the candidate-list and predictive-typing query.
- Compute [mine]: no multiplies. One EF access = a select over a ROM bitvector (byte-table popcount) plus a bit-field read, perhaps 100-300 cycles; a 3-token context lookup is about 20 accesses, a few thousand cycles, under 0.3% of the token budget. RAM: a few words.

## Verdict
Test. Use this layout (sorted successor ranges, frequency-ranked IDs, rank-coded values, optional k=1 remapping) for the exact output-side candidate / typing table; keep blind hashing for input-side embeddings where collisions are trained through. Experiment: build tongrams on our tokenized corpus pruned to 3-5M 1-4-grams, record bytes/gram, and compare held-out top-8 next-token recall per ROM megabyte against a blind-hashed table of equal size.

## Could not verify
- "2.6 bytes per gram" appears neither in this paper nor in the current README; it matches 1.31 + 1.30 on GoogleV2 (probably the SIGIR 2017 abstract, not opened).
- No comparison with Guthrie and Hepple; RandLM is the only lossy competitor.
- All GBA cycle and token-level size figures are mine.

## Links worth following
- https://github.com/jermp/tongrams_estimation ; Rust port https://github.com/kampersanda/tongrams-rs
- Ottaviano and Venturini, Partitioned Elias-Fano Indexes (SIGIR 2014)
