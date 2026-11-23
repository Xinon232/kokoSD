# SuperBPE: Space Travel for Language Models  (https://arxiv.org/abs/2503.13423, v3 Aug 2025, COLM 2025; Liu, Hayase, Hofmann, Oh, Smith, Choi; UW / NVIDIA / AI2)

Read: arXiv HTML v3, main text and appendices A-D (figures as captions only; Table 4 not rendered). Also read: repo README (github.com/PythonNut/superbpe) and Păpușoi & Nisioi, BabyLM 2025 (aclanthology.org/2025.babylm-main.16, PDF text). Tags: [P] paper, [R] README, [B] BabyLM paper, [I] my inference.

## What it is
A two-stage BPE trainer. Stage 1 is ordinary BPE inside whitespace-delimited words; stage 2 continues merging with the whitespace barrier removed, so later tokens are "superwords" (" by the way", " of the"). At a 200k vocabulary it encodes text in up to 33% fewer tokens, and 8B models trained on it score +4.0 points over BPE across 30 tasks while using 27% less inference compute [P]. All of the paper's evidence is at 680M-11B parameters and 200k vocabulary.

## Mechanism
```
T = final vocab, t = transition point (t=T is plain BPE; t=0 never pretokenizes)
Stage 1: split on whitespace (space attached to word start); BPE merges until |V| = t
Stage 2: resume from the same merge list, no whitespace split; merge until |V| = T
  reject a merge if the new token spans > 4 words        [P A.1.2]
  reject a merge if the token contains ":" + space       [P A.1.4]
Both stages: digits pre-split in groups of 3 from the right; no GPT-2 contraction rule
```
- Encoding is ordinary BPE merge-rank application over the whole string [I from P/R].
- t=0 is bad: early merges glue the end of one word to the start of the next [P 2.3].
- Tested t = 80k/160k/180k of 200k; t=80k compresses best, t=180k (90% subword, 10% superword) scores best [P].
- Tokenizer data: 10 GB sample; more than ~10 MB does not change efficiency. Longest 1% of documents truncated; deduplication matters [P A.1].
- Code: fork of HF `tokenizers`; `scripts/train_tokenizer.sh`, then `extend_tokenizer.sh`; `extend_existing_tokenizer.sh` adds superwords to an existing BPE. MIT. Released tokenizer: 128k [R].

## Numbers
Bytes per token (Fig. 1, sec. 2.3) [P]:
- BPE plateaus near 50k; 4.45 at 200k; cannot exceed 4.68 (mean word length).
- SuperBPE passes 4.68 at ~12k, 5.55 at 50k, 6.63 at 200k (t=80k); 6.33 (t=160k), 6.09 (t=180k).

8B models, equal training FLOPs, context fixed in bytes (Tables 1, 2, 5) [P]:

|model|ctx tokens|10^9 FLOPs/byte|30-task avg|
|---|---|---|---|
|BPE|4096|3.75|39.8|
|SuperBPE t=80k|2756|2.42 (-35%)|42.9|
|SuperBPE t=160k|2884|2.54|43.4|
|SuperBPE t=180k|3000|2.65 (-27%)|43.8|
|SuperBPE 11B t=180k|3000|3.75|43.8|

- Wins 25/30; multiple-choice +9.7 average; MMLU 36.5 to 44.7. Losses: LAMBADA 77.0 to 70.6, HumanEval 15.9 to 13.4.
- Bits per byte barely differ: 0.7465 BPE, 0.7482 SuperBPE 8B, 0.7445 11B (Fig. 7). So the gain is not better compression.
- Per-token loss is more uniform: fewer very easy and very hard tokens (Fig. 4).
- Vocabulary needed to cover 99% of data: 57% BPE, 70% t=180k, 90% t=80k (C.2). Flatter Zipf curve.
- Cost model (B.4): with efficiency ratio α (up to 1.49), non-attention compute and KV memory fall by 1/α, attention by 1/α².

## Smallest scale tested, and whether the result held there
- In the paper: 680M and 1.9B, bits-per-byte only, no values printed (Fig. 5/8). SuperBPE at equal parameters is worse than BPE except when under-trained; SuperBPE at equal inference compute (bigger model) is almost always best [P B.4.3].
- Outside the paper [B], 10M words, single runs:
  - 39.2M LSTM, BLiMP: 4k 0.657 vs BPE 0.644; 8k 0.661 vs 0.640; 16k 0.613 vs 0.607 (Unigram 0.646).
  - BLiMP-Supplement at 8k favours BPE (0.581 vs 0.553).
  - GPT-BERT 8k: 0.787 vs BPE 0.794.
- So at our scale the evidence is mixed and within noise. Nothing below 16k is measured for efficiency beyond the "~12k" remark.

## Training recipe and gotchas
- Fix context in bytes and compare in bits per byte, never per-token perplexity.
- Stage 2 needs far more RAM and CPU (hours on 100 CPUs at 200k); trivial at 16k [I].
- Prompt boundary (C.3): a prompt ending mid-superword is tokenized unlike training text. Chat turns separated by special tokens are unaffected. Fixes named but not compared: merge dropout in training, token healing at inference.
- Synthetic chat data is repetitive; without the 4-word cap and deduplication, stage 2 will spend merges on boilerplate [I from A.1].

## Mapping onto the GBA design
- Model compute: zero change per token. The gain is characters per token. If 16k SuperBPE gives 1.1-1.25x bytes per token [I; not measured], we get that much more text per second, or the same text speed with that much more compute per token (the "matched inference" regime the paper favours).
- The 256-token ring buffer and hashed n-gram keys cover proportionally more text. N-gram counts get sparser [I].
- ROM: token strings (~16k x ~8 bytes) plus a merge-rank hash table, about 0.3 MB [I]. RAM: none beyond the input line. On-device encoding of user input is one merge loop per typed line.
- Output head: flatter frequencies mean classes are more balanced; fewer near-free tokens such as " the", so each step is harder for a tiny model [I].
- Predictive typing: superwords give multi-word suggestions from one step, but the user's partial text hits the C.3 boundary problem. Needs token healing: drop the last 1-2 tokens and constrain candidates by byte prefix through a ROM trie [I].
- Simple English is formulaic, so superwords should be common, but the 16k budget competes with word coverage.

## Verdict
Test. It costs nothing at inference and little to try. The headline gains are from 8B models at 200k; the small-scale evidence is mixed. Experiment: train 16k tokenizers at t = 16k, 14.4k, 12k, 8k on our corpus; measure bytes per token and unknown-word fallback rate; then train the same small model on each for equal steps and compare bits per byte and BLiMP. Adopt if bytes per token rises 10% or more without a bits-per-byte loss.

## Could not verify
- Bytes per token for BPE or SuperBPE at 16k, and which t Fig. 1 uses below 200k.
- Numeric values for the 680M/1.9B runs; Table 4.
- Exact stage-2 regex and script arguments (README excerpt incomplete).
- Whether [B] results are seed-averaged.

## Links worth following
- https://github.com/PythonNut/superbpe
- https://huggingface.co/alisawuffles/superbpe-tokenizer-128k
- https://aclanthology.org/2025.babylm-main.16.pdf
