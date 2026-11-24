# Understanding Transformers via N-gram Statistics (https://arxiv.org/abs/2407.12034, NeurIPS 2024, Timothy Nguyen, Google DeepMind)

Read in primary form: arXiv HTML v2, plus the README and the first 100K characters of `load_data.ipynb` in google-deepmind/transformer_ngrams. Reading was through a fetch tool that paraphrases, so table numbers are as it transcribed them; captions are paraphrased.

## What it is
A descriptive study: for each next-token prediction of a trained transformer, find the simple count-based rule (an n-gram statistic over a template of the last M <= 7 tokens) whose distribution is closest to the model's. It reports how often the best rule's top-1 token equals the model's top-1, on TinyStories (160M model) and Wikipedia (1.4B model). It is not a model you can deploy: the rule is chosen per context, after seeing the transformer's output. For us it is evidence about which context keys carry predictive information.

## Mechanism
- Template sigma: a string over {+, -, *} of length <= M applied to the last tokens. `+` keeps the token, `-` deletes it (the remaining tokens are treated as adjacent), `*` replaces it with a wildcard (position kept, token marginalised).
- Rule distribution: R_sigma(t | C) = count(alpha t) / count(alpha *), alpha = the templated context. No smoothing.
- Rulesets: suffix (only `+`, i.e. plain n-grams of every order <= M), subgram (`+`,`-`), all (`+`,`-`,`*`). Distinct rules in "all" for M = 1..7: 2, 5, 13, 34, 89, 233, 378 (footnote 9).
- Backoff baseline: the longest suffix (<= M) with non-zero training count, no backoff weights.
- Optimal rule = argmin over the ruleset of distance to the model's distribution, averaged over 5 training runs; distance is variational, d(p,q) = 1/2 sum |p - q|, or L-infinity.
- Important for us: a position-aware hashed key such as hash(t-3, t-1) is the `+*+` rule (the "all" set). The "subgram" `+-+` rule is something else: the plain bigram table queried with (t-3, t-1) as if adjacent.

## Numbers
TinyStories, 160M model, ~22K tokens from 100 validation stories, top-1 agreement of optimal rule with the model (Table 2, variational selection):

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| all | 30.0 | 44.8 | 54.3 | 62.4 | 68.8 | 74.0 | 78.0 |
| subgram | 30.0 | 44.5 | 53.1 | 60.0 | 64.8 | 68.5 | 71.1 |
| suffix | 30.0 | 44.4 | 52.2 | 57.9 | 61.5 | 63.8 | 65.6 |
| backoff | 30.0 | 42.5 | 48.7 | 52.7 | 54.6 | 55.9 | 56.7 |

- L-infinity selection at M=7 (Table 5): all 78.9, subgram 71.2, suffix 65.2, backoff 56.7. (The prelim figures 65.2/71.2/78.9 are from this table.)
- Mean variational distance at M=7 (Table 4): all 0.274, subgram 0.335, suffix 0.381, backoff 0.466; at M=1 all are 0.738.
- The 160M model's own top-1 accuracy against ground truth: 69.6%.
- Wikipedia, 1.4B model, 10 holdout chunks of 2048 tokens, M=7 (Table 9, variational): all 65.0, subgram 58.3, suffix 52.8, backoff 43.9; L-infinity (Table 7): all 67.7. Backoff peaks at M=4 (43.9) and does not improve after. Model top-1 vs ground truth: 54.9%.
- Figure 2 fits (TinyStories 7-grams): optimal-rule distance vs variance across 5 seeds, slope 1.47, R^2 0.74; variance vs count, R^2 0.11. Contexts where seeds agree are the rule-like ones, even when rare.

Reading of the table: at M <= 3 the template family barely matters (54.3 vs 52.2). The gaps open from M = 5. Nine points at M=7 (65.6 vs 56.7) come purely from choosing the right suffix length instead of the longest match.

## Smallest scale tested, and whether the result held there
Smallest model is 160M parameters (12 layers, width 896, 16 heads) on 480M TinyStories tokens with a 32K vocabulary. Nothing near our size. Larger models (420M, 1.4B) changed rule approximability only modestly, with mixed direction (Tables 11-20). Whether a 5-10M-effective-weight ternary model is more or less rule-like is untested.

## Training recipe and gotchas
- Chinchilla-style models; Adam with weight decay 1e-4; 1000-step warmup to 2e-4, cosine decay; batch 128 x 2048 tokens; 4 epochs on TinyStories.
- SentencePiece BPE; paper says vocabulary 32,678, the notebook says 32768 (paper typo, presumably).
- Counts are computed after chunking into 2048-token sequences; different chunking changes the statistics.
- Curriculum finding: early in training predictions approach short-context rules, later they leave them and keep approaching longer-context ones.
- Overfitting diagnostic: when a model overfits, its loss on inputs truncated to <= 7 tokens rises on the training set too, so short-context training loss can serve as an early-stopping signal without a holdout. No numeric values given (Figure 4).

## Mapping onto the GBA design
- Supports hashed context embeddings summed into the residual, with keys that include gapped, position-aware templates, not only contiguous suffixes. Per key: one hash of 2-4 token ids (shifts, xors, one multiply), one ROM row read, d adds. At d = 256 that is about 256 adds, well under 0.1% of the per-token budget even with 8-12 keys. RAM: only the last 7 token ids.
- ROM is the constraint: a 2^16-row x 256-wide ternary table is 16.8M weights, so 50-120M stored weights buys roughly 3-7 such tables in total. 378 templates cannot each have a table; share tables across templates by salting the hash with the template id, or use narrower rows.
- Do not use longest-match selection. Sum all orders and let training weight them, or gate by a stored count/confidence bit; the suffix-vs-backoff gap says the selector matters as much as the key set.
- The same tables serve predictive typing: they need no dense compute.
- Our vocabulary is 16K, not 32K; the released counts cannot be reused directly, but the template ranking experiment is cheap to rerun on our own tokenization.

## Verdict
Test. The paper is strong motivation but an upper bound with an oracle selector, measured against a 160M model, not against ground truth for a fixed rule set. Settling experiment: on our corpus and tokenizer, train the small model with (a) suffix keys of orders 2-4 only, (b) the same ROM budget split across suffix keys plus 3-4 gapped keys reaching back 5-7 tokens (e.g. (t-1,t-3), (t-1,t-2,t-4), (t-1,t-5)); compare validation loss and top-1. A count-only pre-study (greedy selection of the fixed template subset that maximises ground-truth top-1) costs CPU hours and picks the keys.

## Could not verify
- No table of how often each template is chosen as optimal; the paper has none (only per-story figures), so it does not tell us which gapped keys to pick.
- Rule accuracy against ground truth (Table 1) was not extracted; distinct n-gram counts and table sizes are not reported.
- Fraction of unseen 7-gram contexts is not reported.
- The notebook lists 1094 rule ids versus 378 distinct rules in the paper; the notebook only says there are redundancies. Parquet schema and file sizes not seen; last 62K characters of the notebook not read.
- Whether the short-context losses in the overfitting experiment come from truncating inputs at evaluation or from separately trained models is not stated clearly.

## Links worth following
- https://github.com/google-deepmind/transformer_ngrams (data in `gs://transformer-ngrams`: tokenizer `32768.model`, `TinyStories/`, `Wikipedia/`; Apache 2.0 code, CC-BY 4.0 data)
- Brants et al. 2007, "Large Language Models in Machine Translation" (stupid backoff), cited as the backoff baseline
