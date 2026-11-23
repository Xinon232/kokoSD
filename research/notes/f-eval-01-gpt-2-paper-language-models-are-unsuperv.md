# Language Models are Unsupervised Multitask Learners (GPT-2)

https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf — 2019, Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI). Read in full from the PDF [P]. Items marked [R] come from the official repo/model card, not the paper.

## What it is
Four decoder-only transformers (12/24/36/48 layers) trained on WebText (40 GB, 8M documents from Reddit outbound links with at least 3 karma, Wikipedia removed), evaluated with no fine-tuning. The smallest, called "117M" in the paper, is the model we benchmark against; its true size is 124M (the repo README says the paper's counts were wrong) [R]. Table 3 gives its zero-shot numbers on eight language-modeling sets. This is the source of the baseline, not a method to adopt.

## Mechanism
- Smallest model: 12 layers, d_model 768, context 1024, vocabulary 50,257 (Table 2, section 2.3) [P]. 12 heads, MLP 4x with GELU, dropout 0.1, layer-norm epsilon 1e-5 [R: HF config.json, openai/gpt-2 model.py].
- Pre-norm: layer norm at the input of each sub-block, plus one final layer norm after the last block.
- Init: residual-layer weights scaled by 1/sqrt(N), N = number of residual layers. Batch size 512.
- Tokenizer: byte-level BPE (base vocabulary 256), merges forbidden across character categories, except that a space may attach to the following word. Any string gets a probability, so no OOV.
- Scoring: sum log-probability of the test set, divide by the dataset's canonical unit count (words, bytes or characters). "Invertible de-tokenizers" undo PTB-style artifacts first; worth 2.5 to 5 perplexity for the 1542M model.
- CBT: score each of 10 candidates by the probability of the candidate plus the rest of the sentence; pick the highest.
- LAMBADA accuracy: with a stop-word filter on the predicted final word (52.66% to 63.24% for 1542M; unfiltered figure for the small model is not given).
- Sampling in the appendix: top-k, k = 40. Summarization used "TL;DR:" with k = 2.

## Numbers
Table 3, zero-shot, small model (117M/124M) versus 1542M:

| Set | Small | 1542M |
|---|---|---|
| LAMBADA perplexity | 35.13 | 8.63 |
| LAMBADA accuracy | 45.99 | 63.24 |
| CBT common nouns (validation) | 87.65 | 93.30 |
| CBT named entities (validation) | 83.4 | 89.05 |
| WikiText-2 perplexity | 29.41 | 18.34 |
| PTB perplexity | 65.85 | 35.76 |
| enwik8 bits/byte | 1.16 | 0.93 |
| text8 bits/char | 1.17 | 0.98 |
| WikiText-103 perplexity | 37.50 | 17.48 |
| 1BW perplexity | 75.20 | 42.16 |

- Natural Questions exact match: 1542M 4.1%; the smallest model "does not exceed" a 1.0% most-common-answer baseline (section 3.8).
- CoQA 55 F1, Winograd 70.70%, CNN/DM ROUGE avg 21.40 (random-3 baseline 20.98), WMT14 en-fr 5 BLEU / fr-en 11.5: all 1542M only; small-model values appear only as plot points (Figure 3) or not at all.
- Overlap of test 8-grams with WebText train: 1-6%, mean 3.2% (Table 6); benchmarks' own train sets average 5.9%. WikiText-103 test shares an article with its train set.
- All four models still underfit WebText (Figure 4).

## Smallest scale tested, and whether the result held there
124M is the smallest. Language modeling and cloze transfer hold there (it already beat prior state of the art on LAMBADA perplexity, CBT, WikiText-2). Stored factual knowledge does not: at most 1% on Natural Questions. The paper says models "only begin to outperform trivial baselines when they have sufficient capacity."

## Training recipe and gotchas
- The paper gives no optimizer, learning-rate schedule, step count, or hardware; only "learning rate manually tuned on a 5% held-out sample."
- De-tokenizer code is not in the paper, so Table 3 is not exactly reproducible; later reproductions (lm-eval) get different values. Rerun GPT-2 under our own harness.
- Perplexities are per word of the original dataset, not per BPE token; a per-token perplexity from our 16K vocabulary is not comparable. Use bits per byte.
- CBT is reported on validation because The Jungle Book (test) is in WebText.

## Mapping onto the GBA design
Nothing here runs on the GBA; the value is the target and the eval protocol. My arithmetic from the stated shapes (not from the paper):
- GPT-2 small per token with a key/value cache: 12 x (4 + 8) x 768^2 = about 85M multiply-accumulates in the blocks, plus 50,257 x 768 = 38.6M for the output layer; about 124M total. Koko touches 1.3-2M ternary weights, roughly 60-90x fewer, and about 9-10 bits less per weight.
- GPT-2's 85M block parameters are comparable to what Koko can STORE (50-120M ternary), so parity depends on routing/lookup capacity actually being used.
- Shared ideas already in the design: pre-norm, final norm, 1/sqrt(N) residual scaling (apply to the latent weights in quantization-aware training).
- Tokenizer rule (no merges across character classes, space-prefix words) costs nothing at run time and suits a 16K vocabulary; adopt.
- Bloom-filter 8-gram overlap check (lower-cased alphanumeric words) is cheap to run offline between our synthetic training data and test sets.
- Realistic claim: beat GPT-2 small on in-domain simple English (CBT-style cloze, simple dialogue, bits per byte on simple text); do not expect to match WikiText-103 37.5 or enwik8 1.16 open-domain.

## Verdict
Adopt as the baseline definition and evaluation protocol; skip as architecture guidance. Settling experiment: run GPT-2 small and Koko through one harness reporting bits per byte on (a) held-out simple-English/dialogue text, (b) WikiText-103 test raw, (c) CBT validation CN/NE with the 10-candidate rule; reproduce Table 3's small-model row within a stated tolerance first.

## Could not verify
- Optimizer, schedule, steps, training tokens seen: not in the paper.
- Head count, MLP ratio, GELU, dropout: from repo/config [R], not the paper.
- Small-model Winograd, CoQA, summarization, translation numbers: not tabulated.
- WebText perplexity per model (Figure 4): plot only, no numbers read.
- No BLiMP, HellaSwag, or dialogue results in this paper.

## Links worth following
- https://github.com/openai/gpt-2 (README parameter-count correction; model.py, encoder.py)
- https://huggingface.co/openai-community/gpt2 (model card repeats Table 3 small-model row)
- Children's Book Test: Hill et al. 2015, arXiv:1511.02301 (simple-English cloze where the small model scores 87.65/83.4)
