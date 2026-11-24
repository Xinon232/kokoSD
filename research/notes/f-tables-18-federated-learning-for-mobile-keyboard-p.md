# Federated Learning for Mobile Keyboard Prediction (https://arxiv.org/abs/1811.03604, v2 Feb 2019; Hard, Rao, Mathews, Ramaswamy, Beaufays, Augenstein, Eichner, Kiddon, Ramage; Google)

Read in primary form: full text of the arXiv v2 PDF (all sections, Tables 1-6). No secondary sources used. Items marked [derived] are my arithmetic, not statements in the paper.

## What it is
A short production paper on Gboard's English next-word prediction. It compares the shipped word n-gram FST against a 1.4M-parameter single-layer CIFG LSTM with a 10,000-word vocabulary, trained either on a server (logged snippets) or with FederatedAveraging on phones. Both neural models beat the n-gram on top-1 and top-3 recall, offline and in live traffic. The federated part is irrelevant to us; the value is the metric definitions, the n-gram-vs-tiny-neural gap, and how low absolute next-word recall really is.

## Mechanism
- **Baseline**: Katz-smoothed, Bayesian-interpolated word 5-gram FST, 1.25M n-grams including 164,000 unigrams. Prediction = find the highest-order n-gram state matching the preceding text, return its n-best output labels; back-off paths to lower orders are also considered. Augmented by personal history, contacts and email n-gram models.
- **Neural model**: one CIFG layer (LSTM with coupled gates, `f_t = 1 - i_t`, no peepholes), 670 units; input embedding and "CIFG output projection" both D = 96; tied input/output embedding W (96 x 10,000); logits = W^T h, softmax, cross-entropy.
- Vocabulary V = 10,000 words including BOS, EOS, OOV; logits of these special tokens are ignored at evaluation and inference (never suggested).
- Size: 1.4M parameters, "more than two thirds" in W; 1.4 MB after weight quantization (so about 8 bits/weight [derived]); runs in TensorFlow Lite.
- [derived] Parameter check: W = 960,000; three gate blocks 3 x 670 x (96 + 96 + 1) = 387,930; projection 670 x 96 = 64,320; total about 1.41M. This only matches if the recurrent input is the 96-d projected state (LSTMP style); the paper does not say so explicitly.
- **Metrics**: recall = correct predictions / total tokens (top-1: the best candidate equals the word typed; top-3: any of three). Prediction impression recall (live) = predictions matching entered text / times candidates were shown; lower because zero-state impressions count. CTR = clicks on candidates / candidates proposed. 95% CIs by jackknife over user buckets.

## Numbers
All US English, V = 10K, 1.4M parameters.

| Table | Eval set | N-gram | Server CIFG | Federated CIFG |
|---|---|---|---|---|
| 3 | server logs, top-1 | 13.0% | 16.5% | 16.4% |
| 3 | server logs, top-3 | 22.1% | 27.1% | 27.0% |
| 4 | client caches, top-1 | 12.5 +- 0.2 | 15.0 +- 0.5 | 15.8 +- 0.3 |
| 5 | live impression recall, top-1 | 5.24 +- 0.02 | 5.76 +- 0.03 | 5.82 +- 0.03 |
| 5 | live impression recall, top-3 | 11.05 +- 0.03 | 13.63 +- 0.04 | 13.75 +- 0.03 |
| 6 | live CTR | 2.13 +- 0.03 | 2.36 +- 0.03 | 2.35 +- 0.03 |

- Neural over n-gram: +3.5 points top-1 and +5.0 top-3 on logs (about 27% and 23% relative [derived]); CTR about 10% higher (paper's wording).
- Data: server set about 7.5 billion sentences, mean length 4.1 words, test/eval 25,000 sentences each; 60% chat, 35% web input, 5% long form (Table 1). Client caches: 66% chat, 16% social, 5% web, 12% other (Table 2).
- Latency target stated: visible response within 20 ms of an input event (a requirement, not a measurement; no measured latency is given).

## Smallest scale tested, and whether the result held there
Only one model size (1.4M parameters, 670 units, D = 96). No ablation over width, vocabulary or n-gram order. The result at this single small scale is positive: a 1.4M-parameter recurrent model beats a 1.25M-entry 5-gram with a roughly 16x larger vocabulary and personalization.

## Training recipe and gotchas
- Server: asynchronous plain SGD, learning rate 1e-3, no momentum or weight decay, batch 50 sentences, converged after 150M steps; Adam and AdaGrad "not found to improve convergence". Snippets used only if they start with a sentence-start token.
- Federated: 100-500 clients per round; server update with Nesterov momentum 0.9, server learning rate 1.0; about 400 sentences per client per epoch; 3,000 rounds, 600M sentences, 1.5M clients, 4-5 days. Client learning rate, client batch size and epochs are not given.
- Gotchas: (1) recall is distribution-bound: the same model scores 16.5% on truncated logs and 15.0% on real caches; (2) absolute numbers are low because sentences average 4.1 words, so most predictions have 0-3 words of context; (3) OOV targets count as misses, capping recall; (4) the n-gram baseline was not retrained on the same data, so the gap is not a controlled comparison.

## Mapping onto the GBA design
- Cost of this exact model per token [derived]: recurrent part about 388K + 64K = 452K multiply-accumulates; full tied softmax 96 x 10,000 = 960K. At 7.5 cycles per 8-bit MAC that is about 10.6M cycles, over budget; as ternary at 1.6 cycles/weight about 2.3M cycles, inside budget but spending two thirds on the softmax. Our two-level class/member head removes exactly that dominant term.
- The n-gram baseline is essentially free for us: a hashed 5-gram to top-3 table is one or two ROM lookups and no arithmetic. 1.25M n-grams at, say, 8 bytes each (hash check + three 16-bit ids) is about 10 MB of ROM [derived]; back-off is a retry with a shorter hash.
- Suggestion = top-3 over the model's next-token distribution: with a two-level head, score all classes, expand members of the best 2-3 classes, keep a 3-entry insertion list (no sort, a few compares per logit). RAM: 3 ids plus scores.
- Changes needed: our vocabulary is 16K subword tokens, the paper's is words. A fair recall number needs word-level suggestions: either greedy/beam extension of up to 2-3 tokens per candidate (multiplies compute by the beam) or a ROM table mapping the first token to likely word completions. Mask special tokens as the paper does.
- Precomputable into ROM: the whole n-gram suggestion table; class membership lists; per-class top members as a zero-compute fallback shown instantly while the network computes.

## Verdict
**Adopt as the evaluation template, test as a target; skip the method** (federated learning is not applicable). Use top-1 and top-3 word recall with special tokens masked, and report the hashed-n-gram-only suggester as our own baseline. A realistic bar on chat-like text: n-gram about 13% / 22%, small neural about 16.5% / 27%. Settling experiment: on held-out simple-English chat turns, measure top-1/top-3 word recall for (a) ROM n-gram table alone, (b) Koko alone, (c) n-gram shown immediately and replaced by Koko; also break down by context length, since the paper's 4.1-word sentences will not match ours.

## Could not verify
- Measured on-device inference latency, perplexity, keystroke savings: not reported in the paper.
- Whether the recurrent state is projected to 96 dimensions: inferred from the parameter count only.
- Quantization bit width: inferred from 1.4 MB / 1.4M parameters.
- Top-3 recall on client caches: Table 4 gives top-1 only.
- The numbers are not comparable to GPT-2 small without re-running both on the same text; the paper has no transformer comparison.

## Links worth following
- Ouyang et al., "Mobile keyboard input decoding with finite-state transducers", arXiv 1704.03987 (the FST baseline's mechanics; cited as Ref. 3).
- Yu et al., "On-device neural language model based word prediction", COLING 2018 demos (cited as Ref. 14; reports keystroke savings and word prediction rate under latency/memory limits, per the paper's description).
- Greff et al., "LSTM: A search space odyssey" (CIFG definition; cited as Ref. 20).
