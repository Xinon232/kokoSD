# Federated Learning for Mobile Keyboard Prediction  (https://arxiv.org/abs/1811.03604, v2 Feb 2019; Hard, Rao, Mathews, Ramaswamy, Beaufays, Augenstein, Eichner, Kiddon, Ramage; Google)

Read in primary form: full text of the arXiv v2 PDF (7 pages incl. references). Lines marked [mine] are my arithmetic or inference, not the paper's.

## What it is
The paper describing Gboard's production English next-word predictor: a 1.4M-parameter single-layer CIFG-LSTM word model that replaced a 5-gram FST for the suggestion strip. It trains the same network two ways (server SGD on logged snippets; FederatedAveraging on phones) and compares both to the n-gram baseline. For us the federated part is irrelevant; the value is (a) a shipped, tiny architecture with exact sizes, (b) the metric definitions (top-1/top-3 recall, impression recall, CTR), and (c) the absolute recall numbers a real keyboard reaches, which calibrate what "good predictive typing" means.

## Mechanism
- Vocabulary V = 10,000 words (word-level, not subword), plus BOS, EOS, OOV input tokens. At evaluation and inference the logits of these special tokens are ignored (never suggested).
- Embedding W in R^(D x V), D = 96. Input d = W v. Output logits = W^T h, i.e. input embedding and output projection are tied (Press & Wolf; Inan et al.). Softmax + cross-entropy.
- One CIFG layer, 670 units, no peepholes, with a projection of the output to D = 96. CIFG = LSTM with f_t = 1 - i_t (Eq. 1), 25% fewer parameters per cell than LSTM.
- Total 1.4M parameters, "more than two thirds" in W. Shipped size 1.4 MB after weight quantization (so about 8 bits/weight [mine]); runs in TensorFlow Lite.
- [mine] Parameter reconstruction: W = 960,000; three gate blocks 3 x 670 x (96 + 96 + 1) = 387,930 assuming the recurrent input is the 96-d projection; projection 670 x 96 = 64,320; total 1,412,250, matching 1.4M. The paper does not state the recurrent wiring.
- Suggestion strip: top-1 goes in the centre slot, 2nd and 3rd left and right.
- Baseline: Katz-smoothed, Bayesian-interpolated 5-gram FST, 1.25M n-grams including 164,000 unigrams, plus personal history/contacts/email n-gram models. Prediction = n-best labels from the highest-order matching n-gram state, with back-off paths.
- FedAvg: client k does local SGD, server averages client weights by n_k/N (Eq. 2-3), applied through Nesterov momentum 0.9, server learning rate 1.0.

## Numbers
- Table 3 (server logs, 25,000-sentence test set): top-1 / top-3 recall: n-gram 13.0% / 22.1%; server CIFG 16.5% / 27.1%; federated CIFG 16.4% / 27.0%.
- Table 4 (client caches): top-1 recall n-gram 12.5 +- 0.2%; server CIFG 15.0 +- 0.5%; federated CIFG 15.8 +- 0.3%.
- Table 5 (live traffic, impression recall): top-1 / top-3: n-gram 5.24 / 11.05%; server CIFG 5.76 / 13.63%; federated CIFG 5.82 / 13.75% (errors +-0.02 to 0.04, 95% CI, jackknife over user buckets).
- Table 6 (live CTR): n-gram 2.13%; server CIFG 2.36%; federated CIFG 2.35% (+-0.03). Text: CIFG predictions are about 10% more likely to be clicked.
- Data: server run 7.5 billion sentences, mean length 4.1 words; 60% chat, 35% web input, 5% long-form (Table 1). Federated: 600M sentences from 1.5M clients over 3000 rounds, 4-5 days; caches 66% chat, 16% social, 5% web, 12% other (Table 2).
- Constraint quoted: visible keyboard response within 20 ms; models "tens of megabytes".

## Smallest scale tested, and whether the result held there
Only one model size (1.4M parameters) is reported; no ablation over width, vocabulary or data volume. It is already far below our scale in parameters, and it beat a 1.25M-n-gram 5-gram with personalization by 3.5 points top-1 and 5 points top-3 on logs. The data scale is the opposite: 7.5B sentences (about 31B words [mine]) is far beyond ours.

## Training recipe and gotchas
- Server: asynchronous plain SGD, learning rate 1e-3, no momentum or weight decay, batch 50 sentences, converged at 150M steps. Adam and AdaGrad "not found to improve convergence".
- Federated: 100-500 client updates per round, about 400 sentences per client epoch. Client learning rate, batch size and epoch count are not given.
- Recall = correct predictions / total tokens. Whether OOV or EOS targets stay in the denominator is not stated; this matters when comparing against a subword model.
- Impression recall counts zero-state impressions (keyboard opened, nothing typed), so it is much lower than offline recall. Do not compare our offline numbers with Table 5.
- These are next-WORD metrics with no typed prefix. Word completion (prefix given) is a different, much easier task that this paper does not measure.
- Server vs federated is not a clean comparison (different optimizers and data), as the authors say.

## Mapping onto the GBA design
- Running this exact model on the GBA [mine]: recurrent part 387,930 + 64,320 = about 452K weights/token; at 1.6 cycles/weight ternary = 0.72M cycles, at 7.5 cycles/weight 8-bit = 3.4M cycles. Full tied softmax 10,000 x 96 = 960K weights = 1.5M cycles ternary; a two-level head removes most of that. State RAM: 670 cells + 96 projection, under 2 KB at 16 bits. ROM: 1.4 MB at 8 bits. Sigmoid/tanh become 256-entry ROM tables. So a Gboard-class predictor fits in roughly half our per-token budget; our planned model is larger in dense compute and vastly larger in lookup capacity.
- Transferable: the tied, low-rank (D = 96) output bottleneck; masking special tokens from suggestions; top-3 presentation with the best candidate in the most visible slot.
- For Koko's typing suggestions the user's next-word distribution comes from the same forward pass that ingests the user's tokens, so top-3 costs only a top-k scan of the output head. With a 16K subword vocabulary we need word-level suggestions: either restrict candidates to tokens that are whole words, or do a short greedy expansion (extra forward passes, each a full token budget). A ROM table mapping token to "is whole word" is free.
- The n-gram baseline is itself implementable in ROM as hashed tables (1.25M n-grams is a few MB) and is the natural cheap fallback and comparison.

## Verdict
Adopt as the evaluation protocol and calibration point, not as architecture. Report top-1 and top-3 next-word recall over all word tokens on held-out simple-English chat text, for Koko, a 5-gram Katz baseline, and GPT-2 small, with jackknife CIs. Rough targets from this paper: about 16% top-1 and 27% top-3 on real chat typing, though our simpler, synthetic text should score higher, so the n-gram baseline on our own data is the real yardstick. Settling experiment: train a 1.4M CIFG and a 5-gram on our corpus and check that Koko beats both on top-3 recall at equal or lower cycles.

## Could not verify
- Inference latency, CPU cost and the quantization scheme of the shipped model are not in the paper (only the 20 ms target and 1.4 MB).
- Perplexity is never reported; keystroke-savings is not measured.
- Recurrent wiring of the projection and the handling of OOV targets in recall are not stated.
- Figures 3 and 4 (recall vs step/round) were not readable from extracted text.
- Nothing here is from secondary sources.

## Links worth following
- Ouyang et al., "Mobile keyboard input decoding with finite-state transducers", arXiv:1704.03987 (the n-gram FST baseline and completion/correction mechanics).
- Yu et al., "On-device neural language model based word prediction", COLING 2018 demos (keystroke savings and word prediction rate under memory limits; cited as ref. 14, not read).
- Ghosh & Kristensson, "Neural networks for text correction and completion in keyboard decoding", arXiv:1709.06429 (cited as ref. 15, not read).
