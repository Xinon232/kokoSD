# BabyHGRN: Exploring RNNs for Sample-Efficient Training of Language Models  (https://arxiv.org/abs/2412.15978, v1 20 Dec 2024; Haller, Golde, Akbik; Humboldt-Universität zu Berlin)

Tags: [P] primary text (arXiv HTML v1, whole paper and appendices); [P2] primary text of a followed link (HGRN2 paper 2404.07904v2; FLA `fla/layers/hgrn2.py`; released `config.json`); [I] my inference or arithmetic.

## What it is
A BabyLM-2024 entry built on HGRN2, a gated linear RNN whose state is a decayed sum of outer products [P]. One sweep compares five architectures at 300-360M parameters on 10M words; the final 1B model adds a Pile-derived corpus and same-size self-distillation [P]. The authors call it a "proof of concept" [P]. It has no HGRN2 equations, nothing below 300M parameters, no quantization, no seeds [P].

## Mechanism
HGRN2 layer: pre-RMSNorm mixer, then pre-RMSNorm SwiGLU MLP, residual, no positional encoding [P2]. Per token, with H = d/n heads:
```
q = swish(Wq x);  i = Wi x;  g = sigmoid(Wf x)       # three d x d, no bias
f = beta_l + (1 - beta_l) * g                        # layer 0: beta = 0
S[head] = S[head] * diag(f) + outer(i, 1 - f)        # n x n per head, d*n in total
y = Wo * RMSNorm(S q)                                # Wo is d x d
beta = cumsum(softmax(Gamma, over layers)) - beta_0  # Gamma: L x d, learned
```
The decay floor rises with depth; write key `1-f` is tied to the forget gate [P2]. Configurations: 360M = d 1024, 26 layers, hidden ratio 4, n = 128 [P, App. B]; final = d 2048, 18 layers, n = 128, vocab 16,000, untied head [P2]. About 12d^2 parameters per layer [I]. Transformer baseline: LLaMA-style, d 1024, 22 layers [P].
Distillation [P]: `L = (1-a)*CE + a*KL(softmax(z_teacher) || softmax(z_student))`; the teacher is the same architecture trained first. `a` and temperature are not given.

## Numbers
Table 2 [P]: BabyLM 10M words, best of 5 epochs, cross-entropy only, one run each.

| Model | Params | BLiMP | Supp. | EWoK | Macro |
|---|---|---|---|---|---|
| Transformer | 360M | 62.64 | 54.86 | 50.48 | 55.99 |
| LSTM | 300M | 62.27 | 51.63 | 50.48 | 54.79 |
| Mamba | 350M | 64.44 | 55.39 | 50.39 | 56.74 |
| xLSTM | 340M | 64.66 | 56.72 | 49.48 | 56.95 |
| HGRN2 | 360M | 67.05 | 55.69 | 49.88 | 57.54 |

The gap is all BLiMP (+4.4); EWoK is at chance everywhere [I].
- Corpus (Table 4) [P]: Pile vs BabyLM, BLiMP 67.49 vs 67.05 at 10M, 72.89 vs 69.44 at 100M.
- Final 1B model, 10M track (Table 5) [P], BLiMP / Supp. / EWoK / SuperGLUE / average: CE-only 69.4 / 55.6 / 50.7 / 63.0 / 59.7; distilled 72.1 / 58.6 / 51.3 / 65.8 / 63.3; BabyLlama 69.8 / 59.5 / 50.7 / 63.3 / 60.8.
- 100M track (Table 6) [P]: CE-only 74.5 / 59.1 / 52.88 / 69.1 / 63.9; distilled 77.5 / 58.5 / 51.6 / 70.7 / 64.9; LTG-BERT 69.2 / 66.5 / 51.9 / 68.4 / 64.0.
- [I] The distilled averages do not match their own columns: 61.95, not 63.3 (10M), and 64.58, not 64.9 (100M); the other six rows recompute exactly. Corrected, the 10M lead over BabyLlama is 1.1 points, not 2.5, and distillation adds 2.3, not the stated 5.3.
- HGRN2 paper [P2]: WikiText-103, 44M parameters, test perplexity HGRN2 23.73, Mamba 23.19, Transformer 24.78, HGRN1 24.82 (Table 5). Expand ratio, 385M, 10B Pile tokens, perplexity: n=1 5.10, n=4 4.79, n=8 4.73, n=128 4.62 = Transformer++ (Table 3).

## Smallest scale tested, and whether the result held there
300-360M parameters on 10M words: heavily over-parameterised, roughly 2,000 optimizer steps [I]. HGRN2 led there by 1.5 macro points on single runs [P]. HGRN2's own smallest LM is 44M [P2]. Nothing near our 1-5M computed weights per token.

## Training recipe and gotchas
- Adam, batch 64, sequence 512, clip 1.0, linear schedule; LR swept over {1e-3, 1e-4, 1e-5, 1e-6}; 1e-3 best for every architecture, 4e-4 at 1B; best epoch 2-5 [P]. 16K BPE; 512-token chunks [P]. FLA Triton kernels [P2].
- Confounds [P]: the final model differs from the baselines in size and corpus, and its fine-tuning used 20 epochs, patience 6, LR 1e-5 against the default 10, 3, 5e-5 (possibly for WSC only; the wording is ambiguous).
- Inconsistencies [P]: Section 2 says 330M for the 2048 x 18 model (about 990M [I]); Table 3 says 3 epochs, the text 5.

## Mapping onto the GBA design
All [I], d = 256.
- Projections dominate: Wq, Wf, Wi, Wo = 4d^2 = 262K ternary weights, about 420K cycles per layer, like attention's four. HGRN2 saves no matrix work.
- Update `S += (i - S) * (1 - f)` and readout `S q`: d*n multiplies each. n = 8: about 35K cycles; n = 128: about 500K. Use n = 4-16.
- RAM: 2*d*n bytes per layer at 16 bits: 4 KB at n = 8, 64 KB at n = 128. No ring buffer; context decays instead of ending.
- ROM: fold sigmoid and beta into a 256-entry table per layer and channel (393 KB for 6 layers). Layer-1 q, f, i depend only on the token: 3d bytes x 16K tokens = 12 MB.
- Precision: with f near 1 the increment underflows; 16-bit state reaches roughly 256 tokens, 32-bit further.
- Typing suggestions: snapshot and restore the state (24 KB at n = 8).

## Verdict
**Test, low priority.** A gated linear RNN is not worse than a transformer on BLiMP when data is scarce and parameters abundant; our regime is the reverse, and the evidence is single-run with arithmetic errors. Worth keeping: HGRN2's small-n result and the depth-increasing decay floor. Settling experiment: at our size with ternary QAT, put an n = 8 HGRN2 mixer in none, the lower half, or all layers at equal GBA cycles; compare held-out loss, BLiMP and a 300-token recall probe, three seeds.

## Could not verify
- Distillation weight and temperature; the training code (the README says the pipeline is "still WIP"; file listing was blocked).
- Figures 1 and 3 (images): only captions read.
- Official BabyLM 2024 leaderboard scores (could resolve the average mismatch): not checked.
- The 100M model card names dataset `PatrickHaller/dsir-pile-100M-words` [P2]; the paper describes naive random sampling.
- Nothing on ternary or integer HGRN2.

## Links worth following
- HGRN2: https://arxiv.org/abs/2404.07904 ; HGRN: https://arxiv.org/abs/2311.04823
- FLA code: https://github.com/fla-org/flash-linear-attention (`fla/layers/hgrn2.py`)
- https://github.com/HallerPatrick/BabyLM-2024 ; https://huggingface.co/PatrickHaller/hgrn2_pile_10m_distill_babylm
