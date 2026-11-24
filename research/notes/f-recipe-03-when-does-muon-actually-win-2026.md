# When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark

https://arxiv.org/abs/2610.21555 (v1, 15 Oct 2026; OPT2026 workshop paper, 15 pp). Authors: Mushir Akhtar, Hazem N. Nounou, Majdi Mansouri, Mohamed N. Nounou. Read in full from the PDF (the arXiv HTML version returned 404). Everything below is [P] primary unless marked [I] (my inference).

## What it is
A controlled AdamW-vs-Muon comparison on one A100 with small models: two GPT-style LMs (17.7M and 33.6M params, widths 256/384, which is exactly our width range) and a 2.70M ViT on CIFAR-100 (balanced and 100:1 long-tailed). Each optimizer gets the same 23-candidate tuning pool; the winners are run with 5 paired seeds and tested once on a frozen test set. It reports three budgets separately: equal data, equal wall-clock, and limited tuning. Muon wins on data everywhere, loses on wall-clock for the LMs (reference kernels), and is more tuning-sensitive.

## Mechanism
- Muon on hidden matrices only (attention projections incl. fused QKV, MLP up/down). Embeddings, norms, biases, output head: AdamW ("auxiliary"). Baseline: AdamW on everything.
- Update: Nesterov momentum M_t of the matrix gradient, Frobenius-normalise, then 5 quintic Newton-Schulz steps
  `X <- a*X + b*(X X^T) X + c*(X X^T)^2 X`, (a, b, c) = (3.4445, -4.7750, 2.0315), approximating U V^T of M_t = U S V^T (Eq. 1).
- Shared: decoupled weight decay, grad clip 1.0, 5% linear warmup, cosine decay to 0.1x (LM) / 0.01x (vision), bfloat16, same data order per seed.
- LMs: context 256, GPT-2 tokenizer, tied embeddings, pre-norm, GELU MLP 4d, no dropout. Micro: 6 layers, d=256, 8 heads, 17.66M. Small: 8 layers, d=384, 8 heads, 33.57M. 5,000 steps at 16 sequences x 256 = 4,096 tokens/step (small: batch 8 x 2 accumulation). Data: 100M-token FineWeb-Edu subset, 5M val, 5M test (Table 2, A.1-A.2).
- Selected hyperparameters (Table 3):
  - LM-micro: AdamW lr 1e-3, wd 0, beta2 0.95. Muon matrix lr 0.01, wd 0.10, momentum 0.90, aux AdamW lr 3e-4.
  - LM-small: AdamW lr 3e-4, wd 0.10, beta2 0.99. Muon lr 0.01, wd 0.10, momentum 0.90, aux lr 3e-4.
  - ViT: AdamW lr 3e-4, wd 0.45. Muon lr 0.01, wd 0.05/0.45, momentum 0.85/0.95, aux lr 3e-4.
  - Aux AdamW: beta2 0.95, wd 0.10 (LM) / 0.05 (vision).
- Muon-eligible share: 4.72M of 17.66M (26.7%) in micro, 14.16M of 33.57M (42.2%) in small; the rest is mostly the GPT-2 embedding table.

## Numbers
- Equal-data validation-loss AUC gain of Muon (Table 4, 5 seeds): LM-micro 3.25%, LM-small 4.46%, ViT 16.98% / 22.44%.
- Test loss (Table 5): LM-micro 5.5943 +- 0.0044 (AdamW) vs 5.3277 +- 0.0057 (Muon), -4.77% [4.67, 4.86]; LM-small 5.5112 vs 5.1250, -7.01% [6.95, 7.06]. Perplexity down 23.40% and 32.04%. Muon better in all 20 seed pairs. ViT: -11.52% and -26.03% loss, +6.76 and +7.39 accuracy points.
- Equal-time AUC (Table 4): AdamW better by 6.96% (micro) and 3.38% (small); Muon better by 19.68% / 22.31% on vision. Equal-time endpoint on LM-small: -0.73%, CI [-1.60, 0.22] (a tie).
- Throughput (Table 6): micro 116,300 vs 43,879 tokens/s (ratio 0.377); small 62,884 vs 29,456 (0.468). Optimizer share of step time: 56.1-69.6% Muon vs 6.1-9.1% AdamW.
- Stable 1% crossover at 4.8% / 3.0% of the LM horizon (Sec. 3.1).
- Tuning regret (Fig. 3): 1 random candidate: AdamW 7.52%, Muon 13.66%; 5: 0.42% vs 1.69%; 10: 0.09% vs 0.67%.
- Cheaper Muon (Sec. 4, Table 8, Fig. 6; one seed, LM-small, 1,000-step dev runs): 3 NS steps instead of 5 saves 21.87% of training time (100.59 s vs 128.75 s) for a 0.74% AUC penalty (6.6162 vs 6.5676). Muon only on attention matrices: 28.4% time saved, 2.3% penalty; only on MLP: 29.1% saved, 1.1% penalty; output-proj only or QKV only: about 44-45% saved, 2.5-2.8% penalty.
- AdamW for the first 25% then Muon (Table 7, 3 seeds): 3.97% / 5.92% LM test-loss gain, worse than full Muon. "Variance-Muon" (elementwise second moment after orthogonalisation): 5.16% / 7.45% on LM but -11.71% on long-tail vision, and it selected boundary hyperparameters.

## Smallest scale tested, and whether the result held there
Smallest LM: 17.7M params, of which only 4.72M are transformer matrices (6 x 256) -- close to our dense core. The data-efficiency win held (4.77% test loss, tight CI). Smallest model overall: 2.70M ViT, where the gain was largest. Caveat [I]: 5,000 steps x 4,096 tokens = about 20M tokens seen, under one pass over the data, final loss 5.1-5.6 nats. This is an early-training regime; the paper says nothing about converged quality.

## Training recipe and gotchas
- Starting point for our FP stage: Muon lr 0.01, momentum 0.90, wd 0.10 on matrices; AdamW lr 3e-4, beta2 0.95 on embeddings/head/gains.
- Muon needs a small LR sweep: with a single guess it is about twice as far from its optimum as AdamW. Budget about 5 candidates (regret 1.69%).
- The wall-clock loss is an artefact of tiny batches (4,096 tokens/step) and reference kernels: NS cost is per step, independent of batch size. [I] On a 5090 with batches of 50-100K tokens the optimizer share should fall well below the forward/backward cost, so the equal-data result is the relevant one for us. Not measured in the paper.
- If the optimizer step still dominates: 3 NS steps, or orthogonalise every other step (MuonBP, cited, not tested here).
- Do not switch AdamW->Muon mid-run; do not adopt Variance-Muon on this evidence.

## Mapping onto the GBA design
Training-only: zero cost in cycles, RAM or ROM at inference. Relevant facts: (1) our core is nearly all Muon-eligible matrices at exactly the widths tested; (2) our hashed n-gram tables, per-layer embeddings and two-level head stay on (sparse) Adam, as in the paper's routing; (3) 64 routed expert MLPs per layer means 64x more matrices to orthogonalise per step, each receiving gradient from about 1/64 of tokens. [I] Batch the NS iteration over the expert axis, and expect noisier per-expert momentum; untested anywhere in this paper. (4) Nothing here involves quantisation: Muon's fixed-spectral-norm updates interacting with STE/ternary latent weights is not covered.

## Verdict
**Test (lean adopt for the full-precision stage).** A clean, well-controlled 5-23% perplexity-level gain at equal data at our widths is worth having, and the wall-clock penalty should mostly vanish at our batch size. But the horizon is short, AdamW was not given a longer schedule, and QAT is absent. Settling experiment: our 6x256 model, AdamW vs Muon, 5-candidate LR sweep each, equal GPU-hours on the 5090 at the real batch size, run through the ternary QAT stage, compare final ternary validation loss; log optimizer share of step time.

## Could not verify
- No code or repository link appears in the paper; the "reference implementation" is described, not linked.
- The 23 candidate grids (LR/decay ranges) are not listed, only the selected values.
- No result at convergence, with larger batches, with quantisation, or with MoE/routed layers.
- HTML versions (arxiv.org/html) returned 404; read from PDF text extraction, figures read from their embedded labels.

## Links worth following
- Muon reference note: https://kellerjordan.github.io/posts/muon/
- MuonBP (block-periodic orthogonalisation, cheaper step): https://arxiv.org/abs/2510.16981
- NorMuon: https://arxiv.org/abs/2510.05491
- Muon is scalable for LLM training: https://arxiv.org/abs/2502.16982
- Distance-aware Muon (tuning sensitivity): https://arxiv.org/abs/2605.18999
