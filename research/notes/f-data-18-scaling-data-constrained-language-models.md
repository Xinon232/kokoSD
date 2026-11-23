# Scaling Data-Constrained Language Models  (https://arxiv.org/abs/2305.16264, NeurIPS 2023, Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel; code/runs: https://github.com/huggingface/datablations)

Read in primary form: arXiv HTML, main body and appendices, through a summarising fetch tool in two 100K-character halves. Figures were not seen; per-run loss values that live only in figures are therefore missing. Quoted phrases below are as the tool returned them.

## What it is
An empirical study (400+ runs, about 3M GPU-hours) of what to do when unique training text, not compute, is the limit. GPT-2-architecture models from 7M to about 9B parameters are trained on nested C4 subsets for 1 to 44+ epochs. The authors extend the Chinchilla loss law with terms that make repeated tokens and excess parameters worth exponentially less, and fit the decay constants. They also test filling the data budget with code and with perplexity/dedup filters.

## Mechanism
Chinchilla form: `L(N,D) = A/N^alpha + B/D^beta + E`. Replace N and D by "effective" values.

    U_D = min(D_C, D)            # unique tokens (D_C = unique budget, D = tokens seen)
    R_D = D/U_D - 1              # repetitions = epochs - 1
    D'  = U_D + U_D * R_D* * (1 - exp(-R_D / R_D*))          (Eq. 5)
    U_N = min(N_opt, N)          # N_opt = single-epoch compute-optimal size for U_D
    R_N = N/U_N - 1              # excess-parameter ratio
    N'  = U_N + U_N * R_N* * (1 - exp(-R_N / R_N*))          (Eq. 6)

Fitted law (Eq. 17, C4, GPT-2 arch):

    L = 521 / N'^0.35 + 1488 / D'^0.35 + 1.87,   R_D* = 15.4,  R_N* = 5.3,  U_N = 0.051 * U_D

Effective-data multiplier D'/U_D implied by Eq. 5 with R_D* = 15.4 (my arithmetic, not a table in the paper):

| epochs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | inf |
|---|---|---|---|---|---|---|---|---|
| D'/U_D | 1.00 | 1.97 | 3.73 | 6.62 | 10.6 | 14.3 | 16.1 | 16.4 |

So 4 epochs is worth about 93% of 4x fresh data, 16 epochs about 66% of 16x, and nothing beyond about 16x unique is reachable by repetition.

## Numbers
- "up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data" (abstract). Concrete instance: 8.7B params, 178B tokens seen, 44B unique x 4 epochs ends "only 0.5% higher validation loss" than 178B unique (Sec. 6, Fig. 4).
- Half-life: "R_D* ~ 15, corresponding to 15 repetitions (or 16 epochs)"; beyond that "returns diminish extremely fast" (Sec. 6).
- R_N* = 5.3 < R_D*: excess parameters saturate sooner than excess epochs, so "scaling epochs faster than parameters" (Sec. 5). At 9.3e21 FLOPs and 25B unique tokens, the law's pick (6.3B params, more epochs) beats the Chinchilla pick (8.7B) with 27% fewer parameters; downstream average 25.9 vs 23.5 (Table 8).
- 100M unique tokens (Fig. 3, 93 runs): single-epoch optimum about 7M params; best loss at "20-60x more parameters and epochs", about 7000x the FLOPs, "more than a 50% reduction in loss".
- 100M unique tokens, 146M-param model: 59 epochs was optimal, on both regular and deduplicated C4 (App. E). Models above 2B params were clearly worse than 200M-1B (App. F).
- Epoch-wise double descent on 100M tokens: loss rises around 200 epochs then falls again; such runs were mostly excluded from the fit (App. D). At 44 epochs some runs show loss rising mid-training and the law underestimates their loss.
- Downstream, 2.8B model, 55B tokens seen: average 20.9 at 1 epoch to 14.1 at 44 epochs (Table 4); 8.7B: 26.2 to 18.3 (Table 8).
- Fit quality R^2 about 0.77-0.80 with both decays (Table 1).
- Code filler: up to 50% Python (42B tokens) with no natural-language downstream loss; about 2x effective tokens (Sec. 7, Fig. 6). Perplexity filter (keep best 25%, repeat about 2 epochs) helped, mostly on noisy data (OSCAR); dedup (21B left, 4 epochs) did not help.

## Smallest scale tested, and whether the result held there
Smallest model: 7M params (d_model 128, 3 layers, 4 heads). Smallest unique budget: 100M tokens. The qualitative result (many epochs plus oversizing helps a lot) is strongest there. The headline constants (4 epochs free, R_D* = 15.4) are one global fit dominated by large runs; the 0.5% figure is from an 8.7B model. The 59-epoch optimum at 100M tokens/146M params suggests small-data regimes tolerate more repetition than the 16-epoch half-life implies, but that is a 20x-oversized model, not Koko's regime.

## Training recipe and gotchas
- Adam, beta1 0.9, beta2 0.999 (0.95 for two large budgets), eps 1e-8, max LR 2e-4, 1% linear warmup, cosine decay to 2e-5, batch 256 below 2B params, dropout 0.1, weight decay 0.1, clip 1.0, sequence length 2048, vocab 50,257, bf16.
- "Data is shuffled after each epoch." No early stopping; cosine schedule spans the whole run.
- Training loss is useless under repetition; use held-out loss (App. H).
- Authors' own limitation (App. Q): returns from epochs "may heavily depend on hyperparameters such as learning rate, dropout, or the optimizer choice"; they expect higher LR to bring diminishing returns sooner. Dropout 0.1 and weight decay were on throughout, which helps repetition.
- Law assumes the whole set is repeated uniformly; repeating only a fraction is not modelled.

## Mapping onto the GBA design
Zero inference cost: no lookups, adds, multiplies, RAM or ROM change. It is a training-budget rule.
- Corpus target: let T = tokens the 5090 can push through the final model in the budget. Aim for unique tokens >= T/4; generating beyond T/4 buys almost nothing (under 7%). If generation is the bottleneck, T/8 costs about 17% of effective data and T/16 about 34%.
- Size check: U_N = 0.051 * U_D means about 20 unique tokens per float parameter at the single-epoch optimum. Koko stores 50-120M ternary weights (fewer float-equivalents) but spends most on lookup tables that see sparse gradients, so the rule is only a sanity bound: 1-2.4B unique tokens would be "Chinchilla-matched"; with a few hundred M tokens Koko is in the oversized, multi-epoch regime the paper covers, where R_N* = 5.3 warns that parameters beyond roughly 5x N_opt add little.
- Fact-exposure schedule: this paper does not measure per-fact exposures. Verbatim repetition here is not the same as the 100-1000 reworded exposures in Allen-Zhu and Li (f-data-09). Use rewording for facts, and count epochs only for the general corpus.
- Hashed n-gram tables and 64-expert MLPs see each row rarely; they may want more passes than dense weights. Untested by this source.

## Verdict
Adopt as the budgeting rule (plan 4 passes over the general corpus, never more than about 16; shuffle each epoch; evaluate on held-out text). Test the constants: on a 1/10-scale Koko, fixed token budget, compare 1x, 4x, 16x repeats of proportionally smaller subsets with ternary QAT. If the 4x arm is within about 1% loss, the rule holds for our model class.

## Could not verify
- Per-epoch validation losses of Fig. 4 and the lowest loss in Fig. 3 (figure-only).
- Whether parameter counts include embeddings.
- Nothing in the paper covers quantized/ternary models, synthetic or simple-English data, distillation, or sparse lookup parameters.
- A "40 epochs" figure sometimes cited in summaries was not found; the paper's values are 44 (a tested setting) and 59 (optimum at 100M tokens).
- The alternative fit in App. F reports very different constants (R_D* about 26,530, R_N* about 2,041 under a different parametrisation); I did not resolve its units.

## Links worth following
- https://github.com/huggingface/datablations (all runs and loss curves; would give the figure-only numbers)
- Chinchilla, Hoffmann et al. 2022: https://arxiv.org/abs/2203.15556
