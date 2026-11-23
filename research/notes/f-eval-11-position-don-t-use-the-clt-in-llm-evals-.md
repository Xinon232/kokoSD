# Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints

URL: https://arxiv.org/abs/2503.01747 (ICML 2025 position paper, spotlight per the code README). Authors: Sam Bowyer, Laurence Aitchison, Desi R. Ivanova. Code: https://github.com/sambowyer/bayes_evals (MIT), experiments: github.com/sambowyer/no_clt_paper.

Source status: read in primary form (arXiv HTML, main text plus appendices E-J through a summarising fetch tool; Appendix C/D text and exact figure values not seen). README of bayes_evals read; its source file was not.

## What it is

A position paper about error bars on benchmark scores, not about models. It shows by simulation that CLT intervals (mean +/- 1.96 SE) and bootstrap intervals under-cover when a benchmark has tens to ~100 binary-scored items, and that they fail outright at scores near 0 or 1 (zero width, or bounds outside [0,1]). It recommends Wilson intervals for a single accuracy and simple Bayesian credible intervals (Beta-Bernoulli and extensions) for model comparisons, clustered questions, paired designs and non-average metrics such as F1. Everything is for binary correct/incorrect outcomes.

## Mechanism

All use alpha = 0.05 and uniform priors.

- CLT (the thing to avoid): theta_hat +/- z * sqrt(theta_hat(1-theta_hat)/N). Fails when theta_hat or 1-theta_hat < 1/(1+N/z^2).
- Wilson: centre (theta_hat + z^2/2N)/(1+z^2/N); half-width (z/2N)/(1+z^2/N) * sqrt(4N theta_hat(1-theta_hat) + z^2). `scipy.stats.binomtest(S,N).proportion_ci(method="wilson")`.
- Bayes, single model: posterior Beta(1+S, 1+N-S); report 2.5/97.5% quantiles.
- Bayes, two models, different items: draw ~2,000 samples from each Beta posterior; quantiles of theta_A - theta_B (or odds ratio); P(A>B) = mean(samples_A > samples_B).
- Bayes, paired (same items): theta_A, theta_B ~ Beta(1,1); rho = 2*rho_hat - 1, rho_hat ~ Beta(4,2); latent bivariate Gaussian with means Phi^-1(theta_A), Phi^-1(theta_B), unit variance, correlation rho; y = 1[latent > 0]. Importance sampling: K = 10,000 prior draws weighted by the multinomial likelihood of the 2x2 table (both right, A only, B only, both wrong); bivariate normal CDF needed for one cell.
- Bayes, clustered: d ~ Gamma(1,1), theta ~ Beta(1,1), theta_t ~ Beta(d*theta, d*(1-theta)), y_it ~ Bernoulli(theta_t). Integrate out theta_t: Y_t ~ BetaBinomial(N_t, d*theta, d*(1-theta)). K = 10,000 prior draws of (d, theta), weight = product over clusters of beta-binomial likelihoods, resample, take quantiles of theta.
- Clustered CLT (comparison baseline): SE^2 = SE_CLT^2 + (1/N^2) sum_t sum_i sum_{j!=i} (y_it - ybar)(y_jt - ybar).
- F1 and other confusion-matrix metrics: Dirichlet(1,1,1,1) prior, posterior Dirichlet(1 + counts), compute the metric per sample, take quantiles.
- Library API: DataFrame with one row per question, one 0/1 column per model; `independent_intervals(df, alpha)`, `independent_comparisons(df)`, `paired_comparisons(df)` (the last two return a matrix of P(model i better than model j)).

## Numbers

Simulation: 100 theta values from the prior x 200 datasets each, N in {3, 10, 30, 100}, 5 seeds (coverage SE about 1e-3), 100 nominal levels from 0.8 to 0.995.

- IID, N = 100: the nominal 95% CLT interval covers only 92.5% (Fig. 2). Wilson and Bayes track nominal; Clopper-Pearson is too wide.
- Clustered (Fig. 3): both plain CLT and clustered CLT under-cover at small N; only the clustered Bayes model is right at all N. No numeric coverage in the text.
- Independent comparison (Fig. 4): CLT "far below the target level" at small N; Fisher exact over-conservative, with infinite odds-ratio intervals in 43.5/17/6.5/1.9% of datasets at N = 3/10/30/100 (Fig. 14).
- Paired (Fig. 5): all non-Bayesian methods "severely underperform" at small N; paired Bayes gives narrower intervals than unpaired Bayes.
- F1 (Fig. 6): bootstrap systematically under-covers; Bayes tracks nominal.
- Width (Figs. 13-15): Bayes narrower than CLT/bootstrap at equal coverage.
- Cost (Fig. 11): slowest method about 200 ms on one CPU.
- Real examples: LangChain tool-use benchmark N = 20 and AIME 2025 II N = 15, where CLT bars collapse or leave [0,1] (Figs. 1, 9).

## Smallest scale tested, and whether the result held there

N = 3 is the smallest; the claims are strongest there. The largest is N = 100. Nothing above 100 is simulated, so "a few hundred" in the title is an extrapolation of the trend, and the paper declines to name a threshold N*. Our 200 prompts is outside the tested range on the high side.

## Training recipe and gotchas

No training. Gotchas for use:
- Binary outcomes only. A 1-5 judge rubric does not fit; either binarise (pass/fail per criterion, or win/loss versus GPT-2) or use a different model. The paper offers nothing for ordinal or continuous scores, nor for perplexity.
- Mismatched priors make Bayes intervals too wide at small N; the effect fades as N grows (Figs. 17-19).
- With theta_A = theta_B, paired Bayes and paired CLT cover about the same (fixed-theta appendix); Bayes pulls ahead when the gap is large.
- Judge noise and repeated sampling at temperature are not modelled.

## Mapping onto the GBA design

Zero on-cartridge cost: no lookups, adds, multiplies, RAM or ROM. It affects only the offline evaluation harness.

What changes in our plan:
- A 200-prompt, 50-cluster judged set (4 prompts per cluster) has an effective sample size between 50 and 200 depending on within-cluster correlation, i.e. inside the regime the paper tested. Use the clustered beta-binomial model, not a cluster bootstrap or clustered SE.
- Koko versus GPT-2 on the same prompts is a paired design: score each prompt as a win/loss or as per-model pass/fail and report P(Koko > GPT-2) from the paired model.
- Scores near the floor or ceiling (e.g. safety or format pass rates of 98%, or early checkpoints at 2%) are exactly where CLT bars collapse; use Wilson/Beta.
- My own arithmetic, not from the paper: Wilson half-width at N = 200, p = 0.5 is about +/-6.9 points; at an effective N of 100 it is about +/-9.6. So the judged set can only separate checkpoints that differ by roughly 10+ points unless pairing is exploited. Ablation decisions on small deltas should rest on held-out perplexity over many tokens, with the judged set as a final check.

## Verdict

Adopt, as an evaluation protocol. It is free, the code is one file, and it prevents over-reading small judged-set differences when choosing between design variants. It does not improve the model.

Settling experiment: score two seeds of the same configuration on the judged set; the reported interval for their difference should include zero, and an A/A resampling of prompts should show about 95% coverage. Compare the clustered Bayes interval with a cluster bootstrap on our own data to see how much they differ at 50 clusters.

## Could not verify

- Cluster count T and cluster sizes N_t used in the Fig. 3 simulation: not found in the text I read.
- Numeric coverage values other than 92.5% at N = 100: only in figures, not read.
- Behaviour at N = 200-500: not in the paper.
- HDI versus quantile interval numeric comparison (Appendix J): no numbers seen.
- The paper was read through a summarising fetch tool, so equations are restatements, not verbatim; check against the PDF before implementing the paired model. bayes_evals.py source was not read, only its README.
- "Spotlight" status comes from the repository README.

## Links worth following

- https://github.com/sambowyer/bayes_evals (single-file library)
- https://github.com/sambowyer/no_clt_paper (simulation code; would give T and N_t)
- Miller 2024, "Adding Error Bars to Evals" (arXiv 2411.00640), the CLT-based recommendation this paper argues against, and the source of the clustered SE formula; not read here.
