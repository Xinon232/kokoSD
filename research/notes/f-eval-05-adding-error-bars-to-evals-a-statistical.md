# Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations

https://arxiv.org/abs/2411.00640 (v1, 1 Nov 2024, 14 pages; only version). Evan Miller, Anthropic.
Read in full (primary) from https://arxiv.org/html/2411.00640, including Appendices A-C.

## What it is

A methods paper, not an experiment paper. It treats eval questions as a random draw from an unseen super-population and gives closed-form standard errors for one model, for the difference between two models, and for questions that come in correlated groups. It adds two variance-reduction tricks (resample answers; score with next-token probabilities) and a sample-size formula. Only one table (Table 4) holds real data; every other number is a fictional or analytic illustration.

## Mechanism

Score on question i: `s_i = x_i + eps_i` (x = conditional mean, Var(eps_i) = sigma_i^2).

- Eq.1 `SE = sqrt( (1/(n-1)) * sum (s_i - sbar)^2 / n )`. Eq.2 (0/1 scores only): `sqrt(sbar(1-sbar)/n)`. Eq.3: `CI95 = sbar +/- 1.96 SE`. Bootstrap called unnecessary for a plain mean.
- Eq.4, clustered (cluster c = questions drawn together): `SE_cl^2 = SE^2 + (1/n^2) * sum_c sum_i sum_{j!=i} (s_ic - sbar)(s_jc - sbar)`.
- Variance split: `Var(mu_hat) = (Var(x) + E[sigma_i^2]) / n`. Only the second term is reducible: (a) K samples per question, use the per-question mean, term becomes `E[sigma^2]/K`; choose K so `E[sigma^2]/K << Var(x)`; never pool the K*n answers as independent. (b) score the expected value from next-token probabilities, term becomes 0. Do not lower temperature to cut variance (it moves variance into Var(x) or adds bias).
- Unpaired difference: `SE = sqrt(SE_A^2 + SE_B^2)`; `z = diff/SE` (Eq.5-6).
- Eq.7, paired (same questions): `d_i = s_A,i - s_B,i`; `SE = sqrt( var(d) / n )`. Equivalent: `sqrt(SE_A^2 + SE_B^2 - 2 SE_A SE_B Corr(s_A,s_B))`.
- Eq.8, paired + clustered: `SE = (1/n) * sqrt( sum_c ( sum_i (d_ic - dbar) )^2 )` (the double sum over i,j inside a cluster is the squared cluster sum).
- Eq.9, questions needed: `n = (z_{a/2} + z_b)^2 * (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / delta^2`, with `omega^2 = Var(x_A) + Var(x_B) - 2 Cov(x_A,x_B)`. Eq.10 inverts it for the minimum detectable effect delta. alpha=0.05, power 0.80 gives (1.96+0.84)^2 = 7.85.
- Appendix C: for clustered data plug cluster-adjusted omega^2, sigma^2 into Eq.9; estimate them from a pilot with K >> 1, dividing by (K-1); subsample whole clusters, never single questions.

Reporting format: mean with SE in parentheses, question count, cluster count; for pairs: difference (SE), 95% CI, score correlation (Tables 2, 3, 5).

## Numbers

- Table 4 (real, Anthropic models, model not named): clustered vs naive SE: DROP 1.34 vs 0.44 (ratio 3.05; 9,622 questions / 588 clusters per Table 3), RACE-H 0.51% vs 0.46% (1.10; 3,498 / 1,045), MGSM 1.62% vs 0.86% (1.88; 2,500 / 250).
- Sec 3.1 (analytic, binary scores, difficulty uniform on [0,1]): Var(x)=1/12, E[sigma^2]=1/6; variance falls by 1/3 at K=2, 1/2 at K=4, 5/9 at K=6, limit 2/3. Next-token probabilities reach the 2/3 limit at once.
- Sec 3.3 (analytic): T=0 raises Var(x) from 1/12 to 1/4; second example about five-fold, with mean shifted 2/3 to 3/4.
- Sec 4.2 (analytic): score correlation 0.5 gives 1/3 less variance from pairing (1/6 to 1/9).
- Sec 5 (analytic): omega^2=1/9, sigma=0, delta=0.03 gives n ~ 969, hence "new evals should contain at least 1,000 questions". n=198, sigma^2=1/6 each: MDE 13.2% at K=1, 7.5% at K=10.
- Table 2 (fictional but arithmetically real): 164 questions gives SE ~3.2%; 5,000 gives 0.7%.

## Smallest scale tested, and whether the result held there

Nothing is tested by model size; the statistics are model-agnostic. The smallest n discussed is 164 (HumanEval) and 198. The paper assumes "a large number of questions"; it gives no small-n correction. A follow-up position paper (Bowyer, Aitchison, Ivanova, arXiv 2503.01747, abstract only read) says CLT intervals underestimate uncertainty below a few hundred items.

## Training recipe and gotchas

No training. Gotchas for our eval:
- Our prompts will be clustered: turns of one conversation, prompts from one template/topic/LLM-generation seed, sentences of one story. The cluster is the unit of independence; a 2-3x SE inflation is plausible (Table 4).
- Eq.8 has no finite-cluster correction; use at least ~50 clusters and prefer many small clusters.
- Eq.2 is wrong for fractional scores (the paper faults the Llama 3 report for this).
- Paper is silent on multiple comparisons, judge noise, and tokenizer mismatch. Treat an LLM judge as part of eps (resample it too). Compare Koko and GPT-2 in bits per character/word, not per token.
- Table 1 lists Dreadnought HumanEval as 87.7%, Table 2 as 86.7% (paper typo; -3.1% matches 86.7).

## Mapping onto the GBA design

Zero cost on the cartridge: no lookups, adds, multiplies, RAM or ROM. It is offline analysis on the PC (a few lines of NumPy). The one requirement is a bit-exact PC reference of the integer inference so per-prompt scores can be computed off-device.

Our own sizing from Eq.9 (alpha 0.05, power 0.8; these are our calculations, not the paper's):
- Likelihood metrics (bits/char, top-3 next-word hit rate for predictive typing): deterministic, sigma=0. Run a pilot of ~200 documents, measure var(d) clustered by document, then n = 7.85 var(d)/delta^2.
- Binary per-prompt scores near 50%, correlation rho: var(d) = 0.5(1-rho). rho=0.5: 196 prompts for a 10-point gap, 785 for 5 points, 2,180 for 3 points. rho=0: double.
- Judged head-to-head preference (win=1, loss=0, test mean against 0.5): var <= 0.25, so 785 independent prompts to tell 55/45 from a tie.
- Sampled generations: pilot with K=5-10 to estimate E[sigma^2] vs Var(x), then fix K.

## Verdict

Adopt, as the reporting standard. Fix the prompt count only after a pilot (about 200 clusters, K >= 5) measures var(d) and the cluster ratio; plan on roughly 1,000 independent conversation seeds as the default. It changes nothing in the model.

## Could not verify

- Which Anthropic models produced Table 4 (not stated).
- The "0.3 to 0.7" cross-model score-correlation range is from Anthropic's blog post about the paper, not the paper itself (which only shows fictional 0.37-0.64).
- Small-n behaviour: taken from the Bowyer et al. abstract only.
- No evidence about correlation between a tiny ternary model and GPT-2; must be measured.

## Links worth following

- https://www.anthropic.com/research/statistical-approach-to-model-evals (companion blog, read)
- https://arxiv.org/abs/2503.01747 (small-n alternatives; github.com/sambowyer/bayes_evals named in its abstract)
- https://arxiv.org/abs/2010.06595 (Card et al., "With Little Power Comes Great Responsibility", cited as [2]; not read)
