# Can We Trust Small Language Models as Judges?  (https://neurips.cc/virtual/2026/178372, NeurIPS 2026 workshop poster; Suchir Salhan, Mingchuan Zou, Paula Buttery, Cambridge)

**Read status: abstract only.** The full paper is on OpenReview (forum 6bM9MkX9xz; a second listing 161275 points to forum Gz8290F04b), which returned a browser challenge (HTTP 403) for the forum, PDF and API. No arXiv version was found (arXiv author and title searches, author homepage). Everything below marked [A] is from the official NeurIPS abstract page, as paraphrased by the fetch tool (the numbers were returned consistently across three page fetches). [R] is my reconstruction of standard methods, not confirmed against the paper. [S] is from a different source.

## What it is
A workshop paper (not main track; listed under two NeurIPS 2026 evaluation-trust workshops) that audits eight open-weight small-LM judges on essay scoring and legal evaluation [A]. Its claim is that the usual reliability numbers for a small judge mislead in three ways: agreement can rise without better discrimination, deferral can help without the judge having useful uncertainty, and calibration can help without the human labels being reliably useful [A]. It turns each into an audit test with a counterfactual baseline [A]. The message: deploy a small judge only after the specific capability you need has passed its audit, not because a reliability score looks good [A].

## Mechanism
Four audits [A]; the formulas are [R].

1. Separate ranking from scale. Report Spearman rho (order only) next to QWK (order plus scale). Mean-variance alignment on a small human-labelled anchor set:
   `s' = round(clip(mu_h + (s - mu_j) * sigma_h / sigma_j, lo, hi))`
   This is monotone, so rho cannot change; any QWK gain is scale repair. QWK = 1 - sum(w_ij O_ij) / sum(w_ij E_ij), w_ij = (i-j)^2/(K-1)^2.
2. Deferral versus matched-budget random. If the uncertainty rule escalates a fraction b of items to a human, draw 500 random policies that also escalate exactly b, and count the conditions where the uncertainty rule beats them [A]. The acceptance threshold (e.g. above the 95th percentile of the 500) is not known to me.
3. Anchors are a random variable. Resample the anchor set many times and report the spread of post-calibration agreement [A]. Number of resamples and anchor size: not known (the companion paper uses 60 labelled examples [S, companion abstract]).
4. Test that confidence and self-consistency predict correctness (e.g. AUROC of confidence against "judge matches human"); check score variance, because a near-constant judge looks consistent while discriminating nothing [A].

## Numbers
All [A], abstract level; no table or figure numbers available.
- Mean QWK 0.246 -> 0.433 after aligning judge scores to human mean and variance; eight open-weight judges.
- Ordering "capped" at rho = 0.444 (unchanged by alignment).
- Uncertainty-based escalation beats 500 matched-budget random policies in 37 of 40 legal conditions and 0 of 28 essay conditions.
- Companion paper, same authors ("Towards Trustworthy Compact Judges", NeurIPS 2026 LIGHT workshop oral) [S, abstract only]: 72 SLMs calibrated on 60 human labels; calibration recovers agreement lost to scaling but not ranking, worst under 1B parameters; 4-bit AWQ costs a 7B legal judge 0.026 balanced accuracy with self-consistency statistically unchanged.
- Context from SLMJury (arXiv 2606.07810, read in HTML, first 100K characters) [S]: best small-judge Spearman with humans on SummEval is 0.520 (Phi-4, 14B), Qwen3-8B 0.453; MT-Bench agreement with large judges tops at 0.570 (Phi-4-Reasoning). So rho around 0.45-0.55 is the ceiling two independent studies report for local judges on graded open-ended quality.

## Smallest scale tested, and whether the result held there
Not known for this paper: judge names and sizes are not in the abstract. The companion abstract says ranking is least repairable below 1B [S]. Practical reading: use the largest judge that fits a 32 GB card (7-14B, 4-bit is acceptable per the 0.026 figure [S]), not a sub-1B one.

## Training recipe and gotchas
No training; this is evaluation method. Gotchas that apply to us:
- Calibration to GPT-2-vs-Koko human labels will make scores look more human without making the judge better at telling outputs apart.
- Uncertainty gating is task-dependent (worked on legal, failed on essays); do not assume it works for chat quality.
- A judge that gives nearly every tiny-model reply the same low score will show high self-consistency. Tiny-model outputs cluster at the bottom of a 1-5 scale, so this failure is likely for us.

## Mapping onto the GBA design
Zero inference cost: no lookups, adds, multiplies, RAM or ROM. It affects only the offline evaluation harness and the filtering of LLM-generated training conversations. Consequences:
- A local judge with rho near 0.44 per item is too noisy for per-sample decisions (filtering individual training dialogues, picking per-prompt winners). It can still rank systems if errors average out: use paired comparisons on the same prompts over many prompts (thousands), with a sign test or bootstrap, as in the error-bars note (f-eval-05).
- Primary benchmark against GPT-2 small should stay on judge-free metrics (perplexity on held-out simple English, BLiMP, EWoK, keystroke savings for typing); the judge is a secondary signal.
- If a judge is used, label 100-200 Koko/GPT-2 replies by hand once, then report rho and QWK separately, with resampled anchors.

## Verdict
Adopt the audit protocol; test (do not trust) the local judge. The paper changes nothing on the cartridge. Settling experiment: hand-label about 200 reply pairs (Koko checkpoint versus GPT-2 small, same prompts) for preference; run the candidate local judge pairwise with both orderings; accept the judge only if pairwise accuracy on non-tied human pairs clears a preset bar (say 75%) and its system-level ranking of 4-5 checkpoints of known perplexity order is correct.

## Could not verify
- Full text, tables, the eight judge names and sizes, datasets (essay set is plausibly ASAP-style given QWK, but unconfirmed), prompts, anchor-set size, number of resamples, the exact calibration formula, code link.
- Whether rho = 0.444 is a mean, a maximum, or one dataset.
- Whether the two NeurIPS listings are the same manuscript.
- No result here concerns judging dialogue or tiny-model output; transfer to our setting is an assumption.

## Links worth following
- https://openreview.net/forum?id=6bM9MkX9xz (full paper; needs a real browser)
- https://neurips.cc/virtual/2026/160516 (companion, 72 SLMs, sub-1B result; OpenReview 9m8SaQ7yB1)
- https://arxiv.org/abs/2606.07810 (SLMJury, 16 judges 0.6B-14B, code at github.com/anishh15/SLMJury)
