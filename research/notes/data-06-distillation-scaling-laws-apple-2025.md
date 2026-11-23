# Distillation Scaling Laws  (https://arxiv.org/abs/2502.08606, v2 July 2025, ICML 2025; Busbridge, Shidani, Weers, Ramapuram, Littwin, Webb; Apple)

Read in full: arXiv HTML v2 with appendices; figures as captions only. Tags: [P] paper, [A] my arithmetic from its Table 6, [SLS] Sparse Logit Sampling (read), [G3] Gemma 3 report (two passages), [Z] Zhang et al. (section 3), [I] my inference.

## What it is
A controlled study of logit distillation for LM pretraining: 73 supervised and 697 distilled runs, 143M-12.6B non-embedding parameters, up to 512B tokens of English C4. It fits a law predicting student cross-entropy from student size, distillation tokens and teacher cross-entropy. Distillation beats plain training only at modest student tokens, and only if the teacher is free or amortized.

## Mechanism
Loss per token [P eq. 7]: `(1-λ)·CE(y, z_S) + λ·τ²·CE(softmax(z_T/τ), softmax(z_S/τ)) + 1e-4·(log Z_S)²`; the paper uses λ = 1, τ = 1.

- Supervised law [P eq. 1]: `L(N,D) = E + (A/N^α + B/D^β)^γ`; E 1.220, A 3355, B 18186, α 0.408, β 0.431, γ 0.452.
- Distillation law [P eq. 8]: `L_S = L_T + L_T^(-c0) · (1 + (L_T/(d1·L~S))^(1/f1))^(-c1·f1) · (A'/N_S^α' + B'/D_S^β')^γ'`, where `L~S = L(N_S, D_S)` is the student's own supervised loss; A' 2243, B' 24181, α' 0.321, β' 0.637, γ' 0.764, c0 2.549, c1 522.6, f1 0.090, d1 1.315 [P Table 6]. Fit error is about 1% or less.
- Teacher size and tokens matter only through L_T [P Fig. 3b]; a too-strong teacher hurts (capacity gap).
- Cost [P eq. 9]: `3F(N_S)·D_S + F(N_T)·D_S + 3F(N_T)·D_T` (student, teacher logits, teacher training).
- Sparse targets [SLS; G3 uses 256 samples]: draw 50 ids per position from the teacher distribution, store counts/50 (about 12 unique ids), train with cross-entropy on them. Unbiased, unlike top-k.

## Numbers
- Student loss is "mostly flat" for λ > 0.5 [P Fig. 51, 198M-2.72B].
- τ = 1 best over 0.5-10; τ > 3 hurts small students most [P Fig. 52].
- Top-k 128: +0.11 nats; top-p 0.9: +0.13; k = 128 with λ = 0.7: within 0.01 [P Fig. 54, V = 32168].
- Forward KL 2.42 against reverse KL 2.70, 1.82B to 546M [P Table 7].
- Students trained on teacher top-1 are overconfident, more so with more tokens [P E.8, 198M].
- [A] 198M student: supervised 2.789, best distilled 2.534 at 20 tokens/parameter; gain 0.06 at 200; 0.02 at 2000. 546M: 0.16, 0.03, 0.006. Optimal teacher loss is 0.67x the student's supervised loss at 20, 0.91x at 2000.
- Counting teacher training, supervised always wins [P Figs. 8, 17, 33].
- [SLS], 3B to 300M: at 10B tokens CE 2.807, full KD 2.747, 12 sampled ids 2.746, top-25 2.819 (worse than CE); at 100B tokens 2.480, 2.483, 2.477. Cached targets cost about 10% over CE.
- [Z]: best teacher about 2.5x the student (prune-then-distil, GPT-2/Pythia). [G3 Fig. 8]: small teacher better for short runs, large for long.

## Smallest scale tested, and whether the result held there
143M parameters (sweeps use 198M students), on essentially unrepeated data. Weaker students gained most: a 198M student at 20 tokens/parameter beat supervised with every teacher [P E.4]. Nothing is under 100M. [A] Extrapolated to 3M the law predicts 3.39, below its own infinite-data floor of 3.73: do not extrapolate.

## Training recipe and gotchas
- µP, lr 0.01, weight decay 1e-4, cosine to 1/1000, 2000 warm-up steps, clip 1.0 [P App. I]. Teacher and student see different splits of one distribution.
- Offline targets must use the same sequence packing as the student [SLS D.3].
- Adapt a foreign teacher to the corpus first: 2.994 CE, 2.982 unadapted, 2.960 adapted [SLS Table 13].
- Suggested, untested: distil first, then switch to supervised [P D.5].

## Mapping onto the GBA design
- Inference: no change to lookups, adds, multiplies, RAM or ROM.
- Training [I]: a teacher 10x the student's FLOPs adds 3.3x student cost online, about 10% cached. Sampled targets take 3 bytes per id (14-bit id, 6-bit count): about 36 GB per billion tokens.
- Two-level head [I]: each sampled id gives a class target and a member target, so only the sampled classes' member softmaxes are evaluated.
- Regime [I]: our student sits at thousands of tokens per parameter, where the gain tends to zero. But our unique text is limited and repeated, which this paper does not cover (Baby Llama does).
- We must train the teacher (own tokenizer). That is the paper's losing case, unless cached targets serve many ablation students.
- Pick the teacher by loss, not size; ternary lowers student capacity, so a weaker teacher may be optimal [I].
- LLM-generated text is sequence-level distillation, explicitly out of scope [P App. A]. Sample it near temperature 1; greedy text resembles top-1 targets [I].
- Predictive typing needs calibrated probabilities: avoid top-k targets [I].

## Verdict
**Test, cheaply; adopt the recipe if used** (τ = 1, forward KL, λ 0.7-1, sampled targets). Expect a few hundredths of a nat at long horizons [I]. Experiment: full-precision teachers at three loss levels; ternary student with CE, sampled KD at λ = 1 and 0.7, and KD-then-CE, at about 0.3B and 3B tokens, 2-3 seeds, scored on real held-out text. Keep distillation only if it wins at the long budget.

## Could not verify
- Contour figures were not readable, so crossover compute values are missing.
- c1 = 522.6 has a zero-width interval; it does reproduce sensible in-range values [A].
- Paper inconsistencies: C4 is "180B" tokens (App. I) and "2T" (G.4); D.5 credits a saving (500M student at 3/4 the compute) to a scenario its preceding paragraph calls always worse.
- No official code found. Nothing on ternary or sub-100M students, repeated data, or synthetic text.

## Links worth following
- https://arxiv.org/abs/2503.16870 (sampled targets; code said to be released)
- https://arxiv.org/abs/2311.07052 (capacity-gap law)
- https://arxiv.org/abs/2503.19786 (Gemma 3, section 5.4)
