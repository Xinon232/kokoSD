# GPT or BERT: why not both? (GPT-BERT)

https://arxiv.org/abs/2410.24159 (v2, Dec 2024; Lucas Georges Gabriel Charpentier and David Samuel, University of Oslo; BabyLM 2024 winner). Code: https://github.com/ltgoslo/gpt-bert

Tags: **[P]** paper text (arXiv HTML v2, read in full); **[C]** official repo code; **[M]** my inference.

## What it is
A training objective that lets one transformer be trained on causal LM and masked LM at once, with no extra parameters. The trick is "masked next-token prediction" (MNTP): a masked token at position k+1 is predicted at position k, so the output at k always means "token k+1" in both modes. It also adds attention gating, layer weights and two schedules to LTG-BERT. All tests use 10M or 100M words.

## Mechanism
**Objective [P, C].** Each training sequence is either causal or masked:
- Causal: input `<s> t1..tn`, target at k = t(k+1), lower-triangular attention mask.
- Masked: choose positions with rate p; replace the input by `<mask>` (80%), a random token (10%) or itself (10%); the target at k is t(k+1) only if k+1 was chosen, else ignored; attention is fully bidirectional.
- Loss: cross-entropy for both, all weights shared. The code adds a z-loss, 1e-4 x mean(logsumexp(logits)^2), and keeps an EMA of weights (decay 0.999) [C].
- Mixing: the corpus is duplicated; a fixed fraction of each batch is masked. The code default 15/16 is the *masked* share; each GPU is wholly one type [C].

**Block [P].** `x = norm(x); g = gate(x); x = attention(x) or linear(x); x = glu(x, g); x = norm(x); return residual + output(x)` (parameter-free norms, GEGLU).
**Layer weights [P].** Input of sub-layer i = sum over j <= i of alpha_ij x output_(j-1), alpha unconstrained scalars; attention and feed-forward count as separate layers.
**Schedules [P, C].** Mask rate 30% falling linearly to 15%. Batch 1M rising linearly to 4M tokens. Sequence length 128, then 256 from 70% of steps, 512 from 90%, at constant tokens per batch [C].

## Numbers
Table 1 [P], final models, bidirectional evaluation, 1:15 causal:masked:

| Track | Model | BLiMP | BLiMP-S | GLUE | EWoK |
|---|---|---|---|---|---|
| 10M | LTG-BERT | 80.6 | 69.8 | 74.5 | - |
| 10M | GPT-BERT 30M | 81.2 | 69.4 | 76.5 | 54.6 |
| 100M | LTG-BERT | 85.3 | 76.6 | 77.9 | 56.0 |
| 100M | GPT-BERT 119M | 86.1 | 76.8 | 81.5 | 58.4 |

Ratio sweep (Section 4.2, Fig. 2) [P]:
- Bidirectional evaluation: best from 1:7 to masked-only. Going from causal-only to 6.25% MNTP gives +4.2 BLiMP, +0.9 MNLI, +33.3 LAMBADA.
- Causal evaluation: best at 1:1 and 1:3 causal:masked, "roughly the same"; no numbers in the text. Causal evaluation is worse than bidirectional throughout.
- 20-shot SST-2, causal mask, 100M words (Fig. 3): hybrids score +0.5 to +5.7 over the 51.8% majority baseline (best 3:1); single-objective models fall below it.

Ablation, 10M (Table 2) [P]; base PPL 10.8, BLiMP 81.2, MNLI 80.1, EWoK 54.6. Change in PPL / BLiMP / MNLI / EWoK when removed:
- layer weights: +0.4 / -1.3 / +0.2 / +0.6
- attention gate: +0.3 / -0.3 / +0.3 / -0.9
- mask schedule: +0.1 / -0.1 / -0.7 / -0.6
- batch schedule: +0.7 / -1.1 / 0.0 / +0.8

Other [P]: layer weights slow training almost 1.5x; batch scheduling halves training tokens; greedy decoding loops without repetition penalty 1.5.

## Smallest scale tested, and whether the result held there
30M parameters (12 layers, width 384, FF 1280, 6 heads, vocab 8,192) on 10M words. The hybrid edges out LTG-BERT there on three of four suites (+0.6 BLiMP, -0.4 BLiMP-S). Fig. 2's scale is not stated. Nothing smaller or quantised was tested.

## Training recipe and gotchas
- LAMB, betas 0.9/0.98, weight decay 0.1, clip 2.0, cosine with 1.6% warmup; LR 0.01 to 0.001 (base), 0.0141 to 0.00141 (small); 15,625 or 7,812 steps; dropout 0.1 (Table 3) [P].
- The headline model is tuned for bidirectional use. Our use is causal, where the paper's own best ratio is 1:1 to 1:3 and the gain is shown only in a figure.
- Masked sequences supervise only 15-30% of positions [M].
- BLiMP uses a per-model optimal temperature (about +2 points in masked mode) [P].

## Mapping onto the GBA design
Zero run-time cost: no lookups, adds, multiplies, RAM or ROM. The cartridge only ever runs the causal mode; bidirectional mode would recompute every position for each token and is not usable [M]. Changes needed in training [M]:
- Hashed n-gram embeddings: a `<mask>` inside the n-gram window gives a meaningless hash. Zero the n-gram feature when its window contains a mask.
- The two-level head is unchanged.
- Masked examples need attention over future positions: a training-mask change only.
- The attention gate adds a width x width ternary matrix per attention layer (about 65K-147K weights at width 256-384, 5-10% of the per-token budget); skip unless it pays in a ternary ablation. Layer weights need every earlier layer output kept in RAM (width x layers x 2 bytes) plus scalar multiplies; skip.

## Verdict
**Test.** Free at inference and cheap to try, but evidence for causal use is thin: a figure without numbers, 30M+ float parameters. Experiment: train our model on the 10M corpus with masked share 0, 6.25%, 25% and 50% at equal total tokens, and compare causal validation loss, causal BLiMP and a name-recall probe.

## Could not verify
- Exact values in Figures 2-5 (plots only), including the size of the causal-mode gain.
- Any result under quantisation or below 30M parameters.

## Links worth following
- LLM2Vec (BehnamGhader et al. 2024), origin of the MNTP formulation used
- Ankner et al. 2024, dynamic masking-rate schedules
