# MiniPLM: Knowledge Distillation for Pre-Training Language Models  (https://arxiv.org/abs/2410.17215, ICLR 2025; Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang; code github.com/thu-coai/MiniPLM, MIT)

Read: arXiv HTML (main text and appendices B-F, through an extraction tool, so figures were not seen) and the repository README. Tags: [P] paper text/table, [R] README, [I] my inference.

## What it is
An offline distillation method that changes only the pre-training corpus. A large teacher and a small reference model both score every training sequence; the half of the corpus where the teacher beats the reference by the most is kept, and the student is trained on it from scratch with the ordinary next-token loss. No teacher runs during training, no logits are stored, and the student's tokenizer and architecture are unconstrained. Tested on 200M-1.2B students trained on 50B tokens of the Pile.

## Mechanism
- Score per instance [P eq. 4]: `s(x) = log p_teacher(x) − log p_ref(x)`, i.e. reference loss minus teacher loss. Instances are fixed 1,024-token sequences built by merging documents, so sum and mean are equivalent [P].
- Selection: deterministic `top-K`, `K = α·|D − D_ref|` [P]. No stochastic sampling is described.
- Pipeline [P]:
  1. Sample `D_ref` uniformly from `D` (5B of 105B tokens).
  2. Train the reference LM on `D_ref` (104M: 8 layers, d=512, FFN 1,408, 8 heads).
  3. Run teacher (Qwen1.5-1.8B) and reference over `D − D_ref`; store one score per sequence.
  4. Keep the top α = 0.5 (50B tokens); train the student with plain cross-entropy [P eq. 5].
- Three effects [P §2.5]: common patterns both models fit are down-sampled; text the teacher fits and the reference does not is kept; text where teacher ≤ reference (noise) is dropped.
- Cheaper variant [P App. E]: fit a proxy scorer on a 0.1B-token scored subset, about 0.2% of the compute.

## Numbers
All at 50B training tokens, 1.8B teacher, average zero-shot accuracy over 9 tasks.

|student|no KD|vanilla KD|SeqKD|MiniPLM|MiniPLM + vanilla KD|
|---|---|---|---|---|---|
|200M|39.9|39.9|39.7|41.3|40.7|
|500M|43.2|43.6|42.3|44.8|44.9|
|1.2B|44.9|45.4|44.3|46.6|48.1|

[P Tables 1, 5]
- Test loss extrapolated to 1T tokens, 200M: 3.35 no KD, 3.39 vanilla KD, 3.28 MiniPLM [P Table 2].
- Cross-family [P Table 3]: Llama3.1 212M 41.0 → 41.8 (loss 3.52 → 3.43); Mamba 140M 41.6 → 42.6 (loss 3.24 → 3.15).
- Proxy variant, 200M: 40.9 against 41.3 full [P Table 12].
- Data-limited [P §3.4, App. D.2]: 50B-token corpus, α = 0.25, 4 epochs is best; no-KD needs about 118B tokens to match, the "2.4×" claim (extrapolated from a scaling-law fit).
- Compute [P Fig. 1a]: matches vanilla KD on the 500M student at 2.2× less compute.
- Diversity score [P Table 4]: original 32.25, teacher-generated (SeqKD) 30.16, difference-sampled 36.70.
- Discarded as noise: 6.3% of the Pile [P App. F].
- Reference size, agreement with the student's own ranking [P Table 11]: Spearman 0.774 (45M), 0.840 (104M), 0.879 (200M).
- Wall-clock on 8 A100 [P Tables 8-9]: teacher inference 68 h, reference inference 7 h, reference training 2.6 h, student training 68 h; vanilla KD 139 h.

## Smallest scale tested, and whether the result held there
Smallest student: Mamba 140M (README says 130M); smallest transformer 200M. The gain held: +1.0 to +1.4 accuracy points. Nothing below 140M, nothing quantised, no simple-English corpus. At 200M, adding vanilla KD on top made it worse (40.7 against 41.3), and vanilla KD alone gave nothing, so the trend toward small students favours data selection over logit matching [I].

## Training recipe and gotchas
- AdamW, β = (0.9, 0.98), weight decay 0.1, batch 512 × 1,024 tokens, 100K steps, 2K warm-up, cosine to 1/10 of peak; lr 6e-4 (200M), 3e-4 (500M), 2.5e-4 (1.2B) [P App. B].
- Teacher size matters: for the 200M student a 500M teacher was best; 300M too weak, 4B makes the score scale too small against the reference [P Fig. 6, no numbers in text].
- Bigger references help, with saturation; smaller α (0.3-0.9 tested) helps but needs a larger raw corpus [P App. D.3].
- Scores need 1,024-token windows of equal length; with variable-length dialogues use per-token mean loss difference [I].
- The method raises difficulty: it keeps what a 104M model cannot fit. Our student is far below the reference, so the kept half may be exactly what it cannot learn [I].

## Mapping onto the GBA design
Zero inference cost: no extra lookups, adds, multiplies, RAM or ROM. It is a corpus-preparation step. Cost on our side [I, scaled from Tables 8-9]: 8 A100 × 68 h for 105B tokens is about 5 A100-hours per billion tokens for a 1.8B teacher, so a 1-3B-token corpus costs roughly 5-15 GPU-hours plus a small reference run. Changes needed: reference near our student's effective capacity (or the float version of the student itself); teacher about 0.5B; score per conversation with length normalisation. The noise-discard tail (teacher ≤ reference) and the down-weighting of templated synthetic dialogues are the parts most likely to transfer. The same scores could weight which n-grams get ROM table slots, but the paper does not test that [I].

## Verdict
Test. Cheap (about 5% of the GPU budget), needs no shared tokenizer, and stacks with later logit distillation, but every result is at 140M+ with a 50B-token budget. Settling experiment: score the corpus once; train the float student for a fixed token budget on (a) the raw corpus, (b) top-50% by score, (c) raw minus the bottom ~6%; compare held-out simple-English loss and dialogue evals. Adopt (c) if (b) hurts.

## Could not verify
- Figures 1, 5, 6, 8 (curves) were not seen; teacher-size and α ablations have no numbers in the text.
- The README summary gave the score as a ratio of log-probabilities; the paper's eq. 4 is a difference. I could not open `compute_difference_scores.py` (404 at the guessed path).
- How the Qwen-tokenised windows are re-tokenised for Llama/Mamba (a conversion script exists [R]; details unread).
- No evidence for students under 140M or for ternary training.

## Links worth following
- Code and released 50B-token sampled corpus: https://github.com/thu-coai/MiniPLM
- MiniLLM (reverse-KL baseline, same first author): https://arxiv.org/abs/2306.08543
