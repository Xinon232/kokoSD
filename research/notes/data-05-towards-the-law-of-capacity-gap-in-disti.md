# Towards the Law of Capacity Gap in Distilling Language Models  (https://arxiv.org/abs/2311.07052, 2023; v4 July 2025; Chen Zhang, Qiuchi Li, Dawei Song, Zheyu Ye, Yan Gao, Yan Hu)

Read in full: arXiv HTML v4 and the PDF figures. Tags: [P] paper, [F] read off a figure by eye (approximate), [C] code at github.com/GeneZC/MiniMA, [DSL] Distillation Scaling Laws (arXiv 2502.08606, relevant sections), [MM] MiniMoE (arXiv 2305.12129, same first author, relevant sections), [I] my inference.

## What it is
An empirical study of which teacher size gives the best distilled student. GPT-2 (140M-1.6B) and Pythia (70M-2.8B) models are pruned to several sparsities, and each pruned copy is distilled from its parent on OpenWebText. The best student of a given size comes from a teacher pruned by about 60%, so the optimal teacher is about 2.5 times the student. The rule is then applied at 7B to 3B (MiniMA, MiniChat), where larger LLaMA2 teachers give worse 3B students.

## Mechanism
- Law [P eq. 6]: `T* ≈ 2.498·S − 11.498 ≈ 2.5·S`, in millions of parameters [F Fig. 4], R² = 0.9957. Equivalently student = 0.4 × teacher.
- Student initialisation [P]: a pruned teacher, not random weights (this does not transfer to our ternary student). Gates on heads, FFN neurons and hidden dimensions; score = E|∂L/∂gate|, L = teacher cross-entropy on the true tokens, l2-normalised; lowest scores pruned; per-component sparsity `p' = 1 − sqrt(1 − p)`; ~5% of the data.
- Loss [P eq. 4; C]: `L = 0.5·CE(truth, student) + 0.5·CE(softmax(z_teacher), student)`, per-token mean, temperature 1, logits only.
- Pilot grid [P; F Fig. 3]: sparsities 20/50/60/70/80% (GPT-2 also 90-99%); WikiText-2 perplexity and LAMBADA accuracy; the best teacher per student size is fitted by a line.
- Compute model [P eq. 8]: `C = 6·N_S·D + 2·N_T·D`.

## Numbers
Pilot, OpenWebText (3.9B tokens):
- [F Fig. 1] ~300M student from GPT-2 380M/810M/1.6B: perplexity about 197/97/175. ~500M student from Pythia 1B/1.4B/2.8B: about 41/37.5/44.
- [F Fig. 4] roughly 15 fitted points, students from about 25M to about 1.1B.
- [F Fig. 3a-b] GPT-2 students of 5-40M: perplexity about 1,780 at 5M falling to about 530 at 40M; LAMBADA 0-2%.

7B scale, 3B student, 126B tokens [P Table 2]:

|teacher|MMLU|DROP|GSM8K|HumanEval|
|---|---|---|---|---|
|LLaMA2-7B|28.51|22.50|8.11|10.98|
|LLaMA2-13B|26.82|18.92|5.31|10.37|
|LLaMA2-70B|26.91|19.03|3.56|7.32|
|LLaMA3.1-8B|31.72|30.46|10.42|18.46|
|none (ShearedLLaMA: pruned, retrained)|25.39|20.41|5.08|8.54|

- Ablations, MMLU [P Table 4]: MiniMA 28.51; hidden-state matching 28.69; a 5B teacher assistant 27.69; layer dropping 27.59; global pruning 28.92.

## Smallest scale tested, and whether the result held there
The fit stops at a ~25M student. Below about 40M the GPT-2 curves coincide whatever the teacher, so the law is not shown there [F, I]. The intercept gives T* = S at 7.7M [I]; the line cannot be extrapolated to our size. No seeds or error bars.

## Training recipe and gotchas
- MiniMA [P; C]: 24 layers × 3072; 4,096-token sequences, batch 1,024, lr 3e-4, weight decay 0.1, 1 epoch, 1% warm-up, cosine decay.
- Pilot steps, lr and batch are not reported. Quality is low: a ~1.3B student of GPT-2 xl has perplexity about 75 [F].
- The law is nearly built in [I]: "best teacher" means "best sparsity", 60% on a five-value grid, for students that inherit teacher weights.
- Training loss does not rank teachers; the undistilled run has the lowest [P App. E]. No benchmark row exists for that undistilled student.
- [DSL], 143M-12.6B: teacher size and tokens act only through teacher cross-entropy; the gap sets in at a fixed ratio of teacher loss to the student's own supervised loss; the linear law is "a special case" for compute-optimal teachers; best temperature is 1; if a teacher must be trained for a single student, supervised training is generally preferable.
- [MM], 3 layers × 384 on GLUE: moving from a BERT-base to a BERT-large teacher changes the dense student by −0.5 and a student with top-1 routed FFN experts by +0.5. Learned gating beat hash routing; "smaller students favor fewer experts".

## Mapping onto the GBA design
- Inference: no change in lookups, adds, multiplies, RAM or ROM.
- Training cost [I, eq. 8]: with N_T = 2.5·N_S, online distillation costs 11·N_S·D against 6·N_S·D (1.8 times per token) plus teacher training. Caching teacher top-k probabilities on disk pays the teacher pass once.
- Teachers [I]: our own full-precision models on the 16K tokenizer, as a size ladder. Which "S" applies to us is undefined: about 2M weights touched per token, or 50-120M stored. [MM] suggests routed capacity counts.
- Two-level head [I]: sum teacher probabilities per class for the class loss; renormalise within the class for the member loss.

## Verdict
**Test.** Keep the qualitative rule (a bigger teacher is not a better one), not the 2.5 constant. Settling experiment: fixed ternary student and token budget; hard labels against eq. 4 with teachers of 1, 2.5, 6 and 15 times the student; 3 seeds; held-out hard-label loss; plus a hard-label run with 1.8 times the tokens. Repeat with 1 and 64 experts.

## Could not verify
- Pilot hyperparameters, the "fewer tokens" count, exact figure values.
- Whether the 13B and 70B teachers received the 7B teacher's Chinese-vocabulary adaptation.
- Any ternary, from-scratch or sub-5M student result. I did not find the pilot code in the repository files I opened.

## Links worth following
- https://arxiv.org/abs/2502.08606 (Distillation Scaling Laws)
- https://arxiv.org/abs/2305.12129 (MiniMoE: routed experts against the capacity gap)
- https://github.com/GeneZC/MiniMA
