# Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty  (https://arxiv.org/abs/2308.02019, 2023, Inar Timiryasov and Jean-Loup Tastet)

Read in full: arXiv HTML v2 and the repository code. Tags: [P] paper, [C] code (github.com/timinar/BabyLlama), [B2] BabyLlama-2 (arXiv 2409.17312, read in full), [YP] Yam and Paek, BabyLM 2024 (read in full), [I] my inference.

## What it is
A BabyLM 2023 strict-small entry (10M words). Two large teachers (GPT-2 705M, LLaMA 360M) are trained from scratch on the 10M words; a 58M LLaMA student is then trained from scratch on the same data against the hard labels plus the teachers' averaged logits. The student beats a same-size undistilled model, each teacher, and the teachers' logit-average ensemble on most BLiMP categories. Appendix B lists null results, including mixture-of-experts at width 256.

## Mechanism
`L = a*CE(y, softmax(z_s)) + (1-a)*T^2*KL(softmax(z_t/T) || softmax(z_s/T))` [P eq. 1; C], where `z_t` is the mean of the teachers' logits (not probabilities), a = 0.5, T = 2, used from step 0. Teachers are frozen and run online each step.

- Student [P]: LLaMA, 16 layers, width 512, SwiGLU intermediate 1024, 8 heads, untied embeddings; 58M parameters, about 42M outside the embeddings [I].
- Teachers [P]: GPT-2 24 x 1536, 16 heads, 6 epochs, batch 256, lr 2.5e-4; LLaMA 24 x 1024, intermediate 3072, 8 heads, 4 epochs, batch 128, lr 3e-4.
- Student training [P]: batch 32, lr 3e-4, 6 epochs, cosine schedule, 200 warm-up steps; the code adds weight decay 0.1 and fp16 [C].
- Data [P]: one 16,000-token BPE tokenizer shared by all models; the token stream is cut into contiguous 128-token chunks, reshuffled each epoch.

## Numbers
All at 10M words, 2023 pipeline. Means over the 12 BLiMP categories of Table 1 are my arithmetic.

|model|BLiMP|supplement|
|---|---|---|
|LLaMA 58M, undistilled|70.3|54.5|
|LLaMA 360M teacher|69.6|59.4|
|GPT-2 705M teacher|71.1|59.7|
|Teachers' logit average|71.8|58.8|
|Baby Llama 58M|74.0|59.5|
|OPT-125M / RoBERTa-base / T5-base|62.6 / 69.5 / 58.8|54.7 / 47.5 / 43.9|

- The student beats the undistilled 58M and the ensemble on 10 of 12 categories each; it loses Island Effects (51.2 against 57.3) [P Table 1].
- Cost [P]: student under 3 h, GPT-705M about 12 h, Llama-360M about 2 h, on one RTX 3090.
- Appendix B, GPT2-10M (8 layers, width 256, 16 heads) [P]: averaging the logits of 4 seeds lowers eval loss from 3.77 to 3.66 and adds 1-2 BLiMP points; curriculum gives 3.75 against 3.74; a Switch Transformer got worse in loss and zero-shot scores as experts and capacity grew (no figures given).
- BabyLlama-2 [B2 Table 3]: 345M student, two same-size same-architecture teachers, T = 1, a = 0.5, weight decay 5: BLiMP 73.2 against 71.9 / 72.1 (second run 71.8 against 70.9 / 70.5). A 265-run teacher hyperparameter sweep did not reach the students' cross-entropy. Validation loss explained BLiMP with R^2 = 0.86.
- Yam and Paek [YP Tables 2-3], 2024 pipeline: a GPT2-44M student (2 layers x 768) distilled from GPT2-44M + Llama-60M scores 66.8 against 65.6 / 63.5; BabyLlama-58M 69.8. Undistilled GPT2-18M (2 x 320) 63.7; Llama-20M (2 x 384) 62.8; GPT2-705M 66.1.

## Smallest scale tested, and whether the result held there
Distillation was tested down to a 44M student [YP]: the gain held but was small (+1.2 over the same-architecture teacher, no seed variance reported). The paper itself only distills into 58M. Width-256 models appear only in the null-result appendix. Every result has more parameters than training tokens; the authors list this as a limitation.

## Training recipe and gotchas
- Loss scaling [C, I]: the code calls `KLDivLoss(reduction="batchmean")` on [batch, 128, vocab] logits. That divides by the batch size only (PyTorch docs), while the cross-entropy is a per-token mean, so the soft term weighs about 128 times more than "a = 0.5" suggests. The student learns almost only from the teachers. Choose this weighting deliberately.
- T = 2 was untuned; the tuned follow-up chose T = 1 [B2].
- More than two teachers did not help the 58M student [P].
- Benchmarks are much noisier across seeds than loss [B2]; compare on loss with several seeds.
- Paper and code disagree on student lr (3e-4 against 2.5e-4), teacher epochs and batch, and the random chunk offset.
- LLaMA reaches its minimum loss in about half the epochs of GPT-2 [C README].

## Mapping onto the GBA design
- Inference: no change in lookups, adds, multiplies, RAM or ROM. Only the training target changes.
- Training: one extra forward pass per teacher per step, or cache each teacher's top-k probabilities per corpus position [I].
- Two-level head [I]: sum the teacher's member probabilities into class probabilities; apply the soft loss to the class distribution and to the members within a class.
- Teachers must share our 16K tokenizer, so the open-weight LLM that writes our data cannot supply logits. We train full-precision teachers ourselves; a 16M LLaMA trains "in a few minutes" on 10M words [B2].
- Warning for the 1-of-64 routed MLP: at 8 layers x 256 on 10M words, more experts was worse [P].
- Full-precision reference near our per-token compute: 2 layers x 320-384 scores BLiMP 63-64 on 10M words [YP].

## Verdict
**Test.** It is free at inference and cheap to train, but the evidence is all from the data-starved regime with full-precision students of 44M or more. Ours is compute-starved and ternary, and we can generate more data. Settling experiment: train two full-precision teachers on our corpus (our architecture untied from the cycle budget, and a 16 x 512 LLaMA); train the ternary student with (a) hard labels only, (b) per-token a = 0.5, T = 1, (c) the code's soft-dominated weighting; compare held-out loss over 3 seeds. Repeat (a) with 1, 8 and 64 experts to check the Switch null result.

## Could not verify
- The Switch Transformer degradation has no numbers; the 2023 paper has no seed variance.
- The corpus token count is not stated.
- The 128x weighting is my reading of the code and the PyTorch documentation; the authors do not discuss it. I saw only the titles of the repository's issues.
- Nothing I read tests ternary students or students under 10M parameters.

## Links worth following
- https://arxiv.org/abs/2409.17312 (BabyLlama-2)
- https://aclanthology.org/2024.conll-babylm.27.pdf (smaller teachers and students)
- https://github.com/timinar/BabyLlama (trainer and configs)
- https://arxiv.org/abs/2502.08606 (Distillation Scaling Laws; not read here)
