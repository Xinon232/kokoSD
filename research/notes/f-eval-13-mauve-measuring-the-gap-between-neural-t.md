# MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers (https://arxiv.org/abs/2102.01454, NeurIPS 2021 oral; Pillutla, Swayamdipta, Zellers, Thickstun, Welleck, Choi, Harchaoui)

Read in primary form: ar5iv HTML of v3 (main text, Appendix D/E tables) plus the official README (github.com/krishnap25/mauve). Extraction went through a summarising fetch tool, so numbers below are as it reported them from the tables; a few cells were unreadable and are listed under "Could not verify".

## What it is
A judge-free, reference-distribution metric for open-ended generation. It compares a set of model generations Q with a set of human texts P, not text-by-text but as distributions, and returns one scalar in (0,1] (1 iff Q=P). It summarises two error types at once: model text that humans would not write (Type I) and human text the model cannot produce (Type II). It was validated on GPT-2 small/medium/large/XL, Grover, and several decoders, and correlated with human pairwise judgments better than Gen-PPL, Zipf, Self-BLEU, etc.

## Mechanism
1. Sample n texts from each side (paper: n=5,000; same prompts, human continuation vs model continuation).
2. Embed each text with a frozen LM: terminal (last-token) hidden state of GPT-2 large (README default `featurize_model_name='gpt2-large'`).
3. Pool P and Q features; PCA keeping 90% explained variance; L2-normalise; k-means (FAISS, up to 500 iters, 5 redos, keep best). k=500 default in the paper; README: k ~ n/10.
4. Histogram cluster memberships -> discrete P~, Q~ over k bins.
5. For lambda in a grid over (0,1) (README default 25 points): R_l = l*P~ + (1-l)*Q~; point = (exp(-c*KL(Q~|R_l)), exp(-c*KL(P~|R_l))), c=5.
6. MAUVE = area under that curve.

```
mauve.compute_mauve(p_text=human, q_text=model, featurize_model_name='gpt2-large',
    max_text_length=..., num_buckets=n//10, mauve_scaling_factor=5,
    kmeans_explained_var=0.9, kmeans_num_redo=5, kmeans_max_iter=500)
```
Features can be supplied directly (`p_features`, `q_features`, shape (n,d)), so any embedder works.

## Numbers
Web text = GPT-2 pretrained, 35-token WebText prompt, max generation 1024 tokens, 5,000 generations (Table 2). Std over 5 runs.

GPT-2 small (124M), web text (Tables 6/9/11):
- nucleus p=0.9: **0.878 +/- 0.006**; ancestral sampling: 0.589 +/- 0.018; greedy: 0.008; adversarial: 0.043
- beam b=4: 0.021; b=4 no-4gram-repeat: 0.026; b=8: 0.366; b=8 no-repeat: 0.341
- Human Bradley-Terry score: nucleus -15.78 (Table 6; -15.9 in Table 4), sampling -27.52

Other sizes, nucleus / ancestral (Table 6): medium 0.915 (p=0.9) / 0.373; large 0.936 (p=0.95) / 0.845; XL 0.940 (p=0.95) / 0.882. Greedy <= 0.016 at every size.
Stories, GPT-2 medium finetuned on WritingPrompts (Table 8): sampling 0.905; nucleus 0.9/0.92/0.95 = 0.887/0.901/0.920; top-100 0.884; top-500 0.919; greedy 0.005.
Authors' preferred decoders for GPT-2 large (Fig. 8): p=0.95, top-K with 100<=K<=1000.

Human correlation (Table 5, Spearman vs Bradley-Terry over 8 model/decoder settings; 214 MTurk raters, 90 ratings per pair, texts truncated to 256 tokens): MAUVE 0.952 human-like, 0.810 interesting, 0.857 sensible. Gen-PPL 0.810/0.643/0.738; Self-BLEU 0.595/0.405/0.524.

Robustness: ranks for k in 100..5000 correlate 0.99-1.00 with k=500; k<100 pushes everything toward 1, k>2000 toward 0. RoBERTa-large features: Spearman 0.993 with GPT-2-large features, but absolute values differ a lot (small/sampling 0.174 vs 0.589; small/nucleus 0.723 vs 0.878) (Table 12). MAUVE falls monotonically as max generation length grows (Fig. 4/6, no numbers given).

## Smallest scale tested, and whether the result held there
GPT-2 small (124M) is the smallest model. The size ordering held under nucleus sampling (0.878 < 0.915 < 0.936 < 0.940) and matched human ranking. It did NOT hold cleanly under ancestral sampling: medium (0.373) scored below small (0.589), and humans also preferred small there. Nothing below 124M, and no quantised or simple-English model, was tested.

## Training recipe and gotchas
No training. Gotchas:
- Absolute scores are not portable: they depend on n, k, c, featuriser, text length and domain. README: use for relative comparison with identical settings.
- Small n is optimistically biased and noisy; use a few thousand per side.
- Extremely decoder-sensitive: the same GPT-2 small spans 0.008-0.878. Decoder must be tuned and reported per model.
- If all scores are < 0.4 or > 0.995, change c (ranks are unchanged).
- It measures distribution match, not per-reply correctness or coherence with the prompt. A model copying the training distribution well but answering wrongly can score high.
- Known disagreements with humans exist (e.g. medium-nucleus vs large/XL-sampling).

## Mapping onto the GBA design
Zero on-device cost: no lookups, adds, multiplies, RAM or ROM. It is an offline PC-side evaluation.
Pipeline for Koko: take ~5,000 held-out prompts from our simple-English/chat test split; P = the reference continuations; Q = Koko continuations produced by a bit-exact integer simulator of the cartridge (not the float QAT model), with the real 256-token ring buffer and real decoder. Featurise with GPT-2 large on the 5090 (minutes), k=500.
Changes needed versus the paper: (a) the 0.878 web-text figure cannot be reused; GPT-2 small must be re-run on our prompts and our length cap, ideally both zero-shot and finetuned on our data, since pretrained GPT-2 will be off-distribution for simple English and would lose for the wrong reason; (b) cap lengths identically (e.g. 64-128 tokens) because MAUVE drops with length; (c) score several Koko decoders (temperature/top-k/nucleus), since top-k is cheap on-device and the metric can pick the setting; (d) add a human-vs-human split (P vs second held-out P) as the ceiling and a greedy run as the floor.
Useful secondary role: decoder selection and regression test across QAT checkpoints (ternary vs float teacher) at fixed settings.

## Verdict
**Adopt as a secondary metric (cheap, de-risks evaluation), not design-driving.** It gives a judge-free second opinion that was validated exactly at GPT-2 scale and that correctly punishes degenerate repetitive output, the main failure of weak models. It will not tell us whether replies are correct or on-topic, so it complements, not replaces, perplexity, BLiMP-style tests and judged dialogue quality. Settling experiment: compute MAUVE for Koko, GPT-2 small (zero-shot and finetuned) and a held-out human split on the same 5,000 prompts with 5 k-means seeds; check that the ordering agrees with a small blind human pairwise test (~200 pairs).

## Could not verify
- Which layer exactly ("terminal hidden state" is from the README; the paper excerpt did not state layer/pooling).
- Numeric text-length results and the c-sensitivity table (figures only / not in the extracted text).
- Grover/news MAUVE values (Table 7 headers unreadable).
- Gen-PPL/Zipf/Self-BLEU columns of Table 6 for GPT-2 small (partly unreadable; Table 4 values look like differences from human, not raw values, so not reproduced here).
- No human-vs-human ceiling value and no compute-time figure found.
- Behaviour on sub-100M, quantised or short-dialogue text: not in this source.

## Links worth following
- https://github.com/krishnap25/mauve (official package; read README)
- "MAUVE Scores for Generative Models: Theory and Practice" (Pillutla et al., JMLR 2023): the longer follow-up with estimation theory and sample-size analysis; not read here.
