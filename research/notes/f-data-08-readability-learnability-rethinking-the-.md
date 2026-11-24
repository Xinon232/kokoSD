# Readability ≠ Learnability: Rethinking the Role of Simplicity in Training Small Language Models (LlamaTales)  (https://arxiv.org/abs/2510.13915, v1 15 Oct 2025, Ivan Lee and Taylor Berg-Kirkpatrick, UC San Diego; HF collection says COLM 2025)

Read in full: arXiv HTML v1 (all text, Tables 1, A1-A4, prompt figures A10-A24), figure images 3a/3b/4a/4b/A2a/A2d viewed directly, plus the Hugging Face collection (one model config, dataset card, 8 sample rows). Tags: [P] paper text/table, [Fig] value read off a plot by eye (±2), [HF] Hugging Face, [I] my inference.

## What it is
A controlled re-run of TinyStories with an open-weight generator. Llama-3.1-8B-Instruct writes ~1B-token corpora from one prompt template at two reading levels (child "Jr", college "GRE") and three GRE-level domains (history, sports, news). Six students (262K-33M non-embedding parameters) are trained per corpus and compared with FineWeb-Edu students and public models. Result: reading level does not matter; low n-gram diversity does; and every student collapses on prompts from any corpus other than its own.

## Mechanism
**Generation [P Fig. A20-A24; HF rows].** TinyStories recipe: system persona + "Write a short story (3-5 paragraphs) which only uses simple words that a 5 year old child would understand. The story should use the words: W1, W2, W3. The story has the following features: F1..FK" (features seen: MoralValue, Twist, BadEnding, Foreshadowing; K may be 0). GRE variant: drop the simple-words clause, college-graduate persona, GRE word list. Sampling stored per row: temperature 1.0, max_tokens 512, top_p 1.0 (Jr rows) or 0.95 (GRE rows) [HF].

**Students [P Table A2; HF config].** Llama architecture (RMSNorm, RoPE, SwiGLU with intermediate = 4d, no biases, untied embeddings), context 512, Llama-3 tokenizer (vocab 128,256). Layers/heads/width: 1/2/128 (262K), 2/2/128 (524K), 4/2/128 (1.05M), 4/6/384 (9.44M), 8/6/384 (18.9M), 8/8/512 (33.6M). 10 epochs x 1B = 1e10 tokens each.

**Metrics.**
- Coherence and readability: Llama-3.1-70B-Instruct, 1-100, single-number answer, prompts in Fig. A10/A11 [P]. Chain-of-thought and few-shot judge prompts made "no meaningful differences" [P Fig. A30].
- Evaluation: 1,000 completions per model-corpus pair, top-p 0.95, from 50-token test-split prefixes [P].
- Statistical simplicity: unique 1- to 8-grams in a 100M-token sample [P Fig. 2].
- Learnability ratio = student output coherence / training-corpus coherence [P].

## Numbers
Corpus properties [P Tables 1, A1, A3]:

|corpus|coherence|readability|FK grade|tokens/doc|unique 2-grams|unique 4-grams|ppl under 7-8B LMs|
|---|---|---|---|---|---|---|---|
|TinyStories|90.1|92.6|2.4|186.5|2.89e6|8.75e7|3.9|
|LlamaTales-Jr|89.5|92.7|2.2|282.8|3.71e6|1.10e8|5.2|
|LlamaTales-GRE|94.4|64.8|9.6|500.6|7.90e6|1.61e8|5.5|
|FineWeb-Edu|77.4|68.2|10.7|497.3|4.26e7|5.74e8|9.3|

Other train-split coherence [P Table A1]: Sports 92.4, News 94.5, History 91.0, Dolma 60.2, SlimPajama 67.3.

In-distribution coherence by size [Fig 3, A2]:

|student|Jr on Jr|GRE on GRE|FineWeb on FineWeb|
|---|---|---|---|
|128x1 (262K)|~20|~22|~7|
|128x4 (1.05M)|~28|~41|~8|
|384x4 (9.4M)|~67|~82|~16|
|384x8 (18.9M)|~76|~87|~18|
|512x8 (33.6M)|~80|~89|~21|
|Llama-3.1-70B|~81|~92|~70-73|

- Public ~85M-parameter models (GPT-2 small and pythia-160m; dots not separable) score ~8-10 on Jr prompts, ~17-18 on GRE prompts, ~14-18 on FineWeb prompts; GPT-2-XL ~20 / ~43 / ~38 [Fig].
- Off-distribution, 33M students: GRE-trained on Jr prompts ~26; Jr-trained on GRE prompts ~14; TinyStories-trained on Jr ~76 and Jr-trained on TinyStories ~70 (the one exception); every synthetic-trained student on FineWeb prompts ≤ ~11 [Fig 3, A2].
- Learnability ratio: synthetic corpora 0.88-0.94; FineWeb ~0.27, Dolma ~0.29, SlimPajama ~0.23, against unique 3-grams of ~0.7-1.5e7 versus ~4.4-4.8e7 [Fig 4a].
- Speed: GRE-trained model at ~85.5 after epoch 1, ~89 at epoch 10; Jr ~67 to ~80; TinyStories ~70 to ~81 [Fig 4b; model size not stated].

## Smallest scale tested, and whether the result held there
128 wide, 1 layer. Coherence is poor at every width-128 size (20-41) even with 4 layers; the jump is 128x4 to 384x4 (28 to 67 on Jr). Readability made no difference at any size (GRE is ahead everywhere).

## Training recipe and gotchas
- No optimizer, learning rate, batch size or hardware is given in the paper or model cards (cards are empty templates).
- Table 1 and Table A1 disagree: GRE readability 64.8 vs 72.7, FineWeb coherence 77.4 vs 77.6.
- Table A3's n-gram counts cannot be from 100M-token samples (8-gram counts exceed 1e8) [I]; treat them as full-corpus counts, and Fig. 4a's 3-gram counts as 100M-sample counts.
- Sample rows repeat names and settings ("Benny" in 2 of 4 Jr rows; "Ravenswood"/"Ravenshire" in GRE) [HF, tiny sample]: generator mode collapse is part of the "simplicity".
- Authors did not train on mixed corpora; they say this "might mitigate" brittleness.

## Mapping onto the GBA design
No inference cost: this is a data and evaluation result.
- A Llama block is 16d² weights. 128x4 = 1.05M weights per token, right at our 1.3-2M budget, and scores ~28; 384x4 = 9.4M scores ~67 [I]. Dense compute alone buys incoherent output; routed experts, top-8 attention and ROM tables must close a 5-7x compute gap.
- Low n-gram diversity is exactly what ROM lookups exploit: Jr has 3.7M distinct bigrams, FineWeb 42.6M. Hashed n-gram tables cover a narrow synthetic corpus far better per ROM byte [I].
- Unique n-grams per 100M tokens is a cheap corpus dial to track while widening data.

## Verdict
**Adopt** the generator recipe, the judge protocol and the diversity metric. **Test** breadth: train the same student on (a) one narrow template, (b) a mix of 5-10 templates/domains plus chat, at fixed tokens; score in- and cross-distribution coherence and learnability ratio. This settles how wide the corpus can go before the ratio falls toward 0.3.

## Could not verify
- All per-model scores are eyeballed from plots; no numeric table exists.
- Training hyperparameters; which model Fig. 4b uses; exact GPT-2 small dot.
- Domain-variant datasets are not in the HF collection.

## Links worth following
- https://huggingface.co/collections/ivnle/llamatales-6716dad1a3113c4c3ea1038e
- https://arxiv.org/abs/2406.13069 (n-gram novelty, Rusty-DAWG)
- https://arxiv.org/abs/2404.02204 (emergent abilities in reduced-scale LMs)

## Independent verification
Second reader, 2026-11-19. Re-read from primary sources: arXiv HTML v1 (text, Tables 1, A1-A3), figure PNGs 3a, 3b, 4a, A2 (FineWeb panel) viewed directly, Hugging Face dataset rows (ivnle/llamatales-jr, first rows) and one model config (llamatales_jr_8b-lay4-hs128-hd2-1M). Plot values are my own eyeball readings (±2).

1. **Generator recipe and coherence 89.5 vs 90.1 — confirmed (one sourcing note).** Table 1 and Table A1 both give coherence 90.1 (TinyStories) and 89.5 (LlamaTales-Jr); judge is Llama-3.1-70B-Instruct on a 1-100 scale (Sec. 3.2, Fig. A11); Table A3 gives total tokens 1e9 (TinyStories 9.2e8) and sources "GPT-3.5/4" vs "Llama-3.1-8B". Temperature 1.0 is NOT in the paper; it comes from the per-row fields of the HF dataset (temperature 1.0, top_p 1.0, max_tokens 512), which I confirmed on three rows only. The scores are for train splits of the corpora, not for student output. A 0.6-point gap has no error bar; treat as "equal".
2. **Learnability 0.88-0.94 synthetic vs 0.23-0.29 web — partly.** The plotted values are right (Fig. 4a: TinyStories ~0.90, Jr ~0.89, GRE ~0.94, Sports ~0.94, News ~0.89, History ~0.88; FineWeb ~0.27, Dolma ~0.29, SlimPajama ~0.23). But neither the caption nor the text says which student size Fig. 4a uses. "33M" is an inference; it is a good one (80/89.5 = 0.89, 89/94.4 = 0.94, 21/77.4 = 0.27 match the 33M dots in Fig. 3/A2). 10B tokens (10 epochs of ~1B) is stated in Sec. 4. Missed caveat: the ratio is a largest-model number. Using the same figures, the 128x4 (1.05M) student gives roughly 28/89.5 = 0.31 on Jr and 41/94.4 = 0.43 on GRE, i.e. at our compute scale even the narrow synthetic corpus is recovered at web-like ratios [my arithmetic from eyeballed dots].
3. **Off-distribution collapse — confirmed.** Fig. 3a: GRE-trained 33M on Jr prompts ~26 (TinyStories-trained ~76, the stated exception; FineWeb-trained ~9). Fig. 3b: Jr-trained 33M on GRE prompts ~14 (TinyStories-trained ~21). Fig. A2 FineWeb panel: all synthetic-trained students between ~4 and ~12 (highest is a GRE-trained 128x4 dot at ~11.5, so "at or below ~11" is marginally tight). In-distribution 33M: Jr ~80, GRE ~89. Text agrees: coherence "degrades substantially" on any other dataset's prompts. Caveats: prompts are 50-token prefixes of documents from another corpus, scored by a judge for coherence of the continuation; this is story-continuation shift, not chat. The authors did not train on mixtures ("might mitigate this"), so the paper is silent on whether a mixed corpus fixes it.
4. **"Width matters more than depth" — partly; the numbers are right, the conclusion is not the paper's.** Fig. 3a green dots: 128x4 ~28, 384x4 ~67, 384x8 ~76, and Table A2 confirms 1.05e6 / 9.44e6 / 1.89e7 non-embedding parameters. The paper never makes a width-vs-depth claim, and the six configurations cannot separate the two: there is no 128x8 or 384x1/2, heads change with width (2 to 6), and the 128-to-384 step is a 9x parameter jump versus 2x for 4-to-8 layers. The data are equally consistent with "coherence tracks parameter count" (128x1 to 128x4, 4x parameters, gives only ~20 to ~28). Extra confound for us: the students use the Llama-3 tokenizer (vocab 128,256, untied embeddings, confirmed in the HF config), so a 128-wide model squeezes a 128K vocabulary through a rank-128 output layer, and its softmax alone is 128 x 128,256 = 16.4M multiply-adds per token, far beyond our budget. With a 16K vocabulary the width threshold could sit elsewhere.

**Overall.** Numbers were transcribed accurately, but all per-model values are eyeballed from scatter plots with no table, no seeds and no error bars. The headline "0.9 learnability" belongs to the 33M float student; at ~1M non-embedding weights (our active-compute class) the same plots give coherence ~28-41, so the paper is evidence that dense compute at our scale is insufficient, not that a narrow corpus rescues it. Everything is full-precision, 128K-vocabulary, story-continuation; nothing is ternary, routed or conversational.
