# TinyStories: How Small Can Language Models Be and Still Speak Coherent English?  (https://arxiv.org/abs/2305.07759, 2023, Ronen Eldan and Yuanzhi Li, Microsoft Research)

Read in full: arXiv HTML v2 (Figures 3, 4, 23 from the images) plus Hugging Face cards/configs. Tags: [P] paper, [HF] Hugging Face, [I] my inference.

## What it is
A synthetic corpus of short stories written by GPT-3.5/GPT-4 using only words a 3-4 year old knows, and a study of 1M-80M-parameter GPT-Neo models trained on it. Models under 10M parameters (width 256), or with one transformer block, write fluent multi-paragraph stories. It introduces "GPT-Eval" (GPT-4 grades completions like a teacher) and reports a width-by-depth grid of loss and grades, with GPT-2 baselines.

## Mechanism
**Data generation [P, Sec. 2].** About 1,500 basic words split into nouns, verbs, adjectives. Per story: draw one of each at random, draw a random subset of features (dialogue, plot twist, bad ending, moral value, foreshadowing, conflict), then prompt:
"Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would likely understand. The story should use the verb "X", the noun "Y" and the adjective "Z". The story should have the following features: ... Remember to only use simple words!"
The random triple is the diversity mechanism; high temperature alone gave "a very repetitive dataset".

**TinyStories-Instruct [P, Sec. 2.1].** Each story is prefixed by a random subset of `Summary:` (by GPT-3.5), `Features:`, `Sentence:` (a random non-first sentence), `Words:`, then `Story:`.

**Models [P footnote 2; HF configs].** GPT-Neo, context 512, layers alternating global attention and local attention (window 256), 16 heads at every released width (even 64), no dropout. Tokenizer: GPT-Neo BPE cut to the 10K most common tokens.

**GPT-Eval [P, Sec. 3].** About 50 hand-written openings cut mid-sentence (released file: 44 [HF]); 10 completions each at temperature 1; GPT-4 writes a verbal assessment first, then grades grammar, creativity, consistency out of 10 plus an age group; grades are averaged. The Instruct variant adds a plot grade. GPT-2 outputs were truncated at the first repeated 4-gram.

## Numbers
Figure 4 [P], "after a fixed number of training steps"; loss presumably nats/token on the 10K vocabulary [I].

|width x layers|eval loss|grammar|creativity|consistency|
|---|---|---|---|---|
|64x8 (~1M)|2.08|6.14|4.68|4.45|
|128x8 (~2.5M)|1.65|7.23|5.97|7.10|
|256x2|1.60|7.50|6.23|7.20|
|256x4|1.47|7.64|6.32|7.76|
|256x8 (~8.3M)|1.38|7.72|6.54|8.02|
|512x2|1.39|7.72|6.40|7.90|
|768x8 (best)|1.18|8.62|7.02|9.34|
|GPT-2 small (125M)|-|5.40|3.70|4.32|
|GPT-4|-|8.75|8.21|9.93|

- Instruction-following grade: 768x1 5.81, 256x2 7.23, 256x4 8.07 [P, Fig. 4].
- Heads at 768x2: 2/4/8 heads give loss 1.38/1.34/1.33 [P, Fig. 24].
- Width holds facts, depth holds context: the 1-layer model gets no context-tracking prompt right but some facts; width 64 gets no fact right [P, Sec. 4.2].
- In a 1-layer, width-1024, 16-head model, Figure 19 shows 7 heads attending purely at a fixed relative distance; 3 semantic heads track nouns and character names [P, Sec. 5.1].
- Corpus: 2,119,719 train + 21,990 validation stories, 1.92 GB; V2 (GPT-4 only) 2.23 GB; CDLA-Sharing-1.0 [HF]. A 227-story validation sample averaged 862 characters, 171 words (my count).
- Every model up to ~35M trained on one V100 within 30 hours [P, Sec. 4].

## Smallest scale tested, and whether the result held there
64x2 (loss 2.38, grammar 4.33, consistency 2.41) is the floor. Coherence fails at width 64: consistency "emerges when the hidden size increases from 64 to 128" [P]. Our range (width 256, 2-8 layers) is inside the grid.

## Training recipe and gotchas
- From the TinyStories-33M model card [HF], not the paper: lr 5e-4 constant, weight decay 0.1, Adam betas (0.9, 0.95), context 512, batch 80x16 accumulation.
- Grammar plateaus first, then consistency, then creativity [P, Fig. 3]; do not select checkpoints on grammar.
- The GPT-2 comparison is out of domain for GPT-2, at temperature 1, with truncation: it shows domain match, not general superiority.
- Released configs list vocab 50,257 [HF] although the paper says 10K.

## Mapping onto the GBA design
- A GPT-Neo block has about 12·d² weights (4d² attention, 8d² MLP). At 1.6 cycles per ternary weight, a dense 256x2 costs 1.57M weights/token: our whole budget buys the 1.60-loss row. 256x4 costs 3.15M. So routed experts and top-8 attention must supply the 0.13-0.22 nats between 256x2 and 256x4-8 [I].
- Equal dense compute: 256x2 (1.60) beats 128x8 (1.65); 512x2 (1.39) ties 256x8 (1.38). But under 4 layers, instruction-following and consistency drop: keep 4+ cheap layers.
- Fixed-offset heads are what hashed n-gram embeddings and direct reads of tokens t-1..t-k give for free (ROM lookups, no multiplies); spend top-8 attention on entity tracking [I].
- 16 heads work at width 64 (4 dims per head), so many narrow sign-key heads are plausible.
- Local window 256 matches our ring buffer.
- A dense 10K x 256 output head is 2.56M weights: the two-level head is mandatory.

## Verdict
**Adopt** as calibration corpus, data-generation recipe and evaluation protocol; not the architecture. Settling experiment: reproduce FP32 256x2 and 256x4 baselines with the 10K tokenizer (1.60 / 1.47), train the ternary routed core at 1.6M weights/token on the same data, and report where it lands; grade it and GPT-2 small with one open-weight judge on the 44 released prompts.

## Could not verify
- Which architecture the "roughly 80M" model is; the exponent of the Figure 23 scaling law (plot only).
- Epochs, steps, warm-up; the unit of eval loss; tokens per story.
- No human check of GPT-4 grades, no seeds, no dataset-size ablation, no dialogue evaluation.
- The out-of-distribution instruction claim rests on one example (Fig. 13).

## Links worth following
- https://huggingface.co/datasets/roneneldan/TinyStories (V2, `Evaluation prompts.yaml`, `TinyStories_all_data.tar.gz` with per-story prompts)
- https://huggingface.co/roneneldan/TinyStories-33M
- https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct (named in the paper; not opened)
