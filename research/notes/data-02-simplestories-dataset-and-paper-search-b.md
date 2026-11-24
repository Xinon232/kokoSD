# Parameterized Synthetic Text Generation with SimpleStories (arXiv:2504.09184v3, 2025; Finke, Sreedhara, Dooms, Allen, Zhang, Rodriguez, Nabeshima, Marshall, Braun)

Dataset: https://huggingface.co/datasets/SimpleStories/SimpleStories (MIT). Paper: https://arxiv.org/abs/2504.09184. Tags: [P] = primary text (paper, cards, repository code); [F] = read off a figure; [M] = my arithmetic.

## What it is
A TinyStories replacement: 2,115,696 train + 21,371 test English stories (plus about 2M Japanese) written by gpt-4o-mini-2024-07-18 from prompts whose parameters are sampled from fixed lists and kept as labels [P]. It ships with a lowercase 4,096-token WordPiece tokenizer and five Llama-style models, 1.25M to 35M parameters including embeddings [P]. For us: free pretraining text, a recipe for diverse synthetic data, and a full-precision quality-versus-size curve at our widths and depths.

## Mechanism
Prompt (Appendix A) [P]: "Write {N} short stories ({k} paragraphs each) using very basic words ... about {theme}, include {topic}, be {style} ... feature {narrative feature} ... start the story with {a noun} that begins with the letter {p}."
- Sampled per call: theme (63 options), topic (48), style (23), narrative feature (26); grammar feature (31) in 50% of calls; persona (23) in 33% [P; sizes are my count].
- First word forced to one of 4 parts of speech and a letter drawn from corpus letter frequencies: the main de-duplication device.
- Paragraphs uniform in 1-9; stories per call inversely proportional; split on "The End.". Names from a supplied list. Nucleus p=0.9, temperature 1.
- Stored per story: labels, word count, readability grades. ASCII-only.

Tokenizer (repository code) [P]: lowercase; pre-tokenize on whitespace, punctuation, single digits; WordPiece with "##" continuations; specials [UNK], [EOS].

Models (config.json, model_configs.py) [P]: RoPE, RMSNorm, SwiGLU with hidden 8d/3, no biases, tied embeddings, grouped-query attention with 2 key/value heads, context 512.

## Numbers
- Story length 224.6 ± 103.8 words vs TinyStories 175.4 ± 80.2; Flesch-Kincaid grade 3.08 ± 1.24 vs 2.55 ± 1.49 (Section 3.1) [P]. About 475M words in total [M].
- Top 4-gram: "once upon a time" in 59.38% of TinyStories; SimpleStories' worst is "took a deep breath", 7.49% (Table 1) [P].
- Judge on sets of 4 stories, N=200: simplicity about 85 for both; content diversity about 76 vs 61; style diversity about 63 vs 47 (Fig. 1) [F].
- Label recovery by gpt-4o-mini: topic 0.49, theme 0.225, style 0.145, narrative feature 0.155 [P].
- Model quality, GPT judge, N=200 stories each (model-card figure; paper Fig. 4) [F]; dense weights exclude embeddings [M]:

| Model | layers x d | dense weights | originality | coherence | grammar | quality |
|---|---|---|---|---|---|---|
| 35M | 12 x 512 | n/c | 72.5 | 85.7 | 90.8 | 81.5 |
| 30M | 10 x 512 | n/c | 73.4 | 85.5 | 87.7 | 80.9 |
| 11M | 6 x 384 | 9.4M | 73.8 | 77.0 | 84.2 | 75.5 |
| 5M | 6 x 256 | 4.3M | 76.4 | 75.0 | 82.0 | 74.6 |
| 1.25M | 4 x 128 | 0.72M | 71.7 | 51.4 | 70.0 | 57.4 |

- Tokenizer ablation on the 35M architecture: custom 4,096 vs GPT-2 50,257 gives +26.7 coherence, +27.5 quality (Section 3.5) [P].
- Every model trains in at most 12 hours on one A100 [P].

## Smallest scale tested, and whether the result held there
1.25M parameters. Grammar partly holds (70.0) but coherence collapses (51.4); the knee is between 1.25M and 5M. From 5M to 11M the gain is small (+2 coherence); the next jump is at 30M. The claimed win over TinyStories-33M "in all metrics" is narrow on grammar by the paper's own text; I could not read those bars.

## Training recipe and gotchas
- AdamW, lr 1e-4, batch 128, gradient clipping, cosine decay, 100 warmup steps [P]. Script defaults: betas (0.9, 0.95), weight decay 0.1, clip 1.0, bfloat16 [P]. Epochs not stated.
- A code comment admits a bug: the "affix-aware" alphabet (un, re, ed, ing, ly, er, ness) only adds single characters, so the tokenizer is plain WordPiece [P].
- The tokenizer ablation is confounded: lowercasing, vocabulary size and in-domain fitting change together; no in-domain BPE baseline (Limitations) [P].
- Lowercase only; a chatbot must restore casing by rule.
- Fiction only: no dialogue turns, no facts. Residual tics: "a girl named mia" 2.97%.
- One GPT judge, 200 samples; no perplexity, no BLiMP.

## Mapping onto the GBA design
- Run-time cost: none (training data). The value is the curve: SimpleStories-5M touches 4.3M dense float weights per token, 2-3 times our 1.3-2M ternary budget; the 1.25M model touches 0.72M. A plain dense model inside our budget sits between coherence 51 and 75 before any ternary penalty [M]. Routed experts and ROM lookups must supply the rest.
- Our 256-384 width with 6 layers is the 5M/11M pair: 256 to 384 more than doubles compute for +2 coherence. Prefer 256 wide and spend ROM instead.
- Vocabulary 4,096 lowercase vs our 16K: a two-level head needs about 64+64 row dot-products instead of 128+128 [M] and embedding ROM shrinks 4x, but tokens per word rise (not reported).
- Labels can key ROM-stored topic bias vectors, nearly free at run time.
- The recipe transfers to our dialogue data: parameter grid, forced first word, several samples per call, kept labels, 4-gram and compression checks.

## Verdict
Adopt as phase-1 pretraining data and adopt the prompt-parameterisation recipe. Test, do not assume, the small-vocabulary claim: train our student at 4K, 8K and 16K vocabulary, lowercase vs cased, on identical data, and compare bits per character and judge score per CPU cycle. Use the released 5M and 11M models as full-precision reference points.

## Could not verify
- Bar values for TinyStories-33M (Fig. 4), Fig. 5 and Fig. 3.
- Tokens per story, total tokens, epochs, final losses.
- The generation script itself (README only); the Japanese set.
- NeurIPS 2025 acceptance comes from the conference listing, not the paper.

## Links worth following
- https://github.com/danbraunai/simple_stories_train
- https://github.com/lennart-finke/simple_stories_generate
- https://huggingface.co/SimpleStories/SimpleStories-5M
- https://github.com/cshaib/diversity
