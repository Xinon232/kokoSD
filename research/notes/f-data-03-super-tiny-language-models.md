# Super Tiny Language Models  (https://arxiv.org/abs/2405.14159, v2 26 June 2024, Guertler, Hillier, Agrawal, Chen, Cheng, Tan; A*STAR Singapore)

Read in full: arXiv HTML v2 (all sections, both tables, appendix). Also read: repository README (github.com/LeonGuertler/SuperTinyLanguageModels) and the abstract page of the follow-up dropout report (2409.05423). Tags: [P] paper, [R] repo README, [A] abstract only, [I] my inference.

## What it is
A position paper and research agenda, not a results paper. It announces a series on 10M / 50M / 100M-parameter models trainable in under 48 GPU-hours, releases a training repository, and reports one set of 50M baselines on OpenWebText. Its headline idea, byte-level embedding with pooling at BPE boundaries for "90-95% fewer parameters", is a proposal with no experiment behind it [P 5.2]. The other proposals (FFN sharing with LoRA, early exit, next-thought prediction, dropout schedules, data mixes) are one paragraph each with citations only.

## Mechanism
Baseline actually trained [P Table 1]: 8 layers, hidden 512, SwiGLU FFN dim 1536, 16 heads with query group size 4, RoPE, context 512, GPT-2 BPE (50,257), dropout 0.1, LR 6e-4, weight decay 0.1, batch 24 x 20 accumulation, 5,000 warm-up steps, 2x L40, 30 GPU-hours. Embeddings are 25M of the ~50M parameters [P 4.2].

Byte-level pooling, as proposed [P 5.2], never run:
1. Embed each byte (vocab 256) at a small width, e.g. 64.
2. Run a BPE tokenizer only to get token boundaries; group byte embeddings per token.
3. Pass each group through a "smaller two-layer transformer" and pool to one vector of core width. Sequence length equals the BPE length.
4. Core model unchanged.
5. No LM head: decode each next-token hidden state back into bytes with a mirror-image byte decoder; loss is per byte of the next BPE token.
No pooling operator, decoder shape, or stop rule is given.

Evaluation [P 4.1.2]: perplexity normalised by original string length, not token count, so that tokenizers can be compared; multiple-choice tasks scored by path probability of each option.

## Numbers
Table 2, zero-shot, "custom splits" (so not comparable to published leaderboards), accuracy %:

|model|params|BLiMP|HellaSwag|ARC-easy|WinoGrande|MMLU|
|---|---|---|---|---|---|---|
|Phi-3|3B|83|56|80|73|32|
|TinyLlama|1B|84|43|60|59|26|
|MobiLlama|500M|80|39|51|55|25|
|Qwen2|500M|80|38|52|55|28|
|Baseline (RoPE+SwiGLU)|50M|78|30|39|50|24|
|SinCos PE|47M|77|29|35|50|24|
|Learned PE|48M|77|29|37|50|23|
|Simple FFN (GELU, dim 2304)|50M|79|30|39|51|24|
|DropoutFree|47M|77|30|39|51|24|
|chance|-|50|25|25|50|25|

- All 50M variants are within 1-2 points of each other except ARC-easy for positional encodings (39 vs 35/37). Single runs, no error bars.
- 50M models are at chance on WinoGrande and MMLU; BLiMP is only 2 points below 500M models [P 4.2].
- Dropout 0 gave lower perplexity than dropout 0.1 on OpenWebText (Figure 1; values not printed) [P 4.2].
- Projected, not measured: embedding + head are "45.20% to 62.25%" of a GPT-2-sized model; byte pooling needs "~10%" of the embedding parameters and removes 90-95% overall [P 5.2].
- Simple English Wikipedia overfit a 50M model "after 5,000 epochs" (I read this as iterations), at which point the model had seen ~3B words, each training word ~100 times; they cite Kaplan for needing ~1B words at this size [P 4.1.1].

## Smallest scale tested, and whether the result held there
47-50M only. The 10M and 100M targets were never run in this paper. Nothing byte-level, weight-shared or early-exit was tested at any scale.

## Training recipe and gotchas
- Table 1 says 25,000 iterations; the text says 50,000 steps. Unresolved. At 24 x 20 x 512 = 245,760 tokens per step this is 6.1B or 12.3B tokens [I].
- The Simple-FFN comparison is confounded (FFN dim 2304 vs 1536), as the authors admit.
- The practical lesson is data volume: ~30M words (Simple Wiki, implied by "3B words = 100 passes") is far too little for 50M parameters [I from P].
- Follow-up [A]: the dropout report finds dropout still helps when overfitting a small corpus and may help fit on large data.

## Mapping onto the GBA design
- The paper optimises parameter count. Our scarce resource is per-token compute; ROM is nearly free. A 16K x 384 ternary embedding is ~6.1M weights, roughly 1.2 MB of 32 MB [I]. There is nothing to save.
- Input side: the byte encoder is a pure function of the token's bytes, so for a fixed vocabulary it folds into a 16K-row ROM table at export. Run-time cost is identical to a plain embedding lookup (zero multiplies). It could only matter as a training-time regulariser that shares statistics between spelling-related tokens, which is unproven.
- Output side: byte-wise decoding replaces one two-level head evaluation by one small-decoder pass per byte (about 4 per token for English BPE [I]), each needing dense ternary compute and a 256-way choice. That spends the scarce resource to save the free one. Our class-then-member head is the better trade.
- Usable pieces: (a) length-normalised perplexity (bits per character) is the right metric for Koko (16K vocab) versus GPT-2 (50K vocab); costs nothing. (b) Dropout 0 for web-scale data, dropout on if we are in the overfitting regime on a small simple-English corpus. (c) The 50M row is a reference point: a full-precision 50M model on OpenWebText gets HellaSwag 30 / ARC-easy 39 / BLiMP 78 on custom splits, so GPT-2-level knowledge benchmarks are out of reach and BLiMP-style grammar is the benchmark where tiny models are close.
- Their conditional-computation section (mixture-of-depths, LayerSkip) matches our direction but contributes no method or number.

## Verdict
Skip as a method source. No experiment in it bears on quality per cycle. Keep two habits: character-normalised perplexity, and a data-volume check (words seen per stored weight). If the spelling-sharing idea is ever wanted, the settling experiment is cheap: train the 16K embedding table as `E[token] = table[token] + pool(byte_encoder(bytes))`, fold to ROM, and compare validation bits/char against the plain table at equal steps.

## Could not verify
- Any byte-level, weight-sharing, knowledge-distillation or 10M results: the repo lists these reports as "in progress" [R]; a web search found no published follow-up besides dropout.
- The "45.20% to 62.25%" figure: for GPT-2 small I compute about 31% tied and 47% untied (38.6M embedding of 124M) [I]; the paper does not show its arithmetic.
- Perplexity values (Figure 1 is an image), the exact "custom splits", and whether 25K or 50K steps were run.
- Brandon-Tiny's quoted STLM scores (25.6 / 21) do not match this paper's Table 2 (30 / 39); the prelim note already flags this.
- Dropout report read at abstract level only.

## Links worth following
- https://github.com/LeonGuertler/SuperTinyLanguageModels (MIT; `pre_reports/` holds preregistrations for FFN sharing and distillation)
- https://arxiv.org/abs/2409.05423 (STLM Engineering Report: Dropout)
- https://arxiv.org/abs/2402.16840 (MobiLlama, the FFN-sharing source they build on)
