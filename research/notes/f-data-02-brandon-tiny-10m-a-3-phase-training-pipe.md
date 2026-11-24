# Brandon-Tiny 10M: a 3-phase training pipeline for ultra-small instruction-following language models  (https://xaskasdf.github.io/brandon-tiny/paper.html, February 2026, Samuel Cortes)

Read in full: the HTML report and the repository (github.com/xaskasdf/brandon-tiny, commit cd56eb1). Tags: [P] report, [R] repo `docs/technical_report.md`, [C] code, [D] dataset card, [I] my inference. Self-published; single runs.

## What it is
A 10.7M-parameter Llama-2-style chat model trained in about 6.5 h on one RTX 3090 by pretrain, then logit distillation from a 30M teacher, then ChatML fine-tuning with three anti-repetition loss terms. The report claims the pipeline matters more than parameter count. It also reports a failed ternary variant. Its ablations are uncontrolled.

## Mechanism
- Model [P, C]: dim 256, SwiGLU hidden 720, 24 layers as 12 unique blocks each applied twice in a row, 8 heads with 2 KV heads (KV width 64), RoPE, RMSNorm, tied embeddings, dropout 0.05, 8,192 SentencePiece BPE, context 512.
- DenseFormer [C]: after layer i, `h = sum_{j<=i+1} w[i][j] * all_h[j]` over the embedding and all earlier outputs; `w` initialised to identity.
- Value residual [C]: `v_l = Wv_l x + v_0` for l > 0, no mixing coefficient.
- Registers [C]: 4 learned vectors prepended to the sequence.
- Phase 1 [C]: AdamW betas 0.9/0.95, weight decay 0.1, clip 1.0, bf16, batch 32 x 4 accumulation x 512 = 65,536 tokens/step, 15K steps. LR 8e-4: 500 warm-up steps, constant until the last 20%, then cosine to 8e-5.
- Phase 2 [C]: `L = 0.5*T^2*KL(p_s^T || p_t^T) + 0.5*CE`, T = 2, teacher = the 30M model's *pretrain* checkpoint (dim 384, 28 layers, no chat tuning), LR 4e-4, 7.5K steps, batch 24 x 4.
- Phase 3 [C]: LR 2e-5 cosine to 2e-6, 12K steps, batch 16 x 4, loss on assistant tokens only. `L = CE(label smoothing 0.1) + 0.5*UL - 0.01*H`, where `UL = -log(1 - p(c))` summed over every token c already in the context and H is the mean output entropy.
- Replay [C]: pretrain passages split mid-way and wrapped as "Continue this text:" instructions.

## Numbers
All at 10.7M unless stated.

|variant|pretrain data|pretrain loss|fine-tune loss|
|---|---|---|---|
|baseline|TinyStories+SmolLM|1.92|3.92|
|Enhanced (DWA+VR+registers)|mixed|3.73|2.92|
|Dream (ternary, looped)|mixed|3.81|2.98|
|Enhanced, synthetic-only|synthetic|1.96|3.62|
|Optimal (3-phase)|Wiki+SmolLM+synthetic|4.39|2.40|
|30M reference|TinyStories+SmolLM|3.26|2.61|

[P section 5.1; data column from R.] Validation sets differ between rows [P].
- Phase 2 raised hard-target loss from 4.39 to 4.84 [P 4.3].
- 0-shot, 1,000 examples per task [P 5.3]: BLiMP 73.3, HellaSwag 32.4, ARC-Easy 30.6, PIQA 54.7, Winogrande 50.3, LAMBADA 8.8, WikiText-2 perplexity 224.2.
- Custom suite [R 5.3], WikiText-2 perplexity and keyword-match on 25 prompts: Optimal 224.2, 80%; Enhanced v2 329.2, 80%; synthetic-only 818.0, 96%; 30M 302.8, 84%; 30M Dream 1121.6, 28%.
- Phases take about 3 h, 1.5 h and 2 h [P 7.1].

## Smallest scale tested, and whether the result held there
10.7M is the smallest reported. No component was ablated alone: the architecture row also changes data, and the 3-phase row changes data, distillation, and the fine-tune set together.

## Training recipe and gotchas
- KL scaling [C, I]: `F.kl_div(..., reduction='batchmean')` on [B, 512, V] divides by B only, while CE is a per-token mean. The soft term is about 512 times heavier than "alpha 0.5" says, which fits the hard-target loss rising.
- Unlikelihood term [C, I]: with the assistant mask it is divided by `mask.sum() * V`, so its effective weight is about 0.5/8192. It also does not exclude the gold token, so up-weighted it would punish correct copying of names.
- "Zero loops in 20 generations" was measured with decode-time repetition penalty 1.2 [C].
- Ternary failure is confounded [C, I]: the 28% belongs to the 30M Dream. Dream is 4 layers x 4 loops at dim 576; weights are absmean-ternarised with a plain straight-through estimator, no activation quantisation, no norm before each linear, 15K steps. `model_dream.yaml` says "~15M unique params", not 10.7M. No full-precision Dream result is reported.
- Report and code disagree: replay is "~200 examples" [P, D] but 10% of the set [C]; the baseline is "dim 192, 16 layers" [P 3.4] but dim 256, 24 layers [C].
- Pretrain losses are on different corpora, so the "low pretrain loss does not predict fine-tune loss" finding is confounded.

## Mapping onto the GBA design
All [I].
- Loss terms, schedule, replay: zero inference cost.
- Registers: they sit first under a causal mask, so their keys and values at every layer are input-independent. Store them in ROM: 4 extra key scores per layer, no RAM.
- Value residual: one width-d add per layer per token, plus d bytes of RAM for the current token's layer-0 value.
- DenseFormer: sum over i of (i+2) scaled vectors. At 6 layers, 27 x 256-384 = 7-10K multiply-accumulates, about 52-78K cycles (1-2.5% of budget).
- Two KV heads (KV width 64): a 256-token ring costs 8 bytes of sign key plus 64 value bytes per position, 18.4 KB per layer, 110 KB for 6 layers. Full-width values (73.7 KB per layer) would not fit 6 layers in 256 KB.
- Block sharing saves ROM, which we have, and no compute. Skip.

## Verdict
Test. Adopt the cheap parts now: real text in pretraining, assistant-only loss, label smoothing 0.1, entropy bonus, replay. Treat distillation gain and the "ternary fails" claim as unproven. Settling experiment: at our width, train (a) pretrain+fine-tune, (b) plus per-token-normalised reverse KL, (c) plus forward KL, on one shared validation set, 2 seeds each.

## Could not verify
- Any isolated effect of distillation, DWA, value residual or registers.
- Which LLM produced the 57K chat set; the dataset card says only "curated" [D, via a fetch summary, not read raw].
- The Super Tiny LM comparison (25.6 / 21); our prelim notes found that paper's own table differs.
- The PDF was not opened.

## Links worth following
- MiniPLM, reverse KL for small students: https://arxiv.org/abs/2410.17215
- DenseFormer: https://arxiv.org/abs/2402.02622
- Value Residual Learning: https://arxiv.org/abs/2410.17897
