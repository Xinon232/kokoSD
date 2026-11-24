# MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases  (https://arxiv.org/abs/2402.14905, v2, ICML 2024, Zechun Liu, Changsheng Zhao, ... Vikas Chandra, Meta)

Tags: [P] paper HTML v2 (https://arxiv.org/html/2402.14905), read through a fetch tool that extracts text, so tables were transcribed by a model and should be spot-checked; [D] my derivation; [M] my memory, unverified.

## What it is
An architecture study for 125M and 350M decoder-only LMs trained from scratch on 1T tokens [P]. Four stacked choices (SwiGLU FFN, deep-and-thin shape, tied input/output embeddings, grouped-query attention) give "MobileLLM"; a fifth (adjacent blocks reuse the same weights, run twice) gives "MobileLLM-LS" [P]. All choices are evaluated at a fixed parameter count, not fixed compute. It is the baseline family used by ParetoQ's ternary results.

## Mechanism
- SwiGLU FFN: `y = W_down( silu(W_gate x) * (W_up x) )`, hidden about 2.67 x dim (1536 for 576), three matrices [P config; formula is the standard one, M].
- Deep-thin: 30 layers x 576 dim at 125M; 32 x 960 at 350M (Table 9) [P].
- Embedding sharing: output head = input embedding. At 32k vocab x 512 dim each table is 16M params, "more than 20%" of a 125M model [P]. Saved params are re-spent on depth.
- GQA: query heads share key/value heads. Final: 9 heads / 3 kv-heads (125M), 15 / 5 (350M), head dim 64. K and V projections are dim -> kv_heads x 64 [P].
- Layer sharing: `for block in blocks: x = block(x); x = block(x)` ("immediate block-wise"). Chosen over repeat-all-over for cache locality: the weights stay in about 20 MB of SRAM [P].

| Model | Layers | Heads/KV | Dim | FFN | Params |
|---|---|---|---|---|---|
| 125M | 30 | 9/3 | 576 | 1536 | 124.6M |
| 350M | 32 | 15/5 | 960 | 2560 | 345.3M |

## Numbers
Avg = mean zero-shot accuracy on 8 tasks (ARC-e/c, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande). Ablations at 0.25T tokens [P].

Roadmap, 125M / 350M (Table 10, Fig. 3): baseline 12L x 768 42.6 / 47.4 -> +SwiGLU 43.9 / 48.7 -> deep-thin 44.8 / 49.8 -> tied embeddings 44.6 / 49.2 (params 135.0M -> 118.6M) -> GQA + wider 45.0 / 49.9 -> 1T tokens 46.3 / 51.3 -> LS at 1T 47.0 / 52.1.

Depth at about 135M params (Table 11): 4L x 1280 43.3; 6L x 1024 42.3; 8L x 896 43.0; 12L x 768 43.9; 18L 43.9; 24L x 576 44.8; 30L x 512 44.8; 42L 44.5; 62L x 384 44.7. At about 375M: 5L 47.1; 10L 48.6; 12L 49.1; 24L 49.4; 32L 49.8; 66L 49.5. So the gain is under 2 points end to end, and flat beyond about 24 layers.

GQA (Table 13, 125M, 16 heads): kv = 16 / 8 / 4 / 2 / 1 -> 44.6 / 44.3 / 44.7 / 43.5 / 43.7. 350M: 49.6 / 48.5 / 49.4 / 47.5 / 47.9. A 4x reduction is free; beyond that costs about 1 point.

Layer sharing (Table 2, Table 14), 125M / 350M: none 44.6 / 49.6; immediate x2 45.0 / 50.2; repeat-all 45.2 / 50.7; reverse 44.8 / 50.1; x3 45.0 / 49.4; x4 45.3 / 50.4.

Other:
- vs peers (Table 3): MobileLLM-125M 46.3; GPT-neo-125M 42.9; OPT-125M 42.6; Pythia-160M 42.5; RWKV-169M 43.6. HellaSwag 38.9 vs 29.7-31.9.
- Chat after fine-tuning (Table 5): MT-Bench 2.33 (125M), 3.28 (350M); OPT-125M 1.21, GPT-neo-125M 1.06. AlpacaEval win rate vs text-davinci-001: 24.07% / 47.08%.
- W8A8 post-training quantization (Table 15): drop 0.2 (125M), 0.3 (LS-125M), 0.0 (350M), 0.4 (LS-350M).
- Logit distillation from LLaMA-2 7B (Table 16): 125M label-only 43.9 in 29 h vs label+KD 43.8 in 93 h; 350M 49.1 vs 48.8. No gain, about 3x cost.
- iPhone 13 (Table 7): LS adds 2.6% execute time vs the unshared 30-layer model in the paper's measurement; a real 60-layer model takes 29.0 ms vs 16.0 ms.

## Smallest scale tested, and whether the result held there
125M parameters is the smallest. Nothing near our 1-2M touched weights. Effects are 0.2-1.3 points on a 43-47 scale, single runs, no seeds or error bars reported; several orderings flip between 125M and 350M (e.g. x3 sharing). Treat the directions as hints, not laws, at our scale.

## Training recipe and gotchas
- Adam, weight decay 0.1, initial LR 2e-3 with cosine decay, 32 A100s x batch 32; 120k iterations (0.25T tokens) for ablations, 480k (1T) for final models [P]. Implied about 2k tokens per sequence [D].
- 125M at 0.25T tokens took 29 h on 32 GPUs, about 930 A100-hours [D from Table 16]. We cannot afford this token count; conclusions drawn at "infinite data" may not hold on our small corpus.
- Pretraining corpus is not named. GPT-2 is not a baseline anywhere.
- KD result is logit distillation in pretraining only; it says nothing about training on teacher-generated text.

## Mapping onto the GBA design
The paper optimises stored parameters; we have free ROM and scarce compute, so each item is re-judged [D]:
- GQA: adopt. At d=256, 8 query heads, 2 kv-heads (4x reduction): K and V projections shrink from 131K to 33K ternary weights per layer (saves about 157K cycles/layer at 1.6 c/w). Ring-buffer RAM per layer: 8-bit values 256 x 64 B = 16 KB instead of 64 KB; sign keys 2 KB instead of 8 KB. Six layers: 108 KB vs 432 KB, i.e. the difference between fitting in 256 KB or not. Scoring cost (heads x 256 positions) is unchanged.
- SwiGLU: test. Cost over ReLU is one 256-entry LUT and one 8-bit multiply per hidden unit (512 units = about 3.8K cycles), but three matrices: 3 x d x h weights. Unknown under ternary weights and hard routing.
- Deep-thin: test. Per layer about 8.5 d^2 weights (2.5 d^2 attention with GQA, 6 d^2 SwiGLU at h=2d): 557K at d=256, 313K at 192, 218K at 160. Table 11 suggests more, thinner layers at a fixed budget, but RAM for values grows with layers.
- Tied embeddings: skip. It trades 0.2 points to save ROM we do not need; keep untied tables.
- Layer sharing: skip. Doubles compute per block for +0.4 to 0.9 points and saves only ROM.
- W8A8 result supports 8-bit activations (post-training, float baseline; not ternary).

## Verdict
Adopt GQA (kv width = d/4). Test SwiGLU and layer count. Skip sharing of any kind. Settling experiment: at a fixed 1.6M touched ternary weights, train 3x256, 5x192, 7x160 and 11x128 on our data, each with ReLU and SwiGLU, and with kv width d, d/4, d/8; compare held-out perplexity and BLiMP.

## Could not verify
- Pretraining data, tokenizer, exact vocabulary and context length: not in the paper text I read (32k appears only in the embedding example).
- Per-row transcription of tables was done by the fetch tool; totals agree across tables but individual cells were not double-checked.
- No GPT-2 small comparison exists in this paper.
- I did not open the code repository or model cards.

## Links worth following
- ParetoQ (ternary MobileLLM): https://arxiv.org/abs/2502.02631 (already in notes)
- GQA paper: https://arxiv.org/abs/2305.13245
- MobileLLM code: https://github.com/facebookresearch/MobileLLM (from memory, not opened)
