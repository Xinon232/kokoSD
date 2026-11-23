# Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification
https://arxiv.org/abs/2601.07892 (v1, Jan 2026). Huang, Wu, Hu, Yu, Yang, Zhu, Liu, Wu (CityU Hong Kong / Tencent / McGill). Code: Tencent/AngelSlim, branch `sherry`.
Tags: [P] paper, read in full; [C] released code, read; [S] secondary; [I] my inference.

## What it is
A ternary format where every block of 4 consecutive input weights holds exactly one zero and three ±1: 32 states, 5 bits, 1.25 bits/weight. "Arenas", a training-only full-precision residual annealed to zero, repairs the accuracy loss ("weight trapping", gradient effective rank below 750 of 4096) of naive 3:4 training. Evidence is QAT fine-tuning of pretrained LLaMA-3.2 1B/3B only, with floating-point activations, embeddings and output head. The format is exactly our 32-entry table kernel; the paper adds the training recipe.

## Mechanism
W is (d_out, d_in). Per output row, per block of 4 along d_in [P, C]:
- T = sign(W) on the 3 largest |W|, 0 on the smallest; L2-optimal under the constraint (App. D).
- alpha = mean |W| of the kept weights: per row, per 128 consecutive inputs of a row (default "per-group"), or per tensor.
- Backward: identity straight-through to every latent weight, pruned ones included [C].

Arenas [P, C]:
```
y = x @ (T*alpha).T + lam * (x @ W.T)      # training only
lam: linear 0 -> 1 over the first 10% of steps,
     then 0.5*(1 + cos(pi*p)), p = progress through the remaining 90%
eval / deploy: lam = 0
```
AngelSlim `main` merges this into one matmul, W_eff = Q(W) + lam*W [C].
Inference (App. A) [P]: a 16-entry table per 4 activations plus a sign bit that negates; the split exists only to fit a 16-byte `vpshufb` table.

## Numbers
5-task zero-shot average (ARC-e/c, HellaSwag, PIQA, WinoGrande), 3 seeds, 10B QAT tokens [P], Table 1:

| Method | 1B | 3B |
|---|---|---|
| BF16 | 0.558 | 0.636 |
| AbsMean dense ternary | 0.489 | 0.551 |
| Tequila | 0.519 | 0.576 |
| Sherry 3:4 | 0.519 | 0.567 |

- Arenas ablation, 1B (Fig. 6; baselines derived from printed deltas): binary 0.462 to 0.489 (+2.67 points); 3:4 0.473 to 0.519 (+4.59); dense ternary 0.489 to 0.519 (+2.98). With Arenas 3:4 equals dense ternary; without, 1.6 points worse.
- Scale granularity, 1B (Table 3): per-tensor 0.502±0.010, per-channel 0.513±0.011, per-group 0.519±0.014.
- Schedules (Fig. 8, approximate bar readings): none 0.473; cosine 0.501, with warm-up 0.519; linear and exponential lower.
- Speed, i7-14700HX (Table 4): 0.7B 148.27 t/s vs 132.13 (2-bit) and 116.83 (1.67-bit); 3B 45.55 vs 41.87 and 38.80.

## Smallest scale tested, and whether the result held there
Accuracy at 1B (pretrained, then fine-tuned); speed at 0.7B. It held at 1B (parity with Tequila, +3.0 points over AbsMean) and slipped at 3B (0.9 below Tequila). Nothing from scratch or with quantized activations.

## Training recipe and gotchas
- [C] lr 1e-4 constant, AdamW, weight decay 0, bf16, sequence 1024, batch 16 per GPU, one epoch, no distillation; 22,111 total and 2,211 warm-up steps hardcoded. Only q/k/v/o and the three MLP projections are quantized.
- [C] The step counter increments on every forward call (eval and accumulation micro-steps too). Drive lam from the trainer's global step.
- [I] At lam=1 the effective weight is roughly twice the quantized one, so output scale drifts during training. Calibrate integer activation ranges after lam=0 and add a pure-ternary tail (the paper reaches 0 only at the last step).
- [P] Learned-scale baselines (LSQ, SEQ, DLT) scored 0.378-0.398 at 1B: keep alpha statistical.

## Mapping onto the GBA design
- Kernel: identical to our 32-entry variant; index the table with the 5-bit code directly (no negate). One lookup+add per 4 weights, no multiplies.
- RAM [I]: d_in=256 gives 64 blocks x 32 x 2 bytes = 4 KB of tables, against 10.4 KB for 81 entries.
- Table build [I]: 32 vs 81 entries per block. At an assumed 4 cycles per entry (unmeasured), a 256x256 matrix costs about 8k build + 105k lookup cycles against 21k + 105k: roughly 10% less.
- Scale: per-row alpha is one multiply per output (about 2k cycles per 256-row matrix), foldable into a ROM requantization multiplier. Skip per-group-128: it needs d_in/128 partial sums per row for a 0.006 gain inside one standard deviation.
- ROM [I]: a byte per block is 2 bits/weight (about 128M weights in 32 MB); six codes per 32-bit word is 1.33 bits/weight (about 190M). Measure.

## Verdict
**Test, expecting to adopt.** Arenas is cheap and helped all three formats at 1B, where 3:4 matched dense ternary. Our regime (from scratch or a small teacher, under 10M dense weights, 8-bit activations) is untested. Settling experiment: at the target architecture, {dense AbsMean, 3:4} x {no Arenas, cosine Arenas with 10% warm-up}, equal steps, validation perplexity. Adopt 3:4 if within about 2% of dense.

## Could not verify
- Results below 0.7B, from scratch, or with integer activations: none in paper or repository.
- Fig. 6 and Fig. 8 values are bar readings; dense ternary + Arenas (0.519) is not tabulated.
- llama.cpp PR #22836 (STQ1_0 kernel), read only through an automated page summary [S]: on an M4 Pro with Sherry-1B, 147.47 t/s against 175.06 for 2-bit TQ2_0, so the speed claim did not carry to NEON there.

## Links worth following
- Code: https://github.com/Tencent/AngelSlim/tree/sherry/Sherry (`models/utils_quant.py`)
- Tequila: https://arxiv.org/abs/2509.23809
- llama.cpp kernel: https://github.com/ggml-org/llama.cpp/pull/22836
- BitNet.cpp: https://arxiv.org/abs/2502.11880
