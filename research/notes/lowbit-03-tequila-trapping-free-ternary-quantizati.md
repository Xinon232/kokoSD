# Tequila: Trapping-free Ternary Quantization for Large Language Models  (https://arxiv.org/abs/2509.23809, v2 Oct 2025, ICLR 2026; Hong Huang, Decheng Wu, Rui Cen, Guanghua Yu, Zonghang Li and others, Tencent / CityU Hong Kong; code https://github.com/Tencent/AngelSlim/tree/tequila/TernaryQuant)

Tags: [P] arXiv v2 full text; [C] ICLR 2026 camera-ready PDF; [R] repo, branch `tequila`; [D] my derivation; [S] secondary.

## What it is
A small change to absmean ternary quantization-aware training. Latent weights that quantize to zero ("dead", |w| < Δ) also feed a per-output bias λ·Σw, so they get a clean gradient instead of only straight-through noise [P]. After training the bias is a constant and the matrix stays pure ternary [P]. Evaluated only as fine-tuning of pretrained LLaMA-3.2 1B/3B (and Qwen3-4B [C]) on 10B tokens, with float activations, embeddings and head [P][R].

## Mechanism
Per linear layer with latent W of shape [out, in] [P Eq. 2, 7, 8; R `UltraQuantV2`]:
```
alpha = mean(|W|)                      # repo: per 128 consecutive inputs in a row
T     = sign(W) * (|W| >= alpha/2)     # ternary
dead  = (T == 0)
Wq    = W + (T*alpha - W).detach()     # identity STE; alpha gets no gradient [R]
b     = lam * (W * dead).sum(dim=1)    # true gradient; lam = 1e-3
y     = x @ Wq.T + b
```
- A dead weight gets gradient (x_j + λ)·∂L/∂y_i; the λ part is identical for every dead weight in a row [P Eq. 8].
- Export T, alpha, and b_i = λ·Σ_dead w_ij computed once [P App. C].
- Their kernel: 3 weights per 5 bits, a 14-entry table per 3 inputs, then scale and add the bias [P Fig. 7].

## Numbers
Zero-shot accuracy after 10B-token training [P Tables 1-2; C Table 4].

| Model, average | BF16 | AbsMean | Tequila | Gap closed [D] |
|---|---|---|---|---|
| LLaMA-3.2-1B, 6 tasks | 0.502 | 0.445 | 0.471 | 46% |
| LLaMA-3.2-1B, 5 tasks | 0.558 | 0.489 | 0.519 | 43% |
| LLaMA-3.2-3B, 6 tasks | 0.580 | 0.510 | 0.530 | 29% |
| Qwen3-4B, 5 tasks [C] | 0.667 | 0.565 | 0.599 | 33% |

- Headline claims hold at 1B only: ARC-e 0.645 against 0.603 AbsMean and 0.654 BF16. At 3B ARC-e is 0.702 against 0.745 BF16, and TWN beats Tequila on ARC-c (0.351 vs 0.346). HellaSwag at 1B stays 8.6 points below BF16.
- Learnable-scale baselines (LSQ, SEQ, DLT): 0.354-0.370 at 1B.
- Ablation, ARC-e at 1B (Fig. 5): 60.3 AbsMean, 61.5 signed minima, 62.6 bias-only gradient, 64.5 full; error bars about ±1 to 1.5 over three runs [C].
- λ sweep, 1B (Fig. 6): 0 gives 0.445; 1e-5 0.455; 1e-4 0.468; 1e-3 0.471; 1e-2 0.458; 1e-1 0.453. At 3B the peak is 1e-4 (about 0.535); larger models prefer a smaller λ [C].
- Granularity, 1B (Table 3): per-tensor 0.463, per-channel 0.471, per-group 0.471.
- Training loss, 1B, 10K steps (Fig. 3): Tequila and AbsMean both end near 1.4-1.5; the gap is at most about 0.05 nats by eye, largest before step 3000.
- AbsMedian [C]: 0.431 to 0.477 with Tequila at 1B; 0.466 to 0.513 at 3B.
- Speed (Fig. 4): equal to BitNet; about 2x BF16 at 0.7B, 3x at 3B.

## Smallest scale tested, and whether the result held there
Accuracy: 1B parameters (d=2048), fine-tuned from a pretrained model; 0.7B appears only in the speed test. The share of the gap closed was largest at 1B. Nothing is trained from scratch or under 1B, and no perplexity is reported.

## Training recipe and gotchas
- Paper: fixed lr 1e-4, sequence 1024, 16 GPUs [P]. Released script: AdamW, peak lr 1e-3, warm-up over 61% of steps then decay to 0.1x, weight decay 0, 32 GPUs x 8 sequences, no distillation [R]. These disagree.
- Only q/k/v/o and MLP projections are ternary; activations are not quantized [R].
- No control with an ordinary learned bias, so bias capacity and dead-weight gradients are not separated.
- [D] |b_i| <= λ·n_dead·alpha/2. With λ=1e-3 and fan-in 320 that is about 0.056·alpha: under 0.4% of the pre-activation spread, 0-2 LSB of our accumulator (so it may be droppable at export). At our size any benefit is the training gradient, and λ probably has to rise roughly as 1/fan-in.

## Mapping onto the GBA design
- The table kernel is untouched. Per output row, start the accumulator from a 16-bit ROM constant instead of zero: about 4 cycles (estimate), so 10K rows cost about 40K cycles, roughly 1% of the budget. No RAM.
- ROM: 2 bytes per row, +0.6% to +2.5% on rows of 1280 to 320 weights.
- The bias must be in accumulator units, b_int = round(b·s_x/alpha): use a static activation scale, or add it after the per-token rescale.
- Per-row alpha suffices (Table 3) and folds into the requantization multiplier.
- Works directly with the 81-entry free-ternary table. For the 3:4 format, "dead" would be the forced zero in each block of 4; untested. Sherry (same group) uses an annealed float residual instead and reports parity with Tequila at 1B [S, prelim note].

## Verdict
Test. It costs five lines in training and almost nothing at inference, but the evidence is 1B+ fine-tuning with float activations, and the loss gap is small. Settling experiment on our float-initialised ternary student, equal steps: AbsMean; AbsMean plus a plain learned bias; Tequila at λ in {1e-3, 1e-2, 3e-2}; the Sherry residual. Compare validation loss and the share of latent weights within 10% of ±Δ. Adopt only if it beats the plain-bias arm by 0.02 nats or more.

## Could not verify
- OpenReview reviews (HTTP 403). No independent or small-scale reproduction found.
- Zero fraction after training; which learning rate produced the tables.
- Figure values are read off plots; Sherry parity was not re-read.

## Links worth following
- ICLR version: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b555804d495321df2e3208cc27f4fbc-Abstract-Conference.html
- Code: `TernaryQuant/models/utils_quant.py` in the repo above
- Sherry: https://arxiv.org/abs/2601.07892
- bitnet.cpp kernels: https://arxiv.org/abs/2502.11880
