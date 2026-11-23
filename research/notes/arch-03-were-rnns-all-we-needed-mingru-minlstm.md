# Were RNNs All We Needed? (minGRU / minLSTM)  (https://arxiv.org/abs/2410.01201, v3 28 Nov 2024; Feng, Tung, Ahmed, Bengio, Hajimirsadeghi; Mila / Borealis AI)

Tags: [P] primary text (arXiv HTML v3, whole paper incl. appendices); [P2] primary text of a followed link; [I] my inference or arithmetic.

## What it is
GRU and LSTM with the previous hidden state removed from gates and candidate, and tanh dropped [P]. What remains is a per-channel gated moving average whose coefficients depend only on the current input: training is a parallel prefix scan, inference a one-step O(d) recurrence [P]. Evidence is small: synthetic tasks (Chomsky/LRA average 0.73 vs xLSTM 0.71, Table 4), D4RL, and one character-level Shakespeare LM where it ties a transformer; most baselines are copied from other papers [P]. No quantization, no subword LM.

## Mechanism
minGRU, sequential mode used at inference [P, Alg. 1 and 5]:
```
z   = sigmoid(Wz x_t)            # Wz: dh x dx
c   = g(Wh x_t)                  # Wh: dh x dx
h_t = (1 - z) * h_{t-1} + z * c
```
- Vanilla: g=identity. Log-space version, the one the authors trained: `g(x) = x + 0.5 if x >= 0 else sigmoid(x)`, so c and h stay positive [P, App. B].
- Training scan [P, Listing 5], k=Wz x: `log_a = -softplus(k)`; `log_b = -softplus(-k) + log g(Wh x)`; `a* = pad(cumsum(log_a))`; `h = exp(a* + logcumsumexp(log_b - a*))`.
- minLSTM [P]: two gates normalized to sum to 1; more parameters, less stable. Prefer minGRU.
- Parameters [P]: 2·dh·dx against GRU 3·dh·(dx+dh). With expansion dh=alpha·dx a down-projection is added.
- LM block [P, C.2-C.3]: Conv(kernel 4) -> minRNN -> MLP, residuals and normalization; 3 layers, d=384, alpha=2. Conv type, MLP ratio and norm type are not stated.
- lucidrains reimplementation [P2]: pre-RMSNorm, one bias-free Linear(d -> 2·alpha·d) split into (hidden, gate), alpha=1.5, GELU FFN x4, conv off by default.
- Structural fact [P, Sec. 4.1]: layer-1 gates see only the token, so they are time-independent; context-dependent gating starts at layer 2.

## Numbers
- Selective Copying, length 4096, d=64, alpha=6 (Tables 1-2) [P]: minGRU 37.0/96.8/99.5% for 1/2/3 layers; minLSTM 37.6/85.7/96.0; Mamba S6 99.8.
- Char-level Shakespeare, 1.0M training tokens, 3 layers, d=384 (Fig. 2) [P]: test loss minGRU 1.548, minLSTM 1.555, Mamba 1.575, Transformer 1.547; best at 575/625/400/2000 steps.
- ListOps ablation (Table 6) [P]: bare 0.46, +Conv 0.45, +MLP 0.52, +Conv+MLP 0.59.
- MatMul-free LM, a minGRU-like mixer with four ternary d x d matrices [P2, Tables 1-2]: zero-shot average 40.3 vs Transformer++ 41.1 at 370M; 49.9 vs 50.7 at 2.7B; post-training W8A16 39.8, W8A8 35.4.
- HGRN2 paper, WikiText-103, about 45M parameters [P2, Tables 1 and 5]: Transformer 24.78 test perplexity; real HGRN1 with state d 26.12, with state 2d 24.64; HGRN2 23.73.

## Smallest scale tested, and whether the result held there
The LM is 3 x 384, character-level, on 1M characters; parameter count not stated [P]. Parity with the transformer held, but every model peaked within 2000 steps: a toy. Nothing uses a subword vocabulary. At 45M the closest relative with a d-sized state trails a transformer by 5% perplexity; doubling the state closes the gap [P2, I].

## Training recipe and gotchas
- LM: AdamW, lr 1e-3, dropout 0.2, clip 0.25, batch 64, 5k steps; default PyTorch init [P].
- Use at least 2 layers; keep the conv and MLP [P].
- Ternary recurrent mixer: lr 4e-3 against 3e-4 for FP; post-training 8-bit activations cost over 12% relative, 16-bit under 1.5% [P2].
- A float scan cannot reproduce per-step integer rounding: keep device state wide, fake-quantize only z and c [I].

## Mapping onto the GBA design
All [I].
- Compute per layer at d=256, alpha=1: 2d^2 = 131K ternary weights, about 210K cycles. With alpha=2 and down-projection: 393K weights, about 630K cycles. The elementwise part (d sigmoid lookups, d multiplies for `h += z*(c-h)`) is about 10K cycles, so shift-only gates are not worth it.
- Against attention: narrow attention (64-d Q and K, 128-d V and O) is about 98K weights plus the key scan, so minGRU is not cheaper in cycles. It wins on RAM: 1 KB of state per layer against roughly 18 KB, and no ring buffer.
- ROM precompute: layer-1 z and c depend only on the token, so store both as d-byte rows (16K tokens x 512 B, about 8 MB), or key them on a hashed bigram. That mixer costs d lookups and d multiplies, no matrix work.
- State is a convex mix of bounded c, so it cannot overflow. Keep 16 fractional bits or small updates round to zero.
- Required change: h is positive-only, so add an output projection or per-channel scale and offset.

## Verdict
**Test**, as a hybrid; do not replace all attention on this evidence. Try the ROM-table layer-1 mixer first: it is nearly free. Settling experiment: at equal GBA cycles with the 16K vocabulary, compare all-attention, ROM minGRU layer 1 plus attention, alternating, and all-minGRU, after ternary QAT with 16-bit state.

## Could not verify
- Parameter counts, context length and conv/MLP details of the Shakespeare run; the paper links no official code.
- Any minGRU LM result on TinyStories, BabyLM or WikiText: two searches found none.
- No source tests ROM-tabulated gates, ternary minGRU proper, or integer state.

## Links worth following
- https://github.com/lucidrains/minGRU-pytorch
- MatMul-free LM: https://arxiv.org/abs/2406.02528
- HGRN2: https://arxiv.org/abs/2404.07904
- Merrill et al., "The Illusion of State in State-Space Models" (ICML 2024)
