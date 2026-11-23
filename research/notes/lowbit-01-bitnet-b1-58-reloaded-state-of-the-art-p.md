# BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks  (https://arxiv.org/abs/2407.09527, v1 June 2024, Jacob Nielsen and Peter Schneider-Kamp, Univ. of Southern Denmark; code https://github.com/schneiderkamplab/bitlinear)

Tags: [P] paper HTML; [R] repo source (current `main`, may postdate the paper); [F] follow-up paper; [D] my derivation.

## What it is
A from-scratch study of ternary quantization-aware training on very small decoder LMs (4 layers, hidden 32-256, 8K BPE vocabulary) and tiny vision nets [P]. Every `nn.Linear` becomes a "BitLinear" with ternary weights and 8-bit activations; the one new idea is that the weight scale may be the median of |W| instead of the mean [P]. It claims ternary matches 16-bit when the hidden size is doubled [P]. Results are single runs on an unnamed 135M-token corpus [P].

## Mechanism
Per BitLinear, input I, float shadow weights W [P]:
```
I_hat   = LayerNorm(I)
x_scale = 128 / (max|I_hat| + eps)         # per token, max over last dim [R]
x_q     = clip(round(I_hat * x_scale), -128, 127)
w_scale = 1 / (Measure(|W|) + eps)         # one scalar per matrix; mean or median
W_q     = clip(round(W * w_scale), -1, 1)  # zero iff |w| < 0.5 * Measure
y       = (x_q @ W_q^T + b) / (w_scale * x_scale)
```
- Straight-through estimator: `q = (quant(v) - v).detach() + v` [P][R].
- Repo defaults: eps 1e-5, median, a gain-free LayerNorm (`torch.layer_norm(x, [n_in])`) inside every BitLinear, optional blend `lambda*(quant - v).detach() + v` [R].
- Zero fraction [D]: Gaussian weights give about 31% zeros with mean and 26% with median; Laplace weights 39% and 29%. Median sits near the 25% that a "one zero per 4 weights" format forces.

## Numbers
Table 1 [P]: best perplexity per cell after 10 epochs (1.35B tokens), 4-layer Mistral-like models.

| Hidden (params) | 16-bit | Ternary mean | Ternary median | Best ternary / 16-bit |
|---|---|---|---|---|
| 32 (6M) | 77.8 | 130.1 | 116.6 | 1.50 |
| 64 (12M) | 36.7 | 67.4 | 60.0 | 1.63 |
| 128 (24M) | 21.4 | 36.3 | 37.5 | 1.70 |
| 256 (48M) | 16.6 | 27.1 | 26.8 | 1.61 |

- Loss gap at equal size: 0.40, 0.49, 0.53, 0.48 nats. It does not shrink with width here [D].
- "Double the hidden size": ternary 64 beats 16-bit 32 (60.0 vs 77.8), ternary 128 ties 16-bit 64 (36.3 vs 36.7), but ternary 256 loses to 16-bit 128 (26.8 vs 21.4) [P numbers, D reading].
- Learning rate, Table 1 [P]: 0.01 beats 0.001 only at hidden 32, and at 64 for median. At 256, lr 0.01 gives 63.8-77.7 against 26.8-28.7 at 0.001.
- Table 2 [P], hidden 64: lr 0.1 fails everywhere (ternary 154-240). Weight decay 0-0.10 shifts ternary perplexity by 2-8 with no consistent direction.
- Table 3 [P], vision, weak baselines: MNIST 100K 96.93 (16-bit) vs 96.08; CIFAR-100 2.2M 36.62 vs 42.27.
- Follow-ups [F]: encoders show the same 2x pattern (ternary 384 about equals 16-bit 192; arXiv 2411.05882). OLMo-1B, 10K steps: loss 2.95 for 16-bit, about 3.15 ternary throughout, about 3.088 switching to ternary at step 2K; a gradual lambda ramp gave "little to no benefits" (arXiv 2502.11895).

## Smallest scale tested, and whether the result held there
Hidden 32, 4 layers, 1 head, "6M" parameters: training was stable but perplexity was 1.5x the 16-bit model's. Parameter counts grow linearly with hidden size (6/12/24/48M), which fits the Hugging Face Mistral default MLP width of 14336 (my count: 5.8/11.6/23.3/47.1M) [D, not stated in the paper]. If so, these are narrow streams with huge MLPs, and "2x hidden" means 2x parameters, not 4x.

## Training recipe and gotchas
- Adam, batch 128, 10 epochs, Hugging Face Trainer, from scratch [P].
- At our width use lr 1e-3, not 1e-2 [P].
- Per-token absmax lets one outlier activation crush the rest into few levels [D].
- The bias is added in the quantized domain, before rescaling [R].

## Mapping onto the GBA design
- The forward pass is already integer (8-bit x_q, ternary W_q), so our table kernel applies unchanged: one lookup+add and one ROM byte per 4 weights.
- Norm collapses [D]: with a gain-free LayerNorm the standard deviation cancels against absmax, so `x_q = round((x - mean) * 128 / max|x - mean|)`. Per input of width n: n adds, n compares, one reciprocal lookup from ROM, n multiplies (or shifts with a power-of-two scale). Roughly 6K cycles at n=384 (estimate); no square root.
- `1/w_scale` is one constant per matrix or expert: store as multiplier+shift in ROM. `1/x_scale` is one number per token per BitLinear, needed only where a branch rejoins the residual stream.
- RAM: tables for n=384 are 96 x 32 x 2 bytes, about 6 KB.
- Budget [D]: a ternary weight costs 1.6 cycles against 7.5 for 8-bit (4.7x). If parity needs 2x the weights, ternary wins about 2.3x per cycle; if 4x, only about 1.2x.

## Verdict
Adopt as the baseline quantizer and as the capacity prior: plan for 0.4-0.5 nats lost against a float model of equal shape. Test median vs mean vs the 3:4 format. Settling experiment: at one fixed cycle budget, train ternary width W, 8-bit width about 0.46W and a float reference on our data, with lr in {3e-4, 1e-3, 3e-3}.

## Could not verify
- Corpus identity, sequence length, lr schedule, seeds.
- Whether embeddings and output head were ternarized; the MLP width (inferred).
- The LM training script: not found among the repo's examples.
- Figures: text descriptions only. Follow-ups were read selectively.

## Links worth following
- https://arxiv.org/abs/2502.11895 (16-bit then ternary)
- https://arxiv.org/abs/2411.05882 (encoder and MLP scaling)
- https://github.com/schneiderkamplab/bitlinear (`bitlinear/bitlinear.py`)
