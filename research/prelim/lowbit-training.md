Tags: [P] = read in the primary source this session; [S] = search snippet or abstract only; [M] = my memory, not checked.

## 1. BitNet b1.58 Reloaded — https://arxiv.org/html/2407.09527
- **Method [P]:** s = 1/(Measure(|W|)+eps), Wq = clip(round(W*s), -1, 1); Measure is mean (original) or median (theirs). Activations 8-bit absmax.
- **Smallest LM [P]:** hidden 32, 4 layers ("6M" params, embedding-dominated). All models trained from scratch.
- **Perplexity, 16-bit / ternary-mean / ternary-median [P]:**

| Hidden | 16-bit | Mean | Median |
|---|---|---|---|
| 32 | 77.8 | 130.1 | 116.6 |
| 64 | 36.7 | 67.4 | 60.0 |
| 128 | 21.4 | 36.3 | 37.5 |
| 256 | 16.6 | 27.1 | 26.8 |

- **"~2x hidden size" [P]:** only loosely supported. Ternary h=128 roughly matches 16-bit h=64, but ternary h=256 (26.8) is still behind 16-bit h=128 (21.4). Budget more than 2x width.
- **Median vs mean [P]:** median wins at the smallest sizes, ties above, and converges more slowly.
- **Learning rate [P]:** ternary best at 0.01, ten times the 16-bit rate (0.001). At 0.1 it does poorly, and weight decay hurts at high rates.
- **16-bit pretraining then ternary:** not in this paper [P]. A follow-up by the same group (arXiv 2502.11895, I believe) reports it helps [M].

## 2. Sherry — https://arxiv.org/html/2601.07892; Tequila — https://arxiv.org/html/2509.23809v2
- **3:4 constraint [P]:** in each contiguous block of 4 weights along the input dimension, the smallest |w| becomes 0 and the other three become sign(w). That gives 32 patterns in 5 bits (1 sign + 4-bit index), which matches your table.
- **Scale [P]:** alpha = mean |w| over the non-zero weights. Granularity at 1B: per-tensor 0.502, per-channel 0.513, per-group-of-128 0.519.
- **Enforcement [P]:** the quantizer is recomputed every forward pass with a plain straight-through estimator.
- **Weight trapping [P]:** naive 3:4 training polarises weights towards binary; gradient effective rank collapses to under 750 of 4096.
- **Arenas [P]:** train with Y = X*T*alpha + lambda_t*X*W (W is the latent full-precision weight). lambda_t = 0.5(1+cos(pi*p)) goes from 1 to 0, with warm-up. The residual vanishes at the end, so inference is pure ternary.
- **Accuracy [P]:** 5-task average at 1B is 0.519, against 0.489 for unconstrained absmean ternary, 0.519 for Tequila and 0.558 for BF16. At 3B it is 0.567, against 0.551, 0.576 and 0.636.
- **Scale tested [P]:** quantization-aware fine-tuning of pretrained LLaMA-3.2 1B/3B on 10B tokens at lr 1e-4; speed is benchmarked at 0.7B. There is no from-scratch or sub-100M evidence.
- **Tequila [P]:** absmean with deadzone alpha/2. Deadzone weights add an input-independent bias, Y = X*W_hat*alpha + sum over deadzone of lambda*w_i, with lambda = 1e-3. It folds into a per-output bias offline. "1.67-bit" is its packing of 3 weights into 5 bits.

## 3. Small ternary LMs
- **TernaryLM [P]** — https://arxiv.org/html/2602.07374v2. 132M parameters, 12 layers, d=768, TinyStories.
  - Threshold 0.5*std(W) with a learnable per-layer alpha; embeddings and output head stay FP32.
  - AdamW (0.9, 0.95), lr 1e-3 cosine, 1000 warm-up steps, weight decay 1e-5, clip 1.0, label smoothing 0.1.
  - Ternary validation perplexity is 58.42 against 28.25 for FP32, so the gap is large.
  - Ablations (perplexity): fixed alpha 67.3, binary 72.2, quantized embeddings 89.5.
  - The token count is inconsistent within the paper.
- **BitNet 2B4T [P]** — https://arxiv.org/html/2504.12285. Absmean ternary, per-token absmax 8-bit activations, SubLN, squared ReLU, no biases. Two stages: high lr with weight decay 0.1, then at mid-training an abrupt lr drop and weight decay 0. No numeric learning rates are given.
- **BitNet a4.8 [P]** — https://arxiv.org/html/2411.04965. Train with 8-bit activations (95B tokens), then switch to 4-bit (absmean, scale sqrt(7), clip [-8, 7]) for the last 5B. Smallest model 700M.
- **BitDistill [S]** — https://arxiv.org/abs/2510.13998. SubLN, MiniLM-style attention distillation and continued pretraining from a full-precision model.
- I found nothing credible that is ternary, from scratch and under 10M parameters, beyond Reloaded.

## 4. Quantization-aware training practice
- LSQ (https://arxiv.org/abs/1902.08153): learn the step size, with gradient scale 1/sqrt(N*Qp) and init 2*mean|w|/sqrt(Qp) [M]. TernaryLM's learnable-alpha ablation supports a learned scale [P].
- Recipe: latent FP weights with a plain straight-through estimator; a higher lr than FP; weight decay early and zero late; an FP residual annealed to 0; FP-teacher logit and attention distillation.

## 5. Attention
- **Asymmetric 1-bit keys [P]** — Self-Indexing KVCache, https://arxiv.org/html/2603.14224v1.
  - Keys are mean-centred per channel, then sign bits are grouped by 4 dimensions into 16 centroids per group.
  - Per query, build a 16-entry table per group; score = sum of table lookups. This is the same structure as your weight tables.
  - Used training-free for top-k selection on 8B/14B models: LongBench 58.2 against 58.7 for the full cache.
- **H1B-KV [S]** — https://arxiv.org/abs/2510.05529. 1-bit key sketches and 4-bit values; needs light fine-tuning.
- **Explicit Sparse Transformer [P]** — https://arxiv.org/pdf/1912.11637.
  - Scores below the k-th largest in a row are set to -inf before softmax; gradients flow only through the selected entries.
  - Applied from the start of training. k=8 was best of {4, 8, 16, 32}; k=4 was worse on De-En.
  - Gains are about +0.3 BLEU; enwik8 1.05 bpc at 41M against 1.06 for Transformer-XL. Training with top-k alone also helped.
  - Not tested on tiny models or with quantized keys.
  - Top-k is 2x faster than sparsemax and 10x faster than entmax-alpha at inference.

## 6. Integer-only inference
- **I-LLM [P]** — https://arxiv.org/html/2405.17849v2.
  - Exponent by shifts: log2(e) is approximated as 1 + 1/2 - 1/16; 2^(-q) is a right shift, with linear interpolation of the remainder.
  - Softmax: subtract the max and clip the range to 15 below it (exp(-20) is about 0).
  - RMSNorm: a bitwise integer square root (16 iterations, shift, compare and subtract) and one integer divide. The extracted pseudocode has an apparent typo, so check the repo.
- **I-BERT** — https://arxiv.org/abs/2101.01321. Second-order polynomial GELU and exponent, integer Newton square root [M]; the abstract confirms only "integer-only approximations" [S].
- With top-8 attention, softmax becomes 8 lookups in a 256-entry exp table plus one divide (my inference).

## Recommended recipe for your case (synthesis, not from one paper)
1. Train an FP teacher of the same architecture: RMSNorm or SubLN, squared ReLU, no biases, top-8 masking from step 0.
2. Start the student from the teacher. Use the Sherry 3:4 quantizer on blocks aligned to your table grouping, with a per-row learned scale initialised as mean |w| of the non-zeros and folded into the integer requantization shift.
3. Use the Arenas residual with a cosine 1-to-0 schedule over about 70% of training, then finish pure ternary with weight decay 0 and a low lr.
4. Peak lr about 1e-3 to 1e-2 (sweep), with weight decay 0.05-0.1 early only.
5. Quantize activations to 8 bits throughout; add any lower-bit stage only in the last 5-10%.
6. Keys: mean-centre, take the sign with a straight-through estimator, keep queries FP. Distil the teacher's logits and attention maps.
7. Keep the embeddings and output head at 8 bits or more.

## Three biggest risks
1. **Capacity.** At small widths ternary costs 1.5-2x perplexity; "2x hidden size" is optimistic, and 3:4 adds a further untested constraint below 0.7B.
2. **Train/inference mismatch.** A residual lambda that has not reached exactly 0, floating-point norm or softmax during training against integer tables, and top-k tie-breaking can each silently break the deployed model.
3. **Binary keys with top-8 at tiny head dimensions.** Sign-key scoring is only validated on 8B+ models with training-free selection; with few key bits the ranking noise may select the wrong 8. Use at least 64 key bits and train with the quantized keys in the loop.