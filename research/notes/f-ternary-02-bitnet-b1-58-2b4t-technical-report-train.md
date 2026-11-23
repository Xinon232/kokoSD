# BitNet b1.58 2B4T Technical Report  (https://arxiv.org/abs/2504.12285, April 2025, Ma, Wang, Huang, Zhang, Hu, Song, Xia, Wei; Microsoft Research)

Tags: [P] paper HTML (arxiv.org/html/2504.12285, read in full); [R] official repo `microsoft/BitNet` `gpu/model.py`, `gpu/convert_checkpoint.py`; [C] Hugging Face `config.json` / model card; [D] my derivation.

## What it is
The release report for the first open ternary LLM trained from scratch at scale: 2B parameters, 4T tokens [P]. Linear layers in the blocks have ternary weights and 8-bit activations; embeddings and the output head stay full precision [R]. It describes the recipe in words and reports benchmarks against 1-2B float models [P]. It contains no ablations and no pre-training learning rates, batch size, optimizer or step count [P]. The implementable detail is in the repo, not the paper.

## Mechanism
Shapes [C][R]: 30 layers, d=2560, 20 heads (head_dim 128), 5 KV heads (grouped-query), FFN hidden 6912 (2.7x), RoPE theta 500000, context 4096, vocab 128,256, tied embeddings, RMSNorm eps 1e-5, no biases.

```
# BitLinear [R]; latent float W, straight-through estimator in training
s_x = 127 / max(|x|, over last dim).clamp(min=1e-5)   # per token
x_q = clip(round(x*s_x), -128, 127)
s_w = 1 / mean(|W|).clamp(min=1e-5)                    # one scalar per matrix
W_q = clip(round(W*s_w), -1, 1)                        # zero iff |w| < 0.5*mean|W|
y   = (x_q @ W_q.T) / (s_x*s_w)

# Block (pre-norm plus SubLN) [R]
h = x + wo( RMSNorm_sub( attention( wqkv( RMSNorm(x) ) ) ) )
x1, x3 = split( w13( RMSNorm(h) ) )                    # two d -> 6912 projections
out = h + w2( RMSNorm_sub( relu(x1)**2 * x3 ) )
logits = Linear_fp( RMSNorm(out_final) )               # head is nn.Linear, not BitLinear
```
- SubLN = an extra norm inside each sublayer, before its output projection (`attn_sub_norm` on the concatenated head outputs, `ffn_sub_norm` on the FFN hidden) [R].
- The FFN is gated: `relu(x1)^2 * x3`, three matrices as in SwiGLU [R]. The paper says only "ReLU² instead of SwiGLU" [P].
- Scales are separate for the Q, K, V slices of `wqkv` and for the two halves of `w13` [R].

## Numbers
All at 2B parameters, instruction-tuned, re-run by the authors [P].
- Table 1, 16-benchmark average: BitNet 54.19; Qwen2.5-1.5B 55.23 (18T tokens); SmolLM2-1.7B 48.70; LLaMA-3.2-1B 44.90; Gemma-3-1B 43.74; MiniCPM-2B 42.05.
- Table 1, efficiency: non-embedding memory 0.4 GB vs 1.4-4.8 GB; CPU latency 29 ms/token vs 41-124 ms (i7-13800H, 8 threads, 128 tokens, bitnet.cpp vs llama.cpp); estimated energy 0.028 J vs 0.186-0.649 J.
- Table 2 (MMLU, GSM8K, IFEval average): BitNet 55.01; Qwen2.5-1.5B bf16 55.72, GPTQ-int4 52.15, AWQ-int4 51.17 (0.7 GB).
- Table 3, 11-benchmark average of 1-bit models: BitNet 60.68; Falcon3-1.58bit-7B 50.76; Llama3-8B-1.58 49.75; Bonsai-0.5B 41.06; OLMo-Bitnet-1B 32.22.
- Parameter check [D]: 69.5M per layer x 30 = 2.08B ternary (0.41 GB), plus 328M float embedding.

## Smallest scale tested, and whether the result held there
One scale only: 2B parameters, 4T tokens, about 2,000 tokens per parameter [D]. Nothing is shown for small models or small token budgets. The smaller native ternary models in Table 3 are far worse, but they differ in data and recipe, so that is not a controlled comparison.

## Training recipe and gotchas
- Stage 1: cosine decay from a "relatively high" peak learning rate; weight decay on a cosine schedule peaking at 0.1; bulk web data [P].
- Stage 2, from about the midpoint of the token budget: abrupt drop, then a second cosine from a much lower peak; weight decay 0; higher-quality data [P].
- SFT: cross-entropy summed over tokens, not averaged; larger learning rate and more epochs than a float model; no values given [P].
- DPO: 2 epochs, lr 2e-7, beta 0.1 [P].
- Per-token absmax: one outlier sets the scale for the whole vector. SubLN bounds the input of `wo` and `w2`, the two places where ReLU² would otherwise produce huge ranges [D].

## Mapping onto the GBA design
- Kernel: unchanged; one lookup+add and one ROM byte per 4 weights.
- Gated FFN costs 3·d·h weights against 2·d·h ungated. For a routed expert with d=320, h=640 that is 614K vs 410K weights, about 0.33M extra cycles per layer [D].
- ReLU² and the gate: about 2 multiplies per hidden unit. With 16-bit pre-activations the product reaches 48 bits, so shift x1 down to about 8 bits before squaring [D].
- Sub-norm collapses [D]: RMSNorm followed by per-token absmax is `x_q = round(127 * g_i*u_i / max_j|g_j*u_j|)`. The RMS cancels, so no square root: n multiplies by the gain, n compares, one reciprocal (ROM table), n multiplies. Roughly 20 cycles per element, 13K cycles at h=640.
- Scales [D]: any per-matrix scale that feeds a norm cancels (V, and both halves of w13). Only `wo`, `w2` (residual add) and Q·K (softmax temperature) need a ROM multiplier+shift.
- RAM: partial-sum tables for n=640 are 160 x 81 x 2 bytes = 26 KB, too big for fast RAM with free ternary; 160 x 32 x 2 = 10 KB with the 3:4 format [D].
- The head and embeddings were float here; keep ours at 8 bits.

## Verdict
Adopt the block layout (pre-norm plus SubLN, no biases, absmean weights, per-token absmax activations) and the two-stage schedule as the default: it is the only recipe proven to reach float parity from scratch. Test, do not assume, the gated FFN: it costs 50% more expert weights. Settling experiment at our scale and equal cycles: ungated ReLU² with h, against gated with 2h/3, each with and without SubLN; plus one run with weight decay 0.1 throughout against decay switched off at the midpoint.

## Could not verify
- Peak learning rates, optimizer, batch size, warm-up, pre-training sequence length, SFT epochs: not in the paper, card or repo files I read.
- Any ablation of SubLN, ReLU², the gate or the schedule: none in this source.
- Whether training used a gain in the sub-norms beyond what xformers `RMSNorm` provides: I did not read that class.
- Training code: the forward pass above comes from the inference code.

## Links worth following
- https://github.com/microsoft/BitNet (gpu/model.py, bitnet.cpp kernels)
- https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- https://arxiv.org/abs/2411.04965 (BitNet a4.8: cited for ReLU² sparsity)
- https://proceedings.mlr.press/v202/wang23u.html (Magneto: SubLN)
- https://arxiv.org/abs/2502.11880 (Bitnet.cpp: efficient edge inference for ternary LLMs; cited by the report)
