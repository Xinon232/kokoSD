# N-vium: Mixture-of-Exits Transformer for Accelerated Exact Generation

https://arxiv.org/abs/2605.13190 (v1, 13 May 2026). Aleksander Lorenc, Frédéric Berdoz, Joël Mathys, Roger Wattenhofer (ETH Zurich). Code: https://github.com/ETH-DISCO/n-vium (MIT).

Reading status: arXiv HTML read in full through a fetch tool that extracts text (primary, but figures and Algorithm 1 were not readable); GitHub README read. The paper also calls the 4-exit model "Quadrivium" / "4-vium".

## What it is

A decoder-only LM with prediction heads at N_V evenly spaced depths. The output distribution is a learned per-token mixture of those heads, trained from scratch with plain next-token loss plus a compute penalty. At inference an exit is sampled first, then a token from that exit's head, which samples exactly from the mixture. Layers above the exit are not skipped: they are deferred and computed later, batched with a later token that goes deeper ("piggybacking"). The speedup is therefore a GPU-parallelism gain; total backbone compute equals the dense model.

## Mechanism

- Backbone: L layers, junctions at l_k = k*L/N_V, k = 1..N_V. LLaMA-2 architecture and tokenizer (|V| = 32,000), input embeddings NOT tied to the LM head.
- Per junction: own RMSNorm on the residual stream, then
  - router: bottleneck MLP h -> floor(0.66h) -> h, linear to 2 logits, softmax -> w_k in [0,1]; w_{N_V} = 1.
  - adapter: one-hidden-layer MLP, hidden size h, own weights -> o_k.
  - head: pi_k = softmax(o_k W_lm), W_lm shared by all junctions.
- Exit weights: p_k = w_k * prod_{j<k}(1 - w_j). Mixture: pi_mix = sum_k p_k pi_k.
- Loss: L_main = L_mix + beta * L_compute, with L_mix = -E[log pi_mix(x_{t+1})], L_compute = E[sum_k p_k l_k / L].
- Warmup (first 5% of steps, alpha = 1.0): L_mix + alpha * E[sum_{k<N_V} (w_k - 1/(N_V-k+1))^2], which pushes p_k to 1/N_V. Without it the router collapses to one exit.
- Sampling:
  ```
  for k in 1..N_V:
      run block k on token t (plus any deferred tokens waiting at this depth)
      if k == N_V or Bernoulli(w_k): x_{t+1} ~ pi_k; mark t deferred from l_k; break
  ```
- Added parameters: 3.3% on average. FLOP terms per token (Eq. 2, K early junctions): heads (K+1)*2|V|d, adapters (K+1)*4*d*d_a, routers K*4*d*m_r.

## Numbers

All from-scratch on C4, D = 20N tokens, context 1024; baselines are parameter- and data-matched dense models. Speedup = latency ratio on Spec-Bench, one A100.

Table 2 (4-vium, beta = 0.15):

| Shape | Params | Dense PPL | 4-vium PPL | dPPL | Speedup |
|---|---|---|---|---|---|
| 6L x 768 | 99M | 27.99 | 27.63 | 1.3% | -8.3% |
| 12L x 768 | 141M | 22.99 | 22.29 | 3.1% | 10.9% |
| 24L x 768 | 226M | 18.90 | 18.21 | 3.6% | 26.8% |
| 36L x 768 | 311M | 16.88 | 16.25 | 3.7% | 36.4% |
| 48L x 768 | 396M | 15.94 | 15.18 | 4.8% | 41.6% |
| 24L x 128 | 13M | 78.02 | 65.89 | 15.5% | 53.8% |
| 24L x 256 | 36M | 35.84 | 33.92 | 5.4% | 35.3% |
| 24L x 512 | 112M | 22.26 | 21.28 | 4.4% | 25.0% |
| 24L x 1024 | 381M | 17.50 | 17.12 | 2.2% | 23.8% |
| 24L x 1536 | 806M | 13.35 | 13.34 | 0% | 32.0% |

- 1.5B (48L x 1536, adaptive beta), Appendix G: PPL 11.56 vs 11.63 dense, 57.9% speedup. Table 6: 57.3% on RTX A6000, 57.7% on RTX 3090, 32.0% on RTX 2080 Ti.
- Table 5 (iso-speed): 4-vium 36L (311M, PPL 16.25, 48.86 tok/s, 0.834 GFLOP/token) vs dense 26L (233M, PPL 16.51, 48.36 tok/s, 0.499 GFLOP/token). 4-vium 48L (PPL 15.18, 1.041 GFLOP) vs dense 33L (PPL 15.50, 0.620 GFLOP). So N-vium wins by about 2% PPL at equal GPU speed while spending about 1.67x the FLOPs per token.
- Table 3 (after Tulu-3 fine-tuning): PPL 17.37 vs 22.09 at 6L; 9.96 vs 12.22 at 48L. Benchmark accuracies near chance at these sizes.
- Figure 5: mixture perplexity "lies well below every individual head" (24L, exits at 6/12/18/24).
- Appendix D: retrofitting Llama-3.2-1B (junctions at 4/8/12, beta = 0.1) drove all early-exit probabilities to zero.
- Training costs about 1.2-1.3x FLOPs per step (Limitations section).

## Smallest scale tested, and whether the result held there

Smallest by parameters: 13M (24 layers x 128 wide; with a 32K untied vocabulary most of those parameters are embeddings). The perplexity gain was largest there (15.5%). Smallest by depth: 6 layers x 768 (99M): gain only 1.3% and generation was 8.3% slower; the paper gives no explanation. No model combines small width with small depth, so nothing matches our 4-6 layer, 256-384 wide shape.

## Training recipe and gotchas

- AdamW (0.9, 0.95), weight decay 0.1 (not on biases, norms, embeddings), clip 1.0, cosine to 10% of peak with 1% warmup, peak LR 3e-4 to 1e-3 by scale, batch 128K-512K tokens, bfloat16. About 1200 H200 node-hours (4 GPUs per node) for the whole study.
- Balance warmup is required; from-scratch is required (retrofit collapsed).
- beta = 0.15 fixed for the scaling grid. For 1.5B, beta is driven by a feedback controller on delta = L_baseline - L_mix (gains 8.0 up, 4.0 down, flat zone 0.005, beta in [0, 10]); this needs a dense baseline loss curve.
- The paper never mentions temperature, top-k or greedy decoding. Exactness holds for sampling from pi_mix; greedy or truncated sampling over the mixture would need every head evaluated.

## Mapping onto the GBA design

- The speedup does not transfer. Deferred layers are still computed, one token at a time on our CPU, so the cost is dense backbone + routers + one adapter. The per-token FLOP count is higher than dense (Table 5), and even on a GPU the 6-layer model lost 8.3%.
- Overhead as published, at h = 256: router about 2*256*169 = 86.5K weights, adapter about 131K weights. A token that runs to the last of 4 junctions pays 3 routers + 1 adapter = about 390K weights, roughly 20-30% of the 1.3-2M weights-per-token budget (my arithmetic). A usable version would need a router of one dot product (256 adds) and no adapter; the paper does not test that.
- What could transfer is the quality effect: mixture training improved perplexity at equal parameters. But the gain comes from the mixture, not the final head, so all the extra machinery stays in the inference path; and at 6 layers the gain was 1.3%.
- Deferral itself could move work into idle time: the upper layers of the last tokens of a reply can be finished while the user reads or types. That is a scheduling trick worth a few tokens per turn, not a throughput gain. RAM cost: one hidden vector (256-384 bytes at 8 bits) per deferred token.
- Truly skipping upper layers (the CALM-style baseline, copying the last hidden state into the KV cache) is what would save GBA cycles; the paper reports it faster but worse in perplexity (Figure 3, no numbers readable).
- Nothing new to precompute into ROM; exit heads share the existing output head.

## Verdict

Skip for inference speed. Optional low-priority test of the training objective only: at our real shape (6 layers, 256-384 wide, ternary QAT), train a 2-exit mixture (exits at layers 3 and 6) with a linear router and no adapter, beta = 0, against a matched dense model. If the perplexity gain is under about 3%, drop it; the published 6-layer gain of 1.3% suggests it will be.

## Could not verify

- Exit fractions p_1..p_4, per-exit perplexities, average exit depth: only in Figure 5, not readable.
- Algorithm 1 text; Figure 3 (Pareto plot, CALM* and LayerSkip numbers, beta and N_V sweeps). The README lists sweeps of 13 beta values (0 to 0.6), 11 single-exit positions and N_V = 2..12, but I found no numeric results for them.
- Which token types exit early: not analysed in the paper.
- Why the 6-layer model is slower: not explained.
- No quantization, no CPU, no sub-13M and no simple-English results exist in this source.
- "NeurIPS 2026 poster" status comes from the task description; I did not confirm it.
- The abstract says 57.9% speedup; the introduction says "up to 57%".

## Links worth following

- https://github.com/ETH-DISCO/n-vium (configs in config_experiments/ hold the sweep definitions; checkpoints on the Hugging Face profile "alorenc")
- CALM (Schuster et al., 2022) and LayerSkip (Elhoushi et al., 2024): the true layer-skipping baselines that would actually save sequential compute.
