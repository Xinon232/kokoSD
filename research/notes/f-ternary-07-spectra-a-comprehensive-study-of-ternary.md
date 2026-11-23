# Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language Models (https://arxiv.org/abs/2407.12327, 2024; Kaushal, Vaidhya, Mondal, Pandey, Bhagat, Rish et al.; Nolano / Mila)

Read in primary form: full arXiv HTML (body + appendices A-G, all tables) and the repo README (branch `master`). Items marked [derived] are my arithmetic from the paper's fits, not reported numbers.

## What it is
A suite of 54 LLaMA-style models, 99M-3.9B params, all trained on the same 300B SlimPajama tokens in the same order: FP16 (FloatLM), GPTQ post-quantised 3/4/6/8-bit (QuantLM), from-scratch ternary (TriLM), plus three binary models. It gives a like-for-like ternary-vs-float scaling law, a ternary optimiser-schedule ablation, and a BitNet b1.58 replication. The smallest model (99M) is the closest public controlled ternary/float pair to our size.

## Mechanism
Linear layer, latent FP weights W (n x m), eps = 1e-5 (Table 1):
```
gamma = eps + mean(|W|)                      # ONE scalar per matrix
W_hat = round(clamp(W / gamma, -1, 1))       # {-1,0,+1}
Y     = X @ (gamma * W_hat).T                # STE: dL/dW = dL/dY^T X
```
Inference: cache W_hat and gamma. Binary variant: alpha = mean|W|, W_hat = sign(W - mean(W)).
- Architecture: pre-norm, RMSNorm **with learnable gain**, SwiGLU, RoPE, no biases. Norm only twice per block (before attention, before MLP).
- Unlike BitNet b1.58: no per-linear SubLN, **no activation quantisation** (activations stay FP16).
- Embeddings and LM head stay FP16 (never ternarised).
- 99M config (Table 3): hidden 512, GLU 1280, 8 heads, 16 layers. 190M: 768/2048/12/16.
- Schedule (Sec. 3.3): linear-decay LR with warmup + weight decay, then (1) at ~1/2 of training lower the peak LR, (2) at ~2/3 set weight decay to 0. 99M/190M: LR 2.4e-3 -> 1.5e-3 (float baseline 4e-4, cosine). AdamW betas (0.9, 0.95), batch 1M tokens (float 2M), seq 2048.

## Numbers
- Scaling fit, validation loss at 300B tokens (Eq. 1, Fig. 9b): L = A/N^0.26 + eps; TriLM A=185, eps=1.76; FloatLM A=159, eps=1.67. Same exponent.
  - [derived] gap at 99M about 0.31 nats (3.30 vs 2.99); at 3.9B about 0.17; asymptote 0.09. So "converge" in the text is loose: the fitted offsets differ.
  - [derived] at 99M a ternary model equals a float model of ~45M params; matching float-99M needs ~230M ternary params (~2.3x).
  - Paper: TriLM within 7% / 6% of float loss at 15.6B / 330B (Fig. 10).
- 6-task commonsense average (Table 7), TriLM vs FloatLM: 99M 43.5 vs 44.3 (binary 39.8); 190M 42.4 vs 46.7; 390M 46.4 vs 48.5; 1.1B 51.6 vs 54.9; 3.9B 60.7 vs 61.4.
- LAMBADA perplexity (Table 9), the most LM-like metric: 99M 172.0 vs 85.0 (binary 468.3); 190M 130.7 vs 50.3; 390M 77.7 vs 21.9; 1.1B 17.3 vs 11.7; 3.9B 6.3 vs 6.7. The small-scale gap is far bigger than the commonsense averages suggest.
- GPTQ at 99M (Table 9): 4-bit 211.6, 3-bit 4765 ppl; 190M 4-bit collapsed. Post-training quantisation below 6 bits is unusable at our size; ternary must be trained.
- Schedule ablation, 1.1B, 100B tokens (Tables 10-11 avg): both interventions 50.2; only WD removed 50.1; only LR drop 49.7; baseline 49.1; float 52.2. Training-loss ranking the same (Fig. 6, no numbers given).
- BitNet b1.58 replication at 1.1B/100B: 48.9 vs TriLM 50.2 (Table 11).
- Binary is worse than ternary at every size tested (99M, 560M, 1.1B).
- Sizes in bits (Table 4): TriLM 99M = 0.90e9 bits vs float 1.60e9: FP16 embeddings/head dominate at small scale.

## Smallest scale tested, and whether the result held there
99M (hidden 512, 16 layers), 300B tokens. Ternary trains stably and is usable, but the "matches float" headline does **not** hold: LAMBADA ppl doubles. Also (Sec. 4.3) the mid-training LR drop made training loss **go up** for models below 1B; only WD removal helped at all sizes. The schedule ablation itself was run only at 1.1B.

## Training recipe and gotchas
- LR about 6x the float LR; linear decay, not cosine.
- WD -> 0 at 2/3 is the robust intervention; treat the LR drop as unproven (possibly harmful) at our scale.
- Per-matrix (not per-row/group) scale is enough; released checkpoints have up to MP<=6 scales per matrix (sharding artefact, MP=1 at 99M).
- TriLM underperforms on in-distribution/memorisation corpora (Fig. 13): lower memorisation capacity per parameter.
- Weight-decay value, warmup length and zero-fraction of ternary weights are not stated. Training code is not in the repo (README only; packed kernels not released).

## Mapping onto the GBA design
- Inference cost: none beyond the existing ternary kernel. One gamma per matrix folds into the following RMSNorm gain or an integer shift; 0 extra lookups, 0 extra RAM, a few bytes of ROM.
- Two norms per block instead of BitNet's 4-7 saves inverse-sqrt work (no divide on ARM7); but Spectra kept float activations, so its stability claim is not evidence for our 8-16-bit activations. BitNet-style activation quantisation remains our own risk.
- Keep embeddings non-ternary: ROM-lookup embeddings (hashed n-gram, per-layer) cost nothing as 8-bit, and Spectra never ternarised them. The output head is the same issue: our ternary class/member head is outside what Spectra tested.
- Capacity planning: budget roughly 2x ternary params per float-equivalent param near 100M [derived], likely worse below. This strengthens the case for routed experts and ROM lookups (stored, not touched) over dense width.
- Our 32-entry "one zero per 4" format assumes ~25%+ zeros; Spectra does not report sparsity, but the 99M checkpoint is public and can be measured.

## Verdict
**Test.** Adopt per-matrix absmean and two-norm pre-norm as the baseline (already close to ours). Run a 4-arm schedule ablation (none / WD-off at 2/3 / LR-drop at 1/2 / both) on the pilot model at fixed tokens; expect WD-off to win, LR-drop to be neutral or negative. Second cheap check: histogram zeros in TriLM_99M weights to sanity-check the 1-in-4-zero packing.

## Could not verify
- Exact validation losses per model (figures only; gaps above are from the fits).
- Weight decay value, warmup steps, LR-drop shape (step vs re-scaled decay).
- Zero fraction of trained ternary weights.
- Repo file tree (GitHub API blocked); only README read. Intermediate checkpoints: README says "coming weeks", the abstract says 500+ released.
- Nothing below 99M, nothing on small vocabularies or simple-English data.

## Links worth following
- https://github.com/NolanoOrg/SpectraSuite (README, branch master)
- https://huggingface.co/SpectraSuite/TriLM_99M_Unpacked and FloatLM_99M (paired checkpoints for sparsity / per-layer analysis)
- BitNet b1.58 training tips (Ma et al. 2024), the source of the two-stage LR / WD schedule
