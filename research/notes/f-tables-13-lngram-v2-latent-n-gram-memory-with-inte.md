# Lngram v2: Latent N-Gram Memory with Interpretable Discrete Representations  (https://arxiv.org/abs/2609.03426, Sept 2026; Yunao Zheng, Bin Wen, Xiaojie Wang + 12; BUPT + Kuaishou)

Tags: [P] arXiv HTML (v2), read through a fetch tool that answers questions over the page text, so equations are transcribed, not eyeballed; [E] my arithmetic; [S] secondary.

## What it is
Second version of Lngram (hidden-state sign bits -> per-route symbols -> exact 2/3-gram table lookups). v1 tied the route count to width (R = d/4) and read out through a huge linear layer over the concatenation of all routes. v2 makes routes R, memory width d_m and backbone width d independent, and replaces the readout with a small attention over the R retrieved rows, with a "Sink" that lets the model attend to nothing. It keeps the counterfactual surrogate gradient. Evidence is mostly on 2B and 30B vision-language models; one text-only from-scratch run at 0.6B.

## Mechanism
Per memory layer, position t [P §2, App. A, Table 13]:
```
z = RMSNorm(h_t) W_q                 # W_q: d x (R*M), no bias; M=4 bits, K=16
a[t,r] = pack(z[r*M:(r+1)*M] > 0)    # hard in forward
for n in {2,3}: m(n)[r] = E(n)[ r*K^n + sum_i a[t-n+1+i,r]*K^i ]   # E(n): (R*K^n) x d_m
x[r] = concat_n m(n)[r]              # R memory tokens of 2*d_m
Q = reshape(Norm(h_t) W_Q)           # H_q heads x d_h, H_q*d_h = d
[Kk[r]; V[r]] = reshape(Norm(x[r]) W_KV)   # H_kv heads x d_h each; W_KV shared over routes
s[a,r] = Q[a]·Kk[r, a//G] / sqrt(d_h),  G = H_q/H_kv
alpha[a,r] = exp(s[a,r]) / (exp(b0) + sum_r' exp(s[a,r']))   # Sink: no value vector
b0 = log(R(1-mu0)/mu0), mu0 = 0.5    # fixed, not learned
o = concat_a(sum_r alpha[a,r] V[r, a//G]) W_O          # W_O zero-init
h += o + SiLU(DWCausalConv(o))       # kernel 4, dilation 3, zero-init; before attention
```
One-bit surrogate (the "approximate" mode used in all main runs): `dL/dz_j = λ τ p_j(1-p_j) <g, E_j(1) - E_j(0)>`, p = σ(τz), g = gradient at the retrieved row. Exact variant: `λ τ Σ_c P(c|z)(β_j(c) - p_j)<g, E_c>`. τ, λ values not found.
Configs [P Table 13]: Keye2B: layers {2,14}, R=121, d_m=128, H_q=H_kv=16, d_h=128 (so d=2048 [E]). Keye30B: layers {2,24}, R=256, d_m=256, streamed in 64-route chunks.
Parameter check [E]: r16/m4/KV8 at d=2048, d_m=128, two layers: tables 2·16·4352·128 = 17.8M; W_Q + W_O + W_KV + W_q ≈ 9.05M per layer = 18.1M. Paper: 35.939M total, 18.121M active. Match, so "active parameters" are almost entirely the two d x d projections.

## Numbers
- Table 4 / App. D.3 (text-only; 0.6B DeepSeek-V2-style MoE, 24 layers, d=768, random init, 10,000 steps ≈ 1.31B tokens FineWeb-Edu, one shared seed, layers {2,12}): val loss baseline 2.8899; v1 2.8609; v2 R=192 2.8527; R=64 2.8578; R=16 2.8584. At R=16, v2 has 82.6% fewer module parameters and 95.2% fewer activated parameters than v1. No Engram (token-id) baseline.
- App. A.5 (same family, R=16, d_m=64, seed 42, 10,000 steps): surrogate 2.8585 vs plain STE 7.3349 (diverged).
- Table 2, Keye2B, ~5B continued tokens, 3 seeds, 10-benchmark avg: baseline 47.25±0.15; +v2 47.93±0.18; +PKM 47.24; +matched top-1 sparse FFN (155.8M / 21.0M active) 45.86 (single runs).
- Table 2 / 11, Keye30B, single run: 78.23 -> 79.71; after Holm correction only 2 of 10 tasks significant. Without Sink: 79.39.
- Table 3 (Keye2B, 3-task avg, single runs, confounded): baseline 53.64; r16/KV8 54.20; r64/KV8 55.71; r121/KV16 56.16; r123/KV2 53.69; r16/m5/KV2 52.97; r512/mem256 (1.19B params) 55.14.
- Table 12 (H800): decode 48.26 -> 50.97 ms (r16) / 52.14 ms (r64).
- Table 9: ~7.8 effective codes of 16 per route, 0 dead of 2,048. Table 1: Hamming-KNN on route ids keeps 66-84% of the hidden-state KNN's excess AP.
- App. B: on SigLIP2+Qwen3-1.7B the gain is 47.21 -> 47.38, i.e. almost none.

## Smallest scale tested, and whether the result held there
0.6B MoE, d=768, 1.31B tokens. It held (−0.03 nats vs baseline, R=16 within 0.006 of R=192), one seed. Nothing at d ≤ 512, nothing dense, nothing quantised.

## Training recipe and gotchas
- Table LR 4e-4 (min 1e-5), other Lngram parts 1e-4 (min 2e-6), wd 0 on tables and W_q; backbone 5e-6 (fine-tune). Tables init normal std 1.0 (v1 was zero-init); W_O and conv zero-init, so the module starts as a no-op.
- STE diverges; the surrogate is mandatory. One-bit costs M extra row fetches per route per n-gram position.
- Low KV-head counts looked worst (KV2 rows), but are confounded with R and M.
- Gains were weak on a second backbone (App. B).

## Mapping onto the GBA design
All [E], d=256, R=16, M=4, d_m small.
- W_q: 64x256 ternary = 16K weights ≈ 26K cycles (or reuse attention sign keys).
- Fold W_KV into ROM: if the norm over x[r] is dropped or made per-order, Kk and V are sums of two per-row vectors, so store K|V per table row. With H_kv=2, d_h=32: 128 B/row x 69,632 rows = 8.9 MB per layer; with H_kv=8: 35.6 MB (too much).
- Run time: W_Q 65K ternary weights ≈ 105K cycles; scores R·d = 4,096 8-bit MACs ≈ 31K; exp via LUT over 17 logits x 8 heads; mix 4,096 MACs ≈ 31K; W_O 65K weights ≈ 105K. About 300K cycles per layer (6-10% of budget) versus ~65K for the folded v1-style sum in note f-tables-10.
- Cheaper variants to try: Q = slice of h (no W_Q); V stored already at width d with one head and no W_O; hard top-k routes instead of softmax. None tested by the authors.
- RAM: 8 bytes of codes per position per layer; only two past positions needed. Drop the conv.
- The 95.2% saving is versus v1's concat readout, which our fold already removed; it is not a saving versus the folded design.

## Verdict
Test; stays low priority, but de-risked. The useful facts are (a) 16 routes (64 bits) work nearly as well as 192 at d=768, which is the size we can afford, and (b) surrogate vs STE is now a reported number. Still no comparison against token-id n-gram tables, and one seed. Experiment: at d=256 with QAT, token-id tables vs +latent R=16 with (i) folded gated sum and (ii) Sink-attention readout, at equal ROM bytes; keep (ii) only if it beats (i) by more than its ~235K extra cycles would buy elsewhere.

## Could not verify
- τ, λ, PKM configuration, a definition of "activated parameters", v1's absolute parameter counts.
- Whether any code is released: none found in the paper text or by search.
- Figures not inspected. No independent replication found.
- Task brief lists authors as "Zheng, Wen, Wang"; the paper has 15 authors.

## Links worth following
- https://arxiv.org/abs/2605.24869 (v1; note f-tables-10)
- https://github.com/zyaaa-ux/Lngram (v1 code; check for a v2 update)
- https://arxiv.org/abs/1907.05242 (PKM baseline)
