# Lngram: N-gram Conditional Memory in Latent Space  (https://arxiv.org/abs/2605.24869, v1 24 May 2026; Zheng, Xia, Wang, Ren; BUPT + Li Auto)

Tags: [P] paper HTML v1, read in full incl. appendices; [C] raw `Lngram.py` and README (MIT); [E] my arithmetic; [S] secondary.

## What it is
Engram with learned keys: instead of hashing token-id n-grams, each Lngram layer sign-quantises a projection of the hidden state into d bits, groups them into d/4 "routes" of 4 bits (16 symbols), and uses the 2-gram and 3-gram of each route's symbols over consecutive positions as an exact (unhashed) table address. Retrieved rows are concatenated, projected, gated by the hidden state and added to the residual before attention. Because addressing is hard, the projection is trained with a counterfactual surrogate gradient; plain STE "manifests as a failure to converge" [P §2.5].

## Mechanism
Per Lngram layer, position t [P §2, C]:
```
z = W_q · RMSNorm(h_t)                # W_q: d x d, no bias; code reuses the block's input norm
b = (z > 0)                           # d bits
a[t,r] = sum_j b[4r+j] << j           # R = d/M routes, M=4, K=16
for n in {2,3}:                       # table E(n): (R*K^n) x d_m, zero-init
  g = r*K^n + sum_{i<n} a[t-n+1+i, r] * K^i
  e(n) = concat_r E(n)[g]             # R*d_m dims; zero if t < n
  k = W_K e(n)+b_K ; v = W_V e(n)+b_V       # (R*d_m) -> d
  alpha(n) = sigmoid( RMSNorm(h_t)·RMSNorm(k) / sqrt(d) )
v_t = sum_n alpha(n) v(n)
Y = V + SiLU(DWCausalConv1D(RMSNorm(V)))    # kernel 4, dilation max(n)=3, zero-init
h += Y                                # then attention + MLP as usual
```
Main config [P Table 14]: d=1024, 24 layers, layers {2,12}, M=4, d_m=144, n={2,3}. So R=256; rows 65,536 (2-gram) + 1,048,576 (3-gram); ≈160M table params per layer [E].
One-bit surrogate (recommended by README; what `Lngram.py` single-table implements) [P A.3, C]: for each bit j of each of the n positions in each n-gram, fetch the two rows with that bit forced to 0/1 and set
`dL/dz_j = λ·τ·p_j(1-p_j)·<g, E_j(1) − E_j(0)>`, p=σ(τz), g = dL/d(row), τ=λ=1. Exact variant enumerates all 16 symbols [P A.2]. Tables, readout and conv get ordinary gradients.
Multi-table variant (S independent W_q and tables, softmax fusion): README says it is prone to convergence issues.

## Numbers
All [P]; single seed (42).
- Table 1, 2B total / 0.42B activated MoE, 35B tokens FineWeb-Edu, 5-task avg: MoE 0.5146; +Engram 0.5225; +Lngram 0.5288; Lngram with 23 layers (1.9B/0.4B) 0.5207. Memory replaces 4 of 16 routed experts (~0.5B).
- Table 2: bootstrap significance vs MoE on 4 of 5 tasks; none reported vs Engram.
- Table 3: 2B/140B tokens 0.4718 -> 0.4756 (gain shrinks to 0.38 pt); 8B/1.6B activated, 35B tokens 0.4847 -> 0.4950.
- Table 9 (30B tokens, 4-task avg): {2,3} 0.4829; {3} 0.4799; {2} 0.4698; {1,2,3} 0.4697.
- Table 10: Lngram share of sparse params 25% -> 0.4829; 35% -> 0.4762; 15% -> 0.4765.
- Table 11 (H200): decode 21.798 -> 23.256 ms/token (+6.7%).
- §4.1.2: CKA effective depth +2 to +3 layers mid-stack. §4.4: peak gate values only ~0.085.
- Appendix B (Qwen3-1.7B, window 2048 + retrieval keyed only on 4-bit codes): long-context avg 57.14 vs 59.21 global / 29.41 window-only.

## Smallest scale tested, and whether the result held there
From scratch: 2B total / 0.42B activated, d=1024. Smallest overall: Qwen2.5-0.5B VLM add-on (SeedBench 60.5 -> 61.2, not parameter-matched). Nothing below d=1024, no dense single-stream or quantised backbone, no loss curves. Not shown at our scale.

## Training recipe and gotchas
- Table: Adam, lr 5x base (code default 3x), wd 0, zero-init; backbone AdamW 3e-4, wd 0.01; conv zero-init [P Table 14, C].
- STE fails; use the one-bit surrogate. Cost: 2·M·n extra row gathers per route per order (16 and 24) [E].
- Longer training shrinks the gain (Table 3).
- ≈110 tokens per table parameter (35B / 0.32B) [E]; Engram note suggests ≥50.

## Mapping onto the GBA design
All [E].
- Faithful copy is unaffordable: d=256, R=64, d_m=16 gives e=1,024 dims; W_K and W_V for two orders = 1.05M ternary weights ≈ 1.7M cycles, the whole token budget.
- Fold instead: W_V is linear over a concatenation, so `W_V e = sum_r (W_V^r · row_r)`. Precompute per-route d-dim 8-bit rows into ROM; run time = R row adds per order. Tie key = value for the gate.
- Suggested size: R=16 routes (64 bits). W_q 64x256 ternary = 16K weights ≈ 26K cycles; 32 row fetches x 256 byte-adds ≈ 8K adds ≈ 30K cycles; gate ≈ 8K. About 65K cycles per layer, under 2% of budget. ROM: 2-gram 4,096 rows x 256 B = 1 MB; 3-gram 65,536 x 256 B = 16.8 MB per layer.
- Cheaper still (untested anywhere): drop W_q and use sign bits of 64 fixed residual channels, or reuse the ternary attention-key bits already computed.
- RAM: 8 bytes of codes per past position per layer; two positions needed. Drop the conv (would need 9 past d-vectors).
- Train unfolded (small d_m), fold at export; check 8-bit rounding of folded rows.

## Verdict
Test, second priority behind token-id n-gram tables. It is nearly free at run time once folded and uses our cheapest primitive, but all evidence is one seed at ≥0.42B activated, the gain over Engram is 0.6 pt and untested for significance, and at d=256 the key space (R·K^3 rows) is small. Settling experiment at d=256 with QAT: baseline vs token-id Engram vs Lngram (R=16 and 64, W_q vs raw sign bits) vs both, at equal ROM bytes and equal cycles; report held-out nll. Record whether the surrogate converges with a ternary W_q.

## Could not verify
- Paper says activated parameters rise by only ~40M; my count of the readout projections alone is ≥75M per layer at R·d_m=36,864. Unresolved.
- Paper says W_K/W_V are shared across orders (single-table); the code builds one pair per order.
- Which surrogate (exact or one-bit) the Table 1 run used is not stated.
- Figures 2-4 not read off; text claims only.
- No independent replication found.
- Lngram v2 (arXiv 2609.03426) known only from its abstract page [S]: decouples routes, memory dim and width; grouped-query attention readout; "zero-value sink".

## Links worth following
- https://github.com/zyaaa-ux/Lngram (`Lngram.py`, `Approximate_gradient_backpropagation.py`)
- https://arxiv.org/abs/2609.03426 (Lngram v2: capacity decoupled from width, directly relevant at d=256)
- https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8 (ROSA demo, cited as the origin of low-bit routing codes)
