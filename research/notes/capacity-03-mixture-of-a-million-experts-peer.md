# Mixture of A Million Experts (PEER)  (https://arxiv.org/abs/2407.04153, v1 4 Jul 2024, Xu Owen He, Google DeepMind)

Tags: [P] paper text; [F] eyeballed from its figures (±0.05 ppl); [L] lucidrains/PEER-pytorch source; [C] Csordás 2023 text; [K] Krajewski 2024 text; [N] sibling note capacity-01; [E] my estimate.

## What it is
PEER replaces one transformer FFW with a pool of N≈10^6 single-neuron experts. A product-key router (as in PKM) picks k experts per head; the output is their score-weighted sum. Unlike PKM, each retrieved item is a function σ(uᵀx)·v of the input, not a constant vector. Claim: at equal training FLOPs it beats dense FFW, 128-expert MoE and PKM on C4 perplexity. No code, no appendix.

## Mechanism
Per token, x ∈ R^d [P Eq. 1-8, Alg. 1]:
1. Per head j=1..h: q = BatchNorm(W_q^j x), split into q1, q2.
2. Sub-keys C, C' ∈ R^(√N × d_k/2), shared by all heads. s1 = C·q1, s2 = C'·q2; top-k of each; form k² sums s1[a]+s2[b]; keep top-k; expert id = a·√N + b.
3. g = softmax over that head's k scores.
4. out = Σ_heads Σ_i g_i · σ(u_iᵀx) · v_i, with u_i, v_i ∈ R^d rows of two N×d tables; σ = ReLU or GELU; no bias.

Routing costs O((√N + k²)·d_k). With k=1 the layer is an h-neuron MLP assembled from a shared neuron bank (Eq. 8). Defaults: N=1024², h=8, k=16 (128 active neurons), query BN on.
Third-party code [L] differs: per-head keys, no BN, half-key dim d/2, keys ~N(0, 0.02), default ReLU scores with one expert per head.

## Numbers
Table 1 [P], compute-optimal models, perplexity (Curation / Lambada / Pile / Wikitext / C4):
- 6e18 FLOPs: Dense 23.26/21.95/24.55/29.14/23.84; MoE-128 20.98/19.09/23.26/26.10/21.41; PKM 21.80/19.39/20.49/27.09/21.92; PEER 20.68/17.65/19.01/25.48/20.63.
- 2e19 FLOPs: Dense 17.70/12.28/18.19/21.21/18.31; MoE 16.88/12.97/17.41/20.28/17.12; PKM 17.03/11.18/16.34/20.26/17.36; PEER 16.34/10.33/14.99/19.09/16.45.

Table 2 [P], C4 ppl without/with query BN, hk=128: N=16k 23.4658/23.4690; 65k 22.6121/22.5543; 262k 21.5409/21.4696; 1M 20.7307/20.6449. Usage ≥99.79% everywhere; unevenness (KL to uniform) with BN 0.30/0.44/0.66/1.06.

Fig. 3a [F]: the same backbone without PEER (≈135M) scores ≈24.2, so 16k experts buy ≈0.03 nats and each 4× in N ≈0.04 nats more. Parameter counts (165M/260M/670M/2.2B) imply d≈1024 [E].

Fig. 3b [F], N=1M, best per hk: 32 → 21.50 (h=2; 22.30 at h=8); 64 → 21.03 (h=4); 128 → 20.65 (h=8); 256 → 20.42 (h=16); 512 → 20.26 (h=16). k≈16 per head is best; hk=32 keeps about three quarters of the gain.

Independent check [N]: Meta, 134M base, PEER (1.04B total): NQ 2.46 / TQA 16.34 vs dense 0.91/7.7; Memory+ 3.16/18.77.

## Smallest scale tested, and whether the result held there
Smallest PEER model ≈165M total (≈135M backbone, 16k experts, hk=128): the gain held but was small. Nothing smaller, quantised, or with hk<32.
Nearest small-scale evidence is unfavourable [C]: at 47M (WikiText-103, parameter-equal, every FFW replaced) PKM scores 13.96 (softmax) / 12.77 (ReLU) vs dense 11.81; expert size below 128 neurons "generally degrades performance", and active channels must stay above ≈256-512. Top-4 of 128-neuron experts beats top-1 of 512 at equal compute (11.59 vs 11.90, Table 4).
The granularity law PEER extrapolates was fitted only for G=1-16 at 129M-3.7B; in one wall-clock test routing cost outweighed the gain at G=16 [K].

## Training recipe and gotchas
- Given [P]: batch 128, sequence 2048, one PEER layer in the middle block.
- BN is nearly irrelevant below 65k experts; drop it.
- Softmax needs a division; [C] found ReLU scores better for PKM.
- Each expert gets gradient only when retrieved (hk/N of tokens); keep N ≤ 2^16 [E].
- Ternary QAT of u, v is untested. Single-neuron experts allow a per-expert scale for one multiply each [E].

## Mapping onto the GBA design
Lite config, d=256, h=2, k=8, half-key dim 32, N=2^14 (n=128) [E]:
- Query: 2×64×256 = 32.8K ternary weights ≈ 52K cycles.
- Key scoring: 2×2×128×32 = 16.4K weights ≈ 26K cycles (52K at n=256).
- Top-k scans and 64 candidate sums: ≈10K cycles.
- Down-projection: 16 rows × 256 = 4.1K weights ≈ 7K cycles; rows join the query matvec and reuse its partial-sum tables.
- Up-projection: 4.1K scaled-trit adds (not table-kernel compatible) at 4-6 cycles ≈ 16-25K.
- Total ≈115-140K cycles (3-4% of budget), two thirds of it routing.
- ROM: 512 trits per expert ≈ 103-128 B; 2^14 → 1.7-2.1 MB; 2^16 → 6.7-8.4 MB; 2^18 → 27-34 MB. RAM under 1 KB.
- Cheaper variant (mine, untested): read the hk expert ids from a ROM table keyed on token id or bigram hash (16K×16×2 B = 512 KB), leaving ≈30K cycles. The paper cites work finding fixed routers worse than learned ones.

## Verdict
Test, not adopt. Cheap, but every result is at ≥135M, d≈1024, hk≥32, float, and the affordable N (≤65k) bought only 0.03-0.07 nats there. The firmer lesson is granularity: [C] (top-4 beats top-1) and Table 1 (fine beats coarse MoE) both argue against the 1-of-64 top-1 MLP. Settling experiment at d=256, ternary QAT, matched GBA cycles: (a) 1-of-64 MLP; (b) top-4 of 256 quarter-size experts; (c) PEER, N ∈ {2^14, 2^16}, (h,k) ∈ {(2,8),(2,16),(4,8)}; (d) token-routed PEER; (e) hashed n-gram table with equal ROM.

## Could not verify
- d_k, optimizer, lr, init, σ, backbone shapes: absent from the paper. d≈1024 is my arithmetic from eyeballed figure positions.
- No small-scale PEER replication found (one search; GitHub issues unreadable).
- Meta's numbers come from the sibling note; MoLE (below) only from a search summary.

## Links worth following
- https://arxiv.org/abs/2310.10837 (Csordás: σ-MoE at 41-47M)
- https://arxiv.org/abs/2402.07871 (fine-grained MoE scaling laws)
- https://arxiv.org/abs/2503.15798 (Mixture of Lookup Experts: token-keyed experts baked into LUTs)
- https://github.com/lucidrains/PEER-pytorch
