# Towards an empirical understanding of MoE design choices (https://arxiv.org/abs/2402.13089, Feb 2024, Dongyang Fan, Bettina Messmer, Martin Jaggi, EPFL)

Read in primary form (arXiv HTML v1, via a fetch-and-extract tool; numbers below are from its Tables 1-3, Figure 1 and Appendix A).

## What it is
A short workshop-style ablation that trains MoE versions of GPT-2 small (124M, nanoGPT code base) from scratch and compares routing unit (token vs whole sequence), top-K (1/2/3), number of experts (2/4/6), and router type (learned, frozen at init, re-randomised every step, hard-coded by language). Every block's FFN is replaced by N experts. Main result: the routing unit matters most; token-level routing wins, and token-level top-1 beats a dense model with the same active parameters by a wide margin. Single seed, short training, N never above 6.

## Mechanism
- Per block: logits `h(x) = W_g x` (linear router; a 2-layer MLP router made no difference, App. A.2, plot only), `p = softmax(h)`, `T = top-K(p)`, output `y = sum_{i in T} p_i(x) * E_i(x)` (Eq. 1). Top-1 follows Switch Transformer: the chosen expert's output is scaled by its gate probability (this is how the router gets gradient). No capacity factor.
- Each expert appears to be a full GPT-2 FFN (768 -> 3072 -> 768); not stated, but the parameter counts match (4 experts: 295M total; 12 x 4.7M per FFN).
- Variants: token-level layer-wise (each token routed independently in each layer), sequence-level layer-wise, sequence-level global (one router in block 1, reused).
- Router strategies: learned; frozen (router weights fixed at initialisation, rest of the network trains); random (router re-initialised every iteration).
- Optional load-balancing loss, weight 0.01 (equation not given).

## Numbers
Validation perplexity, GPT-2-small scale, ~6B training tokens.

Table 1 (N=4), MultiWiki / OpenWebText:
- Token top-1 (295M total, 124M active): 10.188 / 23.548
- Token top-2 (295M, 182M active): 9.907 / 22.632
- Sequence layer-wise top-1: 11.573 / 26.742; top-2: 10.667 / 24.980
- Sequence global top-1: 13.632 / 30.292; top-2: 11.605 / 25.689
- Dense 1x FFN (124M): 11.815 / 26.195
- Dense 2x FFN (182M): 10.831 / 24.387
- Dense 4x FFN (295M): 10.071 / 22.700

So token top-1 is 10% (OWT) to 14% (MultiWiki) lower perplexity than the active-matched dense model, and within 1-4% of the dense model that computes all 4x of the FFN.

Table 2 (MultiWiki, (N,K) -> sequence / token): (2,2) 10.648 / 10.761; (4,2) 10.667 / 9.907; (6,2) 10.674 / 9.467; (4,1) 11.573 / 10.188; (4,3) 10.107 / 9.711. Token-level gains from more experts (N: 2 -> 4 -> 6), sequence-level only from more active experts.

Table 3 (layer-wise top-2, sequence / token): MultiWiki frozen 10.601 / 9.810, learned 10.666 / 9.907, random 11.685 / 11.752. OWT frozen 24.590 / 22.687, learned 24.980 / 22.632, random 26.429 / 26.879. A router frozen at init is as good as a learned one; a router that changes every step is no better than dense 1x.

Figure 1: with vs without load-balancing loss, 10.667 vs 10.674 (sequence-level top-2, MultiWiki).

## Smallest scale tested, and whether the result held there
Only one scale: GPT-2 small, 12 layers, 768 wide, N <= 6. Nothing near our 256-384 width or 64 experts. The N trend is three points (2, 4, 6) at K=2 only.

## Training recipe and gotchas
- lr 9.6e-4 decaying to 9.6e-5, weight decay 0.5, dropout 0.2, 1,048,576 tokens per iteration (batch 8 x 1024 x 128 accumulation), ~6B tokens, one A100-40GB. Optimizer and warmup not stated. Results averaged over the last 100 iterations.
- The dense OWT baseline (26.2) is well above what fully trained nanoGPT reaches (my recollection, about 18-19; not checked here), so this is an undertrained regime where extra total parameters may help more than they would at convergence.
- The collapse observation ("early and late layers", "experts can recover") is documented for SEQUENCE-level routing (Fig. 1, App. A.4). No token-level collapse analysis and no token-level with/without-aux-loss number is given. The search summary that selected this paper overstated this.
- Token-level routing showed no topic specialisation and near-even load.
- Top-1 output is multiplied by p_i; an integer inference path must either drop this (train without it, e.g. straight-through) or fold it in.

## Mapping onto the GBA design
- Supports the core bet: 1-of-N hard-routed MLP at fixed active compute beats dense. ROM cost is N x expert size; at 64 experts x ~200K ternary weights x 6 layers = ~77M weights (~19 MB at 4 weights/byte), inside the budget (my arithmetic).
- Router cost: a 64 x 320 ternary router is ~20K weights = ~33K cycles per layer, ~200K per token over 6 layers (4-7% of budget). The frozen-router result suggests a cheaper option: a fixed random projection, e.g. 6 sign bits of 6 fixed ternary rows (~1.9K weights per layer) indexing 64 experts directly, no argmax. The paper's frozen router is a random linear layer plus softmax/argmax, not a sign hash, so this is an extrapolation to test.
- Routing must depend on the hidden state consistently (random-per-step fails), but need not be learned. Keying the route on a token-id or n-gram hash (nearly free on GBA) is not tested here; the authors only remark that their frozen routing differs from hash routing because it depends on embeddings.
- The p_i scale needs one multiply per output element (d multiplies, ~2.4K cycles at 7.5 cycles) or can be removed in training.

## Verdict
Adopt as supporting evidence for hard top-1 token routing; test for everything else. It does not check the 16 -> 64 expert step (max N = 6, K = 2). Settling experiment: at our width, train dense vs 1-of-{4,16,64} with (a) learned router, (b) frozen random router, (c) frozen sign-hash router, all active-matched, with per-layer expert-usage histograms and with/without a balance loss, run to convergence rather than 1 epoch.

## Could not verify
- Load-balancing loss formula, optimizer, warmup, router hidden size, expert dimensions: not in the paper.
- Appendix says "66K iterations" in the extracted text, which conflicts with ~6B tokens at ~1M tokens/iteration (implies ~6K); likely a rendering artifact, unresolved.
- No code release for the MoE variant found in the paper.
- Whether the paper was later published or revised (only v1 read).
- I read the paper through an extraction tool, not the raw HTML; table values were returned consistently across two passes.

## Links worth following
- Switch Transformer (top-1 gating, capacity factor): https://arxiv.org/abs/2101.03961
- Hash Layers for Large Sparse Models (hash routing, already in notes as f-tables-11)
- nanoGPT base code: https://github.com/karpathy/nanoGPT
