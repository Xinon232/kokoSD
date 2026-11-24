# Memory Layers at Scale  (https://arxiv.org/abs/2412.09764, v2 Dec 2024, ICML 2025; Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh; Meta FAIR)

Tags: [P] arXiv v2 text; [I] ICML 2025 text; [C] released code (commit 7f85280); [L] Lample 2019 text; [S] secondary; [E] my estimate.

## What it is
A trainable product-key key-value memory replaces the FFN in a few transformer layers: each token's hidden state retrieves the top-k of N learned value rows and sums them. Capacity grows with N while FLOPs stay almost flat. Meta scale this to 128B memory parameters and report large factual-QA gains over dense, MoE and PEER models at equal compute. "Memory+" shares one value pool across 3 layers and adds a SiLU-gated output projection.

## Mechanism
Per memory layer and token, with x = RMSNorm(h) and model dim d [P, C]:
1. q = Linear(d → H·d_k)(x), bias, no BatchNorm in the code. Split into H queries, each into halves q1, q2.
2. Per head, half-keys K1, K2 ∈ R^(n × d_k/2), n = √N. s1 = K1·q1, s2 = K2·q2; top-k of each.
3. Form the k² sums s1[i]+s2[j]; keep the top-k; slot id = i·n + j.
4. w = softmax over that head's k raw scores (no temperature).
5. y = Σ w·V[slot] over all H·k pairs; V ∈ R^(N×d) is shared by all heads and memory layers.
6. Memory: out = y. Memory+: out = W2(y ⊙ silu(W1·x)), W1, W2 ∈ R^(d×d). h ← h+out.

Hyperparameters: H=4, k=32 (128 rows read) at every size [I App. A; C]; n=1024 (N=2^20); d_k = d/2 for Memory+, d/4 for base Memory; value dim = d; layers 4, 12, 20 of 24 [C config]; vanilla Memory uses one middle layer.
Init [C]: keys U(±1/√d_k); values, W1, W2 ~ N(0, 1/d); query Xavier-uniform.

## Numbers
Table 1 [P], 1T tokens, NQ accuracy / TriviaQA F1 (total parameters in brackets):
- 134M base: dense 0.91/7.7; MoE-16 (984M) 2.49/13.08; PEER (1.037B) 2.46/16.34; Memory (937M) 2.1/16.31; Memory+ (937M) 3.16/18.77. Dense 373M scores 2.58/17.68.
- 373M base: Memory+ 5.76/28.10. Dense 720M scores 3.77/24.85.
- 1.3B base: dense 7.76/32.64; MoE-4 8.14/31.46; Memory+ 13.68/42.89; with 64M slots 20.78/62.14 (Llama2-7B 25.10/64.00).
- Commonsense at 134M: PIQA 62.13→65.94 (MoE 65.78); OBQA 16.40→17.60 (MoE 18.80).

Table 2 [P], 8B base, 16M slots: NQ 25.24→27.06, TQA 63.62→68.15, MMLU 59.68→63.04, MBPP 44.20→42.20 (worse).

Tables 3-4 [P], 373M model, d=1024, train nll / NQ nll / TQA nll:
- Memory layers {12}: 2.11/12.13/8.34; {4,12,20}: 2.06/11.32/7.20; six layers: 2.11/11.79/7.73 (more than ~3 hurts).
- +swilu alone: 2.11/12.05/8.09. Gating, random keys, softmax sink: no consistent gain.
- Fixed memory parameters: v_dim 64 × 16M slots 2.15/12.86/8.75; 1024 × 1M 2.11/12.13/8.34.
- key_dim 256/512/1024/2048: train nll 2.11/2.12/2.11/2.09.
- H×k [I App. A, no numbers]: equal products behave alike; above 128 little gain; 64 gives "non-negligible accuracy degradation".

No dense-baseline training nll is reported, so the general-LM gain is unquantified.
Lample [L Table 2], 6-layer model, CC-News perplexity: no memory 23.0; 16k slots 22.8; 65k 21.7; 262k 19.8; 1M 18.0.

## Smallest scale tested, and whether the result held there
134M base (d=768, 12 layers), memory six times the base size. NQ/TQA gains held, but "training can become unstable, especially for small base models" [P]. Nothing near 1-2M active weights, with H×k below 64, or quantised.

## Training recipe and gotchas
- AdamW β=(0.9, 0.95), weight decay 0.1, clip 1.0; lr 3e-4 (1e-4 at 8B) [I Table 5; C]. Values get lr 1e-3 and their own gradient clipping [C].
- Slot usage starts heavily skewed and evens out over ~13k steps [I App. B]. Lample needed query BatchNorm: at 1M slots usage 25.8%→80.3% [L]. Meta use qk-normalization "when needed".
- "Up to 5x faster decoding than MoE" [I Sec. 3.3] is a GPU-batch argument; irrelevant to batch-1 hard routing.

## Mapping onto the GBA design
Per memory layer at d=256 [E]:
- Meta-proportioned (H=4, k=32, n=1024, d_k=128): query 131K weights, keys 524K weights, 32.8K multiply-accumulates: ~1.3M cycles. Unaffordable.
- Lite (H=2, k=8, n=256, d_k=64, N=65,536): query 32.8K ternary weights (~52K cycles); key scoring 32.8K ternary weights with the existing partial-sum kernel (~52K); top-k scans ~15K; softmax from an exp LUT; gather 16 rows × 256 at 5-7.5 cycles (20-31K). Total ~150K cycles, 3-5% of the budget.
- Memory+ W1, W2 add 131K weights (~210K cycles); swilu alone bought little, so skip them.
- ROM at 5 trits/byte: 65,536 × 256 ≈ 3.4 MB; 2^18 slots ≈ 13.4 MB. Table 4 says not to narrow values to buy slots.
- RAM: ~2.5 KB (partial-sum tables, accumulator).
- Precompute: fold BatchNorm into the query matrix. A second layer sharing the pool costs compute, no ROM.

## Verdict
Test. The effect is proven at 134M and above and concentrated on fact recall; the affordable lite configuration lies outside every tested range (H×k=16, ternary, tiny base). Settling experiment: at d=256, replace the middle expert-MLP with PKM; sweep N ∈ {2^14, 2^16, 2^18} and H×k ∈ {8, 16, 32, 64, 128}; compare validation nll and a fact-recall probe against the 1-of-64 expert MLP and a hashed n-gram table at matched cycles; repeat with ternary QAT values.

## Could not verify
- qk-normalization: absent from the released memory.py.
- H×k ablation numbers; Lample's ablation width.
- Whether Meta's runs applied weight decay to values (the code does).
- Abstract only [S]: Mixture of Chapters, FwPKM. Search snippet only: Sparse Memory Finetuning (Lin et al. 2025).
- Any small-model replication.

## Links worth following
- https://arxiv.org/abs/1907.05242
- https://github.com/facebookresearch/memory
- https://arxiv.org/abs/2407.04153 (PEER)
- https://arxiv.org/abs/2601.00671 (FwPKM)
