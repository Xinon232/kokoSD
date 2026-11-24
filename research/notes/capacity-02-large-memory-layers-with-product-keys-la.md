# Large Memory Layers with Product Keys (https://arxiv.org/abs/1907.05242, NeurIPS 2019; Lample, Sablayrolles, Ranzato, Denoyer, Jégou, Facebook AI Research)

Tags: [P] paper PDF (v2) or authors' code; [F] read off a figure by eye (±0.1 ppl); [C] my arithmetic; [S] secondary.

## What it is
A key-value layer (PKM) that replaces the FFN in 1-3 transformer layers. A query from the hidden state selects k of |K| value rows; the output is their weighted sum. Keys are the Cartesian product of two sub-key codebooks, so exact top-k over |C|² keys costs about 2|C| half-size dot products. Compute depends on heads × k, not memory size [P]. It is the content-addressed complement of our token-id/hash tables.

## Mechanism
Each of H heads has its own query projection and sub-keys; all share one value table [P, code]:
```
q  = BatchNorm(W_h x + b)              # k_dim per head; q = [q1|q2]
s1 = C1_h q1 ; s2 = C2_h q2            # |C| inner products each
I1,I2 = topk(s1,k), topk(s2,k)
cand(i,j) = s1[i]+s2[j], i∈I1, j∈I2   # k² scalar adds
I  = topk(cand,k); slot = i*|C|+j
w  = softmax(cand[I])                  # per head
out = Σ_heads Σ_I w·V[slot]            # then x ← x + out
```
- Paper: d_q = 512, H = 4, k = 32, |C| = 512 (262k slots); L=16, N=2 uses layers 6 and 12 [P].
- Authors' notebook: k_dim = 128 per head (sub-keys 64-d), one Linear(d, H·k_dim) + BatchNorm1d, v_dim = model dim [P, code]. So "512" is probably 4 × 128.
- Init: keys uniform ±1/√(k_dim/2), learned; values N(0, v_dim^-0.5) [P, code].
- Unreported code options: fixed binary ±1 keys (index = query sign bits), no query network, smaller v_dim, values shared across memories [P, code].

## Numbers
CC-News, 28B words, 60k BPE, test perplexity [P].
- Table 1, dim 1024: 12L 17.7 → 15.6 (1 PKM) → 14.8 (2) → 14.5 (3); 16L 16.7 → 14.9 → 14.1; 24L 16.0 → 14.6. Dim 1600: 12L 15.0 → 13.7; 16L 14.4 → 13.2; 24L 14.0. Figure 4: 12L+PKM is "almost twice faster" than 24L.
- 6-layer, 8-head ablation model (width not stated), baseline 22.98; PKM at layer 5, H=4, k=32.
- Table 2, memory size, with BN (ppl / usage%): 16k 23.0/100; 65k 21.9/100; 147k 20.7/99.6; 262k 19.8/97.9; 590k 18.7/90.3; 1M 18.0/80.3. Without BN: 22.8, 21.7, 20.9, 20.5, 20.0, 19.8, usage falling to 25.8% at 1M. In nats vs baseline [C]: 65k +0.05, 262k +0.15, 1M +0.25. **16k slots gave nothing.**
- Table 3, position 1..6: 21.5, 20.7, 20.4, 20.1, 19.8, 20.3.
- Table 4, flat learned keys: 23.2/22.6/22.1 ppl at 16k/65k/147k with usage 19.6/13.6/10.1%; speed 35.0k → 1.2k w/s at 1M, product keys ~36k throughout.
- Figure 6, 262k slots [F]: ppl at k=8/16/32/64: 1 head 23.3/22.7/21.8/21.0; 2 heads 22.6/21.6/20.5/19.95; 4 heads 21.7/20.5/19.8/19.6; 8 heads 20.7/19.85/19.5/19.4. Usage at k=8 for 1/2/4/8 heads: 4/12/33/68%. Text [P]: h×k = 64 gives ~70% usage, ppl ~20.5.

## Smallest scale tested, and whether the result held there
6 layers, 8 heads, 28B words: the gain held (−14% ppl at 262k) but only with ≥65k slots and h×k ≥ ~32. One head with k=8 was worse than no memory [F]. Nothing near 1M active weights. Csordás et al. 2023 (arXiv 2310.10837) [P]: 47M Transformer-XL, WikiText-103, PKM in every layer at parameter parity (62² slots, 4 heads, no BN): 13.96 (softmax) / 12.77 (ReLU) vs dense 11.81.

## Training recipe and gotchas
- Adam 2.5e-4 (β 0.9/0.98, Vaswani schedule); values use Adam at constant 1e-3 [P, notebook].
- Query BatchNorm matters from 147k slots up; padding must not enter its statistics [code warning].
- Place it mid-to-late; never first or last.

## Mapping onto the GBA design
Sub-key scoring is a ternary matvec (|C| rows × k_dim/2 inputs), so our kernel runs it unchanged. Estimates [C] for d = 256, |C| = 256, k_dim = 64:
- Query: H·64·256 ternary weights; keys: H·2·256·32; values: H·k·256 int8 MACs; selection: H·(512 + k²) compares.
- H=4, k=16: 105K + 105K + 123K + ~30K ≈ 365K cycles (7-12% of budget). H=2, k=8: ≈ 150K cycles, the weak corner of Figure 6.
- ROM: 65k × 256 values = 16.8M (3.4 MB ternary, 16.8 MB int8); 262k × 256 = 67M (13-17 MB at 1.6-2 bits). Keys are negligible.
- RAM: scores, k² candidates, kernel tables: about 4 KB.
- Precompute: BN folded into query scale/bias; exp LUT, or ReLU weights to drop the divide.
- Cheaper untested variants: sign-bit keys (no scoring), identity query, v_dim 64 plus a ternary up-projection.
- Table 4 warns about our flat 1-of-64 router: flat keys used 10-20% of slots.

## Verdict
Test. Large gain from lookup-style capacity (+0.15 nats at 6 layers), but it needs ≥65k slots and ≥32-64 reads per token, costs 0.15-0.4M cycles here, and is unproven with ternary keys, low-bit values, or a 1M-weight core. Settle it on the float d=256 model: no memory vs one PKM in the second-to-last layer, |C| ∈ {256, 512}, (H,k) ∈ {(2,8),(4,8),(4,16),(8,8)}, softmax vs ReLU, logging usage and KL; then QAT keys and values. Adopt if it beats an equal-cycle wider MLP by ≥0.05 nats.

## Could not verify
Width of the 6-layer model; memory parameter counts (512²×1024 = 268M is [C]); steps, batch size, context length; any quantised or sub-10M result; numbers for binary keys. Memory Layers at Scale was not re-read [S, prelim notes].

## Links worth following
- https://github.com/facebookresearch/XLM (PKM-layer.ipynb, xlm/model/memory/memory.py)
- https://arxiv.org/abs/2310.10837 (PKM/MoE/top-k at 47M; ReLU fix)
- https://arxiv.org/abs/2412.09764 (Memory Layers at Scale)
