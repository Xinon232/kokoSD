# Engram: Conditional Memory via Scalable Lookup  (https://arxiv.org/abs/2601.07372, v1 Jan 2026, v2 Jul 2026; Cheng, Tian, Zeng, Dai, ... Liang; Peking University + DeepSeek-AI)

Tags: [P] paper v2 full text; [F] read off a paper figure (±0.0005); [C] official demo `engram_demo_v1.py`; [G] Memory Grafting (2605.20948) text; [N] Engram-Nine (2601.16531) text; [E] my estimate.

## What it is
A lookup module: hash the suffix 2- and 3-gram of token ids into large embedding tables, gate the retrieved vector with the hidden state, add it to the residual stream at one or two early layers. Slots add stored capacity at constant per-token compute. DeepSeek show it beats an iso-parameter, iso-FLOPs MoE at 27B and that loss falls log-linearly with slot count. It is the best-evidenced recipe for our "hashed n-gram embeddings from ROM" item, but nothing was tested below 0.56B activated parameters.

## Mechanism
Per Engram layer, position t [P §2; C]:
1. Canonical ids x' = P[x]: a precomputed map merging tokens equal after NFKC, accent stripping, lowercasing and whitespace collapse (129,280 vocab shrinks 23.43%).
2. For n in {2,3}: mix_n = XOR over i<n of (x'_{t-i} * m_i); m_i random odd 64-bit, per layer [C].
3. For heads k=1..K (K=8): z = mix_n mod p_{n,k} (distinct primes just above the slot count); row = E_{n,k}[z]. Heads differ only in modulus [C].
4. e = concat of the 2K rows; d_mem = 1280, so 80 per row [E from Table 5].
5. v = W_V e; key = W_K e; g = RMSNorm(h)·RMSNorm(key)/sqrt(d); alpha = sigmoid(g) [P Eq. 4]. The code first applies g <- sign(g)·sqrt|g| [C].
6. u = alpha·v; Y = u + SiLU(DepthwiseCausalConv(RMSNorm(u))), kernel 4, dilation 3, conv zero-initialised.
7. h <- h + Y, before that block's attention.

Tables are per layer. Engram-27B: layers 2 and 15 of 30, d=2560, 2,262,400 slots per head (5.7B parameters); Engram-40B: 7,239,680 (18.5B) [P Table 5]. The mHC backbone has 4 residual branches: one W_V, four W_K.

## Numbers
- Ablation backbone [P §6.2]: 12-layer MoE, 3B total, 0.56B activated, 100B tokens. Val loss 1.808 -> 1.768 with a 1.6B table at layers 2+6. Single module: layer 2 = 1.770 (best); layer 1 ≈1.7735, layer 6 ≈1.777, layers 11-12 ≈1.782 [F].
- Ablations from 1.768 [F, Fig. 5]: no conv ≈1.7685; +4-grams ≈1.7695; no gate ≈1.7725; no tokenizer compression ≈1.7735; no per-branch gating ≈1.7745. Every variant keeps ≥84% of the gain.
- Slot scaling, same backbone [P §3.2; F Fig. 3]: 2.58e5 -> 1.0e7 slots (up to ≈13B parameters, 23× the activated count): ≈1.788 -> ≈1.743, about 0.028 nats per 10× slots. OverEncoding (n-gram rows averaged into the input embedding) is 0.010-0.015 worse at equal slots.
- Iso-parameter [P §3.1], 993M activated: 1.7248 (pure MoE) -> 1.7109 with ~20% of sparse parameters in Engram.
- Table 1 [P], 3.8B activated, 262B tokens: val loss MoE-27B 1.634, Engram-27B 1.622, Engram-40B 1.610; MMLU 57.4, 60.4, 60.6.
- Zeroing Engram output at inference [P Fig. 6]: TriviaQA keeps 29%, C3 93%. The table holds facts.

## Smallest scale tested, and whether the result held there
Paper: 0.56B activated; held. Independent work is mixed:
- [G] iso-parameter: 2.8B/0.55B activated, 100B tokens: 9-benchmark average MoE 51.95, Engram 52.43. At 0.92B/0.29B, 50B tokens: MoE 45.62, Engram 45.03 (worse; authors blame the small table).
- [N] 12×768 GPT-2 shape, 82M tokens, 128M-parameter table: a collision-free perfect-hash tier scored 4.4799 vs 4.4809 (noise); the gate gave its highest alpha to the highest-loss positions. No no-Engram baseline.

Nothing near 1M active weights, single-stream, or quantised.

## Training recipe and gotchas
- Table: Adam, 5× the base lr (4e-4), no weight decay; backbone Muon, weight decay 0.1 [P Table 5].
- Tokens per table parameter [E]: 62 (1.6B table) and 46 (Engram-27B) worked; at 14 the authors call Engram-40B under-trained; [N] had 0.6. A 60M-weight table wants roughly 3B training tokens.
- The iso-parameter results (§3.1, [G]) are the wrong test for us: ROM is free, so §3.2 applies.

## Mapping onto the GBA design
All [E].
- Hash: 2-3 MULs per order. Mod-prime needs division; use power-of-two tables and one multiply-shift per head (untested deviation). ~100 cycles. Skip perfect hashing [N].
- Faithful small variant (K=4, 16-dim 8-bit rows, d_mem 128, d=256): W_V and W_K are 32.8K ternary weights each; with norms, the gate dot and scaling, ≈115K cycles per layer, 2-4% of budget. ROM: 128 bytes × slots per head.
- ROM-for-compute variant: W_V is linear, so fold it into the table at export (exact). Each row stores its d-wide contribution and v is the sum of 2K rows. K=2: 1,024 adds, ≈5K cycles. 16 MB buys 16K slots per head at 8 bits, ~80K ternary.
- Gate: tie key to value (≈2K cycles, untested) or drop it (≈0.004 of 0.040 nats at 3B scale). Skip the conv (≈0.0003 nats; needs 9 past vectors, 2.3 KB RAM).
- RAM: only the last three canonical ids, already in the ring buffer. Canonical map: 32 KB ROM, or fold case and spacing in the tokenizer.
- Placement: before attention in block 2; optionally a second table mid-depth.

## Verdict
Adopt the retrieval path (multi-head hashed 2/3-gram tables, canonical ids, block 2, Adam 5× lr); test the gate. It costs under 4% of the budget, and the paper stays log-linear to a 23× table-to-active ratio (ours: 30-80×). Settling experiment at d=256 with QAT: no table, token-id per-layer embedding, input-summed n-gram rows, and Engram ungated, tied-key and fully gated; sweep slots 2^14-2^20 and K in {1,2,4,8} at fixed ROM bytes; report validation nll, a fact-recall probe and typing top-3 hit rate; repeat the winner with ternary rows.

## Could not verify
- Absent from the paper: slots versus row width, a K ablation, any dense or single-stream training run, table widths for the 3B ablation and §3.2 sweep.
- Figure values are plot read-offs. v1-v2 differences not checked.
- Not read: OverEncoding (2501.16975); the t-hertel/Engram negative-result replication (README only, so not used here).

## Links worth following
- https://github.com/deepseek-ai/Engram
- https://arxiv.org/abs/2605.20948 (Memory Grafting: rows precomputed from a teacher)
- https://arxiv.org/abs/2601.16531 (Engram-Nine)
- https://arxiv.org/abs/2501.16975 (OverEncoding)
