# Conditional Memory via Scalable Lookup (Engram)  (https://arxiv.org/abs/2601.07372, v2 12 Jul 2026; Cheng, Tian, Zeng, Dai, ... Liang; Peking University + DeepSeek-AI)

Tags: [P] paper v2 text (HTML, read in full); [F] my read-off from PDF Figs. 3/5 (±0.0005); [C] official `engram_demo_v1.py`, via a fetch-tool paraphrase; [R] littlemex replication (config.json, eval.json, README via fetch); [E] my estimate. Earlier note: `capacity-04-engram-...md`; read-offs agree.

## What it is
A lookup module added to 1-2 early transformer blocks: hash the suffix 2-gram and 3-gram of canonicalised token ids into big embedding tables, project, gate by the hidden state, add to the residual stream. Slots add capacity at constant per-token compute. DeepSeek report gains over an iso-parameter, iso-FLOPs MoE at 27B and log-linear loss vs slots.

## Mechanism
Per Engram layer, position t [P §2, C]:
```
x' = P[x]                       # NFKC, accent strip, lowercase, whitespace collapse
for n in {2,3}:
  mix = XOR_{i<n} (x'[t-i] * m[i])   # m: random odd multipliers, seeded per layer
  for k in 1..K (K=8): row[n,k] = E[n,k][ mix mod p[n,k] ]   # distinct primes
e = concat(rows)                # d_mem = 1280 -> 80 per row [E]
v = W_V e ; key = W_K e
alpha = sigmoid( RMSNorm(h)·RMSNorm(key) / sqrt(d) )         # Eq. 4; code adds sign(g)*sqrt|g| first [C]
u = alpha * v
Y = u + SiLU(DepthwiseCausalConv1D(RMSNorm(u)))   # kernel 4, dilation 3, zero-init
h += Y                          # before that block's attention
```
Engram-27B: layers 2 and 15 of 30, d=2560, 2,262,400 slots (x1280 x2 layers = 5.7B); 40B: 7,239,680 (18.5B) [P Table 5]. mHC backbone: 4 residual branches, one W_K each.

## Numbers
- Ablation backbone [P §6.2]: 12-layer MoE, 3B total, 0.56B activated, 100B tokens. Val loss 1.808 -> 1.768 with 1.6B table at layers 2+6.
- Single-module layer sweep [F Fig. 5]: L1 ≈1.7735, L2 1.770 [P], L3 ≈1.775, L6 ≈1.7775, L11-12 ≈1.7825.
- Component ablations from 1.768 [F Fig. 5]: no conv ≈1.7685; +4-gram ≈1.7695; no gate ≈1.7725; no tokenizer compression ≈1.774; no per-branch gating ≈1.7745.
- Slot sweep, same backbone [P §3.2, F Fig. 3 right]: 2.58e5 slots ≈1.788; 6.5e5 ≈1.776; 1.3e6 ≈1.768; 2.6e6 ≈1.760; 1.0e7 ≈1.743 (≈13B table params, 23x activated). Slope ≈0.028 nats per 10x. The smallest table (≈0.33B params [E], 0.6x activated) already gives 0.020 nats. OverEncoding (rows averaged into the input embedding) is ≈0.011-0.014 worse at equal slots.
- Allocation [P §3.1, Fig. 3 left]: at 9.9B total/993M activated, 1.7248 (pure MoE, 99 experts) -> 1.7109 with 20-25% of sparse params in Engram.
- Table 1 [P], 3.8B activated, 262B tokens: val loss MoE-27B 1.634, Engram-27B 1.622, Engram-40B 1.610; MMLU 57.4/60.4/60.6.
- Zeroing Engram [P §6.3]: factual tasks keep 29-44%, reading comprehension 81-93%.

## Smallest scale tested, and whether the result held there
Paper: 0.56B activated; held. Below that, only [R]: single-stream 12x512 GPT (63.8M), WikiText-103, GPT-2 tokenizer (50,257 -> 32,962 canonical ids), one Engram layer at block 1, 8 heads x {2,3}-grams x ~65.5K rows x 64 dims (67.2M table), gate, no conv, table lr x5, 236M tokens, 3 seeds. Val ppl 28.97±0.08 vs 29.26±0.01; test 29.40±0.16 vs 29.36±0.13 (level). Skipping Engram doubles ppl (59.5): the model leans on it without net gain. Not shown to hold at small scale; that run had ~3.5 tokens per table parameter [E].

## Training recipe and gotchas
- Table: Adam, lr 5x base (4e-4), weight decay 0; backbone Muon, wd 0.1; conv zero-init [P Table 5].
- Tokens per table parameter [E]: 62 (ablation), 46 (27B) worked; 14 (40B) is called under-trained by the authors; 3.5 [R] gave nothing. Plan ≥50: a 16M-parameter table needs ~0.8B tokens.
- Val/test disagreement in [R]: evaluate on held-out sources.

## Mapping onto the GBA design
All [E].
- Hash: 2-3 MULs per order in 32 bits (14-bit ids). Mod-prime without divide: reciprocal multiply, or power-of-two tables (untested). <150 cycles.
- Folded variant: W_V is linear, so store W_V-projected rows (d=256, 8-bit) in ROM; v = sum of 2K rows. K=2: 4 rows, 1,024 adds ≈5K cycles. ROM 256 B/slot: 16 MB = 16K slots per table. Gate with tied key=v: dot ≈2K, norms ≈4K, LUT sigmoid/rsqrt, scaling ≈2K. Total ≈15K cycles, <0.5% of budget.
- Unfolded variant: 16-dim rows, K=4, e=128, ternary W_V 128x256 = 33K weights ≈52K cycles (+52K for W_K): 1-3% of budget, but 16 B/slot, so 16 MB = 125K slots per table, 16x more slots per byte.
- RAM: three canonical ids. Canonical map: 32 KB ROM. Drop the conv.
- Our ROM/active ratio (≥40x) exceeds the tested 23x.

## Verdict
Test, high priority. Nearly free for us and the best-evidenced design for ROM tables, but every positive number is at ≥0.56B activated on a multi-branch MoE, and the only small single-stream replication is null on test. Settling experiment at d=256, QAT, ≥50 tokens per table parameter: baseline vs input-summed n-gram rows vs Engram (ungated / tied-key gate / full gate), at layer 1 vs 2, slots 2^14-2^20, folded vs unfolded at equal ROM bytes; report held-out-source nll and typing top-3 hit rate.

## Could not verify
- Figs. 3, 5 values are my plot read-offs; Fig. 6 not re-read (numbers from text).
- No K (head count) ablation, row-width-vs-slots trade-off, dense/single-stream or quantised result in the paper.
- [C] is a tool paraphrase, not raw code. [R] per-seed values not seen.
- t-hertel/Engram (Qwen 0.5-4B: ppl down 22-30%, no downstream gain) known only from a search snippet.

## Links worth following
- https://github.com/deepseek-ai/Engram
- https://github.com/littlemex/zenmai-lab/tree/d845b14d8a80a0fef637091b5f6dde97bd41afca/experiments/2026-10-21-engram-nanogpt
- https://arxiv.org/abs/2501.16975 (OverEncoding baseline)

## Independent verification
Second reader, 2026-11-18. Source read directly: arXiv HTML v2 (https://arxiv.org/html/2601.07372v2), text of Secs. 3.1, 3.2, 6.2 and captions of Figs. 3, 5. Figure values below are NOT eyeball read-offs: I took the marker coordinates from the vector figure files served with the HTML (`scaling_law.svg`, `arch_ablation.svg`) and calibrated against the axis ticks; the calibration reproduces the text values 1.7248 and 1.7109 to 4 decimals. Tags: [P] = stated in paper text, [V] = recovered from the paper's vector figure (primary, but not printed as a number by the authors).

1. **Table at fixed compute, 1.808 -> 1.768; log-linear ~0.028 nats per 10x slots: CONFIRMED.**
   - [P §6.2] 12-layer 3B MoE, 0.56B activated, 100B tokens; baseline 1.808; fixed 1.6B-parameter Engram, {2,3}-grams at layers 2 and 6, 1.768, "a substantial improvement over the MoE baseline" (delta 0.04).
   - [P §3.2] slots swept 2.58e5 to 1.0e7 (up to ~13B params); "strict power law (linear in log-space)".
   - [V Fig. 3 right] eight Engram points (the note lists five): 2.59e5 -> 1.7883; 6.5e5 -> 1.7760; 1.29e6 -> 1.7678; 1.94e6 -> 1.7654; 2.59e6 -> 1.7603; 3.9e6 -> 1.7527; 6.5e6 -> 1.7481; 1.03e7 -> 1.7432. End-to-end slope 0.0281 nats per decade. OverEncoding: 1.7820 / 1.7711 / 1.7670 / 1.7505 at 1.29e6 / 2.59e6 / ~5.2e6 / ~1.3e7 slots, i.e. 0.011-0.014 worse at equal slots.
   - Nuances: the 1.768 figure is the two-module (layers 2+6) model, not a single table; single runs, no seeds or error bars; this sweep ADDS parameters (it is not iso-parameter; only §3.1 is); point-to-point scatter around the line is about 0.002.

2. **Layer 2 best single placement: CONFIRMED (read-offs accurate to ~0.0005).**
   - [P §6.2] "Layer 2 achieves the best single-layer performance (Val Loss = 1.770)".
   - [V Fig. 5] L1 1.7730, L2 1.7705, L3 1.7748, L4 1.7762, L5 1.7786, L6 1.7774, L7 1.7793, L8 1.7802, L9 1.7808, L10 1.7810, L11 1.7824, L12 1.7822. Y-axis ticks are spaced 0.0025 but printed to 3 decimals (1.768 = 1.7675, 1.773 = 1.7725, ...), which is easy to misread; the earlier read-offs handled it correctly.
   - Nuances: L1-vs-L2 gap is 0.0025 nats; the curve is non-monotone (L5 worse than L6), so run-to-run noise is plausibly ~0.001. The paper's own explanation for L2 > L1 is that the gate needs one round of attention and that the mHC parallel branches have not yet diverged at layer 1; the second reason does not exist in a single-stream model, and an ungated/tied-gate variant removes the first. "Layer 2 of 12" does not transfer mechanically to "layer 2 of 4-6".

3. **+4-gram slightly worse; no gate ~0.0045; no tokenizer compression ~0.006: CONFIRMED.**
   - [P §6.2] 4-grams "slightly suboptimal under a fixed 1.6B budget"; authors add they "do not rule out" a benefit at larger memory.
   - [V Fig. 5, deltas vs reference 1.7678] w/o short conv +0.0005 (1.7683); +4-gram +0.0017 (1.7695); w/o gating +0.0046 (1.7724); w/o token compress +0.0057 (1.7735); w/o multi-branch +0.0064 (1.7743).
   - Nuances: the largest ablation is the multi-branch (per-branch gating) one, which has no counterpart in our design. "w/o gating" was measured on the mHC backbone with untied key/value projections; nothing here measures a tied-key or folded gate. Tokenizer compression was 23.43% on a 128k-token vocabulary [P Table 6]; on a 16K lowercase-heavy simple-English vocabulary the mergeable fraction, and so the gain, will differ (unknown direction/magnitude).

4. **20-25% of sparse budget to the table is optimal, 1.7248 -> 1.7109 at 9.9B/993M, same optimum at 5.7B/568M: PARTLY.**
   - [P §3.1] text is as claimed: "reallocating roughly 20%-25% of the sparse parameter budget to Engram yields the best performance"; 1.7248 at rho=100% -> 1.7109 "near the optimum of rho ~ 80%"; optimum "stable across regimes (rho ~ 75%-80%)"; P_tot/P_act ~ 10 at both budgets.
   - What differs: [V Fig. 3 left] the 20-25% figure is the minimum of a fitted U-curve, not the best measured run. Measured points, 6e20: rho 40/50/60/70/75/80/85/90/95/100% -> 1.7247 / 1.7196 / 1.7136 / 1.7122 / 1.7126 / 1.7109 / 1.7095 / 1.7125 / 1.7130 / 1.7248. 2e20: 1.8275 / 1.8227 / 1.8190 / 1.8172 / 1.8177 / 1.8170 / 1.8155 / 1.8172 / 1.8201 / 1.8292. At BOTH budgets the lowest measured loss is at rho ~ 85% (15% moved), and 1.7109 is the rho ~ 80% point, not the minimum (1.7095). The basin from rho = 70% to 95% is flat to within ~0.003 nats and non-monotone, so the data support "anything from 5% to 30% moved gives 0.012-0.015 nats", not a sharp 20-25% optimum.
   - The result is conditional on a fixed sparsity ratio of 10 and on the thing being displaced being inactive MoE experts (99 experts at rho=100%, 43 at rho ~ 40% [P]; roughly 80 at the optimum, my estimate). It says nothing about trading ACTIVE compute for table.

**Bottom line for a 1-2M-active-weight model.** The numbers are transcribed correctly, but every one is a single run at 0.56B-3.8B activated parameters, full precision, 128k vocabulary, multi-branch (mHC) MoE, >=100B training tokens; the total effect is 0.04 nats on 1.81 (2.2%) and the placement/ablation differences being relied on are 0.002-0.006 nats, of the same order as the visible scatter. Treat direction (early placement, 2/3-grams before 4-grams, gate helps, more slots helps) as a prior and none of the magnitudes as transferable.
