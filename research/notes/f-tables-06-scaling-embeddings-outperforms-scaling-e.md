# Scaling Embeddings Outperforms Scaling Experts in Language Models  (https://arxiv.org/abs/2601.21204, v2 Feb 2026; Liu, Zhang, Wang, Hu, Lyu, ... Pei, Xie, Cai; Meituan LongCat)

Tags: [P] paper HTML v2, full text; [F] read off a rendered paper figure (about ±0.001); [C] released `modeling_longcat_ngram.py` / `config.json`; [S] summarising fetch of another paper; [E] my estimate.

## What it is
A technical report on where extra stored parameters should go: more MoE experts, or hashed n-gram embedding tables (NE) summed into the input embedding. From-scratch runs at 280M, 790M and 1.3B activated parameters (300B tokens each) compare NE against an MoE of equal total size. NE wins only once the expert count is past its "sweet spot" and tables are at most about half of all parameters; more in wide models, less in deep ones. It ships LongCat-Flash-Lite (68.5B total, 31.4B in tables).

## Mechanism
[P Eq. 3] `e_i = (E0[t_i] + Σ_{n=2..N} Σ_{k=1..K} W_{n,k} · E_{n,k}[H_{n,k}]) / ((N-1)K + 1)`
- `E_{n,k}`: V_{n,k} × D/((N-1)K); `W_{n,k}`: D × D/((N-1)K), bias-free.
- Hash [P Eq. 2]: `H = (Σ_{j<n} t_{i-j} · V0^j) mod V_{n,k}`; powers of V0 are pre-reduced mod V_{n,k} [C].
- Sub-table `idx = (n-2)K + k` has size `ratio·V0 + 2·idx + 1` [C]. Released model: V0 = 131072, ratio 78, N = 4, K = 4, D = 3072, so 12 tables of width 256 [C].
- Context is zeroed across EOS [C].
- Embedding Amplification [P §3.2.4]: multiply the embedding output by sqrt(D), or LayerNorm it.
- PLE [P Eq. 4]: `FFN(x) = W_d(SiLU(W_g x) ⊙ E^l[t_i])`; the token row replaces the SwiGLU up-projection. PLNE uses an n-gram row instead.

## Numbers
- Fig. 2, 280M activated, x = total/activated [F]. Training loss, MoE: 1.986 (4×), 1.875 (12×), 1.829 (22×), 1.806 (33×), 1.791 (44×). NE added to the 12× base: 1.837 (17×), 1.821 (22×), 1.807 (33×), 1.797 (44×). English validation: NE leads to about 20×, then loses: 2.932 vs 2.907 at 44×. NE added to the 4× base is always worse (1.904 vs 1.875 at 12×).
- Crossover [P]: above 30× NE loses at 280M; at 790M it loses only on English; at 1.3B it still wins at 50×.
- Fig. 4, 790M, training / Chinese / English validation [F]: N2-K1 1.7370 / 2.7038 / 2.8395; N2-K4 1.7284 / 2.6968 / 2.8314; N3-K3 1.7216 / 2.6890 / 2.8241; N5-K3 1.7218 / 2.6869 / 2.8284; N7-K2 1.7260 / 2.6917 / 2.8303; N7-K4 1.7211 / 2.6926 / 2.8339. Whole spread is 0.015 nats.
- Fig. 3b [F]: 2-gram collisions over 100 sequences are about 130 at generic sizes and about 40,000 at exactly 30, 31, 32, 33 × V0, still 400-2,000 right next to them.
- Fig. 5 [P, F]: first attention output norm is about 9× the embedding norm. Amplification: -0.02 on all three losses [P].
- Fig. 7, NE minus MoE at about 50% table share [F]: 6.4B/280M +0.008 / +0.008 / -0.006; 25B/790M +0.011 / +0.008 / -0.014; 34B/1.3B +0.042 / +0.047 / +0.027. Depth at 1.3B: 10 layers 0.042 / 0.047 / 0.027; 20 layers 0.023 / 0.009 / 0.002; 40 layers 0.021 / 0.017 / 0.002.
- Fig. 9, 790M [F]: NE-10B 1.741 / 2.7077 / 2.848; PLE-10B 1.758 / 2.7112 / 2.8635; NE-12B 1.732 / 2.6992 / 2.842; PLNE-12B 1.734 / 2.6967 / 2.837.
- Table 1 [P], 68.5B model, vanilla vs NE: MMLU 64.81 / 64.01; BBH 38.54 / 43.67.

## Smallest scale tested, and whether the result held there
280M activated, 10 "shortcut layers" (the paper counts 40 as 80 conventional). It held only narrowly: +0.008 nats in a window of roughly 12-30×, and negative on English at 23×. The width trend points against us; only depth (ours is shallower than anything tested) points our way. Over-Encoding, the parent method, reports nothing numeric below 1B dense [S].

## Training recipe and gotchas
- No optimizer or learning-rate details for the tables are given.
- Without amplification the table signal is swamped after one attention layer.
- Never size a table at a multiple of V0. Reason [E]: with V_n = c·V0 the hash is `t_i + V0·(t_{i-1} mod c)`, so the previous token keeps only c values.
- The released sizes (78·V0 + 1, +3, ...) sit right next to a multiple, where Fig. 3b shows elevated collisions; unexplained.
- All comparisons are iso-total-parameter with soft top-12 routing over 256 experts, not hard 1-of-64.

## Mapping onto the GBA design
All [E]. D = 256, N = 3, K = 2: four sub-tables of width 64.
- Faithful: 4 row reads plus four 256×64 ternary projections = 65.5K weights, about 105K cycles (2-3% of budget). ROM 64 bytes per slot at 8 bits; 16 MB gives 62K slots per table (3.8× V0); ternary rows give about 19× V0.
- Folded: store `W·E` rows D-wide at export (exact, but 8-bit, 256 bytes per slot); 4 × 256 adds, about 5K cycles.
- Hash: the paper's hash with V0 = 2^14 and power-of-two tables is degenerate (the third token vanishes entirely). Use odd-multiplier multiply-shift, or a non-power-of-two modulus by reciprocal UMULL (about 20 cycles).
- Amplification: sqrt(256) = 16 is a shift; it and the 1/((N-1)K+1) divide fold into the rows.
- RAM: last N-1 token ids only.
- PLE removes one MLP matrix in exchange for a lookup. It lost at equal parameters, but was never compared at equal compute, which is our constraint.
- Our ROM/active ratio (33-80×) is beyond the 280M crossover, so at fixed ROM more experts may beat bigger tables.

## Verdict
Test. Adopt the cheap rules now: N = 3, K >= 2, amplification, no multiples of V0, tables at most 50% of ROM. Settling run at D = 256, 4-6 layers, QAT, fixed ROM bytes: table share 0 / 25 / 50 / 75% against expert count 16-256; N in {2,3,4} × K in {1,2,4}; with and without amplification; PLE at equal cycles. Report validation loss and typing top-3 hit rate.

## Could not verify
- Figure values are read-offs. Table widths, V_n and optimizer settings for the 280M-1.3B runs are not stated.
- Whether the released model applies amplification: not visible in the file read; the parent class was not read.
- Over-Encoding numbers come from a summarising fetch [S], not a direct read.

## Links worth following
- https://huggingface.co/meituan-longcat/LongCat-Flash-Lite
- https://arxiv.org/abs/2501.16975 (Over-Encoding)
- https://arxiv.org/abs/2601.10639 (STEM, per-layer embedding)
- https://arxiv.org/abs/2312.16903 (Spike No More, embedding scaling)
