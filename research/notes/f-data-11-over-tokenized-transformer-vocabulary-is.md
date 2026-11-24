# Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling  (https://arxiv.org/abs/2501.16975, ICML 2025; Huang, Zhu, Wang, Zeng, Wang, Min, Zhou; ByteDance Seed)

Tags: [P] arXiv HTML read through a fetch tool that returns extracted text (three passes; tables returned in full, equations restated, Algorithm 1 only summarised); [E] my estimate. Author list and affiliation are from memory of the abstract page, not re-checked.

## What it is
The paper separates input and output vocabularies. Input side ("Over-Encoding", OE): add hashed 2-gram and 3-gram embedding tables to the normal token embedding. Output side ("Over-Decoding", OD / MTP): extra heads that predict the following tokens as an auxiliary loss, discarded at inference. Claim: bigger input vocabulary always helps, with training loss falling log-linearly in table size; bigger output vocabulary helps only models large enough not to underfit.

## Mechanism
[P Sec. 3.2, Alg. 1 summary]
- n-gram id: `g_n(i) = sum_{j=0..n-1} x_{i-j} * V^j` (missing positions = 0), V = base vocabulary.
- Row index: `g_n(i) mod m_t`, where table t has `m_t = m + 2*t` rows (distinct moduli; without this, slicing does nothing).
- Each n in 2..N gets k sub-tables of width `d/(k(N-1))`, each followed by its own linear projection to d_model.
- `e_i = (E1[x_i] + sum_{n=2..N} sum_{s=1..k} W_{n,s} * E_{n,s}[g_n(i) mod m_{n,s}]) / (1 + k(N-1))`
- E1 is the ordinary V x d table, tied with the output head. The output head stays size V.
- Defaults: N = 3, m = 12.8M ("OE-12.8M"), k chosen so sub-table width is about 256. "OE-m" means m x d_model table parameters in total.
- OD: `L = sum_j lambda_j * CE(h W_j E_j^T, z_j)`, lambda_1 = 1. MTP-DS predicts sequentially, conditioning head j on the embedding of token j-1. "OT" = OE + MTP-DS, one extra head, weight 0.1.

## Numbers
All [P]; training loss unless stated.
- Fig. 5, OLMoE-1.3B, 500B tokens: `L = 2.6754 - 0.0256 * log10(m)`; 4x m gives -0.015.
- Fig. 1/4, OLMo2 dense (V = 100278): OE-12.8M at 400M matches the 1B baseline ("2.5 times larger"). At 1B: -0.12 at 400B tokens, -0.14 at 1T; 5.7x faster loss convergence. No numbers are printed for the 151M and 400M models (figure only).
- Table 1, OLMoE (V = 50280): 1.3B 2.554 -> 2.472 (tables 13.1B params); 7B 2.305 -> 2.229.
- Table 2, OLMoE-1.3B, train / eval loss: baseline 2.554 / 2.924; one 2-gram table 3.2M x d 2.511 / 2.887; same sliced k=2 2.503 / 2.877; 2+3-gram, 3.2M, k=2 2.495 / 2.870; one unsliced 12.8M x d 2.493 / 2.881; 2+3-gram, 12.8M, k=2 2.472 / 2.862. Eval gain (0.062) is smaller than train gain (0.082), and the big unsliced table is worse on eval than a 4x smaller sliced one.
- Table 3, m = 3.2M, 50B tokens: 1-gram only 2.714; 2-gram only 2.785 (worse); 1+2 k=1 2.678; 1+2 k=4 2.670; 1+3 k=4 2.684; 1+2+3 k=4 2.667.
- Table 4, 50B tokens: m = 64V exactly 2.702; m about 3.2M coprime with V 2.678; baseline 2.714. Two thirds of the gain is lost when m is a multiple of V.
- Tables 5/10, output side at 1.3B: MTP 2.556, MTP-DS 2.555, OD lambda=0.1 2.549, lambda=1.0 2.555, against 2.554. Effectively zero.
- Tables 6/8: training throughput -4.6% (1.3B) and -8.3% (7B); forward FLOPs +0.38%.

## Smallest scale tested, and whether the result held there
Real text: OLMo2-151M, figure only, no number. Synthetic: a CFG over 3 characters (sequences up to 729), GPT-2 style, 12 layers, width 768 (85M) and width 128 (2.4M). There, 3-gram input "consistently enhances performance across all model sizes"; 3-gram output "improves performance for larger models but degrades it for smaller ones". No losses are printed for this either. So at our scale (10-100M stored, 256 wide, 4-6 layers) the direction is supported only by a toy grammar with a 3-symbol alphabet and unhashed 3-gram vocabulary of 41 entries.

## Training recipe and gotchas
- CFG runs: AdamW, betas (0.9, 0.98), wd 0.1, peak lr 3e-4, cosine, 10 epochs over 20M sentences. Real-text runs keep the OLMo2 config; no table-specific lr, weight decay or init is given.
- "Slow start": large input vocabularies "require more training tokens to realize their expected benefits" (App. B.2). Risky at a few hundred GPU-hours.
- Never drop the 1-gram table. Never make m a multiple of V.
- Slicing (k >= 2) matters as much as size.
- No code URL in the paper.

## Mapping onto the GBA design
All [E]. V = 16384, d = 256, N = 3, k = 2: four sub-tables of width 64.
- Per token as published: 4 ROM row reads plus four 64->256 ternary projections = 65.5K weights, about 105K cycles (2-3% of budget).
- Folded at export: store `W*E / 5` rows 256 wide; 4 row reads and about 1K adds, a few thousand cycles. Costs 4x the ROM per row and forces 8-bit rather than ternary rows.
- ROM: unfolded 8-bit, 64 B/row; 8 MB total gives m about 32K per table, only 2V. The paper's regime is m/V = 12 to 128. Ternary rows (16 B) at 16 MB give m about 262K = 16V.
- Hash: V = 2^14 with a power-of-two m is the degenerate case of Table 4 (older tokens vanish). Use odd/prime m with a precomputed `V mod m`, `V^2 mod m` and reciprocal-multiply reduction, or a multiply-shift hash; tens of cycles.
- RAM: last two token ids.
- Output head: keep 16K, two-level. Skip MTP/OD; it gave nothing even at 1.3B.

## Verdict
Adopt the input-side structure (it is already in the design); test the size. The log-linear slope was measured at 1.3B with m/V of 60-250 and must not be extrapolated. Settling experiment: 256-wide 4-6 layer QAT model, fixed step budget, tables off / m = 2V / 8V / 32V, k in {1, 2}, 2-gram only vs 2+3-gram; report validation (not training) loss, plus loss versus steps to expose slow start. Compare against spending the same ROM on more experts.

## Could not verify
- Any numeric result at 151M, 400M, 85M or 2.4M: only figures exist.
- Algorithm 1 verbatim, table init, optimizer treatment of tables, code release.
- Table 7 inference speeds came back with garbled column headers; not used.
- The "underfitting" explanation for the output-side result is the authors' interpretation, not an ablation.

## Links worth following
- https://arxiv.org/abs/2601.21204 (LongCat follow-up, tests 280M; notes in f-tables-06)
- https://arxiv.org/abs/2305.13673 (CFG synthetic setup the toy task is based on; recalled, not checked)
