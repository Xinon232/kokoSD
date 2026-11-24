# nanochat dev/LOG.md — running experiment log (https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LOG.md, Jan–May 2026, Andrej Karpathy)

Read in primary form (full 1089-line file, fetched 2026-11-19) plus `nanochat/gpt.py` on master.

## What it is
A dated lab notebook for nanochat (a "GPT-2 capability in ~2-3 h on 8xH100" training repo). Each entry tests one trick, mostly ported from modded-nanogpt, and records adopt/reject. The decision metric is validation bits-per-byte (bpb) versus wall clock on H100s, not versus inference compute, so several "negatives" are really "better per step, slower on GPU". Model sizes are named by depth: d8 = width 512, d12 = 768, d16 = 1024, d20 = 1280 (width = 64 x depth), vocab 32K, context 2048.

## Mechanism (the things that worked)
- **x0 / residual scalars** (before every block i): `x = resid_lambda[i]*x + x0_lambda[i]*x0`, `x0 = norm(wte(idx))`. Log: x0_lambda init 0, resid init 1. Current gpt.py init: resid 1.15→1.05, x0 0.20→0.05 linearly over depth. LR: x0 ~0.5 (AdamW), resid 100x smaller; x0 beta1 0.96 at d20 (0.98 is a cliff: +0.0033 bpb).
- **Value embeddings (VE)**: one extra `Embedding(vocab, kv_dim)` per alternating layer (last layer always has one), added to attention values: `v = v + gate*ve[token]`, `gate = 3*sigmoid(W_g x[:12])` per head (W_g: 12→n_head). Zero matmul cost beyond the 12-channel gate.
- **Bigram hash embedding ("engram-lite")**: `h = ((36313*cur) ^ (27191*prev)) % (table-1)`; table = 5 x vocab rows x d_model, zero-init; added at every layer with per-layer scalar (init 0.1): `x += bigram_lambda[i]*E[h]`. AdamW at token-embedding LR.
- **MLP**: `c_proj(relu(c_fc(x))^2)`, 4x expansion. Logit softcap `c*tanh(logits/c)`.
- **Optimizer**: Muon (+ NorMuon per-row variance, beta2 0.95) for matrices, AdamW for embeddings/scalars; cautious weight decay (decay only where update*weight >= 0), WD linearly ramped to 0; WD ∝ 1/width^2.
- **Data**: every row starts with BOS (BestFit-Crop packing). Batch size B_opt ∝ D^0.383 from d12 reference 2^19 tokens.

## Numbers
- x0 lambdas (table, 2026-01-11): d8 1.0885→1.0782 (-0.0103); d12 0.9770→0.9693 (-0.0077); d16 -0.0057; d20 -0.0039. Best scalar_lr 0.2/0.6/0.2/0.1. resid_lambdas add a "small additional improvement" (no number).
- VE (01-17): "helps a lot", alternating layers best; all parameter sharing, low-rank and projection variants failed; gate helps. No bpb delta given. Optimal tokens:params ratio halved 8→4. At d12: VE 151M params vs 85M transformer matrices.
- Bigram hash (01-27): "consistent improvement" on steps/wall clock/FLOPs at d12 (+126M params); table multiplier 5 best; trigrams no gain; context gating no gain; early-layers-only hurt; norm on the embedding slightly worse; noisy init worse. Reverted 01-28: "tiny" at d25, gone on wall clock, VRAM bloat. No numbers.
- WD optimum: d8 0.40, d12 0.22, d16 0.10, d20 0.08 (alpha ≈ 1.97).
- Hyperparameter sweep (~320 runs): gains ~0.002 at d12, ~0.0007 at d20; d12-tuned values hurt at d20.
- Number tokens at d12/32K vocab: `\p{N}{1,1}` 0.969, `{1,2}` 0.965, `{1,3}` 0.972.
- Softcap at d24: 5 terrible, 10–30 equal, 20 best (0.716→0.715).
- Varlen attention d16: 0.85427→0.85407 (noise). Grad clipping d12/d20: no effect, grad norm never exceeds 1.
- MoE (8 experts, top-2, 1 shared, sigmoid gate, iso-FLOP, d18): better per step, MFU 46%→35%, net loss on wall clock.
- Negative or neutral: SwiGLU (d12, d24: worse on steps, FLOPs and time); DyT (d12, ~10% slower); multi-token prediction (d12, same/slightly worse per step, 34→47 GB); partial/half RoPE; LN scale; orthogonal init; attention gates; skip connection; EMA/SWA; MuonH; skip-AdamW; flipped Muon LR. Better per step but slower: LeakyReLU(0.5)^2, XSA.
- Data: ClimbMix cut time-to-GPT-2 2h46m→2h01m (d26→d24, ~7B tokens); vanilla FineWeb CORE 0.2602→0.2241; Dolma3 mix d16 CORE 15.5→13.8.
- Aspect ratio 64 beats 128 at 1e18 FLOPs; head dim 128 beats 64.

## Smallest scale tested, and whether the result held there
d8 (width 512, 8 layers, 42M matrix params, 0.44B tokens), and only for x0 lambdas and weight decay; both effects were largest there. Everything else is d12 or larger. Nothing is at our width/depth, nothing is quantized.

## Training recipe and gotchas
- Multiplicative scalars need ~100x lower LR than additive ones.
- Zero-init hashed tables, lambda init 0.1 (not 0).
- Do not tune on a small proxy and ship at another size; re-tune WD and LR at target width (WD extrapolates to ~0.8+ at width 256 by the 1/width^2 law — untested).
- BOS-aligned rows lower loss "artificially"; not comparable across the change.
- Mostly single runs; noise floor looks like ~0.0002–0.001 bpb.

## Mapping onto the GBA design
- x0 lambdas: one ROM/RAM copy of x0 (256–384 bytes) and one scaled add per layer (~width multiplies, ~3K cycles/layer; or shift-only if lambdas are rounded to powers of two). Essentially free.
- VE: per-layer ROM table vocab x kv_dim (16K x 256 at 8 bit = 4 MB/layer; ternary ~0.8 MB) and one row read + add per layer per token. The values are stored in the ring buffer anyway, so no extra RAM. Gate: 12 x n_head MACs plus a sigmoid LUT, or fold a constant gate into the table.
- Bigram hash: one hash (2 multiplies, XOR, mask with power-of-two table), one ROM row, one add per layer; per-layer scalar can be baked into per-layer tables (already in our design).
- The wall-clock negatives (MoE, bigram at d25, XSA) do not transfer: our cost model charges lookups nearly nothing. MoE improved per-step loss at iso active FLOPs, which is the relevant fact for hard routing.
- Keep ReLU^2 (one multiply per hidden unit, or a 256-entry LUT) over SwiGLU.

## Verdict
**Adopt** x0/resid lambdas, value embeddings, bigram hash at every layer, ReLU^2, logit softcap, BOS-aligned rows, `{1,2}` digit grouping check. **Test** whether VE survives ternary/8-bit tables and a 16K vocab; ablate {x0, VE, bigram} at our width with QAT on, 2-3 seeds. **Skip** SwiGLU, DyT, MTP, gating on n-gram lookups, trigram tables sharing one budget.

## Could not verify
- No bpb delta is given for VE, bigram hash, MoE per-step gain or resid_lambdas.
- Seeds and run-to-run variance are never stated.
- Log vs code conflict: the log calls smear and backout negative (Jan 16) and softcap 20 the default, but master gpt.py has smear and backout enabled and softcap = 15; no log entry explains this.
- EMA/SWA result is a one-line recollection.

## Links worth following
- modded-nanogpt PR #201 (engram-lite): https://github.com/KellerJordan/modded-nanogpt/pull/201
- Engram paper: https://arxiv.org/abs/2601.07372
- Value residual / ResFormer: https://arxiv.org/abs/2410.17897
- Power Lines (batch size scaling): https://arxiv.org/abs/2505.13738
- Cautious optimizers: https://arxiv.org/abs/2411.16085

## Independent verification

Checked against the primary source (raw `dev/LOG.md` on master, fetched 2026-11-19, 1089 lines). All statements below are from the log itself, not from secondary summaries.

1. **x0 lambdas: -0.0103 bpb at d8, -0.0077 at d12, -0.0039 at d20, no compute cost — CONFIRMED (with caveats).** Entry "2026-01-11: Per-Layer Residual Scalars", table "Experiment Results": d8 1.0885 -> 1.0782, d12 0.9770 -> 0.9693, d16 0.9059 -> 0.9002 (-0.0057), d20 0.8565 -> 0.8526. Summary line: "Solid improvement with essentially no compute overhead." Caveats: (a) each delta is the *best* point of a `scalar_lr` sweep against a single baseline run, so it is an upper-ish estimate (selection over LR, no seeds); best LR is non-monotonic in depth (0.20, 0.60, 0.20, 0.10). (b) "width 512" for d8 is not stated in this entry; it follows from nanochat's width = 64 x depth convention. (c) It is not stated whether the swept runs had resid_lambdas on; the entry only says resid_lambdas add a "small additional improvement over x0 alone" (no number).
2. **Value embeddings on alternating layers "help a lot"; capacity cuts all failed; no number — PARTLY.** Entries "2026-01-17: Various experiments" and the Jan 17 sweep table ("Helps a lot"). Confirmed: alternating layers best of {every layer, alternating, U-shaped, front/back}; gate helps; sharing, low-rank and projections "All failed"; wins on steps, FLOPs and wall clock; no bpb delta anywhere. NOT in the source: the depth range "d12-d20" — the VE entry names no depth at all. Also relevant and omitted: VE made the model so parameter-heavy that the optimal data:param ratio halved from 8 to 4 (d12: value_embeds = 151M of 412M params, vs 85M in transformer matrices). The tables are full-rank vocab x n_embd floats; nothing in the log tests quantized or small-vocab tables.
3. **Zero-init bigram hash table, 5 x vocab rows, every layer, consistent gain at d12; trigrams/gating/early-only did not help — PARTLY.** Entry "2026-01-27: Bigram Hash Embeddings". Confirmed: table size vocab x 5 ("5 was optimal"), zero-init table, per-layer lambda init 0.1, "provides a consistent improvement", no numeric delta; trigrams gave no gain; early-only (first 4 layers) "Actually hurt performance". Differences: (a) it is ONE shared table whose row is added at every layer with a per-layer scalar, not a table per layer; (b) the context-gated Engram variant gave a "Slight improvement" and was dropped for complexity — it is not a clean negative; (c) the trigram test split one budget (concatenated embeddings), so it shows dilution, not that trigrams with their own table are useless; (d) depth of the experiments is not stated — d12 appears only in the parameter count (126M params for the table); (e) MOST IMPORTANT: the next entry (2026-01-28) reverts it — at d25 "the improvement was tiny and disappeared entirely when measured by wall clock time", and the 2026-03-24 entry says it "did help somewhat" but was not justified at scale. So the gain shrinks with model size on a per-step basis too ("tiny"), not only on wall clock.
4. **MoE 8 experts top-2 at equal active FLOPs improves per-step val loss at d18; rejected only for MFU 46% -> 35% — PARTLY.** Entry "2026-02-19: Mixture of Experts (negative)". Confirmed: 8 routed experts, top-2, sigmoid gating, iso-FLOP sizing of expert hidden dim, "improves per-step validation loss" (no number, and no depth given for that statement; d18 is named only for the MFU drop ~46% -> ~35%). Differences: (a) there is also 1 always-on shared dense expert, so 3 expert MLPs are active per token, each with hidden dim ~4*dim/3; this is not pure routing; (b) rejection was not only MFU: the verdict also cites substantial code bloat, missing FP8 support for routed experts, optimizer fixes; (c) the size of the per-step gain is unknown and it "does not compensate for the throughput hit", i.e. it was worth less than a ~24% throughput loss.

**Overall.** Numbers in claim 1 are transcribed correctly; claims 2-4 are qualitatively right but each adds a detail the log does not contain (depth ranges) or drops one that matters (bigram reverted at d25; shared expert in MoE; gating gave a slight gain). All evidence is single-run, float bf16, 32K BPE vocab, widths 512-1600, web text; nothing is tested with ternary weights, 8-bit tables, 16K vocab or widths 256-384. The trend in claims 1 and 3 (gain shrinks as the model grows) favours our small model, but that is extrapolation below d8, which the log never ran. The MoE result (soft top-2 of 8 plus shared expert) is weak support for hard 1-of-64 routing with no shared path.
