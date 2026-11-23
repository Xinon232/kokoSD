# modded-nanogpt speedrun, record #92 "ANVIL2" (https://github.com/KellerJordan/modded-nanogpt, 2024-2026, Keller Jordan + ~60 contributors; commit 4ea6b93, 2026-09-28)

## What it is
Competitive benchmark: fastest 8xH100 training of a GPT-2-small-class model (768 wide, 11 layers, 6 heads) to <=3.28 val CE on FineWeb. 92 records took it from 45 min / 10B tokens to 39.9 s / ~329M tokens (1194 steps; token total is my sum). A trick survives only if it buys loss per wall-clock at p<0.01. Read in primary: README, records table, `train_gpt.py`, the `track_1_short/` package, 14 record READMEs.

## Mechanism (current code unless a record is named)
- **Hashed n-gram table**: V=84,602,880 rows x 768, zero-init; first half bigram, second half trigram. `big = (36313*x[t] ^ 27191*x[t-1]) % (V/2-1)`; `tri = (17351*x[t] ^ 60961*x[t-1] ^ 45259*x[t-2]) % (V/2-1) + V/2` (int32 wraparound). **Sign trick**: fixed pool `sign(randn(8192, D))`, row picked by an independent hash (`(48271*x[t] ^ 30011*x[t-1]) & 8191`; trigram analog). `e = row_big*sign_big + row_tri*sign_tri`, added to the residual's first D dims before layer 0 and at layers 1,4,5,9,10, each with a per-token scalar gate (init 0.05). Record #62: 5x50304 rows x 768, static per-layer lambda 0.1. Record #83: 15x50304 rows x 192.
- **Value embeddings**: 4 planes `[vocab, 768]`, init N(0, 0.01), on layers 1,2,8,10: `v += 2*sigmoid(W . cat(norm(x)[:6], ve[:6])) * ve`; W is `[heads, 12]`, zero-init.
- **Smear**: `x[t] += lam*sigmoid(w . x[t,:12]) * x[t-1]` before the first norm (lam init 0, learned ~0.07).
- **Sparse attention gate** (#28-#89): `y_head *= sigmoid(W_h . x[:12])`; now a 768->64->k GELU MLP gates heads on layers 3 and 10 only.
- **Paired heads** (layers 0,2,5): heads h and h+3 interleave into one length-2T stream, so one softmax sees two keys per position.
- **Partial key offset** (layers 3,10): the non-rotary half of each key comes from token t-1 (one-layer induction).
- **Topology**: attention on layers 0,1,2,3,5,8,10 only; ReLU^2 MLP (hidden 2816) on all but 7, plus a parallel MLP at 8; layer 6 adds gated layer-3 output; attention layers 8 and 10 both read `norm(cache[7])`; x0 re-injected at six layers; final per-token mix of 10 earlier activations (starts as -0.5 x layer 7). QK 64/head on 5 of 7 attention layers.
- **Loss**: `z = 23*sigmoid((logit+5)/7.5)`. MTP: one logit vector scored against the next 3 tokens, weights [1,.5,.25->0] -> [1,.5->0] -> [1] by stage. Prefix-token CE (longest proper byte-prefix token of the target), weight .25 -> 0.
- **Head**: embed tied to lm_head (init std .005) until the last 20 steps (record #53: untie at 2/3).

## Numbers (all 124M-class, FineWeb, 8xH100)
- Table-size dose grid (ANVIL2 README, n=2 each): 377,280 rows 3.3149/3.3152; 21.15M rows 3.2825/3.2863; 84.6M rows 3.2791; wall 39.0 -> 39.5 s.
- Ablations (same README, n=2): n-gram table back to #89's +35.8 mnats; no tail weight-averaging +20.2; no mixed-width attention -6.9 mnats but +4.95 s; no FP8 -0.4 mnats.
- #62 bigram hash: 1765 -> 1600 steps at equal loss (n=6). #83 sign trick + 768->192 dims: 3.27937 (n=10) -> 3.27917 (n=20), -0.96 s.
- Value embeddings: 4.66 -> 4.41 min (#14, n=40). VE gates: ~3.2830 without, 3.2788 with (#55). Moving the last VE earlier: 3.2986 (#70).
- Steps saved: smear 15 of 1660 ("closer to 5" at matched loss); attention gate ~50 of 1750; paired heads 65; MTP + untie 80; softcap 30->15, 100 of 1490 (n=80); prefix CE 15 of 1390.
- Negative: V from the value embedding alone in 6 layers costs ~0.015 (#28 notes).

## Smallest scale tested, and whether the result held there
Only 768-wide/11-layer (plus a 350M track). Nothing at 256-384 width, ternary, or simple English. Seed sd is 1-2 mnats: effects under ~3 mnats need n>=10.

## Training recipe and gotchas
Tables: zero-init, Adam lr 0.008 x 70, beta1=0, one second moment per row, updated every 2-4 steps. Matrices: Muon-family, lr 0.023; zero-init output projections. LR flat, then linear to 0.30 over the last 80%. Weight EMA over the last ~25% of steps. Gating and multi-hash on the bigram path lost to plain addition in #62. The table has more parameters than training tokens; its gain is measured at 329M tokens only.

## Mapping onto the GBA design
- N-gram path per token: ~10 multiplies for hashes (power-of-two tables, no divide), 2 ROM rows, 2 sign rows (pool 8192 x D bits = 64 KB at D=64), 2D negate-adds: ~1-2K cycles, <0.1% of budget. ROM: 1M rows x 64 trits ~ 13 MB. Static per-layer lambdas fold into ROM; per-token gates need a ~16K-weight MLP.
- Value embeddings: D adds per layer when a token's value enters the ring; ROM 16K x 256 trits ~ 0.8 MB per plane. Lookup-only V saves d^2 weights per layer.
- Sublayers reading one normed input (layers 8 and 10; parallel MLP) can share one set of 4-input partial-sum tables (my inference).
- Key offset, MTP, prefix CE, tie->untie, EMA: zero inference cost. Softcap: one 256-entry LUT; bounded logits suit 8-bit fixed point. Attention gate: 12 MACs per head; thresholded, it could skip a head's key scan (my inference). Paired heads double keys scanned.

## Verdict
**Adopt** hashed bigram+trigram table with sign trick, gated value embeddings, softcap, MTP/prefix losses, key offset, EMA. **Test** lookup-only V, gate-as-skip, narrow QK, smear. **Skip** paired heads. Settling experiment: 256-wide ternary, sweep rows {64K, 1M, 4M} x D {32, 64, 192}, with/without trigram channel and value embeddings, 3 seeds.

## Could not verify
PR #299/#337 text came through a summarising fetch (secondary); their statistics files are primary. Record #89's table is 754,560 x 192 in its logged source, 377,280 in the ANVIL2 README. LR floor: code 0.30, README 0.20. No isolated ablation for key offset or MTP; no generation-quality numbers.

## Links worth following
- https://arxiv.org/abs/1709.03933 (hash embeddings, cited in #62)
- https://arxiv.org/abs/2502.12170 (MUDDFormer)
- https://github.com/KellerJordan/modded-nanogpt/pull/360 (ANVIL2)
