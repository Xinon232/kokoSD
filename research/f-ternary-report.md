# Closing the ternary gap and cheap attention keys (focused round, final)

Cluster report, 18 Nov 2026. It covers 20 sources (notes in `research/notes/f-ternary-*.md`) and reads them against our pilot (`research/pilot-results.md`: 128 wide, 4 layers, 491M TinyStories tokens, seed noise about 0.005 nats).

Tags: **[M]** measured by a source, **[P]** measured in our pilot, **[D]** derived by me or a note author, not measured. All GBA cycle, RAM and ROM figures are [D] unless marked.

## 1. Bottom line

1. **Ternary costs about 2.2-2.5x weights for equal loss, and four independent routes now agree.** Spectra's fits put a 99M ternary model at about a 45M float one [D from M]; ParetoQ's tables give 1.7-2.8x [D from M]; the information ceiling (1.585 bits against 3.5-3.6 measured bits per float parameter at 80K-6.9M parameters) gives at least 2.2-2.4x [D from M]; our 2.25x-weights ternary model lands 0.06 nats short of float [P]. Evidence: moderate; no published controlled comparison below 99M.
2. **Widening still beats extra bits or planes per cycle, but the margin depends on which law is assumed.** Ternary beats 8-bit per cycle by about 3x under Frantar's multipliers and about 1.3x under Kumar's fit [D]; both ternary points are extrapolations (interpolated; below a 3-bit floor). Evidence: weak-moderate.
3. **The one-zero-in-four format is a poor trade.** It costs 0.07 nats for about 9% speed [P]; its ceiling is 1.25 bits per weight (2.8-3.0x multiplier) [D]; 2:4 blocks were 0.05 nats worse than large blocks at 100M [M]. Evidence: moderate.
4. **Weights are the safe thing to quantise; attention activations are not.** Binarized NMT: 1-bit weights with float activations match float (1.38 vs 1.39), a fully 1-bit FFN costs 0.01, a 1-bit Q.K product costs 0.38 and 1-bit value mixing is catastrophic (2.81) [M]. Evidence: moderate (encoder-decoder, d=1024).
5. **Late weight-decay removal is the only schedule trick with an ablation** (Spectra, 50.1 vs 49.1 at 1.1B [M]); the mid-run LR drop raised training loss below 1B [M]. Evidence: moderate, far from our scale.
6. **"Float first, then short QAT" is unproven at our budget.** ParetoQ's 90/10 result is one plot at 125M and 100B tokens (verified, with softer wording than the headline) and our 20% warm start did nothing [P]. Evidence: weak.
7. **Sign keys need a scale and a position bias, and 8-bit keys do not fit the budget.** BinaryAttention matches float on DeiT-Tiny (72.88 vs 72.2; scale +0.47, bias +0.44) [M]; our bare sign keys lost 0.15-0.17 nats [P]; an 8-bit scan is about 123K cycles per head per layer, 2-3M per token [D]. Evidence: moderate for vision, none for a causal LM.
8. **Stored capacity, not touched weights, sets what the model can memorise**, so the ternary deficit is best repaid in ROM. Embedding rows counted at the same bits per parameter as other weights [M]; our tables and experts gave -0.13 to -0.25 nats at no compute [P]. Evidence: moderate.

## 2. Ranked recommendations

| # | Technique | Action | Expected gain (source scale) | GBA cost | Evidence | Main risk |
|---|---|---|---|---|---|---|
| 1 | Free ternary (81-entry tables), drop 3:4 | adopt | +0.07 nats [P, 128 wide]; 2:4 costs 0.05 at 100M [M] | 81 vs 32 entries built per group; about 9% slower [P estimate] | Moderate | Table build dominates small routed matrices |
| 2 | Weight decay to 0 for the last third; no mid-run LR drop | adopt | +1.0 point of 6-task average at 1.1B [M] | None | Moderate | Not ablated below 1.1B |
| 3 | High ternary LR; sweep past x10, judge only at end of schedule | adopt | x10 beat x3 by 0.05 nats [P]; 6x in Spectra [M] | None | Moderate | Frantar could not reproduce it (at 3 bits) |
| 4 | Kernel layout: group-outermost tiling, tile-ordered ROM, int16 tables | adopt | No quality change; table RAM 26 KB to 5.2 KB at 640 inputs [D] | One load+store per row per tile | Strong for exactness [M] | Shapes must be multiples of tile size |
| 5 | 8-bit lookup embeddings, never post-hoc ternary | adopt | Perplexity 17.120 vs 17.109 at 2B; ternary embeddings fail [M] | ROM only | Moderate | Ours are hashed, not dense |
| 6 | Foldable per-row gain/shift after every ternary matrix feeding a hard decision | adopt | BinaryConnect without it 125.9 vs 90.7 perplexity (720K weights) [M]; NMT norm-after-matmul needed to converge [M] | 1 multiply + 1 add per output row, under 1% | Moderate (LSTM, NMT) | Gain may be absorbed by existing pre-norm |
| 7 | Learned per-head scale + relative-position bias on sign keys | test first | +0.47 and +0.44 top-1 on DeiT-Tiny [M] | 256 ROM bytes per head per layer; 1 load + 1 add per position | Moderate, vision only | May not survive causal from-scratch training |
| 8 | Asymmetric keys: 8-bit query, 1-bit key, 16-entry group tables | test | 1.2-2.5 LongBench points over sign-only at 8B [M] | 20-35K cycles per head per layer | Weak for us | Scan cost forces fewer heads |
| 9 | Share top-8 selection across adjacent layers | test | 8 of 12 layers shared from scratch matched baseline at 164M, near-chance benchmarks [M] | Saves about 0.26M cycles and 2 KB RAM per head per shared layer | Weak | Early/mid sharing broke 7B models; hurts exact recall |
| 10 | Capacity probe (random-token memorisation) for ternary and tables | test | Float 3.5-3.6 bits/param at 80K-6.9M [M]; ternary unmeasured | None (hours of GPU) | Strong method, no ternary data | Needs very long training |
| 11 | Learned per-row scale (LSQ gradient scaling, SEQ grid) | test | Beats statistics-based scales at 125M (plot only) [M] | Folds into item 6 | Weak-moderate | Spectra: per-matrix is enough |
| 12 | Float twin, then ternary continuation with logit distillation | test | MNLI 74.1 to 86.7 to 88.2 at 0.6B [M]; 90/10 peak at 125M [M, plot] | None | Weak; pilot negative | 10% of our tokens is far below the 30B ParetoQ needs |
| 13 | Sign-input (1-bit activation) expert MLPs | test | +0.01 loss, rest float, d=1024 [M] | 16-entry tables: about 5x cheaper build | Weak | Frantar: 1-bit weights+activations multiplier 0.067 |
| 14 | Squared-ReLU expert + group-of-4 top-K input sparsity | test, low | 2-3% of reducible loss at equal active weights [D from M, 300M+] | Skips builds and lookups in proportion | Weak; group variant untested | Small gain |
| 15 | Flip-ratio logging; sweep lr x wd and lr/init | test (free) | Search cost only; 7 hyperparameters to 3 for binary CNNs under SGD [M] | None | Weak (no Adam, no ternary) | May not transfer |
| 16 | SubLN before output projections; group of 5 (243 entries) for wide matrices | test, low | +2.2 MNLI points at 0.6B [M]; 20% fewer lookups, break-even near 350 rows [D] | About 20K cycles per layer; 486-byte tables | Weak | Overlaps item 6; assumed cycle costs |
| 17 | Two-plane weights (RaBiT), 2-bit, packed binary | skip | 5.84 vs 6.55 perplexity at 7B [M] at 2x lookups | 2 lookups per 4 weights | Absent at our scale | Not shown to beat a 2x-larger ternary layer |
| 18 | MLP hash encoders (HashAttention); tanh-annealed key distillation (HAD) | skip | 75.1 vs 48.9 against LSH at 8B [M]; 0.6-0.7 GLUE points [M] | About 28K weights per encoder per head | Not applicable | No selection inside 256 tokens |
| 19 | Kumar precision law as a planning constant; gated 3-matrix FFN | skip | No data below 3 bits; FFN never ablated | +50% expert weights | None | Double extrapolation |

## 3. Recipes and settling experiments

Pilot shape and budget unless stated; two seeds resolve 0.02 nats.

**(1)-(3) Baseline ternary recipe.** One absmean scale per matrix, plain straight-through estimator, two RMSNorms per block, no biases, AdamW (0.9, 0.95), weight decay 0.05-0.1 until 2/3 of steps, then exactly 0. Peak LR x10, x20, x40 of float. Run every arm to the end of its cosine: Binarized NMT found much of the low-bit gain arrives in the final low-LR steps [M]. *Experiment:* 3x2 grid, LR by weight-decay-off, six runs; then add 8-bit activations to the winner, since Frantar's weights+activations multiplier (0.857 at 8 bits) says they are not free. Log flip ratio per tensor (item 15) in the same runs.

**(4) Kernel layout.** 128-input tiles; per tile build 32 tables of 81 int16 entries incrementally (`T[i + 3^k*t] = T[i] +/- x_k`); park 32-bit row accumulators between tiles; pack ROM tile by tile, row-major. *Experiment:* emulator cycle counts at 320x640 and 640x320, tiled vs untiled, group of 4 vs 5.

**(6) Post-matrix affine.** y_i = m_i * (row_i . x) + c_i with m, c learned per output row, stored as int16 in ROM; apply after key/query projections (before sign), router logits and the class/member head. No batch statistics. *Experiment:* with/without on the ternary baseline and on the sign-key arm. Combine with (11): per-matrix absmean vs learned per-row alpha (init at row mean |w|, gradient scale 1/sqrt(numel*Qp)) vs SEQ levels.

**(7) Scale and bias for sign keys.** Score = c_h * popcount-dot + b[h][distance]; c_h one learned scalar per head and layer (the paper's per-input statistic leaks future tokens); b a 256-entry int8 table in popcount units. Query RMS-normalised, clipped estimator. *Experiment:* 2x2 {scale} x {bias} on the pilot arm at 2.027. Target: recover 0.10 of the 0.15 gap to float keys with top-8 (1.881).

**(8) Asymmetric keys.** Store sign(k - mu), mu a per-channel ROM constant; learn a 16x4 centroid table per group of 4 key bits (16 groups at 64 dims); per query T[g][j] = q_g . c[g][j]; score = 16 lookups, used directly as the top-8 logit. *Experiment:* +/-1 scoring vs group tables vs 8-bit keys, logging top-8 overlap with exact scores; rerun the winner at 1 and 2 heads.

**(9) Shared selection.** Layers 1, 3 and last scan; layers 2 and 4 reuse the previous layer's 8 positions, either with its weights or rescored with an 8-16 bit per-layer key. Tied from step 0. *Experiment:* first log top-8 index overlap between adjacent layers of the existing pilot model (free). If above about 50%, train both variants on a 6-layer model and add a copy probe (names and numbers from earlier turns), where LiSA lost most.

**(10) Capacity probe.** Uniform random tokens, V=2048, S=64, 2 layers x 128 wide, N swept 2^10 to 2^20, train to saturation; report bits per stored weight for float, free ternary, 3:4 and a hashed-table variant. Settles the multiplier and the table-vs-width split in GPU-hours.

**(12) Float-first.** Float twin, then ternary at reduced LR, weight decay 0, KL to teacher logits at temperature 5. *Experiment:* fixed total tokens; from scratch vs 70/30 vs 90/10 on free ternary (the pilot's null used 3:4 and 20%). Drop it unless it wins by 0.03 nats.

**(13)-(14) Cheap experts.** (a) sign of the normalised expert input, 16-entry tables, 8-bit hidden; (b) squared-ReLU hidden, measure the all-zero rate of groups of 4 at the down-projection (about 41% predicted at 80% element sparsity [D]); (c) keep the top 50% of groups by max |x| with gradients passed through the mask. Compare at equal estimated cycles.

## 4. Contradictions and how to resolve them

- **Does ternary match float?** Ardakani's LSTMs do at 720K-36M weights on PTB (90.7 vs 91.5), Binarized NMT does on WMT with distillation; Spectra, ParetoQ, text8 (1.51 vs 1.46 BPC), the NMT 3B-pair corpus and our pilot show a gap. The parity cases are data-limited or distilled, where quantisation acts as a regulariser. With data in excess of capacity, plan for the gap.
- **Size of the multiplier.** NMT: 3.75x parameters for binary. Frantar: about 2.1x for binary, about 1.7x for ternary (interpolated). Kumar's extrapolation: about 4x. Ours: about 2.3x. Quantiser quality explains most of it (Kumar uses max-abs rounding). Use 2.3x, and measure with experiment (10).
- **Learning rate.** Frantar could not reproduce higher-LR tolerance (checked at 3 bits); Spectra and our pilot (x10, still improving) say otherwise. Follow the pilot.
- **From scratch or float first.** ParetoQ and BitDistill say float first; BitNet 2B4T and Spectra train from scratch; our warm start was neutral. Experiment (12).
- **1-bit activations.** NMT: a 1-bit FFN costs 0.01 with float elsewhere. Frantar: 1-bit weights and activations everywhere give a multiplier of 0.067. Position matters; confine to expert inputs and test (13).
- **Do sign keys cost anything?** BinaryAttention: no, at 6M. HAD: 5.42 points on the same DeiT-Tiny with top-N. NMT: +0.38 loss. Us: +0.15. The differences are scale, bias, a float start and symmetric vs asymmetric binarisation. Experiments (7) and (8).

## 5. What nobody has tested

- A controlled ternary-vs-float comparison below 99M parameters with a modern recipe.
- Memorisation capacity of ternary, 3:4 or hashed-table parameters.
- Ternary weights with 8-bit activations, sign keys, top-8 selection and hard routing together; each paper quantises one piece.
- Sign or hashed keys trained from scratch in a causal LM; selection at 3% density (sparsest reported: about 12%).
- Attention sharing in fewer than 12 layers, or with top-k or quantised attention.
- The 90/10 split at a budget of a few billion tokens.
- Any lookup kernel on a scalar CPU; a ternary output head.

## 6. New ideas (none of the sources does these)

1. **Context-indexed attention bias.** Index the bias by (distance, class of the query token): 16 KB per head per layer at 64 classes, still one load and one add per position.
2. **Layer-0 keys and queries from ROM.** They depend only on the token and n-gram hash, so store them as a 16K x 8-byte table per head and skip the projection.
3. **Two-stage selection.** Scan a 32-bit signature (4 table reads per position), keep 24-32 candidates by histogram threshold (scores take 33 values), rescore those with group tables; train the signature with HashAttention's recall-targeted BCE loss. About 4x cheaper scan if recall holds.
4. **Router-conditioned everything.** The hard route that picks an expert also picks the key centroid tables, the bias table and the post-matrix gains: ROM only, zero cycles.
5. **Group top-K sized to the frame budget.** Train with a random keep-fraction so the cartridge can drop low-energy input groups when a token runs late, and run the predictive-typing pass at a lower fraction.
6. **Repay the ternary deficit in ROM.** At 1.585 bits per weight, 100M weights hold at most 158 Mbit against about 446 Mbit for GPT-2 small [D], so add table rows and experts before width and target a narrower distribution than GPT-2's.
7. **Mixed kernels per matrix**, chosen by the offline packer: group of 5 for 640-row up-projections, popcount rows for sign-input experts if (13) holds.

## 7. Sources

Only ParetoQ had an independent verification pass ("No" = not independently verified). "Digest" means read through a summarising fetcher: table cells were not eyeballed.

| Source | URL | Read | Verification |
|---|---|---|---|
| ParetoQ | https://arxiv.org/abs/2502.02631 | primary (paper + repo) | Verified: table numbers and 90/10 claim confirmed; saturation and embedding claims partly confirmed (plots only) |
| BitNet b1.58 2B4T | https://arxiv.org/abs/2504.12285 | primary (paper + repo) | No |
| BitNet Distillation | https://arxiv.org/abs/2510.13998 | primary, digest | No; equations unseen |
| Self-Indexing KVCache | https://arxiv.org/abs/2603.14224 | primary, digest | No |
| microsoft/BitNet | https://github.com/microsoft/BitNet | primary (repo); companion papers via digest | No |
| T-MAC | https://arxiv.org/abs/2407.00088 | primary, digest | No |
| Spectra (TriLM) | https://arxiv.org/abs/2407.12327 | primary | No; gap figures derived from fits |
| RaBiT | https://arxiv.org/abs/2602.05367 | primary, digest + repo function | No |
| Compression Scaling Laws | https://arxiv.org/abs/2502.16440 | primary | No; ternary multiplier interpolated |
| Hamming Attention Distillation | https://arxiv.org/abs/2502.01770 | primary, digest | No |
| BinaryAttention | https://arxiv.org/abs/2603.09582 | primary, digest + repo | No |
| HashAttention | https://arxiv.org/abs/2412.14468 | primary (paper + code) | No |
| Q-Sparse | https://arxiv.org/abs/2407.10969 | primary, digest; BitNet a4.8 figures secondary | No; headline is curves only |
| Shared Attention | https://arxiv.org/abs/2407.12866 | primary | No |
| LiSA | https://arxiv.org/abs/2408.01890 | primary | No; eval-loss figure unread |
| Binarized NMT | https://arxiv.org/abs/2302.04907 | primary (text and tables) | No |
| Recurrent Binary/Ternary Weights | https://arxiv.org/abs/1809.11086 | primary, digest | No |
| How much do LMs memorize? | https://arxiv.org/abs/2505.24832 | primary | No; ternary figures are my arithmetic |
| Scaling Laws for Precision | https://arxiv.org/abs/2411.04330 | primary | No; per-bit factors use an inferred offset sign |
| Weight-magnitude Hyperparameters | https://arxiv.org/abs/2303.02452 | primary | No |
| BitNet b1.58 Reloaded | https://arxiv.org/abs/2407.09527 | secondary here (earlier note) | No |
| Pilot results | `research/pilot-results.md` | own measurement | One seed for most arms |
