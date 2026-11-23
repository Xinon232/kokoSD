# Lookup tables and routed experts in a fixed 32 MB ROM (focused round, final)

Cluster report, 18 Nov 2026. It covers 18 sources (notes in `research/notes/f-tables-*.md`), read against our pilot (`research/pilot-results.md`: 128 wide, 4 layers, 4K vocabulary, 491M TinyStories tokens, seed noise about 0.005 nats) and the emulator benchmark (`research/bench-results.md`).

Tags: **[M]** measured by a source, **[P]** measured in our pilot or benchmark, **[D]** derived by me or a note author, not measured.

## 1. Bottom line

1. **Our pilot is the only evidence near our scale, and it is positive for both mechanisms.** Hashed bigram+trigram tables gave -0.13 nats at 8 MB, 16 hard-routed experts -0.19, both together -0.25 [P]. No source tests tables below a 0.56M dense core or experts below 41M. Evidence: moderate (n-gram-friendly corpus, 1-2 seeds, not ternary).
2. **Tables need about 50 training tokens per table parameter.** Engram worked at 46-62, was called under-trained at 14, and a replication at 3.5 gained nothing on test [M; ratios D]. Lngram ran at about 110, our pilot at about 58 [D]. Evidence: weak-moderate, inferred across unlike setups.
3. **Tables should be the smaller share of ROM.** Engram's allocation basin is flat at 5-30% of sparse parameters, with the best measured point at 15% [M, verified; the paper's "20-25%" is a fitted minimum]. LongCat: at most 50%, and experts win above a 30x stored/active ratio at 280M [M, read-offs]. Ours is 33-80x. Evidence: moderate in direction, none at our scale.
4. **More stored experts at fixed active compute buy quality.** 16 to 128 experts gave -0.11 nats at 47M [M, verified, single seed, authors still see a gap to dense]. Evidence: moderate, 100x our compute.
5. **Token-id routing is a serious replacement for the learned router, but our case is untested.** It beat Switch at 1x64 (23.16 vs 23.65 PPL) and lost slightly at 5x16 (23.21 vs 23.00) [M, verified]. It would return 5.6% of the cycle budget [P]. Evidence: mixed.
6. **A second, output-side table is the main new finding of this round.** A count or teacher-filled n-gram table added as a sparse logit bias gave 4.1% fewer bits on a weak online model [M, StateSMix], and a converted table recovered about 70% of a neural LM's WER gain [M, Arisoy, secondary read]. Stored exactly, it costs 2-3 bytes per n-gram [M, tongrams, word vocabularies] and is also the typing engine. Evidence: weak for a well-trained model; no source combines it with a transformer of our kind.
7. **Keep orders 2 and 3; add gapped keys before longer contiguous ones.** 4-grams cost 0.0017 nats at 0.56B [M, verified]; Lngram's {2,3} beat other sets [M, one seed]; our bigram+trigram beat either alone [P]. Skip-gram features alone nearly equal a 5-gram model (69.8 vs 70.8 PPL on 1B words) and stack with it (54.2) [M, count models]. Evidence: moderate for a count model, untested inside a transformer.
8. **Hash and row format are engineering risks with known answers.** LongCat's polynomial hash is degenerate for power-of-two vocabulary and tables [D, from its code; consistent with its Fig. 3b]. Rows survive 2 bits per parameter post-training at 31B+ parameters [M, unverified, read via summarising fetch]; esp32-ai kept its gain at int4 [M, verified].

## 2. Ranked recommendations

GBA costs are [D] unless marked. One 256-input ternary row costs about 465 cycles [P].

| # | Technique | Action | Expected gain (source scale) | GBA cost | Evidence | Main risk |
|---|---|---|---|---|---|---|
| 1 | Hashed bigram+trigram input tables, zero-init, own optimizer group | adopt | -0.13 nats at 8 MB [P, 128 wide]; -0.040 at 0.56B [M] | 2-4 row reads, 250-1,000 byte adds (<5K cycles); RAM last 2 ids | Moderate | TinyStories flatters n-grams; null test result in one small replication |
| 2 | Hash hygiene: multiply-xor-shift, odd multipliers, power-of-two tables, independent hash per table | adopt | Avoids 130 to 40,000 collisions at bad sizes [M, read-off] | 2-3 multiplies per key | Derivation + one figure | None known |
| 3 | 64 experts per layer instead of 16 | adopt | -0.11 nats for 8x experts at 47M [M]; -0.19 for 16 [P] | ROM only: 55M weights = 13.8 MB at d256 x6 [P] | Moderate | 4x fewer tokens per expert than the pilot |
| 4 | Size tables by training tokens (about 50 per parameter); tables at most half of ROM | adopt | Separates working from null runs [M, D] | None | Weak-moderate | Inferred rule |
| 5 | Typing metric: top-1/top-3 word recall, special tokens masked, n-gram-only baseline | adopt | Reference: n-gram 13.0/22.1%, 1.4M LSTM 16.5/27.1% [M, uncontrolled baseline] | None | Good as a metric | Word-level vs our subwords |
| 6 | Token-id routing (balanced table, different per layer) | test | +0.5 PPL at 1x64, -0.2 at 5x16 [M, 222M base] | Saves about 180K cycles/token (5.6%) [P]; 96 KB ROM | Mixed | All-layers-sparse untested |
| 7 | Output-side exact n-gram table: sparse logit bias + candidate list | test | 4.1% bits over network alone [M, weak LM]; 70% of NNLM WER gain [M, secondary] | 1 probe + up to 8 adds per order (<1K cycles); 2-3 bytes/gram, 8-10 MB for 3-4M grams | Weak | Redundant with input tables; ROM competes with experts |
| 8 | Row shape and precision at fixed bytes: narrow-many vs wide-few; 8-bit, 4-bit, 2-bit trellis, ternary QAT | test | 2-bit: KL 41-63% below 2-bit rounding at 31-197B [M]; narrow+sign: 1.1% time at 768 wide [M] | 2-bit decode 3-5K cycles per 256-dim row; sign = conditional negate | Weak | Trellis is post-training only; modded-nanogpt later returned to wide rows |
| 9 | Gate at equal ROM: none / static importance per row / context gate | test | Context gate 0.0046 of 0.040 [M]; -0.03 at double storage [P]; gating failed in modded-nanogpt [M] | Static: shift on unpack; context: about 10K cycles | Conflicting | Gate bytes may be better spent on rows |
| 10 | Gapped keys (t-2 with t-1 skipped; t-3,t-1) sharing a table by salt | test | Skip-only 69.8 vs 5-gram 70.8 PPL; -17% PPL at 44M tokens [M, count LM]; rule agreement 65.6 to 78.0% at 7 tokens, 52.2 to 54.3 at 3 [M, oracle] | One row per key | Weak-moderate | Attention may already cover it |
| 11 | Multiplicative per-layer token row inside the expert (PLE) | test | +0.098 nats at 0.56M core, 32K vocab [M]; additive version -0.08 [P] | G 8-bit multiplies per layer (about 2K cycles at G=256); ROM 16K x 6 x G | Moderate | Faithful plumbing costs 16-27% of budget |
| 12 | MultiHash experts | test | 0.1-0.3 PPL at 32 experts, non-monotone at 16 [M] | N byte lookups; ROM unchanged | Weak-moderate | Small effect |
| 13 | K=2 or 4 smaller experts at equal active units | test | 0.026-0.043 nats at 47M [M, confounded: top-1 arms had 4 or 32 experts] | 2 active experts measured at +44% cycles [P] unless leaves are halved | Weak | Leaves under 128 units degrade [M] |
| 14 | Teacher-filled tables (sample teacher text, then count) | test | Sampled-text n-grams: about 15% lower PPL when interpolated [M, Krishnan]; probability rewriting clearly worse (632.9 vs 369.9) | Offline only | Weak | Teacher style leaks |
| 15 | Latent keys from hidden sign bits (Lngram v1/v2), 16 routes | test, low priority | -0.03 nats at 0.6B, one seed; 16 routes within 0.006 of 192 [M] | Folded sum about 65K cycles/layer; v2 attention readout about 300K | Weak | Plain STE diverges (7.33 vs 2.86); no token-id baseline |
| 16 | N-Grammer cluster bigrams; tree router; PKM in every layer; 4-grams; Engram conv; SNM as the whole model; Lngram v2 readout; canonical ids (0.0057 nats at 128K vocab, little to merge at 16K) | skip | 5 of 12 N-Grammer configs no better than baseline; PKM worse than dense at 47M; RNN beats SNM by 13% at 44M tokens [M] | - | - | - |

## 3. Recipes and settling experiments

All experiments reuse the pilot rig with 2 seeds, a 16K vocabulary where collision load matters, and a held-out conversational set (the Engram replication gained on validation, not test [M]).

**1, 2, 4. Input tables (adopt).** Index = ((m1*t0) xor (m2*t1) xor (m3*t2)) * odd constant, top bits, per-table multipliers [D; modded-nanogpt's hash uses a modulo we lack]. Zero-init rows; Adam, no weight decay; learning rate 5x base (Engram) to 75x (modded-nanogpt) [M], so sweep 5/20/75. Add before block 1 and again before block 2 or 3 with a per-layer scalar initialised at 0.1 [M], folded into ROM at export. Scale the summed embedding by sqrt(d), a shift (LongCat: -0.02 nats [M]). Reset keys at turn boundaries.
Sizing [D]: d256 x6 with 64 experts takes 13.8 MB [P], leaving 14-16 MB. At 64 dims x 4 bits that is about 500K rows, 32M parameters, wanting about 1.6B training tokens.

**3. 64 experts (adopt).** Keep the pilot's hard top-1 with load-balance loss (usage entropy 0.96-0.99) [P]. Initialise by total stored width, not one expert's (0.02-0.2 PPL at 47M) [M]. Experiment: 16 vs 64 experts at equal tokens.

**5. Typing metric (adopt).** Top-1 and top-3 word recall over all words, special tokens masked [M]. Arms: ROM table alone, Koko alone, table first then Koko.

**6. Token-id routing (test).** Per layer: tokens in descending frequency, each to the currently emptiest expert; different table per layer (sharing costs 0.5 PPL); do not cluster similar tokens (0.7 PPL worse) [M, verified]. Arms at 64 experts in all 4 layers: learned router, token table, token table with the saved cycles spent on width. Adopt if within 0.01 nats.

**7. Output-side table (test).** Offline: count 2-4-grams on the training corpus, keep contexts above a count threshold, store per context a successor list of (token id, 4-8-bit log-ratio bias) in a sorted trie (frequency-ranked ids, Elias-Fano levels) [M for layout; D for sizes]. Run time: longest-first probes; add each stored bias to the member logit, plus a per-context class bias for the two-level head; scale by a learned per-order constant, since StateSMix's hand-set weights are untrustworthy. Train the transformer with the frozen table in the loss (residual training; Arisoy's best row was table plus neural [M, secondary]). Experiment: 8 MB input tables vs 4 MB input + 4 MB output; record loss, top-3 recall and bytes per gram on our tokens.

**8. Row shape and precision (test).** At fixed 8 MB: 64K x 128, 128K x 64, 256K x 32 dims; narrow rows added into the first d_row residual dims without projection; optional fixed sign vector from an 8,192-row pool indexed by an independent hash [M]. Precision arms: 8-bit, 4-bit, ternary QAT, and 2-bit trellis applied post-training with one 16-bit scale per row [M]. Decide by loss per ROM byte.

**9. Gate (test).** Equal-byte arms: no gate; one importance value per n-gram id quantised to {0, 1, 2, 4} (Hash Embeddings [M], adapted [D]); context gate sigmoid(norm(h) . norm(row)) with key tied to value [D]. The pilot's gated arm used 17.4 vs 9.4 MB and does not answer this.

**10. Gapped keys (test).** Add (t-2, skip, t0) and (t-3, skip, skip, t0)-style keys, position-aware, into the existing tables with a template salt [D]. SNM's working template set is dense short gaps plus single-remote-word long gaps to 10 [M]. Experiment: same bytes, 2 contiguous keys vs 2 contiguous + 2 gapped.

**11-13.** PLE: h = act(W1 x) * row[token, layer], row 8-bit centred on 1, vs additive token rows at equal ROM. MultiHash: N in {1, 2, 4} on the token-table arm. K: (K, G) in {(1, 256), (2, 128), (4, 64)} at 64 experts; keep K=1 unless the gap exceeds 0.02 nats under ternary training.

**14.** Sample 1-2B teacher tokens, count, merge with corpus counts for the output table; compare with corpus-only counts.

**15.** Only after 7-10: 16 routes of 4 bits, one-bit surrogate gradient, folded gated sum, against token-id tables at equal bytes.

## 4. Contradictions and how to resolve them

- **Tables or experts for spare ROM.** LongCat favours experts at high sparsity and narrow width; Engram favours a 5-30% table share; our pilot shows they stack. Resolve with a fixed-ROM sweep of table share 0/25/50%.
- **Gating.** Helps in Engram (multi-branch backbone, separate key projection), failed in modded-nanogpt, helps in the pilot only at double storage. Resolve with item 9.
- **Row width and precision.** Modded-nanogpt went narrow, then wide on a larger table; esp32-ai saturates near P=256 [M, single seed]. The ternary round found ternary embeddings fail at large scale, while 2-bit trellis and int4 rows hold post-training [M]. Resolve with item 8; default to 4-bit.
- **Per-layer or input injection.** esp32-ai: per-layer wins by 0.046 (control not clean [verified: partly]); LongCat: input n-grams beat per-layer token rows at equal parameters; pilot: one site and two sites tie. Use two sites; it is nearly free.
- **Learned or fixed routing.** Hash Layers favours tables at 64+ experts in one layer; Csordás favours learned sigmoid routing with K=4, but its top-1 arms do not test 1-of-64. Resolve with items 6 and 13.
- **Is an n-gram table still useful beside a neural model?** StateSMix and Arisoy say yes; the Gboard LSTM beat a similar-size 5-gram by about 25% relative [M, uncontrolled], and RNNs beat SNM at 44M tokens. Compatible: the table supplements, it does not substitute.
- **How to fill a table from a teacher.** Arisoy rewrites probabilities; Krishnan finds sampling and counting clearly better for a GPT-2 teacher. Follow Krishnan.

## 5. What nobody has tested

- Any table or routing result with ternary weights or quantisation-aware training.
- 64 experts in every layer of a 4-6 layer model, with either router.
- Table share against expert count at fixed ROM bytes and fixed cycles.
- Input-side and output-side tables together: do they stack or duplicate?
- Hashed tables at our collision load: 16K vocabulary, dialogue, 0.25-1M rows.
- Exact n-gram storage for subword vocabularies at 1-10M grams (all figures are for word vocabularies).
- Gapped keys as embedding keys inside a transformer.
- Typing recall for a subword model against an n-gram table on the same text.

## 6. New ideas (none of the sources does these; all [D])

1. **One trie, three uses.** The exact output table supplies logit biases, the typing suggestions, and a candidate shortlist so the head expands only classes that contain a listed successor.
2. **Confidence-gated skipping.** When a long context's stored successor list is near-deterministic (a ROM bit), emit from the table and run a shallow path. This turns ROM directly into tokens per second; the stored bit needs calibration against the full model.
3. **Per-token hidden-unit masks.** A ROM bit mask per (token, layer) selects which half of a doubled expert's rows to compute: MultiHash at single-row granularity, at constant cycles.
4. **Two-stage routing.** The token table picks a group of 8 experts; an 8-row learned router picks one (8 rows instead of 64).
5. **Attention sign keys as memory addresses.** Lngram v2 shows 64 address bits suffice at d=768; our attention keys already provide such bits with no extra projection.
6. **Coverage-directed generation.** Have the generator LLM oversample contexts whose table rows are under-visited, attacking the 50-tokens-per-parameter limit.
7. **In-conversation recency table in slow RAM.** A 64-entry "last successor of this bigram" cache (StateSMix's LZ predictor) gives name and phrase repetition to both the model and the keyboard for a few KB.

## 7. Sources

| Source | URL | Read | Verification |
|---|---|---|---|
| Engram (Cheng et al.) | https://arxiv.org/abs/2601.07372 | primary | 3 claims confirmed, allocation optimum partly |
| Hash Layers (Roller et al.) | https://arxiv.org/abs/2106.04426 | primary | 4 confirmed; directional only |
| esp32-ai (slvDev) | https://github.com/slvDev/esp32-ai | primary | 2 confirmed, 2 partly |
| Approximating two-layer FFNs (Csordás et al.) | https://arxiv.org/abs/2310.10837 | primary, via summarising fetch | 2 confirmed, 2 partly |
| LongCat-Flash-Lite (Liu et al.) | https://arxiv.org/abs/2601.21204 | primary; figure read-offs | not verified |
| modded-nanogpt PR 299 / 201 | https://github.com/KellerJordan/modded-nanogpt/pull/299 | primary for diffs and statistics; PR text secondary | not verified |
| N-gram statistics (Nguyen) | https://arxiv.org/abs/2407.12034 | primary | not verified |
| Lngram | https://arxiv.org/abs/2605.24869 | primary | not verified |
| Lngram v2 | https://arxiv.org/abs/2609.03426 | primary, via question-answering fetch | not verified |
| N-gram memories at two bits (Han et al.) | https://arxiv.org/abs/2610.21810 | primary, via summarising fetch; some details uncertain | not verified |
| SNM skip-gram LM (Shazeer et al.) | https://arxiv.org/abs/1412.1454 | primary | not verified |
| StateSMix (Tacconelli) | https://arxiv.org/abs/2605.02904 | primary | not verified |
| NNLM to back-off LM (Arisoy et al.) | https://www.research.ibm.com/publications/converting-neural-network-language-models-into-back-off-language-models-for-efficient-decoding-in-automatic-speech-recognition | secondary (patent text; paper paywalled); Wang 2013 and Krishnan 2023 read directly | not verified |
| Massive n-gram datasets (Pibiri, Venturini) | https://arxiv.org/abs/1806.09447 | primary | not verified |
| Federated keyboard prediction (Hard et al.) | https://arxiv.org/abs/1811.03604 | primary | not verified |
| Fast Feedforward Networks | https://arxiv.org/abs/2308.14711 | primary; figures unread | not verified |
| N-Grammer (Roy et al.) | https://arxiv.org/abs/2207.06366 | primary | not verified |
| Hash Embeddings (Svenstrup et al.) | https://arxiv.org/abs/1709.03933 | primary, via summarising fetch | not verified |
