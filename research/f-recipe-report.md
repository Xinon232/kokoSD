# Cluster report F-recipe (final): training recipe and architecture details that raise quality per unit of inference compute

19 Nov 2026. Based on 18 source notes (`notes/f-recipe-01` to `-18`), two independent verification passes (nanochat log, adaptive softmax) and our pilot (`pilot-results.md`).
Tags: **[M]** measured by the source, **[P]** our pilot, **[I]** my inference or arithmetic. Sources use different units (bits per byte, nats, perplexity, accuracy); their deltas do not convert to pilot nats.

## 1. Bottom line

1. **No source tests our operating point.** The closest are 10.9M parameters at d=384 (one seed), 17.7M at d=256 and a 6M vision model; none is ternary, causal with sign keys, or uses a 16K factorised head. Evidence: certain.
2. **The factorised head is forced by compute and costs quality at equal training tokens:** +0.0115 BPB (d=384) and +0.0101 (d=128), concentrated in rare tokens [M, 1 seed, 50k vocabulary, not peer reviewed]. Inside it, cluster quality dominates: k-means beats random by 0.0572 BPB, a short-list adds 0.0107, full-width tails add about 0.010, cluster count is noise [M]. Evidence: moderate (two unrelated sources agree on direction).
3. **The pilot's 0.15-0.17 nat sign-key loss [P] has one candidate fix with small-model evidence:** a per-head scale plus an additive relative-position bias took sign-QK attention from 71.95 to 72.88 top-1 against 72.2 float on a 6M model; the bias alone gave +0.44 [M, one seed, vision, fine-tuned from float, absolute positions kept]. Evidence: weak for language; cost is one load and one add per scored position.
4. **Hard top-1 routing is supported at fixed active compute, and the router need not be learned:** 4 experts top-1 beat the active-matched dense model by 10-14% perplexity; a router frozen at initialisation matched a learned one (9.810 vs 9.907) [M, GPT-2 small, one seed, at most 6 experts]. Our pilot agrees (-0.19 nats, 16 experts) [P]. Evidence: moderate on sign, none for 64 experts.
5. **Per-layer x0 scalars are the only lookup-style addition with a size-resolved number, and it grows as the model shrinks:** -0.0103 BPB at width 512, -0.0039 at 1280 [M, verified; best-of-LR-sweep against one baseline]. Evidence: moderate.
6. **Do not put attention first; two attention layers in the middle or end are probably enough.** Front placement erased the hybrid gain at 1B (2.750 vs 2.735 NLL) [M]; LFM2-350M starts with two conv layers [M, config]; a d=384 four-block hybrid reached 5.95 perplexity against 6.93 all-attention [M, one seed]. But a pure diagonal recurrence scores at chance on entity tracking (18.8, chance 20) [M]. For us the gain is RAM (64-96 KB per attention layer against under 1 KB), not cycles [I]. Evidence: moderate on direction, weak on size.
7. **Muon is worth 4.8-7.0% test loss at equal data at exactly our widths** [M, 5 paired seeds, 20M-token runs], with about double the tuning regret, untested with ternary latents or routed experts. Evidence: moderate for float, none for our case.
8. **Removing run-time normalisation and adding early exits are not worth pursuing for speed.** Frozen-scalar norms cost +0.067 CE at GPT-2 small for about 2% of our cycles [M; cycle estimate I]; mixture-of-exits uses about 1.67x FLOPs per token and was 8.3% slower at 6 layers even on a GPU [M]. Evidence: moderate.

## 2. Ranked recommendations

| # | Technique | Action | Expected gain (source scale) | GBA cost [I] | Evidence | Main risk |
|---|---|---|---|---|---|---|
| 1 | Two-level head: root short-list + balanced k-means tail clusters, full-width tail rows, free cluster rows | adopt | k-means vs random -0.0572 BPB; short-list -0.0107; full width -0.010 [M, d=128, 50k vocab] | About 211K weights, 337K cycles per sampled token at d=256 (K=512, 250 clusters of 62); 1.7 KB RAM; about 1 MB ROM | Moderate | +0.0115 BPB against flat; rare tokens +0.24 to +0.36 bits |
| 2 | Per-layer x0 and residual scalars | adopt | -0.0103 BPB at width 512 [M, verified] | One scaled add of d values per layer (about 3K cycles, or shifts); d bytes RAM | Moderate | Untested below width 512 |
| 3 | ReLU^2 MLP, logit soft cap, log-class-size bias initialisation | adopt | SwiGLU worse at two sizes [M, nanochat]; rest is hygiene | One table read per hidden unit; one constant per class | Weak, cheap | SwiGLU helped in MobileLLM (section 4) |
| 4 | Sign keys with per-head constant scale + distance-bias table | test first | +0.47 (scale) and +0.44 (bias) top-1 at 6M [M, one seed, vision] | 256 int8 per head per layer (about 6 KB ROM); one load and one add per scored position; no RAM | Weak | Causal, from-scratch, language all untested; source also keeps absolute positions |
| 5 | Router: frozen random sign-bit projection instead of learned | test | Frozen 9.810 vs learned 9.907 ppl [M, N=4, top-2] | About 1.9K weights per layer against 20K (saves about 30K cycles per layer) | Weak for 64 experts | Pilot needed a load-balance loss; frozen routing cannot be balanced |
| 6 | Hybrid layout: cheap mixer in layers 1-2, attention in two later layers | test | PPL 5.95 vs 6.93 at d=384 x 4 [M, 1 seed]; 2.735 vs 2.750 NLL at 1B [M] | Saves 64-96 KB slow RAM and one key scan per replaced layer; mixer 1-4 d^2 weights | Moderate direction | Recurrence shows no binding or copying [M]; late-training decline in recurrent models |
| 7 | Narrow shared key/value heads | test | 16 to 4 KV heads: 44.6 to 44.7 accuracy at 125M [M, single runs] | K and V projections 131K to 33K weights per layer; ring RAM 4x smaller | Weak | 2 and 1 KV heads lost about 1 point |
| 8 | Value embeddings on alternating attention layers + layer-1 value residual | test | "Helps a lot" [M, no number]; +0.032 BPB [M, observational, n=41] | One ROM row and d adds per layer; 0.8-4 MB per table; no RAM | Weak | Shrinking failed in the source; may duplicate our n-gram tables |
| 9 | Muon on matrices, AdamW elsewhere | test | -4.77% / -7.01% test loss, d=256 / 384 [M, 5 seeds] | None | Moderate (float) | Straight-through interaction; 64 sparse experts; tuning regret 13.7% vs 7.5% |
| 10 | Learned candidate lists over a flat head (L2S) as alternative to item 1 | test | 10.6-45.3x softmax speed-up at P@1 0.989-0.998 [M, LSTMs; list sizes not reported] | 0.16-0.26M cycles (1.5-2.5x the class head); 131 KB lists + 1 MB matrix ROM | Weak | Out-of-list tokens get zero probability; list size at 16K unknown |
| 11 | Online context cache over the ring buffer as a logit tilt | test | +0.080 BPB, n=23 [M, observational, long documents] | 1-2K entries x 3 bytes slow RAM; one probe, one add | Weak | Chat repeats less than web documents |
| 12 | Stored top-32 teacher logits with the decoupled loss | test | No ablation [M, LFM2]; plain logit distillation -0.1 to -0.3 points at 3x cost [M, MobileLLM] | None | Conflicting | Capacity gap |
| 13 | 32K vocabulary; weight averaging; masked next-token prediction; frozen-scalar norms | test, low priority | 1.1621 vs 1.1875 BPB (50k two-level vs flat 8k) [M]; +0.025 BPB n=109 [M] but neutral in nanochat; causal gain figure-only; +0.067 CE loss [M] | None, or about 70K cycles saved for norms | Weak | Each could be noise at our scale |
| 14 | Mixture-of-exits; variance-corrected cluster sampling; d/4 tails; tied embeddings; layer sharing; depth recurrence; faithful Mamba or mLSTM; pure recurrent model; learned layer weights; attention gate; multi-token prediction | skip | Unaffordable (8.4M multiplies; 390K extra weights for exits), optimises parameters not compute, or neutral/negative | - | - | - |

## 3. Recipes and settling experiments

All experiments use the pilot harness at the 192-wide ternary package (491M tokens), two seeds. Pilot seed spread is 0.002-0.008 nats [P]; count only effects above about 0.015 nats.

**1. Head (adopt).** Train a flat d=128 proxy for about 50M tokens; run balanced k-means on its embedding rows for tokens outside the K most frequent targets. Root softmax over K short-list rows plus one free row per cluster; tail probability = p(cluster) x softmax over full-width member rows. Start at K=512, 250 clusters. Initialise cluster biases to log cluster unigram mass [I]. Use one fixed sampling temperature: tempering both levels differs from tempering the flat softmax [M, Bendada]; a trained class head needs no size correction. For typing, expand clusters in decreasing p(cluster) and cap at 3-4; exact greedy needed 10.0 on average [M]. *Experiment:* flat (upper bound), random clusters, k-means, k-means plus short-list at K = 256 / 512 / 1024; report loss by frequency band and clusters expanded for top-3.

**2. x0 scalars (adopt).** Before block i: x = r_i x + a_i x0. Initialise r 1.15 to 1.05 and a 0.20 to 0.05 across depth; r needs about 100x lower learning rate; scalar beta1 0.96 [M]. *Experiment:* on / off / rounded to powers of two.

**4. Sign-key repair (test first, because the pilot loss is large).** Score = c_h x popcount-agreement + B[head][distance], distance 0-255, B initialised at std 0.02 and excluded from weight decay [M]. Replace the source's per-input mean-|x| scale with a learned per-head constant c_h, since the statistic leaks future tokens in causal training [I]. Store B in popcount units so top-8 selection stays integer; apply c_h only in the 8-way softmax. Sign gradient passes only where |x| <= 1 [M]. *Experiment:* sign keys + top-8 with {no extras, scale, bias, both} and both with 8-bit keys as reference; success is recovering at least 0.10 of the 0.15 nats.

**5. Frozen router (test).** Six fixed random ternary rows; their sign bits index 64 experts. Drop the gate-probability output scale [I]. *Experiment:* learned argmax vs frozen sign-bit vs token-id hash routing, at 16 and 64 experts; report usage entropy. Also test dense MLPs in layers 1-2, as LFM2 does [M].

**6. Layout (test).** Six layers, equal active weights: attention everywhere; mixer in 1-2 with attention in 3 and 5; mixer in 1, 2, 4, 6. Mixer A: LFM2 gated conv, (B, C, h) = Linear(x), y = B * h, causal depthwise conv kernel 3, out = Linear(C * z), 4 d^2 weights. Mixer B: s_t = a_t * s_(t-1) + b_t * c_t with retention bias initialised at 2.0 and 16-bit state, coupled gates (a = 1 - b), one d^2 output projection. *Experiment:* loss plus a name-copy probe over 256 tokens, checkpointed through training because recurrent models declined late [M].

**7. Narrow KV (test).** 8 query heads on 2 key/value heads of 32 dims: 16 + 2 KB ring per layer instead of 64 + 8 KB. *Experiment:* KV heads 8 / 4 / 2, crossed with the winner of item 4.

**8. Value embeddings (test).** Vocabulary x value-width table on alternating attention layers, last always; gate folded into the ROM row [I]. Value residual: v_l += c_l v_1. *Experiment:* none / residual / float / 8-bit / ternary table, with n-gram tables present.

**9. Muon (test).** Matrices: learning rate 0.01, momentum 0.90, weight decay 0.10, 5 Newton-Schulz steps (3 if the optimiser dominates step time); AdamW 3e-4, beta2 0.95 elsewhere; clip 1.0, 5% warm-up, cosine to 0.1x; five learning-rate candidates. *Experiment:* AdamW vs Muon on float, ternary, and ternary plus 16 experts; sweep learning rate for both, since ternary still improved at 10x [P].

**10. L2S (test, after item 1).** On a flat-head model, dump hidden states and top-k labels; 128 cluster rows initialised by spherical k-means; lists built by greedy knapsack under an average-size budget, with a hard per-cluster cap [I]. *Experiment:* measure the list length needed for P@5 0.98 at 16K; if under about 300 rows, compare sampled-text quality and typing recall against item 1.

**11-13.** Cache: slow-RAM table from hash of the last two tokens to the last successor, a learned constant added to its logits; evaluate on multi-turn dialogue, repeated vs novel tokens. Distillation: cross-entropy + Bernoulli KL on the teacher's top-32 mass + tau^2-weighted KL inside the set, temperature only in the last term [M]; labels-only control at equal steps.

## 4. Contradictions and resolutions

- **Factorised head better or worse than flat?** Better at equal training FLOPs (-0.0137 BPB), worse at equal tokens (+0.0115) [M, same paper]. We are in the equal-token case.
- **Class head vs learned screen.** L2S reports adaptive softmax at only 1.9-3.2x against its 10.6-20.4x [M], but with a frozen model and unreported list sizes; the factorised study trains the head jointly. Not comparable; item 10 decides.
- **Routing balance.** A frozen router matches a learned one [M], and a balance loss made no difference (10.674 vs 10.667), but that was sequence-level routing; our pilot used a balance loss with hard token routing [P]. Test before removing it.
- **Recurrent mixers.** Hybrid best at 10.9M [M, one seed]; pure Mamba best at 100M [M]; pure recurrence wins grammar but sits at chance on tracking [M]; transformers best in 432 controlled runs [secondary, abstract only]. Agreement only on "not attention first, never recurrence alone".

## 5. What nobody has tested

- Sign-key attention that is causal, trained from scratch, on language, with top-k selection, with or without a distance bias.
- Top-1 routing beyond 8 experts with a frozen or hashed router; token-id-keyed routing at all.
- Any factorised head with ternary rows, a vocabulary under 50k or a model under 12M parameters; candidate-list sizes for a subword vocabulary.
- Value embeddings with quantised tables; Muon with straight-through ternary training or sparse expert matrices.
- A diagonal recurrence with ternary projections and 8-bit gates; attention placement in a 4-6 layer model (all ablations are at 13 or more layers).

## 6. New ideas (none tested by any source)

1. **Token-conditioned position bias.** Index the attention bias by (distance bucket, hash of the query token): a ROM row per token class, still one load and one add per position.
2. **Routing from ROM.** Pick the expert from a table keyed on token id or bigram hash, so routing costs zero dense compute and the expert's ROM address is known one token early. The frozen-router result makes this plausible; the risk is losing context-dependence.
3. **Layer-0 mixer as a lookup.** If layer 0 is a gated conv fed only by token-determined inputs, store y = B * h and C per token; the layer becomes three row reads plus one d^2 projection instead of 4 d^2.
4. **Context-keyed candidate lists.** Replace L2S's 128 computed cluster rows with a list keyed on the previous-token hash, unioned with the root short-list; this moves likely rare tokens into the scored set, where the measured rare-token penalty lives.
5. **Flat-teacher, factorised-student.** At 16K the flat distribution is affordable in training; distil a flat-head twin into the two-level head, weighting tail clusters.

## 7. Sources

| Source | URL | Read | Verification |
|---|---|---|---|
| Two-level softmax sampling (Bendada, Salha-Galvan 2026) | https://arxiv.org/abs/2610.10483 | primary | not verified |
| Factorised output layers (Michałek 2026) | https://www.alphaxiv.org/abs/2611.factorised-output-layers-small-language-models | primary (paper text and script from author's repository); not peer reviewed | not verified |
| Parameter Golf: What Really Works? | https://arxiv.org/abs/2607.01517 | primary via summarising fetch only | not verified |
| nanochat dev/LOG.md | https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LOG.md | primary | verified: 1 claim confirmed, 3 partly |
| Hybrid architectures (Bae et al.) | https://arxiv.org/abs/2510.04800 | primary via extraction tool; figures unread | not verified |
| LFM2 technical report | https://arxiv.org/abs/2511.23404 | primary, plus config and code | not verified |
| BLaLM (Haller et al.) | https://arxiv.org/abs/2511.05560 | primary; table errors found | not verified |
| Attention, Dynamic Wave and hybrid mixers | https://arxiv.org/abs/2610.39854 | primary via summarising fetch | not verified |
| When Does Muon Actually Win? | https://arxiv.org/abs/2610.21555 | primary (PDF) | not verified |
| Adaptive softmax (Grave et al. 2017) | https://arxiv.org/abs/1609.04309 | primary | verified: 2 claims confirmed, 2 partly |
| MobileLLM | https://arxiv.org/abs/2402.14905 | primary via extraction tool; tables not cell-checked | not verified |
| GPT-BERT | https://arxiv.org/abs/2410.24159 | primary, plus repository | not verified |
| StateHead-10M | https://aclanthology.org/2026.babylm-main.40.pdf | primary, plus config and code; companion papers main.1 and main.7 abstract only (secondary) | not verified |
| MoE design choices (Fan, Messmer, Jaggi 2024) | https://arxiv.org/abs/2402.13089 | primary via extraction tool | not verified |
| Learning to Screen (Chen et al. 2019) | https://arxiv.org/abs/1810.12406 | primary | not verified |
| BinaryAttention (CVPR 2026) | https://arxiv.org/abs/2603.09582 | primary, plus code | not verified |
| LayerNorm removal (Baroni et al.) | https://arxiv.org/abs/2507.02559 | primary via summarising fetch | not verified |
| N-vium (Lorenc et al. 2026) | https://arxiv.org/abs/2605.13190 | primary (HTML text and README; figures unread) | not verified |
