# Cluster report F-eval (final): evaluating Koko against GPT-2 small, and judging a tiny chatbot

Scope: 17 sources; 16 read in primary form (several through a paraphrasing fetch tool), one abstract-only. **No note carries an independent verification verdict**: every claim is read once, unverified. Tags: [P] measured or stated in a primary source; [S] secondary; [M] our arithmetic or inference.

Everything here is offline protocol with zero cartridge cost, except items 6 and 9 in the table (typing engine, n-gram prior).

## 1. Bottom line

1. **GPT-2 small's published numbers are not a usable baseline; rerun it in our harness.** LAMBADA is 45.99 in the paper, 32.56 +/- 0.65 under lm-evaluation-harness on the same file, 46.67-46.89 by last-BPE-token and 41.98 by whole-word generation [P]. DialoGPT publishes no perplexity at all [P]. Evidence: strong.
2. **Only bits per byte and whole-word scoring are tokenizer-fair.** GPT-2 perplexities are per dataset word [P]; scoring rule alone moves GPT-2 small by about 5 points on one file [P]. On enwik8, GPT-2 small's zero-shot 1.16 bits/byte [P, GPT-2 paper] sits at cmix v21 level (1.17); a static model at our compute plausibly lands between ppmd (1.71) and paq8hp12 (1.30) [M]. Evidence: strong (metric); the bracket is a guess.
3. **Statistics must be paired and clustered, and the judged set resolves only about +/-7 to 10 points.** Clustered SEs were 1.1-3.05x naive on real evals [P]; CLT intervals cover 92.5% instead of 95% at N = 100 and worse below [P]; Wilson half-width at N = 200 is 6.9 points [M]. Decide small ablation deltas on likelihood, not judging. Evidence: strong (method), sizes need a pilot.
4. **Typing is where Koko can match GPT-2 small.** Unadapted gpt2-124M ties a 70M-parameter character 12-gram (2.54 vs 2.55 per-character perplexity); distilgpt2 loses (2.71) [P]. A 1.4M-parameter LSTM got 16.5% / 27.1% top-1 / top-3 next-word recall vs 13.0% / 22.1% for a 5-gram [P]. Evidence: medium; no keystroke-savings figure exists for gpt2-124M, and the n-gram's ROM-prunability is untested.
5. **On a D-pad, prediction is worth about 3.6x more than on a keyboard.** A five-button letter grid costs 3.13 presses per character [P]; context-free unigram completion with one-press selection reaches 0.55 KSPC at 5 slots on a keyboard [P], which we convert to about 1.5 presses per character on the grid [M]. Stepping through a list loses beyond 2 entries [P]. Evidence: strong analytic, no user study.
6. **LLM judging is a secondary signal.** GPT-4 agreed with humans on 85% of non-tie votes but was only 65% order-consistent, 36-42% on subjective categories [P]; small local judges cap near Spearman 0.44-0.52 with humans and calibration fixes scale, not ranking [S, abstract only]. Use forced-choice whole-dialogue comparison in both orders. Evidence: medium; nothing tested on incoherent tiny-model output.
7. **State tracking must be trained in deliberately, with structurally diverse data.** Text-only models up to 175B repeat the initial state [P]; from-scratch specialists score 96-99% on bAbI but 22-36% on compositional mixes [P]; 17K diverse stories beat 190K redundant ones (95.8 vs 71.6, pretrained T5-base only) [P]. Evidence: medium for the need, weak for the fix at our scale.
8. **Poor headlines: HellaSwag, ARC, Natural Questions, MT-Bench questions, absolute MAUVE.** GPT-2 small is 28.6-29.6% on HellaSwag (chance 25), 20.1% on ARC, at most 1% on NQ; LLaMA-13B scores 2.61/10 on MT-Bench; MAUVE for GPT-2 small ranges 0.008-0.878 by decoder alone [P]. Evidence: strong.

## 2. Ranked recommendations

| # | Technique | Action | Expected gain (number, scale) | GBA cost | Evidence | Main risk |
|---|---|---|---|---|---|---|
| 1 | Own-harness rerun of GPT-2 small and DialoGPT-small; bits/byte and whole-word accuracy; GPT-2 also at a 256-token window | adopt | Removes a 13.4-point protocol artefact (GPT-2 124M) | none; needs bit-exact PC simulator | strong | simulator not bit-exact |
| 2 | Paired + clustered analysis: Miller SEs above a few hundred items, Wilson / beta-binomial Bayes below | adopt | Avoids 1.1-3.05x SE understatement (real evals); pairing cuts variance 1/3 at r = 0.5 (analytic) | none | strong | binary outcomes only for the Bayes models |
| 3 | Bits/byte ladder: held-out simple English + enwik8 beside compressor rows | adopt | Places Koko on a compute/memory scale: ppms 2.11 (1.8 MB), ppmd 1.71, GPT-2 1.16 | none | strong | compressors are adaptive; 2026 transformer rows are training-set likelihoods |
| 4 | Presses-per-character simulator on the real button map (KSPC), plus keystroke savings and top-1/top-3 recall | adopt | Baseline 3.13 presses/char; unigram completion ~1.5-2.2 [M]; n-gram 59.7% KS, opt-350m 65.2% (331M) | none | strong (metric) | ideal-user assumption; scanning time ignored |
| 5 | ACUTE-style forced-choice dialogue judging, both orders, audited judge | adopt | 9/9 matchups significant with self-chat vs 5/9 Likert (2018 chatbots) | none | medium | judge near-constant on weak output |
| 6 | Cached top-64 + prefix filter, one-press suggestion slots, ROM n-gram fallback | test | Unknown; must beat unigram completion | 256 B RAM, 32 KB ROM sorted-token table, a few hundred compares per key, 0 multiplies; fallback trie < 1 MB [M] | weak (our design) | multi-token words need extra passes |
| 7 | Diverse synthetic state-tracking data; boxes + Dyna-bAbI-style diagnostics by operations / supporting facts | test | 71.6 -> 95.8 on mix(T7) (T5-base, pretrained) | none at run time | medium / weak at our scale | poor transfer to new phrasing; untested from scratch |
| 8 | Single separator-token chat format | adopt | DialoGPT format; no measured ablation | one token id | medium | no role information |
| 9 | Static n-gram (PPM-style) prior fed to input and mixed into output logits | test | 6M 4-bit model with PPM prior: 0.911 vs 1.23 nats/byte for the LSTM it replaced (confounded; training-set) | a few hashed lookups per token, ROM only | weak | no gain when mixed into a fine-tuned 331M model |
| 10 | LAMBADA whole-word (lambada_openai) | adopt | Baseline 32.56% / ppl 40.06 (GPT-2 124M) | none | strong | passages over 256 tokens |
| 11 | MAUVE, relative only, for decoder selection | adopt (secondary) | Spearman 0.95 with human "human-like" over 8 settings | none | medium | not portable; ignores relevance |
| 12 | CBT cloze (test); HellaSwag completion-style (secondary only) | test | GPT-2 small CBT 87.65 / 83.4; HellaSwag 28.59 / 29.55, SE ~0.45 pp [M] | none | medium | HellaSwag at chance for Koko |
| 13 | ARC, NQ, MT-Bench questions, run-time MMI/beam 10, uncertainty-gated judging | skip | at or below chance / floor; MMI is >= 16x compute | n/a | strong | none |

## 3. Recipes and cheapest settling experiments

**R1. Baseline harness.** Wrap a bit-exact PC simulator of the integer forward pass (256-token ring, top-8 attention, routed experts) as an lm-eval `LM`; log p(token) = log p(class) + log p(member | class), computed in float from integer logits. Run GPT-2 small at 1,024 and 256 tokens, and DialoGPT-small as chat baseline. Record harness commit and task version (32.56 is version 0; current YAML is 1.0). *Settles it:* reproduce 32.56 +/- 0.65 for gpt2 on lambada_openai, then a few hundred items on the emulator for simulator-cartridge parity.

**R2. Statistics.** Per-prompt difference d_i; SE = (1/n) sqrt(sum_c (sum_i (d_ic - dbar))^2) (Miller Eq. 8), clusters = conversation / template / source document, at least ~50. Likelihood metrics are deterministic. For sampled or judged scores draw K >= 5 per prompt and average per prompt first; never lower temperature to cut variance. For the judged set (about 200 prompts in 50 clusters) binarise to win/loss and report P(Koko > GPT-2) from the paired Bayes model and a clustered beta-binomial interval (`bayes_evals`); Wilson for pass rates near 0 or 1. Sizing [M]: n = 7.85 var(d) / delta^2 gives 196 / 785 / 2,180 prompts for 10 / 5 / 3-point gaps. *Settles it:* a 200-cluster pilot with K = 5 estimating var(d), E[sigma^2] and cluster inflation.

**R3. Bits/byte ladder.** Report total bits / total UTF-8 bytes on (a) held-out simple English and chat, (b) enwik8, for Koko, GPT-2 small, a ROM-sized static n-gram, and ppmd/ppms run locally. *Settles it:* if Koko on (a) is not clearly below the static n-gram, the dense compute is not earning its cycles.

**R4. Typing metric.** Offline simulator replaying an ideal user on the exact button map: D-pad over the grid `abcdefghij / klmn_opqr / stuvwxyz` with snap-home, A selects, 3-4 suggestion slots on L / R / B / Select (one press, appends space). Report presses per character and Hick-Hyman scan time 200 ms x log2(n+1) separately. Also report keystroke savings (error-free user; 5 slots as in Gaines & Vertanen, and our 3-4) and top-1/top-3 next-word recall. Systems: grid alone, unigram completion, hashed n-gram, Koko, GPT-2 small as ranker using the released width-8 search. *Settles it:* Koko must clearly beat unigram completion.

**R5. Chat judging.** About 100 fixed-length dialogues per system against one shared scripted or LLM user (GPT-2 small has no chat format; document the prompt used). Show the judge two whole dialogues and one question ("Who would you prefer to talk to for a long conversation?"), both orders, inconsistent = tie. Reference answers for factual items (grading failures 14/20 -> 3/20 [P]). Power [M]: a 60% true win rate needs ~195 judgments, 55% ~780. *Settles it:* hand-label ~200 Koko-vs-GPT-2 pairs once; accept a judge only if it clears a preset bar on non-tied human pairs (say 75%), ranks 4-5 checkpoints of known perplexity order correctly, and is not near-constant.

**R6. On-device suggestions (test).** At each word boundary compute member scores for the top 3-4 classes, cache the top 64 (token id, score) pairs; per keystroke filter by prefix via a ROM table of token ids sorted by lowercased string (16K x 2 B). Never feed partial-word tokens (P('r' | 'yeste') 0.0047 vs 0.9999 with back-off to word start [P]). At most one beam-1 extension pass. *Settles it (PC only):* presses per character for K = 16 / 64 / 256 against full width-8 search on the same float model; under ~2 points of keystroke savings lost means the cheap path is enough.

**R7. State tracking (test).** Own generators: boxes (7 containers, 100 nouns, <= 3 objects, up to 12 Move/Remove/Put, no emptying op) and people-objects-places stories (MOVE/GRAB/DROP/GIVE, co-reference, negation, indefinites), 8-20 sentences with varied length, rejection-sampled so supporting-fact count is roughly uniform, many surface templates. Mix at ~5% of tokens [M]. Score 5-way log-probability (chance 20%), always by operations-on-box and by 1 / 2 / 3+ supporting facts; a copy-the-start heuristic scores below chance there. *Settles it:* float model at Koko's active size, with/without the mix, crossed with top-8 / top-16 / full attention.

**R8. Chat format.** `turn <sep> turn <sep> ...`, plain next-token loss over both speakers, which also trains user-turn prediction for typing.

**R9. N-gram prior (test).** Static order-3 to 5 count tables in ROM; feed top-k candidates as hashed-embedding inputs and add scaled log-counts to class/member logits. *Settles it:* float model with/without the prior at equal dense compute on held-out bits/byte, then under ternary QAT.

**R10-R12.** LAMBADA: context = all words but the last, greedy match of every target token, plus per-word perplexity; count truncated passages. MAUVE: ~5,000 prompts, identical length cap, GPT-2-large features, k = 500, several seeds, zero-shot and fine-tuned GPT-2. CBT and HellaSwag: branch candidates from a cached ring state; report sum, per-token and per-character variants.

## 4. Contradictions and resolutions

- **LAMBADA 45.99 / 46.9 / 41.98 / 32.56 / 31-35 for one model.** Different files and scoring rules. Fix whole-word scoring on lambada_openai and quote only our rerun. The stop-word-filter explanation (~11 points) is a search snippet [S].
- **Miller's CLT formulas vs Bowyer et al.** Not a real conflict: Bowyer simulates only N <= 100 and binary outcomes; Miller assumes large n. Use Bayes/Wilson for the judged set, Miller for likelihood sets of thousands of items.
- **Judge reliability: 85% (GPT-4, 2023) vs rho ~0.44 (small judges, 2026 abstract).** Different judges, tasks and statistics. Resolve by auditing our own judge (R5) and never using it per item.
- **Self-chat.** More discriminating in ACUTE, but one model's rank collapsed under it [P]; headline with a shared partner.
- **N-gram mixing.** No gain for a fine-tuned 331M model [P]; the 2026 compression transformers all use a PPM prior [P, submitter-reported]. The first is ~200x our compute, the second is training-set likelihood. Needs R9.
- **Neural vs n-gram typing.** A 1.4M LSTM beats a 5-gram; gpt2-124M only ties a 12-gram. N-gram sizes differ ~50x and gpt2 was unadapted. R4 settles it on one test set.
- **From-scratch tracking.** Random-init T5 fails on 90K boxes examples; diverse sampling was shown only on pretrained T5. Unknown for us.
- **Minor:** HellaSwag acc_norm 29.55 vs 31.14 (normalisation differs); boxes GPT-2 baseline 13.9 vs 18.92 [S], rerun.

## 5. What nobody has tested

1. Any of these benchmarks below 82M parameters, with a ternary or integer model, or a non-GPT tokenizer.
2. Keystroke savings for gpt2-124M; any presses-per-character figure with a contextual model, with 3-4 slots, or on a D-pad with real users.
3. LLM-judge validity when both systems are often incoherent; small-judge audits on dialogue.
4. Whether diverse synthetic data teaches tracking from scratch at about 10M parameters, and whether top-8 attention causes a cliff near 7 operations (our estimate only).
5. Held-out bits/byte for any small quantised transformer on the compression benchmark.
6. Interval coverage between N = 100 and a few hundred; ordinal judge scores; MAUVE on short dialogue.
7. How many LAMBADA, HellaSwag, boxes and bAbI-style items exceed 256 Koko tokens.

## 6. New ideas (ours; no source does these)

1. **Typing engine mostly from ROM.** A pruned word/character n-gram in tens of MB of ROM handles within-word completion by lookup; the neural top-64 re-ranks at word boundaries. This is the one line where beating GPT-2 small looks reachable.
2. **Prediction-ordered letter grid.** Because a letter costs ~3.6 presses, re-centre the cursor on the most likely next letter (a ROM table per prefix hash, or the cached top-64) instead of on SPACE.
3. **Sign-hash continuation cache in ROM.** The compression leaders' nearest-neighbour state cache, made static: hash the sign pattern of the final hidden state to a ROM table of continuations seen in training, consulted only on low-confidence tokens. Zero multiplies.
4. **Compute-normalised reporting.** GPT-2 small touches ~124M weights per token [M] vs 1.3-2M ternary: publish bits/byte against weights touched and writable RAM, with compressor rows as the classical frontier.
5. **Boxes accuracy vs operations as a per-layer top-k tuner**, and per-input-channel scales folded into activations before the partial-sum tables (free in our kernel, per the 4-bit transformer write-up).

## 7. Sources

Verification status for all rows: no independent verdict available.

| Source | URL | Read |
|---|---|---|
| Radford et al., GPT-2 (2019) | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | primary (full) |
| Hard et al., Mobile Keyboard Prediction | https://arxiv.org/abs/1811.03604 | primary (figures unreadable) |
| Kim & Schuster, Entity Tracking (2023) | https://arxiv.org/abs/2305.02363 | primary; BabyLM numbers secondary |
| Zheng et al., LLM-as-a-Judge (2023) | https://arxiv.org/abs/2306.05685 | primary via summarising fetch |
| Miller, Adding Error Bars to Evals (2024) | https://arxiv.org/abs/2411.00640 | primary (full) |
| karpathy hellaswag.py | https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py | primary (full) |
| openai/gpt-2 issue #131 | https://github.com/openai/gpt-2/issues/131 | primary via paraphrasing fetch |
| HF gpt2 discussion #90 | https://huggingface.co/openai-community/gpt2/discussions/90 | primary (verbatim) |
| Gaines & Vertanen, character AAC (2025) | https://arxiv.org/abs/2501.10582 | primary (full + code) |
| Zhang et al., DialoGPT (2019) | https://arxiv.org/abs/1911.00536 | primary (full) |
| Bowyer et al., Don't Use the CLT (2025) | https://arxiv.org/abs/2503.01747 | primary via summarising fetch; figure values unseen |
| MacKenzie, KSPC (2002) | https://www.yorku.ca/mack/hcimobile02.html | primary (full) |
| Pillutla et al., MAUVE (2021) | https://arxiv.org/abs/2102.01454 | primary via summarising fetch |
| Mahoney, Large Text Compression Benchmark | https://mattmahoney.net/dc/text.html | primary; results submitter-reported |
| Li et al., ACUTE-Eval (2019) | https://arxiv.org/abs/1909.03087 | primary via summarising fetch; table cells less reliable |
| Salhan et al., small LMs as judges (2026 workshop) | https://neurips.cc/virtual/2026/178372 | **secondary: abstract only** |
| Tamari et al., Dyna-bAbI (2021) | https://arxiv.org/abs/2112.00086 | primary (full) |

Full notes: research/notes/f-eval-01 ... f-eval-17.
