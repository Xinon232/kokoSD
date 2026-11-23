# Chat training data, synthetic generation, distillation and vocabulary for Koko (focused round, final)

Cluster report, 19 Nov 2026. 19 sources (`research/notes/f-data-*.md`), read against our pilot (`research/pilot-results.md`: 128x4, 4K vocabulary, 491M TinyStories tokens, seed noise about 0.005 nats). Two sources (LlamaTales, Physics 3.3) were independently verified: no claim refuted, three partly confirmed (flagged).

Tags: **[M]** measured by a source, **[P]** measured in our pilot, **[D]** derived, not measured. Nothing here costs run time unless a GBA cost is given.

## 1. Bottom line

1. **Dense compute at our budget gives incoherent text even on the easiest corpus.** A float 128x4 student (1.05M weights per token) scores about 28/100 coherence on child stories and 41 on college-level ones; 384x4 (9.4M) scores 67-82 [M, read off plots, verified]. Evidence: moderate (no seeds, 128K-vocabulary confound).
2. **Reading level does not matter for learnability; n-gram diversity does, and small students are brittle off-corpus.** A 33M student recovers 0.88-0.94 of corpus coherence on synthetic corpora against 0.23-0.29 on web text [M, partly verified: the 33M size is inferred, and at 1M the ratio is 0.31-0.43 even on synthetic data]; it scores 80-89 in-distribution and 14-26 on a sibling corpus [M, verified]. Evidence: moderate.
3. **An 8B open generator at temperature 1.0 is enough, and its sampled text already is distillation.** Llama-3.1-8B stories score 89.5 against 90.1 for GPT-3.5/4 [M, verified]; text sampled from the untruncated teacher distribution is the N=1 case of an unbiased sparse-logit target [M for N=50 at 300M+; the N=1 reading is D]. Evidence: moderate.
4. **Logit distillation is a low-value item at our size.** Vanilla KD gave 39.9 against 39.9 at 200M [M]; sampled-logit KD gains 0.004 nats over cross-entropy at 300M and the gain shrinks with student size [M] [M]. Evidence: moderate, none below 140M.
5. **Plan unique data at a quarter or more of tokens trained, and never more than about 16 passes.** Four epochs cost 0.5% loss against fresh data (8.7B model); the fitted repetition constant is 15.4 [M, one global fit on C4, float]. Evidence: moderate; not confirmed at our scale.
6. **Facts need exposures and wording variety more than ROM.** About 1 bit per float parameter at 100 exposures and 2 at 1,000 [M, verified, peak values]; one fixed wording gives 9.7% held-out QA against 96.6% with 5 wordings [M, 124M, unverified]; unique look-alike filler cuts capacity about 20x at 100 exposures, a source tag brings that to 2x, repetitive filler is free [M, partly verified: the 3x/1.3x figures at 300/1,000 exposures are relative to the 100-exposure baseline]. Untested for ternary.
7. **Chat and QA must be in training from step one.** Mixed training gives 86.6% held-out QA against about 0% for pretrain-then-finetune [M, 124M]; a 2.4K everyday-conversation set fixes "Hi" and "Who are you" [M, 1.7B].
8. **Predictive typing needs "retrace" decoding, not a new tokenizer.** Completing a partial word under BPE without backing up gives MRR .164 and recoverable length 0.51; backing up to a token boundary gives .294 and 4.64 [M, 1-layer LSTM, vocabulary 256]. Hashed n-gram inputs help (-0.13 nats at 8 MB [P]); multi-token outputs do not (2.556 against 2.554 at 1.3B [M]). Evidence: strong for retrace, at one scale.

## 2. Ranked recommendations

| # | Technique | Action | Expected gain (source scale) | GBA cost | Evidence | Main risk |
|---|---|---|---|---|---|---|
| 1 | Seeded, multi-format corpus from an 8B generator, temperature 1.0, top-p 1.0 | adopt | Coherence 89.5 vs 90.1 for GPT-3.5/4 [M]; under 1% MinHash duplicates at 31M docs [M] | None | Moderate | Template collapse; no simple-English chat evidence |
| 2 | Data budget: unique tokens at least T/4, at most 16 passes, reshuffle each epoch | adopt | 4 epochs: +0.5% loss vs fresh (8.7B); T/8 unique loses about 17% effective data [D from fit] | None | Moderate | Constants from float C4 fit; sparse table rows may differ |
| 3 | Single stage with chat, QA and source tags from step 0 | adopt | 86.6% vs about 0% QA (124M) [M]; junk loss 20x to 2x [M] | 1 ring slot, 1 embedding row per tag | Moderate | Tags tested on synthetic biographies only |
| 4 | Fact list: at least 5 wordings, shuffled, full names, 300 exposures | adopt | 9.7% to 96.6% QA (124M) [M] | None | Strong at 124M+, none at ours | Ternary bits per weight unknown |
| 5 | Dialogue recipe: seed, scene, dialogue, filters; role tokens; greetings grid; fixed name pool | adopt | MTLD 68.0 vs 63.1 without scene [M]; MT-Bench 5.22 to 5.49 at 1.7B [M] | None | Moderate, 1.7B+ only | Adult register; 68.9% filter yield |
| 6 | Diversity dial: unique n-grams per 100M tokens, opener cap, judge | adopt | Separates learnability 0.9 from 0.27 at 33M [M] | None | Moderate | Calibrated on stories |
| 7 | Retrace decoding for word completion, scored by MRL | adopt | MRL 0.51 to 4.64, MRR .164 to .294 [M] | Prefix table about 150-250 KB ROM; 1-3 forward passes per keystroke [D] | Strong at vocab 256 | Paper says BPE degrades as vocabulary grows |
| 8 | Breadth vs brittleness, and 0/30/60% real text | test | 50/50 beats synthetic-only by 0.1-2.0 points (1.2B) [M] | None | Conflicting | Real text has learnability 0.27 |
| 9 | Fact capacity of the ternary routed model | test | Sets fact budget; int4 PTQ 0.7 bit/param [M] | None | None for ternary | Penalties may multiply |
| 10 | Offline register control: sample several replies, filter by word list, rerank | test | Rerank Spearman 0.72 vs 0.23 for hard masks (2.7B) [M] | None offline; optional 16 KB level table | Moderate, one scale | Costs about 5 points sensibleness |
| 11 | Real user turns (WildChat, 48 tokens or fewer) as seeds and typing data | test | Unknown; 837,989 public conversations, ODC-BY [M] | None | None | 61.9% are writing requests; 52.94% English |
| 12 | Generator 1.7B vs 8B for chat | test | 1.7B best of six families for rephrasing [M] | None | Weak for dialogue | Constraint obedience |
| 13 | Over-encoding details: sliced sub-tables, odd moduli, averaging | test | -0.008 train loss at 1.3B [M]; tables -0.13 [P] | 4 row reads; folded about 1K adds [D] | Weak below 1B | "Slow start" |
| 14 | Sampled segmentation (BPE dropout) for typing; PPM trie in ROM; ranked D-pad menu | test | Sampled segmentation MRL 4.48 with no retrace [M]; menu speed follows wpm = 12R/H [M] | Trie lookups and adds; counts in slow RAM [D] | Weak | Reordered menus gave no speed gain (9.61 vs 9.10 wpm) [M] |
| 15 | Difference sampling (MiniPLM), noise tail only | test, low | +1.4 points at 200M, +1.0 at 140M [M] | None | Moderate at 140M+ | Kept half may exceed our capacity |
| 16 | SuperBPE at 16K | test, low | Bytes/token 4.45 to 6.09 at 200K [M]; mixed at 8K | 0.3 MB ROM [D] | Within noise | Harder targets; mid-word prefixes |
| 17 | Logit KD: extra sampled draws, or cross-tokenizer ALM | test, low | 0.004 nats at 300M [M]; ALM +0.9-1.6 points, s.e. 0.7 (8B to 2B) [M] | None | Weak | Tokenizer mismatch; 36 bytes per position |
| 18 | Byte-pooled embeddings, output n-gram heads, run-time masks or 20-candidate rerank, continuous Dasher, cached Top-K under 50, DPO, staged curriculum | skip | Top-12 worse than CE (2.866 vs 2.807) [M] | Rerank: 20x generation [D] | - | - |

## 3. Recipes and cheapest experiments

Experiments use the pilot rig (491M tokens per arm), 2 seeds, a 16K vocabulary, and report bits per byte plus judge coherence (1-100, 1,000 completions).

**1. Generator.** Llama-3.1-8B-Instruct class, temperature 1.0, top-p 1.0, max 512 tokens, no repetition penalty [D: truncation and penalties bias the implicit distillation target]. Prompt = format instruction + seed (first 1,000 characters of a Simple Wikipedia paragraph, or a first turn) + 3 required words + 0-3 feature flags, topic added with p=0.5 [M, Cosmopedia, LlamaTales]. Pilot each template on a few hundred samples; reject it if one opener dominates (7,619 of 10,000 Qwen outputs began "Problem:" [M]). MinHash dedup; 10-gram decontamination. 5090 throughput is unmeasured.

**2. Budget.** If the final run trains T tokens, generate at least T/4 unique (2-4B trained needs 0.5-1B). Effective-data multipliers [D from fit]: 4 epochs 3.73, 8 epochs 6.62, 16 epochs 10.6. *Experiment:* pilot package on 123M unique x 4 against 491M x 1; confirm the gap is under 0.01 nats.

**3. Single stage, tags.** Four reserved ids (FACT, STORY, CHAT, WEB) prepended to documents; two role tokens; loss on both speakers so the user side trains typing.

**4. Facts.** At least 5 independent wordings, order shuffled together with rewording (shuffling alone hurt: 4.4% vs 9.7%), subject named in every sentence, QA turns for a subset; full augmentation of a "celebrity" half transferred to the rest (4.4% to 86.8%) [M]. Budget [D]: 75 rewordings x 4 epochs gives 300 exposures, so 20K facts cost about 45M unique tokens. Start with 20-50K facts.

**5. Dialogues.** Topic x subtopic grid (20x100) or commonsense triple, then a 2-3 sentence scene, then a 4-12 turn dialogue; drop malformed speakers, more than two speakers, turn counts outside 4-20; keep the scene as a prefix with 30% dropout [M, COSMO]. Replies capped near 25 tokens [D]. Incidental names from a pool of about 1K (repetitive filler is free [M]). Generator writes user turns too. Reuse smol-smoltalk (460K rows) and SODA (1.49M) after an LLM rewrite.

**6. Dial.** Per corpus version: unique 2-4-grams per 100M tokens, distinct 5-token openers per 10K documents, share of table rows hit 50 times or more.

**7. Retrace.** Commit context up to the last space; for the partial word, restrict the first generated token to vocabulary entries extending the typed characters (sorted-vocabulary range in ROM, about 14 binary-search probes [D]); decode to the next word boundary only. *Experiment (no GPU training):* on the pilot checkpoint, MRL and MRR with and without retrace on held-out user turns.

**8. Breadth and real share.** Arms at equal tokens on the pilot package (1.780) and float baseline: (a) one story template; (b) 8 formats; (c) = (b) with 30% real simple-English text; (d) 60%. Score each on its own held-out set, the others', and a fixed chat set; take the widest corpus within 5 coherence points of the best on chat.

**9. Fact capacity.** Ternary routed package on bioS(N), N = 20K and 100K, at 100 and 300 exposures, alone and diluted 7:1 with chat text, tags on/off; convert loss to bits per stored weight. Planning range until then: 0.1-0.25 bit per weight [D].

**10. Register.** For each reply sample 4 candidates, drop any with a word outside the list, keep the one closest to target level by summed per-token level [D, adapted from Tyen]. Compare judge sensibleness against single-sample generation on 1,000 dialogues.

**11. Real user turns.** English, unredacted, 48 tokens or fewer, low out-of-vocabulary, capped per hashed IP; discard GPT replies and let the generator answer. *Experiment:* typing MRL on held-out real turns for a student trained with and without 5% real user turns.

**12-13.** 50M tokens from a 1.7B and an 8B generator, compared by judge, opener counts and student loss on a human-written chat set. Pilot arm "bigram 32K + trigram 32K" (1.729) against the same bytes as 2 sliced sub-tables per order, on validation loss.

**15. Difference sampling.** Score = (reference loss - teacher loss) per token on real text only; teacher about 0.5B, reference = our float pilot model. Drop only the tail where the teacher is no better than the reference (6.3% of the Pile [M]).

**14, 16, 17.** One pilot arm with sampled segmentations, scored by MRL. Measure bytes per token for 16K SuperBPE; proceed only above +10%. KD only after a float twin exists, with per-token KL (`batchmean` over-weighted it about 512x in Brandon-Tiny [M]).

## 4. Contradictions and resolutions

- **Repetition: 16 passes is the ceiling, yet facts need 100-1,000 exposures.** Muennighoff measures verbatim whole-corpus epochs; Allen-Zhu and Li re-sample templates each exposure. Resolution [D]: get exposures from rewordings times a few epochs (recipe 4).
- **Synthetic-only: best or worst?** LlamaTales: synthetic is 3x more learnable. FinePhrase and Brandon-Tiny: synthetic-only is worse than a mix. Metrics differ (coherence at 33M; knowledge benchmarks at 0.5B+). Experiment 8 settles it.
- **Diversity: more or less?** Cosmopedia, FinePhrase and SODA maximise it for 1B+ students; LlamaTales shows small students need low n-gram diversity; Physics 3.3 shows unique look-alike detail is the costly kind. Resolution [D]: diversify formats, topics and fact wordings; keep incidental entities and register narrow.
- **Simple vocabulary: needed or not?** LlamaTales says reading level does not affect learnability; Tyen shows word lists barely control perceived difficulty (0.23). Simple English is then a product choice for readers, enforced by reranking, not a training aid.
- **Distillation: helps or not?** Gains appear at 200M-3B, little or none from logits at the small end; Brandon-Tiny's gain is confounded. Rely on teacher-sampled text; rank logit KD last.
- **Predictions on screen.** Dasher gains speed from a model-ordered layout; three-key and gamepad studies found reordering saves keystrokes but not speed. Keep letters static; show suggestions in a separate list.

## 5. What nobody has tested

- Knowledge bits per ternary QAT weight; whether routing, junk, low-exposure and quantization penalties multiply.
- Learnability ratio when a 1M-active student has tables and experts.
- Whether a mixed corpus cures off-distribution collapse, and at what in-distribution price.
- Fact extraction below 124M; multi-turn chat below 135M; any distillation below 140M or into ternary students.
- Repetition tolerance of sparse rows (tables, experts) against dense weights.
- Retrace with a 16K vocabulary, a transformer and a two-level class/member head.
- D-pad typing speed with neural suggestions.
- Whether tiny students benefit from real user turns; every other source's "user" is an LLM.

## 6. New ideas [all D]

1. **Fit the corpus to the tables.** Tune prompts until bigrams and trigrams seen 50 times or more roughly fill the planned rows; ROM size becomes a corpus design target.
2. **Entity-keyed fact table.** Extractable facts are bound to the subject name (Physics 3.1), n-gram rows to preceding words. Add a table keyed on the hash of the latest name-like span in the ring buffer: one lookup, one row add, 16 bits of RAM.
3. **Teacher histograms in ROM.** Sparse-logit sampling stores about 12 ids per position in 36 bytes; store the same histogram per hashed trigram context as a logit prior and as ready-made typing suggestions, with no forward pass.
4. **Retrace through the class head.** Precompute, per typed-prefix range, a bitmask of the output classes that contain an extending token (64 classes = 8 bytes); evaluate members only in those classes, making constrained steps cheaper than free ones.
5. **Two-speed typing.** A ROM character trie (order-5 PPM style) completes within a word between forward passes; the transformer re-ranks at word boundaries.
6. **Table-aware data selection.** Use a tables-only model as the MiniPLM reference, so dense training tokens are spent on what lookups cannot predict.

## 7. Sources

"Extraction" = read through a text-extraction fetch; figures not viewed.

| Source | URL | Read | Verification |
|---|---|---|---|
| Cosmopedia | https://huggingface.co/blog/cosmopedia | primary (blog, card, repo) | none |
| FinePhrase | https://arxiv.org/abs/2604.13977 | primary, extraction; prompts paraphrased | none |
| LlamaTales | https://arxiv.org/abs/2510.13915 | primary; per-model values read off plots | 2 confirmed, 2 partly |
| Physics of LMs 3.3 | https://arxiv.org/abs/2404.05405 | primary, extraction | 3 confirmed, 1 partly |
| Physics of LMs 3.1 | https://arxiv.org/abs/2309.14316 | primary, extraction | none; 96.6% condition uncertain |
| Over-Tokenized Transformer | https://arxiv.org/abs/2501.16975 | primary, extraction | none; sub-1B figure-only |
| SmolLM2 | https://arxiv.org/abs/2502.02737 | primary plus cards | none; 135M ablations not shown |
| Brandon-Tiny 10M | https://xaskasdf.github.io/brandon-tiny/paper.html | primary, report and code | none; self-published, confounded |
| Super Tiny Language Models | https://arxiv.org/abs/2405.14159 | primary | none; headline idea never run |
| SODA | https://arxiv.org/abs/2212.10465 | primary, extraction | none |
| SuperBPE | https://arxiv.org/abs/2503.13423 | primary; BabyLM 2025 numbers via the note | none |
| ALM cross-tokenizer distillation | https://arxiv.org/abs/2503.20083 | primary; Figure 3 unread | none |
| Sparse Logit Sampling | https://arxiv.org/abs/2503.16870 | primary | none |
| MiniPLM | https://arxiv.org/abs/2410.17215 | primary, extraction | none |
| Tyen et al., chatbot for language practice | https://aclanthology.org/2022.bea-1.28/ | primary | none |
| Subword LM for query auto-completion | https://arxiv.org/abs/1909.00599 | primary, paper and code | none |
| Dasher (three-key and H4-Writer studies via the note) | https://arxiv.org/abs/cs/0204030 | primary; follow-ups secondary | none |
| Scaling Data-Constrained LMs | https://arxiv.org/abs/2305.16264 | primary, extraction | none |
| WildChat | https://arxiv.org/abs/2405.01470 | primary plus dataset cards | none |
| Pilot results | `research/pilot-results.md` | own measurement | one seed for most arms |
