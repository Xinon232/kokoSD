# Findings of the Third BabyLM Challenge

https://aclanthology.org/2025.babylm-main.28.pdf (2025; Charpentier, Choshen, Cotterell, Gul, Hu, et al., the BabyLM Team; First BabyLM Workshop, pp. 399-420)

Tags: **[P]** read in the paper; **[C]** read in the raw HuggingFace baseline model cards; **[M]** my inference.

## What it is
The organisers' report on the 2025 shared task: train a language model on at most 10M words (Strict-Small) or 100M words (Strict, Multimodal, Interaction). New in 2025: a cap on word exposures (100M for Strict-Small, 1B otherwise, repeats counted), an Interaction track with an LLM teacher, and "human-likeness" tasks. It ranks training recipes under fixed data and ships a standard evaluation pipeline.

## Mechanism
**Evaluation [P].** Zero-shot tasks need only string log-probabilities:
- BLiMP, BLiMP supplement, EWoK, COMPS: the correct sentence of a minimal pair must score higher.
- Entity tracking: 5 candidate continuations, the correct one must score highest (chance 20% [M]).
- Wug past tense and adjective nominalisation: correlation of model probability with human preferences.
- Reading time and age of acquisition: surprisal correlated with human data.
- Fine-tuning: seven (Super)GLUE tasks, each cut to 10,000 training examples.
- "Fast" evaluation is 20% of each task. BLiMP and EWoK are filtered to BabyLM-corpus words [C].
- Macro average = mean of human-likeness and NLP score [M, checked against Table 3].

**GPT-BERT baseline [P, C].** LTG-BERT body (DeBERTa disentangled attention, pre- and post-norm, GEGLU) trained on a mix of causal and masked-next-token-prediction examples. Strict-Small: ~31M parameters, 12 layers, 6 heads, 16,384 tokens per batch, 9,914 steps, peak LR 7e-3, cosine with 1.6% warmup and 1.6% cooldown, LAMB, weight decay 0.1, clip 2.0, mask ratio 0.3 decaying to 0.15. Sequence length 128 for 60% of steps, 256 for 20%, 512 for 20%, at constant tokens per batch. "Causal focus" = 93.75% causal, 6.25% masked.

**GPT-2 baseline [P, C].** GPT-2 Small (124M), 512-token chunks, batch 16, 10 epochs, LR 5e-5, AdamW; ~0.5 H100-hours on the 10M corpus.

## Numbers
Table 3 [P], scores x100:

| Track | System | Human-like | NLP | Macro |
|---|---|---|---|---|
| 100M | GPT-BERT causal-focus, causal eval | 26.5 | 60.2 | 43.4 |
| 100M | GPT-2 | 17.1 | 55.4 | 36.2 |
| 100M | Simple-Diffusion (NLP winner) | 12.6 | 58.4 | 35.5 |
| 10M | GPT-BERT causal-focus, causal eval | 18.7 | 56.1 | 37.4 |
| 10M | GPT-2 | 19.8 | 49.1 | 34.5 |
| 10M | AMLM-Hard-Decay (NLP winner) | 8.4 | 58.3 | 33.3 |
| 10M | MoEP, token-routed experts | 31.5 | 53.2 | 42.3 |
| Multimodal | BitMar, low-bit (Flamingo baseline: 4.1 / 55.8 / 29.9) | -9.3 | 49.0 | 19.9 |
| Interaction | BLM winner / SimPO baseline | 20.8 / 20.4 | 54.4 / 54.1 | 37.6 / 37.3 |

Per task at 10M words [C]:

| Model | BLiMP | Supplement | EWoK | Entity tracking |
|---|---|---|---|---|
| GPT-2 124M | 66.36 | 57.07 | 49.9 | 13.9 |
| GPT-BERT 31M causal-focus, causal eval | 71.66 | 63.21 | 49.49 | 33.96 |

Other findings [P]:
- No Strict entry beat GPT-BERT (best NLP 63.0, masked-focus). FLOPs track score only in the Interaction track (Fig. 6).
- Footnote 3: zero-shot prompting failed at this data size; induction heads are cited as needing 2.5-5B tokens (Olsson et al. 2022).
- Appendix D, organisers' summaries: synthetic data helped at about half the corpus and hurt when it dominated; morpheme tokenisation gave ~20% on EWoK and ~40% on entity tracking; smaller batches, optimiser-state resets and Muon helped.

## Smallest scale tested, and whether the result held there
10M words, 100M exposures; the smallest model named is the 31M GPT-BERT. The ranking held: GPT-BERT causal beats GPT-2 by 7.0 NLP points at 10M (4.8 at 100M) with a quarter of the parameters. Nothing below 31M parameters, and no ternary text-only model.

## Training recipe and gotchas
- The GPT-BERT/GPT-2 gap is confounded: architecture, optimiser and a 140x learning-rate difference all change.
- "GPT-2" here is trained on 10M-100M words. It is not OpenAI's GPT-2, so these rows are not our GPT-2 reference.
- Entity tracking is below chance for GPT-2 and U-shaped over training. EWoK is at chance for both 10M baselines.
- The only low-bit entry scored below its baselines. MoEP is reported with "later-phase stability trade-offs".
- Organisers: interaction "is not currently as effective as simply pretraining".

## Mapping onto the GBA design
Run-time cost is zero: no lookups, adds, multiplies, RAM or ROM. Score sentences offline in a bit-exact PC simulator of the integer model: log P(token) = log P(class) + log P(member | class), summed per sentence.
- A 6.25% masked share and a 64/128/256 length schedule are free at inference.
- MoEP shows token-routed experts train at 10M words; a router keyed on token id is one ROM lookup.
- If in-context copying needs billions of tokens, name recall over the ring buffer needs that exposure or targeted synthetic recall data.
- Keep real text near half of the pretraining mix.

## Verdict
**Adopt** the zero-shot pipeline and the Strict-Small protocol as a cheap ablation loop. **Test** the 15:1 causal/masked mix and the length schedule. **Skip** teacher-scored reinforcement, GLUE fine-tuning and human-likeness scores. Settling experiment: float and ternary versions of our model on BabyLM-10M for 10 epochs against BLiMP 66.4 (GPT-2) and 71.7 (GPT-BERT), then with and without the masked share.

## Could not verify
- Per-task scores for submissions and the 100M baselines (figures only).
- Human and Llama-70B skyline values (Fig. 4, plot only).
- The NLP and human-likeness aggregation formulas.
- Whether the tokenisation gains are relative or absolute.
- MoEP and BitMar details: submission papers not read; baseline cards at https://huggingface.co/BabyLM-community.

## Links worth following
- MoEP (Tapaninaho 2025)
- GPT or BERT: why not both? (Charpentier and Samuel 2024)
- https://github.com/babylm/evaluation-pipeline-2025
