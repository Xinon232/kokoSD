# BabyLM 2025 evaluation pipeline (https://github.com/babylm/evaluation-pipeline-2025, 2025, BabyLM organisers: Charpentier, Choshen, Jumelet, Mueller, Warstadt et al.)

Tags: **[P-code]** repo source; **[P-readme]** README; **[P-card]** raw model card; **[P-paper]** Findings PDF. All read in primary form.

## What it is
The official scorer for the 2025 BabyLM Challenge (Apache-2.0). Zero-shot tasks rank candidate sentences or continuations by summed token log-probability: BLiMP, BLiMP supplement, EWoK, entity tracking, two wug tests, COMPS. It also has word-surprisal tasks (reading time, age of acquisition) and (Super)GLUE fine-tuning. Backends: `causal`, `mlm`, `mntp`, two encoder-decoder styles. A "fast" subset serves intermediate checkpoints.

## Mechanism
Causal scoring [P-code: `sentence_zero_shot/dataset.py`, `compute_results.py`]:

```
toks = tokenizer(sentence, return_offsets_mapping=True)   # default special tokens
mask[i] = 1 if token i's end char > len(sentence) - len(completion)
inputs = toks[:-1]; targets = toks[1:]; mask = mask[1:]   # right-pad with EOS id
logp   = log_softmax(model(inputs).logits / T)            # T = 1.0 by default
score  = sum_t mask[t] * logp[t, targets[t]]              # no length normalisation
pred   = argmax over candidates                           # label is always index 0
```

- **BLiMP, supplement:** `[sentence_good, sentence_bad]`, whole sentence scored.
- **EWoK:** `[C1+" "+T1, C1+" "+T2]`, only the target scored. The filter script also writes each item with contexts and targets swapped.
- **Entity tracking:** `input_prefix + option` for 5 options [P-paper], only the option scored. Accuracy is averaged per `<split>_<numops>_ops` subset, then per split (regular, ambiref, move_contents), then over splits.
- **COMPS:** `[prefix_acceptable, prefix_unacceptable] + " " + property_phrase`, only the property phrase scored. Four subsets: base, wugs, wugs_dist_before, wugs_dist_in_between.
- **Wug adjective / past tense:** softmax over candidates' sentence scores; the probability of candidate 0 is Spearman-correlated with the human `ratio`.
- **Other tasks:** unweighted mean over subtasks (UIDs).
- **Reading:** surprisal `-ln p(word | context)`. OLS on word frequency, length and context length, with and without surprisal. Score = `100*(R2_model - R2_base)/(1 - R2_base)`, averaged over four eye-tracking measures; self-paced reading adds one-word spillover terms.
- **AoA:** surprisal of 8,005 items at every checkpoint, sigmoid fit, correlation with child data [P-readme].
- **Run:** `./eval_zero_shot.sh <model> causal`; loads `AutoModelForCausalLM` with `trust_remote_code=True`.

## Numbers
Full-set sizes [P-code: `collate_preds.py`; sums are mine]:

| Task | Subtasks | Items |
|---|---|---|
| BLiMP (filtered) | 67 | 59,875 pairs |
| BLiMP supplement | 5 | 5,218 |
| EWoK (filtered, swaps included) | 11 | 7,618 |
| Entity tracking | 18 | 9,483 |
| COMPS | 4 | 91,028 pairs |
| Wug adjective / past | 1 / 1 | 200 / 50 |
| Reading | 1 | 1,726 words |

Fast set: 200 per BLiMP subtask, 50 per supplement subtask, 100 per EWoK domain, the six "regular" entity-tracking subsets (3,152 items), no COMPS.

GPT-2 baselines trained by the organisers (12 layers x 768, 16,384-token vocabulary, 98.4M parameters in safetensors although the card says 124M, 10 epochs) [P-card]:

| Trained on | BLiMP | Supp. | EWoK | Entity | WUGs | Eye track. | Self-paced |
|---|---|---|---|---|---|---|---|
| 100M words | 74.88 | 63.32 | 51.67 | 31.51 | 35.5 | 7.89 | 3.18 |
| 10M words | 66.36 | 57.07 | 49.9 | 13.9 | 52.5 | 8.66 | 4.34 |

- The README holds a commented-out draft table that disagrees (GPT-2 100M: 75.07 / 64.96 / 51.22, entity 30.11, eye tracking 0.47). Trust the cards.
- Findings Table 3 [P-paper]: GPT-2 NLP score 55.4 (strict) and 49.1 (strict-small); best GPT-BERT baseline 63.0.

## Smallest scale tested, and whether the result held there
The smallest baseline is the 10M-word GPT-2 (about 98M parameters); nothing near our 1-2M active weights is reported. At 10M words EWoK is at chance (49.9) and entity tracking is below the 20% chance level (13.9), so the distractors punish copying heuristics. The Findings paper reports U-shaped entity-tracking curves [P-paper].

## Training recipe and gotchas
- **First token:** it is scored only if the tokenizer prepends BOS, because targets start at token 1.
- **Ties:** `torch.max` returns index 0, which is the correct label, so exact ties count as correct. Coarse integer logits can inflate accuracy.
- **Temperature:** a sweep (`--max_temperature`) reports the best temperature on the test items. Leave it at 1.0.
- **Tokenizer:** must be a fast tokenizer, because offsets are required.
- **Filtering:** BLiMP and EWoK are filtered to BabyLM vocabulary; EWoK needs gated Hugging Face access.

## Mapping onto the GBA design
- **Cartridge cost:** zero. This is an off-device benchmark, run on a bit-exact PC simulator wrapped as a Hugging Face causal model that returns `[B, T, 16K]` logits.
- **Two-level head:** `log p(token) = log p(class) + log p(member | class)`. Scoring only needs the target's class, but the stock pipeline wants full-vocabulary logits. Either expand them in the wrapper or patch `compute_causal_results` to take target log-probabilities.
- **Logit scale:** convert integer logits to nats with the trained scale before `log_softmax`.
- **Why not on the console:** assuming about 10 tokens per sentence (my estimate), BLiMP alone is about 1.2M token steps, roughly 3 days at 4 tokens/s.
- **Predictive typing:** nothing here measures it.

## Verdict
**Adopt** as the linguistic-competence leg of the benchmark, and run OpenAI GPT-2 124M through the same script (`gpt2`, causal) for the head-to-head. Use the fast set per checkpoint and the full set at the end. Count ties as 0.5 in our own log. Skip GLUE and AoA. Settling experiment: score a float and a ternary checkpoint plus `gpt2` on fast and full sets; if rankings agree, the fast set is enough for ablations.

## Could not verify
- OpenAI GPT-2 124M scores under this pipeline: not published; must be run.
- Entity-tracking prompt lengths in tokens against our 256-token window: the data is on OSF and was not downloaded.
- The exact formula for the "human-likeness" and "NLP" aggregate scores.
- COMPS and AoA baseline numbers.
- Average BLiMP sentence length.

## Links worth following
- Findings of the Third BabyLM Challenge: https://aclanthology.org/2025.babylm-main.28.pdf
- Evaluation data: https://osf.io/ryjfm/
- Baseline card: https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt2
- Leaderboard: https://huggingface.co/spaces/BabyLM-community/babylm-leaderboard-2025-all-tasks
