# karpathy/build-nanogpt: hellaswag.py — completion-style HellaSwag eval with GPT-2 reference numbers

URL: https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py (2024, Andrej Karpathy, MIT licence). Read in primary form: the full 177-line file, the repo README, the HellaSwag section of `train_gpt2.py`, and the lm-evaluation-harness HellaSwag task files for comparison.

## What it is

A self-contained script that downloads the HellaSwag jsonl files from `rowanz/hellaswag` and scores a causal LM on the 10,042-item validation split. Each item is a context plus 4 candidate endings; the model "chooses" the ending whose tokens it assigns the lowest loss. No prompt template, no few-shot, no fine-tuning: pure next-token likelihood. Its docstring gives reference numbers for GPT-2 124M and GPT-2 XL next to the Eleuther harness numbers, which is exactly the baseline Koko will be compared with. `train_gpt2.py` imports it and runs it every 250 steps as a training-time metric.

## Mechanism

Per item (primary, `render_example` and `evaluate`):

```
ctx_tokens = enc.encode(example["ctx"])            # raw ctx, no BOS, no activity_label
for each of 4 endings:
    end_tokens = enc.encode(" " + ending)          # leading space for GPT-2 BPE
    row  = ctx_tokens + end_tokens
    mask = [0]*len(ctx_tokens) + [1]*len(end_tokens)
pad the 4 rows on the right with token 0 / mask 0 to the longest row   -> tokens, mask: (4, N)
logits = model(tokens)                              # (4, N, V)
loss[r, t] = CE(logits[r, t], tokens[r, t+1])       # t = 0..N-2, reduction none
m = mask[:, 1:]                                     # shifted, so the first ending token is
                                                    # predicted from the last context token
sum_loss = (loss*m).sum(1);  avg_loss = sum_loss / m.sum(1)
pred      = argmin(sum_loss)     -> "acc"       (total log-prob of the ending)
pred_norm = argmin(avg_loss)     -> "acc_norm"  (mean loss per ending TOKEN)
```

Accuracy = correct / 10,042. Batch is the 4 rows of one item; right padding is harmless for a causal model. The training loop uses only `pred_norm` (helper `get_most_likely_row`), sharded across DDP ranks by `i % world_size`.

Differences from the Eleuther harness (primary, `lm_eval/tasks/hellaswag/utils.py` and `api/task.py`): the harness builds the query as `activity_label + ": " + ctx_a + " " + ctx_b.capitalize()`, strips WikiHow bracket artefacts (`" [title]"` -> `". "`, removes `[...]`), and its `acc_norm` divides the log-likelihood by `len(choice)` in characters, not tokens. So the two `acc_norm` values are different metrics that share a name.

## Numbers

All from the script's docstring (primary), full validation set, n = 10,042, zero-shot:

| Model | This script acc | This script acc_norm (per token) | Harness acc | Harness acc_norm (per char) |
|---|---|---|---|---|
| GPT-2 124M | 0.2859 | 0.2955 | 28.92% | 31.14% |
| GPT-2 XL 1558M | 0.3842 | 0.4893 | 40.04% | 50.89% |

Chance is 25%. My arithmetic, not the source: binomial standard error at p = 0.29, n = 10,042 is about 0.45 pp, so a 95% interval is roughly +/-0.9 pp. GPT-2 124M sits only about 4.5 pp above chance on this script, about 10 standard errors.

## Smallest scale tested, and whether the result held there

Smallest model reported is GPT-2 124M. Nothing below that is in the file or README. For a model of Koko's class (tens of millions of ternary weights, simple-English data) the expected result is close to chance; the script gives no evidence either way.

## Training recipe and gotchas

- No training involved in the eval. In `train_gpt2.py` the eval runs under bfloat16 autocast every 250 steps, and `torch.compile` is disabled because it "interferes with HellaSwag eval and Generation".
- Tokenizer dependence: `acc` (summed loss) is a total string log-probability and is comparable across tokenizers; `acc_norm` divides by token count, so a 16K vocabulary that splits endings differently from GPT-2's 50,257 BPE changes the ranking. Report both, plus a character-normalised variant to match the harness.
- The leading-space convention is GPT-2-specific; Koko's tokenizer needs its own equivalent so that the ending starts on a word boundary.
- No BOS is prepended; the first context token is never scored.
- Raw `ctx` keeps WikiHow markers such as `[title]`/`[step]`; a simple-English model will see these as junk. Either copy the harness's cleanup for both models or keep raw for both; never mix.
- Koko-specific: the comparison is only fair if GPT-2 124M is re-run through the same script with the same text preprocessing. Do not quote 29.55 against a Koko number computed with different normalisation.
- Out-of-vocabulary words in endings (ActivityNet/WikiHow vocabulary is not simple English) will fall to byte/character fallback tokens, inflating token counts and making per-token averages favour such endings.

## Mapping onto the GBA design

This is an off-device evaluation; nothing ships in ROM. It should be run in the PyTorch bit-exact simulator of the integer model, with float log-softmax applied to the integer logits only for scoring. With the two-level head, log p(token) = log p(class) + log p(member | class), so each scored token needs the class softmax and one member softmax, not the full 16K. Cost: one forward over context + ending per row, 4 rows per item; an incremental implementation can run the context once and branch the 4 endings from the cached ring-buffer state, which also checks that the 256-token ring buffer and top-8 attention behave under teacher forcing. An on-emulator run of all 10,042 items is possible but pointless beyond a small parity check (a few hundred items) between simulator and cartridge. The item lengths against the 256-token window were not measured here.

## Verdict

Adopt, as a secondary metric. The script is short, dependency-light and gives a reproducible GPT-2 124M number to check our harness against (target: reproduce 0.2859 / 0.2955 exactly before scoring Koko). But it should not be a headline metric: GPT-2 itself is barely above chance, the domain (video captions, WikiHow) is outside simple English, and a 1 pp difference is inside noise. Settling experiment: run GPT-2 124M and Koko through one modified copy that reports sum, per-token and per-character accuracy, with and without harness cleanup, and report paired differences with a bootstrap interval.

## Could not verify

- HellaSwag scores of Karpathy's own from-scratch 124M runs (I recall roughly 30-31% after 10B tokens and about 33% after 40B from the video lecture) are not in the README or the script; treat as unverified memory.
- The harness numbers (28.92 / 31.14, 40.04 / 50.89) are quoted by the docstring; I did not find them in a harness results table. Which harness version produced them is not stated.
- I confirmed the current harness normalises by `len(choice)` (characters); the task description's "byte-normalised" matches the separate `acc_bytes` metric, not `acc_norm`. Whether the 2024 harness version behaved identically was not checked.
- Token-length statistics of HellaSwag items were not computed.

## Links worth following

- https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py (how the eval is wired into training)
- https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/hellaswag (multiple-choice variant and preprocessing)
- https://github.com/rowanz/hellaswag (data files and field definitions)
