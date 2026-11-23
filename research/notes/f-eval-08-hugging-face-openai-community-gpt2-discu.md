# GPT2 Reproduce results with lm-evaluation-harness (HF discussion #90)

https://huggingface.co/openai-community/gpt2/discussions/90 — opened 2024-05-07 by user david5819 ("David Y."), one reply 2024-10-17 by bedio; status still open. Read in primary form through the Hub API JSON (https://huggingface.co/api/models/openai-community/gpt2/discussions/90), so the commands and numbers below are verbatim [P]. Items marked [S] are secondary; [M] is my own arithmetic or recollection.

## What it is
A two-post thread on the GPT-2 124M model page. The opener ran EleutherAI's lm-evaluation-harness on `gpt2` and could not reproduce the model card's LAMBADA accuracy (45.99): he got 32.56% and perplexity 40.06, byte-identical on two harness commits with the cache deleted in between. He did reproduce the Open LLM Leaderboard's ARC-Challenge 25-shot number to within about one standard error. Nobody answered; the only reply asks the same question and adds WikiText. Its value to us is a pinned, re-runnable GPT-2 reference under the harness we will use.

## Mechanism
Exact command [P] (old `main.py` interface, harness commits written as `441e6ac1`/`4416ac1` and `b281b09`; the first two spellings differ in the post, probably a typo):

```
python main.py --model=hf-causal-experimental \
  --model_args="pretrained=gpt2,use_accelerate=True" \
  --tasks=lambada_openai --num_fewshot=0 --batch_size=2 --output_path=output
```
Config echoed in the output: `limit: null`, `bootstrap_iters: 100000`, task version 0.

Task definition, read from the current `lambada_openai.yaml` in the harness repo [P, but current file is version 1.0, the thread ran version 0]:
- dataset `EleutherAI/lambada_openai`, test split; `output_type: loglikelihood`
- context = `text.split(' ')[:-1]` joined by spaces; target = `' ' + text.split(' ')[-1]`
- metrics: `perplexity` and `acc` (mean)

Scoring as I understand the harness [M, not re-read this session]:
```
for each passage:
    ll, is_greedy = loglik(model, context, target)   # sum over target's tokens
    acc_i = is_greedy        # every target token is the argmax, teacher-forced
ppl = exp(-mean(ll))         # per final WORD, not per token
acc = mean(acc_i)
```
No stop-word filter, no candidate list, no detokenizer. The dataset card says the data is "the LAMBADA test split as pre-processed by OpenAI" from `openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl` [P].

## Numbers
All GPT-2 124M, from the post [P]:

| Task | Setting | Metric | Value | Stderr |
|---|---|---|---|---|
| lambada_openai | 0-shot | acc | 0.32564 | 0.00653 |
| lambada_openai | 0-shot | ppl | 40.055 | 1.488 |
| arc_challenge | 25-shot | acc | 0.20051 | 0.01170 |
| arc_challenge | 25-shot | acc_norm | 0.21928 | 0.01209 |
| arc_challenge (leaderboard details_gpt2) | 25-shot | acc / acc_norm | 0.19710 / 0.22014 | 0.01163 / 0.01211 |

- Model card / GPT-2 paper Table 3: LAMBADA acc 45.99, ppl 35.13 (see f-eval-01).
- Gap: 13.4 points of accuracy, 4.9 of perplexity.
- [M] 0.00653 = sqrt(0.3256 x 0.6744 / n) gives n of about 5,153, the full test set, so no subsampling.
- [M] ARC-Challenge is 4-way: 20-22% is chance (25%) or below. GPT-2 gives no signal there.

## Smallest scale tested, and whether the result held there
Only GPT-2 124M. The harness number is stable across commits; the card number is not reproduced. The ARC reproduction shows the setup itself is sound, so the LAMBADA gap is a protocol difference, not a broken install.

## Training recipe and gotchas
No training. Evaluation gotchas:
- Likely cause of the gap [S]: a web-search summary of arXiv:2203.07472 says GPT-2's reported accuracy used a stop-word filter worth about 11 points. The thread itself gives no explanation. The GPT-2 paper states the filter for the 1542M model (52.66 to 63.24); whether 45.99 is filtered is not stated (f-eval-01).
- Accuracy needs ALL sub-tokens of the last word to be greedy. A 16K vocabulary splits more words into several tokens than GPT-2's 50K, which lowers our accuracy for a tokenizer reason. Word-level perplexity is tokenizer-independent, so report both.
- Target is `' ' + word`: our tokenizer must handle a leading-space word the same way or the score is silently wrong.
- LAMBADA passages average roughly 75 words [M, from the original paper, not re-checked]; they fit Koko's 256-token ring only if our tokens-per-word stays low. Count truncated passages and report them.
- Record harness commit and task version (0 vs 1.0) next to every number.

## Mapping onto the GBA design
Zero run-time cost: this is an offline protocol. What it asks of the toolchain:
- A bit-exact Python/C simulator of the integer GBA forward pass, wrapped as an lm-eval `LM` subclass exposing `loglikelihood`. Log-probabilities come from the two-level head: log p(class) + log p(member | class), computed in float offline from the integer logits. No ROM, RAM or cycle impact.
- Evaluate with the same ring-buffer length and top-8 attention as the cartridge, not the float teacher.
- Stderr 0.0065 means two models must differ by about 1.8 points (2 x sqrt(2) x 0.0065) to be separated unpaired; use paired per-item comparison for ablations.
- Predictive typing: lambada acc is exactly "top-1 next-word hit after long context", so it doubles as a keyboard-suggestion metric. Add top-3 offline.

## Verdict
Adopt as the GPT-2 comparison line: 32.56% acc / 40.06 ppl on `lambada_openai` 0-shot, not 45.99. Drop ARC-Challenge as a headline (GPT-2 is at chance). Settling experiment: rerun `gpt2` ourselves on a current harness release (`lm_eval --model hf --model_args pretrained=gpt2 --tasks lambada_openai,lambada_standard`), confirm 0.3256 within one stderr, then run Koko's simulator through the same call.

## Could not verify
- Why the card says 45.99: unanswered in the thread; the stop-word explanation is [S] from a search snippet, not read in the paper.
- Whether task version 1.0 gives the same number as version 0.
- The linked leaderboard comment (open_llm_leaderboard discussion 60) and `details_gpt2` dataset: not opened.
- The harness's is-greedy and perplexity aggregation code: recalled, not re-read.
- No WikiText reproduction exists in the thread (only requested).

## Links worth following
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/lambada/lambada_openai.yaml
- https://huggingface.co/datasets/EleutherAI/lambada_openai
- https://arxiv.org/pdf/2203.07472 (claimed source of the stop-word-filter explanation)
- https://huggingface.co/datasets/open-llm-leaderboard/details_gpt2 (GPT-2 on the other leaderboard tasks)
