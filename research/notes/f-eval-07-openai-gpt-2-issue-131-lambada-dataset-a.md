# openai/gpt-2 issue #131: "Release raw lambada dataset" (https://github.com/openai/gpt-2/issues/131, May 2019 with 2021/2023 follow-ups; yaroslavvb, WuTheFWasThat of OpenAI)

Read through a page-to-text fetch of the issue (comments paraphrased by the fetch tool, short quotes only), plus three followed links. [P] = read in the primary page; [S] = secondary.

## What it is

A 12-comment GitHub thread in which Yaroslav Bulatov (cybertronai) tries to reproduce the GPT-2 paper's Table 3 LAMBADA accuracy for the 124M model (45.99) and cannot on the standard data. Jeff Wu (OpenAI) first points to the standard Zenodo plain-text release, then admits that was wrong and names OpenAI's own file, `lambada_test.jsonl`. With that file the number is reproducible, but only under a last-BPE-token scoring rule. The thread is the origin of the name "lambada-openai", now the `EleutherAI/lambada_openai` dataset and the `lambada_openai` task in lm-evaluation-harness.

## Mechanism

There are two independent choices, and each moves the GPT-2 small number by 5-14 points.

1. Which text file [P].
   - Official LAMBADA (Zenodo record 2630551): tokenised, lower-cased, "don't" stored as "do n't".
   - OpenAI file: `https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl` (the original `gs://gpt-2/data/` path is dead as of June 2021). One JSON object per line, `{"text": ...}`, natural formatting. 5,153 test passages (count from the HF card).
   - OpenAI's stated preprocessing [P]:
     ```python
     def preprocess(text):
         text = text.replace("“", '"')
         text = text.replace("”", '"')
         return '\n' + text.strip()
     ```
     This does nothing on the official set (no smart quotes there), so it does not explain the gap.

2. What counts as correct.
   - (a) Last-BPE-token match [P]: feed the passage minus its final BPE token, take the argmax, compare with the true final token. Batched, one forward pass.
   - (b) Whole-last-word generation [P, bflm commit]: split off the last word, generate 6 BPE tokens (3 before the change), take the first word of the continuation, compare case-insensitively. The issue text describes this as length-5 beam search used for stop-word removal.
   - (c) lm-evaluation-harness `lambada_openai` [P, YAML]: context = `text.split(' ')[:-1]` joined, target = `' ' + last word`; `acc` = greedy decoding reproduces every target token; `perplexity` = exp of mean target-word negative log-likelihood. No stop-word filter.

## Numbers

All GPT-2 small (124M), LAMBADA test.

| File | Scoring | Accuracy | Source |
|---|---|---|---|
| Zenodo post-processed | not stated | about 31% | issue, 15 May 2019 [P] |
| Official | not stated | about 34% (33.63 in bflm code comment) | issue, 23 May [P] |
| Official plain text | last BPE token | 35 | bflm `eval_lambada.py` comment [P] |
| OpenAI jsonl | whole word, (b) | 41.98 (42.01 with 6-token search) | issue 29 May; bflm comment [P] |
| OpenAI jsonl | last BPE token, no flags | 46.67 | bflm comment [P] |
| OpenAI jsonl | last BPE token, preprocess + leading newline | 46.89 | issue 29 May [P] |
| OpenAI jsonl | harness (c) | 32.56 +/- 0.65; perplexity 40.06 +/- 1.49 | HF gpt2 discussion #90, May 2024 [S, one user's run] |
| Paper Table 3 | stop-word filter | 45.99; perplexity 35.13 | GPT-2 paper, via our f-eval-01 note |

So: file choice is worth about 12 points under token scoring (35 to 46.67); scoring rule is worth about 5 points on the same file (41.98 to 46.89); the leading newline is worth 0.2.

## Smallest scale tested, and whether the result held there

Only GPT-2 small (124M) is measured in the thread. Nothing smaller. The closest reproduction of 45.99 is 46.89, i.e. 0.9 points high; the paper's exact procedure (stop-word filter) is never reproduced exactly in the thread.

## Training recipe and gotchas

No training. Evaluation gotchas:
- Last-BPE-token scoring is tokenizer-dependent. It is easier whenever the final word spans several tokens, because the earlier pieces are given. A 16K vocabulary splits more words than GPT-2's 50K, so Koko would be flattered by rule (a). Whole-word scoring (b or c) is the only tokenizer-fair rule.
- The harness figure for GPT-2 small on the same file is 32.56, not 46. Our prelim line "~31 on lambada_openai" is this harness-scoring effect, not a different file.
- The two "41.98" code comments in bflm carry different flag labels; which configuration produced it is ambiguous.
- Passages are book text, about 4-5 sentences; many exceed a 256-token window, which matters for Koko more than for GPT-2 (1024).

## Mapping onto the GBA design

Zero run-time cost: this is an offline benchmark protocol, nothing goes in ROM or RAM. Run it on the PC through the bit-exact integer simulator of the cartridge model.
- Rule (c) needs only teacher-forced log-probabilities of the target tokens through the two-level head: P(token) = P(class) * P(member | class), and "greedy correct" means argmax class and argmax member both match at every target token.
- Context must be truncated to the last 256 tokens (ring buffer). Report that truncation; also run GPT-2 small with a 256-token window as a like-for-like baseline.
- The same harness doubles as the predictive-typing metric: whole-next-word top-1 and top-3 accuracy is what a suggestion bar shows.

## Verdict

Adopt as protocol, not as a headline. Use the `lambada_openai` file (SHA-256 matches OpenAI's, per the HF card), score with the harness whole-word rule, and quote GPT-2 small at 32.56 / perplexity 40.06 from our own re-run, never 45.99. Optionally add rule (a) as a secondary line labelled tokenizer-dependent. Settling experiment: one afternoon running GPT-2 small on the file under rules (a), (b), (c), with 1024 and 256 token contexts, and confirming 46.7-46.9 / about 42 / about 32.6.

## Could not verify

- Exact comment wording: the fetch tool paraphrased; numbers and URLs were cross-checked across two fetches.
- The body of bflm's `predict` function (beam width, stop-word list) was not in the diff I read; "length-5 beam" is from the issue text only.
- Whether OpenAI's 45.99 used token or word matching: not stated anywhere in the thread.
- The 32.56 harness figure is a single third-party run (old harness, task version 0); I found no official table with it.
- I did not open the jsonl file itself or `lambada_comparison.txt`.
- How OpenAI built its file (presumably re-extracted from the raw book text) is not explained.

## Links worth following

- https://huggingface.co/datasets/EleutherAI/lambada_openai (card: 5,153 English rows, test split only, checksum match)
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/lambada/lambada_openai.yaml (scoring rule c)
- https://github.com/cybertronai/bflm/commit/5b8f6beb1e9e582650a82bbac6eedf3f3689909f (rules a and b, code comments with numbers)
- https://huggingface.co/openai-community/gpt2/discussions/90 (harness run: 32.56 / 40.06)
- https://zenodo.org/record/2630551 (official LAMBADA)
