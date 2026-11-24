# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

https://arxiv.org/abs/2306.05685 (2023, NeurIPS D&B). Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica (UC Berkeley, UCSD, CMU, Stanford). Code: FastChat `fastchat/llm_judge`.

Read status: full text read through the ar5iv HTML mirror via a fetch tool that summarises; table numbers below were extracted from that text. Judge prompts were read from the FastChat repo but only in paraphrase (not verbatim).

## What it is
The paper asks whether a strong LLM can replace human raters for open-ended chat quality. It builds MT-Bench (80 hand-written two-turn questions, 8 categories x 10) and uses Chatbot Arena votes (about 30K) as human ground truth. It catalogues judge failure modes (position bias, verbosity bias, self-enhancement, weak math grading), proposes mitigations, and measures judge-human agreement. Headline: GPT-4 agrees with humans as often as humans agree with each other (over 80% on non-tie votes).

## Mechanism
Three judge modes:
1. Pairwise: judge sees question + answer A + answer B, outputs `[[A]]`, `[[B]]` or `[[C]]` (tie) after a short explanation. Criteria in prompt: helpfulness, relevance, accuracy, depth, creativity, detail; prompt tells judge to ignore order and length.
2. Single-answer grading: judge rates one answer 1-10, output `[[rating]]`. FastChat default; final score = mean over all turns and questions.
3. Reference-guided: same, plus a reference answer (used for math/reasoning).

Mitigations to implement:
- Position swap: run every pair in both orders; a win counts only if it wins both; anything else = tie ("conservative").
- Multi-turn: put the whole conversation in one judge prompt, grade the last turn (per-turn prompts caused wrong-turn references).
- Few-shot examples: raise consistency but cost about 4x.
- Chain-of-thought / reference for anything with a right answer.

Agreement metric: probability that two different judges agree on a random question. S1 = all votes, inconsistent counted as tie (chance 33%). S2 = non-tie votes only (chance 50%).

## Numbers
Position bias, pairwise, same answer pairs in both orders (Table 2; GPT-3.5 vs Vicuna-13B answers):
- GPT-4 default: 65.0% consistent, 30.0% favour first, 5.0% second.
- GPT-3.5 default: 46.2% / 50.0% / 1.2%. Claude-v1 default: 23.8% / 75.0% / 0.0%.
- Few-shot (Table 13): GPT-4 65.0 -> 77.5%, Claude-v1 23.8 -> 63.7%, GPT-3.5 46.2 -> 55.0%.
- By category (Table 11): consistent on math 86%, coding 86%, but writing 42%, humanities 36%.
- By quality gap (Table 12): GPT-3.5 vs LLaMA-13B 98.8% consistent; GPT-3.5 vs Claude-v1 67.5%.

Verbosity ("repetitive list" attack, 23 answers, Table 3): failure rate GPT-4 8.7%, GPT-3.5 91.3%, Claude-v1 91.3%.

Math grading by GPT-4, failures = wrong answer accepted (Table 4, 10 questions x 2 orders): default 14/20, CoT 6/20, reference 3/20.

Self-enhancement (Fig. 2b): GPT-4 favours itself by about 10% win rate, Claude-v1 about 25%; authors call the data too thin to conclude.

Agreement, MT-Bench first turn, about 3K votes from 58 expert labelers (Table 5a), S1 / S2:
- GPT-4 pairwise vs human 66% / 85%; GPT-4 single vs human 60% / 85%; human vs human 63% / 81%; GPT-4 pair vs GPT-4 single 70% / 97%.
- Second turn similar (66/85, 59/84, human-human 67/82).
- Arena 3K votes (Table 6): GPT-4 pair vs human 64% / 87%; GPT-3.5 54% / 83%; Claude 53% / 84%.
- Agreement rises from about 70% to near 100% as the win-rate gap between the two models grows (Table 7 / figure).

MT-Bench single-grade scores (Table 9): GPT-4 8.99, GPT-3.5 7.94, Vicuna-13B 6.39, Alpaca-13B 4.53, LLaMA-13B 2.61, LLaMA-7B 2.74. (Our prelim notes record SmolLM2-135M-Instruct at 1.98 — from the SmolLM2 paper, not this one.)

Fine-tuned Vicuna-13B judge (App. F): trained on 20K arena votes as 3-way classification; consistency 65.0%; accuracy 56.8% with ties, 85.5% without (GPT-4: 66% / 87%). Zero-shot Vicuna-13B: 15.0% consistent, 22.5% unparseable.

## Smallest scale tested, and whether the result held there
Smallest judged models are 7B-13B (LLaMA, Alpaca, Vicuna, Dolly-12B). Nothing near our size. Unbase LLaMA-13B already sits at 2.61/10, so MT-Bench has a floor effect well above a GPT-2-small-class model: the 80 questions are useless for us as-is. The method (not the question set) is what transfers; judge reliability on very weak, often incoherent answers was not measured.

## Training recipe and gotchas
No model training except the judge fine-tune above. Gotchas:
- Weak judges are badly order-biased and can be gamed by length; an open-weight local judge needs the swap rule and probably few-shot examples or fine-tuning.
- Prompt wording changes bias direction (renaming assistants flipped Claude to favour second).
- Position bias is worst exactly where answers are close and subjective (writing, humanities) — the regime Koko vs GPT-2 small may be in.
- Single-grade scores are cheaper (N calls, not N^2) but agree less with humans on S1 (60% vs 66%).
- Chat fine-tuning data dominates MT-Bench score: Vicuna-7B with 4.8M selected tokens scores 5.95 vs LLaMA-7B 2.74 (Table 9), while MMLU barely moves.

## Mapping onto the GBA design
Zero runtime cost: no lookups, multiplies, RAM or ROM. This is offline evaluation only. What to build:
- A Koko-level question set (e.g. 80-200 simple-English two-turn prompts in our own categories: chit-chat, simple facts, story, instructions, follow-up/coreference), since MT-Bench questions are far too hard.
- Pairwise Koko vs GPT-2 small (and vs checkpoints), both orders, inconsistent = tie; report S2-style win rate plus tie rate. Also single 1-10 grades for cheap checkpoint tracking.
- Reference answers for any factual/arith items.
- Judge must be a strong model; if a local open-weight judge is used, first measure its swap consistency on our pairs (GPT-4 got only 65%).
- Table 9 supports spending training budget on conversation-format data: format learning needs few tokens.

## Verdict
Adopt (as eval protocol, not architecture). Use position-swapped pairwise judging against GPT-2 small as the headline chat metric, with single grading for ablations. Settling experiment: hand-label about 200 Koko-vs-GPT-2 pairs ourselves, compute S1/S2 agreement and swap consistency for the chosen judge; accept the judge if S2 agreement is at least about 80% and consistency at least about 65%.

## Could not verify
- Exact verbatim judge prompt text (paraphrase only).
- Table 7 exact per-bin values and Table 6 vote counts.
- Generation temperatures per category (not in README; reportedly in FastChat code, not read).
- Any evidence about judge reliability on sub-1B models: not in this paper.
- The paper's text cites "Table 4" for the verbosity result while the data is in Table 3 (numbering mismatch in the ar5iv version).

## Links worth following
- https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge (prompts in `data/judge_prompts.jsonl`, `gen_judgment.py`, `show_result.py`)
- Hugging Face dataset `lmsys/mt_bench_human_judgments` (3.3K human votes, for validating a local judge)
- TinyStories (arXiv 2305.07759) for LLM-graded evaluation of very small models
