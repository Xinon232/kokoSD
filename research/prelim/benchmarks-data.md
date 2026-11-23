## Tiny-chatbot benchmarking and data: research report

Tags: **[P]** = read in the primary source (paper, official card); **[S]** = secondary (search summary or third-party card); **[NF]** = not found.

### 1. BabyLM 2024 / 2025
- **2024 suite [P]:** BLiMP, BLiMP supplement, EWoK, (Super)GLUE fine-tuning, plus multimodal tasks. https://arxiv.org/html/2412.05149
- **2024 winner [P]:** GPT-BERT, a hybrid causal + masked objective on LTG-BERT, won both text tracks.

| Track | System | BLiMP | Supp. | GLUE | EWoK |
|---|---|---|---|---|---|
| 100M | GPT-BERT | 86.1 | 76.8 | 81.5 | 58.4 |
| 10M | GPT-BERT | 81.2 | 69.4 | 76.5 | 54.6 |
| 10M | BabyLlama baseline (58M) | 69.8 | 59.5 | 63.3 | 50.7 |

- **What helped in 2024 [P]:** objective changes, new or synthetic data, architecture changes, and more training FLOPs. Curriculum learning and linguistic-bias methods did not help. Standard BPE beat phoneme/grapheme tokenizers. EWoK stayed near chance (best 58.4).
- **2025 [P]:** added an Interaction track and new tasks: entity tracking, wug tests (past tense, adjective nominalisation), reading-time prediction, age-of-acquisition word learning, COMPS. https://aclanthology.org/2025.babylm-main.28.pdf
- **2025 winners [P]:** Simple-Diffusion (strict, NLP score 58.4), AMLM-Hard-Decay (strict-small, 58.3; masks the hardest tokens), BLM (interaction, 54.4), CLASS-IT and MoEP (human-likeness). Strict-track entries did not beat the GPT-BERT baselines (NLP 60–63). FLOPs predicted performance less well than in 2024. Per-task BLiMP numbers for 2025 are only in figures; I did not extract them.
- **GPT-2 on BLiMP [P]:** 83.0 for GPT-2-large; the paper states all four sizes (117M–1558M) score 84 ± 1. https://arxiv.org/pdf/1912.00582
- **GPT-2 on EWoK [NF]:** the paper's table A4 was not readable. Humans 95, best model 80. https://arxiv.org/html/2405.09605
- **Tiny models trained on 10M words [P]:** GPT2-18M scores BLiMP 63.7 / EWoK 49.7; Llama-20M 62.8 / 50.2. https://aclanthology.org/2024.conll-babylm.27.pdf

### 2. Reference numbers

| Model | HellaSwag | ARC-Easy | LAMBADA acc | Perplexity | BLiMP |
|---|---|---|---|---|---|
| GPT-2 124M | ~31 (acc_norm) [S] | ~39.5 [S] | 45.99 [P]; ~31 on lambada_openai [S] | WikiText-103 37.50 [P] | ~84 [P] |
| SmolLM2-135M | 42.1 [P] | 43.9 (ARC average) [P] | NF | NF | NF |
| MobileLLM-125M | 38.9 [P] | 43.9 [P] | NF | NF | NF |
| GPT-neo-125M | 29.7 [P] | 40.7 [P] | – | – | – |
| Super Tiny LM 50M | 30 [P] | 39 [P] | – | – | 78 [P] |
| Brandon-Tiny 10.7M | 32.4 [P] | 30.6 [P] | 8.8 [P] | WikiText-2 224 [P] | 73.3 [P] |

- **GPT-2 124M, other [P]:** LAMBADA perplexity 35.13, WikiText-2 29.41, enwik8 1.16 bits/byte, text8 1.17 bits/char, all zero-shot with de-tokenizers. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **DistilGPT2 (82M) [P]:** WikiText-103 perplexity 21.1 after fine-tuning, against 16.3 for GPT-2. https://huggingface.co/distilbert/distilgpt2
- **Brandon-Tiny caveat [P]:** self-reported on 1,000-example subsets. Its paper quotes Super Tiny LM at 25.6 / 21, which disagrees with that paper's own table (30 / 39). https://xaskasdf.github.io/brandon-tiny/paper.html
- **TinyStories 1M–33M [NF]:** no standard-benchmark numbers; the paper reports GPT-4 grades only. The paper says the 1M model answers no factual prompt correctly.
- Sources: https://huggingface.co/HuggingFaceTB/SmolLM2-135M, https://arxiv.org/html/2402.14905, https://arxiv.org/html/2405.14159v2

### 3. Datasets

| Dataset | Size | Licence | Chat format? |
|---|---|---|---|
| TinyStories / V2 (GPT-4 only) | 2.14M stories | CDLA-Sharing-1.0 [P] | No |
| TinyStories-Instruct | stories preceded by instructions (words, summary, sentence, features) [P] | size and licence not checked | Instruction, not chat |
| SimpleStories | 2.1M English stories, gpt-4o-mini | MIT [P] | No |
| TinyDialogues | ~29M words, synthetic child-directed dialogue [S] | NF | Yes |
| BabyLM corpus | 100M words: CHILDES 29M, Gutenberg children's 26M, OpenSubtitles 20M, Simple Wikipedia 15M, BNC dialogue 8M, Switchboard 1M [P] | mixed | Partly dialogue |
| Simple English Wikipedia | ~15M words in the BabyLM cut [P] | CC BY-SA (from memory) | No |
| SODA | 1.49M dialogues | CC BY 4.0 [P] | Yes |
| DailyDialog | 13,118 dialogues | CC BY-NC-SA 4.0, non-commercial [P] | Yes |
| smol-smoltalk | 485K conversations | Apache-2.0 [P] | Yes |
| everyday-conversations-llama3.1-2k | 2.4K short chats | Apache-2.0 [P] | Yes |
| SmolLM-Corpus | Cosmopedia v2 39M docs (Mixtral synthetic textbooks and stories); FineWeb-Edu-dedup 190M docs, ~220B tokens | ODC-By [P] | No |
| FineWeb | 10BT / 100BT / 350BT samples | ODC-By [P] | No |
| OpenWebText | not checked | not checked | No |

- **Simple-English factual Q&A [NF]:** no ready-made training set found. You will need to generate one.

### 4. Distillation
- **Logits versus teacher-generated text at 1M–30M [NF]:** I found no controlled comparison. Evidence is indirect.
- **BabyLlama [P]:** logit distillation from an ensemble (GPT2-705M + Llama-360M) into a 58M student beat both teachers and a no-distillation run on 10M words. https://arxiv.org/abs/2308.02019
- **Smaller teachers [P]:** teachers of 44M and 60M gave a 44M student within ~3 BLiMP points (66.8 against 69.8). Raw teacher size barely mattered: GPT2-18M to 705M scored 63.7–66.1 on BLiMP.
- **Capacity gap [P, abstract]:** the optimal teacher scales roughly linearly with student size; too-large teachers hurt. A student-to-teacher ratio of about 0.4 is [S]. https://arxiv.org/abs/2311.07052
- **Distillation Scaling Laws (Apple) [P, abstract]:** distillation beats supervised training only when a teacher already exists or serves many students, and only up to a compute threshold. https://arxiv.org/abs/2502.08606
- **Brandon-Tiny [P]:** 30M teacher to 10.7M student (2.8x), reverse KL, temperature 2.0, alpha 0.5. Pretraining only on synthetic text gave the lowest pretrain loss but poor fine-tune loss (3.62 against 2.40).
- **DistilGPT2 [P]:** DistilBERT-style recipe on OpenWebText.
- **TinyStories:** data-only "distillation" (training on GPT-3.5/4 text).

### 5. Tokenization
- **TinyStories [P]:** GPT-Neo tokenizer cut to the top 10K tokens.
- **Brandon-Tiny [P]:** 8,192-token SentencePiece BPE.
- **Super Tiny LM [P]:** a 50K vocabulary's embeddings take ~25M of 50M parameters.
- **BabyLM 2025 [P]:** tokenization was often higher-leverage than architecture; morphology-aware tokenizers helped EWoK.
- **Vocabulary size [S, mixed]:** one 2025 entry found 8K SuperBPE best; a 2023 entry found 40K unigram best; another 2023 entry found 10K BPE sub-par.
- **Word-level versus subword under 30M [NF]:** no controlled study found.
- Your 16K sits inside the evidenced 8K–16K range.

### 6. Evaluation of tiny story/chat models
- **TinyStories GPT-Eval [P]:** ~50 hand-written story openings ending mid-sentence, 10 completions each at temperature 1. GPT-4 writes an assessment, then grades grammar, creativity and consistency out of 10, plus an age group. The Instruct variant adds a plot score. Grammar saturates first; consistency appears at hidden size ≥128. https://arxiv.org/pdf/2305.07759
- **Standard judge rubric for tiny chatbots [NF]:** none found. SmolLM2 reports MT-Bench (1.98 for 135M-Instruct) and IFEval (29.9) [P].

### Recommended benchmark suite
1. BLiMP and BLiMP supplement (GPT-2 ≈ 84 on BLiMP).
2. EWoK; expect about 50, and run GPT-2 yourself.
3. HellaSwag and ARC-Easy via lm-eval (GPT-2 ≈ 31 / 39.5).
4. LAMBADA accuracy.
5. Bits per byte on WikiText-103 and a TinyStories validation set; this is comparable across tokenizers (GPT-2: 1.16 on enwik8).
6. TinyStories-style LLM-judged completions, with GPT-2 scored by the same judge.
7. A custom chat probe: name recall after N turns, a capitals/everyday-facts list, and BabyLM's entity-tracking task.

Rerun GPT-2 on all of these yourself, because published settings differ.

### Recommended data mix and curriculum (my synthesis, not a published recipe)
- **Phase 1, language:** TinyStoriesV2 + SimpleStories + TinyDialogues + Simple Wikipedia + BabyLM dialogue portions, with a small Cosmopedia slice. Keep real text in the mix; synthetic-only pretraining hurt Brandon-Tiny downstream.
- **Phase 2, distillation (optional):** train a teacher of about 3–10x the student on the same data, then distil logits (temperature 2, alpha 0.5; forward or reverse KL). Do not use a giant teacher.
- **Phase 3, chat fine-tune:** everyday-conversations + smol-smoltalk filtered to short turns + SODA + self-generated simple factual Q&A and name-recall dialogues. Compute loss on assistant turns only, add a little pretrain replay, and use anti-repetition regularisation.