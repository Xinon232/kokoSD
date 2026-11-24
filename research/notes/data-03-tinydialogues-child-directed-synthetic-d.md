# Is Child-Directed Speech Effective Training Data for Language Models? (TinyDialogues) (arXiv:2408.03617v2, EMNLP 2024; Feng, Goodman, Frank, Stanford)

Correct URL: https://arxiv.org/abs/2408.03617; the assigned ID 2408.11208 is another paper (PooDLe). Data: https://huggingface.co/datasets/styfeng/TinyDialogues (MIT). Tags: [P] = primary text via a page-extraction tool; [M] = my arithmetic; [S] = secondary.

## What it is
TinyDialogues (TD) is 129,732 GPT-4-written conversations (28.8M words) in which a child of seed age 2, 5, 10 or 15 talks with family, teachers or friends [P]. The paper trains GPT-2 small and RoBERTa base from scratch on 29M-word corpora (CHILDES, TD, Wikipedia, OpenSubtitles, a BabyLM sample) and compares grammar (Zorro) and word similarity (WS) [P]. Findings: synthetic dialogue matches or beats real child-directed speech; age-ordered curricula do nothing; a diverse mix is best for GPT-2 [P]. For us: ready-made simple-English chat data plus a generation recipe.

## Mechanism
Generation (Appendix A) [P]: model gpt-4-1106-preview. Prompt: "Please construct a realistic, approximately {5, 10}-turn dialogue directly involving a {2, 5, 10, 15}-year-old {toddler, child, teenager} as a participant", plus:
- vocabulary limited to what a child of that age understands;
- type in {explanatory, functional, narrative, argumentative};
- must use one sampled noun, verb and adjective: Wordbank CDI for ages 2 and 5, Kuperman age-of-acquisition ratings cut at the seed age for 10 and 15;
- other participants from an age-specific list (age 2: mom, dad, older sibling, babysitter; older ages add teacher, friend, grandparent, coach and others): one for 5-turn, two for 10-turn dialogues;
- output sections PARTICIPANTS:, SETTING:, DIALOGUE:.

Training format [P]: `**Mom**: text \n\n **Child**: text ... <|endoftext|>`, one conversation per line; child labels normalised to "Child". Metadata files keep type, participants and setting.

Models [P]: GPT-2 small (124M), lr 1e-4, 20 epochs, 1024-token chunks; RoBERTa base (125M), lr 5e-5, 50 epochs. Adam, linear decay, no warmup, three seeds. Per-dataset tokenizer; the repository script retrains GPT-2's to 52,000 entries [P, code]. Split 85/15 (about 24.5M / 4.5M words).

Curriculum test [P]: b age buckets, n passes over each before moving on; age vs reverse vs random order.

## Numbers
Dataset (Appendix C) [P]:

| Age | conversations | words | unique words | utterances/conv. | words/utterance |
|---|---|---|---|---|---|
| 2 | 43,697 | 7,183,704 | 16,269 | 15.75 | 8.32 |
| 5 | 33,248 | 7,194,902 | 15,534 | 14.01 | 13.36 |
| 10 | 27,198 | 7,196,356 | 42,508 | 13.61 | 17.35 |
| 15 | 25,589 | 7,199,752 | 73,484 | 12.88 | 19.77 |

Words per conversation: 164 / 216 / 265 / 281 [M]. CHILDES: 3.85 words per utterance.

GPT-2 small, 20 epochs, 3 seeds (Table 1) [P]:

| Data (29M words) | Zorro | WS |
|---|---|---|
| CHILDES | 78.29 ± 0.51 | 0.24 |
| TinyDialogues | 78.48 ± 0.82 | 0.42 |
| Wikipedia | 78.16 ± 0.61 | 0.32 |
| OpenSubtitles | 81.02 ± 1.03 | 0.38 |
| BabyLM mix | 82.90 ± 1.01 | 0.42 |

- RoBERTa (Table 2): TD 78.52 Zorro vs CHILDES 58.37, Wikipedia 60.84, BabyLM 59.43 [P].
- Global order, TD, n=10 (Table 3): age 78.16, reverse 77.71, random 79.53 [P].
- Shuffling utterances within a conversation (Table 4): TD 78.48 to 78.38, WS unchanged; CHILDES WS 0.24 to 0.19 [P].
- Removing speaker labels (Table 20): TD 78.48 to 77.37, CHILDES 78.29 to 76.61; Zorro differences significant [P].
- Passes (Table 17, one seed): TD Zorro 71.51 (n=3), 74.48 (5), 79.21 (10), 79.65 (20) [P].

## Smallest scale tested, and whether the result held there
Only 124M-125M-parameter models on 24.5M training words. Nothing near 1-20M parameters; no generation, chat or perplexity evaluation. Untested at our scale.

## Training recipe and gotchas
- Many passes are needed on 29M words: 3 gave 71.5 Zorro, 20 gave 79.7.
- GPT-4 overshot: asked for 5 and 10 turns, it wrote about 10 and 20 [P].
- Noise: quotation marks around some utterances, stage directions (*giggles*), asterisks in labels [P, dataset viewer].
- No de-duplication or diversity statistics; quality control was a manual scan [P].
- 10-turn dialogues have three speakers; a two-party chatbot must filter or remap. The "user" side is a simulated child.
- Inconsistencies: Table 19 prints WS 0.52/0.54 where Table 1 gives 0.24/0.42; Table 14 marks p = 0.42 as significant.

## Mapping onto the GBA design
- Run-time cost: zero; this is training data.
- Speaker labels become single role tokens and "\n\n" a turn token: one ROM embedding lookup each. The +1.1 Zorro from labels supports keeping them.
- Role tokens give predictive typing: after its turn the model emits the user-role token and continues as the user, since both sides are trained.
- Ages 2 and 5 (14.4M words, about 16K word types each) suit a 16K vocabulary; ages 10 and 15 (42K-73K types) will fragment.
- Conversations of 164-281 words are close to the 256-token ring buffer [M].
- Age and type labels could key a ROM bias vector for register control.
- No curriculum needed: shuffle.

## Verdict
Adopt as one ingredient of the mix, and adopt the recipe (age x type x participants x seeded noun/verb/adjective, kept as labels) for our own generation. Do not train on dialogue alone: the mix won by 4.4 Zorro points. Experiment: train the student on (a) stories, (b) stories + TD, (c) stories + TD + Simple Wikipedia at equal token budgets; compare bits per character on held-out chat, BLiMP and judged replies.

## Could not verify
- Generation cost, temperature, share of two-party dialogues, token counts, metadata field names.
- Tokenizer size is from the repository script, not the paper.
- The extraction tool paraphrased parts of the prompt; PDF not viewed directly.
- Follow-up "Dialogue Is Not Enough to Make a Communicative BabyLM" (arXiv:2510.20358): abstract summary only [S].

## Links worth following
- https://huggingface.co/datasets/styfeng/TinyDialogues
- https://github.com/styfeng/TinyDialogues
- https://arxiv.org/abs/2510.20358
