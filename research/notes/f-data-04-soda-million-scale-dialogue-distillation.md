# SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization (https://arxiv.org/abs/2212.10465, EMNLP 2023; Kim, Hessel, Jiang, West, Lu, Yu, Zhou, Le Bras, Alikhani, Kim, Sap, Choi; AI2 and others)

Data: https://huggingface.co/datasets/allenai/soda (CC BY 4.0). Tags: [P] = primary text (arXiv HTML and dataset card, read through a page-extraction tool, so wording of numbers is as extracted); [M] = my arithmetic or inference.

## What it is
SODA is 1,486,896 two-party English social dialogues written by GPT-3.5 (text-davinci-002), each grounded in one commonsense triple from the Atomic10x knowledge graph [P]. The pipeline (CO3) expands a triple into a sentence, then a 2-3 sentence narrative, then a conversation, and filters the result [P]. Human judges preferred SODA dialogues over DailyDialog and BlendedSkillTalk on most axes [P]. COSMO, a T5 (3B/11B) trained on SODA, beat BlenderBot, GODEL, Koala and Vicuna on out-of-domain DailyDialog [P]. For us it is both a ready-made chat corpus and a recipe for generating our own.

## Mechanism
Generation, per seed triple (head, relation, tail) [P]:
1. Sample a triple from Atomic10x, relations xReact, xIntent, xAttr, xEffect, xWant, xNeed. Replace PersonX/PersonY with common US names (top-1K Social Security names; a later pass re-names with top-10K names 1990-2021).
2. Template to a sentence (Table 8): xReact "[Head]. Now PersonX feels [Tail]."; xIntent "[Head] because PersonX wants [Tail]."; xAttr "PersonX is [Tail]. [Head]."; xEffect "[Head]. Now PersonX [Tail]."; xWant "[Head]. Now PersonX wants [Tail]."; xNeed "PersonX [Tail, past tense]. [Head]."
3. Narrative prompt: "[sentence] Rewrite this story with more specific details in two or three sentences:"
4. Speaker inference: "[narrative] The following is a conversation in the scene between [PersonX's name] and" (temperature 0, max 16 tokens).
5. Dialogue prompt: "[narrative] The following is a long in-depth conversation happening in the scene between [X] and [Y]" followed by the first speaker prefix "[X]:".
Decoding for steps 3 and 5: temperature 0.9, top-p 0.95, frequency penalty 1.0, presence penalty 0.6, max 1024 tokens.

Filters, applied to 2.2M raw dialogues [P]: malformed speaker patterns (6.3% removed); fewer than 4 or more than 20 turns (5.7%); more than two speakers (11.3%); non-human speaker (5.6%); Canary safety model (4.3%); Rewire toxicity above 0.5 (about 1%). 68.9% survive.

Record fields [P, dataset card]: head, relation, tail, literal, narrative, dialogue (list), speakers (list), PersonX/Y/Z, and four Yes/No/Unknown commonsense-check labels. Splits: 1.19M / 146K / 149K.

COSMO [P]: LM-adapted T5, input = narrative + speaker instruction + dialogue history, target = next response; narrative dropped 30% of the time and role instruction 50%; ProsocialDialog mixed in. Adafactor, constant lr 1e-3, batch 256 (effective about 780), 110K steps (3B) / 130K (11B) on TPUs.

## Numbers
- Size: 1.5M dialogues, over 11M utterances, about 300M tokens; 7.6 turns and 16.1 words per utterance on average; MTLD 68.0 against DailyDialog 63.0 (13K dialogues, 14.6 words), PersonaChat 43.6, BlendedSkillTalk 64.2 (Table 2) [P]. That is roughly 122 words and 200 tokens per dialogue [M].
- Relation mix: xReact 25%, xIntent 23%, xAttr 18%, xEffect 17%, xWant 11%, xNeed 7%. Common topics: friendship, help, support, family, car, school, work (Table 3) [P].
- Grounding: humans found the seed knowledge in 88% of 100 sampled pairs; GPT-3.5 judged 66% to contain the full triple (1,003,595 dialogues) [P].
- Contextless GPT-3.5 dialogues had MTLD 63.1 against 68.0 and lost the human comparison (Figure 3) [P].
- Cost: about $0.02 per dialogue [P]; about $44K for 2.2M raw [M].
- COSMO-3B overall win rate on DailyDialog (Table 5): 72% vs BlenderBot-3B, 86% vs GODEL, 71% vs Koala-7B, 58% vs Vicuna-7B, 55% vs the human reference [P]. COSMO-11B on SODA: 53% vs GPT-3.5, 50% vs ChatGPT (Table 7) [P].
- Human evaluation agreement: Krippendorff alpha 0.25 for dataset quality, 0.5 for responses [P].

## Smallest scale tested, and whether the result held there
The smallest student is 3B parameters. Nothing below that, no from-scratch training, no data-size ablation. Whether SODA helps a 10-100M-weight ternary model is untested here; the claim that transfers is about data quality (naturalness, diversity), not about small students.

## Training recipe and gotchas
- The two-step expansion (seed -> narrative -> dialogue) is what buys diversity; prompting for a dialogue directly gave blander, less specific text.
- 31% of raw generations were discarded, so budget about 1.45 raw dialogues per kept one [M].
- Speakers are named people or roles ("Priest", "coach"), not user/assistant. Dialogues assume a shared physical scene described only in the narrative, so without it some openings are ungrounded.
- 16.1 words per utterance is not simple English: vocabulary is unrestricted adult GPT-3.5 prose.
- Authors state it lacks scientific or historical knowledge, and is two-party only.
- Emotional tone leans to curiosity (12.9%), sadness (8.5%), joy (5.3%) (Table 4) [P]; expect many supportive "that sounds hard" exchanges [M].

## Mapping onto the GBA design
Pure training data: zero lookups, adds, multiplies, RAM or ROM at inference. Fit:
- Length: about 200 tokens per dialogue fits the 256-token ring buffer, so whole dialogues train in one window [M]. A 16-word reply is about 20-25 tokens, 4-8 s at 3-5 tokens/s [M]; prefer the shorter turns.
- Changes needed: (a) filter by our 16K vocabulary and a readability or word-list cut, or have an open LLM rewrite to simple English; (b) map speakers to two fixed role tokens, choosing which side is "Koko"; (c) keep the narrative as an optional prefix with COSMO-style dropout (30%), or drop it and discard dialogues that depend on it; (d) train on both sides, since user-side prediction is exactly the predictive-typing head.
- Recipe reuse: for our own generation with an open-weight LLM, seed with (topic, fact or commonsense triple) -> short simple-English scene -> dialogue, with the same structural filters. Atomic10x triples are a free seed source.
- ROM precompute: none from this source.

## Verdict
Adopt as a data source and as the generation template; test the dose. At about 300M tokens it would dominate a simple-English mix, so start with a vocabulary-filtered subset. Settling experiment: train the same small model with and without a filtered SODA slice (say 20% of chat-phase tokens) and compare held-out simple-chat bits per byte plus an LLM-judged chat probe; also compare raw-filtered SODA against LLM-simplified SODA.

## Could not verify
- Exact vote counts in Figures 2 and 3 (figures only).
- Whether the final 1.49M excludes dialogues failing the commonsense check: 1,003,595 passed the full check, so the released set appears to keep failures with labels [M].
- Tokenizer behind "300M tokens"; whether it counts narratives.
- Download size; the dataset card text was only partly visible.
- COSMO's exact separator tokens. No perplexity or other automatic metric is reported.
- What fraction of SODA survives a simple-English filter: not measured anywhere I found.

## Links worth following
- https://huggingface.co/datasets/allenai/soda
- https://github.com/skywalker023/sodaverse (code; not read)
- https://huggingface.co/allenai/cosmo-xl (COSMO-3B; not read)
- Atomic10x / Symbolic Knowledge Distillation: https://arxiv.org/abs/2110.07178 (not read)
