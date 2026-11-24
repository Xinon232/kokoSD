# ACUTE-Eval: Improved Dialogue Evaluation with Optimized Questions and Multi-turn Comparisons

https://arxiv.org/abs/1909.03087 (2019, v1 only). Margaret Li, Jason Weston, Stephen Roller (Facebook AI Research). Code: ParlAI `parlai/crowdsourcing/tasks/acute_eval`.

Read status: full paper text read via the arXiv HTML rendering, through a fetch tool that summarises. Tables were re-assembled by that tool, so individual cells are less reliable than the prose. ParlAI README, blueprint and `analysis.py` also read.

## What it is
A human evaluation protocol for chatbots. A judge sees two complete multi-turn dialogues side by side, attends to one highlighted speaker in each, and makes a forced binary choice on one fixed question. Question wording was chosen empirically for inter-annotator agreement. The paper shows it separates models that multi-turn Likert scoring cannot, and that model self-chat logs can replace human-model logs, making it cheaper.

## Mechanism
1. Collect about 100 dialogues per model (human-model chats, or self-chats: the model plays both roles).
2. Sample a pair (one dialogue of model A, one of B). Show them side by side; the model's turns are coloured (Speaker 1 light blue, Speaker 2 dark blue), the partner's turns are grey.
3. Ask one question with two radio buttons, plus a strongly encouraged one-line justification.
4. Each specific dialogue pair is shown at most once; maximise pair diversity.
5. Quality control: each worker's first item is a weak-baseline versus human-human pair; workers who do not prefer the human are removed, as are workers who never give a reason. Workers are capped at a few items.
6. Win rate = wins / trials per matchup; significance by a two-sided binomial test against 50%, p < .05.

Chosen wordings (Table 1; agreement on a known human-vs-weak-baseline pair, about 20 annotations per phrasing, 13 phrasings):
- Engagingness: "Who would you prefer to talk to for a long conversation?" / "I would prefer to talk to Speaker 1" (87.5%).
- Interestingness: "If you had to say one of these speakers is interesting and one is boring, who would you say is more interesting?" (86.7%).
- Humanness: "Which speaker sounds more human?" (76.9%).
- Knowledgeable: "If you had to say that one speaker is more knowledgeable and one is more ignorant, who is more knowledgeable?" (100%).
Worst phrasing: "Which speaker is more boring to talk to?" (69.6%).

ParlAI defaults (blueprint source): `eval_question` = the engagingness wording above; `annotations_per_pair=1`; `num_matchup_pairs=2`; `subtasks_per_unit=5`; `onboarding_threshold=0.75`; `block_on_onboarding_fail=True`; `randomize_conversations=True`. `analysis.py` uses `scipy.stats.binom_test([wins1, wins2])`, drops all rows from workers who failed onboarding, and writes a win grid and a p-value table. Input is JSONL with `speakers_to_eval`, `dialogue_dicts` and `is_onboarding`.

## Numbers
All on PersonaChat (ConvAI2 systems, 100 dialogues each) or Wizard of Wikipedia (100 each, unseen topics).
- PersonaChat engagingness, human-model chats (Table 2): ordering Human > PE > LIC > INT > HF > INQ > KV > RC. Example cells: human beats the weakest retrieval baseline (RC) 67%; human beats the best model (PE) only 53%; KV vs RC 50%.
- Self-chat (Table 4): same ordering except HF, which collapses (loses 74-82% to every other model). Human beats RC 74%, PE 61%.
- Significant matchups: engagingness 9 of 9 comparable pairs with self-chat, 3 with human-model chats, 5 of 9 with Likert. Interestingness 6 of 10; humanness 5 of 10; Likert had 2 of 3 known pairs in each. Wizard: 8 significant versus 6 for Likert.
- Wizard knowledgeability (Table 8): human beats the generative no-knowledge model 76%.
- A/A test (same model both sides): close to 50-50.
- Memorisation check: under 1% of PE's self-chat utterance pairs occur in training data; about 10% for INQ (a repeated greeting).
- Figure 5 (RC vs INT): ACUTE reaches significance with fewer person-hours; Likert never does. No numeric values are given in the text.

## Smallest scale tested, and whether the result held there
The systems are 2018-era models (key-value memory network, polyencoder, GPT-1-based fine-tunes); the paper gives no parameter counts. It is a measurement paper, so "scale" means annotation count: 100 dialogues per model, about 20 annotations per phrasing in the wording test. The protocol separated weak, clearly sub-human models from each other, which is our regime.

## Training recipe and gotchas
No training. Gotchas:
- The paper's trial count is literally a placeholder ("We collect NN trials"); the number of annotations per matchup is not recoverable from it.
- Self-chat exposes degenerate behaviour (HF) that human partners hide: a self-chat ranking can differ from the human-chat ranking.
- A model could score well in self-chat by reciting training dialogues; check n-gram overlap with training data.
- Negatively framed questions ("more boring") give lower agreement.
- No tie option: close models need many trials.

## Mapping onto the GBA design
Zero runtime cost: no lookups, adds, multiplies, RAM or ROM. Offline protocol only. Use:
- Generate about 100 dialogues per system (Koko, GPT-2 small, a checkpoint or two, optionally a strong reference) of fixed length, e.g. 6 turns each side. Prefer one shared scripted or LLM "user" partner over pure self-chat for the headline, because GPT-2 small has no chat format and Koko self-chat may loop; run self-chat as a separate degeneracy probe.
- Run the LLM judge on the exact ACUTE screen (two dialogues, one question, forced choice, both orders), and run a small human pass on the same pairs to validate it.
- Sample size (my own normal-approximation calculation, not from the paper; two-sided p<.05, 80% power): true win rate 70% needs about 47 judgments, 65% about 85, 60% about 195, 55% about 780. One or two volunteers can do 200; the LLM judge covers the rest.
- Screening item: Koko-vs-human-written dialogue, expected answer human.

## Verdict
Adopt, as the human-check template (evaluation only). Use the "long conversation" wording as primary plus "sounds more human". Settling experiment: 200 Koko-vs-GPT-2 pairs judged by humans and by the LLM judge; accept the judge if it agrees with the human majority on at least about 80% of pairs and both give the same significant winner.

## Could not verify
- Annotations per matchup, worker cap, pay and time: placeholders or absent in the paper.
- Figure 5 values (plot only).
- Table cells were reconstructed by the fetch tool; the row/column orientation is inferred from human-versus-baseline cells, not read from the header.
- The paper prints no verbatim annotator instructions and no Speaker 2 wording (ParlAI mirrors Speaker 1's).
- Whether the protocol stays reliable when both systems are frequently incoherent: not tested.

## Links worth following
- https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/acute_eval
- Recipes for Building an Open-Domain Chatbot (arXiv 2004.13637): ACUTE-Eval with self-chat at scale.
- What Makes a Good Conversation? (See et al., arXiv 1902.08654): source of the Likert baselines and several models.
