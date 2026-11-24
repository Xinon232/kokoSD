# Entity Tracking in Language Models

https://arxiv.org/abs/2305.02363 (v2, Sep 2023; ACL 2023 pp. 3835-3855; Najoung Kim, Sebastian Schuster). Code and data: https://github.com/sebschu/entity-tracking-lms

Tags: **[P]** read in the paper PDF (full text, appendices included); **[R]** repo or BabyLM pipeline README; **[N]** our earlier notes (eval-01, eval-02); **[M]** my inference.

## What it is
A synthetic "boxes" benchmark: a text gives the initial contents of 7 boxes, then up to 12 Move/Remove/Put operations, and the model must state what one box now contains [P]. It is built so that the answer cannot be read off a single phrase, and scores are reported by how many operations touched the queried box. Result: text-only LMs (GPT-3 davinci, Flan-T5) just repeat the initial state; code-pretrained GPT-3.5 tracks; a fine-tuned T5-base learns the task but a randomly initialised T5 does not [P]. BabyLM 2025 reuses the data as a 5-way multiple-choice log-probability task [R, N].

## Mechanism
**Generator [P, Sec. 3.2].** World W = (O, n, m, e): |O| = 100 nouns (BNC frequency > 27), n = 7 boxes, m = 3 max objects per box, e = 2 expected objects per box at start. A scenario = random initial state + 12 random *valid* operations. 2,200 scenarios; each gives n x (12+1) = 91 cloze examples (every box after every prefix).

Templates (Table 2, Appendix B):
```
Box 0 contains the painting, Box 1 contains the bell, ... Box 6 is empty.
Move the car from Box 1 to Box 3.  | Remove the car from Box 1.  | Put the car into Box 1.
Box 6 __   ->   contains the guitar and the knife.
```
- **AltForms:** "Pick up the furby in Container A and place it into Container C" / "Take the furby out of..." / "Place the furby inside..."; shares only "the" and "into" with base.
- **MoveContents:** adds "Move the contents of Box N to Box M" (objects not named).
- **AmbiRef:** objects carry adjectives ("the blue guitar"); the adjective is dropped in an operation when the box holds only one object of that type; the answer must include it.

**Four desiderata [P, Sec. 3.1]:** (1) probed states must not follow pretraining-like distributions; (2) no single word may fix the state (no "empty the box" operation); (3) train/demo and test text have little lexical overlap; (4) no slot-filling: no train and test example share an initial-state "signature" (objects per box, e.g. 2111111).

**Scoring [P]:** exact set match, any order, with or without "the". Report accuracy versus number of operations affecting the queried box (0-7), separately for state != initial and state = initial. Baseline: 0-3 random objects drawn from those mentioned in the same clauses as the box.

**BabyLM variant [R, N]:** score `prefix + option` for 5 options, highest summed log-prob wins (chance 20%); averaged per `<split>_<numops>_ops` subset, then over regular / ambiref / move_contents; 18 subsets, 9,483 items.

## Numbers
The paper has no result tables; everything below is text or figure description [P].
- Reanalysis of Li et al. (2021) Alchemy probe: 62.7% of probed states equal the initial state, 32.4% are empty, 87.6% trivial in total; probe accuracy 86.8% on trivial cases, **3.1% on non-trivial** (Sec. 2).
- Fig. 2: only text-davinci-003 beats the random baseline when state != initial; still >25% correct after 7 operations. Flan-T5 base (250M) and XL (3B): near zero on changed boxes. GPT-3 davinci (175B) also repeats the initial state.
- Fig. 5: all code-trained GPT-3.5 models beat all GPT-3 models, including code-davinci-002 with no human-feedback tuning.
- Fig. 6: fine-tuned T5-base is "near-perfect" on Base; trained on <= 2 operations and tested on up to 12 (NumOps) it stays above 45%; unseen object names cost little; AltForms drops a lot but stays above baseline; AltForms+NumOps falls to baseline. Random-init T5 "almost exclusively" predicts an empty box.
- Fig. 7: AmbiRef and MoveContents are no longer near-perfect even when trained in-format.
- Tables 6-7: 990 train / 220 dev / 990 test scenarios; 90,090 train examples (20,790 for NumOps), 20,020 dev, 90,090 test.
- BabyLM 10M-word baselines: GPT-BERT causal-focus 33.96; GPT-2 13.9 per model card [N] but 18.92 in a commented-out README table [R]; GPT-2 at 100M words 30.11 [R].

## Smallest scale tested, and whether the result held there
T5-base (~220-250M parameters) is the smallest model. Pretrained: learns the task. From random weights with 90K examples and one epoch: fails. Nothing below 200M parameters in the paper. In BabyLM, 31M-124M models trained on 10M words sit between below-chance and ~34-40% [N, R].

## Training recipe and gotchas
- T5 fine-tune: 1 epoch, batch 8, LR 1e-4; more epochs gave no gain and sometimes hurt; ~3 h on one RTX8000 (Appendix C).
- Aggregate accuracy is misleading: each operation changes at most two of seven boxes, so "repeat the initial state" scores high. Always split by operations-on-box.
- A model that copies the initial clause scores *below* chance in the multiple-choice form [N].
- Data is GPL-3.0 and shipped as a password-protected zip to keep it out of training corpora [P, R]. Do not train on the released test items; generate our own with different nouns.
- Authors' hypothesis (unproven): code helps because variable state must be tracked.

## Mapping onto the GBA design
Run-time cost: zero lookups, adds, multiplies, RAM or ROM. It is an offline test in the bit-exact PC simulator, scored with log P(class) + log P(member | class).

Design consequences [M]:
- **Window.** Initial state is about 45-60 words; each operation 6-10 words; 12 operations plus query is roughly 150-190 words. It fits a 256-token ring buffer only with a near word-level tokenizer; measure before trusting scores.
- **Top-8 attention.** To answer "Box 6" after k operations the model must gather the initial clause plus k operation clauses. At k = 7 that is 8 separate spans, exactly our top-8 limit per head, with nothing left for object tokens. Expect a cliff at high k unless layers accumulate state step by step. This is the one benchmark that directly stresses the top-8 choice.
- **Training data.** Since random-init T5 failed on 90K examples and text-only pretraining did not produce the skill at 175B, our model will not get it for free. Cheap fix: generate millions of our own state-tracking snippets (objects, places, who-has-what, in story and dialogue form, plus simple pseudo-code-like "x is now y" updates) and mix them into pretraining. Costs GPU time only.
- **Chat relevance.** Remembering the user's name, pet or stated preference across turns is the same skill at 0-2 operations.

## Verdict
**Adopt as a diagnostic, test as a training signal.** Use the generator recipe (not the released test set) for training data, and the BabyLM 5-way version for scoring. Settling experiment: train the float model with and without 5% synthetic state-tracking text; score BabyLM entity tracking per operations-on-box; then repeat with top-8 versus top-16 versus full attention to see whether top-8 causes the drop at k >= 4.

## Could not verify
- Exact per-point accuracies: only figures exist; numbers above are the paper's own prose.
- How BabyLM builds the 4 distractors and the prompt token lengths (data on OSF, not downloaded).
- The 13.9 versus 18.92 GPT-2 discrepancy.
- Whether code-like or synthetic tracking data helps models under 10M parameters: no source found here.
- Token-length and top-8 arguments are my own estimates.

## Links worth following
- https://github.com/sebschu/entity-tracking-lms (`scripts/data_generation/`)
- https://github.com/babylm/evaluation-pipeline-2025
- Li, Nye, Andreas 2021, "Implicit Representations of Meaning in Neural Language Models"
- Kim, Schuster, Toshniwal 2024, "Code Pretraining Improves Entity Tracking Abilities of Language Models" (not read)
