# Dyna-bAbI: unlocking bAbI's potential with dynamic synthetic benchmarking

https://arxiv.org/abs/2112.00086 (v1, 30 Nov 2021; *SEM 2022 pp. 101-122 per search result). Tamari, Richardson, Sar-Shalom, Kahlon, Liu, Tsarfaty, Shahaf.

Tags: **[P]** read in the arXiv PDF text (full, with appendices); **[S]** search-result snippet; **[M]** my inference.

## What it is
A Python re-implementation of the Lua bAbI simulator with a config file that sets which concepts (events, linguistic constructs, question types), story length and rejection filters are used [P]. With it the authors build compositional splits: every concept is seen in training, but not every combination. Models that score >96% on original bAbI collapse to 22-67% on the new test sets. Sampling structurally diverse training stories helps far more than tripling the data, but complex compositions still fail [P].

## Mechanism
**World [P, Table 5].** Persons P, locations L, objects O. A story is a sampled event sequence rendered by templates; a QA pair is sampled over it, with supporting-fact line numbers.
```
MOVE   P {moved} to the L.            GRAB  P {grabbed} the O.
DROP   P {dropped} the O.             GIVE  P1 {gave} P2 the O.
COREF  Following that, he <event>.    CONJ  P1 and P2 {moved} to the L.
COMPOUND  ...Then they {moved} to the L2.
NEGATE P is not at the L.             INDEF P is either at the L1 or the L2.
where-P  Where is P?        where-O  Where is the O?
where-was-O  Where was the O before the L?
yes-no  Is P at the L? (yes/no/maybe)   list  What is P carrying?
counting  How many objects is P carrying?   give-qs  Who gave the O to P2? ...
```
**Splits [P, Table 2]** over task sets T2={2,11}, T7={1,2,3,5,11,12,13}, T12=T7+{6..10}:
- `concat`: original bAbI data, 9,000 train per task.
- `inject`: same stories, plus every possible where-P / where-O question.
- `diverse`: rejection-sample so the number of supporting facts is roughly uniform per question type; stories 20 sentences. Held out: COREF/CONJ/NEGATE/INDEF occur only with MOVE.
- `mix` (test only): any combination; 20 sentences; mean supporting facts 2.50 (T7), 3.70 (T12).

**Difficulty dials:** support composition f_c (which concepts are in the supporting facts) and |f| (how many). Report accuracy by |f| = 1 / 2 / 3+.

## Numbers
Table 3, train on concat, accuracy % [P]:

| Test | EntNet | STM | BiDAF | RoBERTa | T5-base |
|---|---|---|---|---|---|
| concat(T7) | 96.8 | 99.4 | 99.98 | 99.98 | 99.8 |
| mix(T2) | 72.0 | 67.6 | 97.2 | 98.7 | 98.1 |
| mix(T7) | 22.2 | 26.7 | 30.5 | 57.7 | 62.66 |
| mix(T12) | 31.97 | 35.65 | - | - | 67.4 |

Table 4, T5-base, test on mix, by supporting facts 1 / 2 / 3+ / total [P]:
- inject(T7), 190,158 examples: 89.82 / 80.55 / 64.16 / 71.57
- diverse(T7), 17,000 examples: 100 / 98.44 / 93.84 / **95.8**
- inject(T12), 368,831: 92.45 / 85.29 / 67.67 / 72.2
- diverse(T12), 24,772: 99.01 / 96.29 / 81.24 / **84.82**
- diverse-trained models drop on long concat stories (3+ column 78.36 / 76.81): length mismatch (footnote 9).

Other [P]:
- Table 9, GIVE in supporting facts, concat(T7)-trained, mix(T7): 2 facts incl. give 3.03 / 7.07 / 12.12 (BiDAF / RoBERTa / T5).
- Table 10: of 446 "yes" questions re-asked as where-P, T5 got 354 right as where-P but only 213 as yes-no. Double disjunctions (Fig. 7): accuracy near zero; T5 answers "maybe".
- A.3: ~500 extra OOD samples per question type lift T5 to >90% on mix(T7) and mix(T12) (Fig. 5).
- Concurrence with SQuAD over 20 models: mix(T7) r=0.92, tau=0.78; bAbI task 2 r=0.08, tau=-0.19 (Fig. 2).

## Smallest scale tested, and whether the result held there
Smallest: EntNet (20 memory slots x 100 dims) and STM, trained from scratch. They solve IID (96-99%) and fail OOD (22-36%). No parameter counts given. **Diverse training was only tried on pretrained T5-base**; nothing shows a small from-scratch model benefits.

## Training recipe and gotchas
- T5-base: 12 epochs, Adam, LR 5e-5, batch 8; inputs over 512 tokens dropped (1,585 train instances). EntNet/STM: 200 epochs, batch 32, best of 10 runs [P].
- Diversity beats volume: 17K diverse > 190K injected.
- IID accuracy on templated data is uninformative; always hold out combinations.
- Models couple answers to question form; test the same fact in two phrasings.
- Uniform story length in training hurts on longer test stories.

## Mapping onto the GBA design
Run-time cost: none. Offline evaluation in the bit-exact PC simulator, plus a recipe for synthetic training text [M].
- **Window:** 20 sentences at ~6-9 tokens plus a question is ~130-190 tokens; fits the 256-token ring. Original task 3 (51.9 sentences on average) does not.
- **Top-8 attention:** mix(T12) needs 3.7 facts on average, each fact a few tokens; multi-hop chains (object -> holder -> holder's location) need several layers. Score by |f| to find where 4-6 layers and top-8 break.
- **Scoring:** answers are one word from a small closed set; score as constrained argmax over candidate tokens through the class/member head, or rephrase as a chat turn ("Where is John?" / "John is in the park.").
- **Training data:** our own generator of who-is-where / who-has-what stories, with uniform-|f| rejection sampling, varied length and paraphrased questions, mixed in at a few percent. Complements boxes (f-eval-03): bAbI adds named people, co-reference, negation, transfer.
- **Chat relevance:** remembering "my dog is called Rex" is |f|=1; corrections ("no, I moved to Leeds") are |f|=2.

## Verdict
**Test (as a diagnostic and data recipe).** Write a ~200-line generator of our own rather than depend on the release. Settling experiment: train the float model with 0% vs 3% diverse-style synthetic stories; evaluate on a held-out-combination set by |f| and by question phrasing; repeat with top-8 vs full attention.

## Could not verify
- Code/data: the paper only says "will be made available" at dyna-babi-project.github.io, which redirects to a Notion page I did not open; no GitHub repo, licence or example config confirmed.
- Entity/location/object vocabulary sizes: not stated.
- Parameter counts: not stated.
- Fig. 3 and Fig. 5 per-cell values (figures only); the "<70% on complex compositions" claim is from the abstract summary.
- Venue (*SEM 2022) is from a search snippet [S].
- Token-length estimates and top-8 argument are mine [M].

## Links worth following
- https://dyna-babi-project.github.io (project page)
- https://github.com/facebookarchive/bAbI-tasks (original Lua generator)
- Kaushik and Lipton 2018, "How Much Reading Does Reading Comprehension Require?" (bAbI shortcuts; not read)
- Liu et al. 2021, concurrence of synthetic and natural benchmarks (not read)
- Oren et al. 2021, structurally diverse sampling (not read)
