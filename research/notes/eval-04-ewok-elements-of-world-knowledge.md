# Elements of World Knowledge (EWoK): A Cognition-Inspired Framework for Evaluating Basic World Knowledge in Language Models
(https://arxiv.org/abs/2405.09605, v2 of 3 July 2025, first posted May 2024; Ivanova, Sathe, Lipkin and 17 others, MIT / Georgia Tech)

Tags: [P] paper, [P-code] code read, [P-poster] authors' pilot poster, [N] sibling notes, [M] my inference.

## What it is
A benchmark generator plus one frozen dataset, EWoK-core-1.0: 4,374 items testing 192 concepts in 11 domains [P]. Each item is a "minimal pair of pairs": two contexts and two targets, where T1 fits C1 and T2 fits C2. The same target is scored under both contexts, so target frequency, length and tokenization cancel. It is the world-knowledge task in BabyLM 2024 and 2025 [N].

## Mechanism
Example [P]: C1 "The piano is in front of Ali. Ali turns left." / C2 "... turns right."; T1 "The piano is right of Ali." / T2 "... left of Ali."

```
s(T|C) = sum_k log P(t_k | C, t_<k)          # target tokens only
acc    = 0.5*[s(T1|C1) > s(T1|C2)] + 0.5*[s(T2|C2) > s(T2|C1)]
```
A context-blind model scores exactly 0.5. Four forward passes per item.

- **Generation [P]:** 880 templates with typed placeholders such as `{object2:can_bounce=True}`; over 500 fillers. Five versions are five random filler draws, fixed within a version: `python -m ewok.compile --compile_dataset=true --fix_fillers=true --num_fillers=1 --version=$v`. 880 x 5 = 4,400; 4,374 remain after exclusions [M].
- **Contrasts [P]:** targets differ by concept swap or variable swap; contexts by antonym, negation or variable swap; templates are direct or indirect.
- **Columns [P-code]:** Domain, Context1, Context2, Target1, Target2, ContextType, ContextDiff, TargetDiff.
- **Other scorers [P]:** Likert (rate 1-5) and Choice (pick context "1" or "2"), both 2-shot with logit-masked output. Baseline: summed word2vec vectors, cosine between context and target.

## Numbers
LogProbs accuracy, mean over 5 versions (range), Table A4 [P]:

| Model | Accuracy |
|---|---|
| Humans | 0.951 (0.942-0.957) |
| falcon-40b-instruct (best) | 0.801 (0.790-0.810) |
| Mistral-7B / Llama-3-70B | 0.775 / 0.775 |
| phi-1.5 | 0.727 (0.686-0.756) |
| gpt2-xl | 0.655 (0.645-0.662) |
| word2vec bag of words | 0.542 (0.539-0.547) |
| phi-1 (code-trained) | 0.522 (0.517-0.530) |

- **By domain, LLM mean/best/human (Table A5) [P]:** social interactions 0.859/0.945/1.000; social properties 0.839/0.905/0.997; material dynamics 0.816/0.885/0.911; social relations 0.761/0.856/0.992; quantitative 0.725/0.823/0.986; physical dynamics 0.706/0.920/0.833; agent properties 0.683/0.778/0.975; physical interactions 0.672/0.759/0.910; material properties 0.669/0.755/0.921; physical relations 0.627/0.723/0.886; spatial relations 0.615/0.749/0.958.
- **Regression (appendix) [P]:** physical relations (0.38) and spatial relations (0.41) are the only domains not significantly above chance; word frequency +0.07***; number of words -0.04**.
- **Prompting [P]:** with strict inequality, Likert gives Llama-3-70B 0.588 and MPT-7B 0.021. Choice via HELM, January 2025 (Table 2): GPT-4 Turbo 0.912, Mistral-7B-Instruct 0.772, Llama-3-8B-Instruct 0.171.
- **Pilot, 175 social-relations items [P-poster]:** GPT2-XL scores 0.72 by LogProbs, 0.53 by Choice, 0.50 by Likert. Prompting is at chance for small models; score Koko by log-probabilities only.

## Smallest scale tested, and whether the result held there
The floor is 1.3B parameters [P]. gpt2-xl sits 15 points above chance and 11 above bag of words. BabyLM models trained on 10M-100M words score 49.5-58.4 [N], on a different comparison (below). Expect near chance at our scale.

## Training recipe and gotchas
- **Terms of use [P]:** no plain-text redistribution; training on EWoK data must be reported; the dataset is gated.
- **BabyLM computes a different metric [P-code, `read_files.py::decode_ewok`]:** it compares s(T1|C1) with s(T2|C1), two targets under one context, plus the swapped item. Target frequency no longer cancels. BabyLM EWoK numbers are not comparable with Table A4.
- **Filler noise [P]:** range across versions is 0.07 for phi-1.5 and phi-2, 0.02 for humans. The set separates domains well but models poorly.
- **Ties:** the metric needs strict inequality; integer logits can tie. Count ties as wrong and report the tie rate [M].
- **Names:** fillers are "culturally diverse" baby names (Fatima, Chao). BabyLM's vocabulary filter keeps 3,809 of 4,374 items [N, halved from 7,618 lines].

## Mapping onto the GBA design
- **Cartridge cost:** zero lookups, adds, multiplies, RAM and ROM. Run in the PC simulator.
- **Work:** 4 x 4,374 = 17,496 sequences. At a guessed 25 tokens each [M] that is about 440K token steps, 30 hours at 4 tokens/s on the console.
- **Two-level head:** s = sum of log P(class) + log P(member | class) over target tokens. Context tokens need state updates only.
- **Attention diagnostic [M]:** variable-swap and spatial items need names bound to roles, which top-8 attention with sign keys could break. Report accuracy per domain and contrast type, float versus ternary.
- **Lookups do not help [M]:** bag of words reaches 0.542, so hashed n-gram tables will not move EWoK. It measures the scarce contextual compute.
- **Data coverage [M]:** check generated conversations against the 192-concept list (Table 3). phi-1.5, trained largely on LLM-generated text, beats gpt2-xl by 7 points.

## Verdict
**Adopt as a secondary offline diagnostic, not a headline metric.** It is cheap, cancels tokenizer effects between our 16K vocabulary and GPT-2's BPE, and exposes attention damage. The range is narrow: chance 0.50, bag of words 0.54, GPT-2 XL 0.655, GPT-2 small unreported. Settling experiment: score GPT-2 small and XL with the paper's metric on all 5 versions (XL must reproduce 0.655), then float and ternary Koko. If GPT-2 small is under about 0.60, keep EWoK as a per-domain sanity check only.

## Could not verify
- GPT-2 small (124M) on EWoK: not in the paper; two searches found nothing.
- Per-model per-domain values: only in Figure 3; the result archive was not opened.
- Per-model parameter counts and item token lengths: not given.
- Dataset card: gated (HTTP 401); column names come from BabyLM's reader.
- BabyLM scores and item counts come from sibling notes, not re-read here.

## Links worth following
- https://github.com/ewok-core/ewok (generator)
- https://github.com/ewok-core/ewok-paper (evaluation code, results)
- https://huggingface.co/datasets/ewok-core/ewok-core-1.0
- https://github.com/babylm/evaluation-pipeline-2025
- https://southnlp.github.io/southnlp2024/papers/southnlp2024-poster-47.pdf
