# BLiMP: The Benchmark of Linguistic Minimal Pairs for English (https://arxiv.org/abs/1912.00582, TACL 2020, arXiv v4 2023; Warstadt, Parrish, Liu, Mohananey, Peng, Wang, Bowman, NYU)

Tags: **[P]** arXiv v4 PDF, read in full; **[R]** files in github.com/alexwarstadt/blimp; **[D]** my counts over the released data and result files; **[S]** fetch-tool summary of another paper.

## What it is
A test set of 67,000 English sentence pairs: 67 "paradigms" of 1,000 pairs, grouped into 12 grammatical phenomena [P]. Each pair has an acceptable and an unacceptable sentence that differ minimally. A model is correct when it gives the acceptable sentence the higher probability; chance is 50%. Sentences come from linguist-written templates over 3,000+ annotated words [P]. Licence CC-BY 4.0 [R].

## Mechanism
- **Full-sentence (all 67 paradigms):** correct if `sum_t log p(w_t | w_<t)` is higher for the good sentence. No length normalisation.
- **One-prefix (20 paradigms):** shared prefix; correct if `p(word_good | prefix) > p(word_bad | prefix)`. Fields `one_prefix_prefix`, `one_prefix_word_good`, `one_prefix_word_bad` [D].
- **Two-prefix (20 others):** correct if `p(word | prefix_good) > p(word | prefix_bad)`. Fields `two_prefix_prefix_good`, `two_prefix_prefix_bad`, `two_prefix_word` [D].
- **Format:** one `.jsonl` per paradigm; `sentence_good`, `sentence_bad`, `UID` [R].
- **Overall score:** fraction of all 67,000 pairs, i.e. the unweighted mean over paradigms [P]; `models_summary.jsonl` reproduces 83.02 for GPT-2 [D].
- **Human check [P]:** 20 raters x 5 pairs per paradigm; majority-vote agreement 96.4%, individual 88.6%.

## Numbers
Table 3 [P], percent correct. GPT-2 is GPT-2-large (774M; about 8B training tokens, the authors' estimate). TXL trained on 103M tokens, LSTM on 83M, 5-gram on 3.1B.

| | All | Anaph. | Arg. | Bind. | Ctrl. | Det-N | Ellip. | Filler | Irreg. | Island | NPI | Quant. | S-V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5-gram | 61.2 | 47.9 | 71.9 | 64.4 | 68.5 | 70.0 | 36.9 | 60.2 | 79.5 | 57.2 | 45.5 | 53.5 | 60.3 |
| LSTM | 69.8 | 91.7 | 73.2 | 73.5 | 67.0 | 85.4 | 67.6 | 73.9 | 89.1 | 46.6 | 51.7 | 64.5 | 80.1 |
| TXL | 69.6 | 94.1 | 72.2 | 74.7 | 71.5 | 83.0 | 77.2 | 66.6 | 78.2 | 48.4 | 55.2 | 69.3 | 76.0 |
| GPT-2 | 83.0 | 99.3 | 81.8 | 80.9 | 81.9 | 95.8 | 89.3 | 81.3 | 91.9 | 72.7 | 76.8 | 79.0 | 86.4 |
| Human | 88.6 | 97.5 | 90.0 | 87.3 | 83.9 | 92.2 | 85.0 | 86.9 | 97.0 | 84.9 | 88.1 | 86.6 | 90.9 |

- **Model size (§6.3) [P]:** four GPT-2 sizes, 117M to 1558M, all score 84 ± 1; per-phenomenon standard deviation at most 3 points.
- **Data size (§6.3, Figure 5) [P]:** LSTM and TXL retrained on 1/8M to 64M tokens. LSTM gain per doubling: anaphor 6.2 points, det-noun 4.3, irregular 3.9, NPI 0.78, islands 0.36.
- **Local statistics:** the 5-gram model is below 50% on 20 of 67 paradigms [D]; one adjective between determiner and noun costs it 35 points, neural models 3-5 (§6.1) [P].
- **Attractors (§6.1) [P]:** an intervening noun of the other number costs GPT-2 22 points and the LSTM 20 on subject-verb agreement.
- **Scoring methods (§6.4, Figure 6) [P]:** all three agree in aggregate; TXL is much worse with full-sentence scoring on binding and agreement.
- **GPT-2's worst paradigms [D]:** sentential_subject_island 35.5, existential_there_quantifiers_2 42.4.
- **Dataset shape [D]:** mean 7.4 words per sentence, maximum 26; 991,414 words in total; 65% of pairs differ by one substituted word.

## Smallest scale tested, and whether the result held there
Smallest model: GPT-2 117M (inside the 84 ± 1 band). Smallest data: 125K tokens, shown only as Figure 5 curves. Nothing near 1-2M active weights. Agreement is learned early at every scale shown; islands and NPIs barely move.

## Training recipe and gotchas
- **Errata [R]:** the TACL Tables 3-4 are wrong; use arXiv v4 or the repository. Even v4 Table 4 disagrees with `models_summary.jsonl` on three rows (GPT-2 `transitive` 76 against 86; both `determiner_noun_agreement_irregular` rows) [D].
- **Averaging [D]:** the mean over the 12 phenomena is 84.8 for GPT-2, against 83.0 over pairs. Fix one convention.
- **Labels [P]:** mainstream US English; stereotyped name-gender pairings.
- **Lexicon:** names and rare nouns (`unicycle`, `casseroles`) are common. BabyLM's filtered set keeps 59,875 pairs (eval-02 note), so its scores are not comparable with these.

## Mapping onto the GBA design
- **Cartridge cost:** none; run on the bit-exact PC simulator.
- **Two-level head:** each scored token needs `log p(class) + log p(member | class)`, as in generation.
- **Console self-test:** one-prefix needs one forward pass and one comparison of two logits. If both words are single tokens in one class, that is an integer compare with no softmax, so a few hundred pairs can ship as a regression test.
- **Predictive typing:** one-prefix accuracy measures whether suggestions rank the grammatical word higher (20,000 prefixes).
- **Hashed n-gram tables:** local statistics; expect about the 5-gram level (61.2) from them alone. The attractor paradigms test whether top-8 attention finds the subject.
- **Context:** the longest sentence is 26 words; the 256-token ring, dialogue and facts go untested.
- **Integer logits:** count ties as 0.5. Report prefix scores too: full-sentence sums include tokens after the critical word (my inference from the TXL result).
- **Noise [mine]:** standard error at most 1.6 points per paradigm, about 0.15 overall.

## Verdict
**Adopt** as the grammar leg of the benchmark, with the one-prefix subset as the typing metric. GPT-2 small should score near 83-84, which our model is unlikely to match; realistic references are 61 (5-gram) and 70 (LSTM/TXL at about 100M tokens). Settling experiment: score `gpt2` and our float, ternary and attention-disabled models on all 67,000 pairs with one script.

## Could not verify
- An exact score for OpenAI GPT-2 124M: the paper gives only "84 ± 1" for four sizes.
- Numeric values behind Figures 2, 5 and 6.
- [S] arXiv 2604.17930: GPT-2 124M trained from scratch scores 74.77 (100M FineWeb tokens) and 80.13 (10B); 1% targeted synthetic text lifts single paradigms (only_npi_scope 20.9 to 69.4) but the overall score by at most 2 points.

## Links worth following
- Data and raw results: https://github.com/alexwarstadt/blimp
- Generation templates: https://github.com/alexwarstadt/data_generation
- Targeted-data study: https://arxiv.org/abs/2604.17930
