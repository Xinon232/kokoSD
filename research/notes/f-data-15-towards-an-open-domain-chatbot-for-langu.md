# Towards an open-domain chatbot for language practice (https://aclanthology.org/2022.bea-1.28/, BEA 2022, Tyen, Brenchley, Caines, Buttery; Cambridge ALTA / CUP&A)

Read in primary form: full PDF (pp. 234-249 incl. appendix) plus the released code (`demo.py`, `controllable_blender/generation_methods.py`, parts of `generation_utils.py`) at https://github.com/WHGTyen/ControllableComplexityChatbot.

## What it is
Five decoding-time strategies that steer an unmodified chatbot (Blender 2.7B) to a target CEFR difficulty level (A1..C2), with no retraining of the generator. Two are hard vocabulary restrictions (word list masks), three are re-ranking of 20 sampled candidates by a sentence-difficulty regressor, optionally with soft sub-token penalties and an out-of-vocabulary filter. 1,600 self-chats were rated by 10 trained Cambridge examiners. Main finding: hard vocabulary restriction alone barely moves judged difficulty; candidate re-ranking does; quality (grammatical/sensible/specific) drops only a few points.

## Mechanism
Base: Blender 2.7B via ParlAI, byte-level BPE, top-k sampling k=40, beam size 20 (= 20 candidate replies), temperature 1.0, length penalty 0.65, 3-gram blocking (Table 5). CEFR is mapped to 0..5.

- **M1 vocab restriction (EVP).** English Vocabulary Profile, 6,750 words/phrases, each collapsed to its *lowest* CEFR level; plus US first/last names and state names treated as A1. If word level > target, probability := 0. Because BPE words are only complete when the next token starts with a word boundary, the mask is applied to boundary-starting next tokens given the partial hypothesis (code: `wordlist.get_allowed_ids(partial_hyp)`, logprobs of the rest set to -inf, per beam, per step).
- **M2 extended EVP.** Random-forest regressor (5000 trees) on context-free word features (length, syllables, WordNet counts, SubIMDB/SimpleWiki frequencies, Ogden Basic English membership, MRC norms) labels 10,000 more frequent words. Done offline.
- **M3 re-rank.** distilroberta-base + regression head, trained on sentences split from the Cambridge Exams dataset (331 texts, ~531 words each). Final rank `R = (r(P(C_i)) + w * r(|L_t - L_Ci|)) / 2`, w = 1 (chosen from 10 hand-ranked sets of 20). Code simply adds the two `scipy.rankdata` vectors and takes the best.
- **M4 = M3 + sub-token penalty.** Each sub-token gets the lowest CEFR level of any Cambridge Exams text it occurs in. `p' = p * phi(L_s - L_t)` if `L_s > L_t`, else unchanged; phi = Normal(0, sigma=2) pdf. In code the non-penalised weight is pdf(0), so the effective ratio is `exp(-d^2/8)`: d=1 -> 0.88, 2 -> 0.61, 3 -> 0.32, 4 -> 0.14, 5 -> 0.04 (my arithmetic). Tokens never seen in the dataset get the maximum penalty (d = 5 - target). Penalties multiply the top-40 probabilities before the multinomial draw; the hypothesis score still uses the unpenalised logprob.
- **M5 = M4 + filter.** Discard any candidate containing a lowercase word not in a 370K-word English list (dwyl `words_alpha`); also discard candidates containing "u/" or "r/".

## Numbers
All at Blender 2.7B, 50 self-chats x 6 levels per method (300), 100 unmodified; 18 messages each; >= 3 examiners per dialogue. Table 4 (target vs examiner-judged CEFR):

| Method | Spearman | Pearson | MAE (levels) | %gramm | %sensible | %specific |
|---|---|---|---|---|---|---|
| Original | - | - | - | 90.2 | 81.9 | 94.1 |
| M1 EVP mask | 0.229 | 0.243 | 1.410 | 89.0 | 76.5 | 91.4 |
| M2 ext. EVP | 0.196 | 0.194 | 1.461 | 89.5 | 77.3 | 91.7 |
| M3 re-rank | 0.719 | 0.707 | 1.120 | 87.4 | 77.0 | 93.0 |
| M4 +penalty | 0.680 | 0.681 | 1.174 | 87.2 | 76.1 | 91.9 |
| M5 +filter | 0.755 | 0.731 | 1.090 | 87.3 | 76.4 | 92.1 |

- Word regressor on held-out EVP: Spearman 0.694, MAE 0.826 (Table 1). Sentence regressor: Spearman 0.701, Pearson 0.734, MAE 0.634 (Table 2).
- Agreement: CEFR weighted Fleiss kappa 0.79; Gwet AC1 0.62 grammatical, 0.23 sensible, 0.67 specific.
- Unmodified Blender was judged mostly B1-B2; no dialogue from any method was judged A1; restriction methods produced no C1/C2 (Fig. 1, Section 5.1).
- 0.36 multi-word expressions per message in a 10-dialogue sample; 6/1600 dialogues had inappropriate content.
- Of 100 "ungrammatical" messages: 51 colloquialisms, 29 awkward/not sensible, 20 real errors.

## Smallest scale tested, and whether the result held there
Only one scale: a 2.7B-parameter generator with an ~82M distilroberta scorer. Nothing at or near our size. No ablation of candidate count (always 20), k, sigma or w.

## Training recipe and gotchas
- Generator untouched. Sentence regressor: lr 9.737e-6, batch 32, 3 epochs, Optuna-tuned (Table 7).
- Hard masking with whole-word lists yields approximations of banned words out of allowed pieces ("I live in wash in n"), hence the added proper-noun list.
- Soft sub-token penalties create non-words ("backpicking"), hence the M5 filter. M4 was *worse* than M3 without it.
- Lowest-sense collapsing and multi-word expressions make word lists a weak proxy for difficulty.
- Sensibleness fell ~5 points for every method, including the mildest.
- The claim that quality is "statistically equivalent" to the original is the authors' test; raw drops are 1-3 points grammatical, 4.6-5.8 points sensible.

## Mapping onto the GBA design
- **Hard mask**: unnecessary if our 16K vocabulary is built from the word list in the first place; this paper's failure modes (fragment-assembled non-words) come from laying a word list over a foreign BPE vocabulary. If we keep sub-word pieces, enforce validity with a ROM trie/DFA over allowed words: 2 bytes of RAM state, one ROM lookup per candidate token, and only for the members of the chosen class in the two-level head. Zero multiplies.
- **Soft penalty**: one int8 per token in ROM (16 KB per difficulty level, or 16 KB of levels plus a 6-entry table), added to the logit in the log domain: one add per scored candidate. Essentially free.
- **Re-ranking 20 candidates**: costs 20x generation (20 replies at 3-5 tok/s is not viable) plus a scorer. A cheap variant: 2-4 candidates, scored by summed per-token level bytes plus length (table lookups only) instead of a neural regressor. RAM: candidates x reply length x 2 bytes, e.g. 4 x 48 x 2 = 384 bytes.
- **Better use**: apply the paper's pipeline offline, at data-generation time. Sample N replies from the teacher LLM, keep the one that passes the word-list filter and ranks best on difficulty; the GBA model then learns the restricted register and needs no run-time control.
- Predictive typing: the same per-token level byte can bias suggestions toward in-list words.

## Verdict
**Test (run-time), adopt (as offline data filter).** The paper's key lesson for us is negative: decoding-time vocabulary masks alone do not make text simpler in examiners' eyes (rho ~0.2) and cost ~5 points of sensibleness even on a 2.7B model; a small model will have less slack. Settling experiment: on our trained model, compare (a) no control, (b) trie mask, (c) int8 logit penalty, (d) best-of-4 with table scorer, measuring out-of-list word rate, non-word rate and held-out perplexity / LLM-judged sensibleness.

## Could not verify
- Whether the annotated self-chat data is actually in the repository (README view did not show it; paper says it is released).
- The EVP list and sub-token level file contents; EVP is licensed, and the repo ships a 5,000-word Google-frequency sample list instead.
- Table 5 prints "decoder layers 2" for Blender; copied as printed, likely a ParlAI option artefact.
- No latency or compute figures are reported.
- No follow-up at small model scale was searched for or found in this pass.

## Links worth following
- Code: https://github.com/WHGTyen/ControllableComplexityChatbot
- EVP: https://www.englishprofile.org/wordlists/evp
- Cambridge Exams readability dataset: Xia, Kochmar, Briscoe 2016 (BEA)
- Gooding and Kochmar 2018, CAMB at CWI shared task (word-difficulty features)
