# KSPC (Keystrokes per Character) as a Characteristic of Text Entry Techniques  (https://www.yorku.ca/mack/hcimobile02.html, 2002, I. Scott MacKenzie, York University; Mobile HCI 2002, LNCS 2411, pp. 195-210)

Read in primary form (full HTML, equation images, Fig. 2, tables, footnotes). [mine] = my arithmetic or inference.

## What it is
Defines KSPC: the average number of key presses per character of text for a given entry technique and language, computed a priori (no users) from a keystroke-level description plus a frequency table of letters, digrams or words. It is computed for 18 techniques, from cursor-key entry (about 10) to word prediction (about 0.5), with Qwerty = 1. It converts prediction quality into button presses, and two of its techniques (cursor keys plus SELECT) are close to a GBA D-pad.

## Mechanism
- Letter or digram level (Eq. 2): KSPC = sum(K_c * F_c) / sum(C_c * F_c); K_c = keystrokes for the character, C_c = 1, F_c = corpus frequency.
- Word level (Eq. 3): KSPC = sum(K_w * F_w) / sum(C_w * F_w). K_w and C_w both include one terminating SPACE.
- Letter table if cost depends only on the character; digram table (729 entries) if on the previous character; word table otherwise.
- Word prediction (Section 6): 1 keystroke per letter; after each letter show the n most frequent words starting with the typed stem (unigram, no context). The user acts when the intended word first appears.
  - "Stylus" (direct selection): K_w = stem length + 1; the selection also adds the SPACE.
  - "Keypad" (NEXT-key stepping): K_w = stem length + min(characters remaining, index of word in list) + 1.
- Example: 'vegetable', n = 5, appears 3rd at stem 'veg': stylus 4 keystrokes, keypad 6, unaided 9.
- Pager grid (Fig. 2), cursor snaps home to SPACE after every entry: rows `abcdefghij` / `klmn_opqr` / `stuvwxyz` (last row starts under 'd'). So a = ULLLLS, SPACE = S.
- Hick-Hyman list-scanning time (Eq. 21): RT = k * log2(n + 1), k = 200 ms/bit.

## Numbers
All on the British National Corpus: about 90M words, 64,566 unique words, mean word length 4.59 characters, SPACE about 18% of input; punctuation excluded.
- Table 1: date stamp (LEFT/RIGHT/SELECT over a line of letters) 10.6598 alphabetical, 6.4458 with central SPACE and snap-home (two variants between). Five-button pager grid 3.1320. Multitap 2.0342. MessagEase 1.8210. LetterWise 1.1500. T9 1.0072 (assumes dictionary words only). Qwerty 1.0000.
- Word prediction, keypad stepping (Eq. 13-16): n = 1: 0.7391; n = 2: 0.7086; n = 5: 0.7483; n = 10: 0.8132. Net loss beyond n = 2.
- Word prediction, direct selection (Eq. 17-20): n = 1: 0.7391; n = 2: 0.6466; n = 5: 0.5506; n = 10: 0.5000.
- Table 2, repeat presses as a share of all presses: date stamp 0.75-0.85; pager 0.3501; word prediction 0.006-0.077.
- Same-key repeat time 128-176 ms (footnote 2); visual reaction 190-260 ms; scanning a 10-word list about 692 ms, additive per keystroke.
- Footnote 1: on the Brown corpus the largest difference was 1.07%, rank order unchanged.

## Smallest scale tested, and whether the result held there
Analytic study: no model sizes, no users. The only scale check is the much smaller Brown corpus, which held within 1.07%. Predictions are context-free unigram, so the figures are a floor for a contextual model.

## Training recipe and gotchas
- Nothing is trained; the gotchas are in the accounting. Numerator and denominator must both include the SPACE, and selection must append it.
- KSPC ignores time: scanning suggestions costs RT per look, and repeated presses are faster than distinct keys. Report presses and time separately.
- Assumes a perfect user who never errs; no punctuation, case, or out-of-dictionary words.
- Selection cost decides the optimum list size (stepping makes n > 2 a loss).

## Mapping onto the GBA design
- The pager grid (3.13 presses/char; D-pad moves, A selects, snap-home) is the no-prediction baseline.
- Bind 3-4 suggestion slots to dedicated buttons (L, R, B, Select) so selection is one press and the direct-selection row applies. Never make the user step through a list.
- [mine] Press estimate. Letters-only cost on the pager grid = (3.132 - 0.179) / 0.821 = 3.60 presses. Letters typed per word under direct selection = KSPC * 5.59 - 1: 3.13, 2.61, 2.08 for n = 1, 2, 5. Presses per word = letters * 3.60 + 1: 12.3, 10.4, 8.5 against 17.5 unaided, i.e. about 2.2, 1.9, 1.5 presses/char versus 3.13 (approximate: word-initial letters have a different cost mix).
- [mine] A letter costs 3.6 presses here, not 1, so each predicted character is worth 3.6 times more than in the paper's settings. Koko also adds context: a correct zero-letter next-word suggestion costs 1 press per word.
- Cost: the next-token distribution comes from the forward pass that ingests the user's previous token, so no extra dense compute. Per letter, re-filter a cached top-64 (token id, score) list in RAM (256 bytes) by prefix against a ROM table of token strings: a few hundred compares, no multiplies. Fallback when nothing matches: a ROM prefix trie with unigram-ranked completions (the paper's method; under 1 MB [mine]).
- The two-level head must expose top-k across several classes, not only the argmax class. The metric itself is computed offline.

## Verdict
Adopt as the predictive-typing metric. Settling experiment: an offline simulator that replays an ideal user on the exact GBA button map over held-out user turns and reports presses/char for (a) pager grid alone, (b) unigram completion with 3 slots, (c) hashed n-gram, (d) Koko, (e) GPT-2 small as ranker, plus Hick-Hyman time per look. Koko must clearly beat (b).

## Could not verify
- KSPC.zip (Java software) was not downloaded; its page only lists file names.
- The paper has no n = 3 or n = 4 figures and no user study; the KSPC-to-words-per-minute relation is asserted, not measured.
- Nothing is from secondary sources.

## Links worth following
- Software: http://www.yorku.ca/mack/KSPC/KSPC.html
- Bellman & MacKenzie 1998, probabilistic (dynamically rearranged) character layout for cursor-key entry: https://www.yorku.ca/mack/gi98.pdf
- Silfverberg, MacKenzie & Korhonen, CHI 2000, predicting text entry speed (time model): https://doi.org/10.1145/332040.332044
