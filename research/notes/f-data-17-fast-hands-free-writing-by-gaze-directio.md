# Fast Hands-free Writing by Gaze Direction (Dasher)

https://arxiv.org/abs/cs/0204030 (v4, 21 Aug 2002; Nature 418, p. 838). David J. Ward, David J.C. MacKay, Cavendish Laboratory.
Companion read for the method: Ward, Blackwell, MacKay, "Dasher - a Data Entry Interface Using Continuous Gestures and Language Models", UIST 2000, https://www.inference.org.uk/djw30/papers/uist2000.pdf. Both read in primary form (full PDF text).

## What it is
A 3-page note on text entry by "inverse arithmetic coding". The space of all strings is laid out as nested intervals whose sizes are language-model probabilities, and the user writes by steering a continuous zoom into the interval of the intended text. Probable continuations are large targets, so one gesture can select several characters, and there is no separate "word completion" mode. The Nature note reports an eye-tracker evaluation against an on-screen keyboard; the UIST paper gives the mechanism, the mouse evaluation and the information-rate analysis.

## Mechanism
- Alphabet A = {a_1..a_I} (27 symbols in UIST: a-z plus space), fixed alphabetical order. Divide [0,1) into intervals of length P(a_i); subdivide each recursively so the interval of string x_1..x_n has length P(x_1..x_n) = prod P(x_k | x_<k) (UIST eq. 1, same as arithmetic coding).
- Display: intervals drawn as boxes down the right edge; child boxes appear inside a parent once large enough (plot threshold 4 pixels on a 600x500 display).
- Control: pointer y picks the point zoomed into; pointer x sets zoom rate; pointing left zooms out (undo). The view zooms so that the point under the pointer reaches the cross-hair after S frames ("Steps"; initialised to 60, auto-adapted with a ~15 s timescale; ~40 frames/s).
- Language model: PPM order 5 with escape method D ("PPM5D+" / "PPMD5"), character level, predicts the next character from the previous five. Trained on text, then adapts online as the user writes.
- Floor: add a fixed probability lambda = 0.002 to every symbol and renormalise, so nothing is unreachable.
- Rate identity (UIST 7.3): wpm = 12 * R / H, with R = bits/s the user delivers through the pointer and H = model bits/char on the text (5 chars/word).

## Numbers
Nature note (Fig. 1b, eye-tracker, dictation from Emma in 5-minute sessions, 2 experts + 2 novices on Dasher, 2 experts on the WiViK on-screen qwerty keyboard with word completion):
- After one hour of practice: Dasher up to 25 wpm; on-screen keyboard 15 wpm.
- On-screen keyboard error rate "about five times that of Dasher" (percentage of words wrong; no exact values in text).
- Mouse-driven expert: 34 wpm (cited from UIST).

UIST 2000 (mouse, 10 novices, 6 sessions with two 5-minute Dasher exercises each, i.e. 60 min total):
- Model trained on 866 KB of Emma; 18 held-out extracts as test (883 KB total).
- PPM5D+ "can compress most English text to around 2 bits per character"; the test text costs 1.7 bits/char under the trained model (sec. 7.3).
- Expert (an author, a few hours' use): 170 chars/min = 34 wpm = 4.8 bits/s.
- With the target path highlighted (no visual search, Fig. 10): 228 chars/min (about 6.5 bits/s by the same identity; my arithmetic).
- Pointing channel estimate from Fitts' law literature: ~14 bits/s, which at 1 bit/char would be ~170 wpm (sec. 3, theoretical).
- Novice speed curves follow the power law of practice (Fig. 8); per-subject numbers are only in figures.

Follow-up search for D-pad/joystick figures (MacKenzie, yorku.ca/mack, read via page fetch):
- Three-key entry (left/right/select), NordiCHI 2002, 10 participants: static alphabetical layout KSPC 10.62 gave 9.10 wpm; a layout reordered after every letter from the two previous characters (FOCL level 2) cut KSPC to 4.23 but gave 9.61 wpm, not significantly different.
- H4-Writer, UIST 2011, 6 participants, one thumb on four gamepad buttons, static base-4 Huffman code from letter frequencies: KSPC 2.321; 7.7 wpm in session 1, 20.4 wpm in session 10; error rate 0.69%.

## Smallest scale tested, and whether the result held there
The language model is tiny by our standards: an order-5 character PPM trained on under 1 MB of one novel, tested on the same novel (best case for in-domain fit). It held: 1.7 bits/char was enough for 34 wpm by mouse and 25 wpm by gaze. Subject counts are small (4 and 2 in the gaze study).

## Training recipe and gotchas
- No training beyond counting n-gram contexts; the model must keep adapting to the user.
- Test text and training text are the same author; out-of-domain bits/char will be worse (their own figure: ~2 bits/char generally).
- The interface needs P(next symbol) for every visible node, many times per second; UIST says model speed was a selection criterion and that opening boxes too eagerly makes the frame rate unusable.
- Dasher requires continuous visual attention.
- The three-key result is a warning: a 2.5x reduction in keystrokes from a dynamic, prediction-driven layout bought no speed, because the user must re-read the layout after each step.

## Mapping onto the GBA design
- Continuous Dasher driven by Koko itself is not affordable: each node expansion is one forward pass (3-5M cycles, 0.2-0.3 s), and Dasher expands several nodes per second, each needing a full next-symbol distribution. The GBA also has no analogue pointer; a D-pad delivers far less than a mouse.
- What transfers is the identity wpm = 12R/H and the interval layout. A discrete version: show the top-k continuations as a menu sized or ordered by probability, D-pad selects. The two-level head already yields a ranked class and members; top-k from it costs nothing beyond the forward pass the model runs anyway. One pass per accepted token fits the budget if suggestions are whole words/tokens rather than characters.
- A character-level order-5 PPM (trie in ROM, built offline from the training corpus; a small adaptive table in the 256 KB RAM) costs a handful of lookups and adds per character and no multiplies. It can fill the gap between forward passes (within-word completion) and gives the lambda-floor trick for free.
- Given the three-key and H4 results, keep a static fallback letter layout so that predictions only ever shorten the path and never move the letters.

## Verdict
Test. The paper establishes that entry rate is proportional to model quality at fixed channel rate, and that integrated prediction beats a keyboard plus completion buttons for a low-rate pointer (25 vs 15 wpm). It does not tell us a D-pad's bits/s. Settling experiment: emulator prototype, three conditions on the same sentences (static grid; static grid plus top-k word row from Koko; probability-ordered letters), measure wpm and presses per character for a few users over several sessions.

## Could not verify
- No Dasher result for joystick, D-pad or button modes was found; Ward's PhD thesis was not located or read.
- Error-rate percentages in UIST sec. 6.2 were lost in PDF text extraction; per-subject wpm values exist only as plots.
- A "39 wpm mouse / 29 wpm motion device" claim surfaced only in a search snippet (secondary, unverified). A CHI 2004 joystick study (Wobbrock et al., EdgeWrite, about 6.4 wpm) was seen only as a search snippet, not read.
- The 3.5 bits/s implied for gaze (25 wpm at 1.7 bits/char) is my inference; the note does not state bits/char for that run.

## Links worth following
- https://www.inference.org.uk/djw30/papers/uist2000.pdf (method and information-rate analysis)
- https://www.yorku.ca/mack/uist2011.html (H4-Writer: four gamepad buttons, 20 wpm)
- https://www.yorku.ca/mack/nordichi2002.html (three-key entry; dynamic layout gave no gain)
- https://faculty.washington.edu/wobbrock/pubs/chi-04.01.pdf (joystick text entry on game controllers; not read)
- https://dasher.acecentre.net/docs/research (Dasher research index, may list button-mode studies; not read)
