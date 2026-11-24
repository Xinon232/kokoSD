# Koko engine: design notes (started 20 Nov 2026)

Model: artifacts/models/h_koko_L8 (8 layers, 256 wide, attention in layers 1,3,5,7, mixer in 0,2,4,6,
128 token-routed experts x 128 hidden, 4-bit bigram+trigram tables at layer 0, 16,384 tokens, context 256).

## Chain of references (each must match the one before)
1. train/model.py (PyTorch)                      - the trained model
2. engine/ref_float.py (numpy, token at a time)  - matches 1 to 1.5e-5 over a 256-token window (check_float.py)
3. engine/ref_int.py (integers only)             - same loss as 2 (1.9381 vs 1.9380 on 1,024 positions), same
                                                   top token 97.7-99.2% (check_int.py)
4. engine/koko.c, native build (build/koko_twin) - equals 3 exactly: state and all 16,384 scores for 4 x 256 tokens (check_twin.py)
5. same koko.c + bench/tern.s in mGBA           - equals 4 exactly: state and all 16,384 scores for 12 tokens
                                                   (engine/test_gba.py 12 --full). ROM 31.05 MB.

## Gotcha
Trits must come from the training code's float32 quantiser (extract.trits). Many latent weights sit on
the +-0.5 rounding threshold; recomputing in float64 flips dozens per matrix and shifts scores by ~0.3.

## Integer formats (ref_int.py is the specification)
- Everything Q12 in int32: stream, normalised stream, projections, hidden units, attention scores, output scores.
- rs(x, n) = (x + 2^(n-1)) >> n, arithmetic shift, everywhere.
- Norm: r = max(isqrt(sum(x^2)/D + 168), 16); inv = 2^40 / r; t = rs(x*inv, 28); out = rs(t*g, 12). g is Q12.
- Embedding: int8 rows, per-row Q16 step: x = rs(e * step, 4).  Inject rows: 4-bit levels * per-row Q12 step.
- Ternary matrices: integer sum of +-inputs, then one Q16 multiplier (per matrix; per expert; per row for the head).
  Keys use only the per-row Q12 gain (the matrix scale cannot change a sign).
- RoPE: cos/sin tables Q14 for positions 0..255, 32 frequencies per head.
- Query per head: r = max(isqrt(sum(q^2)/64 + 168), 1); m = (c_h << 20) / r; qs = rs(q*m, 24); c_h = Q16 exp(temp)/8.
- Score = sum(+-qs) + dbias[h][distance] (Q12). Top 8, earlier position wins ties.
- Softmax over the 8: d = (best - s) >> 5; w = exp_table[d] (uint16, 2,048 entries, 0 beyond); R = 2^31 / sum(w);
  p = (w*R) >> 16 (Q15).
- Values: per position int8 mantissas + one shift (so absmax <= 127). mix = rs(sum(p * mant << shift), 15).
- Mixer: y = sum taps_i * xn[t-1-i] + sum avg_i * ema_i; ema_i = rs(ema_i*r + xn*(65536-r), 16), r Q16 of .5/.8/.95;
  x += rs(y, 12).
- Expert: pre = max(rs(w1sum * w1_m[e], 16), 0); hid = rs(pre*pre, 12); x += rs(w2sum * w2_m[e], 16).
- Output: score[row] = rs(headsum * head_m[row], 16); copy vote: + rs(copy_gain * p, 15) for the token after each
  attended earlier position (last attention layer, head 0).
- Measured ranges (2,000 tokens): stream absmax 1,068; normalised 42; hidden 498; scores 28. No int32 overflow
  (ref_int asserts it).

## Output step (study/, research/check1-output-step.md)
Candidates = 256 most frequent tokens + tokens in context + previous token's 64 successors + members of the
16 best of 256 groups picked by a 2-bit router. About 1,400 rows, model's top token kept 99.2%.

## Still open
- Positions past 255 (ring buffer): RoPE tables and the distance bias are defined for 0..255 only.
- Exporter + ROM layout (pointer-step bytes as in bench/gen.py), C twin, ARM kernels (bench/tern.s), UI.
- Sampling: temperature 0.5, top 40, block repeated phrases (research/check3-4-facts-sampling.md).
- Arithmetic: compute sums in the game.

## First GBA timing (20 Nov 2026, mGBA 0.10.2, unoptimised C around the two assembly kernels)
Body 4.23M cycles/token at short context (3.96 tok/s before the output step); the key scan adds about 0.5M
at a full 256-token window. Per token: key+value projections 818K, query projection 595K, expert out 569K,
mixer 557K (C with 64-bit multiplies; should be ~80K), expert in 526K, table builds 378K, norm 302K
(17 norms; should be ~100K), attention out 294K, top-8 mix 136K.
Scoring an output row costs 1,100 cycles through the one-row C path (465 in a batched kernel).
Memory: slow RAM 228.6 KB of 256 KB used, fast RAM 27.5 KB of 30 KB.

## Status at end of 20 Nov 2026
- Embeddings are 6-bit (4 values in 3 bytes): loss 1.9328 vs 1.9381 at 8 bits on 1,024 positions (5-bit 1.9346,
  4-bit 1.9413). Saves 1 MB.
- Fast output step is in the reference, twin and ROM (engine/outstep.py builds its data: outstep_h_koko_L8.pkl).
  256 frequent + 32 successors + window tokens + 16 of 256 router groups: about 1,000-1,400 rows, model's top
  token kept 99.0%, sampling mass (temperature 0.5, top 40) 98.7%. Streaming top-40, xorshift32 sampling.
- Exactness: check_twin.py 256 (reference vs twin: state, all scores, top-40 list, sampled token, 4 texts) and
  test_gba.py 24 (twin vs ROM in mGBA: state, top-40 list, sampled token) both pass.
- ROM image 31.1 MB (data 29.65 MB). Slow RAM 233 KB of 256 KB, fast RAM 28 KB of 30 KB.
- Speed in mGBA, short context: body 4.30M cycles + output step 1.09M = 5.39M = 3.11 tok/s.
  A full window adds about 0.5M (key scan).
- engine/chat.py talks to the native build of the cartridge engine (build/koko_gen).

## Optimisation list (estimated savings per token)
- Mixer 573K -> ~200K: assembly, 64-bit multiply-accumulate; change the running averages to
  ema += ((xn - ema) * k + 512) >> 10 with k = 512, 205, 51 (32-bit multiply; re-verify loss).
  Slow RAM costs 6 cycles per 32-bit access, so rotate the history by pointer instead of copying.
- Norm 303K -> ~150K: assembly inner loops.
- Output step 1.09M -> ~0.7M: the per-candidate bookkeeping (dedupe, batching, top-40 insert) runs from ROM.
- Putting attention() in ROM instead of fast RAM cost 390K, so fast-RAM placement matters; 2.7 KB of it is free.

## Still to do
- Conversations past 256 tokens (ring buffer; RoPE/bias tables stop at 255).
- Chat screen, on-screen keyboard, predictive typing, tokenizer on the cartridge (vocabulary strings).
- Phrase-repeat block; sums computed by the game.

## Update, later on 20 Nov 2026
Spec changes (all re-verified: ref vs twin exact for 300 tokens on 2 texts, twin vs ROM exact for 300 tokens):
- Mixer running averages: ema += rs((xn - ema) * k, 10), k = 512, 205, 51. State and coefficients interleaved
  per dimension (blob array MIXC). Assembly kernel mixer_k.
- RoPE for any position: phase = pos * inc[f] (32-bit wrap), index = ((phase + 2^19) >> 20) & 4095 into one
  4,096-entry Q14 sine table. No per-position tables.
- Rolling window: ring slot = position mod 256 for keys, values and tokens; bias by distance (slot difference
  mod 256); ties in top-8 go to the lower slot; copy votes use the token in the next slot.
- Integer loss with the final spec: 1.9353 (float 1.9380), 1,024 positions.
- Rolling window quality (float reference, engine/ring_test.py, ring_long.py): tokens 256-511 score 2.1548
  continuing vs 2.1554 in fresh windows (4 domains); out to position 2,047 on chat text the two stay within 0.01.

Speed in mGBA (engine/test_gba.py 300): short context body 3.78M + output 0.84M = 3.55 tok/s;
full window body 4.73M + output about 1.0M = about 2.9 tok/s. The remaining cost is the table kernel itself
(6.4 cycles per stored byte), i.e. the size of the model; the earlier cost model was about 35% optimistic.
Norm stays at 16K each: the multiply by the per-dimension gain read from ROM dominates.

## Tokenizer on the cartridge (plan)
GPT-2 style byte-level BPE: 16,384 entries, 16,125 merges, 107 KB of token text, longest token 18 bytes.
Encoder needs the GPT-2 word-splitting rule for ASCII and a pair -> (rank, id) hash table (about 130 KB).

## Tokenizer on the cartridge: done (engine/koko_tok.c, tok_export.py)
koko_encode / koko_token_text. Tested natively against the training tokenizer on 65,819 ASCII strings
(training turns plus random keyboard input): identical ids, and decoding returns the input text.
Tables add 290 KB.

## Size, in consistent units (1 MiB = 1,048,576 bytes; cartridge limit 32 MiB = 33,554,432 bytes)
Data blob 31,357,360 bytes = 29.90 MiB; code about 12 KB. Free: about 2.1 MiB.
(Earlier notes that said "31.1 MB" were bytes / 10^6; the headroom is larger than those notes implied.)

## Next
Chat screen and on-screen keyboard (src/main.c from stage 1 has a keyboard and text renderer to adapt),
generation loop with a stop at the user marker, phrase-repeat block, sums computed by the game, predictive typing.

## The game: first playable build (20 Nov 2026)
engine/game.c + engine/build_game.py -> build/koko.gba (31,375,360 bytes). Chat log (10 rows, word wrap, scroll),
56-character message line, 4x13 on-screen keyboard (SELECT = capitals, L/R = ready-made messages), START sends,
B stops a reply. Each message is fed as <|user|> text <|koko|>; the reply ends at a marker token or 120 tokens.
Temperature 0.5, top 40, seed from button timing. The conversation continues across turns (rolling window).
engine/test_game.py drives it in mGBA: messages reach the model exactly as the training tokenizer encodes them,
and the first reply equals the native engine's for the same seed. Measured 3.2-3.4 tok/s on short chats.
Not run on real hardware; the header's boot-logo field is blank (emulators and most flash carts accept that).
Not done yet: predictive typing, phrase-repeat block, sums computed by the game, reading the message while
it is typed (now about 0.25 s per message token of "reading" before the reply starts).

## Reading while typing, suggestions, rewind (20 Nov 2026, evening)
- koko_step_begin / koko_step_part: a step in slices of one layer; koko_yield hook polls buttons every few ms.
- koko_mark / koko_rewind: mixer state snapshot (24 KB, kept in spare video memory at 0x06004000 on the GBA),
  position counter back; ring slots written since the mark are lost, so the usable window (nvalid) shrinks by
  that many. In ref_int too; engine/check_rewind.py: reference == twin through 6 rewinds across the ring wrap.
- Editor: every finished word (text before the last space) is fed in the background; deleting into a fed word
  rewinds to the start of the message and feeds the surviving words again.
- koko_suggest: up to 3 whole-word tokens (engine/words.py: 8,170 after a space, 300 message-initial) that
  extend the typed prefix, ranked by the model. Only matching candidates are scored. L and R accept the top two.
- START on an empty line cycles ready-made messages; START on a non-empty line sends.
- engine/test_game.py (mGBA, button presses): ready-made message, typed message with suggestions checked
  against the native engine and one accepted, delete-and-retype with a rewind, two-turn conversation; every
  reply identical to the native engine for the same seed.
- Bug found on the way: koko_suggest writes up to 3 ids; the game's buffer had 2 and the third overwrote the
  message length.

## Repeated-phrase rule and sums (20 Nov 2026, night)
- koko_reply_begin / koko_reply_end; in koko_pick a token that would repeat a 4-token phrase already in the
  current reply gets weight 0; if that leaves no weight, the best allowed candidate is taken. In ref_int too
  (engine/check_repeat.py: reference == engine). Over 95 story/poem/joke/"what is" prompts at the game's
  temperature, 37 replies repeated a 4+ token phrase without the rule and 0 with it.
- engine/sums.c: "<number> <op> <number>" with + - x * / and plus, minus, take away, times, multiplied by,
  divided by; 20,000 generated sums checked against Python. The answer text is tokenized and fed as Koko's
  reply, so the conversation continues normally. "we won 3-2" is not treated as a sum.
- engine/test_game.py now has 5 scenarios, all passing.

## New screen (20 Nov 2026, night)
engine/game.c now draws in bitmap mode 4 (240x160, 256 colours); the old text-mode screen is kept as
engine/game_text.c. Proportional anti-aliased font: Inter Medium 10 px and SemiBold 11 px (SIL Open Font
License 1.1; engine/font_gen.py -> gen/font_ui.h, 4 levels). "Paints" = ink-on-surface palette ramps, so
anti-aliased text needs no blending. User messages sit in a rounded bubble on the right, Koko's text on the
left, input bar with send button, suggestion chips, keyboard panel that appears while typing.
Video memory takes 16-bit writes only, so pixels are read-modify-write. The mixer snapshot moved to
0x0600A000 (the unused second bitmap page). A full change of layout takes roughly 0.15 s to draw.
All 5 scenarios of engine/test_game.py pass with the new screen.
Credits to add in the README: Inter by Rasmus Andersson (OFL 1.1).

## Browser version (web/)
Same C engine compiled with clang --target=wasm32 (freestanding, 20 KB) + build/koko.bin; web/site is a static
page (index.html, style.css, app.js). Read-while-typing, suggestions, rewind, sums and the repeated-phrase rule
all use the cartridge code. web/test_node.mjs: 4 prompts, tokens identical to build/koko_gen. web/test_browser.cjs
(Playwright): typing, Tab suggestion, edit, sum, GBA-speed pacing, phone width, no page errors. out/koko-web.zip
is the itch.io upload.

## Shared screen code; browser version shows the cartridge screen (20 Nov 2026, late)
- engine/ui.c + ui.h hold everything that is drawn (palette, font drawing, chat lines, input bar, chips, keyboard,
  reply layout). engine/game.c is the cartridge logic only. On the GBA the frame buffer is video memory
  (16-bit read-modify-write); elsewhere it is ui_fb[240*160] with ui_pal[48].
- Cartridge change: after a reply the screen stays in the reply layout until a button is pressed (that press is
  then handled by the editor), so long replies are not hidden by the keyboard.
- web/: the page is now a 240x160 canvas of that same screen at a whole-number scale; keyboard typing,
  Enter = START, Tab / Shift+Tab = L / R, Esc = B, and clicking the on-screen keys. Messages are limited to the
  cartridge's 56 characters and its keyboard's character set.
- web/test_browser.cjs compares the typing screen after "what is the " with the same moment in mGBA
  (out/cart_typing_ref.ppm from engine/test_game.py): 0 of 38,400 pixels differ.
- web/artifact/: the same page as one HTML file plus koko.wasm and the data as three data-only wasm modules.

## Stop square, new-chat key, queued presses
- ui_reply_button: the input bar's round button is a stop square while Koko answers (clickable in the browser).
- The ';' key became a new-chat key (UI_KEY_NEW, row 3 column 9, a page with a plus). Pressed twice (the header
  asks "A again: new chat"); anything else cancels. New chat = koko_reset + ui_clear + greeting.
- Button presses are now a queue of edges, not a mask: two quick presses of the same button during a redraw or
  a model slice used to merge into one. ui_yield polls during long redraws.
- engine/test_game.py: 15 checks (adds new chat, and typing straight after a reply). web/test_browser.cjs: 10.
- web/make_artifact.py builds the single-file page for publishing.

New chat is now a single press of the new-chat key (no confirmation), on the cartridge and in the browser.

## Fact table (21 Nov 2026)
engine/facts_build.py -> engine/facts_table.pkl -> blob arrays FACT_HASH / FACT_OFF / FACT_TEXT (1.89 MB):
the opening sentence (two if the first is under 60 characters) of 34,127 Simple English Wikipedia articles
(CC BY-SA; attribution needed in the credits), as 14-bit token ids, keyed by an FNV-1a hash of the normalised
title. Ranked by article length, everyday words x5; chatty words ("you", "up", "love", ...) are never keys;
short names for well-known people ("mozart", "washington") are aliases to the same sentence.
engine/facts.c: fact_reply() recognises who/what/where is|was|are, what's, tell me about, do you know about,
explain, define; strips articles, "please", a trailing "is"; tries plural forms. The sentence is fed as Koko's
reply (like sums), so the conversation continues. No match -> the model answers as before.
Checks: 37,397 of 37,404 stored facts returned exactly when asked in varied wordings (the 7 are titles that
contain the question words themselves); 108 of 115 hand-picked everyday subjects covered; chat such as
"who are you", "what is up", "what is the capital of France?" does not trigger it.
ROM is now 33,296,384 bytes of the 33,554,432 limit (258 KB free).

## Showing what is not the model
- Replies fed from the fact table start with a small book; computed sums with an equals box (Line.mark in ui.c).
- The '=' key became the helpers key (UI_KEY_HELP, row 2 column 12, a book): fact table and calculator on/off.
  The dot beside the title is solid when they are on and hollow in "model only". The browser page has a
  matching switch. With helpers off every reply is generated; engine/test_game.py checks it equals the native engine.
- Tests: 18 cartridge checks, 12 browser checks.

## Model on the cartridge: i_kd (fine-tuned 20 Nov 2026)
See research/finetune-results.md. engine/*.py now default to run "i_kd" (artifacts/models/i_kd);
study/cap_i_kd_*.pt and engine/outstep_i_kd.pkl are its router data. Tests after the swap: check_twin,
check_rewind, check_repeat, test_gba 24, test_game (18), web test_node (4), test_browser (12): all pass.

## Release preparation (21 Nov 2026)
- Title screen: Koko's portrait (art/husky.py -> gen/husky.h, 96x88, 15 colours at palette 48..62), name at twice
  the title font, blinking "Press START". ui_title / ui_title_prompt in ui.c; START or A (Enter or a click in
  the browser) begins the chat. Tests press START first (BOOT in engine/test_game.py).
- release/: CREDITS.md (licences checked against each source's own page), README.md, itch-page.md, screenshots
  and koko.gif (a recorded emulator session). Downloads: out/koko-gba.zip, out/koko-web.zip, out/koko-release-kit.zip.
- The browser page has a "Credits and licences" section; CREDITS.txt is in both zips.
- Not done: Firefox and Safari (only Chromium is available here), real hardware, choosing a licence for the
  project's own code and weights.

## First run on a real Game Boy Advance (21 Nov 2026) and what it changed
The ROM boots and chats from a flash cartridge. Two things the emulator did not show:
- Flicker when a reply wrapped or scrolled: chat_draw blanked the chat area and redrew it over several frames.
  Now chat_update() waits for the end of the frame, slides the existing rows up (rows_up, 32-bit copies) and
  draws only the changed lines. Lines remember where they are drawn (Line.y). A user bubble at the top edge
  and layout changes still use the full redraw. web/test_draw.mjs: after each of 352 tokens across six replies
  the screen equals a full redraw.
- Title screen: 34 snowflakes drift behind the portrait (ui_title_tick, once per frame). A flake is only drawn
  on plain background pixels and restored afterwards, and never in the blinking line's band.

## Keyboard bottom row (21 Nov 2026)
`. , ! ? apostrophe [ space bar, three cells ] / ( ) new-chat delete`. The colon and double-quote keys are gone (division still
works as "divided by"). ui_key_move() treats the bar as one key standing at column 6; the game, the browser page
and the test harness (Pad in engine/test_game.py) all move the cursor through it.

## Licence and source release (21 Nov 2026)
Code: MIT (LICENSE). Model weights, fact table and artwork: CC BY-SA 4.0 (LICENSE-MODEL). Copyright holder is
written as "The Koko authors". out/koko-source.zip is the public source archive: no weights, no downloaded
third-party code or models (ref/, dl/, scoreboard/hf/), no machine notes.
Line wrap without flicker: when a reply's last word moves to a new line, the old line is not cleared and redrawn;
only the pixels after its new end are erased (Line.dn = characters drawn). Every incremental update now starts at the
end of a frame.
