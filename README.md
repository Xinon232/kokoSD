# kokoSD: Koko the Chatbot with an SD card

**Koko the Chatbot** is a language model that runs on an unmodified Game Boy Advance:
72 million two-bit weights in a 32 MB cartridge, writing about three words a second. Koko
was made by **rdsilver** ([Koko the Chatbot on itch.io](https://rdsilver.itch.io/koko-the-chatbot)).
All credit for the model, its training, the engine and Koko himself goes to them.

**kokoSD** is a modified version of Koko for SD-card flash carts (SuperCard SD / SuperChis
with SuperFW, EZ-Flash Omega and Definitive Edition started from the SD card, EverDrive GBA),
through [gbaSD](https://github.com/Xinon232/gbaSD). It adds:
- facts from all of Simple English Wikipedia on the SD card, and "tell me more" to read the article;
- gbawriter typing (the button chords of gbamp3, gbawriter and the other apps), accented letters;
- saved chats that carry on exactly where they were, scrolling, a START+SELECT menu;
- the look of our other apps (gbamp3, gbawriter, gbavocab) and their two fonts.

The model itself is Koko's, unchanged. GBA only (no browser version). kokoSD's changes are by
Halim Jarrar; the original Koko README follows further down.

Current version: **kokoSD 2.3**. Test build: **kokoSD 2.4 (test)** (below).

## Install
From the [Releases](../../releases) page:
1. `kokoSD.gba`: copy it anywhere on the SD card and start it from the flash cart's menu.
2. `koko.zip` (Simple English Wikipedia for Koko's facts, CC BY-SA 4.0): unzip it at the top of the card, so the
   file is `/koko/facts.dat` (250 MB). Without it Koko still chats, but cannot look facts up or show articles.

Works on SuperCard SD / SuperChis with SuperFW, EZ-Flash Omega (and Definitive Edition started from the SD card)
and EverDrive GBA (exFAT and FAT32 cards). Saved chats go to `/koko/chats/`. Buttons: START+SELECT > How to type.

## How much Koko gains from the Wikipedia file
Since 2.0 the cartridge has no fact table of its own: facts come only from the file on the SD card.
Measured with `tools/measure_facts.py` (the cartridge's lookup code run natively, once with only the
cartridge's own table of 34,127 facts, once with the SD card's file of 244,015 articles):

| Questions | Cartridge table only | With the SD file |
|---|---|---|
| 221 everyday questions ("what is a giraffe?", "who was Mozart?", "tell me about Japan", ...) | 208 (94.1%) | 215 (97.3%) |
| 2,000 subjects picked at random from Simple English Wikipedia ("tell me about ...") | 260 (13.0%) | 1,992 (99.6%) |

- Everyday subjects were mostly in the cartridge already; the file adds a few (cow, sheep, carrot, car,
  Cleopatra, robot, mathematics) and gives newer, often longer opening sentences (212 of the 215 answers
  now come from the file).
- Beyond the everyday, the file is what makes Koko an encyclopedia: about 7.7 times as many subjects
  (people, places, animals, events, science), and every answer can be followed by "tell me more" to read the
  article. Without the card there are no articles at all.
- Without a stored fact the model answers alone and makes things up, e.g. (native engine, same model):
  "what is zircon?" -> "Zircon is a town in the north of England."; "what is a swallow?" -> "A swallow is a big
  animal that lives in water, like a big sponge."; the Washington Capitals become "a football team in
  Philadelphia". With the file: "Zircon is a silicate mineral...", "The swallows and martins are a group of
  passerine birds...", "The Washington Capitals are an American ice hockey team in Washington, DC."
- Still not answered: subjects whose opening sentence is too long for Koko (goat, iron, football, love) and
  questions that do not match a title exactly (see KNOWN_ISSUES.md).


## What is new in 2.4 (test)
Four additions, each with its own switch in **START+SELECT > Settings** (kept in `koko/settings.txt`); the
last row there, **What these settings do**, explains them on the GBA:

| Setting | Choices (default first) | What it does |
|---|---|---|
| Facts | Read out / As a hint / Off | Read out: the Wikipedia sentence is Koko's reply (marked Fact), as before. As a hint: the sentence is put into your message, before your question, and Koko answers in his own words (marked Hint). Off: no facts and no sums. |
| Fact matching | Wide / Strict | Wide finds the article for more ways of asking; Strict is 2.3's matching. |
| Memory | On / Off | What you tell Koko about yourself is kept and told to him at the start of each new chat. "What Koko remembers" shows it; SELECT there forgets it all. |
| Word memory | Off / On | Which words follow which in the Wikipedia text is mixed into Koko's choice of the next word. Off by default: it made replies sound like Wikipedia without being more often right. |

**More facts, better matched.** The fact file (`koko/facts.dat`, made again: 323 MB, of which 52 MB word memory) has 264,618
articles (2.3: 244,015): titles such as "Mercury (planet)" are now in it, as "mercury planet" and, where no
other article or redirect has the name, as "mercury". With Wide matching, Koko also understands "can you tell me
about ...", "have you heard of ...", "do you know ...", "what does ... mean", "info about ...", "do you know
anything about ...", "can you tell me who was ...", and subjects with extra words: "the singer Taylor Swift",
"Napoleon from France", "the planet Mercury" (the planet, not the metal). Measured with `tools/measure_match.py`:
50 questions in other wordings, **19 answered with 2.3, 41 with 2.4**, no wrong article among them; the 221
everyday questions of `tools/measure_facts.py`: 212 -> 216.

**Facts as a hint.** Native tests (same model, same seeds) on ten subjects: as a hint, the answers are in Koko's
friendly style and often use the fact ("A volcano is a big, strong volcano that can be very hot...", "Plants use
sunlight to make food through photosynthesis..."), better than with no fact ("The giraffe is the tallest mountain
in the world"), but still wrong at times ("Giuseppe Amadeus Mozart", Einstein "born in Dresden"). Read out is
always right, so it stays the default. The hint costs time: the fact and your message are read again (about 0.4 s
a word: 10 s for a short fact, up to 25 s for a long one).

**Memory.** After each message Koko notes sentences such as "my name is ...", "call me ...", "I am 12 years
old", "I live in ..." / "I'm from ...", "I like / love / enjoy ..." (the last two), "my favorite X is ..." (the
last two), "I have a ...". They are kept in `koko/memory.txt`, one sentence a line, so the file can be read and
changed on a computer. At a new chat Koko is told them while you type the first message (in the background), in
the form that worked best in native tests: your turn "My name is Sam. I live in Berlin." and Koko's "Got it! Your
name is Sam. You live in Berlin." Then "what is my name?" -> "Your name is Sam." and "hi" -> "Hi Sam!"; the
name and favourites are used most reliably, the town and food less so. Saved chats go on without it (they already
have their history). Koko sees about 256 words back, so in a long chat what he was told fades.

**Word memory.** `tools/words_sd.py` counts, in all the article text, which word follows each pair and each
triple of words (seen at least 4 times; the 4 most common next words each) and appends the tables to
`facts.dat` (52 MB: 2.1 million three-word and 1.1 million two-word contexts). For each word Koko writes, the card is asked what follows his
last three words (or two), and that is mixed into his own choice (32/256). On 1.2 million words of held-out
Wikipedia text it makes Koko's guess of the next word slightly better (top-1: 31.1% -> 31.5%). In chat the
effect is small and mixed: replies sound a bit more like an encyclopedia ("about 384,000 km to the moon",
"Paris is the third-largest city in France") without being more often right (it also adds made-up dates).
No speed cost measured in the emulator (2.74 against 2.79 words/s).

## What is new in 2.3
- **Whole articles for "tell me more".** The article is read from the card page by page instead of all at once
  into memory, so it is no longer limited to 6,000 characters: `koko/facts.dat` (made again with
  `tools/facts_sd.py`, 249.8 MB) keeps articles up to 64,000 characters, which is all of nearly every article
  (19 of 244,024 are longer). Eminem's article, for example, was cut after a quarter; it is now 100 pages.
- A cut article ends at a real sentence end, not after an abbreviation such as "Dr." or an initial; empty section
  names ("Notes", "References") are no longer left at the end.
- The reader keeps its line table in video memory the screen does not use (1.9 KB of RAM freed).
- **Copy the new `koko/facts.dat`** to the card with this version (the old file still works, with its shorter
  articles).

## What was new in 2.2
- **No more flicker.** The suggestion line was cleared and written again after every letter, and the chat was
  blanked and redrawn when a message was sent: on a real screen the empty line or chat showed for a frame. Now
  the suggestions stay until the new ones are ready, the line and the header are redrawn in the vertical blank,
  and a full chat redraw writes each line over the old one (only the space beside the text is cleared). Measured
  frame by frame in the emulator while typing, sending and Koko answering: no frame where something vanishes and
  comes back.
- Credits: "kokoSD: halimj" on the title screen and in About.

## What was new in 2.1
- **Presses in the menu are no longer typing.** The buttons pressed in the menu (and in How to type, About,
  Saved chats, Rename, the self test) were handed to the typing engine afterwards: B that closed the menu
  deleted a letter, START sent the message. Now everything is let go and forgotten when the menu closes.
- **Button timing guard.** The buttons' debounce and the typing engine's frames are timed by timers 2 and 3. If
  they are stopped or their count stands still, a button can never change again: START stays "held", SELECT then
  opens the menu, and nothing else answers (in the emulator, 2.0 with the timers stopped freezes exactly so). They
  are started again at once; the self test log says how often ("timers started again").
- **Damaged saved chats are refused** ("That chat could not be opened", a new chat) instead of loaded: every word
  number, position and chat line is checked first. A damaged word number used to make kokoSD write outside its
  table.
- Tests for all three (engine/test_game.py 17 and 18).

## What was new in 2.0
- **A clean rebuild** after crash hunting on a SuperChis with SuperFW: the 1.4.x diagnostics are gone again
  (no interrupts, no start-up ROM check, no restart numbers) and the SuperCard uses gbaSD's own driver, as our
  other apps do.
- **31.5 MB ROM** instead of 33.4 MB: the cartridge's own fact table (1.9 MB) is gone, so the top 2 MB of the
  32 MB cartridge space stay unused. The program, fonts and tables come first, the model data last.
- **Keep-alive (SuperCard):** a SuperCard's "ROM" is SDRAM. Our other apps are a few MB that are read all the
  time; Koko's 31 MB are mostly left alone while one types or waits. kokoSD now reads one word of every 512
  bytes, the whole ROM about every two seconds (from fast RAM, about 4% of Koko's speed).
- **Self test:** on a SuperCard it first leaves the ROM unread for 90 seconds, then checks it ("After 90 s
  unread: ..."). B skips that step.
- Slow RAM is set to its normal speed at start, in case SuperFW's "Fast EWRAM" is on.

## What was new in 1.4 (test)
- **Self test** (START+SELECT menu, last row) for crash hunting on a real cart: round after round, every
  1 MB of the ROM against sums written at build time (tools/fixrom.py), 24 lookups of real titles in the
  fact file (every fourth with its article), the ROM pieces the SuperCard's SD registers could overlay
  again, and a fixed sentence through Koko against the native engine's result. B stops; the results go to
  /koko/selftest.txt.
- **EverDrive GBA:** its register window at 0x09FC0000 (where the end of Koko's ROM is: fonts, tables) is
  now mapped only during card access, as the SuperCard's SD interface since 1.1.
- **Restart detector:** if the cartridge starts again by itself, the title says so (keys held, state).
- `tools/crash_test.py`: random button sessions in the emulator (typing, chords, menu, facts, Tell me
  more, saving and opening chats, scrolling); Koko must answer the buttons after each.

## What is new in 1.3
- **Rename and delete saved chats:** START on a chat in Saved chats: Open, Rename (type the
  new name, START+A saves, START+B cancels) or Delete ("Delete this chat?" A: yes, B: no).
- **Credits in the cartridge:** the title screen and About kokoSD name Koko's author,
  rdsilver (https://rdsilver.itch.io/koko-the-chatbot).

## What is new in 1.2
- **Saved chats.** START+A saves the chat to the SD card (`/koko/chats/chatNNN.kch`, about
  50-250 KB each). START+SELECT > Saved chats lists them by their first message
  (alphabetically); A opens one and the chat carries on exactly where it was: the file
  holds Koko's whole state (attention cache, mixer state, token history), so he does
  not read the conversation again. Saving again after opening overwrites the same file.
  A message being typed is not part of the save. A chat made with other weights is not
  opened.
- **Scrolling.** Hold START and press Up / Down: the chat moves one line (the last 64
  lines are kept); typing returns to the newest lines.
- **START with nothing typed does nothing** (no more ready-made messages), and neither
  does a message Koko cannot read at all (nothing left in its plain copy).

## What is new in 1.1.3 (tested on a SuperChis Prime with SuperFW)
- **Starts on SuperCard-type carts.** Koko set the fastest cartridge timing on its first
  instruction; a SuperCard / SuperChis cannot read that fast, so the GBA crashed at once
  (blank screen). SuperFW could not patch it out (the write was built from 0x04000200 + 4).
  The fast timing is now set only on an EZ-Flash Omega or EverDrive.
- **Nintendo logo in the header** (devkitPro's gbafix), so a start through the BIOS works.
- **Start-up steps on screen:** "Looking for the SD card...", "Opening the fact file...".
- **START with L or R in either order** takes a suggested word (L or R a moment before
  START used to send the half-typed message).
- **B always stops Koko:** presses made while he writes no longer push B out of the queue;
  B while he still reads the message means no reply. The header shows the phase
  (reading..., looking up..., B: stop).

## What is new in 1.1
- **The look of our other apps** (gbamp3, gbawriter, gbavocab): white screen, black text, the
  name in the small font at the top, grey hints, the light blue selection bar, gbamp3's
  Search field for typing. Only their two fonts are used: the SuperFW 16 px font for all
  text to read and the 5x7 font for the header and hints (engine/text, from gbamp3 15.0).
  Koko's portrait, snow, bubbles and Inter font are gone from the cartridge. Your lines
  are on the right in grey, Koko's on the left in black; replies from the helpers are
  marked "Fact" or "Sum".
- **Accented letters** (gbawriter: keep the letter's buttons held and press SELECT) are
  typed and shown as typed. Koko and the helpers read a plain-letter copy ("café in
  München" -> "cafe in Munchen"; ß -> ss, æ -> ae, ø -> o; Greek and Cyrillic are left out),
  since the model knows plain ASCII only.
- **SuperCard fix.** gbaSD leaves the SuperCard's SD interface mapped into the ROM space;
  while it is, the card's registers take the place of ROM at 0x09000000, 0x09100000 and
  0x09800000, where Koko's 32 MB ROM keeps model data. kokoSD now maps the interface only
  during each card read or write (engine/diskio.c). The emulator's SuperCard (test/runner.c)
  now models this, assuming the registers decode a whole 1 MB each: kokoSD 1.0 gave
  wrong replies there.

## What is new in 1.0
- **Facts from the SD card.** Copy `koko/facts.dat` to the folder `/koko` on the SD
  card. "who is X", "what is X", "tell me about X" are then answered from every
  usable Simple English Wikipedia article: 244,024 articles and 68,551 redirects
  (dump of 2 November 2026, 225 MB; the cartridge's own table had 34,127 opening
  sentences; without the file it is used as before). Lookups take a few card
  reads.
- **Tell me more.** After such an answer, "tell me more" (or "more", "go on") opens the
  article to read: Up / Down scroll, Left / Right / L / R turn a page, B goes back.
  "tell me more about X" opens X's article directly. Koko does not read the article
  (that would take minutes); the chat carries on after it.
- **gbawriter typing** instead of the on-screen keyboard: hold a direction and press
  B / A / R for a letter (L as well for the second layer), A space, B delete, R
  capitals, SELECT dot / digits / signs, as in gbawriter, gbamp3 and the other apps.
  START sends (on an empty line: a ready-made message), START+L / START+R take a
  suggested word, START+Left / Right move the cursor, **START+SELECT opens the menu**:
  New chat, Facts and sums On / Off, How to type (all the buttons), About kokoSD.
- Buttons are debounced (a bouncing contact is not a second press), and presses made
  while Koko works are kept in order.

**Known issues:** see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Building kokoSD
The weights are not in git. Take Koko's data block out of a released `koko.gba`,
then build (devkitARM in /opt/devkitpro, Python 3 with numpy):

    git submodule update --init                  # gbaSD
    python3 tools/blob_from_rom.py koko.gba     # -> build/koko.bin
    python3 tools/build.py                      # -> build/koko.gba
    python3 tools/build.py --cardtest           # -> build/koko-cardtest.gba
    python3 tools/build_runner.py               # emulator runners (libmgba), with gbaSD's card models
    python3 tools/facts_sd.py simplewiki-latest-pages-articles.xml.bz2   # -> out/koko/facts.dat
    python3 engine/test_game.py                 # KOKO_RUNNER=build/runner-sc | -ezfo | -edgba | build/runner (no card)

The dump is at dumps.wikimedia.org/simplewiki/latest/ (needs `pip install mwparserfromhell`).

**Card test** (`koko-cardtest.gba`): reads the SD card for about 15 seconds (0.5,
4 and 8 KB pieces from random places, and 8 KB pieces in a row), shows the time
per read and saves the results to `/koko-cardtest.txt`. Nothing else is written.

## Licence and credits (kokoSD)
Koko the Chatbot (code, model, weights, fact table, artwork): **rdsilver**,
https://rdsilver.itch.io/koko-the-chatbot ("The Koko authors" in LICENSE and LICENSE-MODEL).
kokoSD's additions: Halim Jarrar (halimj), with gbaSD, gbawriter's typing engine, gbamp3's text drawing and
GBAVocab's accent folding (all Halim Jarrar, GPL-3.0-or-later) and the SuperFW font
(David Guillen Fandos, GPL-3.0-or-later; UNSCII and GNU Unifont glyphs).

gbaSD and the kokoSD additions are GPL-3.0-or-later (LICENSE-GPL), so the
cartridge as a whole is distributed under GPL-3.0-or-later. Koko's own files keep
their MIT licence (LICENSE); the weights, fact table and artwork stay
CC BY-SA 4.0 (LICENSE-MODEL).

---

# Koko the Chatbot: a language model on a Game Boy Advance cartridge

A language model that runs entirely on an unmodified Game Boy Advance: 72 million two-bit weights, lookup
tables, a 16,384-token tokenizer and a fact table in a 32 MB ROM, generating about three tokens a second.
The same C code is compiled for the GBA, natively (as the reference the ROM is tested against) and to
WebAssembly for the browser version.

See release/README.md for how to play and release/CREDITS.md for data, font and licences.

## Layout
| Folder | What is in it |
|---|---|
| engine/ | The cartridge: inference engine (koko.c), tokenizer, sums, fact table, screen (ui.c), game (game.c), ARM kernels, exporter, and the references the engine is checked against (ref_float.py, ref_int.py) |
| web/ | The browser version: the engine compiled to WebAssembly and a page that shows the cartridge's screen |
| train/ | Model definition and training code (PyTorch), experiment launchers |
| gen/ | Generation and filtering of Koko's own training conversations, fact drills |
| study/ | Post-training studies (fast output step, fact recall, sampling) |
| scoreboard/ | Comparison with GPT-2, SmolLM2 and TinyStories models |
| research/ | Results and notes from each stage |
| art/ | Koko's portrait |
| bench/, src/, tools/, host/, test/ | Lookup kernel and benchmark ROMs, the earlier stories-only ROM, native test programs |

engine/DESIGN.md is the running design log: number formats, the chain of bit-exact references, and what changed after
the first run on real hardware.

## Building
Needs arm-none-eabi-gcc, clang with the wasm32 target, Python 3 with numpy, torch, tokenizers and Pillow, and
libmgba for the emulator tests. The trained weights are not in this archive (they are inside the ROM and koko.bin);
with a model.pt in artifacts/models/<run>/ the pipeline is:

    python3 engine/extract.py <run>      # weights -> integers
    python3 engine/outstep.py            # fast output step data
    python3 engine/facts_build.py        # fact table
    python3 engine/export.py             # build/koko.bin and gen/koko_layout.h
    python3 engine/build_game.py         # build/koko.gba
    python3 engine/test_game.py          # the ROM in mGBA against the native engine
    web/build.sh && node web/test_browser.cjs

## Licence
Code: MIT (LICENSE). Model weights, fact table and artwork: CC BY-SA 4.0 (LICENSE-MODEL).
