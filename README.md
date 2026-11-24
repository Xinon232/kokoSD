# kokoSD: Koko the Chatbot with an SD card

kokoSD 1.1.3 is Koko (below) on SD-card flash carts: SuperCard SD, EZ-Flash Omega (and
Definitive Edition, started from the SD card) and EverDrive GBA, through
[gbaSD](https://github.com/Xinon232/gbaSD). GBA only (no browser version).

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

**Known issues:** see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) (not tested on real hardware yet).

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

## Licence (kokoSD)
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
