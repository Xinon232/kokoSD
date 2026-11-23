# kokoSD: Koko the Chatbot with an SD card

kokoSD is Koko (below) on SD-card flash carts: SuperCard SD, EZ-Flash Omega (and
Definitive Edition, started from the SD card) and EverDrive GBA, through
[gbaSD](https://github.com/Xinon232/gbaSD). Planned: the fact table on the SD card
with all of Simple English Wikipedia and "Tell me more", and gbawriter typing
instead of the on-screen keyboard. GBA only (no browser version).

## Building kokoSD
The weights are not in git. Take Koko's data block out of a released `koko.gba`,
then build (devkitARM in /opt/devkitpro, Python 3 with numpy):

    git submodule update --init                  # gbaSD
    python3 tools/blob_from_rom.py koko.gba     # -> build/koko.bin
    python3 tools/build.py                      # -> build/koko.gba
    python3 tools/build.py --cardtest           # -> build/koko-cardtest.gba
    python3 tools/build_runner.py               # emulator runners (libmgba), with gbaSD's card models
    python3 engine/test_game.py                 # KOKO_RUNNER=build/runner-sc | -ezfo | -edgba

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
