# kokoSD 1.0: known issues and open points

kokoSD 1.0 passes its emulator checks (engine/test_game.py: 25 checks on the
SuperCard SD, EZ-Flash Omega and EverDrive GBA card models), but it has **not been
tested on real hardware yet**. These are known and not fixed yet:

## Not verified yet
- **Real hardware:** none of kokoSD 1.0 (SD card, fact file, Tell me more, typing,
  menu) has run on a real GBA and cart.
- **Card speed:** `koko-cardtest.gba` results from real carts are still missing. The
  emulator's card models leave out the card's own delay, so lookup and start-up times
  on real cards are unknown (models: 0.6-1.1 ms per 0.5 KB read).

## Behaviour that is known to be weak
- **Start-up with the fact file is silent:** opening `koko/facts.dat` takes 1.1-1.6 s
  on the card models (0.2 s without it) before the title appears; nothing is shown
  meanwhile.
- **No SD card at all** (or a cart without one): the start waits until gbaSD gives up
  on the card (a few seconds) before the title appears.
- **Typing while Koko reads in the background:** gbawriter's engine is updated once per
  video frame; while the model works, up to 4 frames are given at once, so holding B
  (delete) can repeat a little unevenly during that time. No presses are lost.
- **Accented letters** (gbawriter's SELECT conversions such as e -> é) are refused
  without a message, because Koko reads plain ASCII only.
- **START+A and START+B** (save / back in gbawriter) do nothing in Koko.

## Fact file limits (tools/facts_sd.py)
- An answer is found only by the exact title (after lowercase, articles and simple
  plurals, as the cartridge's own table); there is no fuzzy or partial match.
- Articles whose opening sentence is longer than 60 tokens, or does not decode back to
  the same text, have no answer (1,184 of 245,208 dropped).
- "Tell me more" shows at most about 6,000 characters of an article (cut at a sentence,
  "...").
- Wikipedia's markup is converted automatically: removed tables, templates and
  references can leave odd phrasing in places.
- The cartridge's own table (2023 snapshot) and the card's file (2 November 2026) can
  differ for the same subject; the card's file wins when present.

## Not done yet
- The cartridge's own fact table (1.9 MB) is still in the ROM as the fallback without
  the card file, so that ROM space is not freed for the model.
