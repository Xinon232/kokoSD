# kokoSD 1.3: known issues and open points

kokoSD 1.3 passes its emulator checks (engine/test_game.py: 37 checks on the
SuperCard SD, EZ-Flash Omega and EverDrive GBA card models, 27 without a card) and about
260 random button sessions without a hang. On real hardware it has run on a SuperChis
Prime with SuperFW (start, typing, replies); the other carts are not tested yet. These are known and not fixed yet:

## Not verified yet
- **Real hardware:** only a SuperChis Prime (SuperFW) so far; EZ-Flash Omega and
  EverDrive GBA are untested. Koko's speed on the SuperChis is not measured yet.
- **SuperCard register windows:** how wide the SuperCard decodes its SD registers is
  assumed (1 MB each at 0x09000000, 0x09100000, 0x09800000). 1.1 maps the SD interface
  only during card access, which is safe whatever the width.
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
- **START+B** does nothing in Koko.
- **Saved chats:** not tested on real hardware (SD card writes). The list shows at most
  48 chats. The chat's last 64 screen lines are kept (Koko himself keeps his
  usual 256-token memory).
- **B while Koko reads a message** takes effect when the reading is done (each word not yet
  read while typing takes about a third of a second); then no reply is written.
- **SuperCard speed:** the cart runs at gbaSD's slower 4/2 timing (the emulator gives about
  3.0 words a second there, 3.4 on an Omega).
- **1.1 screens:** with the 16 px font the chat shows about 7 lines while typing (10
  before); START+Up / Down scroll back through the last 64. An article page takes about
  0.1 s to draw.

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
