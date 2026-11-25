# kokoSD 2.3 / 2.4 (test): known issues and open points

kokoSD 2.3 passes its emulator checks (engine/test_game.py: 47 checks on the SuperCard SD model, 46 on the
EZ-Flash Omega and EverDrive GBA models, 32 without a card) and the random button sessions of
tools/crash_test.py. On real hardware it runs on a SuperChis Prime with SuperFW (stable since 2.1: chats, facts,
tell me more, saved chats); the other carts are not tested on hardware yet. These are known and not## kokoSD 2.4 (test)
- **Word memory on a real card:** each word Koko writes asks the card for 1-2 sectors of the word table. The
  emulator's card models show no slowdown, but they leave out the card's own delay: on a slow card Koko may write
  more slowly with Word memory on (Settings can turn it off).
- **Facts as a hint** can still be wrong (the model invents names and dates around the fact) and takes 10-25 s
  longer, so Read out stays the default.
- **Memory** understands only a few sentence shapes (my name is, call me, I am N years old, I live in, I'm from,
  I like / love / enjoy, my favorite X is, I have a). "I like it" and the like are skipped; "I am happy" is not
  taken as a name. Eight facts at most (the oldest of a kind goes first). Koko sees about 256 words back, so in
  long chats what he was told at the start fades.
- **The menu redraws slowly** (Settings and the 2.4 rows): a press made while a page or the menu is being drawn
  can be lost.
- Not yet tested on hardware.

 fixed yet:

## Not verified yet
- **Real hardware:** only a SuperChis Prime (SuperFW) so far; EZ-Flash Omega and
  EverDrive GBA are untested. Koko's speed on the SuperChis is not measured yet.
- **Why the button timers stopped on the SuperChis** (fixed in 2.1 by starting them again) is not known; the self
  test log (`koko/selftest.txt`) counts how often it happens.
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
- **Saved chats:** the list shows at most 48 chats. The chat's last 64 screen lines are kept (Koko himself keeps his
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
- "Tell me more" shows up to 64,000 characters of an article (19 articles are longer: cut at a sentence end,
  "..."); a page shows 8 lines, an article at most 4,095 lines. A long article takes about 2 s to open.
- Wikipedia's markup is converted automatically: removed tables, templates and
  references can leave odd phrasing in places.
- Without `koko/facts.dat` there are no facts at all (since 2.0 the cartridge has no table of its own).

## Not done yet
- Fuzzy title matching ("who is the eminem rapper") and answers for long opening sentences.
