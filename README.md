# kokoSD: Koko the Chatbot with an SD card

**kokoSD** is made by **halimj** ([halimj.itch.io](https://halimj.itch.io)): a modified version of
**Koko the Chatbot** by **rdsilver** ([rdsilver.itch.io/koko-the-chatbot](https://rdsilver.itch.io/koko-the-chatbot)),
the language model that runs on an unmodified Game Boy Advance (72 million two-bit weights in a 32 MB cartridge,
about three words a second). All credit for the model, its training, the engine and Koko himself goes to rdsilver.
This repository is a fork of Koko; the original README follows further down.

Current version: **kokoSD 2.6.1**.

## How kokoSD differs from the original Koko
The model is Koko's, unchanged: same weights, same engine, same replies for the same words. What changes is
everything around it:

| | Original Koko | kokoSD |
|---|---|---|
| Runs on | any flash cart with 32 MB ROMs, emulators, the browser | SD-card flash carts: SuperCard SD / SuperChis (SuperFW), EZ-Flash Omega (and Definitive Edition from the SD card), EverDrive GBA, through [gbaSD](https://github.com/Xinon232/gbaSD). GBA only |
| Facts | 34,000 opening sentences from Simple English Wikipedia, in the ROM | 264,618 articles on the SD card (`koko/facts.dat`): the opening sentence as the answer, and **"tell me more"** opens the whole article to read |
| Asking for facts | "who is", "what is", "tell me about" | also "can you tell me about", "have you heard of", "do you know", "what does ... mean" and subjects with extra words ("the singer Taylor Swift", "the planet Mercury") |
| Typing | on-screen keyboard | the original's on-screen keyboard (default), or the button chords of gbawriter / gbamp3 (faster once learnt, with a cursor); accented letters |
| Chats | one chat, gone when switched off | saved chats on the SD card that carry on exactly where they were, scrolling, a menu |
| Memory | none | Koko remembers what you tell him about yourself ("my name is ...", "I live in ...") across chats |
| Settings | facts on / off | Facts (read out / as a hint / off), fact matching, memory, word memory, typing, kept on the card |
| Look | blue, Inter font | white, the look and fonts of halimj's other GBA apps (gbamp3, gbawriter, gbavocab) |
| Checks | | a self test (ROM, SD card and model) for real hardware |

## What kokoSD adds
- **Wikipedia on the SD card.** All of Simple English Wikipedia that Koko can say: 264,618 articles with their
  first sentence as Koko's answer (marked Fact) and the article itself, up to 64,000 characters, to read with
  "tell me more". The cartridge has no fact table of its own any more: the card's file is 7.7 times as many
  subjects, and without it Koko makes things up ("what is zircon?" -> "a town in the north of England").
- **Better matching** (2.4): 41 of 50 questions asked in other words find their article (19 before).
- **Facts as a hint** (2.4, optional): the fact is given to Koko and he answers in his own words.
- **Memory** (2.4): what you tell Koko about yourself, kept in `koko/memory.txt` and told to him at each new chat.
- **Word memory** (2.4, optional): which words follow which in Wikipedia, mixed into Koko's choice of words.
- **Two ways of typing** (2.5): the original keyboard, or button chords. SELECT (keyboard) or START+B (chords)
  opens the menu.
- **Saved chats, scrolling, a menu and Settings** with an explanation of each setting on the GBA.
- **Stability work for real carts** (2.0-2.6.1): the SD card is only mapped during card access, a keep-alive for
  the SuperCard, button timers that restart themselves, damaged saved chats refused, no stray memory writes.

## The default settings and why
The defaults were chosen from tests with the native engine (host/, the cartridge's own code on a PC: same model,
same seeds, so the replies are the GBA's) and in the emulator. Examples are real replies.

**Facts: Read out (on).** The Wikipedia sentence is always right; Koko on his own is often wrong. "As a hint"
sounds friendlier and usually keeps the main point, but still invents details and takes 10-25 s longer, so it is
offered, not the default.

| Question | Read out | As a hint | No fact (Off) |
|---|---|---|---|
| who was Julius Caesar | "Gaius Julius Caesar (12 July 100 BC - 15 March 44 BC) was a military commander, politician and author..." | "He was the ruler of the Roman Republic." | "Julius Caesar was a Roman emperor who was killed by Poseidon." |
| what is a penguin | "Penguins are seabirds in the family Spheniscidae..." | "A penguin is a bird that lives in the sea." | "...with a funny-like creature called a polar bear. They live in the Arctic..." |
| what is a tornado | "...fast spinning winds made by a strong rotating thunderstorm..." | "...like a big spinning spinning in the sky. It dances with the wind..." | "...a long, thin, round thing that can be very wild..." |
| who was Leonardo da Vinci | "...an Italian polymath who lived during the Renaissance." | "...a very important person in the Roman Catholic Church. He was born in Rome in 1530." (wrong) | "He was a politician." |

**Fact matching: Wide (on).** The clearest gain and no cost: 41 of 50 questions asked in other words found their
article (Strict, 2.3's matching: 19), none of them the wrong one. Without a match Koko makes things up:

| Question | Strict (2.3) | Wide (2.4) |
|---|---|---|
| have you heard of the beatles | "Yes, I can imagine those! It's so exciting to hear them." | "The Beatles were an English pop/rock band, started in Liverpool, England in 1960." |
| who was the scientist marie curie | "He was a scientist." | "...was a Polish-born French physicist and chemist." |
| what is the planet venus | "...a big red star in the sky... about 50 light years away" | "Venus is the second planet from the Sun..." |
| i want to know about the titanic | "A titanicCristan is a common name, like a sponge." | "The RMS Titanic was a British steam ship..." |
| do you know pikachu | "Pikachu is super cool. It's like a giant playground!" | "Pikachu is a yellow mouse Pokemon... created by Satoshi Tajiri." |

**Memory: On.** Your name is used reliably and makes Koko feel personal; the rest is hit and miss, and a miss does
no harm. It is told to Koko while you type the first message, so normally there is no wait. Told "My name is Lena.
I am 10 years old. I live in Vienna. My favorite food is pancakes. I like horses. I have a cat named Mimi.":

| You say | With memory | Without |
|---|---|---|
| hi koko! | "Hi Lena! What's your favorite food?" | "Hello there! How can I help you today?" |
| what is my name? | "Your name is Lena." (every time) | "Your name is Alex, Alex." |
| what is my cat called? | "It's called Mimi. She's a big cat with a big, fluffy tail." (one seed of two) | "His cat is called the cat." |
| how old am i? | "You are a puppy, Lena." (age is not used) | "You're a good person, I guess." |

**Word memory: Off.** It changes about half of the replies, but makes them sound like an encyclopedia rather than
more often right, and on a slow card it may slow Koko down. Try it if you like:

| Question | Off | On |
|---|---|---|
| who was the first president of america? | "The first president was John F. Kennedy." | "The first president was George Washington." |
| how do plants grow? | "...They use their wings to make their own food..." | "Plants grow from tiny plants with little seeds inside them." |
| what do you know about germany? | "Germany was a country in Europe." | "Germany was founded in the 17th century, but it was called Einsatzgruppen." |
| what is the biggest animal? | "The biggest animal is the lions." | (the same) |

**Typing: Keyboard (kokoSD 2.5).** The original Koko's on-screen keyboard needs nothing learnt: you see every key.
The button chords (gbawriter's) are faster once learnt and have a cursor, so they stay one setting away. With the
keyboard, SELECT opens the menu.

Settings are changed in the menu (SELECT, or START+B with the button chords) > Settings and kept in
`koko/settings.txt`; the last row there, **What these settings do**, explains them on the GBA.

## Install
From the [Releases](../../releases) page:
1. `kokoSD.gba`: copy it anywhere on the SD card and start it from the flash cart's menu.
2. `koko.zip` (Simple English Wikipedia for Koko's facts and word memory, CC BY-SA 4.0): unzip it at the top of the
   card, so the file is `/koko/facts.dat` (323 MB; 2.4 needs this file for word memory, older files still work). Without it Koko still chats, but cannot look facts up or show articles.

Works on SuperCard SD / SuperChis with SuperFW, EZ-Flash Omega (and Definitive Edition started from the SD card)
and EverDrive GBA (exFAT and FAT32 cards). Saved chats go to `/koko/chats/`. Buttons: SELECT (keyboard) or START+B (button chords) > How to type.

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
kokoSD's additions: halimj (Halim Jarrar, https://halimj.itch.io), with gbaSD, gbawriter's typing engine, gbamp3's text drawing and
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
