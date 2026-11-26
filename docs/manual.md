# kokoSD 2.5

## Instruction manual

kokoSD is Koko the Chatbot on your Game Boy Advance, with an SD card. Koko is a
small language model (72 million two-bit weights) by rdsilver that runs on the
GBA itself and writes about three words a second. kokoSD adds facts and whole
articles from Simple English Wikipedia on the SD card, the original on-screen
keyboard or typing with button chords,
saved chats, a memory of what you tell Koko about yourself, and settings for all
of it. It runs on SuperCard SD / SuperChis (SuperFW), EZ-Flash Omega (and
Definitive Edition started from the SD card) and EverDrive GBA.

## Quickstart

For everyone who wants to start chatting right away, here are the basics:

1. **Copy** `kokoSD.gba` anywhere on the SD card, and unzip `koko.zip` at the top
   of the card, so that the file is `/koko/facts.dat`.
2. **Start** `kokoSD.gba` from your flash cart's menu and press START at the title.
3. **Type** on the on-screen keyboard: the D-pad moves, A types the key, B
   deletes, L / R take the suggested words.
4. **Send** with START. Koko answers; B stops him. Press any button to type again.
5. **Ask for facts**: "who was Mozart?", "what is a volcano?", "tell me about
   Japan". Then say "tell me more" to read the whole article.
6. **SELECT** opens the menu: New chat, Saved chats, Settings, How to type,
   About, Self test. (Settings > Typing switches to the button chords of
   gbawriter and the other apps; then START + SELECT opens the menu.)

It is highly recommended to read the rest of this manual: it explains the
typing, the facts, memory, the settings and saved chats.

```screens
title.png The title screen
keyboard.png Typing on the keyboard
fact.png A fact, and "tell me more"
```

## The SD card

- **The ROM**: copy `kokoSD.gba` anywhere on the card and start it from the cart's
  menu (on an EZ-Flash Omega start it from the SD card, not from NOR).
- **The fact file**: unzip `koko.zip` at the top of the card. It makes
  `/koko/facts.dat` (323 MB): the first sentence and the text of 264,618 Simple
  English Wikipedia articles, and the word memory. Without it Koko still chats,
  but cannot look facts up or show articles.
- **Files kokoSD writes**: `/koko/chats/` (saved chats), `/koko/memory.txt` (what
  Koko remembers about you), `/koko/settings.txt` (the settings) and
  `/koko/selftest.txt` (the self test's log).
- Cards formatted FAT32 or exFAT both work.

## Typing on the keyboard

The on-screen keyboard is the original Koko's (13 keys by 4), and the default
way of typing (Settings > Typing: Keyboard).

- **D-pad**: move on the keyboard (hold to repeat). **A**: type the key.
  **B**: delete (hold to repeat).
- **The arrow key** (right end of the digits): capitals on / off.
- **The é key** (before the delete key): an accent on the last letter; again:
  the next accent (e: é è ë ê). Koko reads accents as plain letters.
- **L / R**: the first or second suggested word (shown above the keyboard).
- **START**: send. On an empty line, START offers a ready-made message; START
  again sends it (delete it and press START for the next one).
- **SELECT**: the menu.
- **Hold START** and press **A** to save the chat, **Up / Down** to scroll it.
- The keyboard always types at the end of the message.

```screens
keyboard.png The keyboard (suggestions above it)
typing.png Typing with button chords
accents.png Accents are shown as typed
```

## Typing with button chords

With Settings > Typing: Button chords, kokoSD types with button chords, the same
as gbawriter, gbamp3 and the other apps.
A direction chooses a group of letters; B, A or R picks the first, second or third
letter of it. Hold L as well for the second layer.

| Direction | B / A / R | With L held: B / A / R |
|---|---|---|
| Up | a b c | d e f |
| Right | h i j | k l m |
| Down | n o p | q r s |
| Left | t u w | x y z |

- **g and v**: keep Right or Left held and press R twice (j becomes g, w becomes v).
- **A alone**: a space. **B alone**: delete (hold to repeat).
- **R alone**: a capital for the next letter. Hold R: capitals on; R again: off.
- **SELECT alone** types a dot. Keep SELECT held and press Up / Down for digits,
  R / L for , ' " : ! ? and Right / Left for ( ) / ; @ # % & _ + = -.
- **Accents**: keep the letter's buttons held and press SELECT; SELECT again gives
  the next one (e: é è ë ê). They are shown as typed; Koko reads them as plain
  letters.
- **Suggestions**: Koko reads your words while you type and suggests the next word
  at the bottom. START + L takes the first, START + R the second.
- **START + Left / Right** move the cursor in the message.
- **START + SELECT** opens the menu; **START + A** saves the chat.

## Chatting

- **START** sends the message. With nothing typed, or nothing Koko can read
  (only symbols), START does nothing.
- While Koko writes, **B** stops him.
- When he has finished, the reply stays until you press a button; that button
  already types.
- Hold **START and press Up / Down** to scroll back through the chat.
- **Sums** are worked out by the game, not guessed: "what is 17 + 25?" gives
  "17 + 25 = 42." (marked Sum).
- The header shows what Koko is doing (reading, looking up, B: stop) and, after a
  reply, his speed in words (tokens) a second.

```screens
sum.png A sum, worked out by the game
scroll.png START + Up: older lines
fact.png A fact, and "tell me more"
```

## Facts and Tell me more

Ask "who is ...", "who was ...", "what is ...", "tell me about ..." and Koko answers
with the first sentence of the Simple English Wikipedia article, marked **Fact**.
With the default *Fact matching: Wide* many other ways of asking work too:
"can you tell me about ...", "have you heard of ...", "do you know ...", "what
does ... mean", "info about ...", "do you know anything about ...", and subjects
with extra words ("the singer Taylor Swift", "the planet Mercury").

- **"tell me more"** (also "more", "go on") right after a fact opens the whole
  article to read. "tell me more about Japan" opens Japan's article directly.
- In the article: **Down / Up** scroll by three lines, **Right / Left / R / L**
  turn a page, **B** (or A, START) goes back to the chat.
- Articles are kept whole up to 64,000 characters, which is nearly every article.

```screens
article.png Reading an article
match.png Asked in other words (Wide matching)
hint.png Facts: As a hint
```

## The menu

SELECT while typing on the keyboard (START + SELECT with the button chords) opens
the menu. Up / Down choose, A opens, B goes back.

| Row | What it does |
|---|---|
| New chat | Koko forgets this chat and starts again (what he remembers about you stays). |
| Saved chats | Your saved chats, listed by their first message. |
| Settings | Facts, Fact matching, Memory, Word memory, Typing, What Koko remembers, What these settings do. |
| How to type | The keyboard's or the chords' buttons (whichever is chosen), on the GBA. |
| About kokoSD | Credits and licences. |
| Self test | Checks the cartridge, the SD card and the model; writes `koko/selftest.txt`. |

```screens
menu.png The menu
settings.png Settings (the defaults)
help.png What these settings do
```

## Settings

Each setting is changed with A on its row and kept in `/koko/settings.txt`, so it
stays after switching off. The last row, *What these settings do*, explains them
on the GBA.

| Setting | Choices (default first) | What it does |
|---|---|---|
| Facts | Read out / As a hint / Off | Read out: the Wikipedia sentence is Koko's reply (marked Fact). As a hint: the sentence is put into your message and Koko answers in his own words (marked Hint). Off: no facts and no sums. |
| Fact matching | Wide / Strict | Wide finds the article for many ways of asking; Strict only for the plain ones (kokoSD 2.3). |
| Memory | On / Off | Koko notes what you tell him about yourself and is told it at the start of each new chat. |
| Word memory | Off / On | Which words follow which in the Wikipedia text is mixed into Koko's choice of the next word. |
| Typing | Keyboard / Button chords | The original Koko's on-screen keyboard, or gbawriter's button chords. |

### Why these defaults

- **Facts: Read out**, because the Wikipedia sentence is always right. As a hint,
  Koko keeps the main point in his own friendly words ("He was the ruler of the
  Roman Republic."), but still invents details (Leonardo da Vinci "born in Rome
  in 1530") and needs 10-25 seconds longer to read the fact first.
- **Fact matching: Wide**, because it finds far more articles at no cost: in a
  test 41 of 50 questions asked in other words found their article, against 19
  with Strict, and none found a wrong one.
- **Memory: On**, because Koko uses your name reliably ("Hi Lena!"); other facts
  (town, age, favourites) he uses only sometimes.
- **Word memory: Off**, because it makes Koko sound more like an encyclopedia
  without being more often right, and on a slow card it may slow him down.
- **Typing: Keyboard**, the original Koko's way: nothing to learn. The button
  chords are faster once learnt.

## Memory

Memory works across chats. After each message, Koko notes sentences such as "my
name is ...", "call me ...", "I am 10 years old", "I live in ..." or "I'm from
...", "I like / love / enjoy ...", "my favorite ... is ..." and "I have a ...".
At each new chat (and the first chat after switching on) he is told them while
you type your first message, so he can greet you by name.

- Koko keeps up to eight facts: one name, age and town, two likes, two favourites
  and one "I have". A newer one replaces the oldest of the same kind.
- **Settings > What Koko remembers** shows the list; **SELECT** there forgets it
  all.
- The facts are kept in `/koko/memory.txt`, one sentence a line, so you can read
  and change them on a computer.
- Saved chats go on without it: they already have their own history.
- Koko sees about the last 256 words, so in a long chat what he was told at the
  start fades.

```screens
memory_page.png What Koko remembers
memory_chat.png "what is my name?"
saved.png Saved chats
```

## Saved chats

- **START + A** (hold START, press A) saves the chat on the SD card (in
  `/koko/chats/`).
- **Menu > Saved chats** lists them by their first message; A opens one and Koko
  carries on exactly where you left off.
- **START** on a chat in the list: Open, Rename or Delete. Renaming types the new
  name in the field; START + A saves it, START + B cancels.
- The list shows up to 48 chats.

## Self test

Menu > Self test checks the cartridge's ROM, looks up 24 facts on the SD card,
feeds a fixed sentence through Koko and compares the result with the PC's. It
also checks that the cart keeps its data after 90 seconds unread (B skips that
step). The result is shown and written to `/koko/selftest.txt`. If Koko ever
freezes or misbehaves, run it and keep the log.

## About / licence

Koko the Chatbot (the model, its training, the engine and Koko himself) is by
rdsilver: rdsilver.itch.io/koko-the-chatbot. kokoSD (SD card, typing, facts,
saved chats, memory, settings, this look) is by Halim Jarrar (halimj).

- Code: Koko (MIT); gbaSD and gbawriter typing by Halim Jarrar
  (GPL-3.0-or-later); the cartridge as a whole is GPL-3.0-or-later.
- Weights, the fact file and the artwork: CC BY-SA 4.0. The facts and articles
  are from Simple English Wikipedia (CC BY-SA 4.0).
- Fonts: the SuperFW font by David Guillen Fandos (GPL-3.0-or-later; UNSCII and
  GNU Unifont glyphs) and gbamp3's 5x7 font.
- Source: github.com/Xinon232/kokoSD
