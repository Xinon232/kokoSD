# Koko the Chatbot

A chatbot that runs entirely on a Game Boy Advance, from a 32 MB cartridge. You type on an on-screen keyboard
and Koko, a friendly husky, writes back at about three words a second. Nothing is sent anywhere and there is
no extra chip: the language model is the game.

## Playing
- **In a browser:** the web version on the project page. Type on your keyboard.
- **On an emulator:** open `koko.gba` in mGBA (https://mgba.io) or another Game Boy Advance emulator.
- **On a console:** copy `koko.gba` to a flash cartridge that takes 32 MB ROMs and start it from the cartridge's
  menu. It has run on a real Game Boy Advance this way. The ROM's logo field is blank, which emulators and flash
  cartridges accept; a console booting it directly would not.

## Controls (Game Boy Advance)
| Button | What it does |
|---|---|
| START | Start; send your message; on an empty line, offer a ready-made message |
| D-pad | Move around the keyboard |
| A | Type the highlighted key |
| B | Delete a letter; stop Koko while it is writing |
| SELECT | Capital letters on and off |
| L / R | Accept the first / second suggested word |

Four keys on the on-screen keyboard are special: the bar is a space, the arrow deletes, the page with a plus
starts a new chat, and the book turns the fact table and calculator off and on.

In the browser: Enter is START, Tab and Shift+Tab are L and R, Esc is B, and you can click the keys.

## What it can do
- Chat, tell short stories and jokes, and answer simple questions.
- Suggest your next word as you type.
- Remember roughly the last 190 words of the conversation.
- Answer "who was X" and "what is X" from a built-in table of about 34,000 Simple English Wikipedia opening
  sentences, and work out sums. Those replies are marked with a small book or an equals box; everything else
  is written by the model. The book key turns both off so you can see the model alone.

## What it cannot do
Koko is tiny. Without the fact table it is often wrong about facts, and it loses the thread of long
conversations. It only understands plain English letters, digits and basic punctuation.

## Numbers
- 72 million two-bit weights plus lookup tables: 32 MB including the program, font, tokenizer and fact table.
- About 3.5 tokens per second on short chats and 2.9 once its memory is full, measured in the mGBA emulator.
- Trained in 7.4 hours on one RTX 5090 graphics card.
- Every reply in the emulator matches a reference implementation on a PC, number for number.

## Licence
Open source. The code is under the MIT License; the model weights, fact table and artwork are under
Creative Commons Attribution-ShareAlike 4.0. See LICENSE, LICENSE-MODEL and LICENSE-OFL in the repository root
(LICENSE.txt, LICENSE-MODEL.txt and LICENSE-OFL.txt in the browser download) and CREDITS.md in this folder.
