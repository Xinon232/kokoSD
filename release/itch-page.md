# itch.io page text (copy and adjust)

**Title:** Koko the Chatbot
**Project URL:** koko-the-chatbot

**Short description / tagline:** A tiny language model that runs on a Game Boy Advance. No internet, no extra chip.

**Classification:** Tool or Game (your choice); kind of project: HTML for the browser version, with the ROM as a download.
**Suggested tags:** game-boy-advance, gba, chatbot, homebrew, language-model, retro, experimental

**Embed settings for the browser version:** upload `koko-web.zip`, tick "This file will be played in the browser",
viewport 800 x 720, enable the fullscreen button and "Mobile friendly". First load is 32 MB.

---

## Description

**Koko is a chatbot that fits on a Game Boy Advance cartridge.**

Type a message on the on-screen keyboard and Koko, a friendly husky, writes back, one word at a time, at about
three words a second. There is no internet connection and no helper chip in the cartridge. The language model
itself is the game: 72 million weights squeezed to two bits each, running on a 16.78 MHz processor from 2001.

**Play it here in your browser**, or download the ROM for an emulator or flash cartridge. The browser version
draws the very same 240 x 160 screen with the same code and the same model; only the keyboard is real.

### What Koko does
- Chats, tells short stories and jokes, and answers simple questions
- Suggests your next word while you type
- Answers "who was...?" and "what is...?" from a built-in table of 34,000 Wikipedia opening sentences
- Does sums exactly

### Honest notes
- Koko is tiny and often wrong. Replies marked with a small book come from the fact table, and those with an
  equals box are calculated; everything else is the model. Press the book key to turn the helpers off and see
  what the model does alone.
- It understands plain English letters, digits and basic punctuation, and remembers about the last 190 words.
- It has run on a real Game Boy Advance from a flash cartridge, and is tested in the mGBA emulator.

### Controls
START sends (or offers a ready-made message on an empty line). D-pad and A type. B deletes, or stops Koko.
SELECT switches capitals. L and R accept suggested words. In the browser: type normally, Enter sends,
Tab accepts a suggestion, Esc stops.

### How it compares
On the same tests as small public models (GPT-2, SmolLM2-135M, TinyStories): AI judges preferred Koko's chat
replies to SmolLM2-135M-Instruct's on 18 of 20 messages, and its stories to GPT-2's on 19 of 20. On general
knowledge it loses clearly: 22 of 60 questions against SmolLM2's 43. The other models are 14 to 513 MB;
Koko is 32 MB with everything included. Judges were AI, not people, and the prompts were the kind Koko was
trained for.

### Open source
The code is MIT licensed, and the model weights, fact table and artwork are CC BY-SA 4.0. The full source,
including the training code and every test, is in the source download.

### Credits
Fact table text from Simple English Wikipedia (CC BY-SA). Font: Inter (SIL Open Font License). Trained on
TinyStories, SODA, TinyDialogues, Everyday Conversations, Simple English Wikipedia and conversations written
by Qwen2.5-7B-Instruct. Full credits are in the download. Game Boy Advance is a trademark of Nintendo; this is
an unofficial hobby project.
