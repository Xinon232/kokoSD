# Koko the Chatbot: credits and licences

Koko's code, model, drawing of Koko and screen design were made for this project.

## Licence
- **Code:** MIT License (see LICENSE). Copyright (c) 2026 The Koko authors.
- **Model weights, fact table and the drawing of Koko:** Creative Commons Attribution-ShareAlike 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). See LICENSE-MODEL.

## Text included in the download
- **Fact table:** the opening sentences of about 34,000 articles from **Simple English Wikipedia**
  (https://simple.wikipedia.org), by its contributors, licensed CC BY-SA (https://creativecommons.org/licenses/by-sa/4.0/).
  Sentences were shortened and cleaned (pronunciation guides and non-English script removed). Replies marked
  with a small book come from this table. The fact table stays under CC BY-SA.
- **Font:** Inter by The Inter Project Authors (https://github.com/rsms/inter), SIL Open Font License 1.1,
  drawn here as small bitmaps.

## What the model was trained on
- **TinyStories** (Ronen Eldan and Yuanzhi Li), CDLA-Sharing-1.0. https://huggingface.co/datasets/roneneldan/TinyStories
- **SODA** (Allen Institute for AI), CC BY 4.0. https://huggingface.co/datasets/allenai/soda
- **TinyDialogues** (Steven Y. Feng and others), MIT. https://huggingface.co/datasets/styfeng/TinyDialogues
- **Everyday Conversations** (Hugging Face), Apache 2.0. https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k
- **Simple English Wikipedia** (snapshot of 1 November 2023 from https://huggingface.co/datasets/wikimedia/wikipedia), CC BY-SA and GFDL.
- **Country facts** from mledoze/countries (https://github.com/mledoze/countries), Open Database License; used to write question-and-answer drills.
- **Koko's own conversations:** about 1.3 million short chats written by the Qwen2.5-7B-Instruct model
  (Alibaba Cloud, Apache 2.0) from prompts written for this project, then filtered. They are machine-written.

## Tools
- Tested in the mGBA emulator (https://mgba.io). No emulator is included.
- Built with the GNU Arm toolchain and clang; the browser version is the same C code compiled to WebAssembly.
- The code was written with an AI coding assistant (Claude).

## Not affiliated
Game Boy Advance is a trademark of Nintendo. This is an unofficial hobby project and is not made or endorsed by
Nintendo. No Nintendo code or data is included; the cartridge header's logo field is left blank.
