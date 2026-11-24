# Koko for the browser

The cartridge engine (engine/koko.c, koko_tok.c, sums.c) compiled to WebAssembly, the cartridge's data file,
and a chat page. Nothing is sent anywhere: the model runs in the visitor's browser.

    make web                  # site/koko.wasm (68 KB); refreshes site/koko.bin (the 33 MB cartridge data) from build/
    node web/test_node.mjs    # same prompt + seed -> same tokens as the native cartridge engine
    node web/test_browser.cjs # the real page in headless Chromium: typing, suggestions, edits, a sum, GBA speed

## Uploading to itch.io
- Zip the contents of web/site/ (koko-web.zip) and upload it, kind of project "HTML", tick "This file will be played in the browser".
- index.html is at the top of the zip. Suggested embed size 800 x 720, with "Mobile friendly" and the
  fullscreen button on. The page fills whatever size it is given.
- First load downloads 33 MB; a progress bar shows it.

## Page features
Type on a real keyboard, Enter sends. Suggestions appear as chips (Tab accepts the first, click any).
"GBA speed" paces replies at the cartridge's 3.4 tokens per second; unticked, it runs as fast as the computer
allows. "New chat" starts over. Adding ?seed=7 to the address makes replies repeatable.
Same rules as the cartridge: temperature 0.5, top 40, repeated-phrase rule, sums computed by the game,
120 tokens per reply, rolling 256-token memory. Messages are limited to 160 plain-ASCII characters.
