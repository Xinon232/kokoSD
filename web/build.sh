#!/bin/sh
# Build web/site/koko.wasm from the cartridge engine. Needs clang and wasm-ld.
set -e
cd "$(dirname "$0")/.."
clang --target=wasm32 -O2 -nostdlib -ffreestanding -fno-builtin -Wall -Igen -Iengine \
  -Wl,--no-entry -Wl,--export=__heap_base -Wl,-z,stack-size=262144 \
  -o web/site/koko.wasm web/koko_web.c engine/koko.c engine/koko_tok.c engine/sums.c engine/facts.c engine/ui.c
if [ -f build/koko.bin ]; then cp build/koko.bin web/site/koko.bin; fi   # make web builds it first
ls -la web/site/koko.wasm web/site/koko.bin
