#!/bin/sh
# Model, tokenizer, font and evaluation text. Run from the project root.
set -e
mkdir -p dl/font dl/eval
HF=https://huggingface.co
curl -sSL -o dl/stories260K.bin $HF/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
curl -sSL -o dl/tok512.bin      $HF/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin
curl -sSL -o dl/font/font8x8_basic.h https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_basic.h
curl -sSL -r 0-399999 -o dl/eval/valid.txt $HF/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
