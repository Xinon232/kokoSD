# Koko the Chatbot. Run make from the repository root.
#
#   make              build/koko.gba (the cartridge) and the native engine programs
#   make test         the cartridge in mGBA against the native engine, and the WebAssembly build against it
#   make web          web/site/koko.wasm (the browser version)
#   make stage1       the earlier stories-only ROM (needs tools/fetch.sh first)
#
# The cartridge data (weights, tokenizer, fact table) ships as web/site/koko.bin; engine/export.py rebuilds it
# from a trained model, see README.md.

CROSS  = arm-none-eabi-
HOSTCC = gcc
ENGINE = engine/koko.c engine/koko_tok.c engine/sums.c engine/facts.c
ENGINE_DEPS = $(ENGINE) engine/koko.h engine/ui.h gen/koko_layout.h

$(shell mkdir -p build out)

all: build/koko.gba build/koko_gen build/koko_twin

# ---- Koko ----
build/koko.bin: web/site/koko.bin
	cp $< $@
build/koko.gba: build/koko.bin $(ENGINE_DEPS) engine/game.c engine/ui.c engine/crt0.s engine/kernels.s engine/blob.s \
                bench/tern.s src/gba.ld gen/font_ui.h gen/husky.h tools/fixrom.py
	python3 engine/build_game.py

# native builds of the cartridge engine (the reference the ROM and the browser version are tested against)
build/koko_gen build/koko_twin build/tok_test build/facts_test build/suggest_test: build/%: host/%.c $(ENGINE_DEPS)
	$(HOSTCC) -O2 -Wall -Wextra -Igen -Iengine -o $@ $< $(ENGINE)
build/sums_test: host/sums_test.c engine/sums.c
	$(HOSTCC) -O2 -Wall -Wextra -Igen -Iengine -o $@ $^
# headless mGBA driver for the emulator tests (needs libmgba)
build/runner: test/runner.c
	$(HOSTCC) -O2 -Wall -o $@ $< -lmgba

web: build/koko.bin
	sh web/build.sh

test: build/koko.gba build/koko_gen build/runner
	python3 engine/test_game.py
	node web/test_node.mjs

# ---- Stage 1: TinyStories-260K on the cartridge (src/, tools/export.py; downloads via tools/fetch.sh) ----
CFLAGS = -mcpu=arm7tdmi -mtune=arm7tdmi -marm -O2 -ffreestanding -fno-builtin \
         -Wall -Wextra -Igen -Isrc
KERNEL_OBJ = build/kernels.o
KERNEL_DEF = -DLLM_ASM_KERNELS
OBJS   = build/crt0.o build/main.o build/model.o build/model_data.o $(KERNEL_OBJ)

stage1: build/tinystories.gba build/twin build/runner

gen/model_data.c gen/model_data.h: tools/export.py tools/ref.py
	python3 tools/export.py
gen/font.h: tools/font.py
	python3 tools/font.py

build/crt0.o: src/crt0.s
	$(CROSS)gcc $(CFLAGS) -c $< -o $@
build/%.o: src/%.c gen/model_data.h gen/font.h src/model.h
	$(CROSS)gcc $(CFLAGS) $(KERNEL_DEF) -c $< -o $@
build/%.o: src/%.s
	$(CROSS)gcc $(CFLAGS) -c $< -o $@
build/model_data.o: gen/model_data.c
	$(CROSS)gcc $(CFLAGS) -c $< -o $@

build/tinystories.elf: $(OBJS) src/gba.ld
	$(CROSS)gcc $(CFLAGS) -nostdlib -Wl,-T,src/gba.ld -Wl,-Map,build/tinystories.map \
	    -o $@ $(OBJS) -lgcc
build/tinystories.gba: build/tinystories.elf tools/fixrom.py
	$(CROSS)objcopy -O binary $< $@
	python3 tools/fixrom.py $@

# the same inference code, native: the reference the ROM is tested against
build/twin: host/twin.c src/model.c gen/model_data.c src/model.h
	$(HOSTCC) -O2 -Wall -Wextra -Igen -o $@ host/twin.c src/model.c gen/model_data.c

stage1-test: stage1
	python3 test/test_rom.py
score: build/twin
	python3 tools/score.py 40

clean:
	rm -rf build/* out/* gen/model_data.c gen/model_data.h

.PHONY: all web test stage1 stage1-test score clean
