CROSS  = arm-none-eabi-
CFLAGS = -mcpu=arm7tdmi -mtune=arm7tdmi -marm -O2 -ffreestanding -fno-builtin \
         -Wall -Wextra -Igen -Isrc
KERNEL_OBJ = build/kernels.o
KERNEL_DEF = -DLLM_ASM_KERNELS
OBJS   = build/crt0.o build/main.o build/model.o build/model_data.o $(KERNEL_OBJ)

all: build/tinystories.gba build/twin build/runner

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
	gcc -O2 -Wall -Wextra -Igen -o $@ host/twin.c src/model.c gen/model_data.c
build/runner: test/runner.c
	gcc -O2 -Wall -o $@ $< -lmgba

test: all
	python3 test/test_rom.py
score: build/twin
	python3 tools/score.py 40
clean:
	rm -rf build/* gen/*

.PHONY: all test score clean
