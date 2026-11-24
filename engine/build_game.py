"""Build build/koko.gba (the game)."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import subprocess, os
os.chdir(ROOT)
os.makedirs('build', exist_ok=True)
if not os.path.exists('build/koko.bin'):       # fresh clone: use the shipped cartridge data (weights, tokenizer, fact table)
    import shutil; shutil.copy('web/site/koko.bin', 'build/koko.bin')
CF = '-mcpu=arm7tdmi -mtune=arm7tdmi -marm -O2 -ffreestanding -fno-builtin -Wall -Igen -Iengine'.split()
def sh(*a): subprocess.run(a, check=True)
sh('arm-none-eabi-gcc', *CF, '-nostdlib', '-Wl,-T,src/gba.ld', '-Wl,-Map,build/koko.map', '-o', 'build/koko.elf',
   'engine/crt0.s', 'engine/game.c', 'engine/ui.c', 'engine/koko.c', 'engine/koko_tok.c', 'engine/sums.c', 'engine/facts.c', 'bench/tern.s', 'engine/kernels.s', 'engine/blob.s', '-lgcc')
sh('arm-none-eabi-objcopy', '-O', 'binary', 'build/koko.elf', 'build/koko.gba')
sh('python3', 'tools/fixrom.py', 'build/koko.gba')
sh('arm-none-eabi-size', 'build/koko.elf')
