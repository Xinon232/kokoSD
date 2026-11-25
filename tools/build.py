"""Build kokoSD: build/koko.gba (the game) or, with --cardtest,
build/koko-cardtest.gba (the card test). The game needs build/koko.bin
(python3 tools/blob_from_rom.py koko.gba). GPL-3.0-or-later.

gbaSD's code that runs while the EZ-Flash Omega maps its kernel over the ROM
goes to slow RAM (EWRAM): fast RAM is nearly full with Koko's kernels. Its
objects are named sd-*.o so engine/koko.ld keeps their variables there too.
gbaSD's tests/check_ram_code.py proves that code never reaches ROM."""
import argparse, os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
G = R/'gbasd'
DKA = Path(os.environ.get('DEVKITARM', '/opt/devkitpro/devkitARM'))
P = str(DKA/'bin'/'arm-none-eabi-')
KOKO = ['-mcpu=arm7tdmi', '-mtune=arm7tdmi', '-marm', '-O2', '-ffreestanding', '-fno-builtin', '-Wall',
        f'-I{R/"gen"}', f'-I{R/"engine"}', f'-I{G/"include"}', f'-I{G/"fatfs"}']
SD = ['-mcpu=arm7tdmi', '-mthumb-interwork', '-mlong-calls', '-O2', '-std=gnu11', '-ffreestanding',
      '-ffunction-sections', '-fdata-sections', '-fno-strict-aliasing', '-Wall', '-Wno-unused-parameter',
      f'-I{R/"engine"}', f'-I{G/"include"}', f'-I{G/"fatfs"}']
GBASD = ['-DGBASD_SC_FAST_FN=__attribute__((section(".ewram.sc"),target("arm"),noinline,noclone))',
         '-DGBASD_SC_CRC7_FN=__attribute__((section(".ewram.sc"),target("arm"),noinline,noclone))',
         '-DGBASD_SC_CRC16_FN=__attribute__((section(".ewram.sc"),target("arm"),noinline))',
         '-DGBASD_SC_WRITE_IN_EWRAM', '-DEZFO_SECTION=".ewram.ezfo"']
SD_ARM = {'crc', 'supercard_driver', 'sc_driver', 'supercard_env', 'ezfo_driver'}


def sh(*a):
    r = subprocess.run([str(x) for x in a], capture_output=True, text=True)
    if r.returncode: sys.exit(f'FAILED: {" ".join(map(str, a))}\n{r.stdout}{r.stderr}')
    return r.stdout


def build(cardtest=False):
    B = R/'build'; B.mkdir(exist_ok=True)
    out = 'koko-cardtest' if cardtest else 'koko'
    objs = []
    def obj(src, flags, name=None):
        o = B/f'{name or src.stem}{"-ct" if cardtest else ""}.o'
        f = list(flags) + (['-x', 'assembler-with-cpp'] if src.suffix in ('.s', '.S') else [])
        sh(P + 'gcc', *f, '-c', src, '-o', o); objs.append(o)
    koko = ['crt0.s', 'ui.c', 'rt.c'] + (['cardtest.c'] if cardtest else
            ['game.c', 'koko.c', 'koko_tok.c', 'sums.c', 'facts.c', 'kernels.s', 'blob.s'])
    for n in koko: obj(R/'engine'/n, KOKO)
    T = R/'engine'/'text'                      # kokoSD 1.1: our apps' two fonts (gbamp3's text drawing; its data in slow RAM)
    obj(T/'text.c', KOKO + [f'-I{T}', '-Wno-unused-function'], 'ew-text')
    obj(T/'blit.c', KOKO, 'ew-blit')
    obj(T/'fonts.S', [f'-Wa,-I{T}'], 'fonts')
    if not cardtest: obj(R/'bench'/'tern.s', KOKO)
    for n in ('sd.c', 'diskio.c') + (() if cardtest else ('facts_sd.c', 'chats.c')): obj(R/'engine'/n, SD + ['-marm', f'-I{R/"gen"}'], f'sd-{Path(n).stem}')
    for s in [G/'fatfs'/'ff.c', G/'fatfs'/'ffunicode.c'] + [R/'engine'/'sc_driver.c'] + [G/'src'/n for n in (   # kokoSD 1.4.3: its own SuperCard driver copy
            'cart_io.c', 'crc.c', 'supercard_env.c', 'supercard_io.s',
            'supercard_write.s', 'ezfo_driver.c', 'edgba_driver.c')]:
        isa = '-marm' if s.stem in SD_ARM or s.suffix == '.s' else '-mthumb'
        extra = GBASD if s.parent == G/'src' or s.stem == 'sc_driver' else ['-Wno-sign-compare', '-Wno-implicit-fallthrough']
        obj(s, SD + [isa] + extra, f'sd-{s.stem}')
    if not cardtest:                           # gbawriter's typing engine (C++; its memory in slow RAM: ew-*.o)
        for n in ('writer_core.cpp', 'writer_glue.cpp'):
            o = B/f'ew-{Path(n).stem}.o'
            sh(P + 'g++', '-mcpu=arm7tdmi', '-mthumb', '-mthumb-interwork', '-O2', '-std=gnu++17', '-fno-exceptions', '-fno-rtti',
               '-fno-threadsafe-statics', '-fno-use-cxa-atexit', '-ffreestanding', '-Wall', '-c', R/'engine/writer'/n, '-o', o)
            objs.append(o)
    elf, rom = B/f'{out}.elf', B/f'{out}.gba'
    sh(P + 'gcc', '-nostdlib', '-mcpu=arm7tdmi', '-Wl,-T,' + str(R/'engine/koko.ld'), '-Wl,--gc-sections',
       '-Wl,--no-warn-rwx-segments', f'-Wl,-Map,{B/(out + ".map")}', '-o', elf, *objs, '-lgcc')
    sh(P + 'objcopy', '-O', 'binary', elf, rom)
    sh('python3', R/'tools/fixrom.py', rom, *([] if cardtest else ['--sums']))
    print(sh('python3', G/'tests/check_ram_code.py', elf, 'ezfo_read', 'ezfo_write', 'ezfo_detect').strip())
    nm = {l.split()[-1]: int(l.split()[0], 16) for l in sh(P + 'nm', elf).splitlines() if len(l.split()) == 3}
    print(f'{rom}: {rom.stat().st_size} bytes; IWRAM end {nm["__bss_end"]:#x} (limit 0x3007800), '
          f'EWRAM end {nm["__ewram_sd_bss_end"]:#x} (limit 0x2040000)')
    return rom


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cardtest', action='store_true', help='build the card test ROM')
    build(ap.parse_args().cardtest)
