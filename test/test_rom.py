"""Boot the ROM in headless mGBA, drive it with button presses, and check that
the cartridge produced exactly the tokens the native twin produces from the
same prompt and random seed."""
import struct
import subprocess
import sys

ROM, DBG, HISTMAX = 'build/tinystories.gba', 0x02000000, 2048
A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L = (1 << i for i in range(10))


def run(name, presses, prompt):
    steps = ['f:20'] + [f'k:{k}' for k in presses] + [
        f'k:{START}', f'until:{DBG + 4:#x}:2:60000', f'shot:out/{name}.ppm',
        f'dump:{DBG:#x}:{24 + 2 * HISTMAX}:out/{name}.bin']
    subprocess.run(['build/runner', ROM] + steps, check=True)
    d = open(f'out/{name}.bin', 'rb').read()
    magic, state, seed, nprompt, ntok, ticks = struct.unpack('6I', d[:24])
    assert magic == 0x4C4C4D21 and state == 2, (hex(magic), state)
    rom = list(struct.unpack(f'{ntok}H', d[24:24 + 2 * ntok]))
    twin = subprocess.run(['build/twin', 'gen', 's', str(seed), str(HISTMAX - nprompt), prompt],
                          capture_output=True, text=True, check=True).stdout.split()
    twin = [int(t) for t in twin]
    made = ntok - nprompt
    rate = made * 16384 / ticks
    ok = rom == twin
    print(f'{"PASS" if ok else "FAIL"} {name}: prompt {prompt!r} -> {nprompt} prompt tokens, '
          f'{made} generated, {rate:.2f} tokens/s in the emulator')
    if not ok:
        i = next((i for i, (a, b) in enumerate(zip(rom, twin)) if a != b), min(len(rom), len(twin)))
        print(f'  first difference at token {i}: rom {rom[i:i + 5]} twin {twin[i:i + 5]} '
              f'(lengths {len(rom)} / {len(twin)})')
    return ok


ok = run('default', [], 'Once upon a time')
ok &= run('idea', [R], 'One day, a little girl named Lily')
# two ideas along, then type a space and "a" on the on-screen keyboard
ok &= run('typed', [R, R, UP, LEFT, LEFT, A, DOWN, RIGHT, RIGHT, A], 'The big dog a')
ok &= run('empty', [L], '')
sys.exit(0 if ok else 1)
