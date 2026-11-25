"""kokoSD crash test: random button sessions in the emulator (typing, chords in any order, holds, the menu,
facts and Tell me more from the SD card, sending, stopping, saving and opening chats, scrolling). After each
session Koko must still answer the buttons: back to typing, and two typed letters move the cursor.
Usage: KOKO_RUNNER=build/runner-sc python3 tools/crash_test.py FIRST_SEED LAST_SEED ACTIONS
(needs out/facts-card.img, made by engine/test_game.py). GPL-3.0-or-later."""
import sys, os, subprocess, struct, random, json
os.chdir('/home/user/kokosd')
exec(open('engine/test_game.py').read().split("ok = True")[0])
S = 'out/crash/'
os.makedirs(S, exist_ok=True)
pad = Pad(); h = pad.h
CARET = DBG + 44 + 2 * HISTMAX + 16
def dumps(tag): return [f'dump:{DBG:#x}:{44 + 2 * HISTMAX + 24}:{S}{tag}.bin']
def read(tag):
    d = open(f'{S}{tag}.bin', 'rb').read()
    st = struct.unpack('I', d[4:8])[0]; caret = struct.unpack('I', d[44 + 2 * HISTMAX + 16:44 + 2 * HISTMAX + 20])[0]
    return st, caret
WORDS = ['hi', 'who are you', 'what is a dog', 'tell me more', 'more', 'what is 3 + 4', 'café', 'münchen', 'i like ', 'what is the ', 'x',
         'who is jennifer lopez', 'what is zircon', 'tell me about paris', 'who was mozart', 'tell me more about japan', 'what is a swallow']
def action(rng):
    k = rng.randrange(21)
    if k == 0: return pad.text(rng.choice(WORDS))
    if k == 1: return pad.text(' ')
    if k == 2: return h(B, rng.choice([4, 30, 90])) + h(0, 3)
    if k == 3: return [f'k:{START}']                                       # START (ideas / send)
    if k == 4: return pad.start_with(rng.choice([L, R, LEFT, RIGHT, A, B]))
    if k == 5:
        b = rng.choice([L, R]); return h(b, rng.randint(1, 6)) + h(b | START, rng.randint(1, 6)) + h(rng.choice([b, START, 0]), rng.randint(1, 4)) + h(0, 3)
    if k == 6: return pad.start_with(SELECT) + ['f:10'] + [f'k:{rng.choice([UP, DOWN, A, B, START, LEFT])}' for _ in range(rng.randint(1, 5))]
    if k == 7: return [f'f:{rng.choice([1, 10, 60, 300])}']
    if k == 8:                                                             # random chord
        m = rng.randrange(1, 1024); return h(m, rng.choice([1, 3, 10, 40])) + h(0, rng.choice([1, 3]))
    if k == 9: return [f'k:{rng.choice([A, B, SELECT, START, UP, DOWN, LEFT, RIGHT, L, R])}']
    if k == 10: return h(START, rng.randint(1, 60)) + h(0, 2)
    if k == 11: return h(SELECT, rng.randint(1, 40)) + h(0, 2)
    if k == 12: return h(START | SELECT, rng.randint(1, 20)) + h(0, 2)
    if k == 13: return h(L | R, rng.randint(1, 20)) + h(0, 2)              # flash-cart menu style chord
    if k == 14: return pad.text('what is a husky?') + [f'k:{START}'] + [f'f:{rng.choice([0, 30, 300, 2000])}'] + pad.text('tell me more') + [f'k:{START}', f'f:{rng.choice([0, 100, 2000])}'] + [f'k:{rng.choice([DOWN, UP, R, L, B, A])}' for _ in range(rng.randint(0, 4))]
    if k == 15: return [f'k:{START}', f'f:{rng.choice([0, 5, 50, 500])}'] + [f'k:{rng.choice([A, B, START, UP, L, R, SELECT])}' for _ in range(rng.randint(0, 25))] + [f'k:{B}']   # send, mash, stop
    if k == 16: return h(rng.choice([UP, DOWN, LEFT, RIGHT]) | rng.choice([0, L]), 4) + h(rng.choice([UP, DOWN, LEFT, RIGHT]) | rng.choice([A, B, R]) | SELECT, rng.randint(1, 6)) + h(0, 3)
    if k == 18: return pad.start_with(A) + [f'f:{rng.choice([0, 30, 300])}']                  # save
    if k == 19: return pad.start_with(SELECT) + ['f:10', f'k:{DOWN}', f'k:{A}', f'f:{rng.choice([5, 60])}'] + [f'k:{rng.choice([UP, DOWN, A, B, LEFT, RIGHT])}' for _ in range(rng.randint(0, 4))] + [f'k:{A}', 'f:60']   # Saved chats
    if k == 20: return pad.h(START, 3) + (pad.h(START | rng.choice([UP, DOWN]), rng.randint(1, 40)) + pad.h(START, 2)) * rng.randint(1, 4) + pad.h(0, 3)   # scroll
    return pad.text(''.join(rng.choice('abcdefghijklmnopqrstuvwxyz ') for _ in range(rng.randint(1, 20))))
def run(seed, n):
    rng = random.Random(seed)
    steps = BOOT[:]
    for _ in range(n): steps += action(rng)
    # Liveness: leave any reply / page / menu with B, wait for typing, then type and see the cursor move
    steps += ['f:5', f'k:{B}', 'f:60', f'k:{B}', 'f:60', f'k:{B}', 'f:60', f'k:{SELECT}', f'until:{DBG + 4:#x}:3:3000'] + dumps('a') + pad.text('zz') + ['f:40'] + dumps('b')
    env = dict(os.environ, SD_IMAGE='out/facts-card.img')
    try:
        r = subprocess.run([RUNNER, 'build/koko.gba'] + steps + [f'shot:{S}last-{seed}.ppm'], capture_output=True, text=True, timeout=900, env=env)
    except subprocess.TimeoutExpired:
        return seed, 'TIMEOUT', None
    if r.returncode: return seed, 'NO_IDLE ' + r.stderr.strip()[-60:], None
    (s1, c1), (s2, c2) = read('a'), read('b')
    return seed, 'ok' if c2 == c1 + 2 or c2 == 2 else f'STUCK caret {c1}->{c2} state {s1}->{s2}', None
if __name__ == '__main__':
    a, b, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for seed in range(a, b):
        print(*run(seed, n)[:2], flush=True)
