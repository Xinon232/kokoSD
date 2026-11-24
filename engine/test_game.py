"""Boot the game in headless mGBA and drive it with button presses. Checks: messages reach the model exactly as the
training tokenizer encodes them, replies equal the native engine's for the same seed, typing suggestions equal the
native engine's, and deleting into a finished word rewinds correctly."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import struct, subprocess, sys, os
os.chdir(ROOT)
sys.path.insert(0, 'study')
from common import tokenizer
tok = tokenizer()
ROM, DBG, HISTMAX = 'build/koko.gba', 0x02000000, 1024
A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L = (1 << i for i in range(10))
ROWS = ["abcdefghijklm", "nopqrstuvwxyz", "0123456789+-\x02", ".,!?'___/()\x01<"]   # \x01: new chat, \x02: helpers on/off
IDLE = f'until:{DBG + 4:#x}:3:4000'
BOOT = ['f:20', 'shot:out/koko_title.ppm', f'k:{START}', 'f:10']      # the title screen, then START


class Pad:
    """Turns text into button presses, tracking the keyboard cursor the way the game moves it."""
    def __init__(self): self.x = self.y = 0
    @staticmethod
    def bar(y, x): return y == 3 and 5 <= x <= 7
    def move(self, dx, dy):
        x, y = self.x, self.y
        if dy: y = (y + dy) % 4
        if dx:
            x = (x + dx) % 13
            if self.bar(y, x) and self.bar(self.y, self.x): x = 8 if dx > 0 else 4
        if self.bar(y, x): x = 6
        self.x, self.y = x, y
    def key(self, ch):
        ch = '_' if ch == ' ' else ch
        r = next(i for i, row in enumerate(ROWS) if ch in row); c = 6 if ch == '_' else ROWS[r].index(ch)
        out = []
        dy = (r - self.y) % 4
        for _ in range(dy if dy <= 2 else 4 - dy):
            out.append(DOWN if dy <= 2 else UP); self.move(0, 1 if dy <= 2 else -1)
        best = None
        for d, btn in ((1, RIGHT), (-1, LEFT)):                # try each direction, keep the shorter
            save = (self.x, self.y); n = 0
            while self.x != c and n < 14: self.move(d, 0); n += 1
            self.x, self.y = save
            if best is None or n < best[0]: best = (n, d, btn)
        for _ in range(best[0]): out.append(best[2]); self.move(best[1], 0)
        assert (self.x, self.y) == (c, r), (ch, self.x, self.y)
        return [f'k:{k}' for k in out + [A]]
    def text(self, s): return [st for ch in s for st in self.key(ch)]


def run(name, steps, nturn):
    steps = BOOT + steps + [f'until:{DBG + 12:#x}:{nturn}:400000', 'f:5', f'shot:out/{name}.ppm', f'dump:{DBG:#x}:{44 + 2 * HISTMAX}:out/{name}.bin']
    subprocess.run(['build/runner', ROM] + steps, check=True)
    from PIL import Image
    Image.open(f'out/{name}.ppm').resize((480, 320), Image.NEAREST).save(f'out/{name}.png')
    return parse(f'out/{name}.bin')


def parse(path):
    d = open(path, 'rb').read()
    magic, state, seed, nturn, ntok, ticks, made, nsug, rew = struct.unpack('9I', d[:36])
    assert magic == 0x4B4F4B4F, hex(magic)
    sug = struct.unpack('2H', d[36:40])[:nsug]
    newchats = struct.unpack('H', d[40:42])[0]
    hist = list(struct.unpack(f'{ntok}H', d[44:44 + 2 * ntok]))
    return dict(seed=seed, turns=nturn, hist=hist, rate=made * 16384 / max(ticks, 1), made=made, sug=list(sug), rewinds=rew, state=state, newchats=newchats)


def turns_of(hist):
    text = tok.decode(hist, skip_special_tokens=False)
    return [tuple(p.split('<|koko|>')) for p in text.split('<|user|>')[1:] if '<|koko|>' in p]   # the editor may already have fed the next marker


def native_reply(hist, seed):
    """The reply the native engine gives after the same tokens (up to the last Koko marker) with the same seed."""
    cut = len(hist) - 1 - hist[::-1].index(2)
    out = subprocess.run(['build/koko_gen', 'build/koko.bin', str(seed), '512', '120', '1', '0', '1'],
                         input=' '.join(map(str, hist[:cut + 1])).encode(), capture_output=True).stdout.split()
    out = [int(x) for x in out]
    rest = hist[cut + 1:]
    return out == rest[:len(out)] and rest[len(out):] in ([], [1])   # then at most the next turn's marker


def native_suggest(hist, at_start, prefix):
    out = subprocess.run(['build/suggest_test', 'build/koko.bin', '1' if at_start else '0', prefix],
                         input=' '.join(map(str, hist)).encode(), capture_output=True).stdout.decode()
    return [s.strip() for s in out.replace(']', '').split('[')[1:]][:2]


ok = True
def check(cond, what):
    global ok
    ok &= bool(cond); print('PASS' if cond else 'FAIL', what)

# 1. a ready-made message: START fills it in, START sends it
r = run('koko_quick', [f'k:{START}', IDLE, f'k:{START}'], 1)
t = turns_of(r['hist'])
check(t[0][0] == 'hi', f"ready-made message arrived as typed; reply: {t[0][1].strip()!r} at {r['rate']:.2f} tokens/s")
check(native_reply(r['hist'], r['seed']), 'reply identical to the native engine')

# 2. typed on the keyboard; suggestions checked against the native engine; one accepted with L
pad = Pad()
typed = pad.text('what is the ')
subprocess.run(['build/runner', ROM] + BOOT + typed + [IDLE, 'f:5', 'shot:out/koko_suggest.ppm', f'dump:{DBG:#x}:{44 + 2 * HISTMAX}:out/s.bin'], check=True)
from PIL import Image
Image.open('out/koko_suggest.ppm').resize((480, 320), Image.NEAREST).save('out/koko_suggest.png')
s = parse('out/s.bin')
got = [tok.decode([i]).strip() for i in s['sug']]
want = native_suggest(s['hist'], False, '')
check(got == want and len(got) == 2, f"suggestions after 'what is the ': {got} (native engine: {want})")
check(tok.decode(s['hist'], skip_special_tokens=False) == '<|user|>what is the', 'the model had already read the finished words while typing')
r = run('koko_typed', typed + [IDLE, f'k:{L}'] + pad.text('of') + pad.text(' ') + pad.text('france?') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
check(t[0][0] == f'what is the {got[0]} of france?', f"typed message with an accepted suggestion: {t[0][0]!r} -> {t[0][1].strip()!r}")
check(native_reply(r['hist'], r['seed']), 'reply identical to the native engine')

# 3. delete back into a finished word, retype, send
pad = Pad()
steps = pad.text('i like cats ') + [IDLE] + [f'k:{B}'] * 5 + pad.text('dogs') + [f'k:{START}']
r = run('koko_rewind', steps, 1)
t = turns_of(r['hist'])
check(t[0][0] == 'i like dogs' and r['rewinds'] >= 1, f"after deleting a word: {t[0][0]!r}, rewinds {r['rewinds']} -> {t[0][1].strip()!r}")
check(native_reply(r['hist'], r['seed']), 'reply identical to the native engine')

# 4. two turns in one conversation
r = run('koko_two_turns', [f'k:{START}', IDLE, f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5',
                           ] + Pad().text('who are you') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
print('   conversation:', [(a, b.strip()) for a, b in t])
check(len(t) == 2 and t[1][0] == 'who are you', 'second turn arrived as typed')
check(native_reply(r['hist'], r['seed']), 'second reply identical to the native engine')
# 5. a sum is worked out by the game and said by Koko; the conversation then carries on
pad = Pad()
r = run('koko_sum', pad.text('what is 17 + 25?') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.text('thanks') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
print('   conversation:', [(a, b.strip()) for a, b in t])
check(t[0] == ('what is 17 + 25?', '17 + 25 = 42.'), 'the sum is exact')
check(len(t) == 2 and t[1][0] == 'thanks', 'typing straight after a reply loses no key presses')
check(native_reply(r['hist'], r['seed']), 'the reply after it is identical to the native engine')
# 6. new chat: a turn, then the new-chat key, then another turn; the model must have forgotten the first
pad = Pad()
r = run('koko_newchat', pad.text('my name is sam') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.key('\x01') + ['f:10', 'shot:out/koko_newchat_fresh.ppm']
        + pad.text('hi') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
check(r['newchats'] == 1 and len(t) == 1 and t[0][0] == 'hi', f"after new chat the model's history is only the new message: {t}")
check(native_reply(r['hist'], r['seed']), 'reply after a new chat is identical to the native engine starting fresh')
# 7. a question the cartridge has a stored fact for, then a follow-up that the model answers
pad = Pad()
r = run('koko_fact', pad.text('who was george washington?') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.text('was he nice?') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
print('   conversation:', [(a, b.strip()) for a, b in t])
check(t[0][1] == 'George Washington (February 22, 1732 - December 14, 1799) was the first president of the United States from 1789 to 1797.', 'the stored fact is said word for word')
check(len(t) == 2 and t[1][0] == 'was he nice?' and native_reply(r['hist'], r['seed']), 'the follow-up reply is identical to the native engine')
# 8. helpers off: the same question is answered by the model alone
pad = Pad()
r = run('koko_pure', pad.key('\x02') + pad.text('who was george washington?') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
print('   model only:', t[0][1].strip())
check(native_reply(r['hist'], r['seed']) and not t[0][1].startswith('George Washington (February'), 'with helpers off the reply is the model\'s own, identical to the native engine')
# 9. the slash key: a division typed with it is calculated
pad = Pad()
r = run('koko_divide', pad.text('84 / 7') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
check(t[0] == ('84 / 7', '84 / 7 = 12.'), f"division typed with the slash key: {t[0]}")
sys.exit(0 if ok else 1)
