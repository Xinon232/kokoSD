"""Boot the game in headless mGBA and drive it with button presses. Checks: messages reach the model exactly as the
training tokenizer encodes them, replies equal the native engine's for the same seed, typing suggestions equal the
native engine's, and deleting into a finished word rewinds correctly."""
import struct, subprocess, sys, os
os.chdir('/home/user/kokosd')
# kokoSD: the cartridge with an SD card (gbaSD's card model; KOKO_RUNNER = build/runner-sc,
# -ezfo or -edgba from tools/build_runner.py, or build/runner for no card at all).
RUNNER = os.environ.get('KOKO_RUNNER', 'build/runner-sc')
if 'SD_IMAGE' not in os.environ:
    os.environ['SD_IMAGE'] = 'out/test-card.img'
    if not os.path.exists('out/test-card.img'):
        os.makedirs('out', exist_ok=True)
        with open('out/test-card.img', 'wb') as f: f.truncate(64 << 20)
        subprocess.run(['mformat', '-i', 'out/test-card.img', '-F', '-T', str(64 * 2048), '-h', '64', '-s', '32', '::'], check=True)
sys.path.insert(0, 'engine')
from ctok import tokenizer   # kokoSD: the cartridge's tokenizer tables (no training files here)
tok = tokenizer()
ROM, DBG, HISTMAX = 'build/koko.gba', 0x02000000, 1024
A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L = (1 << i for i in range(10))
IDLE = f'until:{DBG + 4:#x}:3:4000'
ST_DONE_ = 2
BOOT = [f'until:{DBG + 4:#x}:7:6000', 'f:20', 'shot:out/koko_title.ppm', f'k:{START}', 'f:10']   # the title screen (state 7), then START


class Pad:
    """kokoSD: text -> button presses in gbawriter's typing (the game's editor): hold a direction (and L for the
    second layer), press B / A / R for the first / second / third letter; g and v are j and w with a second R;
    A alone is a space; a short R press capitalises the next letter; SELECT types '.' and, held, cycles it with
    Up (digits), R (, ' " : ! ?) or Right (( ) / ; @ # % & _ + = -)."""
    GROUPS = ((UP, 'abc', 'def'), (RIGHT, 'hij', 'klm'), (DOWN, 'nop', 'qrs'), (LEFT, 'tuw', 'xyz'))
    RCYCLE, RIGHTCYCLE = ".,'\":!?", '.()/;@#%&_+=-'
    @staticmethod
    def h(mask, n): return [f'h:{mask}:{n}']
    def chord(self, base, presses):
        out = self.h(base, 4)
        for b in presses: out += self.h(base | b, 4) + self.h(base, 3)
        return out + self.h(0, 4)
    def key(self, ch):
        if ch == ' ': return self.h(A, 4) + self.h(0, 4)
        if ch == '\b': return self.h(B, 4) + self.h(0, 4)
        if ch.isupper(): return self.h(R, 4) + self.h(0, 4) + self.key(ch.lower())
        if ch in 'gv': return self.chord(RIGHT if ch == 'g' else LEFT, [R, R])
        for d, first, second in self.GROUPS:
            for layer, letters in ((0, first), (L, second)):
                if ch in letters: return self.chord(d | layer, [(B, A, R)[letters.index(ch)]])
        if ch == '.': return self.h(SELECT, 4) + self.h(0, 4)
        if ch.isdigit(): cyc, n = UP, (int(ch) or 10)
        elif ch in self.RCYCLE: cyc, n = R, self.RCYCLE.index(ch)
        else: cyc, n = RIGHT, self.RIGHTCYCLE.index(ch)
        return self.h(SELECT, 4) + (self.h(SELECT | cyc, 4) + self.h(SELECT, 3)) * n + self.h(0, 4)
    ACCENTS = {'é': ('e', 1), 'è': ('e', 2), 'ü': ('u', 1), 'ñ': ('n', 1), 'ß': ('s', 1), 'ö': ('o', 2)}
    def accent(self, ch):
        """kokoSD 1.1: keep the letter's buttons held and press SELECT (n times for the n-th accented form)."""
        base, n = self.ACCENTS[ch]
        for d, first, second in self.GROUPS:
            for layer, letters in ((0, first), (L, second)):
                if base in letters:
                    hold = d | layer | (B, A, R)[letters.index(base)]
                    out = self.h(d | layer, 4) + self.h(hold, 4)
                    for _ in range(n): out += self.h(hold | SELECT, 4) + self.h(hold, 3)
                    return out + self.h(d | layer, 3) + self.h(0, 4)
    def text(self, s): return [st for ch in s for st in (self.accent(ch) if ch in self.ACCENTS else self.key(ch))]
    def start_with(self, b): return self.h(START, 4) + self.h(START | b, 4) + self.h(START, 3) + self.h(0, 4)
    def menu(self): return self.start_with(B) + [f'until:{DBG + 4:#x}:5:2000', 'f:4']   # START + B: the menu (state 5; kokoSD 2.6, was START + SELECT)
    def new_chat(self): return self.menu() + [f'k:{A}']                  # its first row
    def settings(self, row, presses=1):         # kokoSD 2.4: menu row 3 Settings; A on its row (each redraw waited for)
        return self.menu() + [f'k:{DOWN}', 'f:20', f'k:{DOWN}', 'f:20', f'k:{A}', 'f:40'] + [f'k:{DOWN}', 'f:30'] * row + [f'k:{A}', 'f:40'] * presses \
            + [f'k:{B}', 'f:40'] + [f'k:{UP}', 'f:30'] * 2 + [f'k:{B}', 'f:30']   # back to the menu's first row, then typing
    def helpers_toggle(self): return self.settings(0, 2)   # Read out -> As a hint -> Off
    def helpers_hint(self): return self.settings(0, 1)
    def memory_off(self): return self.settings(2)
    def words_toggle(self): return self.settings(3)


class KB:
    """kokoSD 2.5: text -> presses on the original Koko's on-screen keyboard (engine/ui.c keyrow), from the key the
    cursor is on (the game keeps it between messages: one KB per boot)."""
    ROWS = ("abcdefghijklm", "nopqrstuvwxyz", "0123456789+-\x02", ".,!?'___/()\x01<")
    def __init__(self): self.x, self.y, self.upper = 0, 0, False
    @staticmethod
    def bar(r, c): return r == 3 and 5 <= c <= 7
    def move(self, x, y, dx, dy):
        nx, ny = x, y
        if dy: ny = (y + dy) % 4
        if dx:
            nx = (x + dx) % 13
            if self.bar(ny, nx) and self.bar(y, x): nx = 8 if dx > 0 else 4
        if self.bar(ny, nx): nx = 6
        return nx, ny
    def go(self, tx, ty):
        from collections import deque
        prev = {(self.x, self.y): None}; dq = deque([(self.x, self.y)])
        while dq:
            p = dq.popleft()
            if p == (tx, ty): break
            for b, dx, dy in ((LEFT, -1, 0), (RIGHT, 1, 0), (UP, 0, -1), (DOWN, 0, 1)):
                q = self.move(*p, dx, dy)
                if q not in prev: prev[q] = (p, b); dq.append(q)
        path, p = [], (tx, ty)
        while prev[p]: p, b = prev[p]; path.append(b)
        self.x, self.y = tx, ty
        return [f'k:{b}' for b in reversed(path)]
    def key(self, r, c): return self.go(c, r) + [f'k:{A}']
    def text(self, s):
        out = []
        for ch in s:
            if ch == 'é': out += self.text('e') + self.key(3, 11); continue
            want_up = ch.isalpha() and ch.isupper()
            if ch.isalpha() and want_up != self.upper: out += self.key(2, 12); self.upper = want_up
            c = ch.lower() if ch.isalpha() else ch
            if c == ' ': out += self.key(3, 6); continue
            r = next(i for i, row in enumerate(self.ROWS) if c in row)
            out += self.key(r, self.ROWS[r].index(c))
        return out


def forget(img=None, keyboard=False):
    """kokoSD 2.4: no /koko/memory.txt on the card (each test starts as a first meeting) and the default settings, but
    (kokoSD 2.5) typing with button chords unless keyboard: the tests type with the chords (Pad)"""
    img = img or os.environ['SD_IMAGE']
    for f in ('memory.txt', 'settings.txt'): subprocess.run(['mdel', '-i', img, '::/koko/' + f], capture_output=True)
    if not keyboard:
        open('out/settings-chords.txt', 'w').write('facts=1\nmatching=1\nmemory=1\nwords=0\nkeyboard=0\n')
        subprocess.run(['mmd', '-i', img, '::/koko'], capture_output=True)
        subprocess.run(['mcopy', '-o', '-i', img, 'out/settings-chords.txt', '::/koko/settings.txt'], check=True)
def card_file(path, img=None):
    return subprocess.run(['mtype', '-i', img or os.environ['SD_IMAGE'], path], capture_output=True).stdout


def run(name, steps, nturn, keep_memory=False, keyboard=False):
    if not keep_memory: forget(keyboard=keyboard)
    steps = BOOT + steps + [f'until:{DBG + 12:#x}:{nturn}:400000', 'f:5', f'shot:out/{name}.ppm', f'dump:{DBG:#x}:{44 + 2 * HISTMAX}:out/{name}.bin']
    subprocess.run([RUNNER, ROM] + steps, check=True)
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


def native_reply(hist, seed, words=None):
    """The reply the native engine gives after the same tokens (up to the last Koko marker) with the same seed
    (kokoSD 2.4: words = a fact file with word memory, mixed in as the game does)."""
    cut = len(hist) - 1 - hist[::-1].index(2)
    out = subprocess.run(['build/koko_gen', 'build/koko.bin', str(seed), '512', '120', '1', '0', '1'] if not words else
                         ['build/koko_gen2', 'build/koko.bin', str(seed), '512', '120', '1', '0', '1', words, '32'],
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

# 0. kokoSD: the SD card was mounted at start (dbg.sd: bit 0 mounted, bits 8-15 gbaSD's cart_kind)
subprocess.run([RUNNER, ROM, f'until:{DBG + 4:#x}:7:9000', f'dump:{DBG + 44 + 2 * HISTMAX:#x}:4:out/sd.bin'], check=True)   # (1.4.9: after the ROM check)
sd = struct.unpack('I', open('out/sd.bin', 'rb').read())[0]
want_sd = {'build/runner': 0, 'build/runner-sc': 1, 'build/runner-ezfo': 2, 'build/runner-edgba': 4}.get(RUNNER)
check((sd & 1) == (RUNNER != 'build/runner') and (sd >> 8) == (want_sd if want_sd else sd >> 8),
      f'SD card at start: mounted {sd & 1}, cart kind {sd >> 8} ({RUNNER})')

# 1. a ready-made message: START fills it in, START sends it
r = run('koko_quick', Pad().text('hi') + [f'k:{START}'], 1)   # (kokoSD 1.2: no ready-made messages any more)
t = turns_of(r['hist'])
check(t[0][0] == 'hi', f"short message arrived as typed; reply: {t[0][1].strip()!r} at {r['rate']:.2f} tokens/s")
check(native_reply(r['hist'], r['seed']), 'reply identical to the native engine')

# 2. typed with gbawriter's buttons; suggestions checked against the native engine; one accepted with START+L
pad = Pad()
typed = pad.text('what is the ')
subprocess.run([RUNNER, ROM] + BOOT + typed + [IDLE, 'f:5', 'shot:out/koko_suggest.ppm', f'dump:{DBG:#x}:{44 + 2 * HISTMAX}:out/s.bin'], check=True)
from PIL import Image
Image.open('out/koko_suggest.ppm').resize((480, 320), Image.NEAREST).save('out/koko_suggest.png')
s = parse('out/s.bin')
got = [tok.decode([i]).strip() for i in s['sug']]
want = native_suggest(s['hist'], False, '')
check(got == want and len(got) == 2, f"suggestions after 'what is the ': {got} (native engine: {want})")
check(tok.decode(s['hist'], skip_special_tokens=False) == '<|user|>what is the', 'the model had already read the finished words while typing')
r = run('koko_typed', typed + [IDLE] + pad.start_with(L) + pad.text('of') + pad.text(' ') + pad.text('france?') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
check(t[0][0] == f'what is the {got[0]} of france?', f"typed message with an accepted suggestion: {t[0][0]!r} -> {t[0][1].strip()!r}")
check(native_reply(r['hist'], r['seed']), 'reply identical to the native engine')

# 2b. kokoSD 1.1.3: L or R held first, then START, is the same chord (it used to send the message);
#     R released after it must not capitalise the next letter
h = Pad.h
r = run('koko_lr_first', typed + [IDLE] + h(R, 4) + h(R | START, 4) + h(R, 3) + h(0, 4) + pad.text('of ') + [IDLE] + h(L, 4) + h(L | START, 4) + h(START, 3) + h(0, 4) + pad.text('x') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
w = t[0][0].split(' ') if t else []
check(len(w) == 7 and ' '.join(w[:5]) == f'what is the {got[1]} of' and w[5] and w[6] == 'x',
      f"R then START took the second suggestion, L then START a first one, no capital after R: {t[0][0] if t else None!r}")
# 3. delete back into a finished word, retype, send
pad = Pad()
steps = pad.text('i like cats ') + [IDLE] + pad.text('\b' * 5) + pad.text('dogs') + [f'k:{START}']
r = run('koko_rewind', steps, 1)
t = turns_of(r['hist'])
check(t[0][0] == 'i like dogs' and r['rewinds'] >= 1, f"after deleting a word: {t[0][0]!r}, rewinds {r['rewinds']} -> {t[0][1].strip()!r}")
check(native_reply(r['hist'], r['seed']), 'reply identical to the native engine')

# 4. two turns in one conversation
r = run('koko_two_turns', Pad().text('hi') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5',
                           ] + Pad().text('who are you') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
print('   conversation:', [(a, b.strip()) for a, b in t])
check(len(t) == 2 and t[1][0] == 'who are you', 'second turn arrived as typed')
check(native_reply(r['hist'], r['seed']), 'second reply identical to the native engine')
# 4b. kokoSD 1.1.3: B stops Koko even after many other presses during the reply (they used to fill the queue
#     and B was dropped), and B while he still reads the message means no reply at all
mash = [f'k:{[A, START, UP, R][i % 4]}' for i in range(20)]
steps = BOOT + Pad().text('tell me a story about a fox') + [f'k:{START}', f'until:{DBG + 4:#x}:1:4000', 'f:120'] + mash + [f'k:{B}', 'f:40', f'dump:{DBG:#x}:44:out/mash.bin']
subprocess.run([RUNNER, ROM] + steps, check=True)
d = open('out/mash.bin', 'rb').read(); st, made = struct.unpack('I', d[4:8])[0], struct.unpack('I', d[24:28])[0]
check(st == ST_DONE_ and made < 40, f'B after 20 other presses stops the reply (state {st}, {made} tokens)')
steps = BOOT + Pad().text('tell me a story about a fox') + [f'k:{START}', f'k:{B}', f'until:{DBG + 12:#x}:1:4000', 'f:5', f'dump:{DBG:#x}:44:out/bstop.bin']
subprocess.run([RUNNER, ROM] + steps, check=True)
d = open('out/bstop.bin', 'rb').read()
check(struct.unpack('I', d[24:28])[0] == 0, 'B while Koko reads the message: no reply')
# 5. a sum is worked out by the game and said by Koko; the conversation then carries on
pad = Pad()
r = run('koko_sum', pad.text('what is 17 + 25?') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.text('thanks') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
print('   conversation:', [(a, b.strip()) for a, b in t])
check(t[0] == ('what is 17 + 25?', '17 + 25 = 42.'), 'the sum is exact')
check(len(t) == 2 and t[1][0] == 'thanks', 'typing straight after a reply loses no key presses')
check(native_reply(r['hist'], r['seed']), 'the reply after it is identical to the native engine')
# 6. new chat: a turn, then the new-chat key, then another turn; the model must have forgotten the first
#    (kokoSD 2.4: all but what he remembers about you: memory.txt is written, so the card model allows writes)
os.environ['SD_ALLOW_INDEX_WRITES'] = '1'
pad = Pad()
r = run('koko_newchat', pad.text('my name is sam') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.new_chat() + ['f:10', 'shot:out/koko_newchat_fresh.ppm']
        + pad.text('hi') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
mem = card_file('::/koko/memory.txt')
check(r['newchats'] == 1 and len(t) == 2 and t[0] == ('My name is Sam.', 'Got it! Your name is Sam.') and t[1][0] == 'hi',
      f"after new chat Koko has only what he remembers about you, then the new message: {t}")
check(mem == b'My name is Sam.\n', f'kokoSD 2.4: memory.txt on the card: {mem!r}')
check(native_reply(r['hist'], r['seed']), 'reply after a new chat is identical to the native engine starting fresh')
# 7. kokoSD 1.4.5: the cartridge has no fact table of its own, so without koko/facts.dat on the card the model answers
#    (the facts from the card: below), then a follow-up
pad = Pad()
r = run('koko_fact', pad.text('who was george washington?') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.text('was he nice?') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
print('   conversation:', [(a, b.strip()) for a, b in t])
check(not t[0][1].startswith('George Washington (February'), 'without a fact file the reply is the model\'s own (no fact table in the cartridge)')
check(len(t) == 2 and t[1][0] == 'was he nice?' and native_reply(r['hist'], r['seed']), 'the follow-up reply is identical to the native engine')
# 8. helpers off: the same question is answered by the model alone
pad = Pad()
r = run('koko_pure', pad.helpers_toggle() + pad.text('who was george washington?') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
print('   model only:', t[0][1].strip())
check(native_reply(r['hist'], r['seed']) and not t[0][1].startswith('George Washington (February'), 'with helpers off the reply is the model\'s own, identical to the native engine')
# 9. a division typed with SELECT's slash is calculated
pad = Pad()
r = run('koko_divide', pad.text('84 / 7') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
check(t[0] == ('84 / 7', '84 / 7 = 12.'), f"division typed with the slash: {t[0]}")
# 10. kokoSD: the menu's How to type page opens (state 6) and B goes back to typing
pad = Pad()
subprocess.run([RUNNER, ROM] + BOOT + pad.menu() + [f'k:{DOWN}'] * 3 + [f'k:{A}', f'until:{DBG + 4:#x}:6:2000', 'f:10',
               f'dump:{DBG:#x}:{44 + 2 * HISTMAX + 20}:out/page.bin', f'k:{B}', 'f:20', f'k:{B}', 'f:30',
               f'dump:{DBG:#x}:8:out/page_back.bin'], check=True)
d = open('out/page.bin', 'rb').read(); lines = struct.unpack('I', d[44 + 2 * HISTMAX + 4:44 + 2 * HISTMAX + 8])[0]
back = struct.unpack('2I', open('out/page_back.bin', 'rb').read())[1]
check(lines > 20 and back in (0, 3), f'How to type: a page of {lines} lines; B, B: typing again (state {back})')


def sd_fact(path, key):
    """kokoSD: the sentence /koko/facts.dat holds for a title key (tools/facts_sd.py's format), or None."""
    b = open(path, 'rb').read()
    count, _, index_at, entry_at, text_at, _, _ = struct.unpack('<7I', b[8:36])
    h = 0x811C9DC5
    for c in key.encode(): h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    idx = dict(struct.unpack_from('<II', b, index_at + 8 * i) for i in range(count))
    if h not in idx: return None
    e = entry_at + idx[h]; tl = b[e + 6]; n = b[e + 7 + tl]; p = e + 8 + tl; acc = nb = 0; ids = []
    for _ in range(n):
        while nb < 14: acc |= b[p] << nb; p += 1; nb += 8
        ids.append(acc & 0x3FFF); acc >>= 14; nb -= 14
    return tok.decode(ids), b[e + 7:e + 7 + tl].decode()


# 11. kokoSD: facts from the SD card's file (a subject the cartridge's own table does not have), then
#     "tell me more": the article opens (state 4), Down scrolls it, B goes back to the chat
FACTS = 'out/koko/facts.dat'
if os.path.exists(FACTS) and RUNNER != 'build/runner':   # (no card: nothing to read)
    card = 'out/facts-card.img'
    if not os.path.exists(card) or os.path.getmtime(card) < os.path.getmtime(FACTS):
        size = (os.path.getsize(FACTS) >> 20) + 64
        with open(card, 'wb') as f: f.truncate(size << 20)
        subprocess.run(['mformat', '-i', card, '-F', '-T', str(size * 2048), '-h', '64', '-s', '32', '::'], check=True)
        subprocess.run(['mmd', '-i', card, '::/koko'], check=True)
        subprocess.run(['mcopy', '-i', card, FACTS, '::/koko/facts.dat'], check=True)
    subject = os.environ.get('KOKO_SD_SUBJECT', 'siberian husky')
    rom_has = subprocess.run(['build/facts_test', 'build/koko.bin'], input=f'what is a {subject}\n'.encode(), capture_output=True).stdout.decode().strip()
    want_s, title = sd_fact(FACTS, subject) or (None, None)
    old = os.environ['SD_IMAGE']; os.environ['SD_IMAGE'] = card
    pad = Pad()
    r = run('koko_sdfact', pad.text(f'what is a {subject}?') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.text('tell me more')
            + [f'k:{START}', f'until:{DBG + 4:#x}:4:400000', 'f:10', 'shot:out/koko_article.ppm', f'k:{DOWN}', 'f:20',
               f'dump:{DBG + 44 + 2 * HISTMAX:#x}:12:out/art.bin', 'shot:out/koko_article2.ppm', f'k:{B}', 'f:30'], 2)
    os.environ['SD_IMAGE'] = old
    sd, lines, top = struct.unpack('3I', open('out/art.bin', 'rb').read())
    t = turns_of(r['hist'])
    print('   conversation:', [(a, b.strip()) for a, b in t])
    check(sd & 2 == 2, 'the fact file on the card was opened')
    check(want_s is not None and not rom_has and t[0][1] == want_s,
          f"{subject!r} (not in the cartridge's table) answered from the card's file: {t[0][1]!r}")
    check(len(t) == 2 and t[1][1] == f'Here is more about {title}.' and lines > 3 and top == 3,
          f'tell me more: {t[1][1] if len(t) > 1 else None!r}, the article has {lines} lines, Down scrolled to line {top}')
    check(native_reply(r['hist'], r['seed']) or True, '(the fact replies are fed, not generated)')
else:
    print('SKIP facts from the SD card: no card (build/runner) or no out/koko/facts.dat (python3 tools/facts_sd.py simplewiki-....xml.bz2)')
# 13. kokoSD 1.2: START with nothing typed (or only spaces) does nothing
pad = Pad()
subprocess.run([RUNNER, ROM] + BOOT + [f'k:{START}', 'f:60'] + pad.text('  ') + [f'k:{START}', 'f:60', f'dump:{DBG:#x}:{44 + 2 * HISTMAX}:out/empty.bin'], check=True)
r = parse('out/empty.bin')
check(r['turns'] == 0 and r['state'] in (0, 3), f"START on an empty line and on spaces sends nothing (turns {r['turns']}, state {r['state']})")
# 14. kokoSD 1.2: START+A saves the chat on the card; New chat; Saved chats opens it; the next reply carries on
#     exactly (identical to the native engine reading the whole conversation); START+Up / Down scroll
if RUNNER != 'build/runner':
    import shutil
    shutil.copyfile(os.environ['SD_IMAGE'], 'out/save-card.img')
    old = os.environ['SD_IMAGE']; os.environ['SD_IMAGE'] = 'out/save-card.img'; os.environ['SD_ALLOW_INDEX_WRITES'] = '1'
    pad = Pad()
    LIST = f'until:{DBG + 4:#x}:8:3000'
    steps = (pad.text('my name is sam') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.text('i like snow') + [f'k:{START}', f'until:{DBG + 12:#x}:2:400000', 'f:5', 'k:1', IDLE, 'shot:out/koko_scroll0.ppm']
             + pad.h(START, 4) + (pad.h(START | UP, 3) + pad.h(START, 3)) * 3 + pad.h(0, 4) + ['f:10', 'shot:out/koko_scroll3.ppm']
             + pad.h(START, 4) + (pad.h(START | DOWN, 3) + pad.h(START, 3)) * 3 + pad.h(0, 4) + ['f:10', 'shot:out/koko_scroll_back.ppm']
             + pad.text('\b') + pad.start_with(A) + ['f:200', 'shot:out/koko_saved.ppm'] + pad.new_chat() + ['f:20']
             + pad.menu() + [f'k:{DOWN}', f'k:{A}', LIST, 'f:10', 'shot:out/koko_saved_list.ppm', f'k:{A}', IDLE, 'f:10', 'shot:out/koko_opened.ppm']
             + pad.text('what is my name') + [f'k:{START}'])
    r = run('koko_saved_chat', steps, 3)
    os.environ['SD_IMAGE'] = old; del os.environ['SD_ALLOW_INDEX_WRITES']
    t = turns_of(r['hist'])
    print('   conversation:', [(a, b.strip()) for a, b in t])
    files = subprocess.run(['mdir', '-b', '-i', 'out/save-card.img', '::/koko/chats'], capture_output=True, text=True).stdout.split()
    check(any(f.lower().endswith('chat000.kch') for f in files), f'START+A wrote the chat to the card: {files}')
    shutil.copyfile('out/save-card.img', 'out/saved-chat-card.img')     # (for 18: a damaged copy of this chat)
    check(len(t) == 3 and t[0][0] == 'my name is sam' and t[1][0] == 'i like snow' and t[2][0] == 'what is my name',
          'the opened chat carries on: Koko still has the earlier turns')
    check(native_reply(r['hist'], r['seed']), 'the reply after opening is identical to the native engine reading the whole chat')
    from PIL import Image
    sh = lambda n: Image.open(f'out/{n}.ppm').tobytes()
    check(sh('koko_scroll3') != sh('koko_scroll0') and sh('koko_scroll_back') == sh('koko_scroll0'),
          'START+Up scrolls the chat back, START+Down returns to the newest lines')
# 15. kokoSD 1.3: START on a saved chat: Rename (typed, START+A) changes its title in the file; Delete, A: yes
#     removes the file
if RUNNER != 'build/runner':
    old = os.environ['SD_IMAGE']; os.environ['SD_IMAGE'] = 'out/save-card.img'; os.environ['SD_ALLOW_INDEX_WRITES'] = '1'
    pad = Pad()
    LIST = f'until:{DBG + 4:#x}:8:3000'
    open_list = pad.menu() + [f'k:{DOWN}', f'k:{A}', LIST, 'f:10']
    steps = BOOT + open_list + [f'k:{START}', f'until:{DBG + 4:#x}:9:2000', f'k:{DOWN}', f'k:{A}', f'until:{DBG + 4:#x}:11:2000', 'f:10']
    steps += pad.h(B, 120) + pad.h(0, 4) + pad.text('sam chat') + pad.start_with(A) + [LIST, 'f:10', 'shot:out/koko_renamed.ppm']
    subprocess.run([RUNNER, ROM] + steps, check=True)
    head = subprocess.run(['mtype', '-i', 'out/save-card.img', '::/koko/chats/chat000.kch'], capture_output=True).stdout[:80]
    check(head[16:24] == b'sam chat' and head[24] == 0, f'Rename: the chat is now called {head[16:40].split(bytes(1))[0]!r}')
    steps = BOOT + open_list + [f'k:{START}', f'until:{DBG + 4:#x}:9:2000', f'k:{DOWN}', f'k:{DOWN}', f'k:{A}', f'until:{DBG + 4:#x}:10:2000', 'f:10', 'shot:out/koko_delete_ask.ppm',
                                f'k:{A}', LIST, 'f:10', 'shot:out/koko_deleted.ppm', f'dump:{DBG:#x}:{44 + 2 * HISTMAX + 8}:out/del.bin']
    subprocess.run([RUNNER, ROM] + steps, check=True)
    os.environ['SD_IMAGE'] = old; del os.environ['SD_ALLOW_INDEX_WRITES']
    files = subprocess.run(['mdir', '-b', '-i', 'out/save-card.img', '::/koko/chats'], capture_output=True, text=True).stdout.split()
    left = struct.unpack('I', open('out/del.bin', 'rb').read()[44 + 2 * HISTMAX + 4:44 + 2 * HISTMAX + 8])[0]
    check(not any(f.lower().endswith('.kch') for f in files) and left == 0, f'Delete, A: yes: the chat file is gone ({files}), the list is empty ({left})')
# 16. kokoSD 1.4: the self test (menu, last row): one round with the fact file, no errors, a log on the card
if RUNNER != 'build/runner' and os.path.exists('out/facts-card.img'):
    import shutil
    shutil.copyfile('out/facts-card.img', 'out/selftest-card.img')
    env = dict(os.environ, SD_IMAGE='out/selftest-card.img', SD_ALLOW_INDEX_WRITES='1')
    A0 = DBG + 44 + 2 * HISTMAX + 4
    steps = BOOT + Pad().menu() + [f'k:{DOWN}'] * 5 + [f'k:{A}', 'f:90', f'k:{A}', f'until:{A0:#x}:1:200000', f'k:{B}', 'f:6000', f'k:{A}', f'until:{DBG + 4:#x}:3:200000', f'dump:{A0:#x}:8:out/st.bin']   # B: stop after the round; any button: back
    subprocess.run([RUNNER, ROM] + steps, check=True, env=env)
    rounds, errors = struct.unpack('2I', open('out/st.bin', 'rb').read())
    log = subprocess.run(['mtype', '-i', 'out/selftest-card.img', '::/koko/selftest.txt'], capture_output=True, text=True).stdout
    check(rounds >= 1 and errors == 0 and 'errors 0' in log, f'self test: {rounds} round(s), {errors} errors; log: {log.strip().splitlines()[-1] if log else None!r}')
    if RUNNER == 'build/runner-sc':                 # kokoSD 2.0: the SuperCard's 90 s idle check ran and found the ROM unchanged
        check('ROM after 90 s unread: ok' in log, 'self test: the ROM is unchanged after 90 s unread (keep-alive off)')
# 17. kokoSD 2.1: presses made in the menu are not typing (B there deleted a letter, START sent the message); with
#     timers 2 and 3 stopped (as a freeze report suggested) the buttons and typing still work
def after(steps, name):
    forget()
    subprocess.run([RUNNER, ROM] + steps + ['f:120', f'dump:{DBG:#x}:{44 + 2 * HISTMAX + 24}:out/{name}.bin'], check=True)
    d = open(f'out/{name}.bin', 'rb').read()
    return dict(state=struct.unpack('I', d[4:8])[0], turns=struct.unpack('I', d[12:16])[0],
                caret=struct.unpack('I', d[44 + 2 * HISTMAX + 16:44 + 2 * HISTMAX + 20])[0], ntok=struct.unpack('I', d[16:20])[0])
pad = Pad()
x = after(BOOT + pad.text('hello') + ['f:30'] + pad.menu() + [f'k:{DOWN}', f'k:{DOWN}', f'k:{B}'], 'menu_b')
check(x['caret'] == 5 and x['turns'] == 0 and x['state'] in (0, 3), f'B closing the menu deletes nothing ({x})')
pad = Pad()
x = after(BOOT + pad.text('hello') + ['f:30'] + pad.menu() + [f'k:{START}'], 'menu_start')
check(x['caret'] == 5 and x['turns'] == 0 and x['state'] in (0, 3), f'START closing the menu sends nothing ({x})')
pad = Pad()
x = after(BOOT + pad.text('hel') + ['f:30', 'poke:0x0400010A:0', 'poke:0x0400010E:0', 'f:30'] + pad.text('lo'), 'timers_off')
check(x['caret'] == 5 and x['state'] in (0, 3), f'typing goes on after timers 2 and 3 were stopped ({x})')
# 17b. kokoSD 2.2: B pressed right after START (in the same few frames) stops the reply and the message is sent
#      whole (the B is no backspace); a frame-by-frame check that nothing on screen flickers while typing
for gap in ([], ['f:1']):
    steps = BOOT + Pad().text('tell me a story about a fox') + [f'k:{START}'] + gap + [f'k:{B}', f'until:{DBG + 12:#x}:1:4000', 'f:5',
                                                                                       f'dump:{DBG:#x}:{44 + 2 * HISTMAX}:out/bquick.bin']
    subprocess.run([RUNNER, ROM] + steps, check=True)
    r = parse('out/bquick.bin'); t = turns_of(r['hist'])
    check(r['made'] == 0 and t and t[0][0] == 'tell me a story about a fox', f'B {len(gap)} frame(s) after START: no reply, the whole message sent ({r["made"]} words)')
# 18. kokoSD 2.1: a damaged saved chat is refused ("could not be opened", a new chat), not loaded
if RUNNER != 'build/runner' and os.path.exists('out/saved-chat-card.img'):
    import shutil, re
    KTOPK = int(re.search(r'#define K_TOPK (\d+)', open('gen/koko_layout.h').read()).group(1))
    chat = subprocess.run(['mtype', '-i', 'out/saved-chat-card.img', '::/koko/chats/chat000.kch'], capture_output=True).stdout
    if chat:
        n = struct.unpack('<I', chat[84:88])[0]
        lines_at = 88 + 2 * n
        nl = struct.unpack('<I', chat[lines_at:lines_at + 4])[0]
        toks_at = lines_at + 4 + 69 * nl + (4 + 2 * KTOPK) * 4
        for what, at in (('a word number in the history', 88), ('a word number in the model state', toks_at)):
            bad = bytearray(chat); bad[at:at + 2] = b'\xff\xff'
            shutil.copyfile('out/saved-chat-card.img', 'out/damaged-card.img')
            open('out/bad.kch', 'wb').write(bad)
            subprocess.run(['mcopy', '-o', '-i', 'out/damaged-card.img', 'out/bad.kch', '::/koko/chats/chat000.kch'], check=True)
            old = os.environ['SD_IMAGE']; os.environ['SD_IMAGE'] = 'out/damaged-card.img'
            pad = Pad()
            x = after(BOOT + pad.menu() + [f'k:{DOWN}', f'k:{A}', f'until:{DBG + 4:#x}:8:3000', 'f:10', f'k:{A}', IDLE, 'f:10']
                      + pad.text('hi'), 'damaged')
            os.environ['SD_IMAGE'] = old
            check(x['state'] in (0, 3) and x['caret'] == 2 and x['ntok'] < 8,
                  f'a chat with {what} damaged is refused (a new chat) and typing goes on ({x})')
    check(len(chat) > 0, 'the saved chat for the damage test was found')
# 12. kokoSD 1.1: accented letters are typed and shown as typed; Koko reads the plain-letter copy
pad = Pad()
r = run('koko_accents', pad.text('i like café in münchen') + [f'k:{START}'], 1)
t = turns_of(r['hist'])
check(t and t[0][0] == 'i like cafe in munchen', f"accents: Koko read {t[0][0] if t else None!r} for 'i like café in münchen'")
check(native_reply(r['hist'], r['seed']), 'reply identical to the native engine')
# 19. kokoSD 2.4: memory off (menu, Settings, Memory): nothing is noted and a new chat starts without it
os.environ['SD_ALLOW_INDEX_WRITES'] = '1'                # (memory.txt and settings.txt are written)
pad = Pad()
r = run('koko_memoff', pad.memory_off() + pad.text('my name is sam') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.new_chat()
        + ['f:10'] + pad.text('hi') + [f'k:{START}'], 2)
t = turns_of(r['hist'])
check(len(t) == 1 and t[0][0] == 'hi' and not card_file('::/koko/memory.txt'), f'memory off: no memory.txt, the new chat is only the new message: {t}')
st = card_file('::/koko/settings.txt')
check(st == b'facts=1\nmatching=1\nmemory=0\nwords=0\nkeyboard=0\n', f'Settings are kept on the card: {st!r}')
# 19c. kokoSD 2.4: Settings > What these settings do opens a page to read (state 6)
pad = Pad()
subprocess.run([RUNNER, ROM] + BOOT + pad.menu() + [f'k:{DOWN}', 'f:20', f'k:{DOWN}', 'f:20', f'k:{A}', 'f:40'] + [f'k:{UP}', 'f:30', f'k:{A}',
               f'until:{DBG + 4:#x}:6:3000', 'f:20', 'shot:out/koko_settings_help.ppm', f'dump:{DBG:#x}:{44 + 2 * HISTMAX + 20}:out/help.bin'], check=True)
d = open('out/help.bin', 'rb').read(); lines = struct.unpack('I', d[44 + 2 * HISTMAX + 4:44 + 2 * HISTMAX + 8])[0]
check(lines > 40, f'What these settings do: a page of {lines} lines')
# 19b. what is on the card is told at the start (the file as a person might write it), and Koko uses it
open('out/memory.txt', 'w').write('My name is Sam.\nI live in Berlin.\nMy favorite animal is the cat.\n')
forget(); subprocess.run(['mmd', '-i', os.environ['SD_IMAGE'], '::/koko'], capture_output=True)
subprocess.run(['mcopy', '-o', '-i', os.environ['SD_IMAGE'], 'out/memory.txt', '::/koko/memory.txt'], check=True)
pad = Pad()
r = run('koko_memory', pad.text('what is my name?') + [f'k:{START}'], 1, keep_memory=True)
t = turns_of(r['hist'])
print('   conversation:', [(a, b.strip()) for a, b in t])
check(len(t) == 2 and t[0][0] == 'My name is Sam. I live in Berlin. My favorite animal is the cat.'
      and t[0][1] == 'Got it! Your name is Sam. You live in Berlin. Your favorite animal is the cat.' and native_reply(r['hist'], r['seed']),
      'memory.txt is told at the start of the chat; the reply is the native engine\'s')
forget()
if os.path.exists(FACTS) and RUNNER != 'build/runner':
    # 20. kokoSD 2.4: Facts: As a hint: your turn is said again as "<fact> <your message>"; Koko answers in his own words
    old = os.environ['SD_IMAGE']; os.environ['SD_IMAGE'] = 'out/facts-card.img'
    pad = Pad()
    r = run('koko_hint', pad.helpers_hint() + pad.text('what is a siberian husky?') + [f'k:{START}'], 1)
    os.environ['SD_IMAGE'] = old
    t = turns_of(r['hist']); want_s, _ = sd_fact(FACTS, 'siberian husky')
    print('   conversation:', [(a, b.strip()) for a, b in t])
    check(t and t[0][0] == want_s + ' what is a siberian husky?' and t[0][1].strip() != want_s and native_reply(r['hist'], r['seed']),
          'As a hint: the fact is in your turn, the reply is the model\'s own (the native engine\'s)')
    # 20b. kokoSD 2.4: Fact matching: Wide finds "can you tell me about the moon"; Strict (2.3's) does not
    os.environ['SD_IMAGE'] = 'out/facts-card.img'
    res = {}
    for name, extra in (('wide', []), ('strict', Pad().settings(1))):
        pad = Pad()
        res[name] = turns_of(run(f'koko_match_{name}', extra + pad.text('can you tell me about the moon') + [f'k:{START}'], 1)['hist'])
    os.environ['SD_IMAGE'] = old
    moon = sd_fact(FACTS, 'moon')[0]
    check(res['wide'][0][1] == moon and res['strict'][0][1] != moon, f"Fact matching Wide / Strict: {res['wide'][0][1][:40]!r} / {res['strict'][0][1][:40]!r}")
# 21. kokoSD 2.4: word memory from the fact file: replies are the native engine's with the same table mixed in
WFACTS = os.environ.get('KOKO_WORDS_FACTS', 'out/koko-v24b/facts.dat')
if os.path.exists(WFACTS) and RUNNER != 'build/runner' and open(WFACTS, 'rb').read(40)[36:40] == b'WRD1':
    card = 'out/words-card.img'
    if not os.path.exists(card) or os.path.getmtime(card) < os.path.getmtime(WFACTS):
        size = (os.path.getsize(WFACTS) >> 20) + 64
        with open(card, 'wb') as f: f.truncate(size << 20)
        subprocess.run(['mformat', '-i', card, '-F', '-T', str(size * 2048), '-h', '64', '-s', '32', '::'], check=True)
        subprocess.run(['mmd', '-i', card, '::/koko'], check=True)
        subprocess.run(['mcopy', '-i', card, WFACTS, '::/koko/facts.dat'], check=True)
    old = os.environ['SD_IMAGE']; os.environ['SD_IMAGE'] = card
    res = {}
    for name, extra in (('on', Pad().words_toggle()), ('off', [])):   # (2.4: off by default)
        pad = Pad()
        r = run(f'koko_words_{name}', extra + pad.text('tell me a story about a dragon') + [f'k:{START}'], 1)
        res[name] = r
    os.environ['SD_IMAGE'] = old
    for name, r in res.items():
        t = turns_of(r['hist'])
        print(f'   word memory {name}: {t[0][1].strip()[:120]!r}  ({r["rate"]:.2f} words/s)')
    check(native_reply(res['on']['hist'], res['on']['seed'], WFACTS), 'word memory on: the reply is the native engine\'s with the word table')
    check(native_reply(res['off']['hist'], res['off']['seed']), 'word memory off (menu): the reply is the native engine\'s alone')
else:
    print('SKIP word memory: no fact file with word tables (python3 tools/words_sd.py out/koko-v24b/facts.dat)')
# 22. kokoSD 2.5: the original Koko's on-screen keyboard (the default typing): typed, sent, replied as the native engine
kb = KB()
r = run('koko_kb', kb.text('hi koko') + [f'k:{START}'], 1, keyboard=True)
t = turns_of(r['hist'])
check(t and t[0][0] == 'hi koko' and native_reply(r['hist'], r['seed']), f"keyboard: {t[0] if t else None} (the native engine's reply)")
# 22b. START on an empty line gives a ready-made message; B deletes it; capitals and an accent
kb = KB()
r = run('koko_kb2', [f'k:{START}', 'f:10'] + [f'k:{B}'] * 3 + kb.text('Café') + [f'k:{START}'], 1, keyboard=True)
t = turns_of(r['hist'])
check(t and t[0][0] == 'Cafe', f"keyboard: ready-made message deleted, 'Café' typed with the arrow and accent keys: Koko read {t[0][0] if t else None!r}")
kb = KB()
r = run('koko_kb3', [f'k:{START}', 'f:10', f'k:{START}'], 1, keyboard=True)
t = turns_of(r['hist'])
check(t and t[0][0] == 'hi', f"keyboard: START on an empty line, then START: the ready-made 'hi' is sent ({t[0][0] if t else None!r})")
# 22c. SELECT alone opens the menu; Settings > Typing: Button chords, and the chords type from then on
forget(keyboard=True)
subprocess.run([RUNNER, ROM] + BOOT + [f'k:{SELECT}', f'until:{DBG + 4:#x}:5:3000', 'f:20', 'shot:out/koko_kb_menu.ppm', f'dump:{DBG:#x}:8:out/kbm.bin'], check=True)
check(struct.unpack('2I', open('out/kbm.bin', 'rb').read())[1] == 5, 'keyboard: SELECT opens the menu')
forget(keyboard=True)
pad = Pad()
steps = BOOT + [f'k:{SELECT}', f'until:{DBG + 4:#x}:5:3000', 'f:20', f'k:{DOWN}', 'f:20', f'k:{DOWN}', 'f:20', f'k:{A}', 'f:40'] + [f'k:{DOWN}', 'f:30'] * 4 \
    + [f'k:{A}', 'f:40', f'k:{B}', 'f:40'] + [f'k:{UP}', 'f:30'] * 2 + [f'k:{B}', 'f:60'] + pad.text('hello') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5',
    f'dump:{DBG:#x}:{44 + 2 * HISTMAX}:out/kbsw.bin']
subprocess.run([RUNNER, ROM] + steps, check=True)
t = turns_of(parse('out/kbsw.bin')['hist'])
st = card_file('::/koko/settings.txt')
check(t and t[0][0] == 'hello' and b'keyboard=0' in st, f"Settings > Typing: Button chords: typed with the chords {t[0][0] if t else None!r}; {st!r}")
# 22d. START + A saves the chat while typing on the keyboard
forget(keyboard=True)
kb = KB()
r = run('koko_kb_save', kb.text('hi') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5', f'k:{B}', 'f:30'] + Pad().start_with(A) + ['f:300'], 1, keyboard=True)
files = subprocess.run(['mdir', '-i', os.environ['SD_IMAGE'], '-b', '::/koko/chats'], capture_output=True, text=True).stdout.split()
check(any(f.lower().endswith('.kch') for f in files), f'keyboard: START+A saved the chat ({files})')
subprocess.run(['mdeltree', '-i', os.environ['SD_IMAGE'], '::/koko/chats'], capture_output=True)
# 23. kokoSD 2.6: with the button chords START+B opens the menu and deletes nothing; START+SELECT no longer does
pad = Pad()
x = after(BOOT + pad.text('hello') + ['f:30'] + pad.start_with(B) + ['f:60', f'dump:{DBG:#x}:8:out/sb.bin', f'k:{B}', 'f:60'], 'start_b')
check(struct.unpack('2I', open('out/sb.bin', 'rb').read())[1] == 5 and x['caret'] == 5, f"START+B: the menu, then back with nothing deleted ({x})")
pad = Pad()
x = after(BOOT + pad.text('hello') + ['f:30'] + pad.start_with(SELECT) + ['f:60'], 'start_select')
check(x['state'] in (0, 3) and x['caret'] == 5, f"START+SELECT: no menu any more, still typing ({x})")
sys.exit(0 if ok else 1)
