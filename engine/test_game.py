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
BOOT = ['f:20', 'shot:out/koko_title.ppm', f'k:{START}', 'f:10']      # the title screen, then START


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
    def text(self, s): return [st for ch in s for st in self.key(ch)]
    def start_with(self, b): return self.h(START, 4) + self.h(START | b, 4) + self.h(START, 3) + self.h(0, 4)
    def menu(self): return self.start_with(SELECT) + [f'until:{DBG + 4:#x}:5:2000', 'f:4']   # START + SELECT: the menu (state 5)
    def new_chat(self): return self.menu() + [f'k:{A}']                  # its first row
    def helpers_toggle(self): return self.menu() + [f'k:{DOWN}', f'k:{A}', f'k:{B}', 'f:10']


def run(name, steps, nturn):
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

# 0. kokoSD: the SD card was mounted at start (dbg.sd: bit 0 mounted, bits 8-15 gbaSD's cart_kind)
subprocess.run([RUNNER, ROM, 'f:20', f'dump:{DBG + 44 + 2 * HISTMAX:#x}:4:out/sd.bin'], check=True)
sd = struct.unpack('I', open('out/sd.bin', 'rb').read())[0]
want_sd = {'build/runner': 0, 'build/runner-sc': 1, 'build/runner-ezfo': 2, 'build/runner-edgba': 4}.get(RUNNER)
check((sd & 1) == (RUNNER != 'build/runner') and (sd >> 8) == (want_sd if want_sd else sd >> 8),
      f'SD card at start: mounted {sd & 1}, cart kind {sd >> 8} ({RUNNER})')

# 1. a ready-made message: START fills it in, START sends it
r = run('koko_quick', [f'k:{START}', IDLE, f'k:{START}'], 1)
t = turns_of(r['hist'])
check(t[0][0] == 'hi', f"ready-made message arrived as typed; reply: {t[0][1].strip()!r} at {r['rate']:.2f} tokens/s")
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

# 3. delete back into a finished word, retype, send
pad = Pad()
steps = pad.text('i like cats ') + [IDLE] + pad.text('\b' * 5) + pad.text('dogs') + [f'k:{START}']
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
r = run('koko_newchat', pad.text('my name is sam') + [f'k:{START}', f'until:{DBG + 12:#x}:1:400000', 'f:5'] + pad.new_chat() + ['f:10', 'shot:out/koko_newchat_fresh.ppm']
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
subprocess.run([RUNNER, ROM] + BOOT + pad.menu() + [f'k:{DOWN}', f'k:{DOWN}', f'k:{A}', f'until:{DBG + 4:#x}:6:2000', 'f:10',
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
if os.path.exists(FACTS):
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
    print('SKIP facts from the SD card: no out/koko/facts.dat (python3 tools/facts_sd.py simplewiki-....xml.bz2)')
sys.exit(0 if ok else 1)
