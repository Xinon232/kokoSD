"""kokoSD's fact file for the SD card: every usable Simple English Wikipedia
article, for "who is / what is" answers (the opening sentence, as Koko's
tokens) and "Tell me more" (the article as plain text). Text: Simple English
Wikipedia, CC BY-SA 4.0 (attribution in CREDITS). GPL-3.0-or-later.

    python3 tools/facts_sd.py simplewiki-latest-pages-articles.xml.bz2 [out/koko/facts.dat]

The answers follow engine/facts_build.py (the cartridge's own table): the
same title key (norm), hash (FNV-1a), sentence rules (first_sentence) and
words never used as keys (STOP); without the ROM's size limit every article
with a usable opening sentence is kept, and redirects ("USA") become aliases.

File (little-endian; each part starts on a 512-byte sector):
  header   "KOKOFAC1", u32 count, u32 entries, u32 index_at, u32 entry_at,
           u32 text_at, u32 tokcheck (FNV-1a of the token texts 3..1023,
           so a file made for other weights is refused), u32 text_bytes
  index    count x (u32 hash, u32 entry offset), sorted by hash: 64 per sector;
           the hashes are even, so a lookup starts at sector hash*count/2^32/64
  entries  u32 text offset, u16 text length, u8 title length, title,
           u8 token count, 14-bit token ids packed (as the ROM's table)
  text     the articles: ASCII, paragraphs separated by blank lines"""
import bz2, re, struct, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R/'engine'))
from facts_build import STOP, norm, fnv, ascii_fold, first_sentence
import mwparserfromhell as mw

TEXT_MAX = 64000                                  # characters of an article kept for Tell me more (kokoSD 2.3: the
                                                  # GBA reads it page by page; the entry's length field is 16 bits)
ABBREV = re.compile(r'(?:\b(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|Prof|Gen|Lt|Col|Capt|Sgt|Rev|Mt|Ft|No|Vol|vs|etc|Inc|Ltd|Co|Corp|U\.S|U\.K|e\.g|i\.e|[A-Z])\.)$')


def sentence_cut(t, limit):
    """The end of the last whole sentence before `limit` (not after an abbreviation such as Dr. or an initial)."""
    best = -1
    for m in re.finditer(r'[.!?]["\')]?(?=\s)', t[:limit]):
        if m.group(0)[0] == '.' and ABBREV.search(t[max(0, m.start() - 6):m.start() + 1]):
            continue
        best = m.end()
    return best

DROP_SECTIONS = re.compile(r'^(references|related pages|other websites|notes|sources|further reading|gallery|external links|see also|bibliography|footnotes)$', re.I)


def plain(wikitext):
    """Wikitext -> paragraphs of plain ASCII text (no tables, references, files, categories)."""
    code = mw.parse(wikitext)
    for t in code.filter_tags(recursive=False):
        if str(t.tag).lower() in ('ref', 'table', 'gallery', 'math', 'references', 'syntaxhighlight', 'timeline'):
            try: code.remove(t)
            except ValueError: pass
    for tpl in code.filter_templates(recursive=False):
        try: code.remove(tpl)
        except ValueError: pass
    for l in code.filter_wikilinks():
        if re.match(r'\s*(file|image|category|media)\s*:', str(l.title), re.I):
            try: code.remove(l)
            except ValueError: pass
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.S)           # tables the parser left
    out, skip = [], False
    for line in text.split('\n'):
        line = line.strip()
        h = re.fullmatch(r'=+\s*(.*?)\s*=+', line)
        if h:
            skip = bool(DROP_SECTIONS.match(h.group(1)))
            if not skip: out.append('\n' + h.group(1))
            continue
        if skip or not line: continue
        if line.startswith(('*', '#')): line = '- ' + line.lstrip('*#: ')
        out.append(line)
    t = '\n'.join(out)
    t = t.replace('–', '-').replace('—', '-').replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace(' ', ' ')
    t = ascii_fold(t)
    t = ''.join(c if 32 <= ord(c) < 127 or c == '\n' else '' for c in t)
    t = re.sub(r'\(\s*[,;]?\s*\)', '', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r' +([,.;:!?)])', r'\1', t)                     # a space left before a comma by a removed template
    t = re.sub(r'\( +', '(', t)
    t = re.sub(r'\n{3,}', '\n\n', t).strip()
    if len(t) > TEXT_MAX:
        cut = sentence_cut(t, TEXT_MAX)
        t = t[:cut if cut > TEXT_MAX // 2 else TEXT_MAX].rstrip() + ' ...'
    return t


def pages(path):
    ns = None
    for ev, el in ET.iterparse(bz2.open(path), events=('end',)):
        if ns is None: ns = el.tag[:el.tag.index('}') + 1] if el.tag.startswith('{') else ''
        if el.tag == ns + 'page':
            if el.findtext(ns + 'ns') == '0':
                red = el.find(ns + 'redirect')
                yield el.findtext(ns + 'title') or '', (red.get('title') if red is not None else None), el.findtext(f'{ns}revision/{ns}text') or ''
            el.clear()


def tokenize(sentences):
    """Koko's own tokenizer (engine/koko_tok.c on build/koko.bin): ids, and whether they decode to the text."""
    exe = R/'build/tok_test'
    if not exe.exists():
        subprocess.run(['gcc', '-O2', f'-I{R/"gen"}', f'-I{R/"engine"}', '-o', str(exe), str(R/'host/tok_test.c'),
                        str(R/'engine/koko.c'), str(R/'engine/koko_tok.c'), '-lm'], check=True)
    out = subprocess.run([str(exe), str(R/'build/koko.bin')], input='\n'.join(sentences).encode(), capture_output=True, check=True).stdout.decode().split('\n')
    res = []
    for s, line in zip(sentences, out):
        ids, _, dec = line.partition('|')
        res.append(([int(x) for x in ids.split()], dec == s))
    return res


def tokcheck():
    out = subprocess.run([str(R/'build/tok_dump'), str(R/'build/koko.bin')], capture_output=True, check=True).stdout.decode().split('\n')
    h = 0x811C9DC5
    for i in range(3, 1024):
        for c in bytes.fromhex(out[i]): h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h


def build(src, dst):
    import pickle
    cache = Path(str(dst) + '.pages.pkl')            # the parsed dump (40 minutes): later runs reuse it
    dst.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_mtime > Path(src).stat().st_mtime:
        arts, redirects = pickle.load(open(cache, 'rb'))
    else:
        arts, redirects = parse(src)
        pickle.dump((arts, redirects), open(cache, 'wb'))
    return write(arts, redirects, dst)


def parse(src):
    arts, redirects = {}, []
    for n, (title, target, wt) in enumerate(pages(src)):
        if n % 20000 == 0: print(f'  {n} pages, {len(arts)} articles', flush=True)
        t = ascii_fold(title)
        if '(' in t or ':' in t or any(ord(c) > 126 for c in t) or re.fullmatch(r'[\d ]+|.*\b(List of|Lists of)\b.*', t):
            continue
        key = norm(t)
        if not key or key in STOP or len(key) < 2 or key.isdigit():
            continue
        if target is not None:
            redirects.append((key, norm(ascii_fold(target)))); continue
        text = plain(wt)
        s = first_sentence(text, t)
        if s is None: continue
        if key not in arts or len(text) > len(arts[key][2]):
            arts[key] = (t, s, text)
    print(f'{len(arts)} articles with an opening sentence, {len(redirects)} redirects', flush=True)
    return arts, redirects


def write(arts, redirects, dst):
    keys = sorted(arts)
    toks = tokenize([arts[k][1] for k in keys])
    entries, rows, text_parts, text_at, ent = bytearray(), [], [], 0, {}
    for k, (ids, same) in zip(keys, toks):
        if not same or len(ids) > 60: continue
        title, s, body = arts[k]
        lines = body.split('\n')                      # kokoSD 2.3: no empty section names left at the end
        while len(lines) > 1 and DROP_SECTIONS.match(lines[-1].strip()): lines.pop()
        tb = '\n'.join(lines).rstrip().encode()
        e = bytearray(struct.pack('<IHB', text_at, len(tb), len(title.encode()[:60]))) + title.encode()[:60]
        e.append(len(ids)); acc = nb = 0
        for i in ids:
            acc |= i << nb; nb += 14
            while nb >= 8: e.append(acc & 255); acc >>= 8; nb -= 8
        if nb: e.append(acc & 255)
        ent[k] = len(entries); entries += e; text_parts.append(tb); text_at += len(tb)
    hashes = {}
    for k, off in ent.items(): hashes.setdefault(fnv(k), off)
    alias = 0
    for k, target in redirects:
        if k in ent or target not in ent or fnv(k) in hashes: continue
        hashes[fnv(k)] = ent[target]; alias += 1
    index = b''.join(struct.pack('<II', h, o) for h, o in sorted(hashes.items()))
    pad = lambda b: b + bytes(-len(b) % 512)
    index_at = 512; entry_at = index_at + len(pad(index)); tat = entry_at + len(pad(bytes(entries)))
    text = b''.join(text_parts)
    head = b'KOKOFAC1' + struct.pack('<7I', len(hashes), len(ent), index_at, entry_at, tat, tokcheck(), len(text))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(pad(head) + pad(index) + pad(bytes(entries)) + text)
    print(f'{dst}: {len(ent)} articles + {alias} redirects = {len(hashes)} keys, {dst.stat().st_size / 1e6:.1f} MB '
          f'(text {len(text) / 1e6:.1f} MB)')
    return arts, ent


if __name__ == '__main__':
    arts, ent = build(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else R/'out/koko/facts.dat')
    for k in ('george washington', 'moon', 'dog', 'france', 'albert einstein', 'volcano', 'siberian husky', 'game boy advance', 'pizza', 'photosynthesis'):
        print(f'  {k:18s} {arts[k][1] if k in ent else None}')
