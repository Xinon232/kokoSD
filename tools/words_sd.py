"""kokoSD 2.4: Koko's word memory: which tokens follow which in Simple English Wikipedia, for the SD card.

    python3 tools/words_sd.py out/koko/facts.dat [PAGES.pkl]

From the articles of the fact file's parse cache (tools/facts_sd.py) tokenised with Koko's own tokenizer
(host/tok_stream.c): for each context of 3 tokens (and, as a fallback, of 2) that occurs at least MIN_COUNT
times, the 4 tokens that most often came next and their share. Appended to facts.dat as a section of 16-byte
records sorted by a hash of the context (32 to a card sector: one sector read per lookup); the header's bytes
36-55 say where ("WRD1", count and offset of the 3-token table, then of the 2-token table). 2 % of the
articles (by title hash) are kept out and saved as out/words-heldout.u16 for tools/words_eval.py.
Records: hash u32, 4 token ids u16, 4 shares u8 (share * 255). Hash: FNV-1a over the order byte (2 or 3) and the
context's tokens (low byte first). GPL-3.0-or-later."""
import pickle, struct, subprocess, sys, zlib
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
MIN_COUNT = 4
SEP = 0xFFFF


def fnv_ctx(order, toks):
    h = 0x811C9DC5
    for c in [order] + [b for t in toks for b in (t & 255, t >> 8)]:
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h


def fnv_vec(order, cols):
    """The same hash for many contexts at once (cols: one array per context position)."""
    h = np.full(len(cols[0]), 0x811C9DC5, np.uint64)
    M = np.uint64(0x01000193); F = np.uint64(0xFFFFFFFF)
    def step(h, c): return ((h ^ c.astype(np.uint64)) * M) & F
    h = step(h, np.full(len(h), order, np.uint64))
    for t in cols:
        h = step(h, t & 255); h = step(h, t >> 8)
    return h.astype(np.uint32)


def table(t, order):
    """Records (sorted by hash) for contexts of `order` tokens in the stream t."""
    n = len(t)
    ok = np.ones(n - order, bool)
    for k in range(order + 1): ok &= t[k:n - order + k] != SEP
    idx = np.nonzero(ok)[0]
    ctx = [t[idx + k].astype(np.uint64) for k in range(order)]
    nxt = t[idx + order].astype(np.uint64)
    key = np.zeros(len(idx), np.uint64)
    for c in ctx: key = (key << np.uint64(14)) | c
    full = (key << np.uint64(14)) | nxt
    del ctx, nxt, key
    u, cnt = np.unique(full, return_counts=True)
    del full
    ck = u >> np.uint64(14); nx = (u & np.uint64(0x3FFF)).astype(np.uint16)
    starts = np.r_[0, np.nonzero(np.diff(ck))[0] + 1]
    tot = np.add.reduceat(cnt, starts)
    keep = tot >= MIN_COUNT
    recs = []
    S, T = starts[keep], tot[keep]
    ends = np.r_[starts[1:], len(u)][keep]
    hashes = None
    for s, e, tt in zip(S, E := ends, T):
        c = cnt[s:e]; o = np.argsort(-c, kind='stable')[:4]
        recs.append((int(ck[s]), [int(nx[s + j]) for j in o], [int(min(255, round(255 * c[j] / tt))) for j in o]))
    # hash each context
    cols = []
    keys = np.array([r[0] for r in recs], np.uint64)
    for k in range(order):
        cols.append(((keys >> np.uint64(14 * (order - 1 - k))) & np.uint64(0x3FFF)).astype(np.uint64))
    hs = fnv_vec(order, cols)
    best = {}
    for h, (k, ids, ps), tt in zip(hs.tolist(), recs, T.tolist()):
        if h not in best or best[h][0] < tt: best[h] = (tt, ids, ps)
    out = bytearray()
    for h in sorted(best):
        _, ids, ps = best[h]
        ids = ids + [0] * (4 - len(ids)); ps = ps + [0] * (4 - len(ps))
        out += struct.pack('<I4H4B', h, *ids, *ps)
    return out, len(best)


def main():
    facts = Path(sys.argv[1])
    pkl = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(str(facts) + '.pages.pkl')
    arts, _ = pickle.load(open(pkl, 'rb'))
    train, held = [], []
    for k, (title, s, text) in arts.items():
        paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 20 and not p.startswith('- ')]
        (held if zlib.crc32(k.encode()) % 50 == 0 else train).extend(paras)
    exe = R/'build/tok_stream'
    if not exe.exists():
        subprocess.run(['gcc', '-O2', f'-I{R/"gen"}', f'-I{R/"engine"}', '-o', str(exe), str(R/'host/tok_stream.c'),
                        str(R/'engine/koko.c'), str(R/'engine/koko_tok.c'), '-lm'], check=True)
    tok = lambda ps: np.frombuffer(subprocess.run([str(exe), str(R/'build/koko.bin')], input=('\n'.join(ps) + '\n').encode(),
                                                  capture_output=True, check=True).stdout, '<u2')
    th = tok(held); th.tofile(R/'out/words-heldout.u16')
    t = tok(train).astype(np.uint32)
    print(f'{len(train)} paragraphs, {len(t) / 1e6:.1f} M tokens (held out: {len(held)} paragraphs, {len(th) / 1e6:.2f} M tokens)', flush=True)
    t3, n3 = table(t, 3); print(f'3-token contexts: {n3}', flush=True)
    t2, n2 = table(t, 2); print(f'2-token contexts: {n2}', flush=True)
    b = bytearray(facts.read_bytes())
    if b[36:40] == b'WRD1':                                  # (made before: cut the old section off)
        b = b[:struct.unpack_from('<I', b, 44)[0]]
    b += bytes(-len(b) % 512)
    at3 = len(b); b += t3; b += bytes(-len(b) % 512)
    at2 = len(b); b += t2
    b[36:56] = b'WRD1' + struct.pack('<4I', n3, at3, n2, at2)
    facts.write_bytes(b)
    print(f'{facts}: word memory {len(t3) / 1e6:.1f} + {len(t2) / 1e6:.1f} MB, file {len(b) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
