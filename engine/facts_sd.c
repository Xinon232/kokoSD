/* kokoSD: the fact file on the SD card. GPL-3.0-or-later.
 * The index is sorted by hash and the hashes are even, so a lookup starts at
 * the index sector the hash points to (hash * count / 2^32 / 64) and moves at
 * most a sector or two: with the entry, two or three card reads. The file is
 * opened once; a link map (FatFs fast seek) makes every seek free of FAT reads. */
#include "facts_sd.h"
#include "sd.h"
#include "koko.h"

#define SD __attribute__((section(".bss.sd")))
int sdf_ready;
uint32_t sdf_count;
static FIL f SD;
static DWORD clmt[96] SD;                   /* link map: room for 47 fragments */
static uint32_t n_entries, index_at, entry_at, text_at;
static uint32_t page[128] SD;               /* one index sector: 64 x (hash, entry) */
static uint8_t ent[192] SD;
static char title[64] SD, text[6200] SD;
static uint32_t last_text, last_len;
static uint32_t win_at, win_n;              /* kokoSD 2.3: the part of the article in text[] */
static int last_ok;
static uint32_t w_n[2], w_at[2];             /* kokoSD 2.4: the word memory's tables (2 and 3 tokens) */

static int read_at(uint32_t at, void *buf, unsigned n) {
    UINT got;
    return f_lseek(&f, at) == FR_OK && f_read(&f, buf, n, &got) == FR_OK && got == n;
}
static uint32_t u32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

int sdf_err;   /* kokoSD 1.4.2: why the fact file is not used: 0 it is, 1 no card, 2 f_open (+ 16 * FRESULT),
               * 3 the header could not be read, 4 not a fact file, 5 made for other weights, 6 empty */
int sdf_open(void) {
    sdf_ready = 0; sdf_err = 1;
    if (!sd_ready) return 0;
    FRESULT r = f_open(&f, "/koko/facts.dat", FA_READ);
    if (r != FR_OK) { sdf_err = 2 + 16 * (int)r; return 0; }
    clmt[0] = sizeof clmt / sizeof clmt[0];
    f.cltbl = clmt;
    if (f_lseek(&f, CREATE_LINKMAP) != FR_OK) f.cltbl = 0;      /* very fragmented: ordinary seeks */
    uint8_t *h = (uint8_t *)page;
    if (!read_at(0, h, 56)) { sdf_err = 3; return 0; }
    if (h[36] == 'W' && h[37] == 'R' && h[38] == 'D' && h[39] == '1') {   /* kokoSD 2.4: the word memory */
        w_n[1] = u32(h + 40); w_at[1] = u32(h + 44); w_n[0] = u32(h + 48); w_at[0] = u32(h + 52);
    } else w_n[0] = w_n[1] = 0;
    if (h[0] != 'K' || h[1] != 'O' || h[2] != 'K' || h[3] != 'O' || h[4] != 'F' || h[5] != 'A' || h[6] != 'C' || h[7] != '1') { sdf_err = 4; return 0; }
    sdf_count = u32(h + 8); n_entries = u32(h + 12); index_at = u32(h + 16); entry_at = u32(h + 20); text_at = u32(h + 24);
    uint32_t check = u32(h + 28), x = 0x811C9DC5u;                 /* made for these weights' tokenizer? */
    for (int id = 3; id < 1024; id++) {
        const char *p; int l = koko_token_text(id, &p);
        for (int i = 0; i < l; i++) x = (x ^ (uint8_t)p[i]) * 0x01000193u;
    }
    if (x != check) { sdf_err = 5; return 0; }
    if (!sdf_count) { sdf_err = 6; return 0; }
    sdf_err = 0;
    return sdf_ready = 1;
}

int sdf_lookup(uint32_t hash, uint16_t *ids, int max) {
    if (!sdf_ready) return 0;
    uint32_t pages = (sdf_count + 63) / 64, pg = (uint32_t)(((uint64_t)hash * sdf_count) >> 32) / 64, off = 0;
    int found = 0;
    for (int tries = 0; tries < 8 && pg < pages; tries++) {
        unsigned n = sdf_count - pg * 64 < 64 ? sdf_count - pg * 64 : 64;
        if (!read_at(index_at + pg * 512, page, n * 8)) return 0;
        if (hash < page[0]) { if (!pg) break; pg--; continue; }
        if (hash > page[2 * (n - 1)]) { pg++; continue; }
        int lo = 0, hi = (int)n - 1;
        while (lo <= hi) {
            int m = (lo + hi) >> 1;
            if (page[2 * m] == hash) { off = page[2 * m + 1]; found = 1; break; }
            if (page[2 * m] < hash) lo = m + 1; else hi = m - 1;
        }
        break;
    }
    if (!found || !read_at(entry_at + off, ent, 4)) return 0;
    UINT got;                                                         /* the entry: header, title, tokens */
    if (f_read(&f, ent + 4, sizeof ent - 4, &got) != FR_OK || got < 4) return 0;
    uint32_t toff = u32(ent);
    unsigned tlen = ent[4] | ent[5] << 8, tl = ent[6];
    const uint8_t *p = ent + 7 + tl;
    int cnt = *p++;
    if (cnt > max || 8 + tl + (cnt * 14 + 7) / 8 > 4 + got) return 0;
    uint32_t acc = 0; int nb = 0;
    for (int i = 0; i < cnt; i++) {
        while (nb < 14) { acc |= (uint32_t)*p++ << nb; nb += 8; }
        ids[i] = (uint16_t)(acc & 0x3FFF); acc >>= 14; nb -= 14;
    }
    for (unsigned i = 0; i < tl && i < sizeof title - 1; i++) title[i] = (char)ent[7 + i];
    title[tl < sizeof title - 1 ? tl : sizeof title - 1] = 0;
    last_text = text_at + toff; last_len = tlen; last_ok = 1; win_n = 0;   /* kokoSD 2.3: the whole article */
    return cnt;
}

/* kokoSD 2.3: the article is read page by page (up to 64,000 characters: the whole of nearly every article), not
 * all at once into memory (6,000 at most). */
int sdf_article(const char **t) {
    if (!sdf_ready || !last_ok || !last_len) return 0;
    *t = title;
    return (int)last_len;
}
const char *sdf_article_text(uint32_t at, uint32_t n) {
    if (at + n > last_len || n > sizeof text - 1) return 0;
    if (!win_n || at < win_at || at + n > win_at + win_n) {
        uint32_t m = last_len - at < sizeof text - 1 ? last_len - at : sizeof text - 1;
        if (!read_at(last_text + at, text, m)) { win_n = 0; return 0; }
        win_at = at; win_n = m; text[m] = 0;
    }
    return text + (at - win_at);
}
void sdf_forget(void) { last_ok = 0; }

/* kokoSD 2.4: the word memory (tools/words_sd.py): for the last 3 tokens (or 2), up to 4 tokens that followed them
 * in Simple English Wikipedia and their shares. 16-byte records sorted by hash, 32 to a sector; an interpolation
 * search finds the sector in about 2-3 card reads. */
int sdf_words_ready(void) { return sdf_ready && w_n[0] && w_n[1]; }
static uint32_t words_hash(int order, const uint16_t *t) {
    uint32_t h = (0x811C9DC5u ^ (uint32_t)order) * 0x01000193u;
    for (int i = 0; i < order; i++) { h = (h ^ (t[i] & 255u)) * 0x01000193u; h = (h ^ (uint32_t)(t[i] >> 8)) * 0x01000193u; }
    return h;
}
static int words_find(int k, uint32_t hash, uint16_t *ids, uint8_t *sh) {
    uint32_t n = w_n[k];
    if (!n) return 0;
    uint32_t pages = (n + 31) / 32, lo = 0, hi = pages - 1, pg = (uint32_t)(((uint64_t)hash * n) >> 32) / 32;
    uint64_t hlo = 0, hhi = 0xFFFFFFFFu;
    for (int tries = 0; tries < 24; tries++) {
        if (pg < lo) pg = lo;
        if (pg > hi) pg = hi;
        uint32_t m = n - pg * 32 < 32 ? n - pg * 32 : 32;
        if (!read_at(w_at[k] + pg * 512, page, m * 16)) return 0;
        uint32_t first = page[0], last = page[4 * (m - 1)];
        if (hash < first) { if (pg == lo) return 0; hi = pg - 1; hhi = first; }
        else if (hash > last) { if (pg == hi) return 0; lo = pg + 1; hlo = last; }
        else {
            for (uint32_t i = 0; i < m; i++)
                if (page[4 * i] == hash) {
                    const uint8_t *r = (const uint8_t *)&page[4 * i];
                    int c = 0;
                    for (int j = 0; j < 4; j++) if (r[12 + j]) { ids[c] = (uint16_t)(r[4 + 2 * j] | r[5 + 2 * j] << 8); sh[c++] = r[12 + j]; }
                    return c;
                }
            return 0;
        }
        pg = lo + (uint32_t)((hash - hlo) * (uint64_t)(hi - lo + 1) / (hhi - hlo + 1));
    }
    return 0;
}
int sdf_words(const uint16_t *c3, uint16_t *ids, uint8_t *sh) {
    if (!sdf_ready) return 0;
    for (int i = 1; i < 3; i++) if (c3[i] < 3) return 0;      /* turn markers: no context */
    int n = 0;
    if (c3[0] != 0xFFFF && c3[0] >= 3) n = words_find(1, words_hash(3, c3), ids, sh);
    if (!n) n = words_find(0, words_hash(2, c3 + 1), ids, sh);
    return n;
}

/* kokoSD 1.4 self test: look up a title picked by r (a real entry of the index) and, when article is set, read its
 * article as Tell me more does. 1: found and read back as it should be. */
int sdf_probe(uint32_t r, int article) {
    if (!sdf_ready) return 0;
    uint32_t pages = (sdf_count + 63) / 64, pg = r % pages;
    unsigned n = sdf_count - pg * 64 < 64 ? sdf_count - pg * 64 : 64;
    if (!read_at(index_at + pg * 512, page, n * 8)) return 0;
    uint32_t hash = page[2 * ((r >> 12) % n)];
    static uint16_t ids[255] SD;
    if (!sdf_lookup(hash, ids, 255)) return 0;        /* a title of the index must be found */
    if (article) {                                    /* its first and last part, as Tell me more reads them */
        const char *t; uint32_t n = (uint32_t)sdf_article(&t), k = n < 512 ? n : 512;
        if (!n || !sdf_article_text(0, k) || !sdf_article_text(n - k, k)) return 0;
    }
    return 1;
}
