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
static int last_ok;

static int read_at(uint32_t at, void *buf, unsigned n) {
    UINT got;
    return f_lseek(&f, at) == FR_OK && f_read(&f, buf, n, &got) == FR_OK && got == n;
}
static uint32_t u32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

int sdf_open(void) {
    sdf_ready = 0;
    if (!sd_ready || f_open(&f, "/koko/facts.dat", FA_READ) != FR_OK) return 0;
    clmt[0] = sizeof clmt / sizeof clmt[0];
    f.cltbl = clmt;
    if (f_lseek(&f, CREATE_LINKMAP) != FR_OK) f.cltbl = 0;      /* very fragmented: ordinary seeks */
    uint8_t *h = (uint8_t *)page;
    if (!read_at(0, h, 36) || h[0] != 'K' || h[1] != 'O' || h[2] != 'K' || h[3] != 'O' || h[4] != 'F' || h[5] != 'A' || h[6] != 'C' || h[7] != '1')
        return 0;
    sdf_count = u32(h + 8); n_entries = u32(h + 12); index_at = u32(h + 16); entry_at = u32(h + 20); text_at = u32(h + 24);
    uint32_t check = u32(h + 28), x = 0x811C9DC5u;                 /* made for these weights' tokenizer? */
    for (int id = 3; id < 1024; id++) {
        const char *p; int l = koko_token_text(id, &p);
        for (int i = 0; i < l; i++) x = (x ^ (uint8_t)p[i]) * 0x01000193u;
    }
    if (x != check || !sdf_count) return 0;
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
    last_text = text_at + toff; last_len = tlen < sizeof text ? tlen : sizeof text - 1; last_ok = 1;
    return cnt;
}

int sdf_article(const char **t, const char **x) {
    if (!sdf_ready || !last_ok || !last_len || !read_at(last_text, text, last_len)) return 0;
    text[last_len] = 0;
    *t = title; *x = text;
    return (int)last_len;
}
void sdf_forget(void) { last_ok = 0; }

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
    if (article) { const char *t, *x; if (!sdf_article(&t, &x)) return 0; }
    return 1;
}
