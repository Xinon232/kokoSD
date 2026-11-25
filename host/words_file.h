/* kokoSD 2.4: the word memory read from facts.dat on the PC (as engine/facts_sd.c does on the card). */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static FILE *wf; static uint32_t w_n[2], w_at[2];
static int words_open(const char *path) {
    wf = fopen(path, "rb"); if (!wf) return 0;
    uint8_t h[64]; if (fread(h, 1, 64, wf) != 64 || memcmp(h + 36, "WRD1", 4)) return 0;
    memcpy(&w_n[1], h + 40, 4); memcpy(&w_at[1], h + 44, 4); memcpy(&w_n[0], h + 48, 4); memcpy(&w_at[0], h + 52, 4);
    return 1;
}
static uint32_t words_hash(int order, const uint16_t *t) {
    uint32_t h = 0x811C9DC5u; h = (h ^ (uint32_t)order) * 0x01000193u;
    for (int i = 0; i < order; i++) { h = (h ^ (t[i] & 255u)) * 0x01000193u; h = (h ^ (t[i] >> 8)) * 0x01000193u; }
    return h;
}
static int words_find(int k, uint32_t hash, uint16_t *ids, uint8_t *sh) {   /* k: 1 = 3 tokens, 0 = 2 */
    uint32_t n = w_n[k]; if (!n) return 0;
    /* interpolation search over the sectors (32 sorted records each): about 2-3 reads */
    uint32_t pages = (n + 31) / 32, lo = 0, hi = pages - 1, pg = (uint32_t)(((uint64_t)hash * n) >> 32) / 32;
    uint64_t hlo = 0, hhi = 0xFFFFFFFFu;
    uint8_t sec[512];
    for (int tries = 0; tries < 24; tries++) {
        if (pg < lo) pg = lo;
        if (pg > hi) pg = hi;
        uint32_t m = n - pg * 32 < 32 ? n - pg * 32 : 32;
        fseek(wf, (long)(w_at[k] + pg * 512), SEEK_SET);
        if (fread(sec, 16, m, wf) != m) return 0;
        uint32_t first, last; memcpy(&first, sec, 4); memcpy(&last, sec + 16 * (m - 1), 4);
        if (hash < first) { if (pg == lo) return 0; hi = pg - 1; hhi = first; }
        else if (hash > last) { if (pg == hi) return 0; lo = pg + 1; hlo = last; }
        else {
            for (uint32_t i = 0; i < m; i++) {
                uint32_t h; memcpy(&h, sec + 16 * i, 4);
                if (h == hash) {
                    int c = 0;
                    for (int j = 0; j < 4; j++) { uint16_t id; memcpy(&id, sec + 16 * i + 4 + 2 * j, 2); uint8_t p = sec[16 * i + 12 + j]; if (p) { ids[c] = id; sh[c++] = p; } }
                    return c;
                }
            }
            return 0;
        }
        pg = lo + (uint32_t)((hash - hlo) * (uint64_t)(hi - lo + 1) / (hhi - hlo + 1));
    }
    return 0;
}
static int words_lookup(const uint16_t *c3, uint16_t *ids, uint8_t *sh) {
    for (int i = 0; i < 3; i++) if (c3[i] < 3 && !(i == 0 && c3[0] == 0xFFFF)) return 0;   /* turn markers: none */
    int n = 0;
    if (c3[0] != 0xFFFF && c3[0] >= 3) n = words_find(1, words_hash(3, c3), ids, sh);
    if (!n) n = words_find(0, words_hash(2, c3 + 1), ids, sh);
    return n;
}
