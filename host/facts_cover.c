/* kokoSD: which source answers a fact question (tools/measure_facts.py). Mode 0: the cartridge table only; 1: the SD file first. GPL-3.0-or-later. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "koko.h"
int fact_reply(const char *msg, int n, uint16_t *ids, int max);
const uint8_t *koko_blob;
static uint8_t *fd; static long fsz; static int use_sd; static int last_src;
static uint32_t u32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
int sdf_ready = 1; uint32_t sdf_count;
int sdf_lookup(uint32_t hash, uint16_t *ids, int max) {
    if (!use_sd) return 0;
    uint32_t count = u32(fd + 8), index_at = u32(fd + 16), entry_at = u32(fd + 24 - 4 + 4) ;
    entry_at = u32(fd + 20);
    int lo = 0, hi = (int)count - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2; uint32_t h = u32(fd + index_at + 8 * m);
        if (h == hash) {
            const uint8_t *e = fd + entry_at + u32(fd + index_at + 8 * m + 4);
            int tl = e[6]; const uint8_t *p = e + 7 + tl; int cnt = *p++;
            if (cnt > max) return 0;
            uint32_t acc = 0; int nb = 0;
            for (int i = 0; i < cnt; i++) { while (nb < 14) { acc |= (uint32_t)*p++ << nb; nb += 8; } ids[i] = acc & 0x3FFF; acc >>= 14; nb -= 14; }
            last_src = 2; return cnt;
        }
        if (h < hash) lo = m + 1; else hi = m - 1;
    }
    return 0;
}
void sdf_forget(void) {}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) return 1;
    koko_blob = b;
    FILE *g = fopen(argv[2], "rb"); fseek(g, 0, SEEK_END); fsz = ftell(g); rewind(g); fd = malloc(fsz); fread(fd, 1, fsz, g);
    use_sd = atoi(argv[3]);
    char line[512]; uint16_t ids[64];
    while (fgets(line, sizeof line, stdin)) {
        int n = (int)strlen(line); if (n && line[n - 1] == '\n') n--;
        last_src = 1;
        int k = fact_reply(line, n, ids, 64);
        printf("%s\t", k ? (last_src == 2 ? "card" : "cart") : "none");
        for (int i = 0; i < k; i++) { const char *p; int l = koko_token_text(ids[i], &p); fwrite(p, 1, l, stdout); }
        printf("\n");
    }
    return 0;
}
