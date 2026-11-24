/* Test ROM: run a fixed token sequence, leave checksums and cycle counts in memory for the test harness. */
#include "koko.h"
#include "test_tokens.h"
struct Dbg { uint32_t magic, done, tokens, cs_h[NTOK], cs_s[NTOK], cyc_step[NTOK], prof[KP_N], cs_t[NTOK], cyc_out[NTOK], nrows[NTOK]; };
volatile struct Dbg dbg __attribute__((section(".bss.dbg")));
void *memset(void *d, int c, unsigned long n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }
static uint32_t now(void) {
    volatile uint16_t *t = (volatile uint16_t *)0x04000100;
    uint16_t h = t[2], l = t[0];
    if (h != t[2]) { h = t[2]; l = t[0]; }
    return ((uint32_t)h << 16) | l;
}
static uint16_t rows[256] __attribute__((section(".bss.ewram")));
static int32_t out[256] __attribute__((section(".bss.ewram")));
int main(void) {
    *(volatile uint16_t *)0x04000204 = 0x4317;          /* fastest cartridge timing, prefetch on */
    *(volatile uint16_t *)0x04000000 = 0x0080;          /* screen off */
    *(volatile uint16_t *)0x04000106 = 0x0084;          /* TM1 counts TM0 overflows */
    *(volatile uint16_t *)0x04000102 = 0x0080;          /* TM0: one tick per CPU cycle */
    dbg.done = 0; dbg.tokens = 0;
    for (int i = 0; i < KP_N; i++) koko_prof[i] = 0;
    koko_reset();
    uint32_t rng = 12345;
    for (int i = 0; i < NTOK; i++) {
        uint32_t t0 = now();
        koko_step(TEST_TOKENS[i]);
        dbg.cyc_step[i] = now() - t0;
        uint32_t cs = 0;
        for (int d = 0; d < K_D; d++) cs = cs * 31 + (uint32_t)koko_h[d];
        dbg.cs_h[i] = cs;
        cs = 0;
#ifdef FULL_SCORES
        for (int b = 0; b < K_V; b += 256) {
            for (int r = 0; r < 256; r++) rows[r] = (uint16_t)(b + r);
            koko_scores(rows, 256, out);
            for (int r = 0; r < 256; r++) cs = cs * 31 + (uint32_t)out[r];
        }
#endif
        dbg.cs_s[i] = cs;
        {   /* fast output step: top-40 list and one sampled token */
            const uint16_t *ids; const int32_t *sc; int scored;
            uint32_t t1 = now();
            int n = koko_topn(&ids, &sc, &scored);
            dbg.cyc_out[i] = now() - t1;
            cs = (uint32_t)n;
            for (int k = 0; k < n; k++) cs = (cs * 31 + ids[k]) * 31 + (uint32_t)sc[k];
            cs = cs * 31 + (uint32_t)koko_pick(&rng, 512);
            dbg.cs_t[i] = cs; dbg.nrows[i] = (uint32_t)scored;
        }
        dbg.tokens = i + 1;
    }
    for (int i = 0; i < KP_N; i++) dbg.prof[i] = koko_prof[i];
    dbg.magic = 0x4B4F4B4F; dbg.done = 1;
    for (;;) {}
}
