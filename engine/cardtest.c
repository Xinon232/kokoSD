/* kokoSD card test (built with tools/build.py --cardtest): how fast the cart
 * and SD card read the small, scattered pieces kokoSD will need (fact
 * lookups, table rows), and longer runs for comparison. Each test reads for
 * 3 seconds from random places in the card's data area (whole sectors,
 * nothing is written there); the results are shown and saved to
 * /koko-cardtest.txt. GPL-3.0-or-later. */
#include <stdint.h>
#include <string.h>
#include "ui.h"
#include "sd.h"
#include "cart_io.h"

#define REG16(a) (*(volatile uint16_t *)(a))
#define REG_DISPCNT REG16(0x04000000)
#define REG_VCOUNT REG16(0x04000006)
#define REG_KEYINPUT REG16(0x04000130)
#define REG_WAITCNT REG16(0x04000204)
#define REG_TM2CNT_L REG16(0x04000108)
#define REG_TM2CNT_H REG16(0x0400010A)
#define REG_TM3CNT_L REG16(0x0400010C)
#define REG_TM3CNT_H REG16(0x0400010E)
#define PAL ((volatile uint16_t *)0x05000000)
#define SECONDS 3u

static uint8_t buf[16 * 512] __attribute__((section(".bss.ewram"), aligned(4)));
static char out[1200] __attribute__((section(".bss.ewram")));
static int olen;

static uint32_t ticks(void) {                     /* 1/16384 s */
    uint16_t hi = REG_TM3CNT_L, lo = REG_TM2CNT_L;
    if (hi != REG_TM3CNT_L) { hi = REG_TM3CNT_L; lo = REG_TM2CNT_L; }
    return ((uint32_t)hi << 16) | lo;
}
static char *num(char *p, uint32_t v) {
    char t[12]; int n = 0;
    do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) *p++ = t[--n];
    return p;
}
static char *dec(char *p, uint32_t v100) {        /* v100 / 100 with two decimals */
    p = num(p, v100 / 100); *p++ = '.';
    *p++ = (char)('0' + v100 / 10 % 10); *p++ = (char)('0' + v100 % 10);
    return p;
}
static char *str(char *p, const char *s) { while (*s) *p++ = *s++; return p; }
static void say(const char *s, int n) {
    ui_say(L_NOTE, s, n); ui_redraw();
    if (olen + n + 1 < (int)sizeof out) { memcpy(out + olen, s, (size_t)n); olen += n; out[olen++] = '\n'; }
}

static uint32_t rng = 2463534242u;
static uint32_t next(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

/* Reads `count` sectors at a time for SECONDS: random places, or one after
 * another. Returns reads done; *el the ticks taken; *err reads that failed. */
static uint32_t run(uint32_t first, uint32_t span, unsigned count, int random, uint32_t *el, uint32_t *err) {
    uint32_t t0 = ticks(), n = 0, at = first + (next() % (span / 2));
    *err = 0;
    while (ticks() - t0 < SECONDS * 16384u) {
        uint32_t lba = random ? first + (next() % (span - count)) : at;
        if (cart_read(buf, lba, count)) (*err)++;
        at += count; if (at + count > first + span) at = first;
        n++;
    }
    *el = ticks() - t0;
    return n;
}

int main(void) {
    /* kokoSD 1.1.2: fastest cartridge timing only on an Omega / EverDrive (a SuperCard's SDRAM is too slow for it) */
    ui_init();
    for (int i = 0; i < UI_PAL_N; i++) PAL[i] = ui_pal[i];
    REG_DISPCNT = 0x0404;
    REG_TM2CNT_H = 0; REG_TM3CNT_H = 0; REG_TM2CNT_L = 0; REG_TM3CNT_L = 0;
    REG_TM3CNT_H = 0x0084; REG_TM2CNT_H = 0x0083;  /* 1/16384 s */
    ui_header("card test");
    ui_say(L_KOKO, "Card test: about 15 seconds.", 28); ui_redraw();
    char line[120], *p;
    int mounted = sd_mount();
    if (cart_kind >= CART_EZFO) REG_WAITCNT = 0x4317;
    if (!mounted) {
        p = str(line, "No SD card or no file system on it ("); p = str(p, sd_cart_name()); p = str(p, ").");
        say(line, (int)(p - line));
        for (;;) {}
    }
    p = str(line, sd_cart_name()); p = str(p, ", ");
    uint32_t first = (uint32_t)sd_fs.database, span = (uint32_t)(sd_fs.n_fatent - 2) * sd_fs.csize;
    p = str(p, sd_fs.fs_type == FS_EXFAT ? "exFAT, " : "FAT32, "); p = num(p, span / 2048); p = str(p, " MB");
    say(line, (int)(p - line));
    static const struct { unsigned count; int random; const char *what; } T[] = {
        {1, 1, "0.5 KB random"}, {8, 1, "4 KB random"},
        {16, 1, "8 KB random"}, {16, 0, "In a row"}};
    for (unsigned i = 0; i < sizeof T / sizeof T[0]; i++) {
        uint32_t el, err, n = run(first, span, T[i].count, T[i].random, &el, &err);
        uint32_t us = n ? (uint32_t)((uint64_t)el * 1000000u / 16384u / n) : 0;            /* per read */
        uint32_t kbs = el ? (uint32_t)((uint64_t)n * T[i].count * 512u * 16384u / el / 1024u) : 0;
        p = str(line, T[i].what); p = str(p, ": ");
        if (T[i].random) { p = dec(p, us / 10); p = str(p, " ms"); }
        else { p = num(p, kbs); p = str(p, " KB/s"); }
        if (err) { p = str(p, ", "); p = num(p, err); p = str(p, " failed"); }
        say(line, (int)(p - line));
    }
    FIL f;
    UINT w;
    int saved = f_open(&f, "/koko-cardtest.txt", FA_WRITE | FA_CREATE_ALWAYS) == FR_OK
                && f_write(&f, out, (UINT)olen, &w) == FR_OK && w == (UINT)olen;
    saved = f_close(&f) == FR_OK && saved;
    say(saved ? "Saved: /koko-cardtest.txt" : "Not saved (card write failed)", saved ? 25 : 29);
    for (;;) {}
}
