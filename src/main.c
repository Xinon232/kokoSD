/* Tiny Stories for Game Boy Advance: prompt editor + story screen.
 * Text mode on BG0, 30x20 characters. No BIOS calls, no interrupts.
 */
#include <stdint.h>
#include "model.h"
#include "font.h"

#define REG16(a) (*(volatile uint16_t *)(a))
#define REG_DISPCNT  REG16(0x04000000)
#define REG_VCOUNT   REG16(0x04000006)
#define REG_BG0CNT   REG16(0x04000008)
#define REG_TM0CNT_L REG16(0x04000100)
#define REG_TM0CNT_H REG16(0x04000102)
#define REG_TM2CNT_L REG16(0x04000108)
#define REG_TM2CNT_H REG16(0x0400010A)
#define REG_TM3CNT_L REG16(0x0400010C)
#define REG_TM3CNT_H REG16(0x0400010E)
#define REG_KEYINPUT REG16(0x04000130)
#define REG_WAITCNT  REG16(0x04000204)
#define PAL   ((volatile uint16_t *)0x05000000)
#define TILES ((volatile uint32_t *)0x06000000)
#define MAP   ((volatile uint16_t *)0x0600F800)

enum { K_A = 1, K_B = 2, K_SELECT = 4, K_START = 8, K_RIGHT = 16, K_LEFT = 32,
       K_UP = 64, K_DOWN = 128, K_R = 256, K_L = 512 };
enum { P_TEXT, P_PROMPT, P_DIM, P_BAR, P_KEY, P_GREEN };
#define RGB(r, g, b) ((r) | ((g) << 5) | ((b) << 10))

/* The tests read this block from the emulator: it sits first in slow RAM (0x02000000). */
#define HISTMAX 2048
struct Dbg {
    uint32_t magic, state, seed, nprompt, ntok, ticks;
    uint16_t hist[HISTMAX];
};
enum { ST_EDIT = 0, ST_RUNNING = 1, ST_DONE = 2 };
volatile struct Dbg dbg __attribute__((section(".bss.dbg")));

void *memset(void *d, int c, unsigned long n) {
    uint8_t *p = d;
    while (n--) *p++ = (uint8_t)c;
    return d;
}
void *memcpy(void *d, const void *s, unsigned long n) {
    uint8_t *p = d;
    const uint8_t *q = s;
    while (n--) *p++ = *q++;
    return d;
}

/* ---------------- screen ---------------- */

static void put(int x, int y, int ch, int pal) {
    if (ch < 32 || ch > 127) ch = '?';
    MAP[y * 32 + x] = (uint16_t)((ch - 32) | (pal << 12));
}
static int print(int x, int y, const char *s, int pal) {
    while (*s) put(x++, y, *s++, pal);
    return x;
}
static void fill(int y0, int y1, int pal) {
    for (int y = y0; y <= y1; y++)
        for (int x = 0; x < 30; x++) put(x, y, ' ', pal);
}
static int print_num(int x, int y, uint32_t v, int pal) {
    char b[12];
    int n = 0;
    do { b[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) put(x++, y, b[--n], pal);
    return x;
}
static void vsync(void) {
    while (REG_VCOUNT >= 160) {}
    while (REG_VCOUNT < 160) {}
}
static void video_init(void) {
    static const uint16_t pal[][2] = {            /* {text, background} */
        { RGB(31, 31, 31), RGB(2, 3, 8) },        /* P_TEXT   */
        { RGB(31, 25, 8),  RGB(2, 3, 8) },        /* P_PROMPT */
        { RGB(15, 19, 27), RGB(2, 3, 8) },        /* P_DIM    */
        { RGB(2, 3, 8),    RGB(31, 25, 8) },      /* P_BAR    */
        { RGB(2, 3, 8),    RGB(31, 31, 31) },     /* P_KEY    */
        { RGB(12, 30, 16), RGB(2, 3, 8) },        /* P_GREEN  */
    };
    for (unsigned p = 0; p < sizeof pal / sizeof pal[0]; p++) {
        PAL[p * 16 + 1] = pal[p][0];
        PAL[p * 16 + 2] = pal[p][1];
    }
    PAL[0] = RGB(2, 3, 8);
    for (int g = 0; g < 96; g++)
        for (int r = 0; r < 8; r++) {
            uint32_t w = 0;
            for (int b = 0; b < 8; b++) w |= (uint32_t)((FONT[g][r] >> b) & 1 ? 1 : 2) << (b * 4);
            TILES[g * 8 + r] = w;
        }
    fill(0, 19, P_TEXT);
    REG_BG0CNT = 31 << 8;
    REG_DISPCNT = 0x0100;
}

static uint16_t keys_prev;
static uint16_t keys_down(void) {                 /* newly pressed since the last call */
    uint16_t k = (uint16_t)(~REG_KEYINPUT & 0x03FF), d = (uint16_t)(k & ~keys_prev);
    keys_prev = k;
    return d;
}

/* ---------------- story text: word wrap and scrolling ---------------- */

#define TOP 2
#define BOT 17
static int cx, cy, wstart;
static uint32_t u8cp;
static int u8need;

static void newline(void) {
    cx = 0; wstart = 0;
    if (cy < BOT) { cy++; return; }
    for (int y = TOP; y < BOT; y++)
        for (int x = 0; x < 30; x++) MAP[y * 32 + x] = MAP[(y + 1) * 32 + x];
    for (int x = 0; x < 30; x++) put(x, BOT, ' ', P_TEXT);
}
static void story_char(int ch, int pal) {
    if (ch == '\n') { if (cx) newline(); return; }
    if (ch == ' ') {
        if (cx == 0) return;
        if (cx < 30) put(cx++, cy, ' ', pal);
        wstart = cx;
        return;
    }
    if (cx >= 30) {                               /* carry the unfinished word to the next line */
        uint16_t word[30];
        int n = 0, y = cy;
        if (wstart > 0 && wstart < 30)
            for (int x = wstart; x < 30; x++) { word[n++] = MAP[y * 32 + x]; put(x, y, ' ', P_TEXT); }
        newline();
        for (int i = 0; i < n; i++) MAP[cy * 32 + cx++] = word[i];
    }
    put(cx++, cy, ch, pal);
}
/* Token bytes are UTF-8; the font is ASCII. Fold the few non-ASCII marks TinyStories uses. */
static void story_byte(int b, int pal) {
    if (b < 0x80) { u8need = 0; story_char(b, pal); return; }
    if (b >= 0xC0) { u8need = b >= 0xF0 ? 3 : b >= 0xE0 ? 2 : 1; u8cp = (uint32_t)b & (0x3F >> u8need); return; }
    if (!u8need) return;
    u8cp = (u8cp << 6) | (b & 0x3F);
    if (--u8need) return;
    switch (u8cp) {
    case 0x201C: case 0x201D: story_char('"', pal); break;
    case 0x2018: case 0x2019: story_char('\'', pal); break;
    case 0x2013: case 0x2014: story_char('-', pal); break;
    case 0x2026: story_char('.', pal); story_char('.', pal); story_char('.', pal); break;
    case 0xA0: story_char(' ', pal); break;
    case 0xE9: story_char('e', pal); break;
    case 0xF1: story_char('n', pal); break;
    default: break;
    }
}

/* ---------------- prompt editor ---------------- */

#define PROMPT_MAX 56
static char prompt[PROMPT_MAX + 1] = "Once upon a time";
static int plen = 16;
static const char *const ideas[] = {
    "Once upon a time", "One day, a little girl named Lily", "The big dog",
    "Tim and his mom went to the park", "",
};
static const char *const keyrow[3] = { "abcdefghijklm", "nopqrstuvwxyz", ".,!?'\"-:;()_<" };
static uint32_t seed = 0x2545F491;

static void bar(void) {
    fill(0, 0, P_BAR);
    print(1, 0, "TINY STORIES", P_BAR);
    print(19, 0, "260K on GBA", P_BAR);
}
static void draw_editor(int kx, int ky, int upper) {
    fill(4, 5, P_PROMPT);
    for (int i = 0; i < plen; i++) put(1 + i % 28, 4 + i / 28, prompt[i], P_PROMPT);
    if (plen < PROMPT_MAX) put(1 + plen % 28, 4 + plen / 28, '_', P_TEXT);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 13; c++) {
            int ch = keyrow[r][c];
            if (upper && r < 2) ch -= 32;
            int sel = (r == ky && c == kx);
            put(1 + c * 2, 8 + r * 2, ' ', sel ? P_KEY : P_TEXT);
            put(2 + c * 2, 8 + r * 2, ch, sel ? P_KEY : P_TEXT);
            put(3 + c * 2, 8 + r * 2, ' ', sel ? P_KEY : P_TEXT);
        }
}
static void edit_screen(void) {
    int kx = 0, ky = 0, upper = 0, idea = 0, rep = 0;
    fill(1, 19, P_TEXT);
    bar();
    print(1, 2, "Start a story:", P_DIM);
    print(1, 14, "_ space  < delete", P_DIM);
    print(1, 16, "A type  B delete  SELECT Aa", P_DIM);
    print(1, 17, "L/R story ideas", P_DIM);
    print(1, 19, "START", P_GREEN);
    print(7, 19, "write the story", P_TEXT);
    dbg.state = ST_EDIT;
    for (;;) {
        draw_editor(kx, ky, upper);
        vsync();
        uint16_t d = keys_down();
        uint16_t held = keys_prev & (K_LEFT | K_RIGHT | K_UP | K_DOWN | K_B);
        if (held && !d) { if (++rep > 14 && !(rep & 3)) d = held; } else rep = 0;
        if (d) seed = seed * 1664525u + REG_TM0CNT_L + 1013904223u;
        if (d & K_LEFT) kx = (kx + 12) % 13;
        if (d & K_RIGHT) kx = (kx + 1) % 13;
        if (d & K_UP) ky = (ky + 2) % 3;
        if (d & K_DOWN) ky = (ky + 1) % 3;
        if (d & K_SELECT) upper = !upper;
        int del = (d & K_B) != 0;
        if (d & K_A) {
            int ch = keyrow[ky][kx];
            if (ky == 2 && kx == 12) del = 1;
            else {
                if (ky == 2 && kx == 11) ch = ' ';
                else if (upper && ky < 2) ch -= 32;
                if (plen < PROMPT_MAX) { prompt[plen++] = (char)ch; prompt[plen] = 0; }
            }
        }
        if (del && plen) prompt[--plen] = 0;
        if (d & (K_L | K_R)) {
            idea = (idea + ((d & K_R) ? 1 : 4)) % 5;
            for (plen = 0; ideas[idea][plen]; plen++) prompt[plen] = ideas[idea][plen];
            prompt[plen] = 0;
        }
        if (d & K_START) return;
    }
}

/* ---------------- story screen ---------------- */

static uint32_t ticks(void) {                     /* 1/16384 s */
    uint16_t hi = REG_TM3CNT_L, lo = REG_TM2CNT_L;
    if (hi != REG_TM3CNT_L) { hi = REG_TM3CNT_L; lo = REG_TM2CNT_L; }
    return ((uint32_t)hi << 16) | lo;
}
static void status(int ntok, uint32_t t) {
    fill(19, 19, P_DIM);
    int x = print_num(0, 19, (uint32_t)ntok, P_DIM);
    x = print(x, 19, " tok", P_DIM);
    if (ntok && t) {
        uint32_t r = (uint32_t)ntok * 163840u / t;          /* tokens/s x 10 */
        x = print_num(x + 2, 19, r / 10, P_DIM);
        put(x++, 19, '.', P_DIM);
        x = print_num(x, 19, r % 10, P_DIM);
        print(x, 19, "/s", P_DIM);
    }
}
/* returns 1 to write another story from the same prompt, 0 to go back to the editor */
static int story_screen(void) {
    uint16_t *hist = (uint16_t *)dbg.hist;
    fill(1, 19, P_TEXT);
    bar();
    cx = 0; cy = TOP; wstart = 0; u8need = 0;
    for (int i = 0; i < plen; i++) story_char(prompt[i], P_PROMPT);
    print(23, 19, "B stop", P_DIM);

    if (!seed) seed = 1;
    int n = tok_encode(prompt, hist, 200);
    dbg.seed = seed; dbg.nprompt = (uint32_t)n; dbg.ntok = (uint32_t)n; dbg.ticks = 0;
    dbg.state = ST_RUNNING;

    REG_TM2CNT_H = 0; REG_TM3CNT_H = 0;
    REG_TM2CNT_L = 0; REG_TM3CNT_L = 0;
    REG_TM3CNT_H = 0x0084;                        /* count TM2 overflows */
    REG_TM2CNT_H = 0x0083;                        /* 16.78 MHz / 1024 */

    llm_reset();
    for (int i = 0; i + 1 < n; i++) llm_forward(hist[i], 0);
    int made = 0;
    uint32_t t0 = ticks();                        /* rate counts generated tokens only */
    while (n < HISTMAX) {
        llm_forward(hist[n - 1], 1);
        int t = llm_pick_sample(&seed);
        if (t == 1) break;                        /* the model ended its story */
        hist[n++] = (uint16_t)t;
        made++;
        const uint8_t *p;
        int len = tok_piece(t, &p);
        for (int i = 0; i < len; i++) story_byte(p[i], P_TEXT);
        uint32_t el = ticks() - t0;
        dbg.ntok = (uint32_t)n; dbg.ticks = el;
        status(made, el);
        print(23, 19, "B stop", P_DIM);
        if (keys_down() & K_B) break;
    }
    dbg.state = ST_DONE;
    print(16, 19, " A again B edit", P_GREEN);
    for (;;) {
        vsync();
        uint16_t d = keys_down();
        if (d) seed = seed * 1664525u + REG_TM0CNT_L + 1013904223u;
        if (d & K_A) return 1;
        if (d & (K_B | K_START)) return 0;
    }
}

int main(void) {
    REG_WAITCNT = 0x4317;                         /* fastest cartridge timing + prefetch */
    REG_TM0CNT_L = 0; REG_TM0CNT_H = 0x0080;      /* free-running, feeds the random seed */
    video_init();
    dbg.magic = 0x4C4C4D21;
    for (;;) {
        edit_screen();
        while (story_screen()) {}
    }
}
