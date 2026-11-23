/* Koko: chat screen and on-screen keyboard. Text mode on BG0, 30x20 characters. No BIOS calls, no interrupts. */
#include <stdint.h>
#include "koko.h"
#include "font.h"
int sum_reply(const char *s, int n, char *out);   /* engine/sums.c */

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

enum { BTN_A = 1, BTN_B = 2, BTN_SELECT = 4, BTN_START = 8, BTN_RIGHT = 16, BTN_LEFT = 32,
       BTN_UP = 64, BTN_DOWN = 128, BTN_R = 256, BTN_L = 512 };
enum { P_TEXT, P_USER, P_DIM, P_BAR, P_KEY, P_GREEN };
#define RGB(r, g, b) ((r) | ((g) << 5) | ((b) << 10))
enum { TOK_END = 0, TOK_USER = 1, TOK_KOKO = 2 };
#define INV_TEMP 512                    /* temperature 0.5 */
#define REPLY_MAX 120

/* The tests read this block from the emulator: it sits first in slow RAM (0x02000000). */
#define HISTMAX 1024
struct Dbg {
    uint32_t magic, state, seed, turns, ntok, ticks, made, nsug, rewinds;
    uint16_t sug[2], pad[2];
    uint16_t hist[HISTMAX];             /* every token the model has taken in (and not taken back), in order */
};
enum { ST_EDIT = 0, ST_RUNNING = 1, ST_DONE = 2, ST_IDLE = 3 };   /* IDLE: editing, model caught up, suggestions shown */
volatile struct Dbg dbg __attribute__((section(".bss.dbg")));

void *memset(void *d, int c, unsigned long n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }

/* ---------------- screen ---------------- */
static void put(int x, int y, int ch, int pal) {
    if (ch < 32 || ch > 127) ch = '?';
    MAP[y * 32 + x] = (uint16_t)((ch - 32) | (pal << 12));
}
static int print(int x, int y, const char *s, int pal) { while (*s) put(x++, y, *s++, pal); return x; }
static void fill(int y0, int y1, int pal) {
    for (int y = y0; y <= y1; y++)
        for (int x = 0; x < 30; x++) put(x, y, ' ', pal);
}
static int print_num(int x, int y, uint32_t v, int pal) {
    char b[12]; int n = 0;
    do { b[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) put(x++, y, b[--n], pal);
    return x;
}
static void vsync(void) { while (REG_VCOUNT >= 160) {} while (REG_VCOUNT < 160) {} }
static void video_init(void) {
    static const uint16_t pal[][2] = {            /* {text, background} */
        { RGB(31, 31, 31), RGB(2, 3, 8) },        /* P_TEXT  */
        { RGB(31, 25, 8),  RGB(2, 3, 8) },        /* P_USER  */
        { RGB(15, 19, 27), RGB(2, 3, 8) },        /* P_DIM   */
        { RGB(2, 3, 8),    RGB(20, 27, 31) },     /* P_BAR   */
        { RGB(2, 3, 8),    RGB(31, 31, 31) },     /* P_KEY   */
        { RGB(12, 30, 16), RGB(2, 3, 8) },        /* P_GREEN */
    };
    for (unsigned p = 0; p < sizeof pal / sizeof pal[0]; p++) { PAL[p * 16 + 1] = pal[p][0]; PAL[p * 16 + 2] = pal[p][1]; }
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
static uint16_t keys_prev, keys_new;
static void poll(void) {                          /* called often, also from inside the model's work */
    uint16_t k = (uint16_t)(~REG_KEYINPUT & 0x03FF);
    keys_new |= (uint16_t)(k & ~keys_prev);
    keys_prev = k;
}
static uint16_t keys_down(void) {                 /* newly pressed since the last call */
    poll();
    uint16_t d = keys_new;
    keys_new = 0;
    return d;
}

/* ---------------- chat log: word wrap and scrolling, rows TOP..BOT ---------------- */
#define TOP 1
#define BOT 9
static int cx, cy = TOP, wstart;
static void newline(void) {
    cx = 0; wstart = 0;
    if (cy < BOT) { cy++; return; }
    for (int y = TOP; y < BOT; y++)
        for (int x = 0; x < 30; x++) MAP[y * 32 + x] = MAP[(y + 1) * 32 + x];
    for (int x = 0; x < 30; x++) put(x, BOT, ' ', P_TEXT);
}
static void log_char(int ch, int pal) {
    if (ch == '\n') { if (cx) newline(); return; }
    if (ch == ' ') {
        if (cx == 0) return;
        if (cx < 30) put(cx++, cy, ' ', pal);
        wstart = cx;
        return;
    }
    if (cx >= 30) {                               /* carry the unfinished word to the next line */
        uint16_t word[30]; int n = 0, y = cy;
        if (wstart > 0 && wstart < 30)
            for (int x = wstart; x < 30; x++) { word[n++] = MAP[y * 32 + x]; put(x, y, ' ', P_TEXT); }
        newline();
        for (int i = 0; i < n; i++) MAP[cy * 32 + cx++] = word[i];
    }
    put(cx++, cy, ch, pal);
}

/* ---------------- message editor ----------------
 * The model reads the message while it is typed: every finished word is fed in the background, one layer
 * per pass of the loop so the buttons stay responsive. Deleting back into a word that was already fed
 * rewinds the model to the start of the message and feeds the surviving words again. When the model has
 * caught up, it proposes two next words (L and R accept them). */
#define MSG_MAX 56
#define MSG_TOK 64
static char msg[MSG_MAX + 1];
static int mlen;
static const char *const quick[] = { "hi", "who are you", "tell me a story about a fox", "what is the capital of France?",
                                     "can you tell me a joke" };
static const char *const keyrow[4] = { "abcdefghijklm", "nopqrstuvwxyz", "0123456789+-=", ".,!?'\"():;/_<" };
static uint32_t seed = 0x2545F491;
static uint16_t fed[MSG_TOK], want[MSG_TOK], sug[3];      /* koko_suggest returns up to 3 */
static int nfed, nwant, user_fed, stepping, dirty, nsug, sug_stale, mark_ntok;

static void bar(void) { fill(0, 0, P_BAR); print(1, 0, "KOKO", P_BAR); }
static void feed_begin(int tok) {
    if (dbg.ntok < HISTMAX) dbg.hist[dbg.ntok] = (uint16_t)tok;
    dbg.ntok++;
    koko_step_begin(tok);
    stepping = 1;
}
static void feed(int tok) { feed_begin(tok); while (koko_step_part()) {} stepping = 0; }
static int last_space(void) { int i = mlen - 1; while (i >= 0 && msg[i] != ' ') i--; return i; }
static void retarget(void) {                     /* the words that are finished: everything before the last space */
    int i = last_space();
    nwant = i > 0 ? koko_encode(msg, i, want, MSG_TOK) : 0;
    if (nwant < nfed) dirty = 1;
    for (int k = 0; k < nfed && k < nwant; k++) if (fed[k] != want[k]) dirty = 1;
    sug_stale = 1; nsug = 0;
}
static int work(void) {                           /* one slice of background work; 0 when there is none */
    if (stepping) { if (!koko_step_part()) stepping = 0; return 1; }
    if (dirty) {
        koko_rewind(); dbg.ntok = (uint32_t)mark_ntok; dbg.rewinds++;
        nfed = 0; user_fed = 0; dirty = 0;
        return 1;
    }
    if (!user_fed) { feed_begin(TOK_USER); user_fed = 1; return 1; }
    if (nfed < nwant) { fed[nfed] = want[nfed]; feed_begin(want[nfed]); nfed++; return 1; }
    if (sug_stale) {
        int i = last_space(), ok = 1;
        for (int k = i + 1; k < mlen; k++) {
            int ch = msg[k] | 32;
            if (ch < 'a' || ch > 'z') ok = 0;
        }
        nsug = ok ? koko_suggest(msg + i + 1, mlen - i - 1, i < 0, sug) : 0;
        if (nsug > 2) nsug = 2;
        sug_stale = 0;
        dbg.nsug = (uint32_t)nsug; dbg.sug[0] = sug[0]; dbg.sug[1] = sug[1];
        return 1;
    }
    return 0;
}
static void draw_editor(int kx, int ky, int upper) {
    fill(11, 13, P_USER);
    for (int i = 0; i < mlen; i++) put(1 + i % 28, 11 + i / 28, msg[i], P_USER);
    if (mlen < MSG_MAX) put(1 + mlen % 28, 11 + mlen / 28, '_', P_TEXT);
    for (int k = 0; k < nsug; k++) {              /* suggestions: L and R */
        const char *p; int len = koko_token_text(sug[k], &p), x = 1 + k * 15;
        put(x, 13, k ? 'R' : 'L', P_GREEN);
        for (int j = 0; j < len && j < 12; j++) if (p[j] != ' ' || j) put(x + 2 + j - (p[0] == ' '), 13, p[j], P_TEXT);
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 13; c++) {
            int ch = keyrow[r][c];
            if (upper && r < 2) ch -= 32;
            int sel = (r == ky && c == kx);
            put(1 + c * 2, 14 + r, ' ', sel ? P_KEY : P_TEXT);
            put(2 + c * 2, 14 + r, ch, sel ? P_KEY : P_TEXT);
            put(3 + c * 2, 14 + r, ' ', sel ? P_KEY : P_TEXT);
        }
}
static void accept(int k) {                       /* replace the word being typed with suggestion k, add a space */
    const char *p; int len = koko_token_text(sug[k], &p);
    if (len && p[0] == ' ') { p++; len--; }
    mlen = last_space() + 1;
    for (int j = 0; j < len && mlen < MSG_MAX; j++) msg[mlen++] = p[j];
    if (mlen < MSG_MAX) msg[mlen++] = ' ';
    msg[mlen] = 0;
}
static void edit(void) {
    static int kx, ky, upper, q = -1;
    int rep = 0;
    fill(10, 19, P_TEXT);
    print(0, 10, "------------------------------", P_DIM);
    print(0, 18, "A type  B delete  SELECT Aa", P_DIM);
    print(0, 19, "START", P_GREEN); print(6, 19, "send   _ space  < delete", P_DIM);
    koko_mark(); mark_ntok = (int)dbg.ntok;
    nfed = nwant = user_fed = stepping = dirty = nsug = 0; sug_stale = 1;
    dbg.state = ST_EDIT; dbg.nsug = 0;
    for (;;) {
        int busy = work();
        dbg.state = busy ? ST_EDIT : ST_IDLE;
        draw_editor(kx, ky, upper);
        if (!busy) vsync();
        uint16_t d = keys_down();
        uint16_t held = keys_prev & (BTN_LEFT | BTN_RIGHT | BTN_UP | BTN_DOWN | BTN_B);
        if (held && !d) { if (++rep > (busy ? 6 : 14) && !(rep & 3)) d = held; } else rep = 0;
        if (!d) continue;
        seed = seed * 1664525u + REG_TM0CNT_L + 1013904223u;
        int old = mlen, changed = 0;
        if (d & BTN_LEFT) kx = (kx + 12) % 13;
        if (d & BTN_RIGHT) kx = (kx + 1) % 13;
        if (d & BTN_UP) ky = (ky + 3) % 4;
        if (d & BTN_DOWN) ky = (ky + 1) % 4;
        if (d & BTN_SELECT) upper = !upper;
        int del = (d & BTN_B) != 0;
        if (d & BTN_A) {
            int ch = keyrow[ky][kx];
            if (ky == 3 && kx == 12) del = 1;
            else {
                if (ky == 3 && kx == 11) ch = ' ';
                else if (upper && ky < 2) ch -= 32;
                if (mlen < MSG_MAX && !(ch == ' ' && (mlen == 0 || msg[mlen - 1] == ' '))) { msg[mlen++] = (char)ch; msg[mlen] = 0; }
            }
        }
        if (del && mlen) msg[--mlen] = 0;
        if ((d & BTN_L) && nsug > 0) { accept(0); changed = 1; }
        if ((d & BTN_R) && nsug > 1) { accept(1); changed = 1; }
        if (d & BTN_START) {
            if (mlen) return;
            q = (q + 1) % 5;                      /* START on an empty line: ready-made messages */
            for (mlen = 0; quick[q][mlen]; mlen++) msg[mlen] = quick[q][mlen];
            msg[mlen] = 0; changed = 1;
        }
        if (changed || mlen != old) retarget();
    }
}

/* ---------------- one exchange ---------------- */
static uint32_t ticks(void) {                     /* 1/16384 s */
    uint16_t hi = REG_TM3CNT_L, lo = REG_TM2CNT_L;
    if (hi != REG_TM3CNT_L) { hi = REG_TM3CNT_L; lo = REG_TM2CNT_L; }
    return ((uint32_t)hi << 16) | lo;
}
static void status(const char *s, int made, uint32_t t) {
    fill(0, 0, P_BAR);
    int x = print(1, 0, "KOKO", P_BAR);
    x = print(x + 2, 0, s, P_BAR);
    if (made && t) {
        uint32_t r = (uint32_t)made * 163840u / t;          /* tokens/s x 10 */
        x = print_num(22, 0, r / 10, P_BAR);
        put(x++, 0, '.', P_BAR);
        x = print_num(x, 0, r % 10, P_BAR);
        print(x, 0, "/s", P_BAR);
    }
}
static void exchange(void) {
    dbg.state = ST_RUNNING;
    while (mlen && msg[mlen - 1] == ' ') msg[--mlen] = 0;    /* a trailing space is not part of the message */
    fill(10, 19, P_TEXT);
    print(0, 10, "------------------------------", P_DIM);
    print(0, 19, "B", P_GREEN); print(2, 19, "stop", P_DIM);
    if (cx) newline();
    log_char('>', P_USER); log_char(' ', P_USER);
    for (int i = 0; i < mlen; i++) log_char(msg[i], P_USER);
    newline();
    status("reading...", 0, 0);
    /* finish what the editor started: the whole message, then Koko's marker */
    while (stepping) work();
    nwant = koko_encode(msg, mlen, want, MSG_TOK);
    if (nwant < nfed) dirty = 1;
    for (int k = 0; k < nfed && k < nwant; k++) if (fed[k] != want[k]) dirty = 1;
    sug_stale = 0;
    while (work()) {}
    feed(TOK_KOKO);
    if (!seed) seed = 1;
    dbg.seed = seed;
    REG_TM2CNT_H = 0; REG_TM3CNT_H = 0; REG_TM2CNT_L = 0; REG_TM3CNT_L = 0;
    REG_TM3CNT_H = 0x0084;                        /* count TM2 overflows */
    REG_TM2CNT_H = 0x0083;                        /* 16.78 MHz / 1024 */
    uint32_t t0 = ticks();
    int made = 0;
    status("writing", 0, 0);
    static char ans[96];
    int al = sum_reply(msg, mlen, ans), an = 0;          /* a sum: the game works it out and Koko says it */
    if (al) an = koko_encode(ans, al, want, MSG_TOK);
    koko_reply_begin();                                  /* repeated-phrase rule on for this reply */
    while (made < REPLY_MAX) {
        int t;
        if (al) { if (made >= an) break; t = want[made]; }
        else t = koko_pick(&seed, INV_TEMP);
        if (t == TOK_END || t == TOK_USER || t == TOK_KOKO) break;
        const char *p;
        int len = koko_token_text(t, &p);
        for (int i = 0; i < len; i++) log_char(p[i], P_TEXT);
        made++;
        feed(t);
        uint32_t el = ticks() - t0;
        dbg.made = (uint32_t)made; dbg.ticks = el;
        status("writing", made, el);
        if (keys_down() & BTN_B) break;
    }
    koko_reply_end();
    newline();
    status("", made, ticks() - t0);
    dbg.turns++;
    dbg.state = ST_DONE;
    mlen = 0; msg[0] = 0;
}

int main(void) {
    REG_WAITCNT = 0x4317;                         /* fastest cartridge timing + prefetch */
    REG_TM0CNT_L = 0; REG_TM0CNT_H = 0x0080;      /* free-running, feeds the random seed */
    video_init();
    bar();
    koko_yield = poll;
    dbg.magic = 0x4B4F4B4F; dbg.turns = 0; dbg.ntok = 0; dbg.made = 0; dbg.rewinds = 0;
    koko_reset();
    cy = TOP; cx = 0;
    {
        const char *hello = "Woof! I'm Koko. Say something!";
        for (int i = 0; hello[i]; i++) log_char(hello[i], P_DIM);
        newline();
    }
    for (;;) { edit(); exchange(); }
}
