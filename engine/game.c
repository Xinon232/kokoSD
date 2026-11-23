/* Koko: the cartridge game. Bitmap mode 4 (240x160, 256 colours); everything drawn comes from ui.c.
 * No BIOS calls, no interrupts. */
#include <stdint.h>
#include "koko.h"
#include "ui.h"
#include "sd.h"
#include "cart_io.h"
int sum_reply(const char *s, int n, char *out);   /* engine/sums.c */
int fact_reply(const char *msg, int n, uint16_t *ids, int max);   /* engine/facts.c */

#define REG16(a) (*(volatile uint16_t *)(a))
#define REG_DISPCNT  REG16(0x04000000)
#define REG_VCOUNT   REG16(0x04000006)
#define REG_TM0CNT_L REG16(0x04000100)
#define REG_TM0CNT_H REG16(0x04000102)
#define REG_TM2CNT_L REG16(0x04000108)
#define REG_TM2CNT_H REG16(0x0400010A)
#define REG_TM3CNT_L REG16(0x0400010C)
#define REG_TM3CNT_H REG16(0x0400010E)
#define REG_KEYINPUT REG16(0x04000130)
#define REG_WAITCNT  REG16(0x04000204)
#define PAL ((volatile uint16_t *)0x05000000)
#define EW  __attribute__((section(".bss.ewram")))

enum { BTN_A = 1, BTN_B = 2, BTN_SELECT = 4, BTN_START = 8, BTN_RIGHT = 16, BTN_LEFT = 32,
       BTN_UP = 64, BTN_DOWN = 128, BTN_R = 256, BTN_L = 512 };
enum { TOK_END = 0, TOK_USER = 1, TOK_KOKO = 2 };
#define INV_TEMP 512                    /* temperature 0.5 */
#define REPLY_MAX 120
#define RGB(r, g, b) ((r) | ((g) << 5) | ((b) << 10))

/* The tests read this block from the emulator: it sits first in slow RAM (0x02000000). */
#define HISTMAX 1024
struct Dbg {
    uint32_t magic, state, seed, turns, ntok, ticks, made, nsug, rewinds;
    uint16_t sug[2], newchats, pure;
    uint16_t hist[HISTMAX];             /* every token the model has taken in (and not taken back), in order */
    uint32_t sd;                        /* kokoSD: bit 0 card mounted, bits 8-15 gbaSD's cart_kind */
};
enum { ST_EDIT = 0, ST_RUNNING = 1, ST_DONE = 2, ST_IDLE = 3 };   /* IDLE: editing, model caught up, suggestions shown */
volatile struct Dbg dbg __attribute__((section(".bss.dbg")));


static void vsync(void) { while (REG_VCOUNT >= 160) {} while (REG_VCOUNT < 160) {} }
static void video_init(void) {
    ui_init();
    for (int i = 0; i < UI_PAL_N; i++) PAL[i] = ui_pal[i];
    REG_DISPCNT = 0x0404;                           /* mode 4, BG2 on, page 0 */
}
/* Button presses are queued, so two quick presses of the same button during a long piece of work both count. */
static uint16_t keys_prev, kq[16];
static uint8_t kq_head, kq_n;
static void poll(void) {                          /* called often, also from inside the model's work and long redraws */
    uint16_t k = (uint16_t)(~REG_KEYINPUT & 0x03FF), d = (uint16_t)(k & ~keys_prev);
    keys_prev = k;
    if (d && kq_n < 16) { kq[(kq_head + kq_n) & 15] = d; kq_n++; }
}
static uint16_t keys_down(void) {                 /* the next queued press (0 if none) */
    poll();
    if (!kq_n) return 0;
    uint16_t d = kq[kq_head];
    kq_head = (uint8_t)((kq_head + 1) & 15); kq_n--;
    return d;
}

/* ---------------- message editor ----------------
 * The model reads the message while it is typed: every finished word is fed in the background, one layer
 * per pass of the loop so the buttons stay responsive. Deleting back into a word that was already fed
 * rewinds the model to the start of the message and feeds the surviving words again. When the model has
 * caught up, it proposes two next words (L and R accept them). */
#define MSG_MAX UI_MSG_MAX
#define MSG_TOK 64
static char msg[MSG_MAX + 1];
static int mlen;
static const char *const quick[] = { "hi", "who are you", "tell me a story about a fox", "what is the capital of France?",
                                     "can you tell me a joke" };
static uint32_t seed = 0x2545F491;
static uint16_t fed[MSG_TOK], want[MSG_TOK], sug[3];      /* koko_suggest returns up to 3 */
static int nfed, nwant, user_fed, stepping, dirty, nsug, sug_stale, mark_ntok;
static int helpers = 1;                           /* fact table and calculator on; off = every word comes from the model */

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

static void chips(void) {
    const char *a = 0, *b = 0; int al = 0, bl = 0;
    if (nsug > 0) al = koko_token_text(sug[0], &a);
    if (nsug > 1) bl = koko_token_text(sug[1], &b);
    ui_chips(nsug, a, al, b, bl);
}
static void accept(int k) {                       /* replace the word being typed with suggestion k, add a space */
    const char *p; int len = koko_token_text(sug[k], &p);
    if (len && p[0] == ' ') { p++; len--; }
    mlen = last_space() + 1;
    for (int j = 0; j < len && mlen < MSG_MAX; j++) msg[mlen++] = p[j];
    if (mlen < MSG_MAX) msg[mlen++] = ' ';
    msg[mlen] = 0;
}
/* Returns 1 to send the message, 0 to start a new chat (the new-chat key). */
static int edit(void) {
    static int kx, ky, upper, q = -1;
    int rep = 0, shown_sug = -1, shown_empty = -1, note = 0;
    ui_typing(msg, mlen, kx, ky, upper);
    koko_mark(); mark_ntok = (int)dbg.ntok;
    nfed = nwant = user_fed = stepping = dirty = nsug = 0; sug_stale = 1;
    dbg.state = ST_EDIT; dbg.nsug = 0;
    for (;;) {
        int busy = work();
        dbg.state = busy ? ST_EDIT : ST_IDLE;
        int sig = nsug ? sug[0] * 3 + sug[nsug - 1] * 7 + nsug : 0;
        if (sig != shown_sug) { chips(); shown_sug = sig; }
        if (!note && (mlen == 0) != shown_empty) { shown_empty = mlen == 0; ui_header(mlen ? "START send" : "START ideas"); }
        if (!busy) vsync();
        uint16_t d = keys_down();
        uint16_t held = keys_prev & (BTN_LEFT | BTN_RIGHT | BTN_UP | BTN_DOWN | BTN_B);
        if (held && !d) { if (++rep > (busy ? 6 : 14) && !(rep & 3)) d = held; } else rep = 0;
        if (!d) continue;
        seed = seed * 1664525u + REG_TM0CNT_L + 1013904223u;
        if (note) { note = 0; shown_empty = -1; }
        int old = mlen, changed = 0, okx = kx, oky = ky;
        if (d & BTN_LEFT) ui_key_move(&kx, &ky, -1, 0);
        if (d & BTN_RIGHT) ui_key_move(&kx, &ky, 1, 0);
        if (d & BTN_UP) ui_key_move(&kx, &ky, 0, -1);
        if (d & BTN_DOWN) ui_key_move(&kx, &ky, 0, 1);
        if (d & BTN_SELECT) { upper = !upper; ui_keyboard(kx, ky, upper); }
        else if (okx != kx || oky != ky) { ui_keycap(oky, okx, 0, upper); ui_keycap(ky, kx, 1, upper); }
        int del = (d & BTN_B) != 0;
        if (d & BTN_A) {
            int ch = ui_key(ky, kx);
            if (ch == UI_KEY_NEW) return 0;
            if (ch == UI_KEY_HELP) {
                helpers = !helpers; ui_helpers(helpers); dbg.pure = (uint16_t)!helpers;
                ui_header(helpers ? "helpers on" : "model only"); note = 1;           /* the usual hint returns with the next press */
                continue;
            }
            else if (ch == '<') del = 1;
            else {
                if (ch == '_') ch = ' ';
                else if (upper && ky < 2) ch -= 32;
                if (mlen < MSG_MAX && !(ch == ' ' && (mlen == 0 || msg[mlen - 1] == ' '))) { msg[mlen++] = (char)ch; msg[mlen] = 0; changed = 1; }
            }
        }
        if (del && mlen) { msg[--mlen] = 0; changed = 1; }
        if ((d & BTN_L) && nsug > 0) { accept(0); changed = 1; }
        if ((d & BTN_R) && nsug > 1) { accept(1); changed = 1; }
        if (d & BTN_START) {
            if (mlen) return 1;
            q = (q + 1) % 5;                      /* START on an empty line: ready-made messages */
            for (mlen = 0; quick[q][mlen]; mlen++) msg[mlen] = quick[q][mlen];
            msg[mlen] = 0; changed = 1;
        }
        if (changed || mlen != old) { retarget(); ui_pill(1, msg, mlen); }
    }
}

/* ---------------- one exchange ---------------- */
static uint32_t ticks(void) {                     /* 1/16384 s */
    uint16_t hi = REG_TM3CNT_L, lo = REG_TM2CNT_L;
    if (hi != REG_TM3CNT_L) { hi = REG_TM3CNT_L; lo = REG_TM2CNT_L; }
    return ((uint32_t)hi << 16) | lo;
}
static void exchange(void) {
    dbg.state = ST_RUNNING;
    while (mlen && msg[mlen - 1] == ' ') msg[--mlen] = 0;    /* a trailing space is not part of the message */
    ui_reply_layout(msg, mlen);
    ui_reply_button(1);
    ui_header("reading");
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
    ui_header("B stop");
    static char ans[96] EW;
    int al = 0, an = 0, mark = 0;
    if (helpers) {
        al = sum_reply(msg, mlen, ans);                  /* a sum: the game works it out and Koko says it */
        if (al) { an = koko_encode(ans, al, want, MSG_TOK); mark = 2; }
        else { an = fact_reply(msg, mlen, want, MSG_TOK); al = an; mark = an ? 1 : 0; }   /* a fact the cartridge has on file */
    }
    koko_reply_begin();                                  /* repeated-phrase rule on for this reply */
    ui_reply_start(mark);
    while (made < REPLY_MAX) {
        int t;
        if (al) { if (made >= an) break; t = want[made]; }
        else t = koko_pick(&seed, INV_TEMP);
        if (t == TOK_END || t == TOK_USER || t == TOK_KOKO) break;
        const char *p;
        int len = koko_token_text(t, &p);
        ui_reply_text(p, len);
        made++;
        feed(t);
        uint32_t el = ticks() - t0;
        dbg.made = (uint32_t)made; dbg.ticks = el;
        if (keys_down() & BTN_B) break;
    }
    koko_reply_end();
    ui_reply_button(0);
    ui_rate(made, ticks() - t0);
    dbg.turns++;
    dbg.state = ST_DONE;
    mlen = 0; msg[0] = 0;
}

int main(void) {
    REG_WAITCNT = 0x4317;                         /* fastest cartridge timing + prefetch */
    REG_TM0CNT_L = 0; REG_TM0CNT_H = 0x0080;      /* free-running, feeds the random seed */
    video_init();
    koko_yield = poll; ui_yield = poll;
    dbg.magic = 0x4B4F4B4F; dbg.turns = 0; dbg.ntok = 0; dbg.made = 0; dbg.rewinds = 0; dbg.newchats = 0; dbg.pure = 0;
    koko_reset();
    sd_mount();                                   /* kokoSD: the SD card (nothing uses it yet; 0 without a card) */
    dbg.sd = (uint32_t)sd_ready | (uint32_t)cart_kind << 8;
    ui_title();                                   /* Koko's portrait until START (or A) is pressed */
    for (int f = 0;; f++) {
        vsync();
        uint16_t d = keys_down();
        if (d) seed = seed * 1664525u + REG_TM0CNT_L + 1013904223u;
        if (d & (BTN_START | BTN_A)) break;
        if ((f & 31) == 0) ui_title_prompt(!(f & 32));
        ui_title_tick();
    }
    ui_init();
    ui_say(L_KOKO, "Woof! I'm Koko. Say something!", 30);
    for (;;) {
        if (!edit()) {                            /* new chat: the model and the screen forget everything */
            while (stepping) work();
            koko_reset();
            ui_clear(); ui_say(L_KOKO, "Woof! I'm Koko. Say something!", 30);
            mlen = 0; msg[0] = 0; dbg.ntok = 0; dbg.newchats++;
            continue;
        }
        exchange();
        while (!kq_n) { vsync(); poll(); }        /* the reply stays up until a button is pressed; edit() gets that press */
    }
}
