/* Koko: the cartridge game. Bitmap mode 4 (240x160, 256 colours); everything drawn comes from ui.c.
 * No BIOS calls, no interrupts. */
#include <stdint.h>
#include "koko.h"
#include "ui.h"
#include "sd.h"
#include "cart_io.h"
#include "facts_sd.h"
#include "writer/writer_glue.h"
#include "text/text.h"
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
    uint32_t sd;                        /* kokoSD: bit 0 card mounted, bit 1 fact file open, bits 8-15 gbaSD's cart_kind */
    uint32_t article, article_top;      /* kokoSD: lines of the article (or page) being read, first line shown */
    uint32_t menu, caret;               /* kokoSD: menu row; cursor in the message */
    uint32_t boot_ticks;                /* kokoSD: 1/16384 s from power-on to the title (card mounted, fact file opened) */
};
enum { ST_EDIT = 0, ST_RUNNING = 1, ST_DONE = 2, ST_IDLE = 3, ST_ARTICLE = 4, ST_MENU = 5, ST_PAGE = 6, ST_TITLE = 7 };   /* kokoSD: an article, the menu, How to type / About */   /* IDLE: editing, model caught up, suggestions shown */
volatile struct Dbg dbg __attribute__((section(".bss.dbg")));


static void vsync(void) { while (REG_VCOUNT >= 160) {} while (REG_VCOUNT < 160) {} }
static void video_init(void) {
    ui_init();
    for (int i = 0; i < UI_PAL_N; i++) PAL[i] = ui_pal[i];
    REG_DISPCNT = 0x0404;                           /* mode 4, BG2 on, page 0 */
}
/* Button presses are queued, so two quick presses of the same button during a long piece of work both count. */
static uint16_t keys_prev, kq[16], hq[32];
static uint8_t hq_n;                              /* kokoSD: the button states in order since the editor last looked */
static uint8_t kq_head, kq_n;
static uint32_t ticks(void);
/* kokoSD: contact bounce (as gbamp3 13.3). A button's state follows the contact at once when the contact had been
 * quiet for 8 ms before the change; a change within 8 ms of another waits until the contact has been quiet for
 * 8 ms. Polling as often as this does, a bounce on release was otherwise a second press. */
#define DEBOUNCE 131u                             /* 8 ms in 1/16384 s */
static uint16_t raw_prev;
static uint32_t moved_at[10];
static void poll(void) {                          /* called often, also from inside the model's work and long redraws */
    uint16_t raw = (uint16_t)(~REG_KEYINPUT & 0x03FF), k = keys_prev;
    if (raw != keys_prev || raw != raw_prev) {
        uint32_t now = ticks();
        for (int i = 0; i < 10; i++) {
            uint16_t b = (uint16_t)(1u << i);
            int quiet = now - moved_at[i] >= DEBOUNCE;
            if ((raw ^ raw_prev) & b) moved_at[i] = now;
            if (((raw ^ k) & b) && quiet) k ^= b;
        }
        raw_prev = raw;
    }
    uint16_t d = (uint16_t)(k & ~keys_prev);
    if (k != keys_prev && hq_n < 32) hq[hq_n++] = k;
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
/* kokoSD 1.1: the message is shown as typed (accented letters too); Koko and the helpers read this plain-letter
 * copy of it (text_plain: "Zürich" -> "Zurich"), since Koko knows plain ASCII only. */
static char plain[2 * MSG_MAX + 1];
static int plen;
static void refold(void) { plen = (int)text_plain(plain, 2 * MSG_MAX, msg, (unsigned)mlen); plain[plen] = 0; }
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
static int last_space_in(const char *s, int n) { int i = n - 1; while (i >= 0 && s[i] != ' ') i--; return i; }
static int last_space(void) { return last_space_in(msg, mlen); }
static void retarget(void) {                     /* the words that are finished: everything before the last space */
    int i = last_space_in(plain, plen);
    nwant = i > 0 ? koko_encode(plain, i, want, MSG_TOK) : 0;
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
        int i = last_space_in(plain, plen), ok = 1;
        for (int k = i + 1; k < plen; k++) {
            int ch = plain[k] | 32;
            if (ch < 'a' || ch > 'z') ok = 0;
        }
        nsug = ok ? koko_suggest(plain + i + 1, plen - i - 1, i < 0, sug) : 0;
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
    refold();
}
/* ---------------- kokoSD: menu and pages ---------------- */
static const char HOW_TO_TYPE[] =
    "Letters: hold a direction, then press B, A or R for its first, second or third letter.\n"
    "Up: a b c    Right: h i j\nDown: n o p    Left: t u w\n\n"
    "Hold L as well for the second layer:\nUp: d e f    Right: k l m\nDown: q r s    Left: x y z\n\n"
    "g and v: keep Right or Left held and press R twice (j becomes g, w becomes v).\n\n"
    "A alone: space. B alone: delete (hold to repeat).\n\n"
    "R alone: a capital for the next letter. Hold R: capitals on; R again: off.\n\n"
    "SELECT alone types a dot. Keep SELECT held and press Up / Down for digits, R / L for , ' \" : ! ? "
    "and Right / Left for ( ) / ; @ # % & _ + = -.\n\n"
    "START: send. START on an empty line: a ready-made message (START again sends it).\n\n"
    "START with L or R (in either order): the first or second suggested word. Hold START and press Left / Right: "
    "move the cursor. START + SELECT: this menu.\n\n"
    "Accents: keep the letter's buttons held and press SELECT; SELECT again gives the next one (e: é è ë ê). "
    "They are shown as typed; Koko reads them as plain letters (é as e).\n\n"
    "B while Koko writes: stop him.\n\n"
    "The same typing as gbawriter, gbamp3 and the other apps.";
static const char ABOUT[] =
    "kokoSD 1.1.3: Koko the Chatbot with an SD card.\n\n"
    "Koko, a friendly husky, is a language model of 72 million two-bit weights that runs on the "
    "Game Boy Advance itself.\n"
    "\nFacts: \"who is\", \"what is\" and \"tell me about\" are answered from Simple English Wikipedia "
    "(CC BY-SA 4.0): the file koko/facts.dat on the SD card, or the cartridge's own smaller table. "
    "After such an answer, \"tell me more\" shows the article. Sums are worked out by the game. "
    "Replies marked Fact or Sum come from these helpers; the menu turns them off.\n"
    "\nCode: Koko (MIT), gbaSD and gbawriter typing by Halim Jarrar (GPL-3.0-or-later); the cartridge "
    "as a whole is GPL-3.0-or-later. Weights, fact table and artwork: CC BY-SA 4.0. "
    "Fonts: the SuperFW font by David Guillen Fandos (GPL-3.0-or-later; UNSCII and GNU Unifont glyphs) and "
    "gbamp3's 5x7 font; accent folding from GBAVocab (Halim Jarrar, GPL-3.0-or-later).";
/* A page of text to read (an article, How to type, About): Up / Down scroll, Left / Right / L / R turn a page,
 * B (or A, START) goes back. */
static void read_pages(const char *title, const char *text, int n, int state) {
    int lines = ui_article_wrap(text, n), rows = ui_article_rows(), top = 0, last = lines > rows ? lines - rows : 0;
    kq_n = 0;                                     /* presses made before the page opened are not page turns */
    dbg.state = (uint32_t)state; dbg.article = (uint32_t)lines; dbg.article_top = 0;
    ui_article(title, text, 0);                   /* presses while it draws count (polled between rows) */
    for (;;) {
        vsync();
        uint16_t d = keys_down();
        int was = top;
        if (d & (BTN_DOWN | BTN_RIGHT | BTN_R)) top += (d & BTN_DOWN) ? 3 : rows;
        if (d & (BTN_UP | BTN_LEFT | BTN_L)) top -= (d & BTN_UP) ? 3 : rows;
        if (top > last) top = last;
        if (top < 0) top = 0;
        if (top != was) { ui_article(title, text, top); dbg.article_top = (uint32_t)top; }
        if (d & (BTN_B | BTN_START | BTN_A)) break;
    }
}
/* START + SELECT while typing. Returns 1 for a new chat. */
static int menu(void) {
    static int sel;
    const char *items[4] = { "New chat", 0, "How to type", "About kokoSD" };
    kq_n = 0;                                     /* presses made while typing are not menu choices */
    for (;;) {
        items[1] = helpers ? "Facts and sums: On" : "Facts and sums: Off";
        dbg.state = ST_MENU; dbg.menu = (uint32_t)sel;
        ui_menu("Menu", items, 4, sel);
        for (;;) {
            vsync();
            uint16_t d = keys_down();
            if (!d) continue;
            if (d & BTN_UP) { sel = (sel + 3) % 4; break; }
            if (d & BTN_DOWN) { sel = (sel + 1) % 4; break; }
            if (d & (BTN_B | BTN_START)) return 0;
            if (d & BTN_A) {
                if (sel == 0) return 1;
                if (sel == 1) { helpers = !helpers; ui_helpers(helpers); dbg.pure = (uint16_t)!helpers; }
                if (sel == 2) read_pages("How to type", HOW_TO_TYPE, (int)sizeof HOW_TO_TYPE - 1, ST_PAGE);
                if (sel == 3) read_pages("About kokoSD", ABOUT, (int)sizeof ABOUT - 1, ST_PAGE);
                kq_n = 0;
                break;
            }
        }
    }
}

/* Returns 1 to send the message, 0 to start a new chat (the menu's New chat).
 * kokoSD: typed with gbawriter's buttons (engine/writer): one update of its engine per video frame (274
 * ticks of 1/16384 s), several when the model's work took longer, so holding B repeats at its usual speed. */
static void msg_from_writer(void) {
    const char *t = writer_text();
    for (mlen = 0; t[mlen] && mlen < MSG_MAX; mlen++) msg[mlen] = t[mlen];
    msg[mlen] = 0;
    refold();
}
static void typing_hint(void) {                  /* kokoSD 1.1: as gbamp3's Search: letter group and Shift / Caps in the header */
    ui_writer_status(writer_group(), writer_caps() ? "Caps" : writer_shift() ? "Shift" : "");
    ui_hint(mlen);
}
static int edit(void) {
    static int q = -1;
    int shown_sug = -1, shown_hint = -1;
    writer_begin(msg);
    refold();
    ui_header(0);
    ui_typing(msg, mlen, mlen);
    koko_mark(); mark_ntok = (int)dbg.ntok;
    nfed = nwant = user_fed = stepping = dirty = nsug = 0; sug_stale = 1;
    dbg.state = ST_EDIT; dbg.nsug = 0;
    kq_n = 0;
    uint32_t last = ticks();
    for (;;) {
        int busy = work();
        dbg.state = busy ? ST_EDIT : ST_IDLE;
        int sig = nsug ? sug[0] * 3 + sug[nsug - 1] * 7 + nsug : 0;
        if (sig != shown_sug) { chips(); shown_sug = sig; }
        if (!busy) vsync();
        poll();
        uint32_t now = ticks();
        if (now < last) last = now;                    /* the timers were started again */
        if (now - last < 274) continue;
        unsigned frames = (now - last) / 274;
        if (frames > 4) frames = 4;
        last = now;
        unsigned fl = 0, f = 0;
        for (unsigned i = 0; i < hq_n; i++, f++) fl |= writer_frame(hq[i]);   /* every change, in order (also while busy) */
        hq_n = 0;
        for (; f < frames; f++) fl |= writer_frame(keys_prev);
        int changed = 0;
        if (fl & WRITER_TEXT) { msg_from_writer(); changed = 1; }
        if (fl & WRITER_PAGE_PREV) { if (nsug > 0) { accept(0); writer_set(msg); changed = 1; } }
        if (fl & WRITER_PAGE_NEXT) { if (nsug > 1) { accept(1); writer_set(msg); changed = 1; } }
        if (fl & WRITER_START_ALONE) {
            if (mlen) return 1;
            q = (q + 1) % 5;                          /* START on an empty line: ready-made messages */
            writer_set(quick[q]); msg_from_writer(); changed = 1;
        }
        if (fl & WRITER_START_SELECT) {
            while (stepping) work();
            if (menu()) return 0;
            writer_set(msg);
            ui_header(0);
            ui_typing(msg, mlen, (int)writer_caret());
            shown_sug = -1; shown_hint = -1;
            continue;
        }
        if (fl) seed = seed * 1664525u + REG_TM0CNT_L + 1013904223u;
        const char *g = writer_group();
        int hint = (g && *g ? (int)g[0] * 4 : 0) + (mlen ? 1 : 0) + writer_caps() * 2 + writer_shift() * 512;
        if (hint != shown_hint) { typing_hint(); shown_hint = hint; }
        if (changed) { retarget(); ui_input(msg, mlen, (int)writer_caret()); dbg.caret = writer_caret(); }
        else if (fl & WRITER_TEXT) ui_input(msg, mlen, (int)writer_caret());
    }
}

/* ---------------- kokoSD: Tell me more ---------------- */
/* "tell me more" (also "more", "go on", with "please") right after a fact from the SD card's file, or
 * "tell me more about X": Koko says "Here is more about <title>." and the article opens to read. */
static int lower_eq(const char *s, int n, const char *w) {
    int i = 0;
    for (; w[i]; i++) { if (i >= n) return 0; int c = (unsigned char)s[i]; if (c >= 'A' && c <= 'Z') c += 32; if (c != w[i]) return 0; }
    return i == n;
}
static int more_request(const char *s, int n) {
    while (n && (s[n - 1] == '.' || s[n - 1] == '!' || s[n - 1] == '?' || s[n - 1] == ' ')) n--;
    static const char *const W[] = { "tell me more", "tell me more please", "more", "more please", "go on",
                                     "tell me more about it", "tell me more about that", 0 };
    for (int k = 0; W[k]; k++) if (lower_eq(s, n, W[k])) return 1;
    return 0;
}
static int more_about(const char *s, int n) {      /* "tell me more about X": length of "tell me more about " or 0 */
    const char *w = "tell me more about ";
    int i = 0;
    for (; w[i]; i++) { if (i >= n) return 0; int c = (unsigned char)s[i]; if (c >= 'A' && c <= 'Z') c += 32; if (c != w[i]) return 0; }
    return n > i + 1 ? i : 0;
}
static void read_article(void) {
    const char *title, *text;
    int n = sdf_article(&title, &text);
    if (n) read_pages(title, text, n, ST_ARTICLE);
    ui_chat_again();
}

/* ---------------- one exchange ---------------- */
static uint32_t ticks(void) {                     /* 1/16384 s */
    uint16_t hi = REG_TM3CNT_L, lo = REG_TM2CNT_L;
    if (hi != REG_TM3CNT_L) { hi = REG_TM3CNT_L; lo = REG_TM2CNT_L; }
    return ((uint32_t)hi << 16) | lo;
}
static void exchange(void) {
    dbg.state = ST_RUNNING;
    kq_n = 0;                                     /* kokoSD: presses made while typing (B in letters) are not "stop" */
    while (mlen && msg[mlen - 1] == ' ') msg[--mlen] = 0;    /* a trailing space is not part of the message */
    refold();
    while (plen && plain[plen - 1] == ' ') plain[--plen] = 0;
    ui_reply_layout(msg, mlen);
    ui_reply_button(1);
    ui_header("reading...");
    /* finish what the editor started: the whole message, then Koko's marker */
    while (stepping) work();
    nwant = koko_encode(plain, plen, want, MSG_TOK);
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
    if (helpers) ui_header(sdf_ready ? "looking up..." : "thinking...");   /* kokoSD 1.1.3: each phase is shown */
    static char ans[96] EW;
    int al = 0, an = 0, mark = 0, article = 0;
    if (helpers && sd_ready) {                           /* kokoSD: Tell me more */
        int ab = more_about(plain, plen);
        if (ab) {
            static char q[80] EW;                        /* "tell me about X": the fact table's own wording */
            const char *pre = "tell me about "; int ql = 0;
            while (pre[ql]) { q[ql] = pre[ql]; ql++; }
            for (int i = ab; i < plen && ql < 79; i++) q[ql++] = plain[i];
            sdf_forget();
            article = fact_reply(q, ql, want, MSG_TOK) > 0;
        } else article = more_request(plain, plen);
        const char *title, *text;
        if (article && sdf_article(&title, &text)) {
            const char *a = "Here is more about "; al = 0;
            while (*a && al < 80) ans[al++] = *a++;
            for (const char *t = title; *t && al < 94; t++) ans[al++] = *t;
            ans[al++] = '.';
            an = koko_encode(ans, al, want, MSG_TOK); mark = 1;
        } else article = 0;
    }
    if (helpers && !article) {
        sdf_forget();
        al = sum_reply(plain, plen, ans);                  /* a sum: the game works it out and Koko says it */
        if (al) { an = koko_encode(ans, al, want, MSG_TOK); mark = 2; }
        else { an = fact_reply(plain, plen, want, MSG_TOK); al = an; mark = an ? 1 : 0; }   /* a fact the cartridge has on file */
    }
    ui_header("B: stop");
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
    if (article) read_article();
    dbg.turns++;
    dbg.state = ST_DONE;
    mlen = 0; msg[0] = 0; plen = 0; plain[0] = 0;
}

int main(void) {
    /* kokoSD 1.1.2: the cartridge timing stays as the loader left it until the cart is known. The fastest timing
     * (0x4317) is too fast for a SuperCard's SDRAM: there the first ROM read after it crashes the GBA (a blank
     * screen), and SuperFW cannot patch the write out (it looks for the address as a constant). */
    REG_TM0CNT_L = 0; REG_TM0CNT_H = 0x0080;      /* free-running, feeds the random seed */
    REG_TM2CNT_H = 0; REG_TM3CNT_H = 0; REG_TM2CNT_L = 0; REG_TM3CNT_L = 0;
    REG_TM3CNT_H = 0x0084; REG_TM2CNT_H = 0x0083;  /* kokoSD: 1/16384 s from the start (debounce, editor frames) */
    video_init();
    koko_yield = poll; ui_yield = poll;
    dbg.magic = 0x4B4F4B4F; dbg.turns = 0; dbg.ntok = 0; dbg.made = 0; dbg.rewinds = 0; dbg.newchats = 0; dbg.pure = 0;
    koko_reset();
    ui_boot("Looking for the SD card...", "");    /* kokoSD 1.1: each step is shown (a hang tells where) */
    sd_mount();                                   /* kokoSD: the SD card (0 without a card) */
    if (cart_kind >= CART_EZFO) REG_WAITCNT = 0x4317;   /* Omega / EverDrive: fastest cartridge timing + prefetch */
    ui_boot(sd_ready ? "Opening the fact file..." : "No SD card found", sd_cart_name());
    sdf_open();                                   /* and its fact file /koko/facts.dat (0 without one) */
    dbg.sd = (uint32_t)sd_ready | (uint32_t)sdf_ready << 1 | (uint32_t)cart_kind << 8;
    ui_title();                                   /* the title until START (or A) is pressed */
    dbg.state = ST_TITLE; dbg.boot_ticks = ticks();   /* kokoSD: the title is up (tests wait for it); start-up time */
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
    while (keys_prev) { vsync(); poll(); }        /* kokoSD: the START that left the title types nothing */
    hq_n = 0;
    for (;;) {
        if (!edit()) {                            /* new chat: the model and the screen forget everything */
            while (stepping) work();
            koko_reset();
            ui_clear(); ui_say(L_KOKO, "Woof! I'm Koko. Say something!", 30);
            mlen = 0; msg[0] = 0; plen = 0; plain[0] = 0; dbg.ntok = 0; dbg.newchats++;
            while (keys_prev) { vsync(); poll(); }    /* the A that chose New chat types nothing */
            hq_n = 0;
            continue;
        }
        exchange();
        hq_n = 0;                                 /* kokoSD: presses during the reply (B: stop) are not typed */
        while (!kq_n) { vsync(); poll(); }        /* the reply stays up until a button is pressed; edit() gets that press */
    }
}
