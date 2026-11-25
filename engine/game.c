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
#include "chats.h"
#include "ff.h"
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
enum { ST_EDIT = 0, ST_RUNNING = 1, ST_DONE = 2, ST_IDLE = 3, ST_ARTICLE = 4, ST_MENU = 5, ST_PAGE = 6, ST_TITLE = 7, ST_LIST = 8, ST_OPTIONS = 9, ST_CONFIRM = 10, ST_RENAME = 11, ST_SELFTEST = 12 };   /* kokoSD: an article, the menu, How to type / About */   /* IDLE: editing, model caught up, suggestions shown */
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
static void life_note(void);                      /* kokoSD 1.3.1: restart detector (below) */
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
    if (k != keys_prev) { keys_prev = k; life_note(); }
    if (d && kq_n < 16) { kq[(kq_head + kq_n) & 15] = d; kq_n++; }
}
/* kokoSD 1.1.3: while Koko answers, B must stop him even when other buttons were pressed first: every queued
 * press is taken (the others mean nothing there) and B among them counts. Before, one press was taken per word,
 * so B waited behind the others, and with 16 queued it was dropped: Koko could not be stopped. */
static int b_pressed(void) {
    int b = 0;
    poll();
    for (; kq_n; kq_n--, kq_head = (uint8_t)((kq_head + 1) & 15)) if (kq[kq_head] & BTN_B) b = 1;
    return b;
}
/* kokoSD 1.3.1: restart detector. This record in slow RAM is not cleared at start (start-up code only clears
 * other sections), so after a crash that restarts the cartridge it still holds what Koko was doing; after a
 * power-on (or a start from the flash cart's menu) it holds something else. The title then says so. */
#define LIFE_MAGIC 0x4B4C4946u
static struct { uint32_t magic, check, keys, state, ntok, count; } life __attribute__((section(".bss.ewram")));
static uint32_t life_check(void) { return life.magic ^ life.keys * 31u ^ life.state * 977u ^ life.ntok * 7919u ^ life.count * 104729u ^ 0x9E3779B9u; }
static void life_note(void) { life.keys = keys_prev; life.state = dbg.state; life.ntok = dbg.ntok; life.check = life_check(); }
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
/* kokoSD 1.2: something Koko can read: the plain copy has more than spaces (Greek, Cyrillic, symbols: nothing) */
static int readable(void) { for (int i = 0; i < plen; i++) if (plain[i] != ' ') return 1; return 0; }
/* kokoSD 1.2: the chat being had: its file number (-1: never saved), first message and number of turns */
static int chat_id = -1, chat_turns;
static char chat_title[CHAT_TITLE];
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
    "START: send (with nothing typed, or nothing Koko can read, START does nothing).\n\n"
    "START+A: save the chat on the SD card. START+SELECT, Saved chats: open a saved chat and carry on where you "
    "left off (listed by their first message); START on a chat in the list: Open, Rename or Delete. "
    "Hold START and press Up / Down: scroll the chat.\n\n"
    "START with L or R (in either order): the first or second suggested word. Hold START and press Left / Right: "
    "move the cursor. START + SELECT: this menu.\n\n"
    "Accents: keep the letter's buttons held and press SELECT; SELECT again gives the next one (e: é è ë ê). "
    "They are shown as typed; Koko reads them as plain letters (é as e).\n\n"
    "B while Koko writes: stop him.\n\n"
    "The same typing as gbawriter, gbamp3 and the other apps.";
static const char ABOUT[] =
    "kokoSD 1.4: Koko the Chatbot with an SD card.\n\n"
    "Koko, a friendly husky, is a language model of 72 million two-bit weights that runs on the "
    "Game Boy Advance itself.\n"
    "\nFacts: \"who is\", \"what is\" and \"tell me about\" are answered from Simple English Wikipedia "
    "(CC BY-SA 4.0): the file koko/facts.dat on the SD card, or the cartridge's own smaller table. "
    "After such an answer, \"tell me more\" shows the article. Sums are worked out by the game. "
    "Replies marked Fact or Sum come from these helpers; the menu turns them off.\n"
    "\nKoko the Chatbot (the model, its training, the engine and Koko himself) is by rdsilver: "
    "rdsilver.itch.io/koko-the-chatbot. kokoSD (SD card, typing, saved chats, this look) is by Halim Jarrar.\n"
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
/* ---------------- kokoSD 1.2: saved chats ---------------- */
#define LIST_MAX 48
static char list_title[LIST_MAX][CHAT_TITLE] __attribute__((section(".bss.ewram")));
static int16_t list_id[LIST_MAX] __attribute__((section(".bss.ewram")));
static const char *list_item[LIST_MAX] __attribute__((section(".bss.ewram")));
static void new_chat_state(void) {
    koko_reset();
    ui_clear(); ui_say(L_KOKO, "Woof! I'm Koko. Say something!", 30);
    mlen = 0; msg[0] = 0; plen = 0; plain[0] = 0; dbg.ntok = 0;
    chat_id = -1; chat_turns = 0; chat_title[0] = 0;
}
/* Write the chat as it was when the message being typed was started (the draft is not part of it). */
static int save_chat(void) {
    while (stepping) work();
    uint32_t n = (uint32_t)(mark_ntok < HISTMAX ? mark_ntok : HISTMAX), sd_ = seed;
    if (!chat_write_begin(&chat_id, chat_title)) return 0;
    chat_io(&sd_, 4, 0); chat_io(&n, 4, 0);
    chat_io((void *)dbg.hist, n * 2, 0);
    ui_log_io(chat_io, 0, 1);
    koko_state_io(chat_io, 0, 1);
    return chat_end();
}
static int load_chat(int id) {
    char title[CHAT_TITLE];
    uint32_t n = 0, sd_ = 0;
    koko_reset();
    if (!chat_read_begin(id, title)) return 0;
    int ok = chat_io(&sd_, 4, 0) && chat_io(&n, 4, 0) && n <= HISTMAX;
    if (ok) {
        ok = chat_io((void *)dbg.hist, n * 2, 0);
        dbg.ntok = n;
    }
    ok = ok && ui_log_io(chat_io, 0, 0) && koko_state_io(chat_io, 0, 0);
    ok = chat_end() && ok;
    if (!ok) return 0;
    for (int i = 0; i < CHAT_TITLE; i++) chat_title[i] = title[i];
    chat_id = id; chat_turns = 1;
    mlen = 0; msg[0] = 0; plen = 0; plain[0] = 0;
    return 1;
}
static int title_less(int a, int b) { return fold_compare(list_title[a], list_title[b], 0) < 0; }
/* kokoSD 1.3: rename a saved chat: gbawriter typing in the field, START+A saves, START+B cancels.
 * Returns 1 when the new name was written. */
static int rename_chat(int id, const char *old) {
    static char name[CHAT_TITLE];
    int n = 0;
    for (; old[n] && n < UI_MSG_MAX; n++) name[n] = old[n];
    while (n > 0 && (name[n - 1] & 0xC0) == 0x80) n--;            /* no half letter at a cut */
    if (n > 0 && (unsigned char)name[n - 1] >= 0xC0) n--;
    name[n] = 0;
    writer_begin(name);
    ui_ask("Rename chat", "", old, "!START+A: save    START+B: cancel");
    ui_input(name, n, n);
    dbg.state = ST_RENAME;
    while (keys_prev) { vsync(); poll(); }       /* the press that chose Rename types nothing */
    hq_n = 0; kq_n = 0;
    uint32_t last = ticks();
    for (;;) {
        vsync(); poll();
        uint32_t now = ticks();
        if (now < last) last = now;
        if (now - last < 274) continue;
        unsigned frames = (now - last) / 274, f = 0, fl = 0;
        if (frames > 4) frames = 4;
        last = now;
        for (unsigned i = 0; i < hq_n; i++, f++) fl |= writer_frame(hq[i]);
        hq_n = 0;
        for (; f < frames; f++) fl |= writer_frame(keys_prev);
        if (fl & WRITER_TEXT) {
            const char *t = writer_text();
            for (n = 0; t[n] && n < UI_MSG_MAX; n++) name[n] = t[n];
            name[n] = 0;
            ui_input(name, n, (int)writer_caret());
        }
        ui_writer_status(writer_group(), writer_caps() ? "Caps" : writer_shift() ? "Shift" : "");
        if (fl & WRITER_START_B) { kq_n = 0; return 0; }
        if (fl & WRITER_START_A) {
            int k = 0;
            while (name[k] == ' ') k++;
            if (!name[k]) { ui_ask("Rename chat", "The name is empty", old, "!START+A: save    START+B: cancel"); ui_input(name, n, (int)writer_caret()); continue; }
            int ok = chat_rename(id, name + k);
            if (ok && id == chat_id) { int i = 0; for (; name[k + i] && i < CHAT_TITLE - 1; i++) chat_title[i] = name[k + i]; chat_title[i] = 0; }
            kq_n = 0;
            return ok;
        }
    }
}
/* kokoSD 1.3: START on a saved chat: Open, Rename, Delete. Returns 1 open, 2 renamed or deleted, 0 nothing. */
static int chat_options(int id, const char *title) {
    static int sel;
    const char *items[3] = { "Open", "Rename", "Delete" };
    sel = 0; kq_n = 0;
    for (;;) {
        dbg.state = ST_OPTIONS; dbg.menu = (uint32_t)sel;
        ui_menu(title, items, 3, sel);
        uint16_t d;
        do { vsync(); d = keys_down(); } while (!d);
        if (d & BTN_UP) sel = (sel + 2) % 3;
        else if (d & BTN_DOWN) sel = (sel + 1) % 3;
        else if (d & (BTN_B | BTN_START)) return 0;
        else if (d & BTN_A) {
            if (sel == 0) return 1;
            if (sel == 1) { rename_chat(id, title); return 2; }
            ui_ask("Delete chat", "Delete this chat?", title, "A: yes    B: no");
            dbg.state = ST_CONFIRM; kq_n = 0;
            for (;;) {
                vsync();
                d = keys_down();
                if (d & BTN_A) { if (chat_delete(id) && id == chat_id) chat_id = -1; return 2; }   /* the chat on screen is then unsaved again */
                if (d & (BTN_B | BTN_START)) break;
            }
        }
    }
}
/* The Saved chats list. Returns 1 when a chat was opened (it is then the current chat), 2 when it could not be. */
static int saved_chats(void) {
    int sel = 0, top = 0, n, order[LIST_MAX];
reload:
    ui_list("Saved chats", list_item, 0, 0, 0, "Reading the SD card...");
    n = chat_list(list_title, list_id, LIST_MAX);
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) { int v = order[i], j = i; while (j > 0 && title_less(v, order[j - 1])) { order[j] = order[j - 1]; j--; } order[j] = v; }
    for (int i = 0; i < n; i++) list_item[i] = list_title[order[i]];
    if (sel >= n) sel = n ? n - 1 : 0;
    kq_n = 0;
    for (;;) {
        if (sel < top) top = sel;
        if (sel >= top + 7) top = sel - 6;
        dbg.state = ST_LIST; dbg.menu = (uint32_t)sel; dbg.article = (uint32_t)n;
        ui_list("Saved chats", list_item, n, sel, top, sd_ready ? "No saved chats yet\nSTART+A while chatting saves one" : "No SD card");
        for (;;) {
            vsync();
            uint16_t d = keys_down();
            if (!d) continue;
            if ((d & BTN_UP) && n) { sel = (sel + n - 1) % n; break; }
            if ((d & BTN_DOWN) && n) { sel = (sel + 1) % n; break; }
            if ((d & BTN_LEFT) && n) { sel = sel > 7 ? sel - 7 : 0; break; }
            if ((d & BTN_RIGHT) && n) { sel = sel + 7 < n ? sel + 7 : n - 1; break; }
            if (d & BTN_B) return 0;
            int open = 0;
            if ((d & BTN_START) && n) {                   /* kokoSD 1.3: Open / Rename / Delete */
                int r = chat_options(list_id[order[sel]], list_item[sel]);
                if (r == 2) goto reload;
                if (!r) break;
                open = 1;
            }
            if (((d & BTN_A) && n) || open) {
                ui_list("Saved chats", list_item, n, sel, top, "");
                ui_message("Opening...");
                if (load_chat(list_id[order[sel]])) return 1;
                new_chat_state();
                return 2;                             /* could not be read: a new chat, with a note */
            }
        }
    }
}
/* ---------------- kokoSD 1.4: self test (crash hunting on the real cart) ----------------
 * Round after round until B: (1) every 1 MB of the ROM against the sums tools/fixrom.py wrote after the image
 * (a misread or overwritten cart memory shows here), (2) 24 lookups of real titles in the fact file, every
 * fourth with its article (SD card reads), then the 1 MB pieces the SuperCard's SD registers could overlay are
 * summed again, (3) a fixed sentence fed through Koko must give the state the native engine gives. The results
 * go to /koko/selftest.txt. */
extern char __rom_end[];
#define MODEL_CHECK 0x0cdc9b2fu                    /* engine/selftest: koko_h after the sentence below (native twin) */
static const uint16_t check_toks[] = { 1, 1546, 3021, 3, 7622, 384, 461, 1229, 14, 3049, 278, 266, 4582, 3972, 7952, 16, 2 };
static int rom_piece(unsigned i, uint32_t *s1, uint32_t *s2) {   /* sums of 1 MB piece i; 0 when there is no table */
    const volatile uint32_t *tab = (const volatile uint32_t *)(((uint32_t)__rom_end + 4095u) & ~4095u);
    if (tab[0] != 0x4D55534Bu || i >= tab[2]) return 0;   /* "KSUM" */
    uint32_t size = tab[1], at = i << 20, n = size - at < (1u << 20) ? (size - at) >> 2 : 1u << 18;
    const volatile uint32_t *p = (const volatile uint32_t *)(0x08000000u + at);
    uint32_t a = 0, b = 0;
    for (uint32_t k = 0; k < n; k++) { uint32_t w = p[k]; a += w; b += w * (k + 1); if (!(k & 4095)) poll(); }
    /* kokoSD 1.4.1: words the loader may have patched (SuperFW: header, WAITCNT constants) count as their originals */
    const volatile uint32_t *may = tab + 3 + 2 * tab[2];
    for (uint32_t j = 0; j < may[0] && j < 255; j++) {
        uint32_t o = may[1 + 2 * j], k = (o - at) >> 2;
        if (o < at || k >= n) continue;
        uint32_t d = may[2 + 2 * j] - p[k];
        a += d; b += d * (k + 1);
    }
    *s1 = a; *s2 = b;
    return tab[3 + 2 * i] == a && tab[4 + 2 * i] == b ? 1 : -1;
}
static unsigned rom_pieces(void) {
    const volatile uint32_t *tab = (const volatile uint32_t *)(((uint32_t)__rom_end + 4095u) & ~4095u);
    return tab[0] == 0x4D55534Bu ? tab[2] : 0;
}
static uint32_t model_check(void) {
    koko_reset();
    for (unsigned i = 0; i < sizeof check_toks / sizeof check_toks[0]; i++) koko_step(check_toks[i]);
    uint32_t c = 0;
    for (int i = 0; i < K_D; i++) c = c * 31u + (uint32_t)koko_h[i];
    return c;
}
#define st_log ((char *)list_title)                  /* (the Saved chats list's buffer: free during the test) */
#define ST_LOG_MAX ((int)sizeof list_title)
static int st_n;
static void st_add(const char *t) { while (*t && st_n < ST_LOG_MAX - 1) st_log[st_n++] = *t++; st_log[st_n] = 0; }
static void st_num(uint32_t v) { char b[12]; int k = 0; do { b[k++] = (char)('0' + v % 10); v /= 10; } while (v); while (k) { char c[2] = { b[--k], 0 }; st_add(c); } }
static void small_note(const char *a, const char *b) { ui_lines(a, b); }
static void self_test(void) {
    ui_ask("Self test", "Run the self test?", "The chat on screen closes: save it first", "A: start    B: back");
    dbg.state = ST_CONFIRM; kq_n = 0;
    for (;;) { vsync(); uint16_t d = keys_down(); if (d & BTN_A) break; if (d & (BTN_B | BTN_START)) return; }
    while (stepping) work();
    st_n = 0; st_log[0] = 0;
    st_add("kokoSD self test\n");
    unsigned pieces = rom_pieces(), round = 0, errors = 0, sd_ok = 0, sd_all = 0, model_bad = 0;
    uint32_t bad_mb = 0;                                         /* which 1 MB pieces ever differed */
    uint32_t r = seed | 1;
    static char line[64] __attribute__((section(".bss.ewram"))), det[96] __attribute__((section(".bss.ewram")));
    if (!pieces) st_add("no ROM sums in this build\n");
    dbg.state = ST_SELFTEST; dbg.article = 0; dbg.article_top = 0;
    for (;;) {
        round++;
        int bad_rom = 0, bad_sd = 0, bad_model = 0; uint32_t a, b;
        for (unsigned i = 0; i < pieces; i++) {                  /* (1) the whole ROM */
            ui_ask("Self test", "Checking the ROM...", "", "B: stop (after this round)");
            if (rom_piece(i, &a, &b) < 0) { bad_rom++; bad_mb |= 1u << (i & 31); st_add("round "); st_num(round); st_add(": ROM MB "); st_num(i); st_add(" differs\n"); }
        }
        if (sdf_ready) {                                          /* (2) the SD card, then the overlay pieces again */
            ui_ask("Self test", "Reading the SD card...", "", "B: stop (after this round)");
            for (int k = 0; k < 24; k++) {
                r = r * 1664525u + 1013904223u;
                sd_all++;
                if (sdf_probe(r >> 4, (k & 3) == 0)) sd_ok++; else { bad_sd++; st_add("round "); st_num(round); st_add(": SD lookup failed\n"); }
            }
            sdf_forget();
            static const unsigned overlay[] = { 16, 17, 24 };     /* 0x09000000, 0x09100000, 0x09800000 */
            for (unsigned j = 0; j < 3; j++)
                if (overlay[j] < pieces && rom_piece(overlay[j], &a, &b) < 0) { bad_rom++; bad_mb |= 1u << overlay[j]; st_add("round "); st_num(round); st_add(": ROM MB "); st_num(overlay[j]); st_add(" differs after SD reads\n"); }
        }
        ui_ask("Self test", "Running Koko...", "", "B: stop (after this round)");
        uint32_t m = model_check();                                /* (3) the model */
        if (m != MODEL_CHECK) { bad_model = 1; model_bad++; st_add("round "); st_num(round); st_add(": model state differs\n"); }
        errors += (unsigned)(bad_rom + bad_sd + bad_model);
        dbg.article = round; dbg.article_top = errors;
        int k = 0; const char *t = errors ? "Round " : "Round ";
        while (*t) line[k++] = *t++;
        { char b2[12]; int q = 0; unsigned v = round; do { b2[q++] = (char)('0' + v % 10); v /= 10; } while (v); while (q) line[k++] = b2[--q]; }
        t = errors ? ": errors found" : ": all good"; while (*t) line[k++] = *t++; line[k] = 0;
        k = 0;
        t = bad_rom ? "ROM: differs   " : pieces ? "ROM: ok   " : "ROM: no sums   "; while (*t) det[k++] = *t++;
        t = !sdf_ready ? "SD: no fact file   " : bad_sd ? "SD: failed   " : "SD: ok   "; while (*t) det[k++] = *t++;
        t = bad_model ? "Koko: differs" : "Koko: ok"; while (*t) det[k++] = *t++; det[k] = 0;
        ui_ask("Self test", line, det, "B: stop    (keeps going)");
        for (int f = 0; f < 60; f++) vsync();
        if (b_pressed()) break;
    }
    st_add("rounds "); st_num(round); st_add(", errors "); st_num(errors); st_add(", SD lookups "); st_num(sd_ok); st_add("/"); st_num(sd_all); st_add("\n");
    int saved = 0;
    if (sd_ready) {
        FIL lf; UINT done;
        f_mkdir("/koko");
        if (f_open(&lf, "/koko/selftest.txt", FA_WRITE | FA_CREATE_ALWAYS) == FR_OK) { saved = f_write(&lf, st_log, (UINT)st_n, &done) == FR_OK && done == (UINT)st_n; f_close(&lf); }
    }
    /* kokoSD 1.4.1: the details on screen too (with no card there is no log) */
    static char d1[64] __attribute__((section(".bss.ewram"))), d2[64] __attribute__((section(".bss.ewram")));
    int k = 0; const char *t;
    if (bad_mb) {
        t = "ROM differs in MB:"; while (*t) d1[k++] = *t++;
        for (int i = 0; i < 32 && k < 58; i++) if (bad_mb >> i & 1) { d1[k++] = ' '; if (i >= 10) d1[k++] = (char)('0' + i / 10); d1[k++] = (char)('0' + i % 10); }
    } else { t = pieces ? "ROM: ok" : "ROM: no sums"; while (*t) d1[k++] = *t++; }
    t = model_bad ? "   Koko: differs" : "   Koko: ok"; while (*t && k < 63) d1[k++] = *t++;
    d1[k] = 0; k = 0;
    t = !sd_ready ? "No SD card at start (" : !sdf_ready ? "SD card, no fact file (" : "SD card and fact file ("; while (*t) d2[k++] = *t++;
    t = sd_cart_name(); while (*t && k < 60) d2[k++] = *t++;
    d2[k++] = ')'; d2[k] = 0;
    ui_ask("Self test", errors ? "Errors found" : "No errors", d1, "");
    small_note(d2, saved ? "Saved: koko/selftest.txt   (any button)" : "Not saved (no SD card)   (any button)");
    kq_n = 0;
    while (!kq_n) { vsync(); poll(); }
    kq_n = 0;
}

/* START + SELECT while typing. Returns 1 for a new chat, 2 when a saved chat was opened (3: it failed). */
static int menu(void) {
    static int sel;
    const char *items[6] = { "New chat", "Saved chats", 0, "How to type", "About kokoSD", "Self test" };
    kq_n = 0;                                     /* presses made while typing are not menu choices */
    for (;;) {
        items[2] = helpers ? "Facts and sums: On" : "Facts and sums: Off";
        dbg.state = ST_MENU; dbg.menu = (uint32_t)sel;
        ui_menu("Menu", items, 6, sel);
        for (;;) {
            vsync();
            uint16_t d = keys_down();
            if (!d) continue;
            if (d & BTN_UP) { sel = (sel + 5) % 6; break; }
            if (d & BTN_DOWN) { sel = (sel + 1) % 6; break; }
            if (d & (BTN_B | BTN_START)) return 0;
            if (d & BTN_A) {
                if (sel == 0) return 1;
                if (sel == 1) { int r = saved_chats(); if (r) return r + 1; }
                if (sel == 2) { helpers = !helpers; ui_helpers(helpers); dbg.pure = (uint16_t)!helpers; }
                if (sel == 3) read_pages("How to type", HOW_TO_TYPE, (int)sizeof HOW_TO_TYPE - 1, ST_PAGE);
                if (sel == 4) read_pages("About kokoSD", ABOUT, (int)sizeof ABOUT - 1, ST_PAGE);
                if (sel == 5) { self_test(); if (dbg.article) return 1; }   /* it ran: a new chat (the model was used) */
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
/* Returns 1 to send the message, 0 to start a new chat, 2 when a saved chat was opened, 3 when it could not be. */
#define SUG_SIG() (nsug ? sug[0] * 3 + sug[nsug - 1] * 7 + nsug : 0)
static int edit(void) {
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
        int sig = SUG_SIG();
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
        if (fl & WRITER_START_ALONE) {                /* kokoSD 1.2: nothing typed (or nothing Koko can read): nothing */
            if (readable()) return 1;
            if (mlen) { ui_message("Koko cannot read that: type letters or numbers"); shown_sug = SUG_SIG(); }
        }
        if (fl & (WRITER_UP | WRITER_DOWN)) {         /* kokoSD 1.2: START+Up / Down scroll the chat */
            if (fl & WRITER_UP) ui_scroll(1);
            if (fl & WRITER_DOWN) ui_scroll(-1);
            if (ui_scrolled()) { ui_message("Older lines    START+Down: newer"); shown_sug = SUG_SIG(); }
            else { chips(); shown_sug = -1; }
        }
        if ((fl & WRITER_TEXT) && ui_scrolled()) { ui_scroll_home(); chips(); shown_sug = -1; }
        if (fl & WRITER_START_A) {                    /* kokoSD 1.2: save the chat (the note stays until the suggestions change) */
            if (!sd_ready) ui_message("No SD card: the chat cannot be saved");
            else if (!chat_turns) ui_message("Nothing to save yet");
            else {
                ui_header("saving...");
                int ok = save_chat();
                ui_header(0); shown_hint = -1;
                ui_message(ok ? "Chat saved (START+SELECT: Saved chats)" : "Could not save the chat");
            }
            shown_sug = SUG_SIG();
        }
        if (fl & WRITER_START_SELECT) {
            while (stepping) work();
            int m = menu();
            if (m == 1) return 0;
            if (m >= 2) return m;
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
    if (!chat_turns) { int i = 0; for (; i < mlen && i < CHAT_TITLE - 1; i++) chat_title[i] = msg[i]; chat_title[i] = 0; }   /* kokoSD 1.2 */
    chat_turns++;
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
    int stopped = b_pressed();                           /* B while Koko was still reading: no reply */
    koko_reply_begin();                                  /* repeated-phrase rule on for this reply */
    ui_reply_start(mark);
    while (!stopped && made < REPLY_MAX) {
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
        if (b_pressed()) { stopped = 1; break; }
    }
    koko_reply_end();
    ui_reply_button(0);
    ui_rate(made, ticks() - t0);
    if (article && !stopped) read_article();             /* B also means: no article */
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
    if (!sdf_ready) {                             /* kokoSD 1.4.2: why there are no facts from the card (FatFs codes) */
        static const char *const why[] = { "", "No SD card (FatFs error ", "No /koko/facts.dat (FatFs error ", "facts.dat could not be read",
                                           "koko/facts.dat is not a fact file", "facts.dat is for other weights", "facts.dat is empty" };
        static char t[56]; int k = 0, e = sdf_err & 15, fr = e == 1 ? sd_err : sdf_err >> 4;
        const char *a = why[e < 7 ? e : 0];
        while (*a) t[k++] = *a++;
        if (e <= 2) {
            if (fr >= 10) t[k++] = (char)('0' + fr / 10);
            t[k++] = (char)('0' + fr % 10);
            if (e == 1 && sd_init_err && sd_init_err < 100) {   /* kokoSD 1.4.3: which step of the card start failed */
                const char *s2 = ", start step "; while (*s2) t[k++] = *s2++;
                if (sd_init_err >= 10) t[k++] = (char)('0' + sd_init_err / 10);
                t[k++] = (char)('0' + sd_init_err % 10);
            }
            t[k++] = ')';
        }
        t[k] = 0;
        ui_title_note2(t);
    }
    if (life.magic == LIFE_MAGIC && life.check == life_check()) {   /* kokoSD 1.3.1: it was running a moment ago */
        static char t[48]; int k = 0;
        const char *a = "Restarted unexpectedly: keys ";
        while (*a) t[k++] = *a++;
        for (int sh = 8; sh >= 0; sh -= 4) t[k++] = "0123456789ABCDEF"[(life.keys >> sh) & 15];
        a = " state "; while (*a) t[k++] = *a++;
        t[k++] = (char)('0' + life.state % 10); t[k] = 0;
        ui_title_note(t);
        life.count++;
    } else life.count = 0;
    life.magic = LIFE_MAGIC; life.state = ST_TITLE; life.keys = 0; life.ntok = 0; life.check = life_check();
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
        int e = edit();
        if (e != 1) {                             /* new chat (0), a saved chat opened (2) or one that could not be (3) */
            while (stepping) work();
            if (e == 0) new_chat_state();
            if (e == 3) ui_say(L_NOTE, "That chat could not be opened.", 30);
            dbg.newchats++;
            while (keys_prev) { vsync(); poll(); }    /* the A that chose New chat types nothing */
            hq_n = 0;
            continue;
        }
        exchange();
        hq_n = 0;                                 /* kokoSD: presses during the reply (B: stop) are not typed */
        while (!kq_n) { vsync(); poll(); }        /* the reply stays up until a button is pressed; edit() gets that press */
    }
}
