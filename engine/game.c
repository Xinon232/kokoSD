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
#include "memory.h"
int sum_reply(const char *s, int n, char *out);   /* engine/sums.c */
int fact_reply(const char *msg, int n, uint16_t *ids, int max);   /* engine/facts.c */
FIL *chat_fil(void);                              /* engine/chats.c: its file object */
extern int fact_wide;                             /* kokoSD 2.4: the wider matching (engine/facts.c) */

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
/* kokoSD 2.0: keep-alive. A SuperCard's "ROM" is SDRAM, which only keeps what is read often enough on some carts:
 * our other apps are a few MB that are read all the time, Koko's 31 MB are mostly left alone while one types or
 * waits, and the model then reads all of it. So one word of every 512 bytes is read, the whole ROM about every
 * two seconds, spread over the calls of poll(). The loop runs from fast RAM (from ROM it cost a third of Koko's
 * speed). Off during the self test's idle check. */
extern char __rom_end[];
extern unsigned cart_kind;
static uint32_t ka_row, ka_last;
static int ka_off;
__attribute__((section(".iwram.text"), target("arm"), noinline))
static uint32_t ka_touch(uint32_t row, uint32_t n, uint32_t rows) {
    const volatile uint16_t *r = (const volatile uint16_t *)0x08000000u;
    while (n--) { (void)r[row << 8]; if (++row >= rows) row = 0; }
    return row;
}
static void keep_alive(void) {
    if (ka_off || cart_kind != CART_SUPERCARD) return;
    uint32_t now = ticks();
    if (now < ka_last) ka_last = now;              /* the timers were started again */
    uint32_t due = (now - ka_last) * 2u;           /* 16384 ticks a second, ~61500 rows: 2 a tick, all in ~2 s */
    if (due < 32) return;                          /* (in small batches) */
    ka_last = now;
    if (due > 4096) due = 4096;
    ka_row = ka_touch(ka_row, due, ((uint32_t)__rom_end - 0x08000000u) >> 9);
}
/* kokoSD 2.1: timers 2 and 3 time the buttons (debounce) and the typing engine's frames. If they were stopped,
 * or their count stood still, a button could never change again (START stayed "held": SELECT then opened the
 * menu, and nothing else answered). Started again here; the self test reports how often that happened. */
static uint8_t start_alone, sent_alone, b_after_send;   /* kokoSD 2.2: B pressed after START sent the message (poll) */
static uint16_t timer_fix;
static uint32_t tick_seen, tick_same;
static void timers_start(void) {
    REG_TM2CNT_H = 0; REG_TM3CNT_H = 0; REG_TM2CNT_L = 0; REG_TM3CNT_L = 0;
    REG_TM3CNT_H = 0x0084; REG_TM2CNT_H = 0x0083;
    timer_fix++; tick_same = 0;
}
static void timer_guard(void) {
    if ((REG_TM2CNT_H & 0xC7) != 0x83 || (REG_TM3CNT_H & 0xC4) != 0x84) { timers_start(); return; }
    uint32_t t = ticks();
    if (t != tick_seen) { tick_seen = t; tick_same = 0; }
    else if (++tick_same > 4000) timers_start();    /* thousands of polls with no tick: the count stands still */
}
static void poll(void) {                          /* called often, also from inside the model's work and long redraws */
    timer_guard();
    keep_alive();
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
    /* kokoSD 2.2: START let go with no other button pressed while it was held (a send), then B: that B stops the
     * reply, whenever the reply starts (the typing loop may see both in one go when it was busy) */
    if (d & BTN_START) { start_alone = 1; b_after_send = 0; }
    else if (d && (k & BTN_START)) start_alone = 0;
    if ((keys_prev & BTN_START) && !(k & BTN_START)) sent_alone = start_alone;
    if ((d & BTN_B) && sent_alone && !(k & BTN_START)) b_after_send = 1;
    if (k != keys_prev && hq_n < 32) hq[hq_n++] = k;
    if (k != keys_prev) keys_prev = k;
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
static int helpers = 1;                           /* fact table and calculator on; off = every word comes from the model;
                                                   * kokoSD 2.4: 2 = facts as a hint (Koko is told the fact and answers) */
static int kb_mode EW;                            /* kokoSD 2.5: typing on the original on-screen keyboard (0: button chords; set in main) */
static int words_on = 0;                          /* kokoSD 2.4: word memory from the fact file (when it has one); off by default:
                                                   * in tests it made replies sound like Wikipedia, not more often right */
/* kokoSD 2.4: what Koko remembers about you, told at the start of each new chat (engine/memory.c). Fed in the
 * background while the first message is typed; the editor's mark moves past it when it is in. */
#define PRE_MAX 200
static uint16_t pre[PRE_MAX] EW;
static int npre, pre_fed, pre_mark;

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
    if (pre_fed < npre) { feed_begin(pre[pre_fed++]); pre_mark = pre_fed == npre; return 1; }
    if (pre_mark) { koko_mark(); mark_ntok = (int)dbg.ntok; pre_mark = 0; return 1; }
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
static const char HOW_TO_TYPE_KB[] =                /* kokoSD 2.5 */
    "Keyboard (Settings > Typing):\n"
    "D-pad: move on the keyboard (hold to repeat). A: type the key. B: delete (hold to repeat).\n\n"
    "The arrow key (right end of the digits): capitals on / off. The e-acute key (before the delete key): an accent "
    "on the last letter; again: the next accent (e: \xc3\xa9 \xc3\xa8 \xc3\xab \xc3\xaa). Koko reads them as plain letters.\n\n"
    "L / R: the first or second suggested word (shown above the keyboard).\n\n"
    "START: send. On an empty line, START offers a ready-made message; START again sends it (delete it and press START for the next one).\n\n"
    "SELECT: the menu (New chat, Saved chats, Settings ...).\n\n"
    "Hold START and press A: save the chat. Hold START and press Up / Down: scroll the chat.\n\n"
    "B while Koko writes: stop him.\n\n"
    "Settings > Typing: Button chords switches to the typing of gbawriter, gbamp3 and the other apps.";
static const char ABOUT[] =
    "kokoSD 2.5: Koko the Chatbot with an SD card.\n\n"
    "Koko, a friendly husky, is a language model of 72 million two-bit weights that runs on the "
    "Game Boy Advance itself.\n"
    "\nFacts: \"who is\", \"what is\" and \"tell me about\" are answered from Simple English Wikipedia "
    "(CC BY-SA 4.0): the file koko/facts.dat on the SD card. "
    "After such an answer, \"tell me more\" shows the article. Sums are worked out by the game. "
    "Replies marked Fact or Sum come from these helpers; the menu turns them off. With Facts: As a hint, Koko is "
    "told the fact and answers in his own words (marked Hint): friendlier, but he can get it wrong.\n"
    "\nMemory: what you tell Koko about yourself (\"my name is ...\", \"I live in ...\", \"I like ...\") is kept in "
    "koko/memory.txt and told to him at the start of each new chat. The menu shows it, turns it off or forgets it.\n"
    "\nWord memory: which words follow which in the Wikipedia text on the card (in facts.dat) is mixed into Koko's "
    "choice of the next word.\n"
    "\nTyping: the original Koko's on-screen keyboard (SELECT: menu), or gbawriter's button chords (Settings > Typing).\n"
    "\nKoko the Chatbot (the model, its training, the engine and Koko himself) is by rdsilver: "
    "rdsilver.itch.io/koko-the-chatbot. kokoSD (SD card, typing, saved chats, this look) is by halimj.\n"
    "\nCode: Koko (MIT), gbaSD and gbawriter typing by Halim Jarrar (GPL-3.0-or-later); the cartridge "
    "as a whole is GPL-3.0-or-later. Weights, fact file and artwork: CC BY-SA 4.0. "
    "Fonts: the SuperFW font by David Guillen Fandos (GPL-3.0-or-later; UNSCII and GNU Unifont glyphs) and "
    "gbamp3's 5x7 font; accent folding from GBAVocab (Halim Jarrar, GPL-3.0-or-later).";
/* A page of text to read (an article, How to type, About): Up / Down scroll, Left / Right / L / R turn a page,
 * B (or A, START) goes back. */
static const char *rom_text;                      /* kokoSD 2.3: How to type / About: the text in the cartridge */
static const char *rom_src(uint32_t at, uint32_t n) { (void)n; return rom_text + at; }
static uint16_t read_pages(const char *title, ui_src src, int n, int state) {
    int lines = ui_article_wrap(src, (uint32_t)n), rows = ui_article_rows(), top = 0, last = lines > rows ? lines - rows : 0;
    kq_n = 0;                                     /* presses made before the page opened are not page turns */
    dbg.state = (uint32_t)state; dbg.article = (uint32_t)lines; dbg.article_top = 0;
    ui_article(title, 0);                         /* presses while it draws count (polled between rows) */
    for (;;) {
        vsync();
        uint16_t d = keys_down();
        int was = top;
        if (d & (BTN_DOWN | BTN_RIGHT | BTN_R)) top += (d & BTN_DOWN) ? 3 : rows;
        if (d & (BTN_UP | BTN_LEFT | BTN_L)) top -= (d & BTN_UP) ? 3 : rows;
        if (top > last) top = last;
        if (top < 0) top = 0;
        if (top != was) { ui_article(title, top); dbg.article_top = (uint32_t)top; }
        if (d & (BTN_B | BTN_START | BTN_A | BTN_SELECT)) return d;
    }
}
/* kokoSD 2.4: Settings > What Koko remembers: the list; SELECT forgets it all */
static void memory_page(void) {
    char *t = mem_buf;
    for (;;) {
        int k = 0;
        const char *h = mem_on ? "Memory is on: Koko notes what you tell him about yourself and is told it at the start of each new chat.\n"
                               : "Memory is off (Settings): nothing is noted and new chats start without it.\n";
        while (*h) t[k++] = *h++;
        h = "SELECT: forget everything.  B: back.\n"
            "It is kept in koko/memory.txt on the SD card (one sentence a line: it can be changed on a computer).\n\n";
        while (*h) t[k++] = *h++;
        if (mem_count()) { h = "Koko remembers:\n"; while (*h) t[k++] = *h++; k += mem_text(2, t + k, MEM_BUF - k - 1); }
        else { h = "Koko remembers nothing yet. Say \"My name is ...\", \"I live in ...\", \"I like ...\", \"My favorite ... is ...\"."; while (*h) t[k++] = *h++; }
        t[k] = 0;
        rom_text = t;
        uint16_t d = read_pages("Memory", rom_src, k, ST_PAGE);
        if (d & BTN_SELECT) { mem_forget(); mem_save(); continue; }
        return;
    }
}
/* ---------------- kokoSD 1.2: saved chats ---------------- */
#define LIST_MAX 48
static char list_title[LIST_MAX][CHAT_TITLE] __attribute__((section(".bss.ewram")));
static int16_t list_id[LIST_MAX] __attribute__((section(".bss.ewram")));
static const char *list_item[LIST_MAX] __attribute__((section(".bss.ewram")));
static void memory_prefix(void) {               /* kokoSD 2.4: "My name is Sam. ..." / "Got it! Your name is Sam. ..." */
    char *t = mem_buf;
    npre = pre_fed = pre_mark = 0;
    if (!mem_on || !mem_count()) return;
    int n = mem_text(0, t, MEM_BUF);
    pre[npre++] = TOK_USER;
    npre += koko_encode(t, n, pre + npre, PRE_MAX / 2 - 2);
    pre[npre++] = TOK_KOKO;
    const char *g = "Got it! "; int k = 0;
    while (g[k]) { t[k] = g[k]; k++; }
    n = k + mem_text(1, t + k, MEM_BUF - k);
    npre += koko_encode(t, n, pre + npre, PRE_MAX - npre);
}
static void new_chat_state(void) {
    koko_reset();
    memory_prefix();
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
        for (uint32_t i = 0; ok && i < n; i++) if (dbg.hist[i] >= K_V) ok = 0;   /* kokoSD 2.1: a damaged file */
        dbg.ntok = ok ? n : 0;
    }
    ok = ok && ui_log_io(chat_io, 0, 0) && koko_state_io(chat_io, 0, 0);
    ok = chat_end() && ok;
    if (!ok) return 0;
    for (int i = 0; i < CHAT_TITLE; i++) chat_title[i] = title[i];
    chat_id = id; chat_turns = 1;
    npre = pre_fed = pre_mark = 0;                /* kokoSD 2.4: a saved chat goes on as it was */
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
static void range_sums(const volatile uint32_t *tab, uint32_t at, uint32_t n, uint32_t *s1, uint32_t *s2);
static int rom_piece(unsigned i, uint32_t *s1, uint32_t *s2) {   /* sums of 1 MB piece i; 0 when there is no table */
    const volatile uint32_t *tab = (const volatile uint32_t *)(((uint32_t)__rom_end + 4095u) & ~4095u);
    if (tab[0] != 0x4D55534Bu || i >= tab[2]) return 0;   /* "KSUM" */
    uint32_t size = tab[1], at = i << 20, n = size - at < (1u << 20) ? (size - at) >> 2 : 1u << 18;
    range_sums(tab, at, n, s1, s2);
    return tab[3 + 2 * i] == *s1 && tab[4 + 2 * i] == *s2 ? 1 : -1;
}
/* the two sums of n words from ROM offset at, with the words a loader may change counted as their originals */
static void range_sums(const volatile uint32_t *tab, uint32_t at, uint32_t n, uint32_t *s1, uint32_t *s2) {
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
    /* kokoSD 2.0: on a SuperCard, does the cart keep what is not read? The ROM is left alone for 90 s (keep-alive
     * off, only the few KB of this loop are read), then every 1 MB is summed again. */
    uint32_t idle_mb = 0; int idle_run = 0;
    if (pieces && cart_kind == CART_SUPERCARD) {
        uint32_t a0, b0;
        ka_off = 1; idle_run = 1;
        for (int left = 90; left > 0 && idle_run; left--) {
            char t[40]; int k = 0; const char *q = "ROM left unread: ";
            while (*q) t[k++] = *q++;
            if (left >= 10) t[k++] = (char)('0' + left / 10);
            t[k++] = (char)('0' + left % 10); t[k++] = ' '; t[k++] = 's'; t[k] = 0;
            ui_ask("Self test", t, "Does the cart keep data that is not read?", "B: skip this step");
            for (int f = 0; f < 60; f++) { vsync(); if (b_pressed()) { idle_run = 0; break; } }
        }
        if (idle_run) {
            ui_ask("Self test", "Checking the ROM...", "after 90 s unread", "");
            for (unsigned i = 0; i < pieces; i++) if (rom_piece(i, &a0, &b0) < 0) idle_mb |= 1u << (i & 31);
            st_add(idle_mb ? "ROM after 90 s unread: differs\n" : "ROM after 90 s unread: ok\n");
        }
        ka_off = 0;
    }
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
    st_add("timers started again: "); st_num(timer_fix); st_add("\n");     /* kokoSD 2.1 */
    st_add("rounds "); st_num(round); st_add(", errors "); st_num(errors); st_add(", SD lookups "); st_num(sd_ok); st_add("/"); st_num(sd_all); st_add("\n");
    int saved = 0;
    if (sd_ready) {
        FIL *lf = chat_fil(); UINT done;                 /* kokoSD 2.5: the chats' file object: a FIL on the stack (550 bytes)
                                                          * left too little for Koko's own check when opened from the keyboard */
        f_mkdir("/koko");
        if (f_open(lf, "/koko/selftest.txt", FA_WRITE | FA_CREATE_ALWAYS) == FR_OK) { saved = f_write(lf, st_log, (UINT)st_n, &done) == FR_OK && done == (UINT)st_n; f_close(lf); }
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
    ui_ask("Self test", errors || idle_mb ? "Errors found" : "No errors", d1, "");
    small_note(d2, saved ? "Saved: koko/selftest.txt   (any button)" : "Not saved (no SD card)   (any button)");
    if (idle_run) {                                             /* kokoSD 2.0 */
        static char d3[64] __attribute__((section(".bss.ewram")));
        int j = 0; t = idle_mb ? "After 90 s unread, MB:" : "After 90 s unread: ROM ok"; while (*t) d3[j++] = *t++;
        for (int i = 0; i < 32 && j < 58; i++) if (idle_mb >> i & 1) { d3[j++] = ' '; if (i >= 10) d3[j++] = (char)('0' + i / 10); d3[j++] = (char)('0' + i % 10); }
        d3[j] = 0;
        ui_title_note2(d3);
    }
    kq_n = 0;
    while (!kq_n) { vsync(); poll(); }
    kq_n = 0;
}

static void words_set(void) {                     /* kokoSD 2.4 */
    koko_words = words_on && sdf_words_ready() ? sdf_words : 0;
    koko_words_mix = 32;                          /* 64 was best on held-out article text; in chat it pulled Koko toward article wording */
}
/* kokoSD 2.4: Settings > What these settings do */
static const char SETTINGS_HELP[] =
    "FACTS (default: Read out)\n"
    "Read out: when you ask \"who is\", \"what is\" or \"tell me about\", Koko says the first sentence of the "
    "Simple English Wikipedia article (marked Fact). It is always right. \"tell me more\" then opens the article.\n"
    "As a hint: the sentence is put into your message, before your question, and Koko answers in his own words "
    "(marked Hint). Friendlier, often uses the fact, but he can still get things wrong (names, dates). It takes "
    "10 to 25 seconds longer, as Koko reads the fact first.\n"
    "Off: no facts and no sums: every word is Koko's own (and often made up).\n\n"
    "FACT MATCHING (default: Wide)\n"
    "Wide: Koko finds the article for many ways of asking: \"can you tell me about\", \"have you heard of\", "
    "\"do you know\", \"what does ... mean\", \"info about\", and with extra words: \"the singer Taylor Swift\", "
    "\"the planet Mercury\". In a test 41 of 50 such questions found their article (Strict: 19).\n"
    "Strict: only the plain ways of asking of kokoSD 2.3.\n\n"
    "MEMORY (default: On)\n"
    "Koko notes what you tell him about yourself: \"my name is ...\", \"call me ...\", \"I am 10 years old\", "
    "\"I live in ...\", \"I like ...\", \"my favorite ... is ...\", \"I have a ...\". At each new chat he is told it "
    "while you type your first message. He uses your name well (\"Hi Lena!\"), other things less often. "
    "It is kept in koko/memory.txt (one sentence a line). What Koko remembers: see it, SELECT forgets it all. "
    "Saved chats go on without it.\n\n"
    "WORD MEMORY (default: Off)\n"
    "For each word Koko writes, the fact file is asked which words usually follow his last words in Wikipedia, "
    "and that is mixed into his choice. His replies sound more like an encyclopedia (\"Germany was founded in ...\"), "
    "but in tests they were not more often right. On a slow SD card it may make him slower. Needs the 2.4 fact file.\n\n"
    "TYPING (default: Keyboard)\n"
    "Keyboard: the original Koko's on-screen keyboard. D-pad: move, A: type the key, B: delete, L / R: the "
    "suggested words, START: send (on an empty line: a ready-made message), SELECT: the menu. The arrow key turns capitals on and "
    "off, the e-acute key puts an accent on the last letter (again: the next accent).\n"
    "Button chords: the typing of gbawriter and the other apps (How to type).\n\n"
    "Settings are kept in koko/settings.txt.";
/* kokoSD 2.4: the settings are kept in /koko/settings.txt ("facts=1" ...: 0 off, 1 read out, 2 as a hint) */
static int set_val(const char *t, int n, const char *key, int dflt) {
    for (int i = 0; i < n; i++) {
        int k = 0; while (key[k] && i + k < n && t[i + k] == key[k]) k++;
        if (!key[k] && (i == 0 || t[i - 1] == '\n') && i + k + 1 < n + 1 && t[i + k] == '=' && i + k + 1 < n) {
            int c = t[i + k + 1] - '0'; return c >= 0 && c <= 2 ? c : dflt;
        }
    }
    return dflt;
}
static void settings_load(void) {
    UINT got = 0;
    if (!sd_ready || f_open(chat_fil(), "/koko/settings.txt", FA_READ) != FR_OK) return;
    int ok = f_read(chat_fil(), mem_buf, 200, &got) == FR_OK;
    f_close(chat_fil());
    if (!ok) return;
    helpers = set_val(mem_buf, (int)got, "facts", 1); fact_wide = set_val(mem_buf, (int)got, "matching", 1) & 1;
    mem_on = set_val(mem_buf, (int)got, "memory", 1) & 1; words_on = set_val(mem_buf, (int)got, "words", 0) & 1;
    kb_mode = set_val(mem_buf, (int)got, "keyboard", 1) & 1;
    ui_helpers(helpers); dbg.pure = (uint16_t)!helpers;
}
static void settings_save(void) {
    if (!sd_ready) return;
    char *t = mem_buf; int n = 0;
    const char *k[5] = { "facts=", "matching=", "memory=", "words=", "keyboard=" }; int v[5] = { helpers, fact_wide, mem_on, words_on, kb_mode };
    for (int i = 0; i < 5; i++) { for (const char *p = k[i]; *p; p++) t[n++] = *p; t[n++] = (char)('0' + v[i]); t[n++] = '\n'; }
    UINT done; f_mkdir("/koko");
    if (f_open(chat_fil(), "/koko/settings.txt", FA_WRITE | FA_CREATE_ALWAYS) != FR_OK) return;
    f_write(chat_fil(), t, (UINT)n, &done);
    f_close(chat_fil());
}
/* kokoSD 2.4: the menu's Settings: the four 2.4 features, each on or off (A changes the row) */
static void settings(void) {
    int sel = 0, changed = 0;
    const char *items[7];
    static char mem_item[32];
    kq_n = 0;
    for (;;) {
        items[0] = helpers == 1 ? "Facts: Read out" : helpers == 2 ? "Facts: As a hint" : "Facts and sums: Off";
        items[1] = fact_wide ? "Fact matching: Wide" : "Fact matching: Strict";
        items[2] = mem_on ? "Memory: On" : "Memory: Off";
        items[3] = !sdf_words_ready() ? "Word memory: none on card" : words_on ? "Word memory: On" : "Word memory: Off";
        { const char *a = "What Koko remembers ("; int k = 0, c = mem_count();
          while (*a) mem_item[k++] = *a++;
          if (c >= 10) mem_item[k++] = (char)('0' + c / 10);
          mem_item[k++] = (char)('0' + c % 10); mem_item[k++] = ')'; mem_item[k] = 0; }
        items[4] = kb_mode ? "Typing: Keyboard" : "Typing: Button chords";
        items[5] = mem_item;
        items[6] = "What these settings do";
        dbg.state = ST_MENU; dbg.menu = (uint32_t)(16 + sel);
        ui_menu("Settings", items, 7, sel);
        for (;;) {
            vsync();
            uint16_t d = keys_down();
            if (!d) continue;
            if (d & BTN_UP) { sel = (sel + 6) % 7; break; }
            if (d & BTN_DOWN) { sel = (sel + 1) % 7; break; }
            if (d & (BTN_B | BTN_START)) { if (changed) settings_save(); return; }
            if (d & BTN_A) {
                if (sel == 0) { helpers = (helpers + 1) % 3; ui_helpers(helpers); dbg.pure = (uint16_t)!helpers; }
                if (sel == 1) fact_wide = !fact_wide;
                if (sel == 2) mem_on = !mem_on;
                if (sel == 3 && sdf_words_ready()) { words_on = !words_on; words_set(); }
                if (sel == 4) kb_mode = !kb_mode;
                if (sel == 5) memory_page();
                if (sel == 6) { rom_text = SETTINGS_HELP; read_pages("What these settings do", rom_src, (int)sizeof SETTINGS_HELP - 1, ST_PAGE); }
                changed |= sel < 5;
                kq_n = 0;
                break;
            }
        }
    }
}
/* START + SELECT while typing. Returns 1 for a new chat, 2 when a saved chat was opened (3: it failed). */
static int menu(void) {
    static int sel;
    const char *items[6] = { "New chat", "Saved chats", "Settings", "How to type", "About kokoSD", "Self test" };
    kq_n = 0;                                     /* presses made while typing are not menu choices */
    for (;;) {
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
                if (sel == 2) settings();                /* kokoSD 2.4 */
                if (sel == 3) { if (kb_mode) { rom_text = HOW_TO_TYPE_KB; read_pages("How to type", rom_src, (int)sizeof HOW_TO_TYPE_KB - 1, ST_PAGE); }
                                else { rom_text = HOW_TO_TYPE; read_pages("How to type", rom_src, (int)sizeof HOW_TO_TYPE - 1, ST_PAGE); } }
                if (sel == 4) { rom_text = ABOUT; read_pages("About kokoSD", rom_src, (int)sizeof ABOUT - 1, ST_PAGE); }
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
/* Returns 1 to send the message, 0 to start a new chat, 2 when a saved chat was opened, 3 when it could not be,
 * 4 when Settings > Typing was changed (kokoSD 2.5). */
#define SUG_SIG() (nsug ? sug[0] * 3 + sug[nsug - 1] * 7 + nsug : 0)
__attribute__((noinline)) static int edit(void) {
    int shown_sug = -1, shown_hint = -1;
    writer_begin(msg);
    refold();
    ui_header(0);
    ui_typing(msg, mlen, mlen);
    koko_mark(); mark_ntok = (int)dbg.ntok;
    nfed = nwant = user_fed = stepping = dirty = nsug = 0; sug_stale = 1;
    dbg.state = ST_EDIT; dbg.nsug = 0;
    kq_n = 0;
    sent_alone = 0; b_after_send = 0;            /* kokoSD 2.2: a send from before does not count */
    uint32_t last = ticks();
    for (;;) {
        int busy = work();
        dbg.state = busy ? ST_EDIT : ST_IDLE;
        int sig = SUG_SIG();
        /* kokoSD 2.2: while the suggestions are being worked out again (after each letter) the old ones stay on
         * screen; before, the line was cleared for a moment and drawn again: it flickered at every letter */
        if (sig != shown_sug && !sug_stale) { chips(); shown_sug = sig; }
        if (!busy) vsync();
        poll();
        uint32_t now = ticks();
        if (now < last) last = now;                    /* the timers were started again */
        if (now - last < 274) continue;
        unsigned frames = (now - last) / 274;
        if (frames > 4) frames = 4;
        last = now;
        unsigned fl = 0, f = 0;
        /* every change, in order (also while busy). kokoSD 2.2: up to START alone (send) only: what came after it
         * is not typing (a B there deleted the message's last letter); a B among it stops the reply at once */
        unsigned hi = 0;
        static char before[MSG_MAX + 1] EW;              /* the text before the change that sent it */
        for (; hi < hq_n && !(fl & WRITER_START_ALONE); hi++, f++) {
            if (hq[hi] & BTN_B) { const char *t = writer_text(); int k = 0; for (; t[k] && k < MSG_MAX; k++) before[k] = t[k]; before[k] = 0; }
            fl |= writer_frame(hq[hi]);
        }
        if (fl & WRITER_START_ALONE) {
            /* START let go and B pressed seen in one change (the loop was busy a few frames): the B is not a
             * backspace in the message being sent (poll() noted it: it stops the reply) */
            if (hi && (hq[hi - 1] & BTN_B)) { writer_set(before); fl |= WRITER_TEXT; }
        } else for (; f < frames; f++) {
            unsigned g;
            if (keys_prev & BTN_B) { const char *t = writer_text(); int k = 0; for (; t[k] && k < MSG_MAX; k++) before[k] = t[k]; before[k] = 0; }
            fl |= g = writer_frame(keys_prev);
            if ((g & WRITER_START_ALONE) && (keys_prev & BTN_B)) { writer_set(before); fl |= WRITER_TEXT; }
            if (g & WRITER_START_ALONE) break;
        }
        hq_n = 0;
        int changed = 0;
        if (fl & WRITER_TEXT) { msg_from_writer(); changed = 1; }
        if (fl & WRITER_PAGE_PREV) { if (nsug > 0) { accept(0); writer_set(msg); changed = 1; } }
        if (fl & WRITER_PAGE_NEXT) { if (nsug > 1) { accept(1); writer_set(msg); changed = 1; } }
        if (fl & WRITER_START_ALONE) {                /* kokoSD 1.2: nothing typed (or nothing Koko can read): nothing */
            if (readable()) return 1;
            sent_alone = 0; b_after_send = 0;         /* (nothing sent: no reply to stop) */
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
            /* kokoSD 2.1: the presses made in the menu (and its pages, Saved chats, Rename) are not typing: B there
             * deleted a letter here, START sent the message. Wait for every button to be let go, then forget them. */
            while (keys_prev) { vsync(); poll(); }
            hq_n = 0; kq_n = 0;
            if (kb_mode) return 4;                           /* kokoSD 2.5: Settings > Typing changed: the keyboard */
            writer_reset_keys();
            last = ticks();
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

/* ---------------- kokoSD 2.5: typing on the original Koko's on-screen keyboard ---------------- */
static const char *const quick[] = { "hi", "who are you", "tell me a story about a fox", "what is the capital of France?",
                                     "can you tell me a joke" };       /* START on an empty line (as the original Koko) */
static const char *const ACC[] = {                 /* the accent key: the last letter, then each of these in turn */
    "a\0\xc3\xa0\0\xc3\xa1\0\xc3\xa2\0\xc3\xa4\0\xc3\xa3\0\xc3\xa5\0", "e\0\xc3\xa9\0\xc3\xa8\0\xc3\xab\0\xc3\xaa\0",
    "i\0\xc3\xad\0\xc3\xac\0\xc3\xae\0\xc3\xaf\0", "o\0\xc3\xb3\0\xc3\xb2\0\xc3\xb4\0\xc3\xb6\0\xc3\xb5\0",
    "u\0\xc3\xba\0\xc3\xb9\0\xc3\xbb\0\xc3\xbc\0", "c\0\xc3\xa7\0", "n\0\xc3\xb1\0", "y\0\xc3\xbd\0\xc3\xbf\0",
    "A\0\xc3\x80\0\xc3\x81\0\xc3\x82\0\xc3\x84\0\xc3\x83\0\xc3\x85\0", "E\0\xc3\x89\0\xc3\x88\0\xc3\x8b\0\xc3\x8a\0",
    "I\0\xc3\x8d\0\xc3\x8c\0\xc3\x8e\0\xc3\x8f\0", "O\0\xc3\x93\0\xc3\x92\0\xc3\x94\0\xc3\x96\0\xc3\x95\0",
    "U\0\xc3\x9a\0\xc3\x99\0\xc3\x9b\0\xc3\x9c\0", "C\0\xc3\x87\0", "N\0\xc3\x91\0", 0 };
static int last_char(void) {                       /* start of the message's last (UTF-8) character */
    int i = mlen - 1;
    while (i > 0 && ((unsigned char)msg[i] & 0xC0) == 0x80) i--;
    return i < 0 ? 0 : i;
}
static int accent_last(void) {
    if (!mlen) return 0;
    int st = last_char(), n = mlen - st;
    for (int a = 0; ACC[a]; a++) {
        const char *p = ACC[a], *first = p;
        while (*p) {
            int l = 0; while (p[l]) l++;
            if (l == n) { int eq = 1; for (int j = 0; j < n; j++) if (p[j] != msg[st + j]) eq = 0;
                if (eq) {
                    const char *nx = p + l + 1; if (!*nx) nx = first;           /* the next accent, after the last: the letter */
                    int nl = 0; while (nx[nl]) nl++;
                    if (st + nl > MSG_MAX) return 0;
                    for (int j = 0; j < nl; j++) msg[st + j] = nx[j];
                    mlen = st + nl; msg[mlen] = 0;
                    return 1;
                } }
            p += l + 1;
        }
    }
    return 0;
}
/* Returns as edit(): 1 send, 0 new chat, 2 a saved chat opened, 3 it could not be. */
__attribute__((noinline)) static int edit_kb(void) {   /* (not inlined into main: its buffers would take stack from the self test) */
    static int kx EW, ky EW, upper EW, q EW;            /* (slow RAM: the fast RAM is full) */
    int shown_sug = -1, shown_empty = -1, st_used = 0;
    uint16_t prev = keys_prev;
    uint32_t held_since = ticks(), last_rep = 0;
    refold();
    ui_header(0);
    ui_typing_kb(msg, mlen, kx, ky, upper);
    koko_mark(); mark_ntok = (int)dbg.ntok;
    nfed = nwant = user_fed = stepping = dirty = nsug = 0; sug_stale = 1;
    dbg.state = ST_EDIT; dbg.nsug = 0; dbg.caret = (uint32_t)mlen;
    kq_n = 0;                                     /* (hq: presses made while the keyboard was drawn are typing) */
    sent_alone = 0; b_after_send = 0;
    for (;;) {
        int busy = work();
        dbg.state = busy ? ST_EDIT : ST_IDLE;
        int sig = SUG_SIG();
        if (sig != shown_sug && !sug_stale) { chips(); shown_sug = sig; }
        if ((mlen > 0) != shown_empty) { shown_empty = mlen > 0; ui_hint(mlen > 0); }
        if (!busy) vsync();
        poll();
        /* every change of the buttons, in order; then the repeat of a held direction or B */
        int n = (int)hq_n, act = 0;
        static uint16_t evs[33] EW;                  /* (not on the stack: the self test, opened from here, needs it all) */
        for (int i = 0; i < n; i++) evs[i] = hq[i];
        hq_n = 0;
        uint32_t now = ticks();
        if (now < held_since) held_since = now;
        if (!n && (prev & (BTN_LEFT | BTN_RIGHT | BTN_UP | BTN_DOWN | BTN_B)) && !(prev & BTN_START)
            && now - held_since > (busy ? 6u : 14u) * 274u && now - last_rep >= 4u * 274u) {
            evs[n++] = 0xFFFF; last_rep = now;                          /* a repeat */
        }
        for (int i = 0; i < n && !act; i++) {
            uint16_t k = evs[i], dn;
            if (k == 0xFFFF) { k = prev; dn = prev & (BTN_LEFT | BTN_RIGHT | BTN_UP | BTN_DOWN | BTN_B); }
            else { dn = (uint16_t)(k & ~prev); uint16_t upm = (uint16_t)(prev & ~k); prev = k; held_since = now; last_rep = now;
                if (dn & BTN_START) st_used = 0;
                if ((upm & BTN_START) && !st_used && !(k & BTN_START)) {     /* START alone: send, or a ready-made message */
                    if (readable()) { act = 1; break; }
                    if (!mlen) {
                        q = q % 5 + 1;                               /* 1..5: the next of the five */
                        for (mlen = 0; quick[q - 1][mlen]; mlen++) msg[mlen] = quick[q - 1][mlen];
                        msg[mlen] = 0; refold(); retarget(); ui_input(msg, mlen, mlen); dbg.caret = (uint32_t)mlen;
                    } else ui_message("Koko cannot read that: type letters or numbers");
                    sent_alone = 0; b_after_send = 0;
                    continue;
                }
            }
            if (!dn) continue;
            seed = seed * 1664525u + REG_TM0CNT_L + 1013904223u;
            if (k & BTN_START) {                                     /* START held: kokoSD's chords */
                if (!(dn & ~BTN_START)) continue;
                st_used = 1;
                if (dn & BTN_A) {
                    if (!sd_ready) ui_message("No SD card: the chat cannot be saved");
                    else if (!chat_turns) ui_message("Nothing to save yet");
                    else { ui_header("saving..."); int ok = save_chat(); ui_header(0);
                           ui_message(ok ? "Chat saved (SELECT: Saved chats)" : "Could not save the chat"); }
                    shown_sug = -2;
                }
                if (dn & (BTN_UP | BTN_DOWN)) {
                    ui_scroll((dn & BTN_UP) ? 1 : -1);
                    if (ui_scrolled()) ui_message("Older lines    START+Down: newer"); else chips();
                    shown_sug = -2;
                }
                if (dn & BTN_SELECT) act = 2;
                continue;
            }
            if (dn & BTN_SELECT) { act = 2; break; }                 /* SELECT: the menu */
            int old_mlen = mlen, okx = kx, oky = ky, del = (dn & BTN_B) != 0;
            if (dn & BTN_LEFT) ui_key_move(&kx, &ky, -1, 0);
            if (dn & BTN_RIGHT) ui_key_move(&kx, &ky, 1, 0);
            if (dn & BTN_UP) ui_key_move(&kx, &ky, 0, -1);
            if (dn & BTN_DOWN) ui_key_move(&kx, &ky, 0, 1);
            if (okx != kx || oky != ky) { ui_keycap(oky, okx, 0, upper); ui_keycap(ky, kx, 1, upper); }
            if (dn & BTN_A) {
                int ch = ui_key(ky, kx);
                if (ch == UI_KEY_CAPS) { upper = !upper; ui_keyboard(kx, ky, upper); }
                else if (ch == UI_KEY_ACCENT) accent_last();
                else if (ch == '<') del = 1;
                else {
                    if (ch == '_') ch = ' ';
                    else if (upper && ky < 2) ch -= 32;
                    if (mlen < MSG_MAX && !(ch == ' ' && (mlen == 0 || msg[mlen - 1] == ' '))) { msg[mlen++] = (char)ch; msg[mlen] = 0; }
                }
            }
            if (del && mlen) { mlen = last_char(); msg[mlen] = 0; }
            if ((dn & BTN_L) && nsug > 0) accept(0);
            if ((dn & BTN_R) && nsug > 1) accept(1);
            if (mlen != old_mlen || (dn & (BTN_A | BTN_L | BTN_R))) {
                refold(); retarget(); ui_input(msg, mlen, mlen); dbg.caret = (uint32_t)mlen;
                if (ui_scrolled()) { ui_scroll_home(); chips(); }
            }
        }
        if (act == 1) return 1;
        if (act == 2) {
            while (stepping) work();
            int m = menu();
            if (m == 1) return 0;
            if (m >= 2) return m;
            while (keys_prev) { vsync(); poll(); }           /* the presses made in the menu are not typing */
            hq_n = 0; kq_n = 0; prev = keys_prev; st_used = 0;
            if (!kb_mode) return 4;                          /* Settings > Typing changed: the chords from now on */
            ui_header(0);
            ui_typing_kb(msg, mlen, kx, ky, upper);
            shown_sug = -1; shown_empty = -1;
        }
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
    const char *title;
    int n = sdf_article(&title);
    if (n) { ui_message("Reading the article..."); read_pages(title, sdf_article_text, n, ST_ARTICLE); }
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
    /* kokoSD 2.4: the helpers look first, then Koko's marker (a hint goes into your turn, before the question) */
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
        const char *title;
        if (article && sdf_article(&title)) {
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
        if (helpers == 2 && mark == 1) {
            /* kokoSD 2.4: As a hint: your turn is said again as "<fact> <your message>" and Koko answers in his
             * own words (the native engine used the fact best this way round: host/, kokoSD 2.4 notes) */
            uint16_t *hint = fed; char *q = mem_buf;       /* (RAM is short: buffers not in use now) */
            int hn = an;
            for (int i = 0; i < hn; i++) hint[i] = want[i];
            ui_header("reading the fact...");
            koko_rewind(); dbg.ntok = (uint32_t)mark_ntok; dbg.rewinds++;
            feed(TOK_USER);
            for (int i = 0; i < hn && !b_pressed(); i++) feed(hint[i]);
            q[0] = ' '; for (int i = 0; i < plen; i++) q[i + 1] = plain[i];
            int qn = koko_encode(q, plen + 1, want, MSG_TOK);
            for (int i = 0; i < qn; i++) feed(want[i]);
            nfed = 0; al = an = 0; mark = 3;
        }
    }
    feed(TOK_KOKO);
    ui_header("B: stop");
    int stopped = b_pressed() || b_after_send;   /* kokoSD 2.2: also a B made between START and here */
    b_after_send = 0; sent_alone = 0;                           /* B while Koko was still reading: no reply */
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
    if (mem_on && !article && mem_learn(plain, plen)) mem_save();   /* kokoSD 2.4: what you said about yourself */
    dbg.turns++;
    dbg.state = ST_DONE;
    mlen = 0; msg[0] = 0; plen = 0; plain[0] = 0;
}

int main(void) {
    /* kokoSD 1.1.2: the cartridge timing stays as the loader left it until the cart is known. The fastest timing
     * (0x4317) is too fast for a SuperCard's SDRAM: there the first ROM read after it crashes the GBA (a blank
     * screen), and SuperFW cannot patch the write out (it looks for the address as a constant). */
    /* kokoSD 1.4.5: slow RAM at its normal speed. SuperFW can leave it overclocked ("Fast EWRAM"), which not
     * every GBA takes: Koko keeps its working memory and the card code there. */
    *(volatile uint32_t *)0x04000800 = 0x0D000020;
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
    kb_mode = 1; mem_load(); settings_load(); words_set();                      /* kokoSD 2.4: memory.txt; the fact file's word memory */
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
    memory_prefix();
    while (keys_prev) { vsync(); poll(); }        /* kokoSD: the START that left the title types nothing */
    hq_n = 0;
    for (;;) {
        int e = kb_mode ? edit_kb() : edit();
        if (e == 4) {                             /* kokoSD 2.5: the other way of typing: the message is read again */
            while (stepping || pre_fed < npre || pre_mark) work();
            koko_rewind(); dbg.ntok = (uint32_t)mark_ntok;
            nfed = 0; user_fed = 0; dirty = 0;
            continue;
        }
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
