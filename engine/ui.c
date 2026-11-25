/* Koko's screen (see ui.h), kokoSD 1.1: drawn like our other apps (gbamp3, gbawriter, gbavocab) with their two
 * fonts only: the SuperFW 16 px font for everything to read and the 5x7 font for the header and hints
 * (engine/text, from gbamp3). White background, black text, grey hints, the light blue selection bar.
 * Mode 4: 240x160 palette indices, page 0 only. Text is UTF-8 (accented letters are shown). GPL-3.0-or-later. */
#include "ui.h"
#include "text/text.h"

#ifdef __arm__
#define FB ((volatile uint16_t *)0x06000000)       /* video memory takes 16-bit writes only */
#define EW __attribute__((section(".bss.ewram")))
#else
uint8_t ui_fb[240 * 160];
#define FB ((volatile uint16_t *)ui_fb)
#define EW
#endif
uint16_t ui_pal[UI_PAL_N] EW;
void (*ui_yield)(void);                           /* optional: called between pieces of a long redraw (button polling) */
#define YIELD() do { if (ui_yield) ui_yield(); } while (0)

/* The apps' colours (gbamp3 ui.c) and their palette indices. */
#define C_WHITE 0x7fff
#define C_BLACK 0x0000
#define C_BLUE  0x7f55                              /* RGB(170,210,255): the selection bar */
#define C_GREY  0x4210
#define C_LIGHT 0x6318                              /* the input field */
enum { P_WHITE, P_BLACK, P_BLUE, P_GREY, P_LIGHT, P_N };
static const uint16_t PAL_RGB[P_N] = { C_WHITE, C_BLACK, C_BLUE, C_GREY, C_LIGHT };

/* The hooks engine/text/text.c draws through (as gbamp3's gfx.c, but straight onto the shown page). */
unsigned gfx_m4 = 1;
volatile uint16_t *gfx_line(int y) { return FB + y * 120; }
uint16_t gfx_pixel(uint16_t rgb) { for (int i = 0; i < P_N; i++) if (PAL_RGB[i] == rgb) return (uint16_t)i; return P_BLACK; }
void gfx_mark(int y0, int y1) { (void)y0; (void)y1; }

static void fill(int x, int y, int w, int h, uint16_t rgb) {
    if (w <= 0 || h <= 0) return;
    unsigned c = gfx_pixel(rgb);
    for (int j = y; j < y + h; j++) {
        volatile uint16_t *l = gfx_line(j);
        int a = x, n = w;
        if (a & 1) { l[a >> 1] = (uint16_t)((l[a >> 1] & 0x00ff) | c << 8); a++; n--; }
        if (n >= 2 && (a & 2)) { l[a >> 1] = (uint16_t)(c | c << 8); a += 2; n -= 2; }
        for (volatile uint32_t *w = (volatile uint32_t *)(l + (a >> 1)); n >= 4; n -= 4, a += 4) *w++ = c * 0x01010101u;   /* 4 pixels a store */
        for (; n >= 2; n -= 2, a += 2) l[a >> 1] = (uint16_t)(c | c << 8);
        if (n) l[a >> 1] = (uint16_t)((l[a >> 1] & 0xff00) | c);
    }
}
#ifdef __arm__
static void rows_up(int y0, int y1, int by) {                 /* move rows y0..y1-1 up by `by` pixels (whole rows are 60 words) */
    volatile uint32_t *d = (volatile uint32_t *)0x06000000 + (y0 - by) * 60, *s = (volatile uint32_t *)0x06000000 + y0 * 60;
    for (int n = (y1 - y0) * 60; n > 0; n--) *d++ = *s++;
}
static void frame_start(void) {                               /* wait for the screen to finish drawing, so a big change is not seen half done */
    volatile uint16_t *vc = (volatile uint16_t *)0x04000006;
    /* kokoSD 2.2: the buttons are read while waiting (a press and release within the waits were missed) */
    while (*vc >= 160) YIELD();
    while (*vc < 155) YIELD();
    while (*vc < 160) {}
}
#else
static void rows_up(int y0, int y1, int by) { for (int i = y0 * 240; i < y1 * 240; i++) ui_fb[i - by * 240] = ui_fb[i]; }
static void frame_start(void) {}
#endif

/* ---------------- text helpers: n bytes of UTF-8 ---------------- */
static char tbuf[256] EW;
static const char *cut(const char *s, int n) { if (n > 255) n = 255; for (int i = 0; i < n; i++) tbuf[i] = s[i]; tbuf[n] = 0; return tbuf; }
static int width_n(const char *s, int n) { return (int)text_width(cut(s, n)); }
/* Text on white, drawn opaque over its own width (gbamp3's two-pixel writes: much faster than pixel by pixel). */
static int text_n(int x, int y, const char *s, int n, uint16_t c) {
    const char *t = cut(s, n);
    int w = (int)text_width(t), x1 = x + w > 240 ? 240 : x + w;
    if (w) text_draw_bg(x, y, x, x1, t, 0, c, C_WHITE);
    return x + w;
}
static int cont(char c) { return ((unsigned char)c & 0xc0) == 0x80; }       /* a UTF-8 continuation byte */
static void centre(int y, const char *s, uint16_t c) { int w = (int)text_width(s), x = (240 - w) / 2; text_draw_bg(x, y, x, x + w, s, 0, c, C_WHITE); }
static void small_centre(int y, const char *s, uint16_t c) { small_draw((240 - (int)small_width(s)) / 2, y, s, c); }
/* The small font's text cut to fit `w` pixels ("..." at the end when cut). */
static const char *small_fit(const char *s, int w) {
    if ((int)small_width(s) <= w) return s;
    int n = 0;
    while (s[n] && n < 250) n++;
    for (; n > 0; n--) {
        if (cont(s[n])) continue;
        const char *t = cut(s, n);
        tbuf[n] = '.'; tbuf[n + 1] = '.'; tbuf[n + 2] = '.'; tbuf[n + 3] = 0;
        if ((int)small_width(t) <= w) return t;
    }
    return "";
}

static void put_num(char *b, int *k, int v);
/* ---------------- header (as gbamp3's: the name in the small font, centred) ---------------- */
#define HDR_H 16
static int helpers = 1;
void ui_helpers(int on) { helpers = on; }
static void header_line(const char *title, const char *right) {
    fill(0, 0, 240, HDR_H, C_WHITE);
    small_centre(5, title, C_BLACK);
    if (!helpers) small_draw(6, 5, "Facts off", C_GREY);
    if (right && *right) small_draw(234 - (int)small_width(right), 5, right, C_GREY);
}
void ui_header(const char *right) { frame_start(); header_line("Koko", right); }   /* kokoSD 2.2: in the vertical blank (no half-drawn status) */
void ui_rate(int made, uint32_t t) {
    char b[24]; int n = 0;
    if (made && t) {
        uint32_t r = (uint32_t)made * 163840u / t;          /* tokens/s x 10 */
        if (r >= 100) b[n++] = (char)('0' + r / 100 % 10);
        b[n++] = (char)('0' + r / 10 % 10); b[n++] = '.'; b[n++] = (char)('0' + r % 10);
        for (const char *u = " tok/s"; *u; u++) b[n++] = *u;
    }
    b[n] = 0;
    ui_header(b);
}
/* While typing: gbawriter's letter group and Shift / Caps on the right, as in gbamp3's Search. */
void ui_writer_status(const char *group, const char *mode) {
    fill(150, 0, 90, HDR_H, C_WHITE);
    int x = 234 - (int)small_width("abc");
    if (group && group[0]) small_draw(234 - (int)small_width(group), 5, group, C_GREY);
    x -= 6;
    if (mode && mode[0]) small_draw(x - (int)small_width(mode), 5, mode, C_GREY);
}

/* ---------------- title screen ---------------- */
void ui_title_tick(void) {}
void ui_title_note2(const char *t) { small_centre(84, t, C_GREY); }   /* kokoSD 1.4.2: no facts from the card: why */
void ui_title_prompt(int on) {
    fill(0, 108, 240, 16, C_WHITE);
    if (on) centre(108, "Press START", C_BLACK);
}
void ui_title(void) {
    fill(0, 0, 240, 160, C_WHITE);
    small_centre(5, "kokoSD", C_BLACK);
    centre(50, "Koko", C_BLACK);
    small_centre(72, "A chatbot on your Game Boy Advance", C_GREY);
    small_centre(128, "Koko: rdsilver   kokoSD: halimj", C_GREY);   /* kokoSD 1.3: credits */
    ui_title_prompt(1);
    small_centre(146, "START+SELECT while typing: menu", C_GREY);
}

/* kokoSD 1.1: what the start is doing (finding the SD card and the fact file take a moment). */
void ui_boot(const char *what, const char *detail) {
    fill(0, 0, 240, 160, C_WHITE);
    small_centre(5, "kokoSD 2.3", C_BLACK);
    centre(64, what, C_BLACK);
    if (detail && *detail) small_centre(88, small_fit(detail, 228), C_GREY);
}
void ui_init(void) {
    for (int i = 0; i < UI_PAL_N; i++) ui_pal[i] = i < P_N ? PAL_RGB[i] : C_WHITE;
    fill(0, 0, 240, 160, C_WHITE);
}

/* ---------------- chat log: wrapped lines; Koko on the left in black, the user on the right in grey ---------------- */
#define NLINE 64                                    /* kokoSD 1.2: kept for scrolling back and saved with a chat */
#define LINE_MAX 64
#define KOKO_W 228
#define USER_W 180
static struct Line { char t[LINE_MAX]; uint8_t n, kind, mark, y, dn; } lines[NLINE] EW;   /* y: where it is drawn (0 = not shown); dn: bytes drawn there */   /* mark: 1 looked up, 2 computed */
#define MARK_W 30                                   /* "Fact" / "Sum" in the small font before such a reply */
static int drawn_bot;                              /* first row below the drawn lines */
#define CHAT_TOP 17
#define LH 16
#define GAP 4                                       /* between a run of the user's lines and Koko's */
#define FIELD_Y 143                                 /* the input field (17 rows, as gbamp3's Search field) */
#define HINT_Y 132                                  /* typing: suggestions or a hint in the small font */
#define CHAT_BOT_TYPING (HINT_Y - 2)
#define CHAT_BOT_REPLY (FIELD_Y - 1)
static int nline, chat_bot = CHAT_BOT_REPLY, cur_y;   /* cur_y: where the last line was drawn (-1: not visible) */
static int scroll_back;                            /* kokoSD 1.2: lines hidden below the view (START+Up / Down) */

static void line_push(int kind) {
    if (nline == NLINE) { for (int i = 1; i < NLINE; i++) lines[i - 1] = lines[i]; nline--; }
    lines[nline].n = 0; lines[nline].kind = (uint8_t)kind; lines[nline].mark = 0; lines[nline].y = 0; lines[nline].dn = 0; nline++;
}
static int layout(uint8_t *ys) {                   /* as many of the newest lines as fit; returns the first shown, fills ys */
    int last = nline - scroll_back, first = last, h = 0;
    while (first > 0) {
        int add = LH + ((first < last && lines[first - 1].kind != lines[first].kind) ? GAP : 0);
        if (h + add > chat_bot - CHAT_TOP - 1) break;
        h += add; first--;
    }
    int y = CHAT_TOP + 1;
    for (int i = first; i < last; i++) {
        if (i > first && lines[i].kind != lines[i - 1].kind) y += GAP;
        ys[i] = (uint8_t)y; y += LH;
    }
    return first;
}
static int line_x(int i) { return 6 + (lines[i].mark ? MARK_W : 0); }
static void draw_line(int i, int y) {
    if (lines[i].kind == L_USER) text_n(234 - width_n(lines[i].t, lines[i].n), y, lines[i].t, lines[i].n, C_GREY);
    else {
        if (lines[i].mark) small_draw(6, y + 5, lines[i].mark == 1 ? "Fact" : "Sum", C_GREY);
        text_n(line_x(i), y, lines[i].t, lines[i].n, lines[i].kind == L_NOTE ? C_GREY : C_BLACK);
    }
    lines[i].y = (uint8_t)y; lines[i].dn = lines[i].n;
}
/* kokoSD 2.2: line by line, each one cleared just before it is drawn (and the gaps between them), not the whole
 * chat blanked first: on a real screen the blank chat was seen for a frame or two when a message was sent. */
static void chat_draw(void) {
    uint8_t ys[NLINE];
    int first = layout(ys);
    cur_y = -1; drawn_bot = CHAT_TOP + 1;
    for (int i = 0; i < first; i++) lines[i].y = 0;
    int last = nline - scroll_back;
    for (int i = last; i < nline; i++) lines[i].y = 0;
    frame_start();
    int y = CHAT_TOP - 1;
    for (int i = first; i < last; i++) {
        if (ys[i] > y) fill(0, y, 240, ys[i] - y, C_WHITE);
        {   /* the text is drawn opaque over its own width: only what is left and right of it is cleared */
            int w = width_n(lines[i].t, lines[i].n), x0 = lines[i].kind == L_USER ? 234 - w : line_x(i), x1 = x0 + w;
            if (x1 < 240) fill(x1, ys[i], 240 - x1, LH, C_WHITE);
            if (x0 > 0) fill(0, ys[i], x0, LH, C_WHITE);
        }
        draw_line(i, ys[i]);
        if (i == nline - 1) cur_y = ys[i];
        drawn_bot = y = ys[i] + LH;
        YIELD();
    }
    if (chat_bot > y) fill(0, y, 240, chat_bot - y, C_WHITE);
}
/* Lines from index `from` on are new or changed; the rest are on screen already. Instead of blanking the chat and
 * drawing it again (a visible flash on a real screen), slide the existing rows up and draw only what changed. */
static void chat_update(int from) {
    uint8_t ys[NLINE];
    if (scroll_back) { scroll_back = 0; chat_draw(); return; }   /* new text: back to the newest lines */
    int first = layout(ys), shift = -1;
    for (int i = from; i < nline; i++) if (lines[i].kind == L_USER) { chat_draw(); return; }
    for (int i = first; i < from; i++) {
        int s = lines[i].y ? lines[i].y - ys[i] : -2;
        if (s < 0 || (shift >= 0 && s != shift)) { chat_draw(); return; }
        shift = s;
    }
    if (shift < 0 || from <= first) { chat_draw(); return; }
    frame_start();
    if (shift > 0) {
        rows_up(ys[first] + shift, drawn_bot, shift);
        fill(0, CHAT_TOP - 1, 240, ys[first] - CHAT_TOP + 1, C_WHITE);   /* the strip above the first line: nothing of the line that left */
        fill(0, drawn_bot - shift, 240, shift, C_WHITE);
    }
    for (int i = 0; i < first; i++) lines[i].y = 0;
    for (int i = first; i < from; i++) lines[i].y = ys[i];
    for (int i = from; i < nline; i++) {
        if (lines[i].y && lines[i].y - shift == ys[i] && lines[i].dn >= lines[i].n) {
            /* already on screen and only shorter (its last word moved down): rub out the end, leave the rest alone */
            int x = line_x(i) + width_n(lines[i].t, lines[i].n);
            fill(x, ys[i], 240 - x, LH, C_WHITE);
            lines[i].y = ys[i]; lines[i].dn = lines[i].n;
            continue;
        }
        fill(0, ys[i], 240, LH, C_WHITE);
        draw_line(i, ys[i]);
    }
    cur_y = ys[nline - 1]; drawn_bot = ys[nline - 1] + LH;
}
/* Add text to the newest line, wrapping by words (a line breaks only between whole UTF-8 characters).
 * Returns how many new lines that started. */
static int line_add(const char *s, int n, int maxw) {
    int wrapped = 0;
    for (int i = 0; i < n; i++) {
        struct Line *l = &lines[nline - 1];
        if (s[i] == ' ' && l->n == 0) continue;
        if (l->n < LINE_MAX) l->t[l->n++] = s[i];
        if (i + 1 < n && cont(s[i + 1]) && l->n < LINE_MAX) continue;          /* the character is not complete yet */
        if (width_n(l->t, l->n) > maxw - (l->mark ? MARK_W : 0) || l->n == LINE_MAX) {             /* carry the unfinished word down */
            int k = l->n - 1;
            while (k > 0 && l->t[k] != ' ') k--;
            int last = l->n - 1;
            while (last > 0 && cont(l->t[last])) last--;                       /* no space: carry the last character */
            int kind = l->kind, from = k > 0 ? k + 1 : last, cnt = l->n - from;
            char tail[LINE_MAX];
            for (int j = 0; j < cnt; j++) tail[j] = l->t[from + j];
            l->n = (uint8_t)(k > 0 ? k : from);
            line_push(kind);
            l = &lines[nline - 1];
            for (int j = 0; j < cnt; j++) l->t[l->n++] = tail[j];
            wrapped++;
        }
    }
    return wrapped;
}
void ui_clear(void) { nline = 0; scroll_back = 0; }
/* kokoSD 1.2: START+Up (d > 0) / START+Down (d < 0) move the view one line; 0 if it could not move. */
int ui_scroll(int d) {
    uint8_t ys[NLINE];
    int was = scroll_back;
    if (d > 0 && layout(ys) > 0) scroll_back++;
    if (d < 0 && scroll_back > 0) scroll_back--;
    if (scroll_back == was) return 0;
    chat_draw();
    return 1;
}
int ui_scrolled(void) { return scroll_back; }
void ui_scroll_home(void) { if (scroll_back) { scroll_back = 0; chat_draw(); } }
/* kokoSD 1.2: the chat's lines, saved with a chat and read back. */
int ui_log_io(int (*io)(void *p, unsigned n, void *ctx), void *ctx, int writing) {
    uint32_t n = (uint32_t)nline;
    if (!io(&n, 4, ctx) || n > NLINE) return 0;
    if (!io(lines, n * sizeof lines[0], ctx)) return 0;
    if (!writing) {                                  /* kokoSD 2.1: a damaged file is refused */
        for (uint32_t i = 0; i < n; i++) if (lines[i].n > LINE_MAX || lines[i].kind > L_NOTE || lines[i].mark > 2) { nline = 0; return 0; }
        nline = (int)n; scroll_back = 0; for (int i = 0; i < nline; i++) { lines[i].y = 0; lines[i].dn = 0; }
    }
    return 1;
}
void ui_say(int kind, const char *s, int n) { line_push(kind); line_add(s, n, kind == L_USER ? USER_W : KOKO_W); }

/* ---------------- the input field (gbamp3's Search field) and the hint line ---------------- */
static int reply_stop = -1;
static void field_frame(void) {
    fill(0, FIELD_Y, 240, 1, C_LIGHT); fill(0, FIELD_Y + 1, 6, 16, C_LIGHT); fill(234, FIELD_Y + 1, 6, 16, C_LIGHT);
}
static void field_placeholder(const char *t, int caret) {
    field_frame();
    fill(6, FIELD_Y + 1, 2, 16, C_LIGHT);
    text_draw_bg(8, FIELD_Y + 1, 8, 234, t, 0, C_GREY, C_LIGHT);
    if (caret) fill(6, FIELD_Y + 1, 1, 15, C_BLACK);
}
void ui_input(const char *msg, int mlen, int caret) {
    reply_stop = -1;
    if (!mlen) { field_placeholder("Type a message", 1); return; }
    if (caret < 0 || caret > mlen) caret = mlen;
    field_frame();
    static char t[UI_MSG_MAX + 1] EW;
    int n = mlen < UI_MSG_MAX ? mlen : UI_MSG_MAX;
    if (n < 0) n = 0;
    for (int i = 0; i < n; i++) t[i] = msg[i];
    t[n] = 0;
    int cx = width_n(msg, caret), skip = cx > 226 ? cx - 226 : 0;   /* the cursor stays in view */
    text_draw_bg(6, FIELD_Y + 1, 6, 234, t, (unsigned)skip, C_BLACK, C_LIGHT);
    fill(6 + cx - skip, FIELD_Y + 1, 1, 15, C_BLACK);
}
static const char *idle_hint = "START+SELECT: menu    START+A: save chat";
static int chips_n;
static void hint_line(const char *t) {
    /* kokoSD 2.2: cleared and written again in the vertical blank: done while the screen was drawing these rows,
     * the empty line was seen for a frame (the suggestions flickered at every letter) */
    frame_start();
    fill(0, HINT_Y - 1, 240, 10, C_WHITE);
    small_draw(6, HINT_Y, small_fit(t, 228), C_GREY);
}
void ui_hint(int has_text) {
    const char *h = has_text ? "START: send    START+A: save chat" : "START+SELECT: menu    START+A: save chat";
    if (h == idle_hint) return;
    idle_hint = h;
    if (!chips_n) hint_line(h);
}
void ui_chips(int n, const char *a, int alen, const char *b, int blen) {
    chips_n = n;
    if (!n) { hint_line(idle_hint); return; }
    static char s[96] EW; int k = 0;
    for (int j = 0; j < n; j++) {
        const char *p = j ? b : a; int len = j ? blen : alen;
        if (len && p[0] == ' ') { p++; len--; }
        const char *pre = j ? "     START+R: " : "START+L: ";
        while (*pre && k < 80) s[k++] = *pre++;
        for (int i = 0; i < len && k < 94; i++) s[k++] = p[i];
    }
    s[k] = 0;
    hint_line(s);
}
void ui_message(const char *t) { hint_line(t); }
void ui_lines(const char *a, const char *b) { small_centre(96, small_fit(a, 228), C_GREY); small_draw(6, 150, small_fit(b, 228), C_GREY); }   /* kokoSD 1.4.1 */
/* kokoSD 1.3: a question on its own screen (Delete this chat? A: yes B: no), or a page to type on (Rename). */
void ui_ask(const char *title, const char *question, const char *detail, const char *hint) {
    frame_start();
    header_line(title, 0);
    fill(0, HDR_H, 240, 160 - HDR_H, C_WHITE);
    YIELD();
    if (question && *question) centre(52, question, C_BLACK);
    if (detail && *detail) small_centre(76, small_fit(detail, 228), C_GREY);
    if (hint && *hint) small_draw(6, hint[0] == '!' ? 128 : 150, hint[0] == '!' ? hint + 1 : hint, C_GREY);
}
/* kokoSD 1.2: Saved chats, as gbamp3's lists: 7 rows of 17 pixels from `top`, the selection blue, long first
 * messages cut with "...". */
#define LIST_ROWS 7
void ui_list(const char *title, const char *const *items, int n, int sel, int top, const char *empty) {
    frame_start();
    header_line(title, 0);
    YIELD();
    for (int r = 0; r < LIST_ROWS; r++) {
        int i = top + r, y = HDR_H + r * 17;
        uint16_t bg = i == sel && i < n ? C_BLUE : C_WHITE;
        fill(0, y, 6, 17, bg); fill(234, y, 6, 17, bg); fill(6, y, 228, 1, bg);
        const char *t = i < n ? items[i] : "";
        unsigned whole; size_t k = text_fit(t, 228, &whole);
        if (!whole) { if (k > 250) k = 250; for (size_t j = 0; j < k; j++) tbuf[j] = t[j]; tbuf[k] = '.'; tbuf[k + 1] = '.'; tbuf[k + 2] = '.'; tbuf[k + 3] = 0; t = tbuf; }
        text_draw_bg(6, y + 1, 6, 234, t, 0, C_BLACK, bg);
        YIELD();
    }
    fill(0, HDR_H + LIST_ROWS * 17, 240, 160 - HDR_H - LIST_ROWS * 17, C_WHITE);
    if (!n) {                                      /* "big line\nsmall line" */
        int k = 0; while (empty[k] && empty[k] != '\n') k++;
        text_draw_bg((240 - width_n(empty, k)) / 2, 56, (240 - width_n(empty, k)) / 2, (240 + width_n(empty, k)) / 2, cut(empty, k), 0, C_GREY, C_WHITE);
        if (empty[k]) small_centre(80, empty + k + 1, C_GREY);
    }
    small_draw(6, 150, n ? "A: open    B: back" : "B: back", C_GREY);
    if (n > LIST_ROWS) {
        char pg[16]; int k = 0;
        put_num(pg, &k, sel + 1); pg[k++] = '/'; put_num(pg, &k, n); pg[k] = 0;
        small_draw(234 - (int)small_width(pg), 150, pg, C_GREY);
    }
}
void ui_typing(const char *msg, int mlen, int caret) {
    chat_bot = CHAT_BOT_TYPING;
    chat_draw();
    fill(0, chat_bot, 240, FIELD_Y - chat_bot, C_WHITE);
    YIELD();
    ui_input(msg, mlen, caret); ui_chips(0, 0, 0, 0, 0);
}
void ui_menu(const char *title, const char *const *items, int n, int sel) {
    frame_start();
    header_line(title, 0);
    YIELD();
    for (int i = 0; i < n; i++) {                                  /* rows as gbamp3's lists: 17 rows each, the selection blue */
        int y = HDR_H + i * 17;
        uint16_t bg = i == sel ? C_BLUE : C_WHITE;
        fill(0, y, 6, 17, bg); fill(234, y, 6, 17, bg); fill(6, y, 228, 1, bg);
        text_draw_bg(6, y + 1, 6, 234, items[i], 0, C_BLACK, bg);
        YIELD();
    }
    fill(0, HDR_H + n * 17, 240, 160 - HDR_H - n * 17, C_WHITE);
    small_draw(6, 150, "A: choose    B: back", C_GREY);
}

/* ---------------- reply layout ---------------- */
void ui_reply_button(int stop) {                      /* the field while Koko answers (B stops him) and after */
    if (stop == reply_stop) return;
    reply_stop = stop;
    field_placeholder(stop ? "Koko is writing...    B: stop" : "Press a button to reply", 0);
}
void ui_reply_layout(const char *msg, int mlen) {
    chat_bot = CHAT_BOT_REPLY;
    line_push(L_USER); line_add(msg, mlen, USER_W);
    fill(0, CHAT_BOT_TYPING, 240, FIELD_Y - CHAT_BOT_TYPING, C_WHITE);
    chat_draw();
    reply_stop = -1; ui_reply_button(1);
}
void ui_reply_start(int mark) { line_push(L_KOKO); lines[nline - 1].mark = (uint8_t)mark; chat_update(nline - 1); }
void ui_redraw(void) { chat_draw(); }
void ui_reply_text(const char *p, int n) {
    int before = lines[nline - 1].n;
    int w = line_add(p, n, KOKO_W);
    if (w) chat_update(nline - 1 - w);
    else if (cur_y < 0) chat_draw();
    else {
        text_n(line_x(nline - 1) + width_n(lines[nline - 1].t, before), cur_y, lines[nline - 1].t + before, lines[nline - 1].n - before, C_BLACK);
        lines[nline - 1].dn = lines[nline - 1].n;
    }
}

/* ---------------- kokoSD: an article to read ("Tell me more"), How to type, About ---------------- */
/* The text is wrapped once into lines; a page shows ART_ROWS of them. kokoSD 2.3: the text comes from a source that
 * gives any part of it (the SD card's article is read in parts, up to 64,000 characters), and the lines' starts
 * and lengths are kept in video memory the screen does not use (0x06014000, for sprites: there is no room left
 * in RAM). */
#define ART_LINES 4095
#define ART_ROWS 8
#define ART_W 228
#ifdef __arm__
#define art_start ((volatile uint16_t *)0x06014000)
#define art_len ((volatile uint16_t *)0x06016000)
#else
static uint16_t art_start[ART_LINES + 1], art_len[ART_LINES];
#endif
static int art_n;
static ui_src art_src;
static uint32_t art_total;
static int char_len(const char *t, int i, int n) { int j = i + 1; while (j < n && cont(t[j])) j++; return j - i; }
int ui_article_wrap(ui_src src, uint32_t n) {
    art_src = src; art_total = n; art_n = 0;
    uint32_t i = 0;
    while (i < n && art_n < ART_LINES) {
        int span = n - i < 300 ? (int)(n - i) : 300;
        const char *t = src(i, (uint32_t)span);
        if (!t) break;                               /* (a card error: the article ends here) */
        if (t[0] == '\n') {                          /* a line break; an empty line (two breaks) is a blank line */
            if (span > 1 && t[1] == '\n') { art_start[art_n] = (uint16_t)i; art_len[art_n++] = 0; i++; }
            i++; continue;
        }
        int j = 0, w = 0, cut_at = -1;
        while (j < span && t[j] != '\n') {
            int k = char_len(t, j, span);
            w += width_n(t + j, k);
            if (w > ART_W || j + k > 255) break;
            if (t[j] == ' ') cut_at = j;
            j += k;
        }
        int end = (j < span && t[j] != '\n' && cut_at > 0) ? cut_at : j;
        if (end == 0) end = char_len(t, 0, span);
        art_start[art_n] = (uint16_t)i; art_len[art_n++] = (uint16_t)end;
        i += (uint32_t)end;
        while (i < n && end < span && t[end] == ' ') { i++; end++; }
        if (!(art_n & 15)) YIELD();
    }
    return art_n;
}
int ui_article_rows(void) { return ART_ROWS; }
static void put_num(char *b, int *k, int v) { char d[6]; int n = 0; do { d[n++] = (char)('0' + v % 10); v /= 10; } while (v); while (n) b[(*k)++] = d[--n]; }
void ui_article(const char *title, int first) {
    frame_start();
    header_line(small_fit(title, 228), 0);
    for (int r = 0; r < ART_ROWS; r++) {
        YIELD();
        int y = CHAT_TOP + r * LH, i = first + r;
        const char *t = i < art_n && art_len[i] ? art_src(art_start[i], art_len[i]) : 0;
        fill(0, y, 6, LH, C_WHITE);
        text_draw_bg(6, y, 6, 240, t ? cut(t, art_len[i]) : "", 0, C_BLACK, C_WHITE);
    }
    fill(0, CHAT_TOP + ART_ROWS * LH, 240, 160 - CHAT_TOP - ART_ROWS * LH, C_WHITE);
    small_draw(6, 150, first + ART_ROWS < art_n ? "Down: more    B: back" : "B: back", C_GREY);
    char pg[16]; int k = 0;
    int pages = (art_n + ART_ROWS - 1) / ART_ROWS, at = (first + ART_ROWS - 1) / ART_ROWS + 1;
    if (pages < 1) pages = 1;
    if (at > pages) at = pages;
    put_num(pg, &k, at); pg[k++] = '/'; put_num(pg, &k, pages); pg[k] = 0;
    small_draw(234 - (int)small_width(pg), 150, pg, C_GREY);
}
void ui_chat_again(void) {                           /* back from an article: the chat as it was (reply layout) */
    fill(0, 0, 240, 160, C_WHITE);
    ui_header(0);
    chat_bot = CHAT_BOT_REPLY;
    chat_draw();
    reply_stop = -1; ui_reply_button(0);
}
