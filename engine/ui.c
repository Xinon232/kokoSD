/* Koko's screen (see ui.h). Bitmap of 240x160 palette indices; proportional anti-aliased font.
 * A "paint" is an ink on a surface: 4 palette entries, surface -> ink, so anti-aliased text needs no blending
 * at draw time. Text must be drawn over its paint's surface. */
#include "ui.h"
#include "font_ui.h"
#include "husky.h"

#define RGB(r, g, b) ((r) | ((g) << 5) | ((b) << 10))
#ifdef __arm__
#define FB ((volatile uint16_t *)0x06000000)       /* video memory takes 16-bit writes only */
#define EW __attribute__((section(".bss.ewram")))
#else
uint8_t ui_fb[240 * 160];
#define EW
#endif
uint16_t ui_pal[UI_PAL_N] EW;
void (*ui_yield)(void);                           /* optional: called between pieces of a long redraw (button polling) */
#define YIELD() do { if (ui_yield) ui_yield(); } while (0)
enum { BG, BG_DIM, BG_ACC, BUB, PILL, PILL_DIM, PANEL, PANEL_DIM, SEL, CHIP, CHIP_ACC, BTN, NPAINT };
#define C(p) ((p) * 4)                              /* the surface colour of a paint */
#define INK(p) ((p) * 4 + 3)
static const uint16_t SURF[NPAINT] = {
    RGB(7, 11, 20), RGB(7, 11, 20), RGB(7, 11, 20), RGB(13, 18, 28), RGB(11, 15, 25), RGB(11, 15, 25),
    RGB(4, 7, 14), RGB(4, 7, 14), RGB(31, 31, 31), RGB(11, 15, 25), RGB(11, 15, 25), RGB(31, 31, 31) };
static const uint16_t INKC[NPAINT] = {
    RGB(31, 31, 31), RGB(17, 21, 28), RGB(15, 27, 31), RGB(31, 31, 31), RGB(31, 31, 31), RGB(17, 21, 28),
    RGB(29, 30, 31), RGB(14, 18, 25), RGB(4, 7, 14), RGB(31, 31, 31), RGB(15, 27, 31), RGB(7, 11, 20) };

#ifdef __arm__
static void pset(int x, int y, int c) {
    volatile uint16_t *p = &FB[(y * 240 + x) >> 1];
    *p = (x & 1) ? (uint16_t)((*p & 0x00FF) | (c << 8)) : (uint16_t)((*p & 0xFF00) | c);
}
static void hline(int x, int y, int w, int c) {
    if (w <= 0) return;
    if (x & 1) { pset(x, y, c); x++; w--; }
    volatile uint16_t *p = &FB[(y * 240 + x) >> 1];
    for (; w >= 2; w -= 2, x += 2) *p++ = (uint16_t)(c | c << 8);
    if (w) pset(x, y, c);
}
#else
static void pset(int x, int y, int c) { ui_fb[y * 240 + x] = (uint8_t)c; }
static void hline(int x, int y, int w, int c) { for (int i = 0; i < w; i++) pset(x + i, y, c); }
#endif
#ifdef __arm__
static int pget(int x, int y) { uint16_t v = FB[(y * 240 + x) >> 1]; return (x & 1) ? v >> 8 : v & 0xFF; }
static void rows_up(int y0, int y1, int by) {                 /* move rows y0..y1-1 up by `by` pixels (whole rows are 60 words) */
    volatile uint32_t *d = (volatile uint32_t *)0x06000000 + (y0 - by) * 60, *s = (volatile uint32_t *)0x06000000 + y0 * 60;
    for (int n = (y1 - y0) * 60; n > 0; n--) *d++ = *s++;
}
static void frame_start(void) {                               /* wait for the screen to finish drawing, so a big change is not seen half done */
    volatile uint16_t *vc = (volatile uint16_t *)0x04000006;
    while (*vc >= 160) {}
    while (*vc < 160) {}
}
#else
static int pget(int x, int y) { return ui_fb[y * 240 + x]; }
static void rows_up(int y0, int y1, int by) { for (int i = y0 * 240; i < y1 * 240; i++) ui_fb[i - by * 240] = ui_fb[i]; }
static void frame_start(void) {}
#endif
static void rect(int x, int y, int w, int h, int c) { for (int i = 0; i < h; i++) hline(x, y + i, w, c); }
static void rrect(int x, int y, int w, int h, int r, int c) {      /* rounded corners, radius 4, 6 or 8 */
    static const uint8_t in4[4] = { 2, 1, 0, 0 }, in6[6] = { 4, 2, 1, 1, 0, 0 }, in8[8] = { 5, 3, 2, 1, 1, 0, 0, 0 };
    const uint8_t *in = r == 4 ? in4 : r == 6 ? in6 : in8;
    for (int i = 0; i < h; i++) {
        int k = i < r ? in[i] : (h - 1 - i) < r ? in[h - 1 - i] : 0;
        hline(x + k, y + i, w - 2 * k, c);
    }
}
static int glyph(int x, int y, int ch, int paint, int title) {
    if (ch < 32 || ch > 126) ch = '?';
    const uint32_t *g = title ? FONT_TITLE[ch - 32] : FONT_BODY[ch - 32];
    int h = title ? FONT_TITLE_H : FONT_BODY_H;
    for (int r = 0; r < h; r++) {
        uint32_t bits = g[r];
        for (int px = x; bits; px++, bits >>= 2)
            if ((bits & 3) && px >= 0 && px < 240 && y + r >= 0 && y + r < 160) pset(px, y + r, paint * 4 + (int)(bits & 3));
    }
    return title ? FONT_TITLE_W[ch - 32] : FONT_BODY_W[ch - 32];
}
static int text_n(int x, int y, const char *s, int n, int paint) { for (int i = 0; i < n; i++) x += glyph(x, y, s[i], paint, 0); return x; }
static int text(int x, int y, const char *s, int paint) { while (*s) x += glyph(x, y, *s++, paint, 0); return x; }
static int width_n(const char *s, int n) { int w = 0; for (int i = 0; i < n; i++) w += FONT_BODY_W[(s[i] < 32 || s[i] > 126 ? '?' : s[i]) - 32]; return w; }
static int width(const char *s) { int n = 0; while (s[n]) n++; return width_n(s, n); }

/* ---------------- title screen ---------------- */
#define HUSKY_PAL0 48                               /* the portrait's 15 colours follow the paints in the palette */
static int glyph2(int x, int y, int ch, int paint) {              /* title font at twice its size */
    const uint32_t *g = FONT_TITLE[ch - 32];
    for (int r = 0; r < FONT_TITLE_H; r++) {
        uint32_t bits = g[r];
        for (int px = x; bits; px += 2, bits >>= 2)
            if (bits & 3) rect(px, y + 2 * r, 2, 2, paint * 4 + (int)(bits & 3));
    }
    return 2 * FONT_TITLE_W[ch - 32];
}
static void centred(int y, const char *s, int paint) { text((240 - width(s)) / 2, y, s, paint); }
/* Snow drifting behind the portrait. A flake is only ever drawn on plain background and put back afterwards. */
#define NFLAKE 34
static struct { int16_t x, y, vx, vy; uint8_t lvl, on, px, py; } flake[NFLAKE] EW;
static uint32_t snow_rng;
static int rnd(int n) { snow_rng = snow_rng * 1664525u + 1013904223u; return (int)((snow_rng >> 16) % (unsigned)n); }
static void flake_new(int i, int top) {
    flake[i].x = (int16_t)(rnd(240) << 4); flake[i].y = (int16_t)(top ? 0 : rnd(160) << 4);
    flake[i].lvl = (uint8_t)(1 + rnd(3));                              /* dim and slow = far away */
    flake[i].vy = (int16_t)(3 + flake[i].lvl * 3 + rnd(4)); flake[i].vx = (int16_t)(rnd(7) - 3);
    flake[i].on = 0;
}
void ui_title_tick(void) {
    for (int i = 0; i < NFLAKE; i++) {
        if (flake[i].on) { pset(flake[i].px, flake[i].py, C(BG)); flake[i].on = 0; }
        flake[i].y = (int16_t)(flake[i].y + flake[i].vy);
        flake[i].x = (int16_t)(flake[i].x + flake[i].vx + ((flake[i].y >> 8) & 1 ? 1 : -1));
        if (flake[i].x < 0) flake[i].x += 240 << 4;
        if (flake[i].x >= 240 << 4) flake[i].x -= 240 << 4;
        if (flake[i].y >= 160 << 4) flake_new(i, 1);
        int x = flake[i].x >> 4, y = flake[i].y >> 4;
        if (y >= 142 || pget(x, y) != C(BG)) continue;                 /* not over the portrait, the words or the blinking line */
        pset(x, y, BG * 4 + flake[i].lvl);
        flake[i].on = 1; flake[i].px = (uint8_t)x; flake[i].py = (uint8_t)y;
    }
}
void ui_title_prompt(int on) {
    rect(0, 143, 240, 15, C(BG));
    if (on) centred(144, "Press START", BG);
}
void ui_title(void) {
    rect(0, 0, 240, 160, C(BG));
    int x0 = (240 - HUSKY_W) / 2, y0 = 6;
    for (int y = 0; y < HUSKY_H; y++)
        for (int x = 0; x < HUSKY_W; x++) {
            int v = (HUSKY[y][x >> 1] >> ((x & 1) * 4)) & 15;
            if (v) pset(x0 + x, y0 + y, HUSKY_PAL0 + v - 1);
        }
    int w = 0;
    for (const char *t = "Koko"; *t; t++) w += 2 * FONT_TITLE_W[*t - 32];
    int x = (240 - w) / 2;
    for (const char *t = "Koko"; *t; t++) x += glyph2(x, 96, *t, BG);
    centred(126, "the chatbot in a Game Boy Advance cartridge", BG_DIM);
    ui_title_prompt(1);
    snow_rng = 2024;
    for (int i = 0; i < NFLAKE; i++) flake_new(i, 0);
}

void ui_init(void) {
    for (int p = 0; p < NPAINT; p++)
        for (int k = 0; k < 4; k++) {
            int c = 0;
            for (int ch = 0; ch < 3; ch++) {
                int a = (SURF[p] >> (5 * ch)) & 31, b = (INKC[p] >> (5 * ch)) & 31;
                c |= ((a * (3 - k) + b * k + 1) / 3) << (5 * ch);
            }
            ui_pal[p * 4 + k] = (uint16_t)c;
        }
    for (int i = 0; i < 15; i++) ui_pal[HUSKY_PAL0 + i] = HUSKY_PAL[i];
    rect(0, 0, 240, 160, C(BG));
}

/* ---------------- header ---------------- */
static int helpers = 1;
void ui_helpers(int on) { helpers = on; }
void ui_header(const char *right) {
    rect(0, 0, 240, 17, C(BG));
    int x = 10;
    for (const char *t = "Koko"; *t; t++) x += glyph(x, 2, *t, BG, 1);
    rect(x + 4, 8, 3, 3, INK(BG_ACC));                                   /* accent dot: solid = helpers on, hollow = model only */
    if (!helpers) pset(x + 5, 9, C(BG));
    if (right) text(232 - width(right), 3, right, BG_DIM);
}
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

/* ---------------- chat log: wrapped lines, user lines in a bubble on the right ---------------- */
#define NLINE 16
#define LINE_MAX 56
#define KOKO_W 220
#define USER_W 150
static struct Line { char t[LINE_MAX]; uint8_t n, kind, mark, y, dn; } lines[NLINE] EW;   /* y: where it is drawn (0 = not shown); dn: characters drawn there */   /* mark: 1 looked up, 2 computed */
#define MARK_W 12
static int drawn_bot;                              /* first row below the drawn lines */
static int nline, chat_bot = 138, cur_y;          /* cur_y: where the last line was drawn (-1: not visible) */
#define CHAT_TOP 19
#define LH 13

static void line_push(int kind) {
    if (nline == NLINE) { for (int i = 1; i < NLINE; i++) lines[i - 1] = lines[i]; nline--; }
    lines[nline].n = 0; lines[nline].kind = (uint8_t)kind; lines[nline].mark = 0; lines[nline].y = 0; lines[nline].dn = 0; nline++;
}
/* A small sign in front of a reply that did not come from the model: a book (looked up) or an equals box (computed). */
static void mark_icon(int x, int y, int mark) {
    int c = INK(BG_ACC);
    if (mark == 1) {
        rect(x, y + 4, 9, 1, c); rect(x, y + 10, 9, 1, c); rect(x, y + 4, 1, 7, c); rect(x + 8, y + 4, 1, 7, c); rect(x + 4, y + 4, 1, 7, c);
    } else {
        rect(x + 1, y + 3, 7, 1, c); rect(x + 1, y + 11, 7, 1, c); rect(x + 1, y + 3, 1, 9, c); rect(x + 7, y + 3, 1, 9, c);
        rect(x + 3, y + 6, 3, 1, c); rect(x + 3, y + 8, 3, 1, c);
    }
}
static int layout(uint8_t *ys) {                   /* as many of the newest lines as fit; returns the first shown, fills ys */
    int first = nline, h = 0;
    while (first > 0) {
        int add = LH + ((first < nline && lines[first - 1].kind != lines[first].kind) ? 6 : 0);
        if (h + add > chat_bot - CHAT_TOP - 4) break;
        h += add; first--;
    }
    int y = CHAT_TOP + 3;
    for (int i = first; i < nline; i++) {
        if (i > first && lines[i].kind != lines[i - 1].kind) y += 6;
        ys[i] = (uint8_t)y; y += LH;
    }
    return first;
}
static void draw_line(int i, int y, int first) {
    if (lines[i].kind == L_USER) {
        if (i == first || lines[i - 1].kind != L_USER) {             /* one bubble behind a run of user lines */
            int j = i, w = 0;
            for (; j < nline && lines[j].kind == L_USER; j++) { int lw = width_n(lines[j].t, lines[j].n); if (lw > w) w = lw; }
            rrect(232 - w - 14, y - 2, w + 14, (j - i) * LH + 4, 6, C(BUB));
        }
        text_n(232 - 7 - width_n(lines[i].t, lines[i].n), y, lines[i].t, lines[i].n, BUB);
    } else {
        if (lines[i].mark) mark_icon(10, y, lines[i].mark);
        text_n(10 + (lines[i].mark ? MARK_W : 0), y, lines[i].t, lines[i].n, lines[i].kind == L_NOTE ? BG_DIM : BG);
    }
    lines[i].y = (uint8_t)y; lines[i].dn = lines[i].n;
}
static void chat_draw(void) {
    uint8_t ys[NLINE];
    int first = layout(ys);
    rect(0, CHAT_TOP, 240, chat_bot - CHAT_TOP, C(BG));
    cur_y = -1; drawn_bot = CHAT_TOP + 3;
    for (int i = 0; i < first; i++) lines[i].y = 0;
    for (int i = first; i < nline; i++) {
        YIELD();
        draw_line(i, ys[i], first);
        if (i == nline - 1) cur_y = ys[i];
        drawn_bot = ys[i] + LH + 2;
    }
}
/* Lines from index `from` on are new or changed; the rest are on screen already. Instead of blanking the chat and
 * drawing it again (a visible flash on a real screen), slide the existing rows up and draw only what changed. */
static void chat_update(int from) {
    uint8_t ys[NLINE];
    int first = layout(ys), shift = -1;
    for (int i = from; i < nline; i++) if (lines[i].kind == L_USER) { chat_draw(); return; }
    for (int i = first; i < from; i++) {
        int s = lines[i].y ? lines[i].y - ys[i] : -2;
        if (s < 0 || (shift >= 0 && s != shift)) { chat_draw(); return; }
        shift = s;
    }
    if (shift < 0 || from <= first) { chat_draw(); return; }
    if (shift > 0 && lines[first].kind == L_USER) { chat_draw(); return; }   /* a bubble cut at the top needs its corners redrawn */
    frame_start();
    if (shift > 0) {
        rows_up(ys[first] + shift, drawn_bot, shift);
        rect(0, CHAT_TOP, 240, ys[first] - CHAT_TOP, C(BG));           /* the strip above the first line: nothing of the line that left */
        rect(0, drawn_bot - shift, 240, shift, C(BG));
    }
    for (int i = 0; i < first; i++) lines[i].y = 0;
    for (int i = first; i < from; i++) lines[i].y = ys[i];
    for (int i = from; i < nline; i++) {
        if (lines[i].y && lines[i].y - shift == ys[i] && lines[i].dn >= lines[i].n) {
            /* already on screen and only shorter (its last word moved down): rub out the end, leave the rest alone */
            int x = 10 + (lines[i].mark ? MARK_W : 0) + width_n(lines[i].t, lines[i].n);
            rect(x, ys[i], 240 - x, LH, C(BG));
            lines[i].y = ys[i]; lines[i].dn = lines[i].n;
            continue;
        }
        rect(0, ys[i], 240, LH, C(BG));
        draw_line(i, ys[i], first);
    }
    cur_y = ys[nline - 1]; drawn_bot = ys[nline - 1] + LH + 2;
}
/* Add text to the newest line, wrapping by words. Returns how many new lines that started. */
static int line_add(const char *s, int n, int maxw) {
    int wrapped = 0;
    for (int i = 0; i < n; i++) {
        struct Line *l = &lines[nline - 1];
        if (s[i] == ' ' && l->n == 0) continue;
        if (l->n < LINE_MAX) l->t[l->n++] = s[i];
        if (width_n(l->t, l->n) > maxw - (l->mark ? MARK_W : 0) || l->n == LINE_MAX) {             /* carry the unfinished word down */
            int k = l->n - 1;
            while (k > 0 && l->t[k] != ' ') k--;
            int kind = l->kind, from = k > 0 ? k + 1 : l->n - 1, cnt = l->n - from;
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
void ui_clear(void) { nline = 0; }
void ui_say(int kind, const char *s, int n) { line_push(kind); line_add(s, n, kind == L_USER ? USER_W : KOKO_W); }


/* ---------------- typing layout ---------------- */
static const char *const keyrow[4] = { "abcdefghijklm", "nopqrstuvwxyz", "0123456789+-\002", ".,!?'___/()\001<" };   /* ___: the space bar (three cells), \001: new chat, \002: helpers on/off */
int ui_key(int r, int c) { return keyrow[r][c]; }
/* The space bar covers columns 5 to 7 of the bottom row and counts as one key, standing at column 6. */
#define BAR(r, c) ((r) == 3 && (c) >= 5 && (c) <= 7)
int ui_key_move(int *kx, int *ky, int dx, int dy) {            /* move the cursor one step; returns 1 if it moved */
    int x = *kx, y = *ky;
    if (dy) y = (y + dy + 4) % 4;
    if (dx) { x = (x + dx + 13) % 13; if (BAR(y, x) && BAR(*ky, *kx)) x = dx > 0 ? 8 : 4; }
    if (BAR(y, x)) x = 6;
    int moved = x != *kx || y != *ky;
    *kx = x; *ky = y;
    return moved;
}
static void button(int y, int stop) {                 /* round button in the input bar: up arrow = send, square = stop */
    rrect(214, y + 2, 12, 12, 6, C(BTN));
    if (stop) { rect(218, y + 6, 4, 4, INK(BTN)); return; }
    for (int i = 0; i < 3; i++) hline(219 - i, y + 5 + i, 2 + 2 * i, INK(BTN));
    rect(219, y + 7, 2, 5, INK(BTN));
}
void ui_reply_button(int stop) { button(141, stop); }   /* while Koko is answering the button is a stop square */
static void pill_at(int y, int typing, const char *msg, int mlen) {
    rrect(6, y, 228, 16, 8, C(PILL));
    button(y, 0);
    if (!typing || !mlen) { text(16, y + 1, typing ? "Type a message" : "Say something to Koko", PILL_DIM); if (!typing) return; }
    int from = 0;
    while (width_n(msg + from, mlen - from) > 186) from++;                 /* a long message shows its end */
    int x = text_n(16, y + 1, msg + from, mlen - from, PILL);
    if (mlen) rect(x + 1, y + 3, 1, 10, INK(PILL));                        /* caret */
    else rect(14, y + 3, 1, 10, INK(PILL));
}
void ui_pill(int typing, const char *msg, int mlen) { pill_at(typing ? UI_PILL_Y : 141, typing, msg, mlen); }
void ui_chips(int n, const char *a, int alen, const char *b, int blen) {
    rect(0, UI_CHIP_Y, 240, 14, C(BG));
    int x = 8;
    for (int k = 0; k < n; k++) {
        const char *p = k ? b : a; int len = k ? blen : alen;
        if (len && p[0] == ' ') { p++; len--; }
        int w = width_n(p, len) + 24;
        rrect(x, UI_CHIP_Y, w, 14, 6, C(CHIP));
        glyph(x + 6, UI_CHIP_Y, k ? 'R' : 'L', CHIP_ACC, 0);
        text_n(x + 16, UI_CHIP_Y, p, len, CHIP);
        x += w + 6;
    }
}
int ui_chip_at(int x, int n, const char *a, int alen, const char *b, int blen) {
    int x0 = 8;
    for (int k = 0; k < n; k++) {
        const char *p = k ? b : a; int len = k ? blen : alen;
        if (len && p[0] == ' ') { p++; len--; }
        int w = width_n(p, len) + 24;
        if (x >= x0 && x < x0 + w) return k;
        x0 += w + 6;
    }
    return -1;
}
void ui_keycap(int r, int c, int sel, int upper) {
    int ch = keyrow[r][c], x = UI_KEY_X + c * UI_KEY_W, y = UI_KEY_Y + r * UI_KEY_H;
    if (BAR(r, c)) {                                                       /* the space bar: one wide key, always outlined */
        if (c != 6) return;
        int p = sel ? SEL : CHIP, bx = UI_KEY_X + 5 * UI_KEY_W;
        rect(bx, y, 3 * UI_KEY_W, UI_KEY_H, C(PANEL));
        rrect(bx + 2, y, 3 * UI_KEY_W - 4, UI_KEY_H - 1, 6, C(p));
        rect(bx + 15, y + 9, 24, 1, INK(p)); rect(bx + 15, y + 7, 1, 3, INK(p)); rect(bx + 38, y + 7, 1, 3, INK(p));
        return;
    }
    if (upper && r < 2) ch -= 32;
    rect(x, y, UI_KEY_W, UI_KEY_H, C(PANEL));
    if (sel) rrect(x + 1, y, UI_KEY_W - 2, UI_KEY_H - 1, 4, C(SEL));
    int paint = sel ? SEL : PANEL;
    if (ch == UI_KEY_HELP) {                                               /* helpers on/off: a book */
        rect(x + 5, y + 4, 9, 1, INK(paint)); rect(x + 5, y + 10, 9, 1, INK(paint));
        rect(x + 5, y + 4, 1, 7, INK(paint)); rect(x + 13, y + 4, 1, 7, INK(paint)); rect(x + 9, y + 4, 1, 7, INK(paint));
        return;
    }
    if (ch == UI_KEY_NEW) {                                                /* new chat: a page with a plus */
        rect(x + 5, y + 3, 9, 1, INK(paint)); rect(x + 5, y + 11, 9, 1, INK(paint));
        rect(x + 5, y + 3, 1, 9, INK(paint)); rect(x + 13, y + 3, 1, 9, INK(paint));
        rect(x + 7, y + 7, 5, 1, INK(paint)); rect(x + 9, y + 5, 1, 5, INK(paint));
        return;
    }
    if (ch == '<') { for (int i = 0; i < 4; i++) rect(x + 5 + i, y + 6 - i, 1, 1 + 2 * i, INK(paint)); rect(x + 9, y + 6, 5, 1, INK(paint)); return; }  /* delete */
    glyph(x + (UI_KEY_W - FONT_BODY_W[ch - 32]) / 2, y, ch, paint, 0);
}
void ui_keyboard(int kx, int ky, int upper) {
    rect(0, UI_KEY_Y - 3, 240, 160 - UI_KEY_Y + 3, C(PANEL));
    for (int r = 0; r < 4; r++) { YIELD(); for (int c = 0; c < 13; c++) ui_keycap(r, c, r == ky && c == kx, upper); }
}
void ui_typing(const char *msg, int mlen, int kx, int ky, int upper) {
    chat_bot = UI_PILL_Y - 4;
    chat_draw();
    rect(0, chat_bot, 240, 160 - chat_bot, C(BG));
    YIELD();
    ui_pill(1, msg, mlen); ui_chips(0, 0, 0, 0, 0); ui_keyboard(kx, ky, upper);
}

/* ---------------- reply layout ---------------- */
void ui_reply_layout(const char *msg, int mlen) {
    chat_bot = 138;
    line_push(L_USER); line_add(msg, mlen, USER_W);
    rect(0, UI_PILL_Y - 4, 240, 160 - UI_PILL_Y + 4, C(BG));
    chat_draw();
    ui_pill(0, 0, 0);
}
void ui_reply_start(int mark) { line_push(L_KOKO); lines[nline - 1].mark = (uint8_t)mark; chat_update(nline - 1); }
void ui_redraw(void) { chat_draw(); }
void ui_reply_text(const char *p, int n) {
    int before = lines[nline - 1].n;
    int w = line_add(p, n, KOKO_W);
    if (w) chat_update(nline - 1 - w);
    else if (cur_y < 0) chat_draw();
    else {
        text_n(10 + (lines[nline - 1].mark ? MARK_W : 0) + width_n(lines[nline - 1].t, before), cur_y, lines[nline - 1].t + before, lines[nline - 1].n - before, BG);
        lines[nline - 1].dn = lines[nline - 1].n;
    }
}
