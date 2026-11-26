/* Koko's screen: everything that is drawn (engine/game.c, engine/cardtest.c). 240x160, one palette index per pixel.
 * kokoSD 1.1: our apps' look and their two fonts (engine/text); text is UTF-8. */
#ifndef KOKO_UI_H
#define KOKO_UI_H
#include <stdint.h>

enum { L_KOKO, L_USER, L_NOTE };                 /* kinds of chat line */
#define UI_PAL_N 64
extern uint16_t ui_pal[UI_PAL_N];                /* 15-bit colours for palette indices 0..63 */
#ifndef __arm__
extern uint8_t ui_fb[240 * 160];                 /* the screen, when it is not the GBA's video memory */
#endif
#define UI_MSG_MAX 56                            /* bytes in a message (UTF-8: an accented letter takes 2) */
extern void (*ui_yield)(void);                   /* optional hook called during long redraws */
void ui_init(void);                              /* palette values and a cleared screen */
void ui_boot(const char *what, const char *detail);   /* kokoSD 1.1: a start-up step on the screen */
void ui_helpers(int on);                         /* off: "Facts off" in the header (every word from the model) */
void ui_title(void);                             /* title screen: the name, "Press START" */
void ui_title_prompt(int on);
void ui_title_note2(const char *t);
void ui_title_tick(void);                        /* one title frame (nothing moves any more) */
void ui_redraw(void);                            /* draw the whole chat again (tests compare this with the incremental drawing) */
void ui_clear(void);                             /* forget the chat lines (new chat) */
int ui_scroll(int d);                            /* kokoSD 1.2: one line up (d > 0) or down; 0 if it could not move */
void ui_scroll_home(void);                       /* back to the newest lines */
int ui_scrolled(void);                           /* lines hidden below the view (0: the newest are shown) */
int ui_log_io(int (*io)(void *p, unsigned n, void *ctx), void *ctx, int writing);   /* the chat's lines for a saved chat */
void ui_message(const char *t);
void ui_lines(const char *a, const char *b);     /* kokoSD 1.4.1: two small lines (self test result) */
void ui_ask(const char *title, const char *question, const char *detail, const char *hint);   /* kokoSD 1.3: a question screen ('!' hint: higher, for a typing page) */                  /* a short note on the hint line (typing layout) */
void ui_list(const char *title, const char *const *items, int n, int sel, int top, const char *empty);   /* a list (Saved chats) */
void ui_say(int kind, const char *s, int n);     /* add a chat line (word-wrapped) without drawing */
void ui_header(const char *right);               /* "Koko" centred, a grey hint on the right (or 0) */
void ui_rate(int made, uint32_t ticks);          /* header hint: tokens per second (ticks of 1/16384 s) */
void ui_typing(const char *msg, int mlen, int caret);   /* kokoSD: the whole typing layout (gbawriter typing, no keyboard) */
void ui_input(const char *msg, int mlen, int caret);    /* kokoSD: the input bar with the cursor at byte `caret` */
/* kokoSD 2.5: the original Koko's on-screen keyboard (13 x 4), in kokoSD's look */
#define UI_KEY_ACCENT 1                          /* row 3, column 11: an accent on the last letter */
#define UI_KEY_CAPS 2                            /* row 2, column 12: capitals on / off */
int ui_key(int r, int c);
int ui_key_move(int *kx, int *ky, int dx, int dy);
void ui_keycap(int r, int c, int sel, int upper);
void ui_keyboard(int kx, int ky, int upper);
void ui_typing_kb(const char *msg, int mlen, int kx, int ky, int upper);
void ui_menu(const char *title, const char *const *items, int n, int sel);   /* kokoSD: START+SELECT menu */
void ui_chips(int n, const char *a, int alen, const char *b, int blen); /* the suggestions (START+L, START+R), or the hint */
void ui_hint(int has_text);                      /* kokoSD 1.1: the hint shown without suggestions (START: send / ideas) */
void ui_writer_status(const char *group, const char *mode);   /* kokoSD 1.1: gbawriter's letter group and Shift / Caps */
void ui_reply_layout(const char *msg, int mlen); /* the user's message on the right, chat full height, the field at the bottom */
void ui_reply_button(int stop);                 /* reply layout: "B: stop" (1) or "Press a button to reply" (0) in the field */
void ui_reply_start(int mark);                   /* a new empty line for Koko; mark 1 = looked up, 2 = computed, 0 = the model */
void ui_reply_text(const char *p, int n);        /* add text to Koko's line and draw it */
/* kokoSD: an article to read ("Tell me more"). Wrap the text once, then draw from line `first`. */
/* kokoSD 2.3: a text to read comes from a source: n bytes of it from offset at (at most 300), 0 on an error */
typedef const char *(*ui_src)(uint32_t at, uint32_t n);
int ui_article_wrap(ui_src src, uint32_t n);     /* lines in the text */
int ui_article_rows(void);                       /* lines on one page */
void ui_article(const char *title, int first);
void ui_chat_again(void);                        /* back to the chat (reply layout) */
#endif
