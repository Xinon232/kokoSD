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
void ui_title_prompt(int on);                    /* blink the prompt */
void ui_title_tick(void);                        /* one title frame (nothing moves any more) */
void ui_redraw(void);                            /* draw the whole chat again (tests compare this with the incremental drawing) */
void ui_clear(void);                             /* forget the chat lines (new chat) */
void ui_say(int kind, const char *s, int n);     /* add a chat line (word-wrapped) without drawing */
void ui_header(const char *right);               /* "Koko" centred, a grey hint on the right (or 0) */
void ui_rate(int made, uint32_t ticks);          /* header hint: tokens per second (ticks of 1/16384 s) */
void ui_typing(const char *msg, int mlen, int caret);   /* kokoSD: the whole typing layout (gbawriter typing, no keyboard) */
void ui_input(const char *msg, int mlen, int caret);    /* kokoSD: the input bar with the cursor at byte `caret` */
void ui_menu(const char *title, const char *const *items, int n, int sel);   /* kokoSD: START+SELECT menu */
void ui_chips(int n, const char *a, int alen, const char *b, int blen); /* the suggestions (START+L, START+R), or the hint */
void ui_hint(int has_text);                      /* kokoSD 1.1: the hint shown without suggestions (START: send / ideas) */
void ui_writer_status(const char *group, const char *mode);   /* kokoSD 1.1: gbawriter's letter group and Shift / Caps */
void ui_reply_layout(const char *msg, int mlen); /* the user's message on the right, chat full height, the field at the bottom */
void ui_reply_button(int stop);                 /* reply layout: "B: stop" (1) or "Press a button to reply" (0) in the field */
void ui_reply_start(int mark);                   /* a new empty line for Koko; mark 1 = looked up, 2 = computed, 0 = the model */
void ui_reply_text(const char *p, int n);        /* add text to Koko's line and draw it */
/* kokoSD: an article to read ("Tell me more"). Wrap the text once, then draw from line `first`. */
int ui_article_wrap(const char *t, int n);       /* lines in the text */
int ui_article_rows(void);                       /* lines on one page */
void ui_article(const char *title, const char *t, int first);
void ui_chat_again(void);                        /* back to the chat (reply layout) */
#endif
