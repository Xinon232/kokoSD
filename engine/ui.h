/* Koko's screen: everything that is drawn, shared by the cartridge (engine/game.c) and the browser build
 * (web/koko_web.c) so both show the same pixels. 240x160, one palette index per pixel. */
#ifndef KOKO_UI_H
#define KOKO_UI_H
#include <stdint.h>

enum { L_KOKO, L_USER, L_NOTE };                 /* kinds of chat line */
#define UI_PAL_N 64
extern uint16_t ui_pal[UI_PAL_N];                /* 15-bit colours for palette indices 0..63 */
#ifndef __arm__
extern uint8_t ui_fb[240 * 160];                 /* the screen, when it is not the GBA's video memory */
#endif
#define UI_MSG_MAX 56                            /* characters in a message */
#define UI_PILL_Y 62                             /* typing layout: input bar, suggestion chips, keyboard */
#define UI_CHIP_Y 81
#define UI_KEY_Y 99
#define UI_KEY_H 15
#define UI_KEY_W 18
#define UI_KEY_X 3

extern void (*ui_yield)(void);                   /* optional hook called during long redraws */
void ui_init(void);                              /* palette values and a cleared screen */
#define UI_KEY_HELP 2                            /* the "helpers on/off" key (row 2, column 12): fact table and calculator */
void ui_helpers(int on);                         /* shown as the dot beside the title: solid = on, hollow = model only */
#define UI_KEY_NEW 1                             /* the "new chat" key (row 3, column 9) */
void ui_title(void);                             /* title screen: Koko's portrait, name, "Press START" */
void ui_title_prompt(int on);                    /* blink the prompt */
void ui_title_tick(void);                        /* one frame of the falling snow */
void ui_redraw(void);                            /* draw the whole chat again (tests compare this with the incremental drawing) */
int ui_key(int row, int col);                    /* the character on a key (4 rows of 13); '_' is space, '<' is delete */
int ui_key_move(int *kx, int *ky, int dx, int dy);   /* one cursor step; the three-cell space bar is one key at column 6 */
void ui_clear(void);                             /* forget the chat lines (new chat) */
void ui_say(int kind, const char *s, int n);     /* add a chat line (word-wrapped) without drawing */
void ui_header(const char *right);               /* title on the left, a dim hint on the right (or 0) */
void ui_rate(int made, uint32_t ticks);          /* header hint: tokens per second (ticks of 1/16384 s) */
void ui_typing(const char *msg, int mlen, int kx, int ky, int upper);   /* the whole typing layout */
void ui_pill(int typing, const char *msg, int mlen);                    /* input bar only */
void ui_chips(int n, const char *a, int alen, const char *b, int blen); /* suggestion chips (L, R) */
int ui_chip_at(int x, int n, const char *a, int alen, const char *b, int blen);   /* which chip is at x, or -1 */
void ui_keycap(int row, int col, int selected, int upper);
void ui_keyboard(int kx, int ky, int upper);
void ui_reply_layout(const char *msg, int mlen); /* the user's message in a bubble, chat full height, input bar at the bottom */
void ui_reply_button(int stop);                 /* reply layout: stop square (1) or send arrow (0) in the input bar */
void ui_reply_start(int mark);                   /* a new empty line for Koko; mark 1 = looked up, 2 = computed, 0 = the model */
void ui_reply_text(const char *p, int n);        /* add text to Koko's line and draw it */
#endif
