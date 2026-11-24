/* Koko for the browser: the cartridge engine (engine/koko.c, koko_tok.c, sums.c) behind a small API. */
#include "koko.h"
#include "ui.h"
int sum_reply(const char *s, int n, char *out);
int fact_reply(const char *msg, int n, uint16_t *ids, int max);
#define EXPORT(name) __attribute__((export_name(#name)))

const uint8_t *koko_blob;
void *memset(void *d, int c, unsigned long n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }

extern uint8_t __heap_base;
EXPORT(web_alloc) void *web_alloc(uint32_t n) {             /* one allocation: the model data */
    uint32_t base = ((uint32_t)&__heap_base + 15) & ~15u, need = base + n;
    uint32_t have = (uint32_t)__builtin_wasm_memory_size(0) << 16;
    if (need > have && __builtin_wasm_memory_grow(0, (need - have + 65535) >> 16) == (unsigned long)-1) return 0;
    return (void *)base;
}
EXPORT(web_init) void web_init(const uint8_t *blob) { koko_blob = blob; koko_reset(); }

static char text[1024];
static uint16_t ids[512];
static uint32_t rng = 1;
EXPORT(web_text) char *web_text(void) { return text; }      /* shared text buffer */
EXPORT(web_ids) uint16_t *web_ids(void) { return ids; }      /* shared token buffer */
EXPORT(web_seed) void web_seed(uint32_t s) { rng = s ? s : 1; }
EXPORT(web_get_seed) uint32_t web_get_seed(void) { return rng; }
EXPORT(web_reset) void web_reset(void) { koko_reset(); }
EXPORT(web_step) void web_step(int tok) { koko_step(tok); }
EXPORT(web_pick) int web_pick(int inv_temp_q8) { return koko_pick(&rng, inv_temp_q8); }
EXPORT(web_reply_begin) void web_reply_begin(void) { koko_reply_begin(); }
EXPORT(web_reply_end) void web_reply_end(void) { koko_reply_end(); }
EXPORT(web_mark) void web_mark(void) { koko_mark(); }
EXPORT(web_rewind) void web_rewind(void) { koko_rewind(); }
EXPORT(web_encode) int web_encode(int n) { return koko_encode(text, n, ids, 512); }            /* text -> ids */
EXPORT(web_token) int web_token(int id) {                                                     /* id -> text */
    const char *p; int n = koko_token_text(id, &p);
    for (int i = 0; i < n; i++) text[i] = p[i];
    return n;
}
static uint16_t sug[3];
static int nsug;
EXPORT(web_suggest) int web_suggest(int plen, int at_start) {                                 /* prefix in text[512..] */
    nsug = plen < 0 ? 0 : koko_suggest(text + 512, plen, at_start, sug);
    if (nsug > 2) nsug = 2;                                                                   /* the cartridge shows two */
    for (int i = 0; i < nsug; i++) ids[i] = sug[i];
    return nsug;
}
EXPORT(web_sum) int web_sum(int n) { return sum_reply(text, n, text + 512); }                  /* answer at text[512..] */

/* ---- the cartridge's screen (engine/ui.c), drawn into ui_fb ---- */
static void chip_text(const char **a, int *al, const char **b, int *bl) {
    *a = *b = 0; *al = *bl = 0;
    if (nsug > 0) *al = koko_token_text(sug[0], a);
    if (nsug > 1) *bl = koko_token_text(sug[1], b);
}
EXPORT(web_fb) uint8_t *web_fb(void) { return ui_fb; }
EXPORT(web_pal) uint16_t *web_pal(void) { return ui_pal; }
EXPORT(web_ui_init) void web_ui_init(void) { ui_init(); }
EXPORT(web_ui_say) void web_ui_say(int kind, int n) { ui_say(kind, text, n); }
EXPORT(web_ui_header) void web_ui_header(int n) { text[512 + n] = 0; ui_header(n ? text + 512 : 0); }
EXPORT(web_ui_rate) void web_ui_rate(int made, uint32_t ticks) { ui_rate(made, ticks); }
EXPORT(web_ui_typing) void web_ui_typing(int mlen, int kx, int ky, int upper) { ui_typing(text, mlen, kx, ky, upper); }
EXPORT(web_ui_pill) void web_ui_pill(int typing, int mlen) { ui_pill(typing, text, mlen); }
EXPORT(web_ui_chips) void web_ui_chips(void) { const char *a, *b; int al, bl; chip_text(&a, &al, &b, &bl); ui_chips(nsug, a, al, b, bl); }
EXPORT(web_ui_chip_at) int web_ui_chip_at(int x) { const char *a, *b; int al, bl; chip_text(&a, &al, &b, &bl); return ui_chip_at(x, nsug, a, al, b, bl); }
EXPORT(web_ui_keycap) void web_ui_keycap(int r, int c, int sel, int upper) { ui_keycap(r, c, sel, upper); }
EXPORT(web_ui_keyboard) void web_ui_keyboard(int kx, int ky, int upper) { ui_keyboard(kx, ky, upper); }
EXPORT(web_ui_reply_layout) void web_ui_reply_layout(int mlen) { ui_reply_layout(text, mlen); }
EXPORT(web_ui_reply_start) void web_ui_reply_start(int mark) { ui_reply_start(mark); }
EXPORT(web_ui_helpers) void web_ui_helpers(int on) { ui_helpers(on); }
EXPORT(web_ui_reply_token) void web_ui_reply_token(int id) { const char *p; int n = koko_token_text(id, &p); ui_reply_text(p, n); }
EXPORT(web_ui_reply_button) void web_ui_reply_button(int stop) { ui_reply_button(stop); }
EXPORT(web_ui_clear) void web_ui_clear(void) { ui_clear(); }
EXPORT(web_fact) int web_fact(int n) { return fact_reply(text, n, ids, 64); }                 /* message in text -> sentence ids */
EXPORT(web_ui_title) void web_ui_title(void) { ui_title(); }
EXPORT(web_ui_title_prompt) void web_ui_title_prompt(int on) { ui_title_prompt(on); }
EXPORT(web_ui_title_tick) void web_ui_title_tick(void) { ui_title_tick(); }
EXPORT(web_ui_redraw) void web_ui_redraw(void) { ui_redraw(); }
static int mvx, mvy;
EXPORT(web_key_move) int web_key_move(int kx, int ky, int dx, int dy) { mvx = kx; mvy = ky; ui_key_move(&mvx, &mvy, dx, dy); return mvx | mvy << 8; }
