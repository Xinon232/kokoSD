/* Koko inference engine. One source for the GBA and for the native twin. */
#ifndef KOKO_H
#define KOKO_H
#include <stdint.h>
#include "koko_layout.h"

#ifdef __arm__
extern const uint8_t koko_blob[];                /* cartridge data (engine/export.py), linked into the ROM */
#else
extern const uint8_t *koko_blob;                 /* the same file, loaded by the native twin */
#endif
/* Optional profiling: cycles per section when built with -DKOKO_PROF (GBA only). */
enum { KP_EMBED, KP_NORM, KP_BUILD, KP_PROJ, KP_KEYS, KP_SCAN, KP_MIX, KP_ATTOUT, KP_MIXER, KP_W1, KP_W2, KP_HEAD, KP_N };
extern uint32_t koko_prof[KP_N];

extern void (*koko_yield)(void);                 /* optional hook, called every few milliseconds of work */
void koko_reset(void);
/* Feed one token; afterwards koko_h holds the normalised final state (Q12). Conversations of any length:
 * attention sees the last K_T tokens. */
void koko_step(int tok);
extern int32_t koko_h[K_D];
/* The same step in slices: begin, then call part until it returns 0 (one layer per call). */
void koko_step_begin(int tok);
int koko_step_part(void);
/* Remember this point / take back every token fed since (see koko.c). */
void koko_mark(void);
void koko_rewind(void);
/* kokoSD 1.2: saved chats: Koko's state at the mark (see koko.c), written (writing 1) or read through io. */
int koko_state_io(int (*io)(void *p, unsigned n, void *ctx), void *ctx, int writing);
/* Q12 output scores for n token ids (copy votes included). */
void koko_scores(const uint16_t *rows, int n, int32_t *out);
/* Fast output step: the K_TOPN best candidates (best first); returns how many. *scored = rows scored. */
int koko_topn(const uint16_t **ids, const int32_t **scores, int *scored);
/* Sample the next token from them. rng: xorshift32 state (non-zero). inv_temp_q8: 512 = temperature 0.5. */
int koko_pick(uint32_t *rng, int inv_temp_q8);
/* Repeated-phrase rule for koko_pick: on between these two calls, for the tokens fed after koko_reply_begin. */
void koko_reply_begin(void);
void koko_reply_end(void);
/* Typing suggestions: up to 3 whole-word tokens that extend the typed prefix; returns how many. */
int koko_suggest(const char *prefix, int plen, int at_start, uint16_t *out);
/* Tokenizer: typed ASCII -> token ids (returns how many); token id -> its text (returns the length). */
int koko_encode(const char *s, int n, uint16_t *out, int max);
int koko_token_text(int id, const char **p);
#endif
