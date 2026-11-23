#ifndef MODEL_H
#define MODEL_H
#include <stdint.h>
#include "model_data.h"

#ifdef __arm__
/* fast RAM (32 KB, 1 cycle/access) for hot code and small buffers; slow RAM (256 KB) for the rest */
#define IWRAM_CODE __attribute__((section(".iwram.text")))
#define IWRAM_DATA __attribute__((section(".bss.iwram")))
#define EWRAM_DATA __attribute__((section(".bss.ewram")))
#else
#define IWRAM_CODE
#define IWRAM_DATA
#define EWRAM_DATA
#endif

/* Q16 logits of the last llm_forward(tok, 1) */
extern int32_t llm_logits[VOCAB];

void llm_reset(void);
/* Feed one token. want_logits=0 for prompt tokens that only need to update the cache. */
void llm_forward(int tok, int want_logits);

/* Decoding. hist/n: every token so far (prompt included). */
int llm_pick_greedy(const uint16_t *hist, int n);          /* argmax, never completing a repeated 4-gram */
int llm_pick_sample(uint32_t *rng);                        /* temperature 0.7 over the top 8 */

/* Tokenizer */
int tok_encode(const char *text, uint16_t *out, int max);  /* BOS + pieces; returns count */
int tok_piece(int id, const uint8_t **p);                  /* raw bytes of a token; returns length */

#endif
