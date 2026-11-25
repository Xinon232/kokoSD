/* kokoSD: the fact file on the SD card (tools/facts_sd.py). GPL-3.0-or-later. */
#ifndef KOKO_FACTS_SD_H
#define KOKO_FACTS_SD_H
#include <stdint.h>
/* Opens /koko/facts.dat after sd_mount(): 1 ready, 0 none (or made for other weights). */
int sdf_open(void);
extern int sdf_ready, sdf_err;                /* kokoSD 1.4.2: sdf_err: why not (facts_sd.c) */
extern uint32_t sdf_count;                  /* titles (with redirects) in the file */
/* The opening sentence for a title hash (engine/facts.c's FNV-1a of the key):
 * token ids (at most max), or 0. Remembers the article for sdf_article(). */
int sdf_lookup(uint32_t hash, uint16_t *ids, int max);
/* The article of the last successful lookup: its title; returns the text length, 0 if none. kokoSD 2.3: the text
 * is read in parts: sdf_article_text(at, n) gives n bytes from offset at (n at most 6000), 0 on a card error. */
int sdf_article(const char **title);
const char *sdf_article_text(uint32_t at, uint32_t n);
void sdf_forget(void);                      /* no article to tell more about */
/* kokoSD 2.4: the word memory in the fact file (tools/words_sd.py): koko_words for engine/koko.c */
int sdf_words_ready(void);
int sdf_words(const uint16_t *ctx3, uint16_t *ids, uint8_t *share);
int sdf_probe(uint32_t r, int article);     /* kokoSD 1.4 self test: a lookup of a real title (1 = ok) */
#endif
