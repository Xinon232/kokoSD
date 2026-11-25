/* kokoSD: the fact file on the SD card (tools/facts_sd.py). GPL-3.0-or-later. */
#ifndef KOKO_FACTS_SD_H
#define KOKO_FACTS_SD_H
#include <stdint.h>
/* Opens /koko/facts.dat after sd_mount(): 1 ready, 0 none (or made for other weights). */
int sdf_open(void);
extern int sdf_ready;
extern uint32_t sdf_count;                  /* titles (with redirects) in the file */
/* The opening sentence for a title hash (engine/facts.c's FNV-1a of the key):
 * token ids (at most max), or 0. Remembers the article for sdf_article(). */
int sdf_lookup(uint32_t hash, uint16_t *ids, int max);
/* The article of the last successful lookup: its title and plain text (read
 * from the card into a buffer kept here). Returns the text length, 0 if none. */
int sdf_article(const char **title, const char **text);
void sdf_forget(void);                      /* no article to tell more about */
int sdf_probe(uint32_t r, int article);     /* kokoSD 1.4 self test: a lookup of a real title (1 = ok) */
#endif
