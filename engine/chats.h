/* kokoSD 1.2: saved chats on the SD card (engine/chats.c). GPL-3.0-or-later. */
#ifndef KOKO_CHATS_H
#define KOKO_CHATS_H
#include <stdint.h>
#define CHAT_TITLE 64                   /* the first message (UTF-8, ending in 0) */
#define CHAT_HEAD 80                    /* magic 8, weights check 4, 4 spare, title 64 */
#define CHAT_MAX_ID 1000
/* Start writing chat *id (a new number when *id < 0; 0 when the card cannot be written). */
int chat_write_begin(int *id, const char *title);
/* Start reading chat id; its title into title. 0: missing, damaged or made with other weights. */
int chat_read_begin(int id, char *title);
int chat_io(void *p, unsigned n, void *ctx);   /* write or read the next n bytes (1 = ok) */
int chat_end(void);                            /* close; 1 when everything went well */
/* The saved chats on the card: titles and numbers (at most max). */
int chat_list(char (*titles)[CHAT_TITLE], int16_t *ids, int max);
int chat_delete(int id);                       /* kokoSD 1.3 */
int chat_rename(int id, const char *title);    /* kokoSD 1.3: a new title (1 = done) */
#endif
