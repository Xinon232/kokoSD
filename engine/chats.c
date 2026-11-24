/* kokoSD 1.2: saved chats on the SD card, /koko/chats/chatNNN.kch. GPL-3.0-or-later.
 * A file: "KOKOCHT1", the weights check (as the fact file's: FNV-1a over token texts 3..1023, so a chat made
 * with other weights is not read), its first message (64 bytes, the title in Saved chats), then what game.c
 * writes through chat_io: the history, the chat's lines and Koko's state (engine/koko.c koko_state_io). */
#include "chats.h"
#include "sd.h"
#include "koko.h"

#define SD __attribute__((section(".bss.sd")))
static FIL cf SD;
static int writing, open_ok;

static uint32_t weights_check(void) {
    uint32_t x = 0x811C9DC5u;
    for (int id = 3; id < 1024; id++) {
        const char *p; int l = koko_token_text(id, &p);
        for (int i = 0; i < l; i++) x = (x ^ (uint8_t)p[i]) * 0x01000193u;
    }
    return x;
}
static void chat_name(char *b, int id) {
    const char *d = "/koko/chats/chat"; int k = 0;
    while (*d) b[k++] = *d++;
    b[k++] = (char)('0' + id / 100 % 10); b[k++] = (char)('0' + id / 10 % 10); b[k++] = (char)('0' + id % 10);
    for (const char *e = ".kch"; *e; e++) b[k++] = *e;
    b[k] = 0;
}
int chat_io(void *p, unsigned n, void *ctx) {
    (void)ctx;
    UINT done;
    if (!open_ok) return 0;
    FRESULT r = writing ? f_write(&cf, p, n, &done) : f_read(&cf, p, n, &done);
    if (r != FR_OK || done != n) open_ok = 0;
    return open_ok;
}
int chat_write_begin(int *id, const char *title) {
    char name[32];
    if (!sd_ready) return 0;
    f_mkdir("/koko"); f_mkdir("/koko/chats");                  /* (already there: an error that does not matter) */
    if (*id < 0) {                                             /* a new chat: the first free number */
        FILINFO fi;
        for (*id = 0; *id < CHAT_MAX_ID; (*id)++) { chat_name(name, *id); if (f_stat(name, &fi) != FR_OK) break; }
        if (*id >= CHAT_MAX_ID) { *id = -1; return 0; }
    }
    chat_name(name, *id);
    if (f_open(&cf, name, FA_WRITE | FA_CREATE_ALWAYS) != FR_OK) return 0;
    writing = 1; open_ok = 1;
    char head[CHAT_HEAD];
    for (int i = 0; i < CHAT_HEAD; i++) head[i] = 0;
    const char *m = "KOKOCHT1";
    for (int i = 0; i < 8; i++) head[i] = m[i];
    uint32_t w = weights_check();
    for (int i = 0; i < 4; i++) head[8 + i] = (char)(w >> (8 * i));
    for (int i = 0; title[i] && i < CHAT_TITLE - 1; i++) head[16 + i] = title[i];
    return chat_io(head, CHAT_HEAD, 0);
}
static int read_head(int id, char *title) {
    char name[32], head[CHAT_HEAD];
    chat_name(name, id);
    if (f_open(&cf, name, FA_READ) != FR_OK) return 0;
    writing = 0; open_ok = 1;
    uint32_t w = 0;
    if (!chat_io(head, CHAT_HEAD, 0)) { f_close(&cf); return 0; }
    for (int i = 0; i < 4; i++) w |= (uint32_t)(uint8_t)head[8 + i] << (8 * i);
    const char *m = "KOKOCHT1";
    for (int i = 0; i < 8; i++) if (head[i] != m[i]) { f_close(&cf); return 0; }
    if (w != weights_check()) { f_close(&cf); return 0; }
    for (int i = 0; i < CHAT_TITLE; i++) title[i] = head[16 + i];
    title[CHAT_TITLE - 1] = 0;
    return 1;
}
int chat_read_begin(int id, char *title) { return sd_ready && read_head(id, title); }
int chat_end(void) {
    int ok = open_ok;
    if (f_close(&cf) != FR_OK) ok = 0;
    open_ok = 0;
    return ok;
}
int chat_list(char (*titles)[CHAT_TITLE], int16_t *ids, int max) {
    DIR dir; FILINFO fi; int n = 0;
    if (!sd_ready || f_opendir(&dir, "/koko/chats") != FR_OK) return 0;
    while (n < max && f_readdir(&dir, &fi) == FR_OK && fi.fname[0]) {
        const char *f = fi.fname; int id = 0, k = 0;
        const char *p = "chat";
        for (; p[k]; k++) if ((f[k] | 32) != p[k]) break;
        if (p[k]) continue;
        int d = 0;
        for (; f[k] >= '0' && f[k] <= '9' && d < 3; k++, d++) id = id * 10 + (f[k] - '0');
        if (d != 3 || f[k] != '.' || (f[k + 1] | 32) != 'k' || (f[k + 2] | 32) != 'c' || (f[k + 3] | 32) != 'h' || f[k + 4]) continue;
        if (read_head(id, titles[n])) { f_close(&cf); ids[n++] = (int16_t)id; }
    }
    f_closedir(&dir);
    return n;
}
/* kokoSD 1.3: delete chat id (1 = done). */
int chat_delete(int id) {
    char name[32];
    chat_name(name, id);
    return sd_ready && f_unlink(name) == FR_OK;
}
/* kokoSD 1.3: give chat id a new title (its first message is what the list shows). */
int chat_rename(int id, const char *title) {
    char name[32], head[CHAT_HEAD];
    UINT done;
    if (!sd_ready) return 0;
    chat_name(name, id);
    if (f_open(&cf, name, FA_READ | FA_WRITE) != FR_OK) return 0;
    int ok = f_read(&cf, head, CHAT_HEAD, &done) == FR_OK && done == CHAT_HEAD && head[0] == 'K' && head[4] == 'C';
    if (ok) {
        for (int i = 0; i < CHAT_TITLE; i++) head[16 + i] = 0;
        for (int i = 0; title[i] && i < CHAT_TITLE - 1; i++) head[16 + i] = title[i];
        ok = f_lseek(&cf, 16) == FR_OK && f_write(&cf, head + 16, CHAT_TITLE, &done) == FR_OK && done == CHAT_TITLE;
    }
    if (f_close(&cf) != FR_OK) ok = 0;
    return ok;
}
