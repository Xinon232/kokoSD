/* kokoSD 2.4: what Koko remembers about you, across chats (GPL-3.0-or-later).
 *
 * After each message, a few plain sentence shapes are noted: "my name is ...", "call me ...", "I am N years
 * old", "I live in ...", "I like / love ...", "my favorite X is ...", "I have a ...". They are kept in
 * /koko/memory.txt on the SD card, one sentence a line in your own words ("My name is Sam."), so the file can
 * be read and changed on a computer too. At each new chat Koko is told them first, in this form (it worked best
 * in a test with the native engine, host/): your turn "My name is Sam. I live in Berlin." and Koko's turn
 * "Got it! Your name is Sam. You live in Berlin." */
#include <stdint.h>
#include "memory.h"
#ifndef MEM_HOST
#include "ff.h"
#include "sd.h"
#define EW __attribute__((section(".bss.ewram")))
#else
#define EW
#endif

enum { K_NONE, K_NAME, K_AGE, K_LIVE, K_FAV, K_LIKE, K_HAVE };
#define SLOTS 8
#define VAL 32
typedef struct { uint8_t kind; char key[16]; char val[VAL]; } slot;
static slot m[SLOTS] EW;
static int mn;                                    /* slots in use, oldest first */
int mem_on = 1;

static int low(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
static int is_word(int c) { c = low(c); return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'' || c == '-'; }
static int same(const char *a, const char *b) { while (*a && low(*a) == low(*b)) { a++; b++; } return !*a && !*b; }
/* s[i..] starts with the words w (lower case) at a word start: the length matched, else 0 */
static int at(const char *s, int n, int i, const char *w) {
    if (i > 0 && is_word(s[i - 1])) return 0;
    int k = 0;
    for (; w[k]; k++) if (i + k >= n || low(s[i + k]) != w[k]) return 0;
    if (i + k < n && is_word(s[i + k]) && is_word(w[k - 1])) return 0;
    return k;
}
/* the value after a pattern: up to `words` words, to the end of the sentence or "and", "but", "because" ... */
static int value(const char *s, int n, int i, int words, char *out) {
    static const char *const stop[] = { "and ", "but ", "because ", "so ", "too", "very much", "a lot", "who ", "which ", "that ", 0 };
    int o = 0, w = 0;
    while (i < n && s[i] == ' ') i++;
    while (i < n && w < words) {
        for (int k = 0; stop[k]; k++) if (at(s, n, i, stop[k])) { i = n; break; }
        if (i >= n || !is_word(s[i])) break;
        if (o) { if (o >= VAL - 1) break; out[o++] = ' '; }
        int st = o;
        while (i < n && is_word(s[i]) && o < VAL - 1) out[o++] = s[i++];
        if (i < n && is_word(s[i])) { o = st ? st - 1 : 0; break; }   /* a word cut off: not kept */
        w++;
        while (i < n && s[i] == ' ') i++;
    }
    while (o && out[o - 1] == ' ') o--;
    out[o] = 0;
    return o;
}
static void put(int kind, const char *key, const char *val) {
    int i = 0;
    for (; i < mn; i++) {                         /* the same fact again: the new wording, as the newest */
        if (m[i].kind != kind) continue;
        if (kind == K_FAV ? same(m[i].key, key) : (kind == K_LIKE || kind == K_HAVE) ? same(m[i].val, val) : 1) break;
    }
    int same_kind = 0;
    for (int j = 0; j < mn; j++) same_kind += m[j].kind == kind;
    if (i == mn && (kind == K_LIKE || kind == K_FAV) && same_kind >= 2) {   /* two likes, two favourites: the oldest goes */
        for (i = 0; m[i].kind != kind; i++) {}
    } else if (i == mn && kind == K_HAVE && same_kind >= 1) {
        for (i = 0; m[i].kind != kind; i++) {}
    }
    if (i == mn && mn == SLOTS) i = 0;
    if (i < mn) { for (int j = i; j < mn - 1; j++) m[j] = m[j + 1]; mn--; }
    slot *t = &m[mn++];
    t->kind = (uint8_t)kind;
    int k = 0; for (; key[k] && k < 15; k++) t->key[k] = (char)low(key[k]); t->key[k] = 0;
    for (k = 0; val[k] && k < VAL - 1; k++) t->val[k] = val[k];
    t->val[k] = 0;
    if (kind == K_NAME && t->val[0] >= 'a' && t->val[0] <= 'z') t->val[0] = (char)(t->val[0] - 32);
}
static int unwanted(const char *v) {              /* "I like it", "I have no ...": nothing to remember */
    static const char *const w[] = { "it", "that", "this", "them", "you", "him", "her", "to", "no", "not", "nothing",
                                     "a", "an", "the", "some", "one", "fun", "too", "so", "really", "what", "how", "why", 0 };
    for (int k = 0; w[k]; k++) if (same(v, w[k])) return 1;
    return !v[0] || (low(v[0]) == 'n' && low(v[1]) == 'o' && v[2] == 't' && (v[3] == ' ' || !v[3]));
}

int mem_learn(const char *s, int n) {
    char v[VAL], key[16];
    int got = 0, k;
    for (int i = 0; i < n; i++) {
        if (i && is_word(s[i - 1])) continue;
        if ((k = at(s, n, i, "my name is ")) || (k = at(s, n, i, "my name's ")) || (k = at(s, n, i, "call me ")) ||
            (k = at(s, n, i, "i am called ")) || (k = at(s, n, i, "i'm called "))) {
            if (value(s, n, i + k, 1, v) && !unwanted(v) && !(v[0] >= '0' && v[0] <= '9')) { put(K_NAME, "", v); got = 1; }
            continue;
        }
        if ((k = at(s, n, i, "i live in ")) || (k = at(s, n, i, "i'm from ")) || (k = at(s, n, i, "i am from ")) || (k = at(s, n, i, "i come from "))) {
            if (value(s, n, i + k, 3, v) && !unwanted(v)) { put(K_LIVE, "", v); got = 1; }
            continue;
        }
        if ((k = at(s, n, i, "i am ")) || (k = at(s, n, i, "i'm "))) {
            int j = i + k, o = 0;
            while (j < n && s[j] >= '0' && s[j] <= '9' && o < 3) v[o++] = s[j++];
            v[o] = 0;
            if (o && ((j < n && s[j] == ' ' && (at(s, n, j + 1, "years old") || at(s, n, j + 1, "year old"))) || j == n || s[j] == '.' || s[j] == '!' || s[j] == ',')) { put(K_AGE, "", v); got = 1; }
            continue;
        }
        if ((k = at(s, n, i, "i really like ")) || (k = at(s, n, i, "i really love ")) || (k = at(s, n, i, "i like ")) ||
            (k = at(s, n, i, "i love ")) || (k = at(s, n, i, "i enjoy "))) {
            if (value(s, n, i + k, 3, v) && !unwanted(v)) { put(K_LIKE, "", v); got = 1; }
            continue;
        }
        if ((k = at(s, n, i, "my favorite ")) || (k = at(s, n, i, "my favourite "))) {
            int j = i + k, o = 0;                          /* the thing: one or two words before "is" */
            while (j < n && o < 15 && !at(s, n, j, "is ") && !at(s, n, j, "are ")) {
                if (!is_word(s[j]) && s[j] != ' ') { o = -1; break; }
                key[o++] = s[j++];
            }
            if (o <= 0 || o >= 15) continue;
            while (o && key[o - 1] == ' ') o--;
            key[o] = 0;
            int c = at(s, n, j, "is ") ? 3 : at(s, n, j, "are ") ? 4 : 0;
            if (o && c && value(s, n, j + c, 3, v) && !unwanted(v)) { put(K_FAV, key, v); got = 1; }
            continue;
        }
        if ((k = at(s, n, i, "i have a ")) || (k = at(s, n, i, "i have an ")) || (k = at(s, n, i, "i've got a ")) || (k = at(s, n, i, "i have got a "))) {
            if (value(s, n, i + k, 4, v) && !unwanted(v)) {
                char a[VAL]; int o = 0; const char *p = (s[i + k - 2] == 'n' && s[i + k - 3] == 'a') ? "an " : "a ";
                while (*p) a[o++] = *p++;
                for (int q = 0; v[q] && o < VAL - 1; q++) a[o++] = v[q];
                a[o] = 0;
                put(K_HAVE, "", a); got = 1;
            }
            continue;
        }
    }
    return got;
}

/* one remembered fact as a sentence: yours ("My name is Sam.") or Koko's ("Your name is Sam.") */
static int sentence(const slot *t, int you, char *out, int max) {
    const char *a = "", *b = "", *c = "";
    switch (t->kind) {
    case K_NAME: a = you ? "Your name is " : "My name is "; break;
    case K_AGE: a = you ? "You are " : "I am "; c = " years old"; break;
    case K_LIVE: a = you ? "You live in " : "I live in "; break;
    case K_LIKE: a = you ? "You like " : "I like "; break;
    case K_FAV: a = you ? "Your favorite " : "My favorite "; b = " is "; break;
    case K_HAVE: a = you ? "You have " : "I have "; break;
    default: return 0;
    }
    int o = 0;
    for (const char *p = a; *p && o < max; p++) out[o++] = *p;
    if (t->kind == K_FAV) { for (const char *p = t->key; *p && o < max; p++) out[o++] = *p; for (const char *p = b; *p && o < max; p++) out[o++] = *p; }
    for (const char *p = t->val; *p && o < max; p++) out[o++] = *p;
    for (const char *p = c; *p && o < max; p++) out[o++] = *p;
    if (o < max) out[o++] = '.';
    return o;
}
int mem_text(int you, char *out, int max) {
    int o = 0;
    for (int kind = K_NAME; kind <= K_HAVE; kind++)     /* your name first, then the rest, in a fixed order */
        for (int i = 0; i < mn; i++) {
            if (m[i].kind != kind) continue;
            if (o && o < max) out[o++] = you == 2 ? '\n' : ' ';
            o += sentence(&m[i], you == 1, out + o, max - o);
        }
    if (o < max) out[o] = 0;
    return o;
}
int mem_count(void) { return mn; }
void mem_forget(void) { mn = 0; }

#ifndef MEM_HOST
FIL *chat_fil(void);                              /* engine/chats.c: its file object (RAM is short) */
char mem_buf[MEM_BUF] EW;
#define mf (*chat_fil())
#define buf mem_buf
int mem_load(void) {
    mn = 0;
    if (!sd_ready) return 0;
    UINT got = 0;
    if (f_open(&mf, "/koko/memory.txt", FA_READ) != FR_OK) return 0;
    int ok = f_read(&mf, buf, sizeof buf - 1, &got) == FR_OK;
    f_close(&mf);
    if (!ok) return 0;
    for (UINT i = 0; i < got; i++) if (buf[i] == '\r' || buf[i] == '\n') buf[i] = '.';
    mem_learn(buf, (int)got);                     /* the file's sentences are the patterns themselves */
    return mn;
}
int mem_save(void) {
    if (!sd_ready) return 0;
    int n = mem_text(2, buf, (int)sizeof buf - 1);
    if (n < (int)sizeof buf - 1) buf[n++] = '\n';
    UINT done = 0;
    f_mkdir("/koko");
    if (f_open(&mf, "/koko/memory.txt", FA_WRITE | FA_CREATE_ALWAYS) != FR_OK) return 0;
    int ok = f_write(&mf, buf, (UINT)n, &done) == FR_OK && done == (UINT)n;
    return f_close(&mf) == FR_OK && ok;
}
#endif
