/* Byte-level BPE tokenizer for typed ASCII text (same result as the training tokenizer). */
#include "koko.h"
#define BU16(o) ((const uint16_t *)(koko_blob + (o)))
#define BU32(o) ((const uint32_t *)(koko_blob + (o)))

static int is_l(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int is_d(int c) { return c >= '0' && c <= '9'; }
static int is_o(int c) { return c != ' ' && !is_l(c) && !is_d(c); }

/* Length of the next piece of s[0..n) under the GPT-2 splitting rule (ASCII, spaces as the only whitespace). */
static int piece(const char *s, int n) {
    if (s[0] == '\'' && n >= 2) {
        if (s[1] == 's' || s[1] == 't' || s[1] == 'm' || s[1] == 'd') return 2;
        if (n >= 3 && ((s[1] == 'r' && s[2] == 'e') || (s[1] == 'v' && s[2] == 'e') || (s[1] == 'l' && s[2] == 'l'))) return 3;
    }
    int i = s[0] == ' ' && n >= 2 ? 1 : 0;                  /* optional leading space */
    if (i < n && s[i] != ' ') {
        int (*cls)(int) = is_l(s[i]) ? is_l : is_d(s[i]) ? is_d : is_o;
        int j = i;
        while (j < n && cls(s[j])) j++;
        return j;
    }
    int j = 0;                                              /* a run of spaces */
    while (j < n && s[j] == ' ') j++;
    return (j < n && j > 1) ? j - 1 : j;                    /* leave the last space for the word after it */
}

static int merge(uint32_t a, uint32_t b, uint32_t *val) {
    const uint32_t *k = BU32(O_TOK_MKEY);
    uint32_t key = a << 14 | b;
    int lo = 0, hi = K_NMERGE - 1;
    while (lo <= hi) {
        int m = (lo + hi) >> 1;
        if (k[m] == key) { *val = BU32(O_TOK_MVAL)[m]; return 1; }
        if (k[m] < key) lo = m + 1; else hi = m - 1;
    }
    return 0;
}

int koko_encode(const char *s, int n, uint16_t *out, int max) {
    int no = 0;
    while (n > 0) {
        int len = piece(s, n), w0 = no;
        for (int i = 0; i < len && no < max; i++) {
            int c = (unsigned char)s[i];
            out[no++] = BU16(O_TOK_BYTE)[c >= 32 && c < 127 ? c : '?'];
        }
        for (;;) {                                           /* merge the best-ranked pair until none is left */
            uint32_t best = 0xFFFFFFFFu, v; int at = -1;
            for (int i = w0; i + 1 < no; i++)
                if (merge(out[i], out[i + 1], &v) && v < best) { best = v; at = i; }
            if (at < 0) break;
            out[at] = (uint16_t)(best & 0xFFFF);
            for (int i = at + 1; i + 1 < no; i++) out[i] = out[i + 1];
            no--;
        }
        s += len; n -= len;
    }
    return no;
}

int koko_token_text(int id, const char **p) {
    const uint32_t *off = BU32(O_TOK_OFF);
    *p = (const char *)(koko_blob + O_TOK_TEXT + off[id]);
    return (int)(off[id + 1] - off[id]);
}
