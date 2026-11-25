/* The fact table: "who was X", "what is X", "tell me about X" ... are answered with the stored opening sentence
 * of X's Simple English Wikipedia article when the cartridge has one (engine/facts_build.py). Otherwise the
 * model answers as usual. */
#include "koko.h"
#ifdef __arm__
#include "facts_sd.h"
#endif
#define BU8(o) ((const uint8_t *)(koko_blob + (o)))
#define BU32(o) ((const uint32_t *)(koko_blob + (o)))

static int lookup(const char *s, int n, uint16_t *ids, int max) {
    uint32_t h = 0x811C9DC5u;
    for (int i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 0x01000193u;
#ifdef __arm__
    int r = sdf_lookup(h, ids, max);                         /* kokoSD: the SD card's file first (every article) */
    return r;     /* kokoSD 1.4.5: the cartridge has no fact table of its own any more (the ROM is 1.9 MB smaller) */
#else
    const uint32_t *hs = BU32(O_FACT_HASH);
    int lo = 0, hi = K_NFACT - 1;
    while (lo <= hi) {
        int m = (lo + hi) >> 1;
        if (hs[m] == h) {
            const uint8_t *p = BU8(O_FACT_TEXT) + BU32(O_FACT_OFF)[m];
            int cnt = *p++;
            if (cnt > max) return 0;
            uint32_t acc = 0; int nb = 0;
            for (int i = 0; i < cnt; i++) {
                while (nb < 14) { acc |= (uint32_t)*p++ << nb; nb += 8; }
                ids[i] = (uint16_t)(acc & 0x3FFF); acc >>= 14; nb -= 14;
            }
            return cnt;
        }
        if (hs[m] < h) lo = m + 1; else hi = m - 1;
    }
    return 0;
#endif
}
static int starts(const char *s, int n, const char *w);
int fact_wide = 1;                                /* kokoSD 2.4: 0 = the 2.3 matching (Settings) */
static int subject(const char *s, int len, uint16_t *ids, int max) {   /* the title, or its singular */
    int r = lookup(s, len, ids, max);
    if (!r && len > 3 && s[len - 1] == 's') {                /* plurals: dogs -> dog, foxes -> fox, puppies -> puppy */
        r = lookup(s, len - 1, ids, max);
        if (!r && s[len - 2] == 'e') r = lookup(s, len - 2, ids, max);
        if (!r && len > 4 && s[len - 3] == 'i' && s[len - 2] == 'e') {
            char t[64];
            for (int i = 0; i < len - 3; i++) t[i] = s[i];
            t[len - 3] = 'y';
            r = lookup(t, len - 2, ids, max);
        }
    }
    return r;
}
/* kokoSD 2.4: when the whole subject is no title, shorter parts of it that are safe to try:
 *  - before a possessive or an apposition: "eminem s songs", "eminem the rapper", "paris in france" -> the first part
 *    (only before "s", "the", "a", "an", "in", "from", "who", "that");
 *  - after descriptive words: "the rapper eminem", "famous singer elvis presley" -> the rest (never across "of",
 *    "in", "from", "for", "on", "at", "to", "with", "by", "and": "president of france" stays unanswered). */
static int is_word(const char *s, int n, const char *const *w) {
    for (int k = 0; w[k]; k++) { int l = 0; while (w[k][l]) l++; if (l == n && starts(s, n, w[k])) return 1; }
    return 0;
}
static int parts(const char *s, int len, uint16_t *ids, int max) {
    static const char *const CUT[] = { "s", "the", "a", "an", "in", "from", "who", "that", "is", "was", 0 };
    static const char *const LINK[] = { "of", "in", "from", "for", "on", "at", "to", "with", "by", "and", "or", "about", 0 };
    int w0[12], wl[12], nw = 0;
    for (int i = 0; i < len && nw < 12;) {
        while (i < len && s[i] == ' ') i++;
        if (i >= len) break;
        w0[nw] = i; while (i < len && s[i] != ' ') i++; wl[nw] = i - w0[nw]; nw++;
    }
    if (nw < 2) return 0;
    for (int k = 1; k < nw && k < 3; k++) {                  /* "planet mercury" -> "mercury planet": Mercury (planet) */
        if (is_word(s + w0[k - 1], wl[k - 1], LINK) || is_word(s + w0[k - 1], wl[k - 1], CUT)) break;
        char t[64]; int n = 0;
        for (int i = w0[k]; i < len && n < 62; i++) t[n++] = s[i];
        t[n++] = ' ';
        for (int i = 0; i < w0[k] - 1 && n < 63; i++) t[n++] = s[i];
        int r = lookup(t, n, ids, max); if (r) return r;
    }
    for (int k = 1; k < nw; k++)                             /* the first words, up to a cut word */
        if (is_word(s + w0[k], wl[k], CUT)) {
            int e = w0[k - 1] + wl[k - 1];
            if (e >= 3) { int r = subject(s, e, ids, max); if (r) return r; }
            break;
        }
    for (int k = 1; k < nw; k++) {                           /* the last words, after descriptive ones */
        if (is_word(s + w0[k - 1], wl[k - 1], LINK)) break;
        int b = w0[k], n = len - b;
        if (n >= 3) { int r = subject(s + b, n, ids, max); if (r) return r; }
    }
    return 0;
}
static int starts(const char *s, int n, const char *w) {
    int i = 0;
    for (; w[i]; i++) if (i >= n || s[i] != w[i]) return 0;
    return i;
}

/* Token ids of the stored sentence about the subject of the message (at most max), or 0 if there is none. */
int fact_reply(const char *msg, int n, uint16_t *ids, int max) {
    static const char *const LEAD[] = { "who is ", "who was ", "who are ", "who were ", "who s ", "whos ", "what is ", "what was ",
        "what are ", "what were ", "what s ", "whats ", "tell me about ", "what do you know about ", "do you know about ",
        "do you know who ", "do you know what ", "where is ", "where are ", "explain ", "define ", 0 };
    static const char *const LEAD2[] = {   /* kokoSD 2.4: more ways to ask (Settings: Fact matching: Wide) */
        "can you tell me about ", "could you tell me about ", "tell me something about ", "tell me a bit about ",
        "i want to know about ", "i would like to know about ", "have you heard of ", "have you heard about ",
        "do you know ", "what is the meaning of ", "what does the word ", "what does ", "info about ", "information about ",
        "facts about ", "say something about ", "talk about ", "what can you tell me about ",
        "can you tell me who ", "can you tell me what ", "tell me who ", "tell me what ", "do you know anything about ",
        "what do you know of ", 0 };
    char q[64]; int m = 0;
    for (int i = 0; i < n && m < 63; i++) {                  /* lowercase letters, digits, single spaces */
        int c = (unsigned char)msg[i];
        if (c >= 'A' && c <= 'Z') c += 32;
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) c = ' ';
        if (c == ' ' && (m == 0 || q[m - 1] == ' ')) continue;
        q[m++] = (char)c;
    }
    while (m && q[m - 1] == ' ') m--;
    const char *s = q; int len = m, hit = 0;
    for (int again = 1; again;) {                            /* "do you know who" + "is" ... : leads can chain */
        again = 0;
        int best = 0;                                        /* kokoSD 2.4: the longest lead ("do you know anything about ") */
        for (int k = 0; LEAD[k]; k++) { int l = starts(s, len, LEAD[k]); if (l > best) best = l; }
        for (int k = 0; fact_wide && LEAD2[k]; k++) { int l = starts(s, len, LEAD2[k]); if (l > best) best = l; }
        if (best) { s += best; len -= best; hit = 1; again = 1; }
        int l = starts(s, len, "is ") + starts(s, len, "was ") + starts(s, len, "are ");
        if (hit && l) { s += l; len -= l; }
    }
    if (!hit || len < 2) return 0;
    if (len > 7 && starts(s + len - 7, 7, " please")) len -= 7;
    if (fact_wide && len > 5 && starts(s + len - 5, 5, " mean")) len -= 5;              /* kokoSD 2.4: "what does X mean" */
    if (len > 3 && starts(s + len - 3, 3, " is")) len -= 3;        /* "do you know what a dinosaur is" */
    else if (len > 4 && (starts(s + len - 4, 4, " are") || starts(s + len - 4, 4, " was"))) len -= 4;
    int l = starts(s, len, "the ") + starts(s, len, "a ") + starts(s, len, "an ");
    s += l; len -= l;
    if (len < 2) return 0;
    int r = subject(s, len, ids, max);
    if (!r && fact_wide) r = parts(s, len, ids, max);        /* kokoSD 2.4 */
    return r;
}
