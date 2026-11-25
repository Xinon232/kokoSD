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
        for (int k = 0; LEAD[k]; k++) {
            int l = starts(s, len, LEAD[k]);
            if (l) { s += l; len -= l; hit = 1; again = 1; break; }
        }
        int l = starts(s, len, "is ") + starts(s, len, "was ") + starts(s, len, "are ");
        if (hit && l) { s += l; len -= l; }
    }
    if (!hit || len < 2) return 0;
    if (len > 7 && starts(s + len - 7, 7, " please")) len -= 7;
    if (len > 3 && starts(s + len - 3, 3, " is")) len -= 3;        /* "do you know what a dinosaur is" */
    else if (len > 4 && (starts(s + len - 4, 4, " are") || starts(s + len - 4, 4, " was"))) len -= 4;
    int l = starts(s, len, "the ") + starts(s, len, "a ") + starts(s, len, "an ");
    s += l; len -= l;
    if (len < 2) return 0;
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
