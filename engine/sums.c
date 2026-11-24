/* Sums are computed by the game, not by the model (which gets them wrong about four times in five).
 * sum_reply() looks for "<number> <operator> <number>" in a message and writes Koko's answer. */
#include <stdint.h>

static int lc(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
static int word(const char *s, int n, const char *w) {       /* does s start with w (case ignored)? returns its length */
    int i = 0;
    for (; w[i]; i++) if (i >= n || lc(s[i]) != w[i]) return 0;
    return i;
}
static int number(const char *s, int n, int64_t *v) {        /* up to 9 digits; returns digits read */
    int i = 0; int64_t x = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9' && i < 9) x = x * 10 + (s[i++] - '0');
    if (i < n && s[i] >= '0' && s[i] <= '9') return 0;        /* too long */
    *v = x;
    return i;
}
static char *put_s(char *o, const char *s) { while (*s) *o++ = *s++; return o; }
static char *put_n(char *o, int64_t v) {
    char b[24]; int k = 0;
    uint64_t u = v < 0 ? (uint64_t)(-v) : (uint64_t)v;
    if (v < 0) *o++ = '-';
    do { b[k++] = (char)('0' + (int)(u % 10)); u /= 10; } while (u);
    while (k) *o++ = b[--k];
    return o;
}

/* Returns the length of the answer written to out (at most 80 characters), or 0 if the message has no sum. */
int sum_reply(const char *s, int n, char *out) {
    static const struct { const char *w; char op; const char *say; } OPS[] = {
        { "+", '+', " + " }, { "plus", '+', " plus " }, { "-", '-', " - " }, { "minus", '-', " minus " },
        { "take away", '-', " take away " }, { "x", '*', " x " }, { "*", '*', " x " }, { "times", '*', " times " },
        { "multiplied by", '*', " multiplied by " }, { "/", '/', " / " }, { "divided by", '/', " divided by " },
    };
    for (int i = 0; i < n; i++) {
        int64_t a, b;
        if (i && ((s[i - 1] >= '0' && s[i - 1] <= '9') || s[i - 1] == '.')) continue;   /* not the middle of a number */
        int la = number(s + i, n - i, &a);
        if (!la) continue;
        int j = i + la;
        while (j < n && s[j] == ' ') j++;
        for (unsigned k = 0; k < sizeof OPS / sizeof OPS[0]; k++) {
            int lw = word(s + j, n - j, OPS[k].w);
            if (!lw) continue;
            int letters = OPS[k].w[0] >= 'a';
            if (letters && OPS[k].w[1] && j == i + la) continue;   /* "3x4" is fine, "3times4" is not a sum */
            if (OPS[k].w[0] == '-' && j == i + la && i) continue;  /* "we won 3-2" is a score; "3-2" alone is a sum */
            int q = j + lw;
            if (letters && OPS[k].w[1] && q < n && s[q] != ' ') continue;   /* a word operator must end at a space */
            while (q < n && s[q] == ' ') q++;
            int lb = number(s + q, n - q, &b);
            if (!lb) continue;
            if (q + lb < n && (s[q + lb] == '.' || lc(s[q + lb]) >= 'a') && !(s[q + lb] == '.' && (q + lb + 1 >= n || s[q + lb + 1] == ' '))) continue;
            char *o = out;
            int symbol = !letters || !OPS[k].w[1];
            if (OPS[k].op == '/' && b == 0) { o = put_s(o, "Nobody can divide by zero, not even a husky!"); return (int)(o - out); }
            int64_t v = OPS[k].op == '+' ? a + b : OPS[k].op == '-' ? a - b : OPS[k].op == '*' ? a * b : a / b;
            o = put_n(o, a); o = put_s(o, OPS[k].say); o = put_n(o, b);
            o = put_s(o, symbol ? " = " : " is ");
            o = put_n(o, v);
            if (OPS[k].op == '/' && a % b) { o = put_s(o, " remainder "); o = put_n(o, a % b); }
            *o++ = '.';
            return (int)(o - out);
        }
    }
    return 0;
}
