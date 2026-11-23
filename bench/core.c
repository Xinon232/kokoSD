/* Candidate cartridge architecture with random weights: measures cycles only.
 *
 *   token -> embedding lookup
 *   L layers:  norm -> attention (ternary keys, top-8 of a T-slot ring) -> residual
 *              norm -> router picks KACT of E experts -> expert MLP -> residual
 *   norm -> class head -> members of the chosen class -> next token
 *
 * Every matrix is ternary and runs through the lookup kernel in tern.s.
 * Compiled for the GBA and natively; the two must agree (checksum).
 */
#include "data.h"

#ifdef __arm__
#define IWRAM_CODE __attribute__((section(".iwram.text")))
#define IWRAM_DATA __attribute__((section(".bss.iwram")))
#define EWRAM_DATA __attribute__((section(".bss.ewram")))
void tern_build81(int32_t *tbl, const int32_t *x, int ngroups);
void tern_build32(int32_t *tbl, const int32_t *x, int ngroups);
#define tern_build (NE == 81 ? tern_build81 : tern_build32)
void tern_rows(int32_t *out, const uint8_t *steps, const int32_t *tbl, const int32_t *mult, int nwords, int nrows);
void tern_scan(int32_t *out, const uint8_t *steps, const int32_t *tbl, int nwords, int nrows);
void axpy8(int32_t *acc, const int8_t *v, int32_t p, int n);
#else
static void axpy8(int32_t *acc, const int8_t *v, int32_t p, int n) { for (int i = 0; i < n; i++) acc[i] += p * v[i]; }
#define IWRAM_CODE
#define IWRAM_DATA
#define EWRAM_DATA
static void tern_build(int32_t *tbl, const int32_t *x, int ngroups) {
    static const int sg[3] = { 0, 1, -1 };
    for (int g = 0; g < ngroups; g++, x += 4)
        for (int i = 0; i < NE; i++) {
            if (NE == 81) {
                *tbl++ = sg[i % 3] * x[0] + sg[i / 3 % 3] * x[1] + sg[i / 9 % 3] * x[2] + sg[i / 27] * x[3];
            } else {                    /* input i/8 has weight 0; bits 0..2 negate the other three in order */
                int32_t v = 0;
                for (int j = 0, b = 0; j < 4; j++)
                    if (j != i / 8) { v += (i >> b & 1) ? -x[j] : x[j]; b++; }
                *tbl++ = v;
            }
        }
}
static int32_t rowsum(const uint8_t **s, const int32_t *tbl, int nwords) {
    const int32_t *p = tbl - 1;
    int32_t acc = 0;
    for (int i = 0; i < nwords * 4; i++) { p += *(*s)++; acc += *p; }
    return acc;
}
static void tern_rows(int32_t *out, const uint8_t *s, const int32_t *tbl, const int32_t *mult, int nwords, int nrows) {
    for (int r = 0; r < nrows; r++)
        out[r] = (int32_t)(((int64_t)rowsum(&s, tbl, nwords) * mult[r] + 0x8000) >> 16);
}
static void tern_scan(int32_t *out, const uint8_t *s, const int32_t *tbl, int nwords, int nrows) {
    for (int r = 0; r < nrows; r++) out[r] = rowsum(&s, tbl, nwords);
}
#endif

enum { C_EMBED, C_NORM, C_BUILD, C_PROJ, C_SCAN, C_MIX, C_WO, C_ROUTER, C_W1, C_W2, C_HEAD, C_PKM, C_N };
struct Dbg { uint32_t magic, done, tokens, checksum, cyc[C_N]; };
volatile struct Dbg dbg __attribute__((section(".bss.dbg")));

#ifdef __arm__
static uint32_t tlast IWRAM_DATA;
static inline uint32_t now(void) {
    volatile uint16_t *t = (volatile uint16_t *)0x04000100;
    uint16_t h = t[2], l = t[0];
    if (h != t[2]) { h = t[2]; l = t[0]; }
    return ((uint32_t)h << 16) | l;
}
#define TICK(k) do { uint32_t n_ = now(); dbg.cyc[k] += n_ - tlast; tlast = n_; } while (0)
#else
#define TICK(k)
#endif

#define HS (DK / NH)                 /* key dims per head */
#define VS (DV / NH)                 /* value dims per head */
#define TOPK 8
static int32_t tbl[(D > HD ? D : HD) / 4 * NE] IWRAM_DATA;
static int32_t x[D] IWRAM_DATA, xn[D] IWRAM_DATA, sc[T] IWRAM_DATA;
static int32_t proj[DK + DK + DV] EWRAM_DATA, att[DV] EWRAM_DATA, tmp[D] EWRAM_DATA;
static int32_t rt[E + 1] EWRAM_DATA, hid[KACT + 1][HD] EWRAM_DATA, logit[NCLS > NMEM ? NCLS : NMEM] EWRAM_DATA;
static uint8_t kc[L][NH][T][HS / 4] EWRAM_DATA;     /* ternary keys as pointer steps */
static int8_t vc[L][T][DV] EWRAM_DATA;
static uint32_t pos IWRAM_DATA;
#if NPK
static int32_t pkA[NHK] EWRAM_DATA, pkB[NHK] EWRAM_DATA, pkacc[D] IWRAM_DATA;
#endif

IWRAM_CODE static uint32_t isqrt64(uint64_t v) {
    uint64_t r = 0, b = 1ULL << 62;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return (uint32_t)r;
}
IWRAM_CODE static void norm(int32_t *out, const int32_t *in) {      /* out ~ Q10 of in / rms(in) */
    uint64_t ss = 0;
    for (int i = 0; i < D; i++) ss += (uint64_t)((int64_t)in[i] * in[i]);
    uint32_t s = isqrt64(ss / D + 1);
    uint32_t inv = (uint32_t)((1ULL << 32) / (s + 1));
    for (int i = 0; i < D; i++) {
        int32_t t = (int32_t)(((int64_t)in[i] * inv) >> 22);
        out[i] = (int32_t)(((int64_t)t * 4000) >> 12);                /* stand-in for the gain */
    }
}
static int sat8(int32_t v) { return v > 127 ? 127 : v < -127 ? -127 : v; }
IWRAM_CODE static int argmax(const int32_t *v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

IWRAM_CODE static void topk(const int32_t *v, int n, int *top) {      /* indices of the TOPK largest, sorted */
    int k = 0;
    for (int t = 0; t < n; t++) {
        int32_t s = v[t];
        if (k == TOPK && s <= v[top[TOPK - 1]]) continue;
        int j = k < TOPK ? k++ : TOPK - 1;
        while (j > 0 && v[top[j - 1]] < s) { top[j] = top[j - 1]; j--; }
        top[j] = t;
    }
}

void bench_reset(void) {
    for (int l = 0; l < L; l++)
        for (int h = 0; h < NH; h++)
            for (int t = 0; t < T; t++)
                for (int g = 0; g < HS / 4; g++) kc[l][h][t][g] = g ? NE : 1;   /* all-zero key */
    for (int l = 0; l < L; l++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < DV; d++) vc[l][t][d] = 0;
    pos = 0;
}

IWRAM_CODE int bench_forward(int tok) {
    const int8_t *e = EMB + (tok & 63) * D;
    for (int i = 0; i < D; i++) x[i] = e[i] << 5;
    int slot = pos % T;
    TICK(C_EMBED);
    for (int l = 0; l < L; l++) {
        /* ---- attention ---- */
        norm(xn, x); TICK(C_NORM);
        tern_build(tbl, xn, D / 4); TICK(C_BUILD);
        tern_rows(proj, S_qkv, tbl, M_qkv, D / 16, DK + DK + DV); TICK(C_PROJ);
        const int32_t *q = proj, *k = proj + DK, *v = proj + 2 * DK;
        for (int h = 0; h < NH; h++) {                               /* ternarize and store the key */
            int prev = 0;
            for (int g = 0; g < HS / 4; g++) {
                const int32_t *kk = k + h * HS + g * 4;
                int idx = 0;
                if (NE == 81) {
                    for (int j = 0, m = 1; j < 4; j++, m *= 3)
                        idx += m * (kk[j] > 300 ? 1 : kk[j] < -300 ? 2 : 0);
                } else {                /* zero the smallest of the four, keep the signs of the rest */
                    int z = 0;
                    for (int j = 1; j < 4; j++)
                        if ((kk[j] < 0 ? -kk[j] : kk[j]) < (kk[z] < 0 ? -kk[z] : kk[z])) z = j;
                    idx = z * 8;
                    for (int j = 0, b = 0; j < 4; j++)
                        if (j != z) { idx |= (kk[j] < 0) << b; b++; }
                }
                kc[l][h][slot][g] = (uint8_t)(g ? NE + idx - prev : idx + 1);
                prev = idx;
            }
        }
        for (int d = 0; d < DV; d++) vc[l][slot][d] = (int8_t)sat8(v[d] >> 4);
        tern_build(tbl, q, DK / 4); TICK(C_BUILD);
        for (int h = 0; h < NH; h++) {
            tern_scan(sc, &kc[l][h][0][0], tbl + h * (HS / 4) * NE, HS / 16, T); TICK(C_SCAN);
            int top[TOPK];
            topk(sc, T, top);
            uint32_t w[TOPK], sum = 0;
            for (int i = 0; i < TOPK; i++) {
                uint32_t d = (uint32_t)(sc[top[0]] - sc[top[i]]) >> 6;
                w[i] = d >= 1024 ? 0 : T_exp[d];
                sum += w[i];
            }
            uint32_t R = 0x80000000u / sum;
            int32_t acc[VS];
            for (int d = 0; d < VS; d++) acc[d] = 0;
            for (int i = 0; i < TOPK; i++) {
                int32_t p = (int32_t)((w[i] * R) >> 16);
                const int8_t *vv = &vc[l][top[i]][h * VS];
                for (int d = 0; d < VS; d++) acc[d] += p * vv[d];
            }
            for (int d = 0; d < VS; d++) att[h * VS + d] = acc[d] >> 11;
            TICK(C_MIX);
        }
        tern_build(tbl, att, DV / 4); TICK(C_BUILD);
        tern_rows(tmp, S_wo, tbl, M_wo, DV / 16, D);
        for (int i = 0; i < D; i++) x[i] += tmp[i];
        TICK(C_WO);
        /* ---- routed expert MLP ---- */
        norm(xn, x); TICK(C_NORM);
        tern_build(tbl, xn, D / 4); TICK(C_BUILD);
#if KACT
        tern_rows(rt, S_rt, tbl, M_rt, D / 16, E);
        int ex[KACT];
        for (int a = 0; a < KACT; a++) { ex[a] = argmax(rt, E); rt[ex[a]] = -0x7fffffff; }
        TICK(C_ROUTER);
#endif
#if NPK
        /* ---- product-key memory, in parallel with the MLP: it reuses the same input tables ----
         * Each head scores NHK half-keys against the first half of the stream and NHK against the
         * second, combines the 8 best of each into 64 candidates, and mixes the 8 best value rows
         * out of NHK*NHK slots. */
        for (int i = 0; i < D; i++) pkacc[i] = 0;
        for (int h = 0; h < NPK; h++) {
            tern_scan(pkA, S_pk + (h * 2) * NHK * (D / 8), tbl, D / 32, NHK);
            tern_scan(pkB, S_pk + (h * 2 + 1) * NHK * (D / 8), tbl + (D / 8) * NE, D / 32, NHK);
            int ta[TOPK], tb[TOPK], ca[TOPK], cb[TOPK];
            int32_t cs[TOPK];
            topk(pkA, NHK, ta);
            topk(pkB, NHK, tb);
            int n = 0;
            for (int i = 0; i < TOPK; i++)
                for (int j = 0; j < TOPK; j++) {
                    int32_t v = pkA[ta[i]] + pkB[tb[j]];
                    if (n == TOPK && v <= cs[TOPK - 1]) continue;
                    int q = n < TOPK ? n++ : TOPK - 1;
                    while (q > 0 && cs[q - 1] < v) { cs[q] = cs[q - 1]; ca[q] = ca[q - 1]; cb[q] = cb[q - 1]; q--; }
                    cs[q] = v; ca[q] = ta[i]; cb[q] = tb[j];
                }
            uint32_t w[TOPK], sum = 0;
            for (int i = 0; i < TOPK; i++) {
                uint32_t d = (uint32_t)(cs[0] - cs[i]) >> 6;
                w[i] = d >= 1024 ? 0 : T_exp[d];
                sum += w[i];
            }
            uint32_t R = 0x80000000u / sum;
            for (int i = 0; i < TOPK; i++) {
                int32_t pw = (int32_t)((w[i] * R) >> 16);
                if (!pw) continue;
                /* all slots share 64 random rows here; a real ROM indexes V_pk by slot */
                const int8_t *vv = V_pk + ((ca[i] * NHK + cb[i]) & 63) * D;
                axpy8(pkacc, vv, pw, D);
            }
        }
        TICK(C_PKM);
#endif
#if KACT
        for (int a = 0; a < KACT; a++) {
            /* all experts share one random matrix here; a real ROM indexes S_w1 by ex[a] */
            tern_rows(hid[a], S_w1 + (ex[a] & 0) * HD, tbl, M_w1, D / 16, HD);
            for (int i = 0; i < HD; i++) hid[a][i] = hid[a][i] > 0 ? hid[a][i] : 0;
        }
        TICK(C_W1);
        for (int a = 0; a < KACT; a++) {
            tern_build(tbl, hid[a], HD / 4); TICK(C_BUILD);
            tern_rows(tmp, S_w2, tbl, M_w2, HD / 16, D);
            for (int i = 0; i < D; i++) x[i] += tmp[i];
            TICK(C_W2);
        }
#endif
#if NPK
        for (int i = 0; i < D; i++) x[i] += pkacc[i] >> 14;
        TICK(C_PKM);
#endif
    }
    pos++;
    norm(xn, x); TICK(C_NORM);
    tern_build(tbl, xn, D / 4); TICK(C_BUILD);
    tern_rows(logit, S_cls, tbl, M_cls, D / 16, NCLS);
    int c = argmax(logit, NCLS);
    tern_rows(logit, S_mem, tbl, M_mem, D / 16, NMEM);   /* a real ROM indexes S_mem by class */
    int m = argmax(logit, NMEM);
    TICK(C_HEAD);
    uint32_t cs = dbg.checksum;
    for (int i = 0; i < D; i++) cs = cs * 31 + (uint32_t)x[i];
    dbg.checksum = cs * 31 + (uint32_t)(c * NMEM + m);
    return c * NMEM + m;
}

#ifdef __arm__
void *memset(void *d, int c, unsigned long n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }
int main(void) {
    *(volatile uint16_t *)0x04000204 = 0x4317;
    *(volatile uint16_t *)0x04000000 = 0x0080;          /* screen off */
    bench_reset();
    dbg.checksum = 0; dbg.tokens = 0; dbg.done = 0;
    for (int i = 0; i < C_N; i++) dbg.cyc[i] = 0;
    *(volatile uint16_t *)0x04000106 = 0x0084;          /* TM1 counts TM0 overflows */
    *(volatile uint16_t *)0x04000102 = 0x0080;          /* TM0: one tick per CPU cycle */
    int tok = 1;
    tlast = now();
    for (int i = 0; i < NTOK; i++) { tok = bench_forward(tok); dbg.tokens++; }
    dbg.magic = 0xBE7C4000; dbg.done = 1;
    for (;;) {}
}
#else
#include <stdio.h>
int main(void) {
    bench_reset();
    int tok = 1;
    for (int i = 0; i < NTOK; i++) tok = bench_forward(tok);
    printf("%u\n", dbg.checksum);
    return 0;
}
#endif
