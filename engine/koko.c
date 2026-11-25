/* Koko inference: integers only. engine/ref_int.py is the specification; this file must reproduce it exactly. */
#include "koko.h"

#ifdef __arm__
#define IWRAM_CODE __attribute__((section(".iwram.text"), target("arm")))
#define IWRAM_DATA __attribute__((section(".bss.iwram")))
#define EWRAM_DATA __attribute__((section(".bss.ewram")))
#else
#define IWRAM_CODE
#define IWRAM_DATA
#define EWRAM_DATA
#endif

#define D K_D
#define T K_T
#define NH K_NH
#define HK (K_DK / K_NH)
#define HV (K_DV / K_NH)
#define NE 81
#define QB K_Q
#define RS(x, n) (((x) + ((int64_t)1 << ((n) - 1))) >> (n))

#define B8(o) ((const int8_t *)(koko_blob + (o)))
#define BU8(o) ((const uint8_t *)(koko_blob + (o)))
#define B16(o) ((const int16_t *)(koko_blob + (o)))
#define BU16(o) ((const uint16_t *)(koko_blob + (o)))
#define B32(o) ((const int32_t *)(koko_blob + (o)))
#ifdef __arm__        /* kokoSD 1.4.5: the cartridge's copy of the data block leaves out the fact table (tools/build.py), */
#define O_LAYERS_AT O_FACT_HASH   /* so the layer table follows the word tables directly */
#else
#define O_LAYERS_AT O_LAYERS
#endif
#define LAY(l, f) (((const uint32_t *)(koko_blob + O_LAYERS_AT))[(l) * LO_N + (f)])

int32_t koko_h[D] EWRAM_DATA;
void (*koko_yield)(void);                                  /* optional: called every few milliseconds of work (button polling) */
#define YIELD() do { if (koko_yield) koko_yield(); } while (0)
uint32_t koko_prof[KP_N] EWRAM_DATA;
#if defined(__arm__) && defined(KOKO_PROF)
static uint32_t tlast IWRAM_DATA;
static inline uint32_t now(void) {
    volatile uint16_t *t = (volatile uint16_t *)0x04000100;
    uint16_t h = t[2], l = t[0];
    if (h != t[2]) { h = t[2]; l = t[0]; }
    return ((uint32_t)h << 16) | l;
}
#define TICK(k) do { uint32_t n_ = now(); koko_prof[k] += n_ - tlast; tlast = n_; } while (0)
#else
#define TICK(k)
#endif
static int32_t tbl[(D / 4) * NE] IWRAM_DATA;
static int32_t x[D] IWRAM_DATA, xn[D] IWRAM_DATA;
static int32_t proj[K_DK > D ? K_DK : D] EWRAM_DATA, mixv[K_DV] EWRAM_DATA, hid[K_HD] EWRAM_DATA, sc[T] EWRAM_DATA;
static uint8_t kc[K_NATT][NH][T][HK / 4] EWRAM_DATA;        /* sign keys as pointer steps, one strip per head */
static int8_t vc[K_NATT][T][K_DV] EWRAM_DATA;               /* values: int8 mantissas */
static uint8_t vsh[K_NATT][T] EWRAM_DATA;                   /* and one shift per position */
static int32_t mst[K_L - K_NATT][D][6] EWRAM_DATA;           /* mixer state */
static uint32_t voted[K_V / 32] EWRAM_DATA;                 /* tokens that have a copy vote this step */
static uint16_t toks[T] EWRAM_DATA;
static uint32_t npos;                                      /* tokens fed so far; ring slot = position mod T */
static uint32_t nvalid;                                    /* how many of the most recent positions are usable */
#define SLOT(p) ((p) & (T - 1))
_Static_assert((T & (T - 1)) == 0, "the ring needs a power-of-two window");
static int a_top[K_TOPK], a_p[K_TOPK], a_n;

/* ---- kernels: ARM assembly in fast RAM on the GBA (engine/kernels.s), plain C natively ---- */
#ifdef __arm__
void tern_build81(int32_t *tbl, const int32_t *x, int ngroups);
void tern_scan(int32_t *out, const uint8_t *steps, const int32_t *tbl, int nwords, int nrows);
void tern_gather(int32_t *out, const uint16_t *ids, int n, const uint8_t *steps, const int32_t *tbl, const uint16_t *mult);
#define tern_build tern_build81
#define tern_rows(out, s, nbytes, nrows) tern_scan(out, s, tbl, (nbytes) / 4, nrows)
#define scan_keys(out, s, tb, nrows) tern_scan(out, s, tb, HK / 16, nrows)
static inline int32_t rowsum1(const uint8_t *s, int nbytes) { int32_t v; tern_scan(&v, s, tbl, nbytes / 4, 1); return v; }
#else
static void tern_build(int32_t *t, const int32_t *v, int ngroups) {
    static const int sg[3] = { 0, 1, -1 };
    for (int g = 0; g < ngroups; g++, v += 4)
        for (int i = 0; i < NE; i++)
            *t++ = sg[i % 3] * v[0] + sg[i / 3 % 3] * v[1] + sg[i / 9 % 3] * v[2] + sg[i / 27] * v[3];
}
static void scan(int32_t *out, const uint8_t *s, const int32_t *tb, int nbytes, int nrows) {
    for (int r = 0; r < nrows; r++) {
        const int32_t *p = tb - 1;
        int32_t acc = 0;
        for (int i = 0; i < nbytes; i++) { p += *s++; acc += *p; }
        out[r] = acc;
    }
}
#define tern_rows(out, s, nbytes, nrows) scan(out, s, tbl, nbytes, nrows)
#define scan_keys(out, s, tb, nrows) scan(out, s, tb, HK / 4, nrows)
static inline int32_t rowsum1(const uint8_t *s, int nbytes) { int32_t v; scan(&v, s, tbl, nbytes, 1); return v; }
#endif

static uint32_t isqrt64(uint64_t v) {
    uint64_t r = 0, b = (uint64_t)1 << 62;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return (uint32_t)r;
}
#ifdef __arm__
uint64_t sumsq_k(const int32_t *in, int n);
void norm_k(int32_t *out, const int32_t *in, const int32_t *g, int32_t inv, int n);
#else
static uint64_t sumsq_k(const int32_t *in, int n) {
    uint64_t ss = 0;
    for (int i = 0; i < n; i++) ss += (uint64_t)((int64_t)in[i] * in[i]);
    return ss;
}
#endif
static void norm(int32_t *out, const int32_t *in, const int32_t *g) {
    uint32_t r = isqrt64(sumsq_k(in, D) / D + 168);
    if (r < 16) r = 16;
    int64_t inv = (int64_t)(((uint64_t)1 << 40) / r);
#ifdef __arm__
    if (inv < ((int64_t)1 << 31)) { norm_k(out, in, g, (int32_t)inv, D); return; }
#endif
    for (int i = 0; i < D; i++) {                            /* general path: 64-bit reciprocal */
        int64_t t = RS((int64_t)in[i] * inv, 28);
        out[i] = (int32_t)RS(t * g[i], QB);
    }
}
static void rope(int32_t *v, int n, uint32_t pos) {        /* n values = NH heads of HK, rotated in pairs */
    const uint32_t *inc = (const uint32_t *)B32(O_ROPE_INC);
    const int16_t *sn = B16(O_SIN);
    for (int i = 0; i < n; i += 2) {
        uint32_t idx = ((pos * inc[(i % HK) / 2] + (1u << 19)) >> 20) & 4095;   /* phase wraps: any position works */
        int32_t si = sn[idx], co = sn[(idx + 1024) & 4095];
        int64_t a = v[i], b = v[i + 1];
        v[i] = (int32_t)RS(a * co - b * si, 14);
        v[i + 1] = (int32_t)RS(a * si + b * co, 14);
    }
}
static uint32_t mix32(uint32_t h) {
    h = (h ^ (h >> 15)) * 0x2C1B3C6Du;
    h = (h ^ (h >> 12)) * 0x297A2D39u;
    return h ^ (h >> 15);
}
static void inject(uint32_t off, uint32_t off_step, uint32_t rows, uint32_t salt, int tri, int tok, int prev, int prev2) {
    uint32_t h = (uint32_t)tok * 0x9E3779B1u + (uint32_t)prev * 0x85EBCA77u + salt * 0x27D4EB2Fu;
    if (tri) h += (uint32_t)prev2 * 0xC2B2AE3Du;
    uint32_t i = mix32(h) % rows;
    const uint8_t *p = BU8(off) + i * (D / 2);
    int32_t st = BU16(off_step)[i];
    for (int d = 0; d < D; d += 2) {
        int b = *p++;
        x[d] += ((int32_t)((b & 15) ^ 8) - 8) * st;         /* signed nibbles */
        x[d + 1] += ((int32_t)(b >> 4 ^ 8) - 8) * st;
    }
}

void koko_reset(void) {
    npos = 0; a_n = 0; nvalid = 0;
    for (unsigned i = 0; i < K_V / 32; i++) voted[i] = 0;
#if defined(__arm__) && defined(KOKO_PROF)
    tlast = now();
#endif
    for (unsigned l = 0; l < K_L - K_NATT; l++)
        for (int d = 0; d < D; d++)
            for (int i = 0; i < 6; i++) mst[l][d][i] = 0;
}

IWRAM_CODE static void attention(int l, int ai, uint32_t pos) {
    int t = (int)SLOT(pos), n = pos + 1 < T ? (int)pos + 1 : T;
    const int32_t *att = B32(LAY(l, LO_ATT));
    tern_build(tbl, xn, D / 4); TICK(KP_BUILD);
    /* key: per-row gain, rotate, keep the signs as pointer steps */
    tern_rows(proj, BU8(LAY(l, LO_K)), D / 4, K_DK);
    const int32_t *kg = B32(LAY(l, LO_KG));
    for (int i = 0; i < K_DK; i++) proj[i] = (int32_t)RS((int64_t)proj[i] * kg[i], QB);
    rope(proj, K_DK, pos);
    for (int h = 0; h < NH; h++) {
        int prev = 0;
        for (int g = 0; g < HK / 4; g++) {
            const int32_t *kk = proj + h * HK + g * 4;
            int idx = (kk[0] >= 0 ? 1 : 2) + 3 * (kk[1] >= 0 ? 1 : 2) + 9 * (kk[2] >= 0 ? 1 : 2) + 27 * (kk[3] >= 0 ? 1 : 2);
            kc[ai][h][t][g] = (uint8_t)(g ? NE + idx - prev : idx + 1);
            prev = idx;
        }
    }
    /* value: int8 mantissas and a shift */
    tern_rows(proj, BU8(LAY(l, LO_V)), D / 4, K_DV);
    int32_t mx = 0;
    for (int i = 0; i < K_DV; i++) {
        proj[i] = (int32_t)RS((int64_t)proj[i] * att[0], 16);
        int32_t a = proj[i] < 0 ? -proj[i] : proj[i];
        if (a > mx) mx = a;
    }
    int sh = 0;
    while ((mx >> sh) > 127) sh++;
    for (int i = 0; i < K_DV; i++) {
        int32_t m = sh ? (int32_t)RS((int64_t)proj[i], sh) : proj[i];
        vc[ai][t][i] = (int8_t)(m > 127 ? 127 : m < -127 ? -127 : m);
    }
    vsh[ai][t] = (uint8_t)sh;
    TICK(KP_KEYS);
    /* query */
    tern_rows(proj, BU8(LAY(l, LO_Q)), D / 4, K_DK);
    rope(proj, K_DK, pos);
    for (int h = 0; h < NH; h++) {
        int32_t *q = proj + h * HK;
        uint64_t ss = 0;
        for (int i = 0; i < HK; i++) ss += (uint64_t)((int64_t)q[i] * q[i]);
        uint32_t r = isqrt64(ss / HK + 168);
        if (r < 1) r = 1;
        int64_t m = ((int64_t)att[2 + h] << 20) / r;
        for (int i = 0; i < HK; i++) q[i] = (int32_t)RS((int64_t)q[i] * m, 24);
    }
    TICK(KP_PROJ);
    tern_build(tbl, proj, K_DK / 4); TICK(KP_BUILD);
    const int32_t *db = B32(LAY(l, LO_DBIAS));
    for (int h = 0; h < NH; h++) {
        scan_keys(sc, &kc[ai][h][0][0], tbl + h * (HK / 4) * NE, n);
        for (int j = 0; j < n; j++) sc[j] += db[h * T + SLOT(t - j)];
        TICK(KP_SCAN);
        int top[K_TOPK], k = 0;
        for (int j = 0; j < n; j++) {
            int32_t s = sc[j];
            if (SLOT(t - j) >= nvalid) continue;             /* slot lost to a rewind */
            if (k == K_TOPK && s <= sc[top[K_TOPK - 1]]) continue;
            int p = k < K_TOPK ? k++ : K_TOPK - 1;
            while (p > 0 && sc[top[p - 1]] < s) { top[p] = top[p - 1]; p--; }
            top[p] = j;
        }
        uint32_t w[K_TOPK], sum = 0;
        for (int i = 0; i < k; i++) {
            uint32_t d = (uint32_t)(sc[top[0]] - sc[top[i]]) >> (QB - K_EXP_BITS);
            w[i] = d < K_EXP_N ? BU16(O_EXP)[d] : 0;
            sum += w[i];
        }
        uint32_t R = 0x80000000u / sum;
        int64_t acc[HV];
        for (int d = 0; d < HV; d++) acc[d] = 0;
        for (int i = 0; i < k; i++) {
            int32_t p = (int32_t)(((uint64_t)w[i] * R) >> 16);
            const int8_t *vv = &vc[ai][top[i]][h * HV];
            int s2 = vsh[ai][top[i]];
            for (int d = 0; d < HV; d++) acc[d] += (int64_t)(p * vv[d]) << s2;
            if (h == 0) { a_top[i] = top[i]; a_p[i] = p; }
        }
        if (h == 0) a_n = k;
        for (int d = 0; d < HV; d++) mixv[h * HV + d] = (int32_t)RS(acc[d], 15);
        TICK(KP_MIX);
    }
    tern_build(tbl, mixv, K_DV / 4);
    tern_rows(proj, BU8(LAY(l, LO_O)), K_DV / 4, D);
    for (int i = 0; i < D; i++) x[i] += (int32_t)RS((int64_t)proj[i] * att[1], 16);
    TICK(KP_ATTOUT);
}

/* Mixer: weights on the previous three normalised inputs and on three running averages of them.
 * State per dimension: h[t-1], h[t-2], h[t-3], ema0, ema1, ema2. Coefficients per dimension: 3 taps, 3 average weights. */
#ifdef __arm__
void mixer_k(int32_t *x, const int32_t *xn, int32_t *state, const int32_t *coef, int n);
#else
static void mixer_k(int32_t *xx, const int32_t *xnn, int32_t *st, const int32_t *cf, int n) {
    static const int32_t k[3] = { 512, 205, 51 };
    for (int d = 0; d < n; d++, st += 6, cf += 6) {
        int64_t y = (int64_t)cf[0] * st[0] + (int64_t)cf[1] * st[1] + (int64_t)cf[2] * st[2];
        for (int i = 0; i < 3; i++) {
            st[3 + i] += (int32_t)RS((int64_t)(xnn[d] - st[3 + i]) * k[i], 10);
            y += (int64_t)cf[3 + i] * st[3 + i];
        }
        st[2] = st[1]; st[1] = st[0]; st[0] = xnn[d];
        xx[d] += (int32_t)RS(y, QB);
    }
}
#endif
static void mixer(int l, int mi) { mixer_k(x, xn, mst[mi][0], B32(LAY(l, LO_MIXC)), D); }

/* A step can be run in slices (one layer per call) so the game can poll buttons in between. */
static int s_tok, s_layer = -1, s_ai, s_mi;
void koko_step_begin(int tok) {
    uint32_t t = npos;
    toks[SLOT(t)] = (uint16_t)tok;
    nvalid = nvalid < T ? nvalid + 1 : T;
    const uint8_t *e = BU8(O_EMB) + (uint32_t)tok * (D / 4 * 3);          /* 6-bit values, 4 in 3 bytes */
    int32_t es = BU16(O_EMB_STEP)[tok];
    for (int i = 0; i < D; i += 4, e += 3) {
        uint32_t u = e[0] | (uint32_t)e[1] << 8 | (uint32_t)e[2] << 16;
        for (int k = 0; k < 4; k++, u >>= 6)
            x[i + k] = (int32_t)RS((int64_t)((int32_t)((u & 63) ^ 32) - 32) * es, 16 - QB);
    }
    s_tok = tok; s_layer = 0; s_ai = s_mi = 0;
}
int koko_step_part(void) {                                   /* returns 1 while there is more to do */
    uint32_t t = npos;
    int l = s_layer, tok = s_tok;
    if (l < 0) return 0;
    if (l == 0) {
        int prev = t >= 1 ? toks[SLOT(t - 1)] : 0, prev2 = t >= 2 ? toks[SLOT(t - 2)] : 0;
        inject(O_INJ0, O_INJ0_STEP, K_INJ0_ROWS, K_INJ0_SALT, K_INJ0_TRI, tok, prev, prev2);
        inject(O_INJ1, O_INJ1_STEP, K_INJ1_ROWS, K_INJ1_SALT, K_INJ1_TRI, tok, prev, prev2);
    }
    TICK(KP_EMBED);
    norm(xn, x, B32(LAY(l, LO_N1))); TICK(KP_NORM);
    if (K_ATT_MASK >> l & 1) attention(l, s_ai++, t); else { mixer(l, s_mi++); TICK(KP_MIXER); }
    YIELD();
    norm(xn, x, B32(LAY(l, LO_N2))); TICK(KP_NORM);
    int ex = BU8(LAY(l, LO_ROUTE))[tok];
    tern_build(tbl, xn, D / 4); TICK(KP_BUILD);
    tern_rows(hid, BU8(LAY(l, LO_W1)) + (uint32_t)ex * K_HD * (D / 4), D / 4, K_HD);
    int32_t m1 = B32(LAY(l, LO_W1M))[ex], m2 = B32(LAY(l, LO_W2M))[ex];
    for (int i = 0; i < K_HD; i++) {
        int64_t p = RS((int64_t)hid[i] * m1, 16);
        hid[i] = p > 0 ? (int32_t)RS(p * p, QB) : 0;
    }
    TICK(KP_W1);
    tern_build(tbl, hid, K_HD / 4); TICK(KP_BUILD);
    tern_rows(proj, BU8(LAY(l, LO_W2)) + (uint32_t)ex * D * (K_HD / 4), K_HD / 4, D);
    for (int i = 0; i < D; i++) x[i] += (int32_t)RS((int64_t)proj[i] * m2, 16);
    TICK(KP_W2);
    if (++s_layer < K_L) return 1;
    norm(koko_h, x, B32(O_NF));
    TICK(KP_NORM);
    tern_build(tbl, koko_h, D / 4); TICK(KP_BUILD);          /* left ready for koko_scores */
    npos = t + 1;
    s_layer = -1;
    return 0;
}
void koko_step(int tok) {
    koko_step_begin(tok);
    while (koko_step_part()) {}
}

/* Mark / rewind: take back the tokens fed since the mark (the user edited their message). Ring slots written
 * since then are lost, so the usable window shrinks by that many and grows again as tokens arrive. */
#ifdef __arm__
#define MSNAP ((int32_t *)0x0600A000)                        /* spare video memory (the unused second bitmap page): 24 KB */
#else
static int32_t MSNAP[(K_L - K_NATT) * D * 6];
#endif
static uint32_t m_npos, m_nvalid;
static int m_top[K_TOPK], m_p[K_TOPK], m_n;
void koko_mark(void) {
    const int32_t *s = &mst[0][0][0];
    for (unsigned i = 0; i < (K_L - K_NATT) * D * 6; i++) MSNAP[i] = s[i];
    m_npos = npos; m_nvalid = nvalid; m_n = a_n;
    for (int i = 0; i < K_TOPK; i++) { m_top[i] = a_top[i]; m_p[i] = a_p[i]; }
}
void koko_rewind(void) {
    int32_t *s = &mst[0][0][0];
    uint32_t k = npos - m_npos;
    for (unsigned i = 0; i < (K_L - K_NATT) * D * 6; i++) s[i] = MSNAP[i];
    npos = m_npos; a_n = m_n; s_layer = -1;
    nvalid = k >= T ? 0 : (m_nvalid < T - k ? m_nvalid : T - k);
    for (int i = 0; i < K_TOPK; i++) { a_top[i] = m_top[i]; a_p[i] = m_p[i]; }
}

static void mark_votes(void) {
    int t = (int)SLOT(npos - 1);
    for (int a = 0; a < a_n; a++)
        if (a_top[a] != t && a_p[a]) { unsigned id = toks[SLOT(a_top[a] + 1)]; voted[id >> 5] |= 1u << (id & 31); }
}
static void clear_votes(void) {
    int t = (int)SLOT(npos - 1);
    for (int a = 0; a < a_n; a++)
        if (a_top[a] != t) { unsigned id = toks[SLOT(a_top[a] + 1)]; voted[id >> 5] = 0; }
}
void koko_scores(const uint16_t *rows, int n, int32_t *out) {
    int t = (int)SLOT(npos - 1);
    if (n <= 0) return;                                      /* the assembly kernel counts down from n */
#ifdef __arm__
    tern_gather(out, rows, n, BU8(O_HEAD), tbl, BU16(O_HEAD_M));
#else
    for (int i = 0; i < n; i++)
        out[i] = (int32_t)RS((int64_t)rowsum1(BU8(O_HEAD) + (uint32_t)rows[i] * (D / 4), D / 4) * BU16(O_HEAD_M)[rows[i]], 16);
#endif
    mark_votes();
    for (int i = 0; i < n; i++) {
        unsigned id = rows[i];
        if (!(voted[id >> 5] >> (id & 31) & 1)) continue;
        for (int a = 0; a < a_n; a++)
            if (a_top[a] != t && a_p[a] && toks[SLOT(a_top[a] + 1)] == id) out[i] += (int32_t)RS((int64_t)K_COPY_GAIN * a_p[a], 15);
    }
    clear_votes();
    TICK(KP_HEAD);
}

/* ---- fast output step: score only the rows worth scoring, keep the best K_TOPN ---- */
#define MPROBE 16
static uint32_t seen[K_V / 32] EWRAM_DATA;
static uint16_t cbuf[64] EWRAM_DATA;
static int32_t cscore[64] EWRAM_DATA, rsc[K_NGROUP] EWRAM_DATA;
static uint16_t top_id[K_TOPN];
static int32_t top_sc[K_TOPN];
static int top_n, cn, n_scored, top_cap = K_TOPN;
/* Optional filter (typing suggestions): only whole-word tokens that extend the typed prefix are scored. */
static const uint32_t *flt_mask;
static const char *flt_pre;
static int flt_len, flt_skip;
static int lower(int ch) { return ch >= 'A' && ch <= 'Z' ? ch + 32 : ch; }
static int flt_ok(unsigned id) {
    const uint32_t *off = (const uint32_t *)B32(O_TOK_OFF);
    const char *p = (const char *)(koko_blob + O_TOK_TEXT + off[id]) + flt_skip;
    int len = (int)(off[id + 1] - off[id]) - flt_skip;
    if (len <= flt_len) return 0;
    for (int i = 0; i < flt_len; i++) if (lower(p[i]) != lower(flt_pre[i])) return 0;
    return 1;
}

IWRAM_CODE static void flush(void) {
    YIELD();
    koko_scores(cbuf, cn, cscore);
    for (int j = 0; j < cn; j++) {
        int32_t s = cscore[j];
        if (top_n == top_cap && s <= top_sc[top_cap - 1]) continue;
        int p = top_n < top_cap ? top_n++ : top_cap - 1;
        while (p > 0 && top_sc[p - 1] < s) { top_sc[p] = top_sc[p - 1]; top_id[p] = top_id[p - 1]; p--; }
        top_sc[p] = s; top_id[p] = cbuf[j];
    }
    n_scored += cn; cn = 0;
}
static inline void add(unsigned id) {
    if (seen[id >> 5] >> (id & 31) & 1) return;
    seen[id >> 5] |= 1u << (id & 31);
    if (flt_mask && !((flt_mask[id >> 5] >> (id & 31) & 1) && flt_ok(id))) return;
    cbuf[cn++] = (uint16_t)id;
    if (cn == 64) flush();
}
IWRAM_CODE int koko_topn(const uint16_t **ids, const int32_t **scores, int *scored) {
    for (unsigned i = 0; i < K_V / 32; i++) seen[i] = 0;
    top_n = cn = n_scored = 0;
    const uint16_t *sl = BU16(O_SHORT), *su = BU16(O_SUCC) + (uint32_t)toks[SLOT(npos - 1)] * K_KSUCC;
    int nw = npos < T ? (int)npos : T;
    for (int i = 0; i < K_SHORT; i++) add(sl[i]);
    for (int i = 0; i < K_KSUCC; i++) add(su[i]);
    for (int i = 0; i < nw; i++)
        if (SLOT(npos - 1 - (uint32_t)i) < nvalid) add(toks[i]);
    flush();                                                 /* the table still holds koko_h */
    tern_rows(rsc, BU8(O_ROUTER), D / 4, K_NGROUP);
    const int32_t *rm = B32(O_ROUTER_M);
    for (int g = 0; g < K_NGROUP; g++) rsc[g] = (int32_t)RS((int64_t)rsc[g] * rm[g], 16);
    const uint16_t *gs = BU16(O_GSTART), *gm = BU16(O_GMEM);
    for (int m = 0; m < MPROBE; m++) {
        int b = 0;
        for (int g = 1; g < K_NGROUP; g++) if (rsc[g] > rsc[b]) b = g;
        rsc[b] = (int32_t)0x80000000;
        for (unsigned i = gs[b]; i < gs[b + 1]; i++) add(gm[i]);
    }
    flush();
    *ids = top_id; *scores = top_sc;
    if (scored) *scored = n_scored;
    return top_n;
}
/* Repeated-phrase rule: once koko_reply_begin() has been called, a token that would repeat a 4-token phrase
 * already written in the current reply is not sampled (unless that would leave nothing). */
#define NO_REPLY 0xFFFFFFFFu
static uint32_t reply_at = NO_REPLY;
void koko_reply_begin(void) { reply_at = npos; }
void koko_reply_end(void) { reply_at = NO_REPLY; }
int koko_pick(uint32_t *rng, int inv_temp_q8) {
    const uint16_t *ids; const int32_t *sc;
    int n = koko_topn(&ids, &sc, 0);
    uint16_t ban[16]; int nb = 0;
    if (reply_at != NO_REPLY && npos - reply_at >= 3) {
        unsigned a = toks[SLOT(npos - 3)], b = toks[SLOT(npos - 2)], c = toks[SLOT(npos - 1)];
        for (uint32_t j = reply_at + 3; j < npos && nb < 16; j++)
            if (toks[SLOT(j - 3)] == a && toks[SLOT(j - 2)] == b && toks[SLOT(j - 1)] == c) ban[nb++] = toks[SLOT(j)];
    }
    uint32_t w[K_TOPN], sum = 0, sumb = 0;
    uint8_t isb[K_TOPN];
    for (int i = 0; i < n; i++) {
        uint32_t d = (uint32_t)(((int64_t)(sc[0] - sc[i]) * inv_temp_q8) >> (8 + QB - K_EXP_BITS));
        w[i] = d < K_EXP_N ? BU16(O_EXP)[d] : 0;
        isb[i] = 0;
        for (int k = 0; k < nb; k++) if (ban[k] == ids[i]) isb[i] = 1;
        sum += w[i];
        if (!isb[i]) sumb += w[i];
    }
    uint32_t r = *rng;
    r ^= r << 13; r ^= r >> 17; r ^= r << 5;
    *rng = r;
    if (nb && !sumb)                                         /* every weighted candidate is banned: best allowed one */
        for (int i = 0; i < n; i++) if (!isb[i]) return ids[i];
    if (sumb) { sum = sumb; for (int i = 0; i < n; i++) if (isb[i]) w[i] = 0; }
    r %= sum;
    for (int i = 0; i < n; i++) { if (r < w[i]) return ids[i]; r -= w[i]; }
    return ids[0];
}

/* Typing suggestions: up to 3 whole words the model expects next that start with the typed prefix
 * (case ignored). at_start: the message is empty so far (first words have no leading space). */
int koko_suggest(const char *prefix, int plen, int at_start, uint16_t *out) {
    const uint16_t *ids; const int32_t *sc;
    flt_mask = (const uint32_t *)B32(at_start ? O_WORD_START : O_WORD_SP);
    flt_pre = prefix; flt_len = plen; flt_skip = at_start ? 0 : 1; top_cap = 3;
    int n = koko_topn(&ids, &sc, 0);
    flt_mask = 0; top_cap = K_TOPN;
    for (int i = 0; i < n; i++) out[i] = ids[i];
    return n;
}

/* kokoSD 1.2: a saved chat. The state as it was at the mark (the start of the message being typed: the
 * draft's tokens are left out, exactly as koko_rewind would), written or read in pieces through io (1 = ok).
 * Only the ring slots in use are stored. The next steps rebuild everything else (tbl, koko_h). */
int koko_state_io(int (*io)(void *p, unsigned n, void *ctx), void *ctx, int writing) {
    uint32_t hdr[4 + 2 * K_TOPK];
    if (writing) {
        uint32_t k = npos - m_npos;
        hdr[0] = m_npos; hdr[1] = k >= T ? 0 : (m_nvalid < T - k ? m_nvalid : T - k); hdr[2] = (uint32_t)m_n; hdr[3] = 0;
        for (int i = 0; i < K_TOPK; i++) { hdr[4 + i] = (uint32_t)m_top[i]; hdr[4 + K_TOPK + i] = (uint32_t)m_p[i]; }
    }
    if (!io(hdr, sizeof hdr, ctx)) return 0;
    if (!writing) {
        if (hdr[1] > T || hdr[2] > K_TOPK) return 0;
        npos = hdr[0]; nvalid = hdr[1]; a_n = (int)hdr[2]; s_layer = -1; reply_at = NO_REPLY;
        for (int i = 0; i < K_TOPK; i++) { a_top[i] = (int)hdr[4 + i]; a_p[i] = (int)hdr[4 + K_TOPK + i]; }
        for (unsigned i = 0; i < K_V / 32; i++) voted[i] = 0;
    }
    unsigned used = hdr[0] < T ? hdr[0] : T;                    /* slots 0..used-1 hold the window (ring) */
    if (!io(toks, used * sizeof toks[0], ctx)) return 0;
    for (int a = 0; a < K_NATT; a++) {
        if (!io(vsh[a], used, ctx) || !io(vc[a], used * K_DV, ctx)) return 0;
        for (int h = 0; h < NH; h++) if (!io(kc[a][h], used * (HK / 4), ctx)) return 0;
    }
#ifdef __arm__
    if (writing) {                                             /* the mixer state at the mark is in video memory */
        static int32_t row[256] EWRAM_DATA;
        for (unsigned at = 0; at < (K_L - K_NATT) * D * 6; at += 256) {
            for (int i = 0; i < 256; i++) row[i] = MSNAP[at + i];
            if (!io(row, sizeof row, ctx)) return 0;
        }
        return 1;
    }
#else
    if (writing) return io(MSNAP, sizeof mst, ctx);
#endif
    return io(mst, sizeof mst, ctx);
}
