/* Integer-only inference for stories260K. No floats, no libc.
 *
 * This one file is compiled twice: for the GBA (ARM code in fast RAM) and
 * natively as the "twin" that the tests compare the cartridge against.
 * Anything that changes an integer here changes both.
 *
 * Formats: the residual stream and matvec outputs are int32 Q16. A vector
 * about to be multiplied by a matrix is narrowed to int16 range (kept in
 * int32 slots) with the Q_* fraction bits from model_data.h.
 */
#include "model.h"

int32_t llm_logits[VOCAB] EWRAM_DATA;

/* attention cache, [layer][kv head][slot][dim]: each head's scan is sequential */
static int16_t kcache[NLAYER][NKVHEAD][WINDOW][HEADSZ] EWRAM_DATA;
static int16_t vcache[NLAYER][NKVHEAD][WINDOW][HEADSZ] EWRAM_DATA;

static uint32_t pos IWRAM_DATA;
static int32_t x[DIM] IWRAM_DATA;      /* residual stream, Q16 */
static int32_t vin[HID] IWRAM_DATA;    /* input of the next matvec */
static int32_t bq[DIM] IWRAM_DATA, bk[KVDIM] IWRAM_DATA, bv[KVDIM] IWRAM_DATA;
static int32_t h1[HID] IWRAM_DATA, h3[HID] IWRAM_DATA, tmp[DIM] IWRAM_DATA;
static int32_t sc[WINDOW] IWRAM_DATA;

#if Q_QK != 8 || HEADSZ != 8
#error "attention score scaling below assumes Q_QK == 8 and HEADSZ == 8"
#endif
/* score/sqrt(8) in 1/256 steps = raw * 362 >> 18; weights below exp(-16) are zero */
#define EXP_MUL 362
#define EXP_LIM 2965821u

static inline int32_t sat16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return v;
}
static inline int32_t rshr(int32_t v, int s) { return (v + (1 << (s - 1))) >> s; }

/* Weights are stored +128 (see tools/export.py). */
#ifdef LLM_ASM_KERNELS      /* src/kernels.s */
void matvec(int32_t *out, const int32_t *in, const uint8_t *w, const int32_t *mult, int n_in, int n_out);
int32_t att_scores(int32_t *sc, const int16_t *k, const int32_t *q, int n);
#else
static void matvec(int32_t *out, const int32_t *in, const uint8_t *w,
                   const int32_t *mult, int n_in, int n_out) {
    for (int r = 0; r < n_out; r++) {
        int32_t acc = 0;
        for (int i = 0; i < n_in; i++) acc += (w[i] - 128) * in[i];
        w += n_in;
        out[r] = (int32_t)(((int64_t)acc * mult[r] + (1 << (MSH - 1))) >> MSH);
    }
}
/* sc[t] = q . k[t]; returns the largest */
static int32_t att_scores(int32_t *sc, const int16_t *k, const int32_t *q, int n) {
    int32_t mx = -0x7fffffff - 1;
    for (int t = 0; t < n; t++, k += HEADSZ) {
        int32_t s = 0;
        for (int d = 0; d < HEADSZ; d++) s += q[d] * k[d];
        sc[t] = s;
        if (s > mx) mx = s;
    }
    return mx;
}
#endif

IWRAM_CODE static uint32_t isqrt64(uint64_t v) {
    uint64_t r = 0, b = 1ULL << 62;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; }
        else r >>= 1;
        b >>= 2;
    }
    return (uint32_t)r;
}

/* out = sat16(gain * x / rms(x)) with qout fraction bits */
IWRAM_CODE static void rmsnorm(int32_t *out, const int32_t *in, const int32_t *g, int qout) {
    uint64_t ss = 0;
    for (int i = 0; i < DIM; i++) ss += (uint64_t)((int64_t)in[i] * in[i]);
    ss = ss / DIM + 42950;                              /* mean square, Q32, + 1e-5 */
    uint32_t s = isqrt64(ss);                           /* rms, Q16 */
    uint32_t inv = (uint32_t)((1ULL << 39) / s);        /* 1/rms, Q23 */
    int sh = 16 + Q_G - qout;
    for (int i = 0; i < DIM; i++) {
        int32_t t = (int32_t)(((int64_t)in[i] * inv) >> 23);          /* Q16 */
        out[i] = sat16((int32_t)(((int64_t)t * g[i] + (1 << (sh - 1))) >> sh));
    }
}

IWRAM_CODE static int32_t isin(uint32_t ph) {            /* Q14 sine of ph/2^32 turns */
    int i = ph >> 20, f = (ph >> 12) & 0xFF;
    int a = T_sin[i], b = T_sin[i + 1];
    return a + (((b - a) * f) >> 8);
}

void llm_reset(void) { pos = 0; }

IWRAM_CODE void llm_forward(int tok, int want_logits) {
    const uint8_t *e = W_emb + tok * DIM;
    int32_t me = M_emb[tok];
    for (int i = 0; i < DIM; i++) x[i] = ((e[i] - 128) * me + 128) >> 8;

    int32_t cs[HEADSZ / 2], sn[HEADSZ / 2];
    for (int j = 0; j < HEADSZ / 2; j++) {
        uint32_t ph = pos * ROPE_STEP[j];
        sn[j] = isin(ph);
        cs[j] = isin(ph + 0x40000000u);
    }
    int slot = pos % WINDOW;
    int n = pos + 1 < WINDOW ? (int)pos + 1 : WINDOW;

    for (int l = 0; l < NLAYER; l++) {
        /* ---- attention ---- */
        rmsnorm(vin, x, G_att + l * DIM, Q_N1);
        matvec(bq, vin, W_wq + l * DIM * DIM, M_wq + l * DIM, DIM, DIM);
        matvec(bk, vin, W_wk + l * KVDIM * DIM, M_wk + l * KVDIM, DIM, KVDIM);
        matvec(bv, vin, W_wv + l * KVDIM * DIM, M_wv + l * KVDIM, DIM, KVDIM);
        /* RoPE: rotate each (even, odd) pair of q and k by this position's angle */
        for (int i = 0; i < DIM; i += 2) {
            int j = (i % HEADSZ) >> 1;
            int32_t a = sat16(rshr(bq[i], 16 - Q_QK)), b = sat16(rshr(bq[i + 1], 16 - Q_QK));
            bq[i] = (a * cs[j] - b * sn[j] + 8192) >> 14;
            bq[i + 1] = (a * sn[j] + b * cs[j] + 8192) >> 14;
        }
        for (int i = 0; i < KVDIM; i += 2) {
            int j = (i % HEADSZ) >> 1;
            int32_t a = sat16(rshr(bk[i], 16 - Q_QK)), b = sat16(rshr(bk[i + 1], 16 - Q_QK));
            int16_t *kd = &kcache[l][i / HEADSZ][slot][i % HEADSZ];
            int16_t *vd = &vcache[l][i / HEADSZ][slot][i % HEADSZ];
            kd[0] = (int16_t)sat16((a * cs[j] - b * sn[j] + 8192) >> 14);
            kd[1] = (int16_t)sat16((a * sn[j] + b * cs[j] + 8192) >> 14);
            vd[0] = (int16_t)sat16(rshr(bv[i], 16 - Q_V));
            vd[1] = (int16_t)sat16(rshr(bv[i + 1], 16 - Q_V));
        }
        for (int h = 0; h < NHEAD; h++) {
            const int16_t *K = &kcache[l][h / (NHEAD / NKVHEAD)][0][0];
            const int16_t *V = &vcache[l][h / (NHEAD / NKVHEAD)][0][0];
            int32_t mx = att_scores(sc, K, bq + h * HEADSZ, n);
            uint32_t sum = 0;
            for (int t = 0; t < n; t++) {
                uint32_t d = (uint32_t)(mx - sc[t]);
                uint32_t w = d >= EXP_LIM ? 0 : T_exp[(d * EXP_MUL) >> 18];
                sc[t] = (int32_t)w;
                sum += w;
            }
            uint32_t R = 0x80000000u / sum;             /* sum >= 65535: the max itself */
            int32_t acc[HEADSZ];
            for (int d = 0; d < HEADSZ; d++) acc[d] = 0;
            for (int t = 0; t < n; t++) {
                int32_t p = (int32_t)(((uint32_t)sc[t] * R) >> 16);    /* Q15 */
                if (!p) continue;
                const int16_t *v = V + t * HEADSZ;
                for (int d = 0; d < HEADSZ; d++) acc[d] += p * v[d];
            }
            for (int d = 0; d < HEADSZ; d++)
                vin[h * HEADSZ + d] = sat16(rshr(acc[d], 15 + Q_V - Q_ATT));
        }
        matvec(tmp, vin, W_wo + l * DIM * DIM, M_wo + l * DIM, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += tmp[i];

        /* ---- SwiGLU MLP ---- */
        rmsnorm(vin, x, G_ffn + l * DIM, Q_N2);
        matvec(h1, vin, W_w1 + l * HID * DIM, M_w1 + l * HID, DIM, HID);
        matvec(h3, vin, W_w3 + l * HID * DIM, M_w3 + l * HID, DIM, HID);
        for (int i = 0; i < HID; i++) {
            int32_t h = h1[i];
            int idx = (h >> 8) + 2048;
            int32_t sig;
            if (idx < 0) sig = 0;
            else if (idx >= 4096) sig = 32768;
            else {
                int a = T_sig[idx], b = T_sig[idx + 1];
                sig = a + (((b - a) * (h & 0xFF)) >> 8);
            }
            int32_t silu = (int32_t)(((int64_t)h * sig) >> 15);
            int32_t hb = (int32_t)(((int64_t)silu * h3[i]) >> 16);
            vin[i] = sat16(rshr(hb, 16 - Q_HB));
        }
        matvec(tmp, vin, W_w2 + l * DIM * HID, M_w2 + l * DIM, HID, DIM);
        for (int i = 0; i < DIM; i++) x[i] += tmp[i];
    }
    pos++;
    if (want_logits) {
        rmsnorm(vin, x, G_fin, Q_NF);
        matvec(llm_logits, vin, W_emb, M_cls, DIM, VOCAB);
    }
}

/* ---------------- decoding ---------------- */

static int repeats4(const uint16_t *hist, int n, int c) {
    if (n < 4) return 0;
    uint16_t a = hist[n - 3], b = hist[n - 2], d = hist[n - 1];
    for (int i = 0; i + 3 < n; i++)
        if (hist[i] == a && hist[i + 1] == b && hist[i + 2] == d && hist[i + 3] == c) return 1;
    return 0;
}

int llm_pick_greedy(const uint16_t *hist, int n) {
    uint16_t rej[16];
    int nrej = 0;
    for (;;) {
        int best = -1;
        for (int i = 0; i < VOCAB; i++) {
            if (best >= 0 && llm_logits[i] <= llm_logits[best]) continue;
            int skip = 0;
            for (int j = 0; j < nrej; j++) if (rej[j] == i) skip = 1;
            if (!skip) best = i;
        }
        if (nrej == 16 || !repeats4(hist, n, best)) return best;
        rej[nrej++] = (uint16_t)best;
    }
}

#define TOPK 8
int llm_pick_sample(uint32_t *rng) {
    int top[TOPK], k = 0;
    for (int i = 0; i < VOCAB; i++) {
        int32_t v = llm_logits[i];
        if (k == TOPK && v <= llm_logits[top[TOPK - 1]]) continue;
        int j = k < TOPK ? k++ : TOPK - 1;
        while (j > 0 && llm_logits[top[j - 1]] < v) { top[j] = top[j - 1]; j--; }
        top[j] = i;
    }
    /* exp((l - lmax) / 0.7): 1/0.7 * 256 = 366 table steps per unit */
    uint32_t w[TOPK], sum = 0;
    for (int i = 0; i < TOPK; i++) {
        uint32_t d = (uint32_t)(llm_logits[top[0]] - llm_logits[top[i]]);
        uint32_t idx = ((d >> 6) * 366) >> 10;
        w[i] = idx >= 4096 ? 0 : T_exp[idx];
        sum += w[i];
    }
    uint32_t r = *rng;
    r ^= r << 13; r ^= r >> 17; r ^= r << 5;
    *rng = r;
    uint32_t pick = (uint32_t)(((uint64_t)r * sum) >> 32);
    for (int i = 0; i < TOPK; i++) {
        if (pick < w[i]) return top[i];
        pick -= w[i];
    }
    return top[0];
}

/* ---------------- tokenizer ---------------- */

int tok_piece(int id, const uint8_t **p) {
    *p = TOK_bytes + TOK_off[id];
    return TOK_off[id + 1] - TOK_off[id];
}

static int find_piece(const uint8_t *s, int len) {
    int lo = 0, hi = NSORTED - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1, id = TOK_sorted[mid];
        const uint8_t *p;
        int pl = tok_piece(id, &p), c = 0;
        for (int i = 0; i < len && i < pl && !c; i++) c = (int)s[i] - (int)p[i];
        if (!c) c = len - pl;
        if (!c) return id;
        if (c < 0) hi = mid - 1; else lo = mid + 1;
    }
    return -1;
}

int tok_encode(const char *text, uint16_t *out, int max) {
    int n = 0;
    out[n++] = 1;
    if (!*text) return n;
    out[n++] = TOK_char[' '];
    for (const uint8_t *c = (const uint8_t *)text; *c && n < max; c++)
        out[n++] = *c < 128 ? TOK_char[*c] : (uint16_t)(*c + 3);
    for (;;) {
        int best = -1, bi = 0;
        for (int i = 1; i + 1 < n; i++) {
            if (out[i] < 259 || out[i + 1] < 259) continue;
            uint8_t buf[32];
            const uint8_t *a, *b;
            int la = tok_piece(out[i], &a), lb = tok_piece(out[i + 1], &b);
            for (int j = 0; j < la; j++) buf[j] = a[j];
            for (int j = 0; j < lb; j++) buf[la + j] = b[j];
            int id = find_piece(buf, la + lb);
            if (id >= 0 && (best < 0 || id < best)) { best = id; bi = i; }
        }
        if (best < 0) break;
        out[bi] = (uint16_t)best;
        for (int i = bi + 1; i + 1 < n; i++) out[i] = out[i + 1];
        n--;
    }
    return n;
}
