/* kokoSD 2.4: how the word memory changes Koko's next-token prediction on held-out Wikipedia text.
 * args: blob, facts.dat (with the word memory), held-out tokens (uint16, 0xFFFF between paragraphs), tokens to
 * score, mixes (/256)... Prints per mix: cross-entropy (bits per token, 1/4096 floor outside the top 40) and top-1
 * accuracy. GPL-3.0-or-later. */
#include <stdlib.h>
#include <math.h>
#include "koko.h"
#include "words_file.h"
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) return 1;
    koko_blob = b;
    if (!words_open(argv[2])) { fprintf(stderr, "no word memory in %s\n", argv[2]); return 1; }
    FILE *hf = fopen(argv[3], "rb"); long want = atol(argv[4]);
    static uint16_t t[1 << 20]; size_t n = fread(t, 2, 1 << 20, hf);
    int nm = argc - 5; double ce[16] = {0}; long hit[16] = {0}, cnt = 0, found = 0;
    for (int m = 0; m < nm; m++) {
        koko_words = words_lookup; koko_words_mix = atoi(argv[5 + m]); cnt = 0; found = 0;
        koko_reset(); int fed = 0;
        for (size_t i = 0; i < n && cnt < want; i++) {
            if (t[i] == 0xFFFF) { koko_reset(); fed = 0; continue; }
            if (fed >= 3 && fed < 200) {                 /* score this token given the paragraph so far */
                const uint16_t *ids; uint32_t w[64]; int k = koko_dist(1024 /* T 0.25 */, &ids, w);
                (void)k;
                k = koko_dist(256 /* T 1 */, &ids, w);
                uint64_t sum = 0; int best = 0; for (int j = 0; j < k; j++) { sum += w[j]; if (w[j] > w[best]) best = j; }
                double p = 1.0 / 4096;
                for (int j = 0; j < k; j++) if (ids[j] == t[i]) p = (double)w[j] / sum > p ? (double)w[j] / sum : p;
                ce[m] += -log2(p); if (ids[best] == t[i]) hit[m]++;
                cnt++;
                uint16_t wid[4]; uint8_t sh[4];
                if (m == 0) { uint16_t c3[3] = { fed >= 3 ? t[i - 3] : 0xFFFF, t[i - 2], t[i - 1] }; if (words_lookup(c3, wid, sh)) found++; }
            }
            if (fed < 200) koko_step(t[i]);
            fed++;
        }
        printf("mix %3d/256: %.3f bits per token, top-1 %.1f %%  (%ld tokens", koko_words_mix, ce[m] / cnt, 100.0 * hit[m] / cnt, cnt);
        if (m == 0) printf(", a table entry for %.0f %% of them", 100.0 * found / cnt);
        printf(")\n"); fflush(stdout);
    }
    return 0;
}
