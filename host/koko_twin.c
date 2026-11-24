/* Native twin: read token ids from stdin, write the final state and all output scores per step (binary). */
#include <stdio.h>
#include <stdlib.h>
#include "koko.h"
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = fopen(argc > 1 ? argv[1] : "build/koko.bin", "rb");
    if (!f) { perror("blob"); return 1; }
    uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) { fprintf(stderr, "short blob\n"); return 1; }
    koko_blob = b;
    static uint16_t rows[K_V];
    static int32_t out[K_V];
    for (int i = 0; i < K_V; i++) rows[i] = (uint16_t)i;
    koko_reset();
    uint32_t rng = 12345;
    int tok;
    while (scanf("%d", &tok) == 1) {
        if (tok == -1) { koko_reset(); continue; }
        if (tok == -2) { koko_mark(); continue; }
        if (tok == -3) { koko_rewind(); continue; }
        koko_step(tok);
        koko_scores(rows, K_V, out);
        fwrite(koko_h, 4, K_D, stdout);
        fwrite(out, 4, K_V, stdout);
        const uint16_t *ids; const int32_t *sc;
        int32_t rec[3 + 2 * K_TOPN] = { 0 }, scored;
        int n = koko_topn(&ids, &sc, &scored);
        rec[0] = n; rec[1] = scored;
        for (int i = 0; i < n; i++) { rec[3 + i] = ids[i]; rec[3 + K_TOPN + i] = sc[i]; }
        rec[2] = koko_pick(&rng, 512);
        fwrite(rec, 4, 3 + 2 * K_TOPN, stdout);
    }
    return 0;
}
