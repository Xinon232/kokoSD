/* Native generation with the cartridge engine: args = blob, seed, inverse temperature (Q8), max new tokens,
 * stop token; prompt token ids on stdin; prints generated ids. */
#include <stdio.h>
#include <stdlib.h>
#include "koko.h"
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) { fprintf(stderr, "blob?\n"); return 1; }
    koko_blob = b;
    uint32_t rng = (uint32_t)atoi(argv[2]);
    int inv = atoi(argv[3]), maxn = atoi(argv[4]), stop = atoi(argv[5]), stop2 = atoi(argv[6]), tok, n = 0;
    koko_reset();
    while (scanf("%d", &tok) == 1) { koko_step(tok); n++; }
    if (argc > 7 && atoi(argv[7])) koko_reply_begin();      /* repeated-phrase rule */
    for (int i = 0; i < maxn; i++, n++) {
        tok = koko_pick(&rng, inv);
        if (tok == stop || tok == stop2) break;
        printf("%d ", tok);
        koko_step(tok);
    }
    printf("\n");
    return 0;
}
