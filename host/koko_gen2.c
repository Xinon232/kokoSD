/* kokoSD 2.4: host/koko_gen.c with the word memory: args = blob, seed, inverse temperature (Q8), max new tokens,
 * stop, stop2, repeated-phrase rule, facts.dat (with word tables), mix (/256); prompt ids on stdin. */
#include <stdio.h>
#include <stdlib.h>
#include "koko.h"
#include "words_file.h"
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) { fprintf(stderr, "blob?\n"); return 1; }
    koko_blob = b;
    if (argc < 10 || !words_open(argv[8])) { fprintf(stderr, "words?\n"); return 1; }
    koko_words = words_lookup; koko_words_mix = atoi(argv[9]);
    uint32_t rng = (uint32_t)atoi(argv[2]);
    int inv = atoi(argv[3]), maxn = atoi(argv[4]), stop = atoi(argv[5]), stop2 = atoi(argv[6]), tok;
    koko_reset();
    while (scanf("%d", &tok) == 1) koko_step(tok);
    if (atoi(argv[7])) koko_reply_begin();
    for (int i = 0; i < maxn; i++) {
        tok = koko_pick(&rng, inv);
        if (tok == stop || tok == stop2) break;
        printf("%d ", tok);
        koko_step(tok);
    }
    printf("\n");
    return 0;
}
