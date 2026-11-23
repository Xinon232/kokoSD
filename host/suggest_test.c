/* Suggestions test: args blob, at_start, prefix; context token ids on stdin; prints up to 3 suggested ids. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "koko.h"
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) return 1;
    koko_blob = b;
    int tok; uint16_t out[3];
    koko_reset();
    while (scanf("%d", &tok) == 1) koko_step(tok);
    const char *pre = argc > 3 ? argv[3] : "";
    int n = koko_suggest(pre, (int)strlen(pre), atoi(argv[2]), out);
    for (int i = 0; i < n; i++) { const char *p; int l = koko_token_text(out[i], &p); printf("[%.*s] ", l, p); }
    printf("\n");
    return 0;
}
