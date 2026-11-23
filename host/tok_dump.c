/* Every token's text, one per line as hex (so tests can decode without the
 * training tokenizer file): kokoSD, GPL-3.0-or-later. */
#include <stdio.h>
#include <stdlib.h>
#include "koko.h"
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = argc > 1 ? fopen(argv[1], "rb") : 0;
    uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) return 1;
    koko_blob = b;
    for (int id = 0; id < K_V; id++) {
        const char *p; int l = koko_token_text(id, &p);
        for (int i = 0; i < l; i++) printf("%02x", (unsigned char)p[i]);
        printf("\n");
    }
    return 0;
}
