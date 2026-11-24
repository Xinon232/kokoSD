/* Tokenizer test: one line of text in, token ids and decoded text out. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "koko.h"
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) return 1;
    koko_blob = b; (void)argc;
    static char line[4096]; static uint16_t out[4096];
    while (fgets(line, sizeof line, stdin)) {
        int n = (int)strlen(line);
        if (n && line[n - 1] == '\n') n--;
        int k = koko_encode(line, n, out, 4096);
        for (int i = 0; i < k; i++) printf("%d ", out[i]);
        printf("|");
        for (int i = 0; i < k; i++) { const char *p; int l = koko_token_text(out[i], &p); fwrite(p, 1, l, stdout); }
        printf("\n");
    }
    return 0;
}
