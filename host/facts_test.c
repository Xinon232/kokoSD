/* Fact table test: one message per line in; the stored sentence (or an empty line) out. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "koko.h"
int fact_reply(const char *msg, int n, uint16_t *ids, int max);
const uint8_t *koko_blob;
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    uint8_t *b = malloc(KOKO_BLOB_BYTES);
    if (!f || fread(b, 1, KOKO_BLOB_BYTES, f) != KOKO_BLOB_BYTES) return 1;
    koko_blob = b; (void)argc;
    char line[512]; uint16_t ids[64];
    while (fgets(line, sizeof line, stdin)) {
        int n = (int)strlen(line);
        if (n && line[n - 1] == '\n') n--;
        int k = fact_reply(line, n, ids, 64);
        for (int i = 0; i < k; i++) { const char *p; int l = koko_token_text(ids[i], &p); fwrite(p, 1, l, stdout); }
        printf("\n");
    }
    return 0;
}
