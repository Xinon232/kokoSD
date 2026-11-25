/* kokoSD 2.4: a text corpus to Koko's token ids, for tools/words_sd.py. One paragraph per input line; out: the
 * ids as little-endian uint16 with 0xFFFF after each paragraph. GPL-3.0-or-later. */
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
    static char line[1 << 16]; static uint16_t out[1 << 16];
    const uint16_t sep = 0xFFFF;
    while (fgets(line, sizeof line, stdin)) {
        int n = (int)strlen(line);
        if (n && line[n - 1] == '\n') n--;
        int k = koko_encode(line, n, out, 1 << 16);
        fwrite(out, 2, (size_t)k, stdout); fwrite(&sep, 2, 1, stdout);
    }
    return 0;
}
