/* Native build of the cartridge's inference code, for scoring and tests.
 *
 *   twin score            stdin: one line of token ids per sequence.
 *                         stdout (binary): for every position after the first,
 *                         VOCAB int32 logits (Q16) predicting the next token.
 *   twin gen MODE SEED N "prompt"
 *                         MODE g = greedy/no-repeat, s = sampled. Prints the
 *                         token ids, prompt included, one line.
 *   twin encode "text"    prints token ids
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/model.h"

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "score")) {
        static char line[1 << 16];
        while (fgets(line, sizeof line, stdin)) {
            llm_reset();
            char *p = line, *end;
            int first = 1;
            for (;;) {
                long t = strtol(p, &end, 10);
                if (end == p) break;
                p = end;
                char *q = p;
                strtol(q, &end, 10);
                int last = (end == q);
                if (!last) {
                    llm_forward((int)t, 1);
                    fwrite(llm_logits, sizeof(int32_t), VOCAB, stdout);
                }
                (void)first;
            }
        }
        return 0;
    }
    if (argc >= 6 && !strcmp(argv[1], "gen")) {
        static uint16_t hist[4096];
        uint32_t rng = (uint32_t)strtoul(argv[3], 0, 0);
        int n = tok_encode(argv[5], hist, 256), max = n + atoi(argv[4]);
        llm_reset();
        for (int i = 0; i + 1 < n; i++) llm_forward(hist[i], 0);
        while (n < max) {
            llm_forward(hist[n - 1], 1);
            int t = argv[2][0] == 'g' ? llm_pick_greedy(hist, n) : llm_pick_sample(&rng);
            if (t == 1) break;
            hist[n++] = (uint16_t)t;
        }
        for (int i = 0; i < n; i++) printf("%d ", hist[i]);
        printf("\n");
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "encode")) {
        static uint16_t ids[1024];
        int n = tok_encode(argv[2], ids, 1000);
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n");
        return 0;
    }
    fprintf(stderr, "usage: twin score | gen MODE SEED N PROMPT | encode TEXT\n");
    return 1;
}
