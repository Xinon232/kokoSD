#include <stdio.h>
#include <string.h>
int sum_reply(const char *s, int n, char *out);
int main(void) {
    char line[256], out[128];
    while (fgets(line, sizeof line, stdin)) {
        int n = (int)strlen(line);
        if (n && line[n - 1] == '\n') n--;
        int k = sum_reply(line, n, out);
        printf("%.*s\n", k, out);
    }
    return 0;
}
