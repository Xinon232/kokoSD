/* Headless mGBA driver for the tests.
 *
 *   runner rom.gba STEP...
 *     f:N                 run N frames
 *     k:MASK              press keys (GBA key bits) for 3 frames, then release for 3
 *     until:ADDR:VAL:MAX  run until the 32-bit word at ADDR equals VAL (at most MAX frames)
 *     shot:FILE.ppm       write the screen
 *     dump:ADDR:LEN:FILE  write LEN bytes of memory
 *     frames              print how many frames have run
 */
#include <mgba/core/core.h>
#include <mgba/core/log.h>
/* kokoSD: built with CART_MODEL=1/2/3 (tools/build_runner.py), the cartridge's
 * SD card is gbaSD's model of a SuperCard SD / EZ-Flash Omega / EverDrive GBA
 * with the card image SD_IMAGE (environment). */
#ifdef CART_MODEL
#include <mgba/core/timing.h>
#include <mgba/internal/gba/gba.h>
#endif
#if CART_MODEL == 1
#include "../gbasd/tests/models/scsd_model.h"
/* kokoSD 1.1: a stricter SuperCard. Its SD registers are mapped into the ROM space only while the mode register's
 * SD-interface bit (bit 1) is set; we assume the card decodes them by 1 MB (0x09000000, 0x09100000, 0x09800000), so
 * while it is set every read there gives register values instead of the game's data. Koko's 32 MB ROM has model
 * data at those addresses: it must turn the interface off after each card access (engine/diskio.c). */
static int sc_window(uint32_t a) { uint32_t m = a >> 20; return m == 0x090 || m == 0x091 || m == 0x098; }
static uint32_t strict_load32(struct ARMCore *c, uint32_t a, int *cy) {
    if (!sc_window(a) || !(sd_mode & 2)) return original_memory.load32(c, a, cy);
    if (a == 0x09000000 || (a >= 0x09100000 && a < 0x09100020)) return sd_load32(c, a, cy);
    return 0xA5A5A5A5u;
}
static uint32_t strict_load16(struct ARMCore *c, uint32_t a, int *cy) {
    if (!sc_window(a) || !(sd_mode & 2)) return original_memory.load16(c, a, cy);
    if (a == 0x09000000 || a == 0x09100000 || a == 0x09800000) return sd_load16(c, a, cy);
    return 0xA5A5u;
}
static uint32_t strict_load8(struct ARMCore *c, uint32_t a, int *cy) {
    if (!sc_window(a) || !(sd_mode & 2)) return original_memory.load8(c, a, cy);
    return 0xA5u;
}
static uint32_t strict_load_multiple(struct ARMCore *c, uint32_t a, int mask, enum LSMDirection d, int *cy) {
    if (!sc_window(a) || !(sd_mode & 2)) return original_memory.loadMultiple(c, a, mask, d, cy);
    if (a == 0x09100000) return sd_load_multiple(c, a, mask, d, cy);
    uint32_t at = a;
    for (int i = 0; i < 16; i++) if (mask & (1 << i)) { c->gprs[i] = 0xA5A5A5A5u; at += 4; }
    return at;
}
static void strict_install(struct mCore *core) {
    struct ARMCore *cpu = core->cpu;
    cpu->memory.load32 = strict_load32; cpu->memory.load16 = strict_load16; cpu->memory.load8 = strict_load8;
    cpu->memory.loadMultiple = strict_load_multiple;
}
#elif CART_MODEL == 2
#include "../gbasd/tests/models/ezfo_model.h"
#elif CART_MODEL == 3
#include "../gbasd/tests/models/edgba_model.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void quiet(struct mLogger *l, int c, enum mLogLevel lv, const char *f, va_list a) {
    (void)l; (void)c; (void)lv; (void)f; (void)a;
}
static struct mLogger logger = { .log = quiet };

static uint32_t rd32(struct mCore *core, uint32_t a) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (core->busRead8(core, a + i) & 0xFF) << (8 * i);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    mLogSetDefaultLogger(&logger);
    struct mCore *core = mCoreFind(argv[1]);
    if (!core) { fprintf(stderr, "not a ROM mGBA recognises\n"); return 1; }
    core->init(core);
    mCoreInitConfig(core, NULL);
    unsigned w, h;
    core->desiredVideoDimensions(core, &w, &h);
    color_t *buf = malloc(w * h * sizeof(color_t));
    core->setVideoBuffer(core, buf, w);
    if (!mCoreLoadFile(core, argv[1])) { fprintf(stderr, "load failed\n"); return 1; }
    core->reset(core);
    if (getenv("KOKO_JUNK")) {                         /* kokoSD 1.4.5: memory as hardware leaves it (not zeros) */
        uint32_t x = (uint32_t)strtoul(getenv("KOKO_JUNK"), 0, 0) | 1, mark = *getenv("KOKO_JUNK") == 'm';
        if (mark) { for (uint32_t a = 0x03000000; a < 0x03007F00; a++) core->busWrite8(core, a, 0xA5); }
        else {
            for (uint32_t a = 0x02000000; a < 0x02040000; a++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; core->busWrite8(core, a, x >> 24); }
            for (uint32_t a = 0x03000000; a < 0x03007F00; a++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; core->busWrite8(core, a, x >> 24); }
        }
    }
#ifdef CART_MODEL
    model_install(core);
#endif
#if CART_MODEL == 1
    strict_install(core);
#endif
    long frames = 0;
    for (int i = 2; i < argc; i++) {
        char *s = argv[i];
        if (!strncmp(s, "f:", 2)) {
            for (long n = atol(s + 2); n > 0; n--, frames++) core->runFrame(core);
        } else if (!strncmp(s, "k:", 2)) {
            core->setKeys(core, (uint32_t)strtoul(s + 2, 0, 0));
            for (int n = 0; n < 3; n++, frames++) core->runFrame(core);
            core->setKeys(core, 0);
            for (int n = 0; n < 3; n++, frames++) core->runFrame(core);
        } else if (!strncmp(s, "h:", 2)) {           /* kokoSD: h:MASK:FRAMES holds exactly these buttons (gbawriter chords) */
            char *e;
            uint32_t mask = (uint32_t)strtoul(s + 2, &e, 0);
            long n = atol(e + 1);
            core->setKeys(core, mask);
            for (; n > 0; n--, frames++) core->runFrame(core);
        } else if (!strncmp(s, "until:", 6)) {
            char *e;
            uint32_t addr = (uint32_t)strtoul(s + 6, &e, 0), val = (uint32_t)strtoul(e + 1, &e, 0);
            long max = atol(e + 1), n = 0;
            while (rd32(core, addr) != val && n < max) { core->runFrame(core); n++; frames++; }
            if (rd32(core, addr) != val) { fprintf(stderr, "until: timed out after %ld frames\n", n); return 2; }
        } else if (!strncmp(s, "shot:", 5)) {
            FILE *f = fopen(s + 5, "wb");
            fprintf(f, "P6\n%u %u\n255\n", w, h);
            for (unsigned p = 0; p < w * h; p++) {
                uint32_t c = buf[p];
                fputc(c & 0xFF, f); fputc((c >> 8) & 0xFF, f); fputc((c >> 16) & 0xFF, f);
            }
            fclose(f);
        } else if (!strncmp(s, "dump:", 5)) {
            char *e;
            uint32_t addr = (uint32_t)strtoul(s + 5, &e, 0), len = (uint32_t)strtoul(e + 1, &e, 0);
            FILE *f = fopen(e + 1, "wb");
            for (uint32_t a = 0; a < len; a++) fputc((int)(core->busRead8(core, addr + a) & 0xFF), f);
            fclose(f);
        } else if (!strncmp(s, "poke:", 5)) {         /* kokoSD 1.4.7: poke:ADDR:WORD (a crash on purpose) */
            char *e;
            uint32_t addr = (uint32_t)strtoul(s + 5, &e, 0), val = (uint32_t)strtoul(e + 1, &e, 0);
            for (int i = 0; i < 4; i++) core->busWrite8(core, addr + i, (val >> (8 * i)) & 0xFF);
        } else if (!strcmp(s, "frames")) {
            printf("%ld\n", frames);
        } else { fprintf(stderr, "bad step %s\n", s); return 1; }
    }
    if (getenv("KOKO_JUNK") && *getenv("KOKO_JUNK") == 'm') {   /* kokoSD 1.4.5: the deepest stack use */
        uint32_t a = 0x03007764;
        while (a < 0x03007F00 && (core->busRead8(core, a) & 0xFF) == 0xA5) a++;
        FILE *f = fopen("/tmp/koko_stack.txt", "a"); if (f) { fprintf(f, "%#x\n", a); fclose(f); }
    }
    return 0;
}
