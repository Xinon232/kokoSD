/* kokoSD: mounting the SD card. GPL-3.0-or-later.
 * Koko uses no interrupts, so the Omega driver keeps them masked during each
 * card command (ezfo_irq_live stays 0); its code runs from slow RAM (EWRAM)
 * because the Omega maps its kernel over the game ROM during card access. */
#include "sd.h"
#include "cart_io.h"

FATFS sd_fs __attribute__((section(".bss.sd")));
int sd_ready;

int sd_mount(void) {
    sd_ready = f_mount(&sd_fs, "", 1) == FR_OK;
    return sd_ready;
}

const char *sd_cart_name(void) {
    switch (cart_kind) {
    case CART_SUPERCARD: return "SuperCard";
    case CART_EZFO: return "EZ-Flash Omega";
    case CART_EZFO_NO_CARD: return "EZ-Flash Omega (no SD card)";
    case CART_EDGBA: return "EverDrive";
    case CART_EDGBA_NO_CARD: return "EverDrive (no SD card)";
    default: return "unknown cart";
    }
}
