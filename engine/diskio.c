/* kokoSD: FatFs disk glue for gbaSD's cart_io, after gbavideo's diskio.c:
 * requests are split into commands of at most 16 sectors. GPL-3.0-or-later.
 * kokoSD 1.1: on a SuperCard the SD interface is mapped into the ROM space only during card access. While it is
 * mapped, the card's registers take the place of ROM at 0x09000000, 0x09100000 and 0x09800000, and Koko's 32 MB
 * ROM keeps model data there (gbaSD leaves the interface on, which suits ROMs below 16 MB). kokoSD 1.4: the
 * EverDrive GBA's register window at 0x09FC0000 likewise. */
#include "ff.h"
#include "diskio.h"
#include "cart_io.h"
#include "supercard_driver.h"

static void sc_io(int on) {
    if (cart_kind == CART_SUPERCARD) set_supercard_mode(MAPPED_SDRAM, on, on);
    /* kokoSD 1.4: the same on an EverDrive GBA: gbaSD leaves its registers mapped at 0x09FC0000, where the end of
     * Koko's 32 MB ROM is (fonts, tables). Unlock (key 0xA5), then the configuration register: PSRAM mapped, with
     * or without the register window (as gbaSD's edgba_detect does). */
    if (cart_kind == CART_EDGBA) {
        *(volatile uint16_t *)(0x09FC0000u + 0x5A * 2) = 0xA5;
        *(volatile uint16_t *)0x09FC0000u = (uint16_t)(on ? 3 : 2);
    }
}

#define CHUNK 16u
unsigned sd_init_err;                         /* kokoSD 1.4.3: cart_sd_init's last result */
static DSTATUS state = STA_NOINIT;

DSTATUS disk_status(BYTE d) { return d ? STA_NOINIT : state; }

DSTATUS disk_initialize(BYTE d) {
    if (d) return STA_NOINIT;
    /* kokoSD 1.4.3: up to three tries; the last card-start error is kept for the title (SuperCard: the
     * failed step, gbaSD's SD_ERR_* 1-6) */
    for (int t = 0; t < 3; t++) { sd_init_err = cart_sd_init(); if (!sd_init_err) break; }
    state = sd_init_err ? STA_NOINIT : 0;
    sc_io(0);
    return state;
}

DRESULT disk_read(BYTE d, BYTE *buf, LBA_t sector, UINT count) {
    if (d || state) return RES_NOTRDY;
    while (count) {
        unsigned k = count < CHUNK ? count : CHUNK;
        sc_io(1);
        unsigned e = cart_read(buf, sector, k);
        sc_io(0);
        if (e) return RES_ERROR;
        buf += 512 * k; sector += k; count -= k;
    }
    return RES_OK;
}

DRESULT disk_write(BYTE d, const BYTE *buf, LBA_t sector, UINT count) {
    if (d || state) return RES_NOTRDY;
    while (count) {
        unsigned k = count < CHUNK ? count : CHUNK;
        sc_io(1);
        unsigned e = cart_write(buf, sector, k);
        sc_io(0);
        if (e) return RES_ERROR;
        buf += 512 * k; sector += k; count -= k;
    }
    return RES_OK;
}

DRESULT disk_ioctl(BYTE d, BYTE cmd, void *buf) {
    if (d) return RES_PARERR;
    if (cmd == CTRL_SYNC) return RES_OK;
    if (cmd == GET_SECTOR_SIZE) { *(WORD *)buf = 512; return RES_OK; }
    return RES_PARERR;
}
