/* kokoSD: the SD card through gbaSD (SuperCard SD, EZ-Flash Omega, EverDrive GBA).
 * GPL-3.0-or-later. */
#ifndef KOKO_SD_H
#define KOKO_SD_H
#include "ff.h"
extern FATFS sd_fs;
/* Detects the cart and mounts the card's file system: 1 ready, 0 not. */
int sd_mount(void);
extern int sd_ready, sd_err;
extern unsigned sd_init_err;                    /* kokoSD 1.4.3: the card start's error (SuperCard: SD_ERR_* step) */
/* The cart for messages: "SuperCard", "EZ-Flash Omega", "EverDrive", ... */
const char *sd_cart_name(void);
#endif
