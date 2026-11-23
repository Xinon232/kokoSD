/* kokoSD: Koko's src/gba.ld plus gbaSD. gbaSD's code that must run from RAM
 * (the Omega maps its kernel over the ROM during card access) and its
 * variables go to slow RAM (EWRAM) after Koko's own buffers, because fast RAM
 * (IWRAM) is nearly full. Objects named sd-*.o (gbaSD, FatFs, kokoSD's SD
 * glue) keep their data and bss there. GPL-3.0-or-later. */
ENTRY(_start)
MEMORY {
    rom   (rx)  : ORIGIN = 0x08000000, LENGTH = 32M
    ewram (rw)  : ORIGIN = 0x02000000, LENGTH = 256K
    iwram (rwx) : ORIGIN = 0x03000000, LENGTH = 0x7800   /* leaves 1.75 KB of stack below 0x03007F00 */
}
SECTIONS {
    .text : {
        KEEP(*(.crt0))
        *(.text .text.*)
        *(.glue_7 .glue_7t .vfp11_veneer .v4_bx)
        *(.rodata .rodata.*)
        . = ALIGN(4);
    } > rom
    .iwram : {
        __iwram_start = .;
        *(.iwram.text .iwram.text.*)
        EXCLUDE_FILE(*sd-*.o) *(.data .data.*)
        . = ALIGN(4);
        __iwram_end = .;
    } > iwram AT> rom
    __iwram_lma = LOADADDR(.iwram);
    .ewram (NOLOAD) : {
        KEEP(*(.bss.dbg))
        *(.bss.ewram .bss.ewram.*)
        . = ALIGN(4);
    } > ewram
    .ewram_sd : {                                /* copied out of ROM at start */
        __ewram_sd_start = .;
        *(.ewram .ewram.*)
        *sd-*.o(.data .data.*)
        . = ALIGN(4);
        __ewram_sd_end = .;
    } > ewram AT> rom
    __ewram_sd_lma = LOADADDR(.ewram_sd);
    .ewram_sd_bss (NOLOAD) : {                   /* cleared at start */
        __ewram_sd_bss_start = .;
        *(.bss.sd .bss.sd.*)
        *sd-*.o(.bss .bss.* COMMON)
        . = ALIGN(4);
        __ewram_sd_bss_end = .;
    } > ewram
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > iwram
    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) *(.comment) *(.note*) }
    ASSERT(__ewram_sd_bss_end <= 0x02040000, "EWRAM full")
}
