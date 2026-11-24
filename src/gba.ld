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
        *(.data .data.*)
        . = ALIGN(4);
        __iwram_end = .;
    } > iwram AT> rom
    __iwram_lma = LOADADDR(.iwram);
    .ewram (NOLOAD) : {
        KEEP(*(.bss.dbg))
        *(.bss.ewram .bss.ewram.*)
        . = ALIGN(4);
    } > ewram
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > iwram
    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) *(.comment) *(.note*) }
}
