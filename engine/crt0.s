@ Cartridge header and startup. No BIOS calls, no interrupts.
    .section .crt0, "ax"
    .arm
    .global _start
_start:
    b       rom_start
    .fill   156, 1, 0               @ 0x04 boot logo: the console BIOS checks these bytes;
                                    @ left blank here (emulators don't need them)
    .ascii  "KOKO\0\0\0\0\0\0\0\0"         @ 0xA0 title
    .ascii  "KOKE"                  @ 0xAC game code
    .ascii  "00"                    @ 0xB0 maker
    .byte   0x96, 0, 0              @ 0xB2 fixed, unit, device
    .fill   7, 1, 0
    .byte   0                       @ 0xBC version
    .byte   0                       @ 0xBD header checksum (tools/fixrom.py)
    .byte   0, 0
rom_start:
    @ kokoSD 1.4.7: first the registers a restart leaves behind (engine/game.c shows them on the title when
    @ Koko restarted by itself): the lr and status of the mode it starts in, then lr and saved status of the
    @ undefined-instruction, abort, supervisor and IRQ modes (a crash into the BIOS leaves its address there)
    ldr     r1, =boot_regs          @ (in slow RAM, not cleared below)
    str     lr, [r1, #0]
    mrs     r0, cpsr
    str     r0, [r1, #4]
    msr     cpsr_c, #0x9B           @ undefined-instruction mode, interrupts off
    mov     r2, lr
    mrs     r3, spsr
    msr     cpsr_c, #0x97           @ abort
    mov     r4, lr
    mrs     r5, spsr
    msr     cpsr_c, #0x93           @ supervisor
    mov     r6, lr
    mrs     r7, spsr
    msr     cpsr_c, #0x92           @ IRQ
    mov     r8, lr
    mrs     r9, spsr
    msr     cpsr_c, #0x9F           @ system
    add     r1, r1, #8
    stmia   r1, {r2-r9}
    mov     r0, #0x04000000         @ kokoSD 1.4.7: no interrupt until the game asks (a restart can leave the
    str     r0, [r0, #0x208]        @ freeze detector's on, and its code in slow RAM is copied below): IME off
    mov     r1, #0
    strh    r1, [r0, #4]            @ no video-frame interrupt
    str     r1, [r0, #0x200]        @ IE off
    mov     r0, #0x12               @ IRQ mode stack
    msr     cpsr_c, r0
    ldr     sp, =0x03007FA0
    mov     r0, #0x1F               @ system mode stack
    msr     cpsr_c, r0
    ldr     sp, =0x03007F00
    ldr     r0, =__iwram_lma        @ copy fast-RAM code and data out of ROM
    ldr     r1, =__iwram_start
    ldr     r2, =__iwram_end
1:  cmp     r1, r2
    ldrlo   r3, [r0], #4
    strlo   r3, [r1], #4
    blo     1b
    ldr     r0, =__ewram_sd_lma     @ kokoSD: copy gbaSD's slow-RAM code and data out of ROM
    ldr     r1, =__ewram_sd_start
    ldr     r2, =__ewram_sd_end
4:  cmp     r1, r2
    ldrlo   r3, [r0], #4
    strlo   r3, [r1], #4
    blo     4b
    ldr     r1, =__ewram_sd_bss_start   @ and clear its variables (FatFs expects zeros)
    ldr     r2, =__ewram_sd_bss_end
    mov     r3, #0
5:  cmp     r1, r2
    strlo   r3, [r1], #4
    blo     5b
    ldr     r1, =__bss_start        @ clear fast-RAM bss (slow-RAM buffers are not cleared)
    ldr     r2, =__bss_end
    mov     r3, #0
2:  cmp     r1, r2
    strlo   r3, [r1], #4
    blo     2b
    ldr     r0, =main
    mov     lr, pc
    bx      r0
3:  b       3b
    .pool
