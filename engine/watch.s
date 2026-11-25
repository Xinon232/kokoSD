@ kokoSD 1.4.6: freeze detector. A video-frame interrupt counts the frames in which poll() did not run
@ (engine/game.c) and how long L+R+SELECT are held. After 16 s without poll(), or 2 s of L+R+SELECT, it notes
@ where the program was (the interrupted address, its mode, the program's sp and lr) and makes the program
@ continue in watch_stuck, which shows that on the screen. GPL-3.0-or-later.
@ The interrupt code and its record are in slow RAM: the Omega maps its kernel over the ROM during card access.
@ watch: 0 beat (poll counts), 1 beat seen, 2 frames still, 3 frames L+R+SELECT, 4 on, 5 why (1 still, 2 keys),
@        6 address, 7 status (spsr), 8 sp, 9 lr
    .section .ewram.watch, "ax"
    .arm
    .align 2
    .global watch_irq
    .global watch
watch_irq:                              @ called by the BIOS in IRQ mode: sp -> its saved r0-r3, r12, lr
    mov     r3, #0x04000000
    add     r3, r3, #0x200
    ldr     r2, [r3]                    @ IE | IF << 16
    and     r2, r2, r2, lsr #16
    strh    r2, [r3, #2]                @ acknowledge
    adr     r1, watch
    ldr     r0, [r1, #16]
    cmp     r0, #0
    bxeq    lr
    ldr     r0, [r1, #0]
    ldr     r2, [r1, #4]
    str     r0, [r1, #4]
    cmp     r0, r2
    ldr     r2, [r1, #8]
    addeq   r2, r2, #1
    movne   r2, #0
    str     r2, [r1, #8]
    ldrh    r0, [r3, #-0xD0]            @ buttons (0x04000130; 0 = held)
    tst     r0, #0x300                  @ L and R
    tsteq   r0, #4                      @ and SELECT
    ldr     r12, [r1, #12]
    addeq   r12, r12, #1
    movne   r12, #0
    str     r12, [r1, #12]
    cmp     r12, #120
    movhs   r0, #2
    bhs     1f
    cmp     r2, #960
    bxlo    lr
    mov     r0, #1
1:  str     r0, [r1, #20]
    mov     r0, #0
    str     r0, [r1, #16]               @ once
    ldr     r0, [sp, #20]
    sub     r0, r0, #4
    str     r0, [r1, #24]               @ where the program was
    mrs     r2, spsr
    str     r2, [r1, #28]
    add     r0, r1, #32
    stmia   r0, {sp, lr}^               @ the program's sp and lr
    nop
    bic     r2, r2, #0x20               @ continue in ARM state, interrupts off
    orr     r2, r2, #0x80
    msr     spsr_fsxc, r2
    ldr     r0, =watch_stuck
    add     r0, r0, #4
    str     r0, [sp, #20]
    bx      lr
    .pool
watch:
    .space  40

    .text
    .arm
    .align 2
    .global watch_stuck
watch_stuck:                            @ a stack of its own (the program's may be what went wrong)
    mov     sp, #0x03000000
    add     sp, sp, #0x7F00
    b       watch_report
    .pool
