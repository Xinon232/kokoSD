@ Hand-written ARM7TDMI kernels, run from fast RAM. The C versions in model.c
@ are the definition; these must produce the same integers (test/test_rom.py).
@
@ Timing notes: MLA's cost depends on the magnitude of its last multiplier
@ operand, so the 8-bit weight always goes there (1 internal cycle).
    .section .iwram.text, "ax"
    .arm

@ ---------------------------------------------------------------------------
@ void matvec(int32_t *out, const int32_t *in, const uint8_t *w,
@             const int32_t *mult, int n_in, int n_out)
@ w holds weight+128 as unsigned bytes, n_in a multiple of 4:
@   out[r] = ((sum(w*in) - 128*sum(in)) * mult[r] + 2^19) >> 20
@ ---------------------------------------------------------------------------
    .macro MAC4 wreg
    ldmia   r1!, {r6-r9}
    and     r14, \wreg, #0xFF
    mla     r0, r6, r14, r0
    and     r14, r12, \wreg, lsr #8
    mla     r0, r7, r14, r0
    and     r14, r12, \wreg, lsr #16
    mla     r0, r8, r14, r0
    mov     r14, \wreg, lsr #24
    mla     r0, r9, r14, r0
    .endm

    .global matvec
matvec:
    push    {r4-r11, lr}
    ldr     r4, [sp, #36]           @ n_in
    ldr     r5, [sp, #40]           @ n_out
    mov     r10, r0                 @ out
    mov     r11, r3                 @ mult
    mov     r6, #0                  @ bias = 128 * sum(in)
    mov     r7, r1
    mov     r8, r4
1:  ldr     r9, [r7], #4
    add     r6, r6, r9
    subs    r8, r8, #1
    bne     1b
    mov     r6, r6, lsl #7
    push    {r1, r4, r5, r6}        @ [sp]=in  +4=n_in  +8=rows left  +12=bias
    mov     r12, #0xFF
.Lrow:
    ldr     r1, [sp]
    ldr     r3, [sp, #4]
    mov     r0, #0
    movs    r3, r3, lsr #3          @ blocks of 8; carry set = one odd block of 4
    bcc     2f
    ldr     r4, [r2], #4
    MAC4    r4
    cmp     r3, #0
    beq     3f
2:  ldmia   r2!, {r4, r5}           @ 8 weights from ROM in one sequential read
    MAC4    r4
    MAC4    r5
    subs    r3, r3, #1
    bne     2b
3:  ldr     r6, [sp, #12]
    ldr     r7, [r11], #4
    sub     r0, r0, r6
    smull   r4, r5, r0, r7
    adds    r4, r4, #0x80000
    adc     r5, r5, #0
    mov     r4, r4, lsr #20
    orr     r4, r4, r5, lsl #12
    str     r4, [r10], #4
    ldr     r7, [sp, #8]
    subs    r7, r7, #1
    str     r7, [sp, #8]
    bne     .Lrow
    add     sp, sp, #16
    pop     {r4-r11, lr}
    bx      lr

@ ---------------------------------------------------------------------------
@ int32_t att_scores(int32_t *sc, const int16_t *k, const int32_t *q, int n)
@ sc[t] = dot(q[0..7], k[t][0..7]); returns the largest score. q lives in
@ registers for the whole scan.
@ ---------------------------------------------------------------------------
    .global att_scores
att_scores:
    push    {r4-r11, lr}
    ldmia   r2, {r4-r11}
    mov     r2, #0x80000000
1:  ldrsh   r12, [r1], #2
    mul     r14, r4, r12
    ldrsh   r12, [r1], #2
    mla     r14, r5, r12, r14
    ldrsh   r12, [r1], #2
    mla     r14, r6, r12, r14
    ldrsh   r12, [r1], #2
    mla     r14, r7, r12, r14
    ldrsh   r12, [r1], #2
    mla     r14, r8, r12, r14
    ldrsh   r12, [r1], #2
    mla     r14, r9, r12, r14
    ldrsh   r12, [r1], #2
    mla     r14, r10, r12, r14
    ldrsh   r12, [r1], #2
    mla     r14, r11, r12, r14
    str     r14, [r0], #4
    cmp     r14, r2
    movgt   r2, r14
    subs    r3, r3, #1
    bne     1b
    mov     r0, r2
    pop     {r4-r11, lr}
    bx      lr
