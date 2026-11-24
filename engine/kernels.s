@ Koko kernels beyond bench/tern.s (which provides tern_build81 and tern_scan).
    .section .iwram.text, "ax"
    .arm

    .macro LK4 w
    and     r10, r12, \w
    ldr     r11, [r6, r10, lsl #2]!
    add     r7, r7, r11
    and     r10, r12, \w, lsr #8
    ldr     r11, [r6, r10, lsl #2]!
    add     r7, r7, r11
    and     r10, r12, \w, lsr #16
    ldr     r11, [r6, r10, lsl #2]!
    add     r7, r7, r11
    mov     r10, \w, lsr #24
    ldr     r11, [r6, r10, lsl #2]!
    add     r7, r7, r11
    .endm

@ void tern_gather(int32_t *out, const uint16_t *ids, int n, const uint8_t *steps, const int32_t *tbl, const uint16_t *mult)
@ For each id: the 256-input row at steps + id*64, summed through tbl, times mult[id], rounded >> 16.
    .global tern_gather
tern_gather:
    push    {r4-r11, lr}
    sub     sp, sp, #12
    str     r3, [sp, #4]            @ steps base
    mov     r5, r2                  @ rows left
    mov     r3, r1                  @ ids
    ldr     r2, [sp, #48]           @ tbl
    sub     r2, r2, #4
    ldr     r6, [sp, #52]
    str     r6, [sp, #8]            @ mult base
    mov     r12, #0xFF
.Lg_row:
    ldrh    r9, [r3], #2
    ldr     r1, [sp, #4]
    str     r9, [sp]
    add     r1, r1, r9, lsl #6
    mov     r6, r2
    mov     r7, #0
    mov     r8, #8                  @ 16 words, two per pass
.Lg_in:
    ldmia   r1!, {r9, r14}
    LK4     r9
    LK4     r14
    subs    r8, r8, #1
    bne     .Lg_in
    ldr     r9, [sp]
    ldr     r10, [sp, #8]
    add     r10, r10, r9, lsl #1
    ldrh    r9, [r10]
    smull   r10, r11, r7, r9
    adds    r10, r10, #0x8000
    adc     r11, r11, #0
    mov     r10, r10, lsr #16
    orr     r10, r10, r11, lsl #16
    str     r10, [r0], #4
    subs    r5, r5, #1
    bne     .Lg_row
    add     sp, sp, #12
    pop     {r4-r11, lr}
    bx      lr

@ void mixer_k(int32_t *x, const int32_t *xn, int32_t *state, const int32_t *coef, int n)
@ Per dimension: state = h1 h2 h3 e0 e1 e2, coef = 3 taps then 3 average weights (see koko.c).
    .global mixer_k
mixer_k:
    push    {r4-r11, lr}
.Lm:
    ldr     r10, [r1], #4           @ xn
    ldmia   r2, {r4-r9}
    ldr     r14, [r3], #4
    smull   r11, r12, r14, r4
    ldr     r14, [r3], #4
    smlal   r11, r12, r14, r5
    ldr     r14, [r3], #4
    smlal   r11, r12, r14, r6
    sub     r6, r10, r7             @ e0 += (xn - e0 + 1) >> 1
    add     r6, r6, #1
    add     r7, r7, r6, asr #1
    ldr     r14, [r3], #4
    smlal   r11, r12, r14, r7
    sub     r6, r10, r8             @ e1 += ((xn - e1) * 205 + 512) >> 10
    mov     r14, #205
    mul     r6, r14, r6
    add     r6, r6, #512
    add     r8, r8, r6, asr #10
    ldr     r14, [r3], #4
    smlal   r11, r12, r14, r8
    sub     r6, r10, r9             @ e2 += ((xn - e2) * 51 + 512) >> 10
    mov     r14, #51
    mul     r6, r14, r6
    add     r6, r6, #512
    add     r9, r9, r6, asr #10
    ldr     r14, [r3], #4
    smlal   r11, r12, r14, r9
    mov     r6, r5                  @ history moves down one
    mov     r5, r4
    mov     r4, r10
    stmia   r2!, {r4-r9}
    adds    r11, r11, #0x800
    adc     r12, r12, #0
    mov     r11, r11, lsr #12
    orr     r11, r11, r12, lsl #20
    ldr     r14, [r0]
    add     r14, r14, r11
    str     r14, [r0], #4
    ldr     r14, [sp, #36]
    subs    r14, r14, #1
    str     r14, [sp, #36]
    bne     .Lm
    pop     {r4-r11, lr}
    bx      lr

@ uint64_t sumsq_k(const int32_t *in, int n)
    .global sumsq_k
sumsq_k:
    mov     r2, #0
    mov     r3, #0
1:  ldr     r12, [r0], #4
    smlal   r2, r3, r12, r12
    subs    r1, r1, #1
    bne     1b
    mov     r0, r2
    mov     r1, r3
    bx      lr

@ void norm_k(int32_t *out, const int32_t *in, const int32_t *g, int32_t inv, int n)
@ out = rs(rs(in * inv, 28) * g, 12), for inv below 2^31.
    .global norm_k
norm_k:
    push    {r4-r8, lr}
    ldr     r4, [sp, #24]
1:  ldr     r5, [r1], #4
    smull   r6, r7, r5, r3
    adds    r6, r6, #0x08000000
    adc     r7, r7, #0
    mov     r6, r6, lsr #28
    orr     r6, r6, r7, lsl #4
    ldr     r5, [r2], #4
    smull   r8, r7, r6, r5
    adds    r8, r8, #0x800
    adc     r7, r7, #0
    mov     r8, r8, lsr #12
    orr     r8, r8, r7, lsl #20
    str     r8, [r0], #4
    subs    r4, r4, #1
    bne     1b
    pop     {r4-r8, lr}
    bx      lr
