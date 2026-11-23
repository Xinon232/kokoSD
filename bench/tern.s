@ Ternary matrix-vector product by table lookup, for ARM7TDMI in fast RAM.
@
@ For every 4 inputs, tern_build writes the 81 sums those inputs can make with
@ weights -1/0/+1. A row of the matrix is then a string of bytes, one per 4
@ weights. The byte is a pointer STEP: "ldr t, [p, step, lsl #2]!" moves p
@ from the entry used in one table straight to the entry wanted in the next
@ (the exporter knows both), so a lookup is 3 instructions and no multiply.
    .section .iwram.text, "ax"
    .arm

@ void tern_build(int32_t *tbl, const int32_t *x, int ngroups)
@ entry index = ta + 3*tb + 9*tc + 27*td, trit codes 0: 0, 1: +x, 2: -x
    .macro BLOCK9
    str     r12, [r0], #4
    add     r14, r12, r3
    str     r14, [r0], #4
    add     r14, r12, r4
    str     r14, [r0], #4
    add     r14, r12, r5
    str     r14, [r0], #4
    add     r14, r12, r6
    str     r14, [r0], #4
    add     r14, r12, r7
    str     r14, [r0], #4
    add     r14, r12, r8
    str     r14, [r0], #4
    add     r14, r12, r9
    str     r14, [r0], #4
    add     r14, r12, r10
    str     r14, [r0], #4
    .endm
    .global tern_build81
tern_build81:
    push    {r4-r11, lr}
1:  ldmia   r1!, {r3, r5, r11}      @ a, b, c
    add     r1, r1, #4              @ d stays in memory at [r1, #-4]
    rsb     r4, r3, #0              @ -a
    add     r6, r3, r5              @ a+b
    sub     r7, r5, r3              @ b-a
    rsb     r8, r5, #0              @ -b
    sub     r9, r3, r5              @ a-b
    rsb     r10, r6, #0             @ -a-b
    mov     r12, #0
    BLOCK9
    add     r12, r12, r11           @ c
    BLOCK9
    sub     r12, r12, r11, lsl #1   @ -c
    BLOCK9
    ldr     r14, [r1, #-4]
    add     r12, r12, r11
    add     r12, r12, r14           @ d
    BLOCK9
    add     r12, r12, r11           @ d+c
    BLOCK9
    sub     r12, r12, r11, lsl #1   @ d-c
    BLOCK9
    ldr     r14, [r1, #-4]
    add     r12, r12, r11
    sub     r12, r12, r14, lsl #1   @ -d
    BLOCK9
    add     r12, r12, r11           @ -d+c
    BLOCK9
    sub     r12, r12, r11, lsl #1   @ -d-c
    BLOCK9
    subs    r2, r2, #1
    bne     1b
    pop     {r4-r11, lr}
    bx      lr

@ void tern_build32(int32_t *tbl, const int32_t *x, int ngroups)
@ "One zero in every four weights": entry = zero position * 8 + sign bits of the
@ other three inputs in order (bit set = minus). 32 sums per group instead of 81.
    .macro Z8 u, v, w
    add     r12, \u, \v
    add     r12, r12, \w
    str     r12, [r0], #4
    sub     r14, r12, \u, lsl #1
    str     r14, [r0], #4
    sub     r14, r12, \v, lsl #1
    str     r14, [r0], #4
    sub     r14, r14, \u, lsl #1
    str     r14, [r0], #4
    sub     r12, r12, \w, lsl #1
    str     r12, [r0], #4
    sub     r14, r12, \u, lsl #1
    str     r14, [r0], #4
    sub     r14, r12, \v, lsl #1
    str     r14, [r0], #4
    sub     r14, r14, \u, lsl #1
    str     r14, [r0], #4
    .endm
    .global tern_build32
tern_build32:
    push    {r4-r6, lr}
1:  ldmia   r1!, {r3-r6}            @ a, b, c, d
    Z8      r4, r5, r6
    Z8      r3, r5, r6
    Z8      r3, r4, r6
    Z8      r3, r4, r5
    subs    r2, r2, #1
    bne     1b
    pop     {r4-r6, lr}
    bx      lr

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
    .macro ROWSUM
    mov     r6, r2
    mov     r7, #0
    movs    r8, r4, lsr #1
    bcc     2f
    ldr     r9, [r1], #4
    LK4     r9
    cmp     r8, #0
    beq     3f
2:  ldmia   r1!, {r9, r14}
    LK4     r9
    LK4     r14
    subs    r8, r8, #1
    bne     2b
3:
    .endm

@ void tern_rows(int32_t *out, const uint8_t *steps, const int32_t *tbl,
@                const int32_t *mult, int nwords, int nrows)
@ out[r] = (rowsum * mult[r] + 2^15) >> 16.  nwords = inputs / 16.
    .global tern_rows
tern_rows:
    push    {r4-r11, lr}
    ldr     r4, [sp, #36]
    ldr     r5, [sp, #40]
    sub     r2, r2, #4
    mov     r12, #0xFF
.Lrow:
    ROWSUM
    ldr     r9, [r3], #4
    smull   r10, r11, r7, r9
    adds    r10, r10, #0x8000
    adc     r11, r11, #0
    mov     r10, r10, lsr #16
    orr     r10, r10, r11, lsl #16
    str     r10, [r0], #4
    subs    r5, r5, #1
    bne     .Lrow
    pop     {r4-r11, lr}
    bx      lr

@ void tern_scan(int32_t *out, const uint8_t *steps, const int32_t *tbl, int nwords, int nrows)
@ Same row sum, unscaled: attention scores of one query against cached ternary keys.
    .global tern_scan
tern_scan:
    push    {r4-r11, lr}
    mov     r4, r3
    ldr     r5, [sp, #36]
    sub     r2, r2, #4
    mov     r12, #0xFF
.Lscan:
    ROWSUM
    str     r7, [r0], #4
    subs    r5, r5, #1
    bne     .Lscan
    pop     {r4-r11, lr}
    bx      lr

@ void axpy8(int32_t *acc, const int8_t *v, int32_t p, int n)
@ acc[i] += p * v[i] for a row of 8-bit values read from ROM a word at a time; n a multiple of 4.
@ The 8-bit value is the multiplier operand, so each multiply takes one internal cycle.
    .global axpy8
axpy8:
    push    {r4-r9, lr}
1:  ldr     r12, [r1], #4
    ldmia   r0, {r4-r7}
    mov     r8, r12, lsl #24
    mov     r8, r8, asr #24
    mla     r4, r2, r8, r4
    mov     r8, r12, lsl #16
    mov     r8, r8, asr #24
    mla     r5, r2, r8, r5
    mov     r8, r12, lsl #8
    mov     r8, r8, asr #24
    mla     r6, r2, r8, r6
    mov     r8, r12, asr #24
    mla     r7, r2, r8, r7
    stmia   r0!, {r4-r7}
    subs    r3, r3, #4
    bne     1b
    pop     {r4-r9, lr}
    bx      lr
