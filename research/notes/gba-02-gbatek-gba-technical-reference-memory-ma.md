# GBATEK: GBA technical reference (https://problemkaputt.de/gbatek.htm, living document, copy "extracted from no$gba version 3.06", Martin Korth / nocash)

## What it is
The standard reverse-engineered hardware reference for the GBA. I read the raw HTML directly: Memory Map, LCD timings, DMA, Interrupts, System Control, GamePak Prefetch, the cartridge chapters, and the ARM chapters on cycle times, multiply and load/store. It is a specification, not a benchmark. Figures are GBATEK entries, or my arithmetic from its formulas where marked "derived".

## Mechanism
Memory (bus width; cycles for 8/16/32-bit access):
- IWRAM 03000000h, 32 KB, 32-bit, 1/1/1. The top 256 bytes are reserved for BIOS/IRQ use.
- EWRAM 02000000h, 256 KB, 16-bit, 3/3/6.
- ROM 08000000h-09FFFFFFh, max 32 MB, 16-bit, 5/5/8 at boot (WAITCNT=0000h).
- Cart SRAM 0E000000h, max 64 KB, 8-bit only, 5 cycles at boot.

WAITCNT (4000204h): bits 2-3 = ROM first-access waits (0..3 = 4,3,2,8); bit 4 = sequential waits (0..1 = 2,1); bit 14 = prefetch. Access time = 1 + waits. Commercial setting: 4317h (ROM 3,1, prefetch on). A 32-bit ROM read is two 16-bit reads, the second always sequential. Sequential timing breaks at each 128 KB ROM boundary.

Instruction costs: ALU 1S; LDR 1S+1N+1I; STR 2N; LDM nS+1N+1I; STM (n-1)S+2N; B/BL 2S+1N; MUL 1S+mI; MLA 1S+(m+1)I. m=1 if bits 31-8 of Rs are all 0 or all 1, m=2 for bits 31-16, m=3 for 31-24, else 4. When code and data sit in different regions, the data access is 1N (+(n-1)S for LDM) at the data region's timing.

Prefetch buffer: eight 16-bit values, filled while the cart bus is idle. It helps only opcodes fetched from ROM, never data reads issued by RAM code.

Port 4000800h can cut EWRAM to 2/2/4 cycles on GBA and SP only (Micro locks up, DS ignores it).

## Numbers
From GBATEK:
- Clock 16*1024*1024 Hz; frame = 280,896 cycles, 59.737 Hz.

Derived (hot code in IWRAM, WAITCNT=4317h, so ROM N=4, S=2 cycles):
- MLA with an 8-bit Rs = 3 cycles; MUL = 2.
- LDRB/LDRH from ROM = 1+4+1 = 6 cycles even at consecutive addresses: a lone load is always N.
- LDR 32-bit from ROM = 1+(4+2)+1 = 8 cycles.
- LDMIA of n words from ROM = 4n+4 cycles: 36 cycles per 32 bytes = 1.125 cycles/byte. At boot defaults 6n+4 = 52 cycles = 1.6 cycles/byte.
- IWRAM: LDR/LDRSH 3, STR 2. EWRAM: LDRB/LDRH 5, LDR 8.
- ARM opcode fetched from ROM = 4 cycles sequential, versus 1 in IWRAM.
- 3-5 M cycles/token = 3.4-5.6 tokens/s.

## Smallest scale tested, and whether the result held there
Not applicable. The project's emulator figures agree: 7.5 cycles per MAC is roughly LDRSB (3) + amortised ROM weight (1-2) + MLA (3). 1.6 cycles/weight is 6.4 cycles per ROM byte against a derived floor of about 6 (stream 1, index extract 1, table load 3, add 1).

## Training recipe and gotchas
No training content. Implementation gotchas:
- "Sequential ROM reads are cheap" holds only inside LDM, DMA3, or the second half of a 32-bit LDR. Fetch index bytes several words at a time, never with LDRB.
- LDRSH has no scaled-register addressing: store pre-doubled table indices in ROM (max 160 fits a byte).
- Put the 8-bit operand in Rs.
- Third-party flashcards are "reportedly running unstable" at 3,1: ship a WAITCNT fallback and time both.
- Byte writes to VRAM land in both halves of the halfword; draw text with 16-bit writes.
- SWI 06h Div is a software routine; use reciprocal tables in ROM.

## Mapping onto the GBA design
- IWRAM (32,512 usable bytes) holds kernel code, tables, activations and stack. 16-bit tables cost (n_in/4) x 81 x 2 bytes: 10,368 B at 256 inputs, 15,552 B at 384, 41,472 B at 1024 (does not fit). A 4x-wide MLP hidden layer therefore needs 32-entry tables (16,384 B at 1024) or a ROM layout blocked by input chunk: for each block of at most 384 inputs, build tables, then stream every row's bytes for that block. Tables in EWRAM would cost 5 cycles per lookup, not 3.
- Five trits per byte (243-entry tables) would give about 1.2 cycles/weight, but needs 24,786 B of tables at 255 inputs and 2.4x the build adds per input. Four per byte is the IWRAM-bounded optimum.
- KV cache: EWRAM gives 1,024 B per position at 256 positions. Sign keys plus full-width 8-bit values need 1,152 B (d=256, 4 layers) or 2,592 B (d=384, 6 layers). Neither fits. Options: narrower or 4-bit stored values, a shorter ring, or one shared un-projected vector per position with each layer's value matrix applied after mixing (exact by linearity, but an architecture change).
- ROM: 32 MB = 134 M weights at 4 per byte. A random 384-byte embedding row via LDM costs about 390 cycles: effectively free.

## Verdict
Adopt as the cost model. Settling experiment: time the kernel with hardware timers at WAITCNT 0000h and 4317h, in mGBA and on real hardware; produce a linker map proving the IWRAM budget.

## Could not verify
- GBATEK gives no cycle count for BIOS Div.
- Its ROM-size paragraph is dated; 32 MB appears only as the address-space maximum.
- Flashcart and emulator fidelity for 3,1 timing and the 128 KB rule: not checked.
- The date of the v3.06 text.

## Links worth following
- https://mgba.io/2015/06/27/cycle-counting-prefetch/ (read via fetch summary: prefetch example, 7 cycles to 4)
- https://gbadev.net/tonc/asm.html (read via fetch summary: tables 23.7 and 23.8 agree with GBATEK, except it gives the LDR opcode fetch as 1N)
