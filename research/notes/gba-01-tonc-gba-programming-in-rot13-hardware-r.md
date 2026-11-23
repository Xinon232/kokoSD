# Tonc: GBA Programming in rot13
https://gbadev.net/tonc/ . Jasper "cearn" Vijn, text to v1.4.1 (Mar 2013); mdBook port by gbadev.org contributors (Dec 2023). Build read: commit 7920233.
Tags: [P] Tonc, read in full from its print view; [G] GBATEK; [F] gbadev forum thread 9602; [I] my arithmetic.

## What it is
The standard GBA programming text. Three parts matter to Koko: the memory map with wait states, the ARM/Thumb cycle-cost model (ch. 23), and two text systems with measured per-glyph costs (ch. 19, 22). It also gives a timer-based profiler and a table of emulator timing error against hardware. It contains no machine learning; it is the cost model behind our cycles-per-weight figures.

## Mechanism
Memory, default waits; cycles per 8/16/32-bit access (Table 23.8) [P]:

| Section | Size | Bus | Cycles |
|---|---|---|---|
| IWRAM | 32 KB | 32 | 1/1/1 |
| VRAM | 96 KB | 16 | 1/1/2 |
| EWRAM | 256 KB | 16 | 3/3/6 |
| ROM | up to 32 MB | 16 | 5/5/8 (waits N/S 4/2) |

Instruction costs (Table 23.7) [P]: data op 1S; `ldr` 1N+1Nd+1I; `str` 1N+1Nd; `ldm` of n words 1N+1Nd+(n-1)Sd+1I; taken branch 2S+1N; `mul` 1S+mI, `mla` one I more, m = significant bytes of the second operand. N and S cycles cost 1 + the section's wait; I cycles never wait; "d" cycles are charged at the data's section. A 32-bit access on a 16-bit bus is two accesses.

Rules [P]:
- ARM code only in IWRAM (`.iwram`, `-marm -mlong-calls`); Thumb in ROM. Thumb lacks conditional execution, inline shifts and write-back addressing.
- `const` data stays in ROM; other globals default to IWRAM.
- Use int locals (78% faster than s8/u16 in the ch. 3 example), count loops down, load words with `ldm`.
- Halfword and signed-byte loads cannot take a shifted-register offset.
- No divide: BIOS Div costs about 90 + 13 cycles per significant bit. For constants: `m = (2^F + A - 1)/A; y = (x*m) >> F`.

Profiler [P]:
```
start: TM2D=0; TM3D=0; TM2CNT=0; TM3CNT=0; TM3CNT=ENABLE|CASCADE; TM2CNT=ENABLE
stop:  TM2CNT=0; cycles = (TM3D<<16) | TM2D
```
ROM waits [G]: WAITCNT = 0x4317 sets waits 3/1 (4 cycles for the first halfword, 2 per sequential one) and prefetch. Prefetch does not help ROM data read by IWRAM code. Sequential timing restarts at every 128 KB ROM boundary.

## Numbers
- Clock 2^24 Hz; frame 280,896 cycles (59.73 Hz) (Table 4.1) [P].
- One mode-4 bitmap copied EWRAM to VRAM on hardware (Table 19.3) [P]: u16 loop 614,571 cycles; u32 loop 289,825; memcpy 195,156; memcpy32 86,816; DMA32 76,889.
- Emulator error, same table [P]: no$gba 0 to -1.7%; VisualBoyAdvance -9 to -21%, and -99.7% for DMA.
- Cycles per character; 194 characters, verdana 9, code in ROM, default waits, -O2 (Table 22.7) [P]: engine overhead 221; tilemap 8x16 370; generic tilemap 595; variable-width rendering into 4bpp tiles: plain C 3,049, tuned C 2,044, ARM assembly in IWRAM 631.
- SuperCard carts cannot run 3/1 ROM waits (ch. 3) [P].

## Smallest scale tested, and whether the result held there
Not applicable: no model. Timings: short routines on the author's console. `ldr` = 1N+1Nd+1I was measured with code in ROM, data in ROM and IWRAM, four wait settings, no prefetch [F].

## Training recipe and gotchas
No training content. Implementation gotchas [P]: VRAM rejects byte writes, and `memcpy` may write bytes; word casts need 4-byte alignment; `printf` takes about 1,300 B of IWRAM stack; the top of IWRAM holds the interrupt vector (0300:7FFC).

## Mapping onto the GBA design
All [I], from the tables above at 3/1 waits.
- Ternary kernel: mask 1 + table `ldr` 3 + add 1 = 5 cycles per ROM byte; two-word `ldm` from ROM 12 per 8 bytes; loop 4. Total 56 cycles per 32 weights = 1.75 cycles/weight (1.63 with rows unrolled), against about 1.6 measured.
- 8-bit multiply-accumulate: `mla` with the 8-bit operand second is 3 cycles; with fetch and sign extension about 7.5, as measured.
- The partial-sum table must live in IWRAM: with 32-bit entries in EWRAM a lookup costs 10 cycles instead of 5 and a build store 7 instead of 2. The table takes (widest matrix input) x (entries) bytes: 256x81 = 20.7 KB fits; 384x81 = 31.1 KB does not; 384x32 = 12.3 KB does. So 384 wide needs the 32-entry format or 16-bit entries (15.6 KB at 81). For 16-bit entries, pre-double the offset in the mask (`and r10, r12, w, lsr #7`, r12 = 0x1FE) and use `ldrsh r11, [r6, r10]!`: same instruction count; activations must stay within 13 bits.
- EWRAM (6 cycles/word) is slower than sequential ROM (4). Spare VRAM, about 60 KB, reads 16-bit values in 1 cycle and could hold keys, values or tables; GBATEK adds 1 cycle when the display reads VRAM at the same moment [G].
- A byte load from ROM costs 6 cycles; keep word loads. A row crossing a 128 KB boundary costs 2 extra cycles.
- Text: 25 glyphs/s x 631 = 16K cycles/s, 0.1% of the CPU, so use the variable-width tile renderer.

## Verdict
Adopt as cost model and coding standard; the kernels already follow it. Experiments: (1) hand-count the kernel against mGBA, then run the profiler on NanoBoyAdvance or SkyEmu, which Tonc calls cycle accurate, and on a console; (2) time a 16-bit table in VRAM with the display on.

## Could not verify
- mGBA's timing: Tonc calls it "highly accurate", but Table 19.3 covers only VBA and no$gba.
- WAITCNT bits, prefetch scope, the 128 KB rule and VRAM contention are GBATEK's; Tonc only names REG_WAITCNT.
- Interrupt entry cost, and VRAM used as work RAM: not in Tonc.
- Why the hand count (1.75) exceeds the measured 1.6.
- libtonc source: GitHub was blocked here.

## Links worth following
- GBATEK: https://problemkaputt.de/gbatek.htm
- `ldr` timing thread: https://gbadev.net/forum-archive/thread/8/9602.html
- libtonc, linked from Tonc: https://github.com/gbadev-org/libtonc
