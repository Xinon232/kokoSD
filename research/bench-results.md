# Emulator benchmark results (measured, mGBA 0.10.2, random weights, timing only)

All shapes: attention over a full 256-slot ring (4 heads, 64 key dims as ternary/sign keys, 64 value dims
at 8 bits, top-8 mixing), 16K vocabulary through a 128-class x 128-member head, integer RMSNorm.
Every ROM matched a native build of the same C code exactly (checksum). 16.78M cycles = 1 second.
Source: bench/ (gen.py, core.c, tern.s, run.py).

## Kernel costs
- 8-bit weights, multiply-accumulate (stage 1 kernel): about 7.5 cycles per weight in the inner loop,
  about 9 cycles per weight all-in for the stories260K model.
- Ternary lookup kernel: one lookup + add per 4 weights = about 6.4 cycles per ROM byte
  (1.6 cycles/weight) plus a per-row epilogue of about 25 cycles and the table build.
- Table build per 4 inputs: 81-entry (free ternary) about 270 cycles; 32-entry (exactly one zero per
  4 weights, Sherry-style) about 110 cycles. 32-entry tables also cut table RAM from 324 to 128 bytes
  per 4 inputs, which is what allows widths above 256 in the 32 KB fast RAM.
- A row of 256 inputs costs about 465 cycles; of 64 inputs about 120 cycles.

## Shapes with 64 hard-routed experts per layer (1 active)
| shape | tables | cycles/token | tokens/s | weights used/token | weights stored |
|---|---|---|---|---|---|
| d256 x4 | 81 | 2,388,283 | 7.02 | 0.92M | 38M |
| d256 x6 | 81 | 3,504,783 | 4.79 | 1.34M | 55M |
| d256 x6 | 32 | 3,286,292 | 5.11 | 1.34M | 55M |
| d256 x8 | 32 | 4,332,084 | 3.87 | 1.77M | 72M |
| d320 x6 | 32 | 4,359,957 | 3.85 | 1.93M | 85M |
| d384 x4 | 32 | 3,797,864 | 4.42 | 1.77M | 82M |
| d384 x6 | 32 | 5,591,626 | 3.00 | 2.61M | 120M |
| d512 x4 | 32 | 5,783,574 | 2.90 | 2.88M | 143M |
| d256 x6, 2 experts active | 81 | 5,059,434 | 3.32 | 2.13M | 55M |
| d256 x6, context 128 | 81 | 3,335,752 | 5.03 | 1.34M | 55M |

ROM note: ternary weights take 1 byte per 4 weights, so 55M weights = 13.8 MB, 82M = 20.6 MB,
120M = 30 MB (leaves no room for embeddings in a 32 MB cartridge).

## Where a token's time goes (d256 x6, 32-entry tables, 64 experts): 3.28M cycles
norm 5.0% | table builds 5.9% | q/k/v projections 15.7% | key scan over 256 slots 7.9% |
top-8 selection and mix 5.3% | attention output 7.2% | router (64 rows) 5.6% |
expert input matrix (256 rows) 21.9% | expert output matrix (256 rows) 21.8% | output head 3.7%.
Per layer: attention about 200K cycles, routed expert MLP about 270K cycles.

## Product-key memory layer (in parallel with the MLP, reusing its input tables)
Each head scores N half-keys against each half of the normalised stream (ternary rows of 128 inputs),
takes the top 8 of each, forms 64 candidates, and mixes the 8 best 8-bit value rows.
| shape | cycles/token | tokens/s |
|---|---|---|
| d256 x6, experts only | 3,283,977 | 5.11 |
| + 1 memory head, 128 half-keys (16K slots) per layer | 4,101,390 | 4.09 |
| + 2 heads, 256 half-keys (65K slots each) per layer | 5,642,314 | 2.97 |
| no experts, 4 heads x 128 half-keys | 4,752,694 | 3.53 |
| no experts, 2 heads x 256 half-keys | 3,972,713 | 4.22 |
| x8 layers, no experts, 2 heads x 256 | 5,247,622 | 3.20 |
| d384 x6, no experts, 2 heads x 256 | 5,502,428 | 3.05 |

Cost per memory head per layer as measured: about 136K cycles at 128 half-keys, about 196K at 256.
About 50K of that is the value mix (8 rows x 256 dims), which is unoptimised C reading ROM bytes;
an assembly kernel should bring it to about 16K. Key scoring is 2N rows of 128 inputs = about 230
cycles each.
ROM note: values dominate storage. 16K slots x 256 dims at 8 bits = 4 MB per head per layer;
65K slots = 16 MB. So memory layers are limited by the 32 MB ROM, not by compute.

## Gated lookup injection (not yet benchmarked, estimate)
Adding a looked-up row to the residual stream costs 256 adds (about 1-2K cycles); a scalar gate from
one 256-dim dot product with an 8-bit row costs about 2K cycles. So token / hashed n-gram tables at
every layer are effectively free in compute; they are limited by ROM (32 MB = 125K rows of 256
8-bit values, or 500K rows at 2 bits).

## Product-key memory, after moving the value mix to an assembly kernel (supersedes the table above)
| shape (all d256 unless noted, 32-entry tables) | cycles/token | tokens/s |
|---|---|---|
| x6, 64 experts, no memory | 3,283,977 | 5.11 |
| x6, 64 experts + 1 memory head, 128 half-keys (16K slots) | 3,871,555 | 4.33 |
| x6, 64 experts + 1 memory head, 256 half-keys (65K slots) | 4,257,573 | 3.94 |
| x6, no experts, 2 memory heads x 256 half-keys | 3,526,791 | 4.76 |
| x6, one small dense MLP (hidden 128) + 1 memory head x 256 | 3,371,989 | 4.98 |
| x8, one small dense MLP (hidden 128) + 1 memory head x 256 | 4,450,902 | 3.77 |
| d384 x6, small dense MLP (hidden 192) + 1 memory head x 256 | 5,170,917 | 3.24 |

Cost per memory head per layer: about 98K cycles at 128 half-keys (16K slots), about 162K at 256
half-keys (65K slots). For comparison a routed expert MLP (256 hidden) costs about 270K per layer and
a dense MLP with 128 hidden units about 126K.
