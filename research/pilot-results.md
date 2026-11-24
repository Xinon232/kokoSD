# Pilot experiment results (18 Nov 2026, RTX 5090)

Setup: decoder-only model, 128 wide, 4 layers, 4 heads (64 key dims, 64 value dims), MLP hidden 256,
4,096-token BPE vocabulary, context 256, trained from scratch on 1 GB of TinyStories V2 (248M tokens)
for 15,000 steps x 128 x 256 = 491M tokens, AdamW, one seed unless noted. Score: validation loss in
nats/token (lower is better). GBA speed and ROM are cost-model estimates for this small shape, not
measurements. Raw results: train/results/*.json.

Seed noise: baseline 1.860 / 1.862 / 1.862; ternary 2.104 / 2.096; routed experts 1.667 / 1.660.

| Arm | Loss | vs baseline | Est. GBA tokens/s | Est. ROM MB |
|---|---|---|---|---|
| Baseline, full precision | 1.860 | 0 | 4.8 | 1.4 |
| 8-bit weights (per-row scale) | 1.863 | +0.00 | 4.8 | 1.4 |
| Ternary (BitNet absmean), lr x3 | 2.104 | +0.24 | 12.8 | 0.7 |
| Ternary, lr x1 | 2.171 | +0.31 | 12.8 | 0.7 |
| Ternary, lr x10 | 2.051 | +0.19 | 12.8 | 0.7 |
| Ternary, 1.5x wider (192) | 1.920 | +0.06 | 7.9 | 1.1 |
| One-zero-in-four ternary (Sherry 3:4) | 2.177 | +0.32 | 14.0 | 0.7 |
| ...with faded full-precision residual (Arenas) | 2.209 | +0.35 | 14.0 | 0.7 |
| ...with 20% full-precision warm start | 2.172 | +0.31 | 14.0 | 0.7 |
| Top-8 attention | 1.881 | +0.02 | 4.8 | 1.4 |
| Sign keys + top-8 (query RMS-normalised, clipped STE) | 2.027 | +0.17 | 4.8 | 1.4 |
| One-zero-in-four keys + top-8 | 2.021 | +0.16 | 4.8 | 1.4 |
| Routed experts: 16 per layer, hard top-1, load-balance loss | 1.667 | -0.19 | 4.7 | 5.1 |
| Tables: token (4 layers) + hashed bigram + trigram, 65K rows, gated (oversized) | 1.615 | -0.25 | 4.7 | 69 |
| Tables: bigram 32K + trigram 32K rows at layer 0, no gate | 1.729 | -0.13 | 4.8 | 9.4 |
| ...same with a learned gate per row | 1.701 | -0.16 | 4.8 | 17.4 |
| Tables: bigram 16K + trigram 16K at layers 0 and 2, no gate | 1.721 | -0.14 | 4.8 | 9.4 |
| Tables: bigram only, 64K rows | 1.747 | -0.11 | 4.8 | 9.4 |
| Tables: trigram only, 64K rows | 1.785 | -0.08 | 4.8 | 9.4 |
| Tables: per-layer token rows only, gated | 1.777 | -0.08 | 4.8 | 5.4 |
| Routed experts + bigram/trigram 16K at layers 0 and 2 | 1.613 | -0.25 | 4.7 | 13.1 |
| Package: ternary + top-8 + 16 experts + 8 MB tables, 8-bit keys | 1.780 | -0.08 | 12.6* | 9.7 |
| Package, 1.5x wider (192) | 1.655 | -0.20 | 7.8* | 15.2 |
| Old package (sign keys, oversized tables), Sherry weights | 1.939 | +0.08 | 13.6 | 69 |
| Old package with free ternary weights | 1.881 | +0.02 | 12.5 | 69 |

* The cost model assumes the cheap ternary key scan; with 8-bit keys the scan needs multiplies and the
real figure will be somewhat lower.

Main readings:
1. Ternary costs 0.19-0.24 nats at equal shape and improves with a higher learning rate (x10 best so
   far; not yet at an optimum). A 1.5x wider ternary model is within 0.06 of the float baseline.
2. One-zero-in-four costs a further 0.07 over free ternary; Arenas and warm start did not help here.
3. Top-8 attention is nearly free. Quantising keys to signs or one-zero-in-four costs about 0.15.
4. Hard top-1 routing over 16 experts trained without collapse (usage entropy 0.96-0.99) and gave
   -0.19, reproducibly.
5. Hashed n-gram tables give -0.13 to -0.14 at 8 MB; bigram + trigram together beat either alone;
   gates help slightly but double storage.
6. Experts and tables partly stack (-0.25 together).
7. The cartridge-shaped package at 192 wide beats the float baseline by 0.20 nats at an estimated
   1.6x its GBA speed.
Caveats: TinyStories is repetitive, which favours n-gram tables; pilot scale only.
