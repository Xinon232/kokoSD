# Batch D results: combined package (19 Nov 2026)

16 runs, same budget and corpus as batch C (128 wide unless named, 4 layers, 491M tokens). The
package is 64 experts routed by token identity, 4-bit two- and three-word tables with 32,768 rows,
table learning rate x5, copy head, top-8 attention. Ternary runs add learning rate x10 and per-row
gain. Batch C baseline: float 2.660, old ternary package 2.527.

## Findings

1. **The combined package in ternary beats every float run from batch C.** 128 wide: 2.167 (two
   seeds: 2.167, 2.169). Float package: 1.988.
2. **Width buys the ternary gap back.** 192 wide ternary 1.993 matches the 128-wide float package;
   256 wide reaches 1.900.
3. **Fact recall is fixed by the new package, at every width.** Ternary 128 wide: 98.7% trained
   wording, 95.4% unseen wording. 192 and 256 wide: 99.9% / 99.6%+. The old package (16 learned
   experts) stays near chance, and a marker token does not help it. Token-routed experts are what
   hold facts.
4. **1-bit attention keys cost 0.01** in the 192-wide ternary package (2.003 vs 1.993).
5. **Ternary damage is spread across the model.** Keeping one part 8-bit in a plain ternary model
   (3.034; float 2.660) recovers only part of the gap: expert layers 2.859, attention 2.883, head
   2.907. No single part is worth keeping 8-bit for likelihood; width is the better trade.

## All runs

| Run | Overall | Chat | Facts | Stories | Fact recall (trained / unseen wording) | Speed (tok/s, est.) |
|---|---|---|---|---|---|---|
| pkg_tern_256 | 1.900 | 1.537 | 2.879 | 1.647 |  | 3.74 |
| pkg_tern_256_facts | 1.908 | 1.546 | 2.885 | 1.656 | 99.9% / 99.7% | 3.74 |
| pkg_fp | 1.988 | 1.604 | 3.017 | 1.727 |  | 4.37 |
| pkg_tern_192 | 1.993 | 1.628 | 2.981 | 1.735 |  | 5.8 |
| pkg_fp_facts | 2.001 | 1.613 | 3.035 | 1.742 | 99.9% / 94.3% | 4.37 |
| pkg_tern_192_sign | 2.003 | 1.633 | 2.998 | 1.751 |  | 5.83 |
| pkg_tern_192_facts | 2.019 | 1.649 | 3.017 | 1.764 | 99.9% / 99.6% | 5.8 |
| pkg_tern | 2.167 | 1.789 | 3.189 | 1.902 |  | 10.26 |
| pkg_tern_seed2 | 2.169 | 1.789 | 3.194 | 1.905 |  | 10.26 |
| pkg_tern_facts | 2.190 | 1.810 | 3.214 | 1.926 | 98.7% / 95.3% | 10.26 |
| facts_old_pkg_int8_head | 2.417 | 2.022 | 3.507 | 2.120 | 32.8% / 27.8% | 10.15 |
| facts_old_pkg_int8_mlp | 2.481 | 2.083 | 3.601 | 2.155 | 5.6% / 5.1% | 10.15 |
| facts_old_pkg_marker | 2.530 | 2.126 | 3.631 | 2.235 | 8.9% / 7.0% | 10.15 |
| tern_int8_mlp | 2.859 | 2.475 | 4.004 | 2.481 |  | 10.3 |
| tern_int8_att | 2.883 | 2.527 | 4.027 | 2.449 |  | 10.3 |
| tern_int8_head | 2.907 | 2.540 | 4.052 | 2.497 |  | 10.3 |

The `int8_*` runs are diagnostics; their speed column does not account for the 8-bit part.
