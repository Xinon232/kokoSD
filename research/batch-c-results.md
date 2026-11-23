# Batch C results: chat-first pilots (19 Nov 2026)

55 of 56 runs finished (one ran out of GPU memory and is rerun in batch D). Every run: 128 wide,
4 layers, 4,096-token vocabulary, 15,000 steps of 128 x 256 tokens (491M tokens) on the mixed
corpus, trained at 50% chat / 25% facts / 25% stories unless named `mix_*`. Scores are validation
loss in nats per token, lower is better; "Overall" weights the three domains 50/25/25 whatever the
training mix. Two seeds of the float baseline differ by 0.014, so gaps under about 0.02 are noise.
Speed is the cost-model estimate with a two-step output head, at this pilot size.

## Findings

1. **Routing experts by the current word is the best single change and costs nothing to compute.**
   64 experts picked by token identity: 2.059 against the 2.660 baseline. It beats a learned router
   (2.174) and a frozen random one (2.314) and needs no router matrix on the GBA.
2. **Lookup tables and experts stack.** Tables alone 2.387, 16 experts alone 2.374, both 2.218.
   4-bit table rows with twice as many rows beat 8-bit rows at the same bytes (2.325 vs 2.387).
3. **Ternary weights are the main remaining cost, and no training trick here fixed it.** Plain
   ternary is 3.03 against 2.66 (+0.37). Learning rates x10/x20/x40, weight decay off late, Muon
   and per-row gain all land within 0.04 of each other. With experts the gap is +0.29, with the
   full package +0.31.
4. **Ternary weights wrecked fact memorisation.** The float model with experts and tables recalled
   98.9% of 30,000 drilled facts (94.4% in a wording it never saw for that person). The ternary
   package recalled 12.0%, barely above the 5.8% chance level. The float baseline without experts
   or tables recalled 6.8%, so the experts and tables are what hold the facts.
5. **1-bit attention keys are affordable.** Sign keys with a distance bias and twice the key
   width: 2.683 against 2.680 for the 8-bit top-8 reference.
6. **Small wins:** copy head 2.587 (-0.07), hash-addressed memory 2.594 (-0.07), Muon on float
   weights 2.619 (-0.04).
7. **No gain:** input word copied into every layer, distance bias on 8-bit keys, attention in only
   half the layers (same score, about 25% faster), key-bit routing.
8. **Two-step output head costs 0.07 at best** (similar-word groups plus a 128-word short list,
   2.729); random groups cost 0.16.
9. **Training mix moves loss between domains about one for one.** Chat-heavy: chat 2.20, stories
   2.55. Story-heavy: stories 1.96, chat 2.50. The 50/25/25 default stays.

## All runs

| Run | Overall | Chat | Facts | Stories | Speed (tok/s, est.) | ROM (MB) |
|---|---|---|---|---|---|---|
| s24_moe64_token | 2.059 | 1.678 | 3.116 | 1.765 | 4.38 | 17.12 |
| s20_moe64 | 2.174 | 1.794 | 3.252 | 1.856 | 4.11 | 17.16 |
| ref_moe16_tables | 2.218 | 1.837 | 3.303 | 1.896 | 4.3 | 13.13 |
| s61_facts_moe_tab | 2.241 | 1.855 | 3.332 | 1.921 | 4.3 | 13.13 |
| s22_moe16_token | 2.257 | 1.879 | 3.340 | 1.930 | 4.38 | 5.12 |
| s25_moe64_frozen | 2.314 | 1.952 | 3.388 | 1.963 | 4.11 | 17.16 |
| s50_tab_4bit_2x_rows | 2.325 | 1.922 | 3.421 | 2.034 | 4.37 | 9.38 |
| s51_tab_lr5 | 2.366 | 1.961 | 3.481 | 2.062 | 4.37 | 9.38 |
| ref_moe16 | 2.374 | 2.000 | 3.481 | 2.015 | 4.31 | 5.13 |
| s52_tab_lr20 | 2.376 | 1.961 | 3.500 | 2.083 | 4.37 | 9.38 |
| s53_tab_8bit_ref | 2.387 | 1.986 | 3.490 | 2.087 | 4.37 | 9.38 |
| ref_tables | 2.387 | 1.985 | 3.492 | 2.088 | 4.37 | 9.38 |
| s21_moe16_frozen | 2.429 | 2.062 | 3.534 | 2.057 | 4.31 | 5.13 |
| s23_moe16_keybits | 2.485 | 2.134 | 3.577 | 2.096 | 4.38 | 5.12 |
| ref_pkg_tern | 2.527 | 2.127 | 3.626 | 2.228 | 10.15 | 9.66 |
| s62_facts_pkg_tern | 2.535 | 2.130 | 3.640 | 2.240 | 10.15 | 9.66 |
| s31_copy_head | 2.587 | 2.256 | 3.632 | 2.203 | 4.38 | 1.38 |
| s32_hashmem | 2.594 | 2.225 | 3.714 | 2.210 | 4.31 | 9.38 |
| s17_fp_muon | 2.619 | 2.259 | 3.748 | 2.209 | 4.38 | 1.38 |
| ref_fp_seed2 | 2.646 | 2.282 | 3.782 | 2.238 | 4.38 | 1.38 |
| ref_fp | 2.660 | 2.289 | 3.790 | 2.272 | 4.38 | 1.38 |
| s07_topk8_bias | 2.664 | 2.291 | 3.776 | 2.299 | 4.35 | 1.38 |
| s30_x0mix | 2.665 | 2.297 | 3.792 | 2.273 | 4.38 | 1.38 |
| s26_tern_moe16 | 2.665 | 2.280 | 3.810 | 2.292 | 10.19 | 1.66 |
| s27_tern_moe16_gain | 2.667 | 2.273 | 3.823 | 2.300 | 10.19 | 1.66 |
| s33_att_upper_half | 2.673 | 2.295 | 3.804 | 2.298 | 5.53 | 1.31 |
| mix_even | 2.675 | 2.394 | 3.713 | 2.199 | 4.38 | 1.38 |
| s34_att_alternate | 2.676 | 2.307 | 3.809 | 2.282 | 5.53 | 1.31 |
| ref_topk8 | 2.680 | 2.304 | 3.804 | 2.309 | 4.38 | 1.38 |
| s06_sign_dk128_bias | 2.683 | 2.308 | 3.794 | 2.322 | 4.04 | 1.44 |
| s04_pq16_bias | 2.702 | 2.320 | 3.809 | 2.359 | 4.7 | 1.38 |
| s60_facts_fp | 2.702 | 2.325 | 3.847 | 2.312 | 4.38 | 1.38 |
| mix_chat_heavy | 2.703 | 2.201 | 3.856 | 2.554 | 4.38 | 1.38 |
| s00_sign_bias | 2.709 | 2.327 | 3.809 | 2.373 | 4.74 | 1.38 |
| s05_rescore32 | 2.716 | 2.338 | 3.846 | 2.343 | 4.69 | 1.38 |
| s43_head_kmeans_short128 | 2.729 | 2.352 | 3.857 | 2.354 | 4.24 | 1.38 |
| s35_att_upper_half_no_mixer | 2.729 | 2.357 | 3.846 | 2.356 | 5.59 | 1.31 |
| s42_head_kmeans | 2.749 | 2.370 | 3.881 | 2.376 | 4.38 | 1.38 |
| mix_story_heavy | 2.754 | 2.501 | 4.054 | 1.963 | 4.38 | 1.38 |
| s41_head_freq | 2.795 | 2.409 | 3.938 | 2.424 | 4.38 | 1.38 |
| s03_pq16 | 2.801 | 2.418 | 3.927 | 2.442 | 4.73 | 1.38 |
| s40_head_random | 2.824 | 2.431 | 3.964 | 2.470 | 4.38 | 1.38 |
| s15_tern_lr10_gain | 2.998 | 2.621 | 4.168 | 2.582 | 10.3 | 0.72 |
| s08_tern_sign_bias_gain | 3.004 | 2.622 | 4.141 | 2.632 | 12.55 | 0.72 |
| s11_tern_lr20 | 3.022 | 2.643 | 4.188 | 2.612 | 10.3 | 0.72 |
| s13_tern_lr40 | 3.023 | 2.645 | 4.192 | 2.610 | 10.3 | 0.72 |
| s12_tern_lr20_wdoff | 3.027 | 2.648 | 4.191 | 2.619 | 10.3 | 0.72 |
| s14_tern_lr40_wdoff | 3.027 | 2.648 | 4.193 | 2.618 | 10.3 | 0.72 |
| ref_tern_lr10 | 3.034 | 2.650 | 4.202 | 2.636 | 10.3 | 0.72 |
| s10_tern_lr10_wdoff | 3.035 | 2.650 | 4.198 | 2.642 | 10.3 | 0.72 |
| s18_tern_muon_lr3 | 3.048 | 2.667 | 4.232 | 2.628 | 10.3 | 0.72 |
| s16_tern_muon | 3.055 | 2.669 | 4.237 | 2.643 | 10.3 | 0.72 |
| s19_tern_muon_lr10 | 3.062 | 2.679 | 4.238 | 2.650 | 10.3 | 0.72 |
| s02_tern_sign_gain | 3.119 | 2.731 | 4.275 | 2.738 | 12.79 | 0.72 |
| s01_tern_sign | 3.152 | 2.764 | 4.316 | 2.764 | 12.79 | 0.72 |

## Fact recall (5,000 made-up people, 30,000 facts)

| Run | Recall, trained wording | Recall, wording unseen for that person | Chance |
|---|---|---|---|
| s60_facts_fp | 6.8% | 8.7% | 5.8% |
| s61_facts_moe_tab | 98.9% | 94.4% | 5.8% |
| s62_facts_pkg_tern | 12.0% | 10.2% | 5.8% |

## Batch D (running)

The best options combined (64 token-routed experts, 4-bit tables, copy head, top-8) in float and
ternary at 128, 192 and 256 wide; fact recall for each ternary width; and a diagnostic that keeps
one part of a ternary model 8-bit at a time (head, attention, expert layers) to find where the
ternary damage comes from.
