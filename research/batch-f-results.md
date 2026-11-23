# Batch F: cartridge space (19 Nov 2026)

Baseline is e_256_sign_half: 256 wide, 4 layers, attention in layers 2 and 3, 64 token-routed experts
with 512 hidden units, 4-bit tables (32,768 rows) at layers 0 and 2, ternary weights, 1-bit keys.
Pilot scale: 4,096-token vocabulary, 491M training tokens, mix0. Overall = 50/25/25 chat/facts/stories,
nats per token, lower is better. Speed and ROM are cost-model estimates for the pilot shape.

| Arm | Overall | Chat | Facts | Stories | tok/s | ROM MB |
|---|---|---|---|---|---|---|
| baseline E64 H512 | 1.951 | 1.583 | – | – | 4.85 | 33.3 |
| tables at layer 0 only | 1.944 | 1.575 | 2.932 | 1.694 | 4.86 | 25.3 |
| tables with 16,384 rows | 1.958 | 1.592 | 2.937 | 1.711 | 4.85 | 25.3 |
| E128 H256 | 1.959 | 1.586 | 2.942 | 1.719 | 6.81 | 33.3 |
| E32 H512 | 1.995 | 1.626 | 2.986 | 1.742 | 4.85 | 25.3 |
| E64 H256 | 2.007 | 1.632 | 3.005 | 1.757 | 6.81 | 25.3 |
| E128 H128 | 2.020 | 1.643 | 3.019 | 1.776 | 8.54 | 25.3 |
| E128 H128 + tables at layer 0 | 2.008 | 1.644 | 2.993 | 1.751 | 8.55 | 17.3 |
| E256 H64 | stopped at step 3,700 of 15,000 | | | | 9.78 | 25.3 |

Findings
- The second copy of the tables (layer 2) is worth nothing: dropping it saves 8 MB at no cost.
- Quality follows total expert weights (experts x hidden units). Halving the total costs 0.04 to 0.07.
- At a fixed total, twice as many experts of half the size costs about 0.01 and is 40% faster on the
  GBA, because only one expert runs per token.
- E256 H64 trained ten times slower: with 256 experts and a 4,096-token vocabulary the routing is lopsided
  and the code fell back to a per-expert loop. The padded-batch limit was raised afterwards (model.py).
- One seed per arm; seed noise measured earlier is about 0.01.
