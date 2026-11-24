# Final runs (20 Nov 2026)

Four models, 120,000 steps each (3.9 billion tokens, about four passes over mix1: Koko's own data 93M tokens,
chat 158M, Simple Wikipedia 22M, stories 240M; 16,384-token vocabulary). Training mix koko .4 / chat .25 /
facts .1 / stories .25. Overall = koko .4 / chat .2 / facts .2 / stories .2, nats per token, lower is better.
Speed and ROM are cost-model estimates, not measurements on the GBA engine.

| Run | Overall | bits/char | Koko | Chat | Facts | Stories | tok/s | ROM MB | Train h |
|---|---|---|---|---|---|---|---|---|---|
| h_koko_L8 (8 layers, 128 experts x 128) | 2.188 | 0.762 | 2.105 | 1.620 | 3.505 | 1.607 | 4.32 | 29.2 | 5.4 |
| h_L6_E192 (6 layers, 192 experts x 128) | 2.201 | 0.766 | 2.117 | 1.626 | 3.523 | 1.623 | 5.68 | 31.1 | 5.1 |
| h_turbo_L4_E256 (4 layers, 256 experts x 128) | 2.268 | 0.789 | 2.178 | 1.697 | 3.594 | 1.692 | 8.30 | 29.1 | 4.4 |
| h_koko_L8_fp (full-precision twin) | 2.068 | 0.720 | 1.962 | 1.441 | 3.497 | 1.480 | – | (80.8) | 4.9 |

- No overfitting: validation fell at every checkpoint in every run; the final weights are the best weights.
- 2-bit weights cost 0.12 overall against the full-precision twin (0.18 on chat, 0.01 on facts).
- One pass (shape check) to four passes: 8-layer 2.270 -> 2.188.
- h_L6_E192 is 31.1 MB, not the 28.1 MB estimated before launch; it leaves under 1 MB for code and font.
- Files: artifacts/models/<run>/{model.pt,best.pt,result.json,log.jsonl}, artifacts/models/code_as_run.tgz,
  artifacts/data/{koko_data,koko_raw,mix1}.tar.zst.
- tools/chat_local.py <run> "prompt" ... generates replies on CPU from a downloaded model.
