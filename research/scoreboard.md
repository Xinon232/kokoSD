# Koko scoreboard (20 Nov 2026)

Model on the cartridge: i_kd. Scripts and raw results: scoreboard/ (models.py, eval_text.py, eval_qa.py, qa_sets.py,
gen_samples.py, judge/, build_page.py; text.json, qa.json, qa_extra.json, samples.json, judging.json, size_speed.json).

## Text prediction, bits per character (200 held-out stories, 200 held-out Simple Wikipedia passages; lower is better)

| Model | Stories | Encyclopedia |
|---|---|---|
| Koko (cartridge, 2-bit) | 0.545 | 1.178 |
| Koko full-precision twin | 0.496 | 1.176 |
| GPT-2 small | 0.862 | 1.003 |
| SmolLM2-135M | 0.695 | 0.898 |
| SmolLM2-135M-Instruct | 0.731 | 0.981 |
| TinyStories-33M | 0.435 | 3.119 |
| TinyStories-8M | 0.513 | 2.834 |
| TinyStories-1M | 0.781 | 2.892 |

## Questions (greedy answer, pattern-checked)

| Model | Capitals /40 | General /60 | Sums /30 |
|---|---|---|---|
| Koko (cartridge, model only) | 35 | 22 | 14 |
| Koko (cartridge, with fact table and calculator) | 35 | 22 | 30 |
| SmolLM2-135M-Instruct | 39 | 38 | 6 |
| SmolLM2-135M | 36 | 43 | 14 |
| GPT-2 small | 3 | 12 | 0 |
| TinyStories-33M | 0 | 2 | 0 |

SmolLM2-135M-Instruct sums with 140 tokens to show its working: 11/30.

## Blind judging (3 AI judges per item, 20 items per pairing)

| Pairing | Items won Koko-other | Votes | Unanimous |
|---|---|---|---|
| chat: Koko vs GPT-2 small | 20-0 | 60-0 | 20/20 |
| chat: Koko vs SmolLM2-135M-Instruct | 18-2 | 56-4 | 18/20 |
| stories: Koko vs GPT-2 small | 19-1 | 58-2 | 19/20 |
| stories: Koko vs SmolLM2-135M-Instruct | 9-11 | 28-32 | 15/20 |
| stories: Koko vs TinyStories-33M | 15-5 | 44-16 | 17/20 |
| stories: Koko vs TinyStories-8M | 16-4 | 51-9 | 17/20 |

## Size and speed

| Model | Parameters | float32 bytes | tokens/s on this 4-core CPU |
|---|---|---|---|
| GPT-2 small | 124,439,808 | 497,759,232 | 64.7 |
| SmolLM2-135M | 134,515,008 | 538,060,032 | 41.7 |
| SmolLM2-135M-Instruct | 134,515,008 | 538,060,032 | 47.4 |
| TinyStories-33M | 68,514,048 | 274,056,192 | 128.8 |
| TinyStories-8M | 19,702,528 | 78,810,112 | 243.5 |
| TinyStories-1M | 3,745,984 | 14,983,936 | 366.7 |
| Koko (native cartridge engine, 1 thread) | 72.1M 2-bit + 21.0M table values | ROM 33,296,384 | 1471.4 |

Caveats: one run each; AI judges (Claude), not people; chat and story prompts are in Koko's training style; Koko and the TinyStories models
were trained on the same public sources as the held-out text; TinyStories/GPT-2 do not follow instructions; GBA speed is from mGBA.
Training facts for other models: SmolLM2-135M model card (2T tokens, 64 H100), TinyStories paper (under a day on one GPU), OpenAI GPT-2 announcement (40 GB, 8M pages).
