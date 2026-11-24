# slvDev/esp32-ai: 28.9M-parameter LM on an ESP32-S3 with Per-Layer Embeddings (https://github.com/slvDev/esp32-ai, 2026, owner slvDev)

Read in primary form: clone at commit 52ebcbb (2026-09-23): README, RESULTS.md, src/model.py, src/quantize.py, research/tinystories (README, train.py, all three ablation scripts), runtime/llm.h, both Hugging Face cards.

## What it is
An independent project that ports Gemma-3n-style Per-Layer Embeddings (PLE) to a microcontroller. A tiny dense transformer core (~0.56M params) is conditioned at every layer by one row of a 25M-parameter table indexed by the current token id; the table stays memory-mapped in flash and only 6 rows (~450 B at int4) are read per token. It ships a TinyStories model (28.9M stored params, 14.9 MB at int4, 9.88 tok/s) plus a second "Barista" Q&A model with separate input and output vocabularies.

## Mechanism
Core: pre-norm decoder, RMSNorm, RoPE, SwiGLU FFN, no biases, tied head. Deploy config: d=96, L=6, H=4, ffn_hidden=66, ple_dim P=128, seq 256, vocab 32,768 (25,353 rows reachable).

Per token (once, depends only on token id t):
```
e      = tok_emb[t]                                   # [D]
c_l    = RMSNorm_P( (W_mp e)[l] / sqrt(D) )           # W_mp: [L*P, D], one shared norm gain [P]
ple_l  = (c_l + sqrt(P) * table[t][l]) / sqrt(2)      # table: [V, L*P]
```
Per layer l, AFTER attention and FFN:
```
g = gelu(W_gate_l x)             # [P, D]
x = x + RMSNorm_D(W_proj_l (g * ple_l))   # [D, P]; elementwise product
```
Init: `ple_norm` gain zeroed so the branch starts as an exact no-op; residual-writing projections std 0.02/sqrt(2L); the sqrt(P) table scale is applied at run time, called "undocumented in Gemma's config but load-bearing" (model.py comment).

Arms (all with the dense core matched by solving ffn_hidden): `baseline`; `ple`; `ple_notable` (plumbing, no table); `fatembed` (same table budget as a V x (L*P) input embedding projected down once); `bigcore` (table budget spent on width).

## Numbers
All TinyStories validation loss, from RESULTS.md unless noted.
- Headline table (vocab 32,768, core ~559K, 2 seeds): baseline ppl 12.58; ple 11.41 (+0.098 nats, +/-0.006); fatembed 11.94 (+0.052). Per-layer beats bottom injection by 0.046 nats.
- Small-vocab table (vocab 4096, core 1.5M, 2 seeds): baseline 8.21; ple_notable 8.35 (-0.017); fatembed 8.26 (-0.006); ple 8.00 (+0.025); bigcore (2x core) 6.93 (+0.170).
- Table-width sweep (vocab 4096, FFN fixed 256, 1 seed, ple minus ple_notable): P=64 +0.045; 128 +0.071; 256 +0.094; 512 +0.087. Saturates near a 6M table.
- PTQ int4, group 64: degradation baseline +0.079/+0.088, ple +0.055/+0.061, fatembed +0.046/+0.050. In the shipping format (group 128, fp16 scales) the PLE edge is retained 121-126%. No QAT.
- int8 activations: 2.4793 -> 2.4796 nats (+0.0003) (HF card).
- Parameters: 28,867,968 = 25,165,824 table + 3,145,728 tied emb/head + 556,416 core (HF card; RESULTS.md says ~558K).
- Flash random 512 B row read 20.3 us; 6 rows ~0.12 ms/token.
- Profile, ms/token (current runtime): head 59.4, attention 20.5, PLE path 6.4, FFN 6.5, input 2.2, total 94.9. The head reads 2.43 MB of int8 per token.
- Barista (HF card): 8,903,808 params, d=128, input vocab 8,057, output 854 classes; 49.6 ms per forward. No quality score published.

## Smallest scale tested, and whether the result held there
Everything is tiny: core 0.56M-1.5M, d 96-128. The gain held at the smallest core (0.56M) but only with a large vocabulary: +0.098 at V=32,768 vs +0.025 at V=4,096 (different core sizes, so the vocab effect is confounded with core size and training length). Plumbing without a table is slightly harmful.

## Training recipe and gotchas
- Data: first 300 MB of TinyStories; BPE trained on that slice.
- AdamW, lr 1e-3, betas (0.9, 0.95), wd 0.1 (none on norms, tables, tok_emb), warmup 200, cosine to 0.1x peak, grad clip 1.0.
- Deploy runs: 5000 steps x batch 16 x seq 256 = 20.5M tokens. Small-vocab runs: 3000 x 32 x 512 = 49M tokens. These are very short, probably under one epoch, so all gaps are early-training gaps; rare-token table rows have seen few updates.
- Trained on Apple MPS, no determinism flags; mixing run cohorts gives +0.074 instead of +0.098.
- PLE plumbing is not cheap in dense terms: W_mp (73,728) + 6 x (gate 12,288 + proj 12,288) = 221,184 weights, about 40% of the 0.56M core (my arithmetic from model.py shapes). The core-matched comparison already charges for this.

## Mapping onto the GBA design
- ROM: V=16K x L=6 x P=128 = 12.3M entries: 12.3 MB at int8, 6.1 MB at int4, ~2.5 MB ternary. Fits easily; rows are contiguous sequential reads.
- Precompute: `c_l` depends only on the token id, so the whole `ple_l` vector can be folded into the ROM table offline. That removes the W_mp matvec (the repo computes it at run time) and one RMSNorm. My inference; the repo does not do it.
- Per layer at d=320, P=128: gate 41K + proj 41K ternary weights, about 131K cycles at 1.6 c/w; plus 128 8-bit multiplies (~1K cycles), a GELU LUT, and a d-wide RMSNorm (sum of squares + rsqrt LUT). Six layers is roughly 0.8M cycles, 16-27% of the token budget. P=64 halves this (sweep: +0.045 vs +0.071 nats at V=4096).
- RAM: one L*P row (768 B at int8), or read per layer straight from ROM: 0 extra.
- Changes needed: ternary/int8 QAT instead of int4 PTQ; integer RMSNorm; 16K vocab sits between the two tested regimes. Natural extension, untested here: key extra tables on hashed bigrams/trigrams, which adds rows (the axis that did not saturate) without growing the vocabulary.
- The profile confirms the head problem: a full scan is 63% of time even at d=96. Barista's reduced output alphabet (854 classes) roughly halves time per step; supports our two-level head.

## Verdict
Adopt per-layer token-keyed tables in gated form as the default, with the context projection folded into ROM. Evidence is small-scale and short-trained, so settle it with one experiment at our scale: d=320 ternary QAT, V=16K, arms baseline / gated PLE P=64 and 128 / additive per-layer row (x += table_l[t], zero dense cost) / hashed-bigram-keyed PLE, core-matched in cycles rather than parameters, trained to convergence.

## Could not verify
- No ablation of gated versus plain additive per-layer injection; the "multiply makes it conditioning" claim is a code comment only.
- No runs at d>=256, with ternary weights, with QAT, on non-TinyStories data, or trained beyond ~50M tokens. No sample-quality evaluation beyond perplexity.
- The vocab-size effect is confounded (core 1.5M vs 0.56M, different batch/seq/steps).
- `runs/` logs are not in the repo; numbers are the author's reports. Core count differs between sources (556,416 vs ~558K).
- Barista corpus and scores are unpublished. Gemma 3n's own PLE description was not read here.

## Links worth following
- https://huggingface.co/slvDev/esp32-ai-tinystories and https://huggingface.co/slvDev/esp32-ai-barista (model cards, read)
- https://ai.google.dev/gemma/docs/gemma-3n (PLE origin)

## Independent verification
Second reader, 2026-11-18. Read in primary form from raw.githubusercontent.com at HEAD (not pinned to 52ebcbb): RESULTS.md, research/tinystories/README.md, research/tinystories/train.py, the three scripts in research/tinystories/scripts/, src/model.py (arm definitions, `param_budget`, `make_model`). Not read: analyze.py, HF cards, runtime. The `runs/` logs are not published, so every number is the author's report; I could only check that the text, scripts and code agree with each other.

1. PLE vs same-core baseline, +0.098 nats (ppl 12.58 -> 11.41), 2 seeds, core ~559K, V=32,768, ~20M tokens: **confirmed**. RESULTS.md headline table and bullet ("0.098 nats (2 seeds, ±0.006)"); run_deploy_ablation.sh gives d=96, L=6, P=128, target-core 560000, 5000 steps x 16 x 256 = 20.48M tokens. Per-seed fp32 edges are given in the PTQ section as +0.101 / +0.095. Caveats: validation is 40 random batches of 16x256 (~164K tokens, same batches for every arm); training is far from converged; cohort-mixed archive gives +0.074.
2. Per-layer beats the same table budget as a wide input embedding by 0.046 nats (11.41 vs 11.94): **confirmed as reported, with an unflagged confound**. RESULTS.md headline bullet. But in model.py the `fatembed` arm differs from `ple` in more than injection site: its head is untied (`head_is_tied` is false for fatembed) while `ple` ties head and input embedding, and its 768-wide embedding is squeezed through one 768->96 linear whose 73.7K weights are charged to the core. So the 0.046 is "PLE vs one particular bottom-injection design", not a clean where-to-inject test. At V=4096 the same arm is -0.006 vs baseline.
3. Vocabulary dependence; V=4096 / core 1.5M gives PLE +0.025, plumbing only -0.017, 2x core +0.170: **partly**. The three numbers are exactly as in the RESULTS.md small-vocab table (2 seeds). What differs is the interpretation "vocabulary-dependent": run_small_vocab_ablation.sh passes no --ple-dim or --d-model, so it runs at the train.py defaults P=64, d=128, 49M tokens, batch 32 x seq 512, whereas the V=32,768 run uses P=128, d=96, 20M tokens. The table is 1.57M params vs 25.2M (8x rows AND 2x row width), the core is 2.7x bigger, and nats/token are not comparable across tokenizers (a V=4096 token carries less text). The author's own sweep shows P=128 at V=4096 yields +0.071 isolated table benefit vs +0.045 at P=64, so part of the "4x" is row width, not vocabulary. Also "2x core" is derived: bigcore core = 1.5M + 4096*6*64 = ~3.07M. "Plumbing is harmful" rests on -0.017 at 2 seeds with no stated noise for that cohort (headline cohort noise was 0.006), so weakly supported.
4. Row-width sweep +0.045 / +0.071 / +0.094 / +0.087 at P=64/128/256/512, V=4096, 1 seed: **confirmed**, numbers match research/tinystories/README.md table (RESULTS.md prints only 64, 256, 512). Clarifications: the quantity is `ple` minus `ple_notable`, not vs baseline; FFN is pinned at 256 and the core is NOT matched, so the dense plumbing (and core cost) grows with P in both arms; seed 0 only, so the 256 -> 512 dip (0.007) is within the ~0.006 seed noise seen elsewhere. "Saturates" is a fair reading of "stops improving" but not evidence of a decline. Saturation was measured only at V=4096 with ~49M tokens; no width sweep exists at V=32,768.

Overall: the numbers were transcribed correctly; the weak points are causal readings (vocab effect, per-layer vs bottom) that the experimental design does not isolate. All evidence is fp32 training, d<=128, TinyStories, <=49M tokens, 1-2 seeds, unpublished logs.
