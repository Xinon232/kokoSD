# Koko research report (overnight run, 18 Nov 2026)

## Status of this report

The overnight run stopped early: the weekly usage limit was reached about 15 minutes after the
large sweep started. What exists:

- 5 first-pass overview reports (`prelim/`), all complete.
- 28 deep-read source notes (`notes/`), out of roughly 250 planned.
- Emulator benchmarks (`bench-results.md`), complete and exact against the reference build.
- Training harness: only the data pipeline and quantizers were written. TinyStories is downloaded
  and tokenised (4,096-token vocabulary). No model code, no tests, nothing trained.

Not done: the independent verification pass (every verifier was cut off, so each claim below rests
on one reader), the per-cluster reports, the second "what is missing" round, and the design panel.
Treat the conclusions as a well-read first draft, not as verified.

## What changed in my thinking

### 1. The routed-expert block is the weakest part of the design
I benchmarked "pick 1 of 64 expert blocks per layer" because it stores a lot and runs little.
The literature does not support it at our size:

- A 47M-parameter study that replaced every ordinary block with lookup-style memory at equal
  parameters did worse than the plain model (12.77-13.96 vs 11.81 perplexity).
- The same work and the "million experts" paper both find that picking several small experts
  beats picking one large one.
- Tree-routed blocks cost 9-24% in perplexity against a plain block at 125M parameters, and the
  gap grows as models shrink.
- Nobody has published hard 1-of-N routing below about 50M parameters. Collapse onto a few
  experts is the known failure.

So routing moves from "the plan" to "one arm of an experiment". The default becomes a plain block.

### 2. Lookup tables keyed on recent tokens are the best-supported free capacity
- The ESP32 hobby project measured +0.098 nats from per-layer token tables on a model with only
  556K computed weights. This is the only result at our scale.
- Two large-model papers (Engram, Scone) and the GPT-2 speedrun all add tables indexed by a hash
  of the last two or three tokens, and agree on roughly +0.03 nats per tenfold increase in rows.
- On the GBA a table read plus a gate costs a few thousand cycles per layer, under 4% of a token.
- The limit is ROM: 32 MB holds about 125K rows of 256 eight-bit values, or 500K rows at 2 bits.

This also serves predictive typing directly, since "what follows these two words" is exactly what
the tables store.

### 3. Key-value memory layers only pay off when they are large and read often
- The original paper needed at least 65K slots and about 32-64 reads per token. One head reading
  8 slots was worse than no memory at all.
- I benchmarked exactly that weak setting (1 head, 8 reads): 162K cycles per layer. A setting
  the paper says works (4 heads) would cost about 650K cycles per layer.
- Gains are concentrated on fact recall and are proven only at 134M parameters and above.

Worth one memory layer in the middle of the model as an experiment, not one per layer.

### 4. Three-valued weights cost more than I assumed
- The small-model study shows about 1.5 times the perplexity of a full-precision model of the
  same shape, roughly 0.4-0.5 nats. "Twice the width" does not fully recover it.
- A 132M ternary model on TinyStories scored perplexity 58 against 28 at full precision.
- The one-zero-in-four scheme matched free ternary at 1B parameters. It is untested from scratch
  or below 700M.
- Helpful recipe pieces: a full-precision residual faded out during training; a full-precision
  warm start followed by a ternary stage at about ten times the learning rate (fragile: at a low
  rate it was 7-15% worse than ternary from scratch); median scaling at the smallest sizes.

Our kernel is about 4.7 times cheaper per weight than 8-bit multiply, so ternary probably still
wins per cycle, but this must be measured first: ternary at width W against 8-bit at about half W.

### 5. Attention may not be the best mixer at this size
- A 60K-parameter study found a simple gated recurrence beat the best transformer by a wide
  margin at full precision. It lost more under ternary (28% vs 5-9%) but stayed ahead.
- Evidence is thin: one size, one author, byte-level text. The Game Boy Color project's ternary
  recurrent model is also clearly behind a float transformer, though that was not a controlled test.
- Sign-bit attention keys grouped four at a time already exist in the literature for cache
  compression, with almost no quality loss on large models. Mixing only the top 8 positions was
  the best setting in the paper that introduced it. Neither is tested on tiny models.

A hybrid (recurrence in some layers, attention in others) goes on the ladder.

### 6. Learned-lookup inference is not a better kernel
Replacing matrix products with learned tables loses accuracy in every paper, and all of them keep
early layers exact. Our kernel is already the exact version of that idea. The GBA reference manual
puts the floor at about 6 cycles per ROM byte; we measure 6.4.

## Revised candidate design

| Part | Before | Now |
|---|---|---|
| Block | 1 of 64 routed experts | Plain block by default; routing, tree and memory variants as experiments |
| Free capacity | "maybe" tables | Hashed two- and three-word tables plus per-layer token tables, sized to fill the ROM |
| Memory layer | not planned | One, mid-model, as an experiment |
| Mixer | attention everywhere | Attention, with a recurrent hybrid on the ladder |
| Weights | free ternary | One-zero-in-four ternary with a faded residual, if within about 2% of free ternary |
| Output | two-step head | Two-step head with a short-list of the most frequent words at the first step |
| Vocabulary | 16K | Test 4K, 8K and 16K; phrase-level tokens are worth a look |

Speed for these shapes is already measured: 256 wide and 6 layers runs at about 5 tokens/s with
either block type.

## First GPU experiments, in order

1. **Baselines:** reproduce published full-precision TinyStories models at width 256, so our
   numbers are anchored to known ones.
2. **Weight format:** ternary, one-zero-in-four and 8-bit at equal GBA cycles, with a learning-rate
   sweep for each.
3. **Block type:** plain vs 1-of-64 vs several-of-many vs tree vs one memory layer, at equal cycles.
4. **Tables:** none vs token tables vs hashed word-pair tables, at equal ROM.
5. **Mixer:** attention vs hybrid vs recurrent, after ternary training.
6. **Data amount:** 1, 4 and 16 GB.

Each needs two or three seeds; the speedrun notes put seed noise at 1-2 thousandths of a nat.

## Data, distillation and evaluation

- **Data:** coherence appears at width 128 and above in TinyStories; SimpleStories puts the knee
  between 1.25M and 5M parameters. Mixing stories, dialogue and simple encyclopedia text beat
  dialogue alone in the one study that compared.
- **Distillation:** expect a few hundredths of a nat at best. A bigger teacher is not a better one.
  All evidence is from models above 40M.
- **Benchmarks:** the BabyLM pipeline (grammar, world knowledge, entity tracking) runs on any
  causal model. GPT-2 small scores about 83-84 on the grammar test; a 5-gram model scores 61 and
  small recurrent models about 70, which are the realistic comparisons. The world-knowledge test
  will sit near chance at our size.

## Key sources

Full notes with URLs are in `notes/` and `prelim/`. The ones this report leans on most:

- Memory Layers at Scale: https://arxiv.org/abs/2412.09764
- Large Memory Layers with Product Keys: https://arxiv.org/abs/1907.05242
- Mixture of A Million Experts: https://arxiv.org/abs/2407.04153
- Engram: https://arxiv.org/abs/2601.07372 and Scone: https://arxiv.org/abs/2502.01637
- Tree-structured feedforward networks: https://arxiv.org/abs/2604.08565
- ESP32 28.9M-parameter model: https://github.com/slvDev/esp32-ai
- BitNet b1.58 Reloaded: https://arxiv.org/abs/2407.09527
- Sherry: https://arxiv.org/abs/2601.07892 and Tequila: https://arxiv.org/abs/2509.23809
- 16-bit to 1.58-bit transition: https://arxiv.org/abs/2502.11895
- 60K-parameter mixer study: https://arxiv.org/abs/2609.29397
- Self-indexing key cache: https://arxiv.org/abs/2603.14224
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- TinyStories: https://arxiv.org/abs/2305.07759
- BabyLM evaluation pipeline: https://github.com/babylm/evaluation-pipeline-2025
- ChatGBC / Rei: https://github.com/crashtheuniverse/chatgbc
