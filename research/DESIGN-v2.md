# Koko design v2 (19 Nov 2026)

Built from 31 pilot runs (`pilot-results.md`), the emulator benchmarks (`bench-results.md`) and the
focused research round: 92 sources across five reports (`f-ternary`, `f-tables`, `f-data`,
`f-recipe`, `f-eval`, each `-report.md`). Only 9 of those sources had an independent second reader,
and no published source tests a model like ours (ternary, routed, tabled, under 30M weights), so
outside numbers are hypotheses. Our own pilots are the only evidence at our scale, and they are
small (128-192 wide, children's stories).

## Decided (pilot evidence, backed by the reading)

| Choice | Evidence |
|---|---|
| Free ternary weights; drop one-zero-in-four | Pilot: 0.07 nats better. One paper found a similar restricted format worse at 100M. |
| Widen to pay for ternary | Pilot: 1.5x wider ternary is within 0.06 of float. Four routes agree on about 2.2-2.5x weights. |
| High ternary learning rate | Pilot: x1 2.171, x3 2.104, x10 2.051. Best not yet found. |
| Top-8 attention | Pilot: 0.02 nats. |
| 8-bit attention values | One study found 1-bit value mixing catastrophic. |
| Hard-routed experts, one per token | Pilot: -0.19 nats, two seeds, no collapse. |
| Hashed two- and three-word input tables, plain rows | Pilot: -0.13 to -0.14 at 8 MB; gates double storage for 0.03. |
| Experts and tables together | Pilot: -0.25. |
| 8-bit rows in lookup tables | Ternary embeddings broke a 2B model in one project. |
| Chat and question data in training from step one | One study: 87% vs about 0% held-out answers. |
| No distillation | No gain shown at our size. |
| Drop: faded residual, float warm start | Pilot: no gain. |

## Open, to settle in the next pilot batch

1. **Attention keys.** Sign keys cost 0.15 nats; 8-bit keys do not fit the cycle budget (estimated
   2-3M cycles per token). Test: per-head strength, distance bias, both; then 8-bit query against
   1-bit keys through small tables.
2. **Ternary learning rate and schedule.** x10, x20, x40, each with and without weight decay turned
   off for the last third.
3. **Expert count and routing.** 16 vs 64 experts; learned router vs frozen random router vs routing
   by token identity (the last two would save about 180K cycles per token).
4. **Per-row gain after matrices that feed hard decisions** (key signs, router, head).
5. **Scaled copy of the input word at every layer.**
6. **Two-step output head.** Flat head as the upper bound, then random groups, similar-word groups,
   and similar-word groups with a frequent-word short-list.
7. **Table details.** Better hash; 8-bit vs 4-bit rows at equal bytes; table share of ROM.
8. **Fact capacity.** How many drilled facts the ternary routed model holds, with and without a
   marker token on factual text.
9. **Attention in fewer layers**, with a cheap mixer in the others.
10. **Muon optimiser** against AdamW under ternary training.

## Target shape for the first full-size run

Subject to the batch above: 256 wide (384 if RAM tiling works), 6 layers, 64 experts per layer,
16,000-token vocabulary with the two-step head, 8-16 MB of two- and three-word tables, context 256.
Benchmarked speed for the 256-wide shape: about 5 tokens/s.

## Data plan

- Generate with an open model of about 8B parameters at temperature 1.0; seeded templates across
  stories, dialogues and question-answer pairs; check that no single opening dominates.
- Filter by variety of word sequences, not by reading age.
- Curated fact list: at least five wordings per fact, about 300 exposures, marker token on factual
  text.
- Dialogues: seed, short scene, dialogue; role tokens; a greetings and identity grid; a fixed pool
  of names.
- Plan unique tokens at a quarter or more of total training tokens, at most about 16 passes.

## Evaluation plan

- Rerun GPT-2 small in our own harness; published numbers differ by up to 13 points with protocol.
- Headline scores: bits per byte on held-out simple English, and whole-word accuracy.
- Predictive typing: presses per character on the real GBA button layout, plus top-1/top-3 word
  recall. This is where Koko has the best chance of matching GPT-2 small.
- Judged conversations: forced choice between whole dialogues, both orders, with the judge audited
  against about 200 hand labels. About 200 prompts resolve only 7-10 point gaps, so small design
  choices are decided on likelihood.
- Skip HellaSwag, ARC and similar as headlines; tiny models sit near chance.

## Known gaps

- Verification was thin: 9 of 92 sources re-read. The evaluation report has none.
- The GBA engine for this design exists only as a random-weight benchmark.
- The key scan for 8-bit queries against 1-bit keys is not yet benchmarked.
- TinyStories flatters lookup tables; the table gain on chat data is unknown.
