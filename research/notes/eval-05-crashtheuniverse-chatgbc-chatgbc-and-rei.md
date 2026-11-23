# ChatGBC and Rei: a ternary minGRU language model in SM83 assembly on Game Boy Color  (https://github.com/crashtheuniverse/chatgbc, 2026, owner "crashtheuniverse", MIT)

Primary read: clone at commit 319d8d0 (2026-09-29) - README, VERSIONS, CHANGELOG, the four docs, py/{train,fit,model5,twin5}.py, src/{sweep,matvec3,cls4,name}.asm headers.

## What it is
A 374K-parameter TinyStories model and a conversational sibling ("Rei") on a 2.1M-cycle/s 8-bit CPU with no multiply. v0.3 ported karpathy's stories260K transformer with 4-bit weights; v0.4 trained a model for the machine (minGRU core, hard-routed ternary experts); v0.9 rewrote every kernel as tables and skips, bit-exact; v1.0 added Rei and "engine opcode" tokens. It is Koko's closest published relative: same partial-sum-table kernel, hard MoE, integer-only QAT.

## Mechanism
Shape: dim 64, 3 layers, 4 experts x 176, vocab 1024 BPE, tied ternary embedding/classifier. Per layer:
```
xn=rms(x); z=sigmoid(wz xn); ht=wh xn; h=(1-z)h+z*ht; x=clip(x+wo h, +-16)
xf=rms(x); e=argmax(router xf); x=clip(x+w2[e] relu(w1[e] xf)^2, +-16)
```
Integers: int8 activations/state, exact int16 row sums, `out=sat8(shr_round(acc,s_row))`.
- Weights: per-row `scale=2^round(log2(mean|w|))`, `q=clamp(round(w/scale),-1,1)*scale`, STE. Classifier and router use one scale per tensor, so argmax needs no rescale.
- Kernel, output-major: per input vector build 22 tables of 27 sums (381 cycles each); a ROM byte is the table address; a row is 22 lookups in a register pair with requant fused: 302+8s cycles. Classifier: blocks of 4 (81 sums), only best and runner-up kept.
- Sparse w2: the ReLU^2 loop lists nonzero activations; w2 is stored per input column as +1/-1 output lists; one add per nonzero pair.
- Codomain audit: a stage with few distinct outputs becomes a table (ReLU^2 page, gate table, pre-requantized embeddings).
- Decode: greedy, skipping any token that completes an already-emitted 4-gram (<=8 rejects, 176-token history).
- Prompt tokens stop at the last layer's state update.
- Rei opcodes: `<SN>` store last typed word, `<N>` print it, `<W>` echo last word, `<NK>` header while a name is known. The model learns when; the engine holds the text.

## Numbers
Emulator DIV/TIMA counters; quality on the bit-exact twin, 200 held-out TinyStories V2 stories, teacher-forced.
- VERSIONS: cycles/token 12.92M (v0.3) -> 2.26M (v0.4.1) -> 962,790 (v1.0) = 0.46 s/token, 3.46 chars/token.
- Bits/char shipped: 1.149 (v0.3), 1.126 (v1.0). Float-activation eval of the same checkpoints: 0.876 and 1.123. Integer loss 31% (post-hoc 4-bit, 24-token window) vs 0.3% (QAT).
- ROADMAP census: classifier 423,496 -> 246,712; w2 344,696 -> 67,624; norms 326,024 -> 162,720; total 2,164,704 -> 973,032.
- THE-MODEL: 170,752 MACs/token for 373,952 parameters; classifier 25% of the token.
- fit.py: 17.4% of ReLU^2 outputs nonzero (6,291 tokens); 34-40% of w2 weights zero.
- cls4.asm: 3.55% of tokens retry once, 0.23% more (15,500 tokens).
- Post-hoc ternary: 2.9% top-1.
- Rei review: 401 lines judged ok 158, off 112, garbled 86, wrong 45.
- Training: 59M tokens, 20,000 steps, 3.5 h on a laptop GPU.

## Smallest scale tested, and whether the result held there
Everything is 260K-374K parameters at dim 64; nothing larger. QAT held to 0.3% there, but the ternary minGRU scores 1.123 bpc against 0.876 for the float transformer (not a controlled comparison). Rei is weak: 39% of probed lines sensible.

## Training recipe and gotchas
AdamW lr 3e-3, wd 0.01, betas (0.9, 0.95), OneCycle pct_start 0.1, clip 1.0, batch 96, seq 96. Arenas residual `W_eff=Q(W)+lam*W`: lam=1 for 10% of steps, cosine to 0; grade at lam=0. MoE: hard top-1 forward, output times the chosen router probability, aux `0.01*E*sum_e(frac_e*meanprob_e)`. State fake-quantized to int8 every step.
Gotchas (code comments):
- An unclamped stream grew to |x|=233 against embeddings of 2.5 and the ROM emitted `<unk>` forever; clamp to +-16 on a fixed 127-step grid in training.
- A 4x coarser stream grid cost 1.146 vs 1.126 bpc.
- Halved block sums with round-half-up biased rows ("little little little"); keep sums exact.
- Sherry 3:4 zeros on wz/wh "killed recall while costing nothing in loss".
- 4-bit recurrent state: recall 0/12.
- Emulator zeroes RAM; hardware does not.

## Mapping onto the GBA design
- Their kernel is ours with blocks of 3, ~4 cycles/weight against our 1.6. Adopt output-major order, fused requant, one table build shared by matrices reading the same vector (the router rides on w1's tables).
- A flat head is linear in vocab: 16K x 256 would be ~6.7M GBA cycles (my arithmetic), so the two-level head is mandatory. They priced one (32x32) but never trained it.
- Sparse w2: pairs are ~11% of weights. If an ARM pair costs 6-8 cycles (my estimate) that is ~0.8 cycles/weight against 1.6: perhaps 2x, not their 5x, and the lists roughly double w2's ROM.
- Prefill shortcut for attention: prompt tokens need only K/V writes in the last layer.
- Opcode tokens: a few hundred bytes of code, tens of bytes of RAM, exact name/echo copying.
- Copy the method: bit-exact Python twin, per-stage census ROM, a price function within 1% of the census, bits/char as the metric against GPT-2.

## Verdict
Adopt the engineering (twin, census, stream clamp, Arenas schedule, one-scale router/head, no-repeat greedy, opcodes, prefill shortcut). Test sparse w2 and Sherry zeros on attention K/Q. Skip minGRU as sole memory. Settling experiments: ReLU^2 nonzero rate and ARM pair-loop cost at dim 256; 4 vs 64 experts under this loss; Sherry vs free ternary on K/Q judged by a recall probe, not loss.

## Could not verify
docs/LOG.md, FACTS.md, DESIGN.md and docs/audit are cited but absent, so ablations (minGRU vs attention, MoE vs dense, Arenas) are not public. Rei's corpus ("friend5c"), its generator and her bits/char are unpublished. Real-hardware evidence covers v0.3 only. Secondary (fetched page summaries): 40 stars; gbc-transformer's 169 s/token.

## Links worth following
- Sherry / Arenas: https://arxiv.org/abs/2601.07892
- minGRU: https://arxiv.org/abs/2410.01201
- https://github.com/maddiedreese/gbc-transformer
