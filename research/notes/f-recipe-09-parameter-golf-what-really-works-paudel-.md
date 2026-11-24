# Parameter Golf: What Really Works?  (https://arxiv.org/abs/2607.01517, 2026, Prashanna Mani Paudel & Shivanand Venkanna Sheshappanavar; BabyLM Workshop @ EMNLP 2026; v1 1 Jul 2026, v2 5 Sep 2026)

Provenance: I read the arXiv HTML, the authors' analysis repo (README + `results/SUMMARY.md`), the contest repo README and winning PR #2135, all through a fetch tool that returns a model-written digest, not raw text. Numbers below are as that digest reported them; none are quoted verbatim.

## What it is
An observational meta-analysis of OpenAI's Parameter Golf contest: best FineWeb-validation bits-per-byte (BPB) from an artifact (code + compressed weights) of at most 16,000,000 bytes, trained in under 10 minutes on 8xH100, with a separate 10-minute eval budget. From 2,037 pull requests the authors keep 1,810 de-duplicated rows, 1,478 scored, 1,430 "clean" (48 entries below 0.9 BPB dropped as suspected leakage), and tag each with 84 binary technique flags (17 quantization, 37 architecture, 4 tokenization, 16 training, 10 evaluation) by keyword rules on README / `submission.json`. It reports per-technique score differences over all entries and again inside the competitive tier (429 entries below 1.10 BPB). It is not a method paper: no ablations, no parameter counts.

## Mechanism
Impact metric: `Delta_k = mean(BPB | technique k absent) - mean(BPB | k present)`; positive = helps. Frontier version: same, restricted to BPB < 1.10, with 95% percentile-bootstrap CI, two-sided permutation p, Benjamini-Hochberg q. Bundling measured by lift.

Techniques as described (paper gives almost no hyperparameters):
- N-gram backoff: PPM-style fallback to shorter contexts; mixed as `P = lam*P_neural + (1-lam)*P_ngram`; no learned parameters.
- Winning entry's variant (PR #2135, 1.0565 BPB): "token-only n-gram tilt", `TOKEN_ORDER=16, TOKEN_THRESHOLD=0.800, TOKEN_BOOST=2.625`: when the n-gram hint clears the threshold, `logit[hint] += boost`, then softmax. Strictly causal; precompute runs inside the eval timer; repo files are named `online_ngram_state.c` / `online_ngram_tilt.py`.
- Value residual: layer-1 attention value vectors are added into every later layer's values.
- SWA: average weights along the training trajectory (one leaderboard entry lists "SWA(0.4)"). EMA similar.
- Gated DeltaNet: attention replaced by a gated delta-rule recurrence with fixed-size state.
- Parallel residuals: attention and MLP read the same input. Depth recurrence: loop some layers (PR #2135 loops layers 3-5).
- Sliding-window eval: stride 64, 960 tokens of left context, about 16x forward passes.
- int6 QAT-STE at about 0.75 byte/weight freed about 4 MB, spent on MLP 2x -> 3x and 9 -> 11 layers.

## Numbers
All at the contest scale (16 MB artifact, FineWeb BPB).
- Baseline 9 layers, 512 wide, SP-1024 vocab, tied embeddings, int8 + zlib: 1.2244. Best verified in paper 1.058 (-13.6%); repo leaderboard top now 1.0565. Phases: ~1.12, ~1.064, 1.058.
- Frontier table (`results/SUMMARY.md`, n=429): N-gram backoff n=23, +0.080 [0.060, 0.098], p<0.0001; Gated DeltaNet n=31, +0.032 [0.021, 0.042]; Value residual n=41, +0.032 [0.013, 0.051]; PPM byte mixture n=47, +0.029 [0.014, 0.045]; SWA n=109, +0.025 [0.014, 0.036]; zstd n=101, +0.026; SP1024 vocab n=82, +0.024; Complementary training n=9, +0.036 [0.010, 0.057]; Crown-Q n=6, +0.039 (CI crosses 0); KV sharing n=7, +0.032 (q=0.093).
- Frontier, text only: GPTQ, Brotli, depth recurrence and common TTT variants become small but significantly negative (values not given).
- All-data (Table 3, n=1,430): N-gram backoff +0.180 (n=28); sparse attention gate +0.159 (71); GPTQ +0.155 (664); n-gram cache +0.147 (218); TTT global/phased +0.146 (101); int6 +0.130 (968); SP8192 +0.115 (351); CaseOps +0.135 (135); depth recurrence +0.079 (405); ternary -0.013 (188); binary -0.071 (42); universal transformer -0.098 (38); state-space model -0.156 (60); int8-only -0.200 (258).
- Sliding-window eval: about 0.032 BPB with no architecture change.
- Stacks: 222 distinct combinations of the 12 commonest frontier techniques; depth recurrence x parallel residuals lift 1.80.
- Contest README: ternary entry, 73.7M weights, 1.1570; 1-bit, 106M weights, 2 h training (non-record), 1.1239.

## Smallest scale tested, and whether the result held there
One scale only: about 16 MB of compressed weights (roughly 20-35M int6 weights by my arithmetic, not stated), 512-wide, 9-11 layers, 10 minutes of training. Nothing at Koko's compute scale. The frontier-restricted n-gram result held with n=23 only.

## Training recipe and gotchas
- Deltas are correlations confounded by adoption date: late techniques ride on better stacks. All-data GPTQ +0.155 flips negative inside the frontier.
- The n-gram effect is, at least in the winning entry, an online structure built from the eval stream, with a 10-minute eval allowance. It rewards long-range repetition in FineWeb documents, not stored knowledge.
- BPB scoring rewards calibrated mixing; a chatbot samples, so the gain in sample quality is unmeasured.
- Ternary's deficit is per byte under a size cap; compute per token was unconstrained.
- Keyword flagging is noisy; "value residual" was removed in one strong entry (PR #1218, 1.0979).
- A 10-minute training budget favours fast-converging tricks; several-hundred-hour QAT may rank them differently.

## Mapping onto the GBA design
- N-gram path: two forms. (a) ROM tables keyed by hashed context: one hash, one sequential ROM read, and either a few logit adds (tilt form: add a stored boost to 1-8 candidate tokens) or a bias on the class and member logits of the two-level head. No multiplies. The tilt form avoids the divide that linear probability mixing needs. (b) Online cache over the 256-token ring buffer: a small hash table in slow RAM (e.g. 1-2K entries x 3 bytes) mapping context hash to last-seen next token; cost one lookup and one add per token. This also gives typing suggestions for names the user introduced.
- Value residual: keep layer-1 values for the 256 positions (already stored for layer 1) and add them to each later layer's top-8 mix: 8 x d_v extra adds per layer, or fold as `v_l + a*v_1` at write time, zero extra RAM if done at write.
- SWA: free at inference. With ternary QAT, average latent weights, re-ternarise, then re-estimate norm statistics.
- Gated DeltaNet: a d_k x d_v state per head updated densely each token; at 64x64 that is about 8K 8-bit MACs (about 60K cycles) and 4-8 KB RAM per head. Affordable but competes with attention for the 32 KB fast RAM.
- Depth recurrence: doubles dense compute for no ROM saving we need. Not useful here.
- Ternary vs int6: not transferable; our kernel makes ternary about 4.7x cheaper per weight than 8-bit MAC and ROM is abundant.

## Verdict
Adopt the n-gram path in both forms as a logit tilt, and adopt value residuals and weight averaging as cheap defaults; test Gated DeltaNet only if attention RAM becomes the binding limit; skip depth recurrence, GPTQ and TTT. Settling experiment: on the Koko student, ablate {ROM n-gram tilt, online ring-buffer cache, value residual, SWA} one at a time at fixed cycles per token, reporting held-out bits per byte and repetition-sensitive chat samples.

## Could not verify
- Raw paper text: I saw only tool digests. Per-technique frontier values for GPTQ, depth recurrence, TTT, ternary, int6 were not found (said to be in Figure 4).
- N-gram backoff order, hashing, table size and whether tables are stored or built online are not in the paper; the online reading rests on file names in PR #2135. No isolated BPB gain for the n-gram tilt was reported there.
- No parameter counts anywhere; AsymLogit formula not found.
- Whether the paper's final version reports the 1.0565 entry.

## Links worth following
- https://github.com/PMP56/pmgolf-analysis (`results/SUMMARY.md`, `data/technique_taxonomy.csv`)
- https://github.com/openai/parameter-golf (leaderboard; ternary 73.7M entry)
- https://github.com/openai/parameter-golf/pull/2135 (`online_ngram_tilt.py`, `online_ngram_state.c`)
