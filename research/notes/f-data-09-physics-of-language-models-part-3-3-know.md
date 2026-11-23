# Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws  (https://arxiv.org/abs/2404.05405, 2024, Zeyuan Allen-Zhu & Yuanzhi Li, Meta FAIR / MBZUAI)

Read in primary form: arXiv HTML, main body plus appendices A-E (read through a summarising fetch tool in two 100K-character halves, so figures were not seen, only their captions and the text).

## What it is
A controlled study of how many bits of factual knowledge a transformer stores per parameter. The authors generate synthetic (name, attribute, value) knowledge whose information content is known exactly, train GPT2/LLaMA/Mistral-style models from scratch on it, and convert the final loss into a lower bound on stored bits. Dividing by parameter count gives a "capacity ratio" R in bits/param. They then vary exposures per fact, architecture, MLP size, quantization, mixture-of-experts sparsity and the share of junk data. It is not a language-quality paper: nothing here measures fluency.

## Mechanism
Datasets
- bioS(N): N people, 6 attributes each (birth date, birth city, university, major, employer, work city); 50 sentence templates per attribute, re-sampled and re-ordered at every exposure. About 47.6 bits per person excluding the name (Remark 4.4). Variants: bioS-simple (one fixed biography), bioR (40 LLaMA2 rewrites per person).
- bioD(N,K,C,D,L,T): N names, K attributes, each value = C chunks drawn from a per-attribute set of D random strings of length L over a vocabulary of T tokens.

Capacity measurement (Def. 4.1, Thm. 3.2, Eq. A.1): with p1, p2, p3 the model's summed cross-entropy (nats) on name tokens, value tokens and diversity-set tokens,

    bits >= N*log2(N0/e^p1) + N*K*log2(D^C/e^p2) + K*D*log2(T^L/(D*e^p3))
    R = bits / P        (P excludes embeddings of unused tokens)

For bioS: R = [N*log2(N0/e^p1) + N*log2(S0/e^p2)] / P. This is directly reusable for Koko: build a held-out synthetic fact set, measure loss on the value tokens only, and convert to bits.

Model: GPT2 with rotary positions, no dropout, 64 dims per head, context 512, GPT2 tokenizer; written GPT2-l-h (layers, heads). Trained from scratch, AdamW, cosine decay to 0.1x, 1K warmup, fp16.

"Exposure" = number of times a fact appears in training in any wording, not the number of passes.

## Numbers
All on synthetic data; R is relative to total parameters.
- Base law (Result 1, Fig. 1a): 1000 exposures, N = 10K-10M, models 1M-0.5B params: peak R >= 2 bits/param, never above about 2.3. Models sized so that R_max <= 1.8 reach near-perfect accuracy.
- 100 exposures (Result 4, Fig. 1b): peak R >= 1 bit/param.
- bioD sweep (Result 3, Fig. 2): K, C in 1-50, D in 10-10,000, L in 1-50, T in 20-40,000: still about 2. Value term usually dominates the bit count (Fig. 13).
- Architecture, 1000 exposures (Result 5, Fig. 3): LLaMA, Mistral, GPT2 with a quarter-size MLP (d->d->d) and GPT2 with NO MLP all reach about 2.
- Architecture, 100 exposures (Results 6-7, Fig. 4-5): LLaMA/Mistral up to 1.3x worse than GPT2, traced to the gated MLP; quarter-size MLP negligible loss; no MLP loses more than 1.5x.
- Quantization (Result 8, Fig. 6, 14, 15): post-training GPTQ (auto_gptq, calibrated on 1,000 biographies). int8: negligible loss. int4: more than 2x loss; the introduction gives 0.7 bit/param. Authors say int4 "may" need quantization during training; not tested.
- MoE (Result 9, Fig. 7, 16): GPT2 with 32 experts (tutel), top-1, cap_factor 2 in training and 8 at test. Total/effective parameter ratio about 11.3 (8.8% of parameters used per token), yet capacity falls only 1.3x at 1000 exposures and 1.5x at 100. Top-2 behaves similarly.
- Junk (Result 10, Fig. 8): 1/8 of tokens useful bioS(N), 7/8 junk bioS(N'=100M). Useful capacity falls about 20x at 100 exposures; 3x at 300; 1.5x at 600; 1.3x at 1000.
- Repetitive junk (Result 11, Fig. 8f): junk = bioS(N'=1K): no loss at 100 exposures.
- Source tag (Result 12, Fig. 8g-h): one special token prepended to every useful passage: the 20x loss becomes about 2x at 100 exposures; at 300 exposures capacity equals the junk-free 100-exposure baseline.

## Smallest scale tested, and whether the result held there
About 1M parameters, with N down to 10K people. The 2 bit/param law is reported to hold across the whole 1M-0.5B range. Caveats stated: 1-layer models are slightly lower (the paper focuses on >= 2 layers); under 10M parameters LLaMA is slightly worse, tokenizer choice matters, and weight tying helps. Koko's "touched per token" size (1.3-2M weights) sits at the bottom edge of the tested range; its stored size (50-120M) is well inside it.

## Training recipe and gotchas
- Hyperparameters scale with N (Parameters 1-2, 5-7): lr 1e-3 for small N down to 3e-4 for N >= 2M; weight decay 0.02 down to 0.001; batch 24 up to 1536. Best of 2-3 learning rates reported.
- MoE needs HIGHER lr than dense (e.g. 1e-3 to 2e-3 at N = 10K-2M) and a capacity cap per expert (each expert receives at most cap_factor/M of the tokens).
- Fixed-wording facts (bioS-simple) are memorised but, per the authors' earlier Part 3.1, almost not extractable by questions; varied wording is what makes facts usable and does not cost capacity.
- Gated MLPs (SwiGLU style) hurt in the under-trained regime; plain MLP is safer.
- The exposure threshold is sharp under junk: 100 exposures is a cliff, 300 is mostly fine.

## Mapping onto the GBA design
Runtime cost: zero. Everything here is a training-data and budgeting result.
- Fact budget (my arithmetic, not the paper's): at 100-300 exposures a full-precision model gets 1-2 bits/param; int4 PTQ gets 0.7. Ternary with QAT is untested; planning with 0.25-0.5 bit/weight and a further 1.3-1.5x for hard routing gives roughly 17-40 Mbit for 100M stored weights, i.e. a few hundred thousand 6-attribute records at 47.6 bits each. Grammar and style consume capacity too, and the paper does not measure that share.
- Exposure budget is the tighter limit: F facts x about 30 tokens x 300 exposures. 100K facts = 0.9B training tokens; 1M facts = 9B. Choose a curated fact list and generate about 300 re-wordings each rather than hoping facts arrive from filtered text.
- 1-of-64 hard-routed experts: supported in spirit (32 experts, top-1, 1.3-1.5x loss). Copy the cap-factor balancing and the higher lr. The paper's routing is learned; Koko's routing may differ.
- Source tag: reserve 2-4 vocabulary ids (e.g. FACT, STORY, CHAT) and prepend to each training document; at run time the chat template supplies the tag, costing one ring-buffer slot and one embedding lookup.
- Junk: LLM-generated conversations full of unique invented names, numbers and places are "junk" in this paper's sense. Draw such fillers from a small fixed pool (about 1K entities) so the junk is repetitive (Result 11).
- No-MLP result says attention/embeddings alone can hold facts when exposure is high, which is mildly encouraging for moving capacity into ROM lookup tables, but the paper does not test lookup memories.

## Verdict
Adopt as a budgeting rule and corpus recipe (source tags, 300+ exposures per wanted fact, small entity pool for fillers); test the ternary number. Settling experiment: train the Koko architecture with ternary QAT on bioS(N) at N = 20K, 50K, 100K, 200K with 100 and 300 exposures, with and without 7/8 junk and tags, and report bits per stored weight using the formula above. A few GPU-hours each at this size.

## Could not verify
- Ternary or any QAT result: absent from the paper. The 0.25-0.5 bit/weight figure is my guess.
- Exact per-N entries of Parameters 1, 2 and 10, step counts and GPU counts: not recovered.
- Whether the MoE used an auxiliary balancing loss, and which layers were MoE: not found in the text I read.
- Whether the tag token must be present at test time: not stated in what I read.
- Figures were not viewed; all numbers come from the paper's text.
- Morris et al. number below is from the abstract only.

## Links worth following
- https://arxiv.org/abs/2309.14316 Part 3.1 (knowledge storage vs extraction; why re-wording matters). Cited by this paper; not re-read here.
- https://arxiv.org/abs/2505.24832 "How much do language models memorize?" (Morris et al., 2025): abstract reports about 3.6 bits/param for GPT-style models from 500K to 1.5B params; a second, differently measured estimate.
- GPTQ and the tutel MoE library, as the tools used (named in appendices C and D; URLs not checked).

## Independent verification
Second reader, 2026-11-19. Read in primary form: the arXiv HTML of 2404.05405 (v1), downloaded and searched as raw text (Sections 1, 5, 6, 8, 9, 10; Appendices A, C, D, E). Figures themselves were not viewed, only captions and text, so no per-model data point was checked.

1. MoE, 32 experts top-1, 8.8% of params, 1.3x / 1.5x loss: CONFIRMED, with scope limits.
   - Result 9 (Sec. 9) gives exactly 1.3x (1000 exposures) and 1.5x (100 exposures) for the PEAK capacity ratio, relative to total parameters, versus dense GPT2. Section 1 gives the 8.8% figure.
   - How 8.8% is derived (Sec. 9): only the MLP is replaced by 32 experts, each a narrow d->d->d block (2d^2 params, a quarter of a normal MLP); attention (4d^2) stays dense and shared. Ratio (4d^2+64d^2)/(4d^2+2d^2) = 11.3. Embeddings are ignored in that ratio.
   - Routing is learned (tutel), topk=1, cap_factor=2 in training and raised to 8 at test (App. D). The paper calls topk=1 with cap_factor=1 "generally not advisable". Deep models (GPT2-16-4) trained more easily with topk=2.
   - Not in the paper: 64 experts, sparsity below 8.8%, fixed/hash routing, ternary experts. No per-size breakdown of the MoE loss was readable from text, so whether 1.3x holds at the ~1M end is not established here.

2. Junk data 20x / 3x / 1.3x; source token 20x -> 2x: PARTLY. Numbers are right; the baseline is misstated.
   - Result 10: 20x at 100 exposures ("may degrade by 20x"), and 3x / 1.5x / 1.3x at 300 / 600 / 1000 exposures. But the 3x, 1.5x and 1.3x are measured against the junk-free 100-EXPOSURE law (about 1 bit/param), not against junk-free training at the same exposure count. So with 7/8 junk, 1000 exposures gives roughly 1/1.3 = 0.77 bit/param, about 2.6x below the junk-free 1000-exposure figure of 2. "Only 1.3x at 1000" overstates the recovery if read as same-exposure.
   - Result 12: with the special token, 2x at 100 exposures and parity with the junk-free 100-exposure law at 300 exposures. Confirmed. Footnote 21 adds that tagging the junk instead also works.
   - N ranges (App. E): 100-exposure junk runs used N = 50K to 1M; 300/600 used N <= 500K; 1000 used N <= 200K. So "N up to 200K-500K" applies to the longer runs only.
   - Setup detail: each 512-token window was all-useful or all-junk; junk is bioS with N' = 100M (same format as useful data, each junk person seen far less than once). This is "unlearnable look-alike facts", not arbitrary low-quality text.
   - Authors state they tuned to strengthen negative results and did not tune positive ones as hard (Parameter 10), so the 20x is a best-effort figure, the 2x a lightly tuned one.
   - Whether the tag must be present at test time: still not stated in the text I searched.

3. 2 bits/param at 1000 exposures, 1 bit/param at 100, for 1M-0.5B params: CONFIRMED, as a statement about the peak.
   - Result 1: peak R >= 2, R <= 2.3 for all models, N from 10K to 10M, sizes 1M to 0.5B. Result 4: peak R >= 1 at 100 exposures, same ranges.
   - These are peaks over (model, N) pairs, i.e. models loaded to the brim; they are lower bounds on bits computed from loss on the synthetic data.
   - Stated exceptions: 1-layer models are slightly lower, more so at 100 exposures (footnote 15, Fig. 9 caption). Parameter count excludes unused embedding rows (GPT2-small counted as 88M).
   - For N >= 2M only one model size per N was trained (App. A), so the large-N end rests on few points.

4. int8 free, int4 PTQ ~0.7 bit/param, ternary/QAT untested: CONFIRMED.
   - Result 8 (Sec. 8): int8 "negligible impact"; int4 "more than 2x" loss. The 0.7 bit/param number appears only in the Section 1 summary of Result 8. Tool: auto_gptq, calibrated on 1,000 biographies (App. C).
   - The authors say int4 "may" need quantization during training and that comparing quantization methods is left as future work. No QAT, no ternary, no sub-4-bit result exists in the paper. The statement that Koko's bits per weight must be measured is the first reader's inference, and a correct one.
   - Worth keeping in view: int8 at 2 bits/param is 1/4 of the 8-bit ceiling (Corollary 8.1); int4 at 0.7 is about 1/6 of its ceiling. A ternary weight has a ceiling of 1.58 bits, so the planning range 0.25-0.5 bit/weight in this note is 1/6 to 1/3 of ceiling, i.e. not conservative by this paper's own evidence.

Caveats the first read under-weighted
- Sparsity gap: Koko touches roughly 1-4% of stored weights per token (1.3-2M of 50-120M) and puts part of the store in lookup tables; the paper's sparsest case is 8.8% with dense shared attention and learned balanced routing. The 1.3-1.5x penalty is an extrapolation for Koko, not a measurement.
- Stacking: the MoE penalty, the junk penalty, the low-exposure penalty and the quantization penalty were each measured alone on fp16 GPT2. The paper never combines them; multiplying them is an assumption.
- Capacity here is memorisation of synthetic records measured by loss; nothing about fluency, and no measurement of how much capacity language itself consumes.
