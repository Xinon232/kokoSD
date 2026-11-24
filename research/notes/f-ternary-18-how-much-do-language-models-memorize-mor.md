# How much do language models memorize?  (https://arxiv.org/abs/2505.24832, v3, 2025; Morris, Sitawarin, Guo, Kokhlikyan, Suh, Rush, Chaudhuri, Mahloujifar — Meta FAIR / Google DeepMind / Cornell / NVIDIA)

Read in primary form (arXiv HTML, full text and Tables 1-4). Figures were read by caption only.

## What it is
A measurement paper, not an architecture. It defines "unintended memorization" as the bits a trained model saves when compressing its training set beyond what a reference model of the true distribution saves, and uses it to measure the storage capacity of GPT-2-style transformers. Trained on uniform-random token strings (nothing to generalise), memorised bits plateau at about 3.5-3.6 bits per parameter. On real text, models memorise until capacity is full, then per-sample memorisation falls and generalisation starts (double descent at dataset-bits ~ capacity).

## Mechanism
- Per-sample estimate (arithmetic-coding code length = NLL in bits):
  `mem_U(x) = -log2 p(x|ref) - ( -log2 max{p(x|model), p(x|ref)} )`; sum over the training set.
- Synthetic capacity test: N sequences of S tokens, each uniform over V. Exact `H(X) = N*S*log2(V)`. Reference is the uniform distribution, so `mem = H(X) - NLL_bits(X | model)`. Train to saturation for a sweep of N; `Capacity = max_N mem`; `alpha = Capacity / params`.
- Predictor verified in the appendix: `mem ~= min(alpha * params, H(X))`, with params including the embedding matrices.

## Numbers
- Table 1 (synthetic, V=2048, S=64; 1/2/4/8 layers x d_model 32/64/128/256; 8.04e4 to 6.86e6 params): mean alpha = 3.51 bpp in bfloat16, 3.83 bpp in float32 (stated +-0.1). Outliers: 1 layer x 256 gives 2.88; 8 layers x 32 gives 2.86.
- Fig. 6 / A.1: fitted alpha = 3.64 (3.642) for half precision; abstract rounds to 3.6. Text: "between 3.5 and 3.6".
- Doubling stored bits (16 -> 32) adds about 9% capacity: storage is not limited by float precision.
- Tables 3-4 (2 layers, d=128, N=4096): the min() predictor is within 0.41-4.19% across S=4..256 and 0.36-2.78% across V=128..4096. Extra embedding rows (4.21e5 -> 9.29e5 params) add capacity at the same alpha.
- Convergence: datasets of 16K-4M samples all end at 3.56-3.65e6 bits; the two largest were still rising at 1e6 steps, so capacity is a lower bound.
- Table 2: GPT-2 small-size (123.7M params, labelled "GPT2-Medium") predicted F1 0.55/0.75/0.95 at |D| = 13.57M/6.10M/1.50M samples, observed 53.44/65.69/97.98.
- Text (FineWeb, 64-token deduplicated sequences): generalisation starts at about 1e5 samples, depending on model size (Fig. 4).

## Smallest scale tested, and whether the result held there
80K parameters (1 layer, d=32). It held roughly: alpha 3.93 (bf16). The whole synthetic study is 80K-6.9M params (text says up to d=512 / 20M, Table 1 stops at 256), which is our scale. Spread across shapes is about +-15%; very wide-shallow and very narrow-deep are worst.

## Training recipe and gotchas
- GPT-2 architecture from scratch, Adam, 1e6 steps, batch 2048, bfloat16, five seeds, single A100. Learning rate and schedule are not given in the body text.
- Memorising random data needs very many epochs; under-training under-reports alpha.
- No quantised, ternary or QAT model was measured. No code link found in the paper.

## Mapping onto the GBA design
No runtime cost: this is a budgeting tool. My own arithmetic from here on, not the paper's:
- Hard ceilings: free ternary = log2(3) = 1.585 bits/weight; the "one zero per 4" format = 5 bits per 4 weights = 1.25 bits/weight. Against 3.5-3.8 bpp for float, a ternary weight holds at most 0.41-0.45 of a float weight's memorised information (0.33-0.36 in the 32-entry format). So the "ternary multiplier" for equal memorisation is at least 2.2-2.4x weights (2.8-3.0x for the 32-entry format), and more if QAT does not reach the ceiling.
- ROM: 100M free-ternary weights hold at most 158 Mbit (19.8 MB), the information of about 44M float parameters. GPT-2 small at 3.6 bpp is about 446 Mbit; the whole 32 MB cartridge is 256 Mbit. We cannot match GPT-2 small on stored facts; we must win on a narrower distribution (simple English).
- Lookup tables: embedding parameters counted at the same alpha in Table 4, so hashed n-gram and per-layer embedding rows are a legitimate place for capacity that costs almost no cycles. Caveat: that test used uniform tokens, V <= 4096, dense fp rows; Zipfian hashed rows will fill unevenly.
- Data: with a few hundred million tokens against 100-160 Mbit of capacity we are far past the double-descent point, so sample memorisation is not a concern; rare-token content is what gets memorised first, so vocabulary filtering matters.

## Verdict
Adopt as a measurement method; test the ternary number. Experiment: rerun the synthetic test (V=2048, S=64, 2 layers x d=128, about 0.67M params, N swept 2^10..2^20) with float, ternary-QAT, the one-zero-per-4 format, and a hashed-embedding variant; report alpha per stored weight. It is small enough for hours on one GPU and settles the multiplier and the table-versus-width split.

## Could not verify
- Any measured capacity for ternary/low-bit weights: not in this paper.
- Learning rate, warmup, exact dataset-size grid: not found in the text read.
- The paper quotes Allen-Zhu & Li 2024 at about 2 bpp; I did not open that source.

## Links worth following
- Allen-Zhu & Li 2024, "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws" (capacity under quantisation).
- Boo et al. 2019, "Memorization capacity of deep neural networks under parameter quantization" (cited in the paper's references).
