# Large Text Compression Benchmark (https://mattmahoney.net/dc/text.html, 2006-present, last update Nov. 17 2026, Matt Mahoney)

Read in primary form (full page downloaded, 713 KB), plus three linked writeups: fx2-cmix-transformer `writeup.md`, cmix-neif-beehive `README.md`, zmix `writeup.md`. bpc figures below are my arithmetic (8 x bytes / 10^8 or 10^9) from the page's byte counts.

## What it is
An open ranking of ~200 lossless compressors on enwik9 (first 10^9 bytes of the 2006-03-03 English Wikipedia XML dump) and enwik8 (first 10^8). Rank = compressed enwik9 + zipped decompressor, so model weights count. Each row also gives ns/byte and peak memory, which makes it a likelihood-versus-compute-versus-memory table. In 2026 the leading entries are all cmix plus a small offline-trained, low-bit transformer run on one CPU core.

## Mechanism
- Score: bpc = 8 x compressed bytes / input bytes; arithmetic coding makes this equal to the model's cross-entropy plus a negligible coder overhead.
- Families (page glossary): PPM (longest matching context, back off on zero counts), CM (mix bit predictions of many hashed-context models), LSTM/Tr (CM with a neural model).
- fx2-cmix-transformer (writeup, primary): 6M-parameter byte-level transformer (205-symbol alphabet after dictionary preprocessing), no tokenizer, PPMD next-byte distribution fed in as an extra input; 75% of layers use Kimi Linear attention, the rest sliding-window attention of 1024; RoPE, QK-norm, ReLU^2 MLP, Muon. Weights 4-bit in {-7..7} (stored at log2(15)=3.81 bits), activations 8-bit, one scale per INPUT channel so the matmul stays pure int8; embeddings and unembeddings quantized too. Modified ParetoQ QAT.
- cmix-neif-beehive (README, primary): 16 layers, d_model 192, MLP width 1024, 208-symbol vocabulary, window 1536, one KV head shared by 3 query heads, value embeddings on layers 3/7/11/15, output head that mixes PPMd-prior logits per token; row-wise mixed precision (99.52% of rows 4-bit, 0.47% 2-bit) allocated by Fisher information, with a differentiable weight-container-size term in QAT.
- cmix-neif 2.0 (page): adds SEMCACHE, a nearest-neighbour cache over transformer states: when the transformer is unsure, find earlier positions with a similar state and predict from the bytes that followed.

## Numbers
Main table (page), enwik8 bytes / enwik9 bytes / decomp ns/byte / MB:
- cmix-neif 2.0: - / 89,346,392 (0.715 bpc) / 177,480 / 9,770. Top entry.
- fx2-cmix-transformer: - / 96,996,198 (0.776) / 131,561 / 9,547.
- nncp v3.2 (199M-parameter transformer trained online, GPU): 14,915,298 (1.193) / 106,632,363 (0.853) / 238,670 / 7,600.
- cmix v21: 14,623,723 (1.170) / 107,963,380 (0.864) / 638,442 / 30,950.
- paq8hp12any: 16,230,028 (1.298) / 132,045,026 (1.056) / 37,584 / 1,850.
- drt|lpaq9m: 17,964,751 (1.437) / 143,943,759 (1.152) / 898 / 1,542.
- lstm-compress v3: 20,318,653 (1.625) / 173,874,407 (1.391) / 91,876 / 9.
- ppmd J1 -o10: 21,388,296 (1.711) / 183,964,915 (1.472) / 895 / 256.
- ppms J -o5: 26,310,248 (2.105) / 233,442,414 (1.868) / 354 / 1.8.
- ha 0.98 (PPM): 31,250,524 (2.500) / 285,739,328 (2.286) / 1,800 / 0.8.
- gzip -9: 36,445,248 (2.916) / 322,591,995 (2.581) / 17 / 1.6.

From the linked writeups:
- fx2-cmix-transformer: quantized model 0.911 nats/byte (1.314 bits) versus the replaced 0.4M-parameter online LSTM at 1.23 nats/byte; 4-bit/8-bit QAT costs +2.6% loss versus bfloat16; weights 2.9 MB; trained 8 x RTX 5090 for 26 h (320 epochs float, 256 QAT, 16 float32 QAT).
- beehive: deployed cross-entropy 1.27404 bits/token versus 1.2915 for cmix-neif-1.0; weight container 3,649,198 B; "dense weights (~6 MB per token)" streamed per predicted byte.
- fx2-cmix-t-ft8 (page): 8 further epochs (~$20 of GPU) gain only 51 KB (96,945,005 versus 96,996,198).

## Smallest scale tested, and whether the result held there
Memory floor in the table: ha at 0.8 MB (2.50 bpc enwik8) and ppms at 1.8 MB (2.11). Smallest neural models: lstm-compress in 9 MB (1.63 bpc enwik8, online-trained) and the 6M-parameter 4-bit transformer. That transformer still reads about 6 MB of int8 weights per byte, against our 1.3-2M ternary weights per token. Nothing in the table is ternary or runs under 256 KB of writable memory with a neural model.

## Training recipe and gotchas
- The 2026 transformer numbers are training-set likelihoods: the model is trained for hundreds of epochs on the exact file it codes. They show what a 3-4 MB quantized model can memorize and generalize jointly, not held-out bpc. Do not compare them with GPT-2's zero-shot figure.
- The losses are over a preprocessed stream (dictionary-coded, 888,096,401 B for enwik9), not raw bytes.
- All classical entries are adaptive (learn while coding), so their bpc on enwik8 includes the cold start; a static ROM model has no cold start but no adaptation either.

## Mapping onto the GBA design
- Evaluation: report Koko's bits/byte on enwik8 and on held-out simple English next to these rows. Realistic bracket for a static held-out model at our compute: between ppmd (1.71) and paq8hp12 (1.30) on enwik8; GPT-2 small at 1.16 (GPT-2 paper, not this page) sits at the cmix v21 level.
- PPM-prior input: a static order-3..5 count model stored in ROM costs a few hashed lookups per token and no multiplies; feed its distribution (or top-k candidates) into the network and mix its logits in the output head, as fx2 and beehive do. This matches our hashed n-gram embeddings but adds an explicit probability path.
- Per-input-channel weight scales keep the matmul integer-only; with ternary weights the scale folds into the 8-bit activation before the partial-sum table is built, at zero kernel cost.
- Shared KV across query heads cuts ring-buffer RAM and key scans by 3x for 3 heads.
- SEMCACHE maps to a ROM-side nearest-neighbour table keyed on a hash of the sign pattern of the final hidden state; only on low-confidence tokens (conditional execution).
- Not transferable: online learning and adaptive mixers need writable memory we lack.

## Verdict
Adopt as an evaluation yardstick; test the PPM-prior input/logit-mixing head and shared-KV attention. Settling experiment: train the float model with and without an order-4 static n-gram prior fed to input and output head, and compare held-out bits/byte at equal dense compute; then repeat with ternary QAT.

## Could not verify
- GPT-2's 1.16 bits/byte is not on this page (no GPT-2 entry found in the text).
- No held-out numbers exist for any 2026 transformer entry; fractal-zip's enwik8 figure (12,303,845) is presumably from a model that saw the data, so I did not use it.
- cmix-neif 2.0 repository not read; SEMCACHE details are Mahoney's summary only. Beehive's parameter count is not in its README.
- Results are submitter-reported; Mahoney states he has not verified them, and timings come from different machines.

## Links worth following
- https://github.com/astOwOlfo/fx2-cmix-transformer-v1/blob/main/writeup.md
- https://github.com/galqiwi/cmix-neif-beehive (training/README.md section 12: Fisher bit allocation, size term)
- https://github.com/marwoelk/cmix-neif-2.0 (SEMCACHE)
- https://arxiv.org/abs/2502.02631 (ParetoQ), https://arxiv.org/abs/2510.26692 (Kimi Linear)
