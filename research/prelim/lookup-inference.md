## Bottom line
End-to-end-trained PQ/LUT inference is unlikely to beat your ternary kernel on quality per cycle for a 256-wide autoregressive LM. The one promising variant is untested in the literature: comparison-only tree encoding with large K stored in ROM.

Verification note: "verified" means I pulled the figure from the arXiv/ar5iv full text through a fetch tool that summarises pages; I did not read raw PDFs. "Abstract only" and "secondary" are marked.

## 1. MADDNESS (verified, ar5iv full text)
- **Encode:** each codebook is a 4-level balanced binary tree with 16 leaves. Each level compares one input coordinate against a threshold, so encoding is 4 comparisons per codebook and zero multiplies.
- **Tables:** K=16 prototypes per codebook, fitted by ridge regression; tables are 8-bit and lookups are aggregated by pairwise averaging.
- **Claims:** about 100x faster than exact matmul (abstract).
- **Limits:** tested only on single layers (e.g. the final softmax layer on 512-d CIFAR features). The authors state they have no results on full networks, and the hash is not differentiable.
- https://arxiv.org/abs/2106.10860

## 2. LUT-NN (verified, ar5iv full text)
- **Training:** the forward pass uses hard argmin to the nearest centroid; the backward pass uses softmax(-dist²/t) via stop-gradient. Temperature is learned per layer; centroids start from k-means.
- **Temperature matters:** learned t gives 94.4% on ResNet18/CIFAR-10, versus 91.55% annealed and 89.85% fixed.
- **Settings:** K=16 with V=9 for 3x3 convs; K=16 with V=16 or 32 for BERT. Tables are INT8.

| Model / task | Original | LUT-NN | MADDNESS |
|---|---|---|---|
| ResNet18, CIFAR-10 | 95.26 | 94.40 | 10.01 |
| ResNet18, ImageNet | 69.76 | 67.38 | 0.10 |
| BERT, GLUE average | 80.4 | 78.5 | not reported |

- **Caveat on BERT:** only the last 6 layers' FC operators are replaced. The paper says replacing the first two layers "results in 80% accuracy loss". The first conv layer is also kept exact.
- **Encode cost:** N·D·K multiply-adds for distance computation. Overall FLOPs reduction is M/(K + M/V).
- https://arxiv.org/abs/2302.03213

## 3. Follow-ups
- **Stella Nera** (verified): makes the MADDNESS tree differentiable with hard sign forward, tanh/softmax backward and a straight-through estimator. ResNet9/CIFAR-10 reaches 92.6%, about 1.2 points under FP32. It uses 9 inputs per codebook, K=16, INT8 tables, and keeps the first and last layers exact. https://arxiv.org/abs/2311.10207
- **PECAN** (verified): the multiplier-free PECAN-D variant encodes by L1 distance, about 2·K·d additions per subvector. ResNet20/CIFAR-10 drops from 92.55 to 87.88; CIFAR-100 from 69.55 to 58.01. It needs K=64–128 with d=3–4. https://arxiv.org/abs/2208.13571
- **TabConv** (abstract only): keeps "over 93% of original performance" with a 36.5% operation reduction on ResNet18/CIFAR-10. https://arxiv.org/abs/2404.05872
- **"Look-ups are not (yet) all you need"** (verified): a negative result. A fully replaced MNIST MLP reaches 36% at 2 codebooks and 84.9% at 16, by which point it is slower than matmul. The encoder is named as the bottleneck. https://arxiv.org/abs/2207.05808
- **LUT-LLM** (verified, arXiv HTML): the only generative-LM data point found, Qwen 3 1.7B on FPGA.
  - It needs subvector length 2, 64 activation centroids and 16 weight centroids (about 3 bits per activation).
  - MMLU-Pro falls from 33.1 to 30.8 and SQuAD v2 from 72.8 to 69.7. No perplexity is reported.
  - Centroid search uses Chebyshev distance.
  - https://arxiv.org/abs/2511.06174
- **NoMAD-Attention** (abstract only): PQ lookups for attention scores only, up to 2x on LLaMA-7B. https://arxiv.org/abs/2403.01273

## 4. A 256->256 layer (my arithmetic from the papers' formulas)
Your current kernel costs 64 groups x 81 table builds (about 5.2k adds) plus 256 x 64 = 16.4k lookup-adds.

| Config | Encode | Lookup-adds | ROM table (int8) |
|---|---|---|---|
| LUT-NN, V=16, K=16 (C=16) | 4,096 MACs (~29k cycles) | 4,096 | 64 KB |
| V=8, K=16 (C=32) | 4,096 MACs | 8,192 | 128 KB |
| MADDNESS tree, V=8, K=16 | 128 compares | 8,192 | 128 KB |
| LUT-LLM-like, V=2 | 128 x 64 distances | 32,768 | — |

- Distance encoding at 7 cycles per MAC erases the gain on your CPU; only tree or L1 encoding makes sense.
- At LUT-LLM's V=2, lookups alone are double your current cost.
- Your ternary kernel is effectively exact PQ with V=4 and K=81. PQ only wins at V≥8, which is where the accuracy loss occurs.

## 5. Retro and MCU language models (secondary sources; not checked against repos)
- llama2.c64 runs TinyStories-260K on a C64 with a 2 MB REU: https://github.com/ytmytm/llama2.c64
- ESP32 ports of the same model: https://github.com/eric-humane/esp32-llm
- A 28.9M-parameter ESP32-S3 model keeps about 25M embedding parameters in flash and reads only the needed rows: https://hackster.io/news/running-a-28-9m-parameter-llm-on-an-8-microcontroller-173f1f370708
- None use PQ/LUT matmul; they use int8/4-bit weights, with "lookup" meaning embedding rows from flash. I found no PQ-inference LM on retro hardware.

## Assessment (my inference, not from a source)
- **Quality per compute: worse.** K=16 over 8–16 activations passes 4 bits per subvector, a harsh activation bottleneck. Every paper keeps early layers exact, and the only generative-LM result needed V=2. Autoregressive models also compound error, and ternary with exact arithmetic has no activation bottleneck.
- **Your ROM changes one thing.** Papers fix K=16 for SIMD shuffles. You could use depth-8 trees (K=256, 8 compares per codebook) at V=8, costing 2 MB per matrix. That gives 8 bits per 8 activations at about 8.2k lookup-adds plus 256 compares, roughly 2.5x fewer operations than now. Nobody has published this, so it needs your own experiment.
- **Hybrid:** use LUT for the MLP down-projection and for the unembedding (wide, tall, followed by argmax/sampling, and MADDNESS's proven case). Keep exact ternary for attention projections, early layers and anything writing to the residual stream. Train with a Stella Nera-style straight-through tree and learned temperature, and distil from the ternary model.
- **Cheaper first step:** since the inner loops are identical, try storing the 81-entry-per-group idea differently, or simply V=5 (243 entries) or V=6 (729), to cut lookups exactly with no accuracy loss.