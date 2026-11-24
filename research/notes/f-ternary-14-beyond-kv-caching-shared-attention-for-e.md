# Beyond KV Caching: Shared Attention for Efficient LLMs (https://arxiv.org/abs/2407.12866, 2024, Bingli Liao and Danilo Vasconcellos Vargas, Kyushu University)

Read in primary form (arXiv HTML v1, via a fetch-and-extract tool, so table cells were machine-extracted). Code: https://github.com/metacarbon/shareAtt (not read).

## What it is
A short empirical paper proposing "Shared Attention" (SA): compute the softmax attention weights once in one layer and reuse the same weight matrix in a span of following layers, each of which still applies its own value projection. The motivation is a measured high cosine similarity between attention maps of different layers in pretrained 7B-72B LLMs. It is tested only by patching pretrained Llama2-7B and Llama3-8B (32 layers), with and without a light instruction fine-tune. No model is trained from scratch with SA, and no speed, FLOP or memory measurement is reported.

## Mechanism
Algorithm 1 in the paper:

```
S = contiguous span of layers, e.g. 23..30
for l in layers:
    if l == first(S):  A = softmax(Q_l K_l^T / sqrt(d_k)); A_l = A
    elif l in S:       A_l = A                 # no Q, K, softmax
    else:              A_l = softmax(Q_l K_l^T / sqrt(d_k))
    O_l = A_l V_l                              # V_l is always per-layer
```

- Shared object: the post-softmax weights (all heads), not Q/K parameters. Layers inside the span need no W_Q, W_K, no key cache, no softmax.
- Spans tested: 4 layers (15-18, 19-22, 23-26, 27-30) and 8 layers (23-30) of 32.
- Span choice: cosine similarity between layer attention matrices on MMLU prompts (zero-padded to max length) plus a "weighted cumulative variance" analysis; early layers have higher variance, so only late layers were shared.
- Layer grouping reported for Llama2-7B-chat: layers 0-1 (distinct), a transitional group up to about layer 5, one large highly similar group up to the penultimate layer, and a distinct final layer.

## Numbers
All from Table 1 (GLUE / GSM8K / HellaSwag / MMLU), 7-8B models, SA applied without retraining unless marked SFT:

| Model | GLUE | GSM8K | HellaSwag | MMLU |
|---|---|---|---|---|
| Llama2-7B base | 0.4050 | 0.1395 | 0.5713 | 0.4119 |
| SA 27-30 | 0.3882 | 0.1243 | 0.5616 | 0.4056 |
| SA 23-26 | 0.4351 | 0.1122 | 0.5681 | 0.3994 |
| SA 23-30 | 0.3819 | 0.0728 | 0.5575 | 0.3794 |
| SA 19-22 | 0.3996 | 0.0834 | 0.5553 | 0.3926 |
| SA 15-18 | 0.3731 | 0.0220 | 0.4790 | 0.3378 |
| Llama2-7B SFT | 0.5372 | 0.1440 | 0.5772 | 0.3722 |
| Llama2-7B SFT + SA 23-30 | 0.5401 | 0.0758 | 0.5671 | 0.3717 |
| Llama3-8B base | 0.4804 | 0.5155 | 0.6009 | 0.6198 |
| SA 27-30 | 0.5532 | 0.4526 | 0.6060 | 0.6163 |
| SA 23-26 | 0.5024 | 0.4556 | 0.5993 | 0.6189 |
| SA 23-30 | 0.5595 | 0.3275 | 0.6011 | 0.6122 |
| SA 19-22 | 0.5115 | 0.3745 | 0.5829 | 0.6181 |
| SA 15-18 | 0.4685 | 0.0136 | 0.5307 | 0.3019 |

- Pattern: knowledge/multiple-choice scores barely move for late spans; multi-step arithmetic (GSM8K) drops sharply (Llama3-8B 0.5155 to 0.3275 for an 8-layer span) and fine-tuning did not recover it (Llama2 SFT 0.1440 vs 0.0758).
- Sharing layers 3-6 of Llama2-7B gave "an explosion of perplexity" (text only, no number).
- Similarity inside the large middle group rose from about 0.8 to 0.9 over Baichuan2-7B pretraining checkpoints (0.2T to 2.6T tokens) - i.e. similarity is a product of long training. No per-layer similarity table is given.
- No perplexity, latency, FLOP or cache-size numbers anywhere.

## Smallest scale tested, and whether the result held there
7B parameters, 32 layers. Nothing smaller. Even there the result only partly held: sharing 4 late layers (12% of depth) is nearly free on MMLU/HellaSwag but costs 9-12% relative on GSM8K; anything in the first half of the network breaks the model. A 4-6 layer model has no "redundant late middle" by this paper's own grouping (2 distinct early layers + transitional + distinct last layer already use up the depth).

## Training recipe and gotchas
- Fine-tune only: Alpaca-style instruct data, AdamW, lr 2e-5, 2 epochs, bf16, micro-batch 16 per GPU on 2x A100-80GB, DeepSpeed ZeRO-3.
- Authors state from-scratch pretraining with SA was not done for cost reasons; they speculate it would work better. Untested.
- The similarity metric is whole-matrix cosine on padded maps, dominated by a few high-mass positions (BOS, recent tokens); high cosine does not imply that the informative low-mass differences are shared.
- Similarity analysis used chat/instruct variants; the SA experiments used base models.

## Mapping onto the GBA design
What SA would mean for Koko: one key scan + top-8 selection produces 8 (position, weight) pairs (about 16 bytes of RAM); a following layer skips its query projection, its 256-position key scan, its top-8 selection and its softmax, and just gathers its own 8-bit value rows at the same 8 positions.

My own estimates (not from the paper), per shared layer, per head with 64-dim sign/ternary keys: scan 256 x 64 = about 16K weight-equivalents (about 26K cycles at 1.6 c/w); query projection 384 x 64 = about 25K ternary weights (about 39K cycles). With 4 heads that is roughly 0.26M cycles per layer, i.e. 5-9% of a 3-5M token budget per layer shared. RAM: one fewer key ring per shared layer (256 x 64 bits = 2 KB per head); value rings are unaffected and remain the larger RAM cost. ROM: saves W_Q/W_K, irrelevant. Nothing is precomputable to ROM since the weights depend on context.

Changes required: train with tied attention from the start (a layer pair shares one selection), and the freed cycles should be re-spent (wider MLP expert or an extra value-only layer). A cheaper middle ground suggested by the evidence, not by the paper: share only the top-8 index set and let each layer rescore those 8 candidates with its own query (8 x 64 ops instead of 256 x 64), which keeps per-layer weighting and removes the scan.

## Verdict
Test, low priority; do not rely on it. The paper's evidence is for late layers of deep, long-trained 7B models, with no speed data and real damage on reasoning tasks; our 4-6 layer from-scratch model is exactly the regime it did not test and where its own layer grouping predicts little redundancy. Settling experiment (cheap): train the float/QAT baseline at 6 layers three ways with equal per-token cycle budget - (a) independent attention, (b) layers 3-4 and 5-6 share full top-8 weights, (c) shared top-8 indices with per-layer rescoring, with saved cycles re-spent on MLP width - and compare held-out perplexity and a multi-turn coherence probe. Also first just measure top-8 index overlap between adjacent layers of the baseline; if overlap is below about 50%, skip.

## Could not verify
- Per-layer similarity values (figures only; extraction gave "about 0.8 to 0.9" for the middle group).
- Any efficiency number: the paper reports none.
- Whether a fine-tuned Llama3-8B SA row exists; the extracted table had none.
- Table column mapping: the HTML header was garbled; the four-column order GLUE/GSM8K/HellaSwag/MMLU is the extractor's reading.
- Whether the paper was later peer-reviewed or revised; only v1 read. Code repository not opened.
- LiSA and EchoAtt descriptions below come from search-result summaries, not primary reading.

## Links worth following
- EchoAtt (https://arxiv.org/abs/2409.14595): same idea with distillation on TinyLlama-1.1B; secondary summary claims 15% faster inference, about 4% fewer parameters. Smallest-scale evidence found.
- LiSA, Cross-layer Attention Sharing for LLMs (https://arxiv.org/abs/2408.01890): shares attention across adjacent layers with small alignment networks plus low-rank corrections - closer to the "shared candidates, per-layer rescoring" variant.
- https://github.com/metacarbon/shareAtt: the paper's code.
