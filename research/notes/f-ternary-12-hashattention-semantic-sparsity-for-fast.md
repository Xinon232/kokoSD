# HashAttention: Semantic Sparsity for Faster Inference

https://arxiv.org/abs/2412.14468 (v2; first posted Dec 2024). Aditya Desai, Shuo Yang, Alejandro Cuadron, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica (UC Berkeley; Cuadron ETH Zurich). Code: github.com/xAlg-ai/HashAttention-1.0.

Read in primary form (arXiv HTML and `hashattention/hashattention_llama.py`), through a fetch tool that condenses pages. Table values are as that tool reported them; I did not see raw HTML.

## What it is

A post-hoc sparse-attention add-on for frozen pretrained LLMs. Per head, two small learned MLPs map the query and each key to a 32-bit signature; Hamming distance picks candidate tokens, and ordinary attention with the original q, k, v runs only on those. Selection is framed as maximum inner product search in a learned, asymmetric Hamming space. The base model is never retrained; only the encoders are.

## Mechanism

Per head (code: separate `nn.ModuleList` entries per head; input is `head_dim`, not the residual width):

```
phi_k(k) = sign(F_k(k)),  phi_q(q) = sign(F_q(q))        # b = 32 bits each
F = Linear(head_dim,128) -> SiLU -> Linear(128,128) -> SiLU -> Linear(128,32)
score(q, i) = <phi_q(q), phi_k(k_i)>  in {-32..32}  = 32 - 2*Hamming   (XOR + popcount)
S = top-n(score) over the candidate region, plus first 128 and last 128 tokens (always kept)
out = softmax(q.K_S) V_S                                # exact, original vectors
```

- The paper defines phi_kv on key+value (R^2d) but uses the key only.
- Labels: paper says the top-k tokens by a_i * ||v_i|| (Lemma 4.1: a lone token's contribution is proportional to this), k "such as 64". The code takes `topk(attn_weights, k=32)` on attention logits, with no value norm visible.
- Training forward: tanh replaces sign; `sigmoid(tanh(F_q q) . tanh(F_k k) - thold)`, thold = 0; binary cross-entropy per (query, key) pair; Adam. A hard path uses a sign STE whose backward is (1 - tanh^2).
- Class weight for positives = alpha + beta * context_length (values not given).
- Code also supports 2-layer, 1-layer (plain linear) and 0-layer (random projection = the LSH baseline) encoders.
- Repo quick-start config: `init_budget` 128, `recent_budget` 128, `heavy_budget` 0.25, 3 layers, 128 hidden, 32 bits.

## Numbers

All on Llama-3.1-8B-Instruct (also Mistral-7B) at 16K-64K context.

- Table 8, learned vs random bits, average of four LongBench tasks (passage_retrieval_en / multifieldqa_en / hotpotqa / triviaqa), 16x sparsity: learned 32 bits 75.07 (99.43 / 55.18 / 52.34 / 93.32); LSH 32 bits 48.90; LSH 256 bits 69.79; LSH 512 bits 68.64; LSH 1024 bits OOM. At 8x: LSH 32 bits 60.07, 512 bits 72.61.
- Table 9, BCE of top-k prediction vs width: 16 bits 0.243, 32 bits 0.212, 64 bits 0.204 (250 x 32K samples). Diminishing returns past 32.
- Head-to-head LongBench table (512-token budget; includes 25 task examples per dataset in training): full 63.55, oracle top-k 63.43, HashAttention 32 bits 64.15, Quest 32 bits 62.36, Double Sparsity 32 bits 61.88, H2O 43.47, StreamingLLM 33.28.
- Table 2, LongBench at 16x without task tuning: 48.00 vs full 48.78. Table 3, RULER@16K at 16x: 89.53 vs 90.66.
- Table 7: mean query-key cosine of the true top-32 tokens is about -0.10 to -0.12 in the original space, +0.13 to +0.49 after the learned (tanh) encoders. This is the argument for asymmetric encoders: raw q and k do not live in a common space where sign bits agree.
- Table 6: packed-bit Hamming scoring beats the float inner product up to 512 bits (GPU).
- 32x sparsity needs task-specific fine-tuning (abstract).

## Smallest scale tested, and whether the result held there

Smallest model is 7-8B; shortest context about 16K. Critically, the first 128 and most recent 128 tokens are always attended densely, so inside a 256-token window the method as published does no selection at all. Nothing here is evidence for hashing within a 256-token ring, for head_dim below 128, or for a model trained from scratch with the hash in the loop. Encoders trained at 64K did not transfer to 128K.

## Training recipe and gotchas

- Online: run the frozen model in 256-token chunks over OpenWebText (concatenated to 32K), one Adam step per chunk, 1 epoch (repo README). Learning rate, alpha, beta, cost: not stated.
- Class imbalance is the main difficulty (k positives among thousands); hence the weighted loss.
- Paper/code disagree on label k (64 vs 32) and on value-norm weighting.
- Quality dropped more on Chinese than English: encoders are distribution-specific.
- Top-k itself became the bottleneck at long lengths.

## Mapping onto the GBA design

- RAM: 4 bytes per position per head. 256 positions x 4 heads x 6 layers = 24.6 KB (my arithmetic), fits slow RAM.
- Scoring: per position one 32-bit XOR plus popcount (no ARM7 popcount: four 256-entry LUT reads). My estimate is 20-30 cycles per position, about 5-8K cycles per head per token, against roughly 16 lookup+adds per position for a 64-dim sign key in the existing nibble-table kernel. Not measured.
- Top-8: scores take 33 values, so a histogram pass picks the threshold; ties broken by recency. No sort.
- Encoders are the problem. The paper's 3-layer MLP is about 28K weights per encoder per head; two per head across 24 heads would exceed the whole per-token budget. Only the 1-layer form (head_dim or d_model -> 32 ternary rows, about 2-8K weights each) is affordable, and the paper has no numbers for it.
- ROM precompute: layer-0 key signatures depend only on the token (and any hashed n-gram input), so they can be a ROM table (16K tokens x 4 bytes per head).
- Our keys are already trained sign/ternary projections, i.e. a learned hash. What this paper adds is (a) separate, asymmetric query/key encoders for selection, (b) a recall-targeted BCE loss, (c) exact rescoring of the 8 survivors with 8-bit keys (8 x 64 MACs is about 3.8K cycles per head).

## Verdict

Test. Borrow the structure (short asymmetric signature for selection, exact rescoring of top-8, auxiliary BCE recall loss), not the published module. Settling experiment: in the QAT model at 256 context, compare recall@8 against full-attention top-8 and validation loss for (i) current sign keys scored directly, (ii) a 32-bit linear asymmetric hash plus 8-bit rescoring, (iii) the same with a 2-layer encoder as an upper bound; also sweep 16/32/64 bits.

## Could not verify

- Learning rate, alpha/beta, training cost: not in paper or the file I read; `train_usa.py` returned 404 at two guessed paths.
- Latency units in Table 6; whether Table 9 is top-32 or top-64 (text is inconsistent).
- Table 5 (Chinese) direction of the difference is ambiguous as printed.
- All GBA cycle figures above are my estimates.

## Links worth following

- Code: https://github.com/xAlg-ai/HashAttention-1.0
- Baselines named in the paper (not read here): Double Sparsity, Quest, InfLLM.
