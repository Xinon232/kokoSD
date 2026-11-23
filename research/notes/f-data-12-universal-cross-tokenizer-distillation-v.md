# Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching (ALM)

(https://arxiv.org/abs/2503.20083, NeurIPS 2025; Minixhofer, Vulic, Ponti. Code: https://github.com/bminixhofer/tokenkit, Apache-2.0, JAX.)

Source status: full arXiv HTML read in primary form (main text, appendices A-I, all tables). Figure 3 is an image; its values were not readable. The tokenkit README was read through a summariser.

## What it is
A distillation loss for teacher and student with different tokenizers. It aligns chunks of tokens covering the same bytes and matches the probability each model gives to that chunk of the observed text. It needs no vocabulary mapping and no precomputation. It is tested on tokenizer transfer of 2-12B models, on 8B to 2B distillation, and on hypernetwork training; never on a from-scratch or sub-2B student.

## Mechanism
1. One forward pass per model gives log-probs of the observed tokens: `lpT` (b x n), `lpS` (b x m).
2. Greedy, non-overlapping alignment by byte offset: a chunk ends wherever both tokenizations share a boundary. Encode as binary matrices N (b x n x k), M (b x m x k), k = min(m, n). Runs in the data loader.
3. Chunk log-probs: `CT = lpT @ N`, `CS = lpS @ M`.
4. Debiasing: add the log of the mass each model puts, after the chunk, on next tokens starting with a pretoken-boundary byte (space, \n, \t).
5. Keep a chunk only if the teacher's boundary mass >= gamma (default 0.1).
6. Loss per chunk (losses combined by weighting each with 1 / its last-layer gradient norm; an optional hidden-state matching loss exists), p = exp(CT), q = exp(CS): `f(p^(1/tau) || q^(1/tau)) + f(1-p^(1/tau) || 1-q^(1/tau))`, f_KL(p||q) = p log(p/q), tau = 100. With TVD and tau -> infinity it reduces to `mean |CS - CT|`.

The teacher signal is one scalar per chunk, not a distribution over the vocabulary (stated limitation, Section 3.2).

## Numbers
Averages over PiQA, ARC-C, BoolQ, MMLU, AGIEval-EN/ZH, IFEval; about 0.6B training tokens.

| Setting | orig. | SFT | DSKD | MinED | ALM+SFT | ALM |
|---|---|---|---|---|---|---|
| Gemma2 2B -> Qwen2 tok. (T1) | 58.0 | 51.6 | 51.6 | 53.0 | 54.9 | 55.1 |
| Gemma2 2B -> bytes (T1) | 58.0 | 46.5 | 47.4 | 47.1 | 51.3 | 50.6 |
| Llama3.2 3B -> Qwen2 tok. (T1) | 59.3 | 55.7 | 51.1 | 57.3 | 58.0 | 58.6 |
| Llama3.2 3B -> bytes (T1) | 59.3 | 52.6 | 48.7 | 52.4 | 53.5 | 52.0 |
| Gemma3 12B -> Qwen2 tok. (T5) | 71.2 | 64.6 | - | - | - | 69.2 |
| OpenMath2 8B -> Gemma2 2B, GSM8K/MATH avg, teacher 74.6 (T3) | - | 51.7 | 50.3 | 49.8 | 53.3 | 52.6 |

- Large-to-small (Table 3): gain over SFT is 0.9-1.6 points, standard error about 0.7. Both prior methods were worse than SFT.
- Temperature (Table 7, 5k steps): KL tau=1 56.1, tau=5 56.3, tau=100 55.5. Authors say tau=5 "might be a better choice".
- Debiasing (Table 8): Gemma2 -> Qwen2 54.1 without, 55.5 with. Llama3 -> bytes 43.2 without, 45.3 with, 47.2 with the hidden-state loss too. That loss slightly hurt subword-to-subword (55.5 -> 55.0).
- Cost: loss overhead "negligible" next to the backbone passes (Appendix I). Runs took 5-24 h on up to 32 TPU v4 chips (Appendix B).

## Smallest scale tested, and whether the result held there
Smallest student: pretrained Gemma2 2B. TinyLlama 1.1B appears only in a hypernetwork ablation (Table 10). Nothing from scratch, below 1B, quantized, or near a 16K vocabulary. The claim we care about is untested.

## Training recipe and gotchas
- Defaults: Adam, no weight decay, LR 1e-5, 2k warmup, linear decay, 20k steps, batch 64 x 512 tokens, LoRA r = alpha = 64.
- Keep the teacher's special tokens; moving Gemma2 to a tokenizer without <bos> degraded it badly (Appendix H).
- Section 3 calls the binarised divergence an upper bound; Appendix E proves a lower bound. Lower is correct.
- Stated limits: unknown behaviour beyond 0.6B tokens; only supervised KD.

## Mapping onto the GBA design
Training-time only: zero lookups, adds, multiplies, RAM or ROM on the cartridge; nothing to precompute into ROM. On the 5090 it costs one teacher forward pass per sequence, cacheable as about 2 floats per teacher token, plus alignment in the data loader.

Two routes (my analysis):
- Direct: off-the-shelf teacher -> our 16K ternary student. One scalar per chunk says how surprising the teacher found the text, nothing about alternatives; much thinner than full-vocabulary KL for a from-scratch student.
- Two-step: ALM self-distillation moves a small pretrained teacher onto our 16K tokenizer, then ordinary KL into the student. Step one is supported at 2-12B (2-6 points lost). Our vocabulary is about 10x smaller than the teachers', so expect a loss between the subword and byte rows.

## Verdict
Test, low priority. It removes the vocabulary-mismatch blocker in principle and is about 50 lines to implement, but the evidence stops at 2B and the one large-to-small gain is about one standard error. Training on teacher-generated text stays the main path.

Settling experiment: 10-20M parameters, 16K vocabulary, same tokens: (a) cross-entropy; (b) plus ALM from a cached off-the-shelf teacher; (c) plus full KL from our own same-tokenizer teacher. Compare bits per byte and BLiMP. Adopt only if (b) beats (a) beyond seed noise.

## Could not verify
- Figure 3's FLOPs and memory values; the text gives no ratio.
- Single-GPU wall-clock.
- Use Case 2 batch size ("doubling to 64" contradicts the default of 64).
- How the "34% additional gap reduction" is calculated.

## Links worth following
- https://github.com/bminixhofer/tokenkit (implementation, ALM notebook)
- Pimentel and Meister 2024, "How to Compute the Probability of a Word" (debiasing step)
- Minixhofer et al. 2024, Zero-Shot Tokenizer Transfer
