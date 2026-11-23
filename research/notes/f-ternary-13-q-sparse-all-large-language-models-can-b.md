# Q-Sparse: All Large Language Models can be Fully Sparsely-Activated
https://arxiv.org/abs/2407.10969 (2024). Hongyu Wang, Shuming Ma, Ruiping Wang, Furu Wei (Microsoft Research / UCAS).
Tags: [P] paper HTML, read through a fetch-and-extract tool (figure curves not readable, so figure-only results have no numbers); [B] follow-up BitNet a4.8 (arXiv 2411.04965), same access; [I] my inference or arithmetic.

## What it is
A training method that makes the *input of every linear layer* sparse by keeping only the top-K activations by magnitude, with a straight-through estimator so masked positions still pass gradient. It is shown for training from scratch, continued training and finetuning, with full-precision and BitNet b1.58 (ternary) weights. It also fits a scaling law for loss versus size and sparsity, and proposes a block-structured variant. It is a training-side paper: no kernel, speed or energy measurement.

## Mechanism
[P] For each linear projection, input X (N x D), weight W (M x D):
```
M   = Top_k(|X|)                      # 1 for the K largest magnitudes, else 0
Y   = (X * M) . W^T                   # full precision; X*M is rescaled by its L2 norm after masking
Q(X)= RoundClip(127/(gamma+eps) * X, -128, 127),  gamma = max|X|
Qw(W)=RoundClip(W/(alpha+eps), -1, 1),            alpha = mean|W|
Y   = (Q(X) * M) . Qw(W)^T            # 1.58-bit version
backward: dY/dX = dY/d(X*M)           # STE: gradient is NOT multiplied by M
```
- [P] FFN uses squared-ReLU GLU: `(X W_up^T) * ReLU^2(X W_gate^T)`, which makes the down-projection input much sparser than K alone.
- [P] Pretrained models without ReLU^2: top-K is applied after the existing activation (SiLU).
- [P] Block Q-Sparse: top-K is taken independently inside each block of M consecutive values; experiments use 50% sparsity, block 32 ("N:M = 16:32").
- [P] The paper does not state whether K is per token; Eq. is written over the whole tensor. [I] Per token is the only reading that works at inference.
- [I] Because gamma = max|X| survives masking, mask-then-quantize equals quantize-then-mask, so top-K can run on int8 values.

## Numbers
- [P] From scratch, 50B tokens RedPajama, 700M and 7B, full precision (Fig. 5) and 1.58-bit (Fig. 6): "40% overall sparsity" matches dense loss. Curves only; no loss values in text.
- [P] Scaling law, 300M-7B: `L(N,S) = E + A(S)/N^alpha`, `A(S) = B + C exp(beta/(1-S))`; reported fit E=1.86, B=0.01, C=1.89, alpha=0.10, beta=0.05 (rounded).
- [P] Inference-optimal sparsity at fixed activated parameters Na: 45.58% full precision (1.84 Na total), 61.25% for 1.58-bit (2.58 Na total).
- [I] Plugging in the fit: at equal total size, S=0.5 raises the reducible loss term by about 5%; at equal activated parameters, the 2x larger sparse model is only about 2-3% better on that term. The gain is real but small.
- [P] Table 1, Mistral 7B continued on 40B tokens FineWeb-Edu, 5-task average: dense 64.6; Q-Sparse 3.8B activated 63.7; 2.9B activated 61.7; ReLUfication (5.0B) 60.8; dReLU (5.4B) 61.0.
- [P] Table 2, per-projection sparsity at 2.9B activated: QKV 51.4, Out 50.0, Up 50.0, Gate 50.0, Down 80.0, overall 58.2%. At 3.8B: Down 60.4, overall 45.7%.
- [P] Table 3/4, finetuned Mistral 7B: dense 66.8; Q-Sparse 4.3B 66.4; Block Q-Sparse 4.3B 65.9, 3.8B 65.0 (versus 65.9 unstructured).
- [P] Ablation, 300M (Fig. 8/9): removing STE or using ReLU instead of top-K "significantly hurt"; with ReLU the sparsity decays during training.
- [B] BitNet a4.8 (100B tokens): top-K 50% only on the attention output projection, ReLU^2 gives >80% sparse down-projection input; overall 42.5% sparse at 700M; C4 perplexity 700M: 12.40 versus 12.32 for b1.58 (FP16 11.44).

## Smallest scale tested, and whether the result held there
[P] 300M parameters (hidden 1024, 24 layers), used for the ablation, the Block variant and the scaling fit. The headline "matches dense" claim is shown at 700M and 7B. Nothing below 300M, about 20-50x our size. Whether a 256-384 wide layer tolerates 40-60% masking is untested; fewer dimensions means less redundancy.

## Training recipe and gotchas
- [P] From scratch, same recipe as BitNet b1.58: LR 1.8e-3 -> 1.5e-3 (300M), weight decay 0.1 -> 0, batch 0.5M tokens, Adam (0.9, 0.95).
- [P] STE is mandatory: without it gradients vanish in the bottom layers (Fig. 2).
- [P] Use top-K, not ReLU thresholds, to keep sparsity fixed.
- [B] a4.8 trains dense-ish first (95B tokens) and switches on the hybrid scheme for the last 5B, reusing optimizer state.
- [P] Text says gate sparsity exceeds 60% while Table 2 shows 50.0/40.0: an inconsistency. "Top-K as 70% giving 40% overall sparsity" is not defined.

## Mapping onto the GBA design
- [I] Our kernel only saves work when all 4 inputs of a table group are zero. With unstructured 50% sparsity that happens for about 1/16 of groups: about 6% saving. Block Q-Sparse spreads non-zeros evenly, which is the wrong structure for us.
- [I] What we need is *group top-K*: score each group of 4 inputs (max or sum of |x|), keep the top-K groups, zero the rest. Then cycles fall in proportion to group sparsity (no table build, no lookups, ROM bytes skipped by pointer stride). Selection over 64-96 group scores via a 256-bin histogram threshold is a few hundred cycles versus about 100K for a 256x256 matrix. This variant is not in the paper.
- [I] The ReLU^2 down-projection input (60-80% zeros) is the best natural target: at 80% element sparsity about 41% of groups are all-zero for free, and a group-level variant would do better. Applies inside each routed expert.
- [I] The L2 rescale needs a divide; fold it into the following norm or use a reciprocal table. RAM cost: none beyond a bitmask (D/32 words). ROM: unchanged; nothing to precompute.
- [I] Overlaps with what we already have (top-8 attention, 1-of-64 experts), which are harsher forms of the same idea.

## Verdict
Test, low priority. The paper's own scaling law says the win at equal compute is a few percent of reducible loss, at 300M+, and the structure we can exploit (group-of-4) is untested. Settling experiment: at our size, train (a) dense width d, (b) width 1.4-2x d with group top-K at 50%, same touched-weight count, plus (c) ReLU^2 MLP with measured group-zero rate; compare validation loss and emulator cycles.

## Could not verify
- Loss values in Figs. 5-8 (curves only). Whether the fitted constants are for full precision or 1.58-bit; the 1.58-bit constants.
- No code release found in the paper. No results below 300M. No speed or energy measurement.
- All of the group-of-4 analysis is my inference.

## Links worth following
- BitNet a4.8: https://arxiv.org/abs/2411.04965 (the hybrid recipe actually shipped).
- BitNet b1.58 recipe: https://arxiv.org/abs/2402.17764
