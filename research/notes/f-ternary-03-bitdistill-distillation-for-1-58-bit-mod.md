# BitNet Distillation (BitDistill)  (https://arxiv.org/abs/2510.13998, Oct 2025; Xun Wu, Shaohan Huang, Wenhui Wang, Ting Song, Li Dong, Yan Xia, Furu Wei — Microsoft Research)

Read: arXiv HTML v1, through a fetch tool that returns a model-made digest of the page (not raw text). Numbers below are from that digest of the primary source; equations are reconstructed from its description and from my prior knowledge of BitNet/MiniLM, and are marked where so.

## What it is
A three-stage recipe to convert an already-trained full-precision (FP16) LLM into a ternary (1.58-bit) BitNet for one downstream task, without the ~4T-token from-scratch BitNet pretraining. Stages: (1) add extra normalisation layers (SubLN), (2) a short "continue pre-training" of the ternary model on 10B tokens, (3) task fine-tuning with logit distillation plus MiniLM-style attention-relation distillation from the FP16 fine-tuned teacher. It closes a 14-15 point accuracy gap between naive ternary fine-tuning and FP16 on GLUE tasks at 0.6B-4B. It is task-specific (classification, summarisation), not a general chat model result.

## Mechanism
Quantisers (standard BitNet b1.58, Eq. 1-3):
- Weights, per tensor: `D = mean(|W|); Wq = D * clip(round(W/(D+eps)), -1, 1)`.
- Activations, per token, 8-bit absmax: `g = max(|x|); xq = (g/127) * clip(round(127*x/(g+eps)), -128, 127)`.
- STE through both.

Stage 1, SubLN (Eq. 4-6): keep the pre-norm blocks, and insert one more norm inside each sublayer, immediately before its output projection:
```
attn_out = W_o( SubLN( concat(heads) ) )
ffn_out  = W_down( SubLN( act(W_gate x) * (W_up x) ) )     # Qwen3 gated FFN
```
Purpose given: the input to a ternary projection otherwise has very large variance, which destabilises training. (Norm type inside SubLN: not confirmed; RMSNorm in BitNet practice — my assumption.)

Stage 2: plain next-token loss on the ternary student, initialised from FP weights, 10B tokens of the "FALCON corpus". Observed effect (Fig. 2): latent weights move from Gaussian to a BitNet-like shape with more mass near the 0/±1 rounding boundaries, so small updates can flip ternary values.

Stage 3: `L = L_CE + lambda*L_LD + gamma*L_AD`
- `L_LD = KL(softmax(z_T/tau) || softmax(z_S/tau))`, tau = 5.0. (tau^2 scaling: not confirmed.)
- `L_AD` (MiniLM-v2 style, Eq. 10-12, Algorithm 1): at ONE chosen layer, for each of the Q, K, V states: split into `split_heads` relation heads, L2-normalise, form `R = softmax(A A^T / sqrt(d_r))` (a T x T token-token distribution), then `KL(R_T || R_S)` averaged over heads and positions, summed over Q, K, V with alpha_i = 1.0. Because R is T x T, teacher and student widths need not match.
- lambda = 10, gamma = 1e5 (classification); lambda = 1, gamma = 1e3 (summarisation).
- Distil a single layer, a late one; all-layer distillation was worse (Fig. 3b).

## Numbers
All Qwen3 unless noted; FP16-SFT teacher / naive BitNet-SFT / BitDistill.
- Table 1, MNLI acc: 0.6B 88.01 / 74.09 / 88.17; 1.7B 89.61 / 75.27 / 89.53; 4B 91.48 / 76.11 / 91.40.
- Table 1, QNLI: 0.6B 93.72 / 78.32 / 93.66. SST-2: 0.6B 94.21 / 79.92 / 94.30.
- Naive gap widens with size: 13.9 points at 0.6B to 15.3 at 4B (intro).
- Table 2, CNN/DM at 0.6B: BLEU 13.98 / 11.47 / 14.41; ROUGE-L 27.72 / 24.84 / 27.49.
- Table 3, MNLI: Gemma3-1B 89.77 / 78.02 / 89.61; Qwen2.5-0.5B 79.91 / 60.80 / 79.98.
- Table 5 ablation, 0.6B, MNLI: none 74.09; +SubLN 76.30; +SubLN+continue-pretrain 86.73; +SubLN+distill (no continue-pretrain) 88.04; all three 88.17. CNN/DM BLEU for the same rows: 11.47, 11.69, 13.96, 13.70, 14.41.
- Table 6, MNLI (stages 1+2 present): no distill 86.73; logits only 87.32; attention only 87.67; both 88.17.
- Table 4: weight quantiser choice barely matters (88.05-88.25 MNLI).
- Efficiency: 427 vs 1,135 tok/s on CPU, 16 threads; 1.20 vs 0.11 GB (abstract: 2.65x, 10x).
- Fig. 3c: a 1.7B or 4B teacher improves the 0.6B student beyond the same-size FP16 model (values only in the figure; not read).

## Smallest scale tested, and whether the result held there
Qwen2.5-0.5B (Table 3) and Qwen3-0.6B. It held: student matches teacher within 0.2 points. Nothing below 0.5B was tested, and nothing trained from scratch. Our 50-120M-weight model is 5-10x below the smallest point, and has no pretrained FP checkpoint to start from.

## Training recipe and gotchas
- Max length 512, batch 32, 8x AMD MI300X. LR and epochs chosen by "greedy search"; values, optimiser and warmup not given.
- The teacher is the FP16 model fine-tuned on the same task; the student is initialised from the FP checkpoint.
- Most of the gain on classification comes from stage 2 or stage 3 alone (86.7 and 88.0 of 88.2); SubLN alone gives only +2.2. For generation (CNN/DM), both stages matter more evenly.
- Layer choice matters a lot for L_AD ("results vary significantly"); the layer index is not stated in the text.
- gamma is huge (1e3-1e5) because relation KL values are tiny; expect to retune.
- Algorithm 1 has inconsistent variable names (`split_head`/`split_heads`, undefined temperature), so do not copy it blindly.
- Evidence is classification accuracy and ROUGE; no perplexity or open-ended chat quality.

## Mapping onto the GBA design
- Stages 2 and 3 are training-only: zero run-time cost, zero ROM.
- SubLN is the only inference cost. Per norm over n values: n multiplies for sum of squares, one reciprocal-sqrt (ROM table, no divide), n multiplies to rescale. At ~7.5 cycles per MAC: width 320 attention output ~4.8K cycles; a 1024-wide expert hidden layer ~15K. About 20K per layer, ~120K for 6 layers, i.e. 2-4% of a 3-5M budget (my estimate). Cheaper: fold SubLN into the activation quantiser that already has to run before each ternary matrix (the norm then only sets the 8-bit scale), and use a power-of-two shift instead of an exact rsqrt; this must be trained that way.
- Per-token absmax activation scaling needs a max scan and a divide; replace by a shift or a table reciprocal.
- Relation distillation is width-agnostic, so a wider FP teacher of our own architecture can supervise the ternary student. With top-8 hard attention the student's Q/K/V states still exist, so the loss is defined; ternary/sign keys make the K relation coarse — untested.
- Suggested pipeline: train an FP twin (same routing, same vocabulary) -> insert SubLN -> ternary continue-pretraining on the full corpus -> distil from the FP twin (or a larger teacher) with L_LD + single-late-layer L_AD.

## Verdict
TEST. The ingredients are cheap and the ablations are clean, but every number is task-specific fine-tuning at >=0.5B from a pretrained checkpoint. Settling experiment: at our target size on the simple-English corpus, compare held-out perplexity for (a) ternary QAT from scratch, (b) FP twin -> ternary continue-training, (c) b + logit KD, (d) c + attention-relation KD at one late layer, each with and without SubLN, at equal GPU-hours.

## Could not verify
- Raw equation text and Algorithm 1 (digest only); tau^2 factor; the norm type in SubLN; which layer was distilled; LR/epochs/optimiser.
- Figure 3 numeric values (layer sweep, teacher-size gains).
- What "FALCON corpus" is; no citation was resolved.
- The paper's text says "up to 2x" speedup in Sec. 4.2 while the abstract says 2.65x (table ratio is 2.66x).
- Whether BitDistill training code is actually in github.com/microsoft/BitNet (the paper says so; repository not checked).
- MiniLM details are from my memory of that paper, not re-read.

## Links worth following
- MiniLMv2 (multi-head self-attention relation distillation): https://arxiv.org/abs/2012.15828
- Code location claimed by the paper: https://github.com/microsoft/BitNet
- BitNet b1.58 2B4T technical report (the from-scratch baseline; already in notes f-ternary-02).
