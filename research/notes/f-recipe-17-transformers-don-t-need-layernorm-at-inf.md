# Transformers Don't Need LayerNorm at Inference Time: Scaling LayerNorm Removal to GPT-2 XL

https://arxiv.org/abs/2507.02559 (v1 July 2025; ICLR 2026 poster). Baroni, Khara, Schaeffer, Subkhankulov, Heimersheim. Code: https://github.com/submarat/removing-layer-norm ; models: https://huggingface.co/schaeff/models

## What it is
A fine-tuning protocol that removes every LayerNorm from pretrained GPT-2 small/medium/large/XL by replacing the per-token division by the standard deviation with a frozen scalar, one LN at a time, while continuing to train on OpenWebText. The result is a model whose "norm" is a fixed affine map. The motivation is interpretability (exact direct logit attribution), not efficiency. It extends Heimersheim 2024 (arXiv 2409.13710, GPT-2 small only). It does not train LN-free from scratch and does not touch quantisation.

## Mechanism
Per token (batch b, position s, width H):
- sigma_{b,s} = sqrt( (1/H) sum_h (x_h - mu)^2 + eps )
- sigma_avg = mean of sigma_{b,s} over all B x S tokens (one scalar per LN, not per position)
- FakeLN(x) = (x - mu) / sigma_avg * gamma + beta   (Eq. 2; mu, gamma, beta kept)

Before removal, sigma_avg is recomputed every batch (EMA for Large/XL, where micro-batches are small); at the removal step it is frozen forever.

Schedule (Sec. 3, Table 4): plain fine-tune warm-up; then remove LN_MLP layer 0..L-1 with gap g_mlp steps; then LN_qk (the LN feeding query/key) with gap g_qk; then LN_v (the LN feeding values) with gap g_v; final LN last. The attention LN is split into a qk copy and a v copy so they can be removed separately.

| | Small | Medium | Large | XL |
|---|---|---|---|---|
| gaps g_mlp / g_qk / g_v | 2/2/3 | 2/2/3 | 4/4/6 | 4/4/6 |
| LN_MLP start | 20 | 20 | 30 | 50 |
| LN_qk start | 44 | 68 | 174 | 242 |
| LN_v start | 68 | 116 | 318 | 434 |
| final LN | 104 | 188 | 534 | 722 |

Auxiliary loss (Eq. 3), applied at the final LN only: L_aux = lambda * E[(sigma_{b,s} - sigma_hat)^2], where sigma_hat is the mean sigma over tokens excluding position 0 and end-of-text tokens (those have much larger norms in GPT-2). It pulls every token's std to one value so that a single constant is a good substitute.

Hyperparameters (Table 3): about 0.52M tokens per step (micro-batch 32 x 16 accumulation for Small; implied context 1024); weight decay 0.01; LR 6e-4 (Small, Medium), 3e-4 (Large), 1e-4 (XL), cosine to 3e-4 / 3e-4 / 4e-5 / 2e-5; warm-up 25/10/15/20 steps; lambda 0.1/0.1/0.03/0.01.

## Numbers
Table 1, mean cross-entropy (original / vanilla fine-tune / LN-free, same step count):
- Small, 300 steps: OWT val 3.1006 / 3.0126 / 3.0797 (+0.067); Pile 2.8450 / 2.8511 / 2.8852; Pile-filtered 2.7899 / 2.8112 / 2.8757.
- Medium, 500: OWT 2.8145 / 2.7390 / 2.7642 (+0.025); Pile 2.5163 / 2.5752 / 2.6579.
- Large, 600: OWT 2.6623 / 2.6240 / 2.6384 (+0.014); Pile 2.5320 / 2.6233 / 2.7504.
- XL, 800: OWT 2.5567 / 2.4799 / 2.5052 (+0.025); Pile 2.4436 / 2.4673 / 130.2197; Pile-filtered 2.3739 / 2.3821 / 2.3992.

The XL Pile blow-up comes from "a couple of" sequences; median and 95th percentile are similar to the original. Medium output entropy drops from 2.86 to 2.53 (overconfidence; all LN-free models show it). The gap to the vanilla fine-tune stays roughly constant with longer fine-tuning. Compute (Table 2): 1.5 GPU-h Small vs 1 vanilla; XL 26 vs 14. Predecessor (Heimersheim 2024): +0.05 CE on GPT-2 small after 500M tokens, -0.5% HellaSwag.

## Smallest scale tested, and whether the result held there
GPT-2 small (124M, 12 layers, 768 wide). It held but with the largest OWT gap of the four sizes (+0.067). Nothing below 124M, nothing narrower than 768, no quantised model, no seeds. The paper cites Nabeshima's tiny models trained LN-free from scratch and expects from-scratch training to work "only for small language models", which is our regime, but gives no numbers.

## Training recipe and gotchas
- LN_v removal is the dangerous step: cascading exploding gradients. It always needed gaps; LN_MLP and LN_qk could sometimes be removed in all layers at once (Sec. 6.2).
- Some runs failed irrecoverably; keep checkpoints and restart. Mitigations per the authors' LessWrong post: lower LR, bigger batch, EMA of sigma, wider gaps.
- Without the auxiliary loss the method worked on Small only.
- Position-0 handling matters: a bug there made removal rougher (LessWrong post).
- A frozen sigma that mismatches the true std is the stated cause of failures; freeze at the moment of removal, not from a precomputed table.
- Mean-centring, gamma and beta fold into neighbours (TransformerLens fold_ln). The paper does not spell out folding 1/sigma_avg; that it folds is my inference from it being a constant.

## Mapping onto the GBA design
My estimates, not from the paper. A run-time RMS norm at width 384 costs roughly 384 squares-accumulate (about 2.9K cycles at 7.5), one reciprocal-sqrt via ROM table, and 384 scale multiplies (about 2-3K): about 5-6K cycles. With 2 norms x 6 layers + 1 final that is about 70K cycles per token, 1.5-2.5% of a 3-5M budget. So the compute saved is small; the gain is simplicity (no rsqrt table, no 32-bit sum of squares) and static, analysable fixed-point ranges.

With a frozen scalar the norm costs zero operations: 1/sigma_avg merges with the ternary layer's scale alpha and the activation quantiser's step into one constant, realised as a shift (or one multiply per output if not a power of two; it can be trained toward a power of two). ROM: one constant per norm. RAM: none.

What must change:
- Per-channel gamma cannot fold into ternary weights; use scalar or no gain (as RMSNorm-in-BitNet setups already do), and drop mean subtraction from the start so nothing needs centring.
- The quantiser before each ternary layer must use a fixed clip range (not per-token absmax), otherwise a dynamic scale re-enters through the back door.
- Top-8 selection and expert argmax are scale-invariant, but the softmax over the 8 survivors and the output head are not: the paper's overconfidence finding predicts over-sharp attention and logits. With saturating 8-bit activations a blow-up becomes clipping rather than loss 130, which may be a natural guard, but is untested.
- Add the aux loss on every residual read point, not only the final one, since each of our constants is a hard shift.

## Verdict
Test, low priority. It is a valid route to a divide-free norm, but (a) the run-time norm it removes costs only about 2% of our budget, (b) the smallest model shows the largest penalty and that penalty did not shrink with more training, (c) nothing is known under ternary QAT. Settling experiment: train the candidate with RMSNorm + QAT, then run the removal schedule (MLP norms, then qk, then v, then final; aux loss on) for about 5% extra steps; compare val CE, entropy, and the 99.9th-percentile per-sequence loss against (i) the normed model and (ii) a cheap run-time norm using mean-absolute-value and a ROM reciprocal table. Adopt only if the CE gap is under about 0.02 and tails are clean.

## Could not verify
- Read through a fetch tool that summarises the arXiv HTML (both halves, about 155K chars); numbers are as that extraction reported them, not eyeballed in the rendered tables.
- Optimizer, EMA decay, gradient clipping: not found in the text.
- Table 2 gives 600/800 fine-tune steps for Large/XL while Table 3 gives max steps 1,200 for both; unexplained.
- Per-layer sigma_avg values: not reported.
- I did not read the GitHub repo or the Hugging Face model cards; URLs come from the authors' LessWrong post (secondary to the paper, same authors).
- Heimersheim 2024 numbers are from its abstract page only.

## Links worth following
- https://github.com/submarat/removing-layer-norm (schedule configs, position-0 handling)
- https://arxiv.org/abs/2409.13710 (predecessor, GPT-2 small)
- https://github.com/noanabeshima/tinymodel (tiny LN-free models trained from scratch; closest to our scale)
- Dynamic Tanh, Zhu et al. 2025 (concurrent; element-wise tanh instead of norm, table-lookup friendly)
