# Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit pre-training for BitNet language models?  (https://arxiv.org/abs/2502.11895, v1 Feb 2025; Findings of ACL 2025, https://aclanthology.org/2025.findings-acl.694; Jacob Nielsen, Peter Schneider-Kamp, Lukas Galke, Univ. of Southern Denmark)

Tags: [P] paper text (arXiv v1 HTML and ACL PDF, both read in full; same numbers); [F] my reading of Figure 2; [R] linked files (OLMo-1B.yaml, olmo/model.py, bitlinear.py); [D] my derivation; [S] secondary.

## What it is
A training-schedule study: train in 16-bit precision first, then make the same weights the "shadow weights" of ternary quantization-aware training [P]. On OLMo-1B with a fixed 10K-step budget, switching at 20% ends with lower training loss than ternary from step 0 [P]. Two transition aids (keeping the AdamW state; ramping quantization strength lambda) only shrink a temporary loss spike [P]. One model size, one run per condition. Inference is unchanged.

## Mechanism
BitLinear with blend factor lam (Sec. 3.1, 3.4 [P]; scale and norm details [R]):
```
x_n = layer_norm(x)                  # gain-free, inside every BitLinear [R]
x_s = 128 / max|x_n|  (per token)    # 8-bit activations [R]
w_s = 1 / measure(|W|) (per matrix)  # paper: mean; repo default: median
x_q = x_n*x_s + lam*sg(clip(round(x_n*x_s), -128, 127) - x_n*x_s)
W_q = W*w_s  + lam*sg(clip(round(W*w_s), -1, 1) - W*w_s)
y   = (x_q @ W_q.T) / (w_s*x_s)      # sg = stop-gradient
```
Schedule [P]: lam = 0 for t <= s; 2*sigmoid(5(t-s)/(t*-s)) - 1 for s < t <= t*; 1 after. That is tanh(2.5u): 0.46 at u=0.2, 0.85 at 0.5, 0.987 at 1, then a jump to 1 [D].
Variants [P]: default = ramp + fresh optimizer; `-opt` = AdamW state kept; `-nowarm` = hard switch. Switch at 2K, 4K or 6K of 10K steps.

## Numbers
All at OLMo-1B (d_model 2048, 16 layers, 16 heads, tied embeddings [R]), Dolma, 2048 x 2048 tokens per step, 10K steps, about 42B tokens [P; total D].

Training loss at 10K steps (Sec. 5.1, Fig. 2a; smoothed) [P]:

| Run | Loss | Ternary gap closed [D] |
|---|---|---|
| 16-bit throughout | 2.95 | - |
| ternary throughout | ~3.15 | 0 |
| switch at 2K | ~3.088 | 31% |
| switch at 4K | 3.097 | 27% |
| switch at 6K | ~3.12 | 15% |

- Best case: 0.062 nats, about 6% perplexity, over ternary from scratch [D].
- `-opt` and `-nowarm` end at "comparable" loss; no values given [P].
- Transient [F]: the default 2K run spikes above 4.4 for about 100 steps [P], bumps to about 3.55 near step 2.6K, and is below the from-scratch curve by 3K. `-nowarm` spikes off the chart and rejoins after roughly 700 steps. `-opt` has the bump, no spike.
- Table 1, zero-shot [P]: HellaSwag 0.3607 (16-bit), 0.3212 (ternary), 0.3375 / 0.3328 / 0.3262 (switch at 2K / 4K / 6K). Unweighted 11-task means [D]: 0.5369, 0.5333, 0.5365, 0.5277, 0.5224, so the 4K and 6K runs average below ternary-from-scratch; "consistently outperforming" holds for loss and HellaSwag only. The authors call significance unclear [P].

## Smallest scale tested, and whether the result held there
Only 1B parameters; scaling is "left for future work" [P]. A sibling note (arch-02, arXiv 2609.29397) covers a 60K-parameter test [S]: 90% float then 10% ternary beat all-ternary by 2-5% in loss only when the ternary stage used about 10x the pretraining peak learning rate, and was 7-15% worse at 1e-4. So the effect is learning-rate fragile at small scale.

## Training recipe and gotchas
- OLMo defaults [R]: AdamW, lr 4e-4, betas (0.9, 0.95), weight decay 0.1, grad clip 1.0, cosine with 2000 warmup steps, bf16. Same lr for both arms [P, ACL version]; the ternary baseline was not tuned.
- The 2K switch lands on the end of the 2000-step warmup [D from R]: the from-scratch arm warms up under quantization, the winner does not. Untested confound.
- Earlier was better among {2K, 4K, 6K}; nothing below 20% was tried, so the optimum lies in (0, 40%) [D].
- Only `nn.Linear` is ternarized. With tied weights the head is `F.linear` on the embedding, so embeddings and head stay 16-bit [R].
- `replace_modules` copies weights into a layer with an added input LayerNorm [R]; whether the 16-bit phase had it is unstated.
- Hard switch plus fresh optimizer is fine if about 1K or more steps remain [P, F].

## Mapping onto the GBA design
Inference cost: zero lookups, adds, multiplies, RAM or ROM. It changes the training plan only:
- Run the first 10-20% of steps with rounding off (lam = 0) but every other part in final form (norms, activation scaling, top-8 attention, expert routing), so the switch only turns on rounding [D].
- Embedding tables, hashed n-gram tables and the output head are not covered by this evidence (they stayed 16-bit here).
- A "one zero per 4 weights" constraint would switch on at the same step; untested.

## Verdict
Test, then probably adopt. The gain is modest (a third of the ternary gap at 1B), single-seed and confounded with warmup, but free, and the 60K study agrees in direction. Settling experiment at our scale, equal tokens, 2-3 seeds: ternary from scratch with tuned lr and warmup, against float fractions {5, 10, 20, 40}% crossed with stage-2 lr {1x, 3x, 10x}, hard switch, judged on validation loss. Skip the lambda ramp and optimizer-state plumbing.

## Could not verify
- Ramp length (s, t* per run); whether lr warmup restarts at the switch; whether the cosine horizon was 10K or OLMo's 739,328 steps.
- Mean or median weight scale actually used; seeds; validation loss (only smoothed training loss is given).
- 60K-parameter numbers are from the sibling note; I read only that paper's abstract.

## Links worth following
- https://arxiv.org/abs/2609.29397 (small-scale float-then-ternary, stage-2 lr sweep)
- https://huggingface.co/blog/1_58_llm_extreme_quantization (Mekkouri et al. 2024, origin of the ramp; seen via automated summary only)
- Kumar et al., "Scaling laws for precision", ICLR 2025 (cited for "overtrained" models quantizing worse; no URL in the paper)
