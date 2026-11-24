# Cross-layer Attention Sharing for Pre-trained Large Language Models - LiSA (https://arxiv.org/abs/2408.01890, arXiv v2 17 Oct 2025, TACL; Yongyu Mu, ..., Tong Xiao, Jingbo Zhu; Northeastern Univ. Shenyang / WeChat AI)

Read in primary form (arXiv HTML v2, full text and appendix). Figures are images: only captions and prose were readable.

## What it is
An analysis-plus-method paper. Analysis: in 7-70B LLMs the attention distributions of adjacent layers are nearly identical, but only after heads are matched, and early layers break when their attention is perturbed. Method (LiSA): a retrofit for frozen pretrained LLMs in which a layer reuses the previous layer's pre-softmax scores, passed through a tiny head-mixing FFN plus a low-rank Q/K correction. A small side experiment trains 12-layer, 164M models from scratch with sharing in 8 of 12 layers.

## Mechanism
Shared object: pre-softmax score tensor A_{n-1} of shape (h, l, l). V and output projection stay per-layer.

```
# layer n is a LiSA layer
Qlr = H @ Wq_lr ; Klr = H @ Wk_lr          # Wq_lr, Wk_lr in R^{d x (h*r)}, r << d_k
A_delta = Qlr Klr^T / sqrt(r)               # (h, l, l)            Eq. 3
x = concat(A_{n-1}, A_delta) over head axis # per (i,j): 2h-vector
A_n = FFN(x)                                # 2h -> 256 -> h, ReLU  (LiSA)
                                            # or single linear 2h -> h (LiSA_SL)
P_n = softmax(A_n); out = P_n V_n
# "LiSA_plus" (from-scratch variant): A_n = A_{n-1} + A_delta, no FFN
```

- Sizes (App. C.1), h=32, d=4096, d_k=128: FFN 64x256, 256x32; W_lr 4096x640 (r=20 per head, "6x"). LiSA_SL: FFN 64x32, r=32. Chains allowed (A_n feeds n+1).
- Layer choice (Sec. 5.1): skip first and last few layers; rank layers by robustness to direct sharing on 100 GSM8K items; alternate (e.g. 2 LiSA layers then 1 standard). LLaMA3-8B LiSA(17) = layers 5,6,17-31 of 32 (1-indexed); LiSA(27) = 4-30 (Table 1).
- Loss: L = beta * Huber(A_n, teacher A_n*; delta=1), averaged over LiSA layers, + (1-beta) * LM loss; beta=0.25 (Eq. 5-7).

## Numbers
Analysis (LLaMA3-8B, 100 samples per task, last-token attention):
- Head-averaged JS divergence between layer pairs mostly ~0.1; adjacent layers <0.05 (Fig. 3). Layers 1-2 and 11-14 differ from the rest.
- Per-head, same-index matching: mean JS ~0.2, same as random pairing; best-match pairing: <0.1 in most layers (Fig. 5b-d).
- Sharing or uniform attention on one adjacent pair: shallow pairs collapse, deep pairs tolerate even uniform attention. To keep 90%: PIQA needs own attention in shallow layers, MMLU in the first half, GSM8K in most layers (Fig. 6).

Retrofit, LLaMA3-8B, Table 2 (average of 12 benchmarks preserved): direct sharing DS(17) 46.67% (LAMBADA ppl 3.48 -> 936); DS + LoRA(17) 67.83%; LiSA(17) 96.77% (ppl 3.79, GSM8K 51.71 -> 45.94); LiSA(21) 92.63%; LiSA(27) 88.38% (GSM8K 31.77, TriviaQA 63.39 -> 43.65).

Ablation, DS(21), 1B tokens, Table 7 (BoolQ / GSM8K; base 81.13 / 51.71): DS 40.46 / 2.65; +align 74.34 / 7.96; +low-rank diff only 37.86 / 0.68; both 76.85 / 26.84. Table 8: additive "Plus" integration 68.72 / 6.14, i.e. clearly worse than FFN mixing in the retrofit setting.

Speed: throughput +10.1-19.5% on LLaMA3-8B, up to 40.1% on 13B (Table 3, 80 GB A800); the gain comes mainly from a smaller K cache (Sec. 5.2), not fewer operations.

From scratch, 12 layers x 12 heads x d_k 64, 164M params, 10B tokens, sharing in layers 2,3,4,6,7,8,10,11 (Table 5; OBQA / HellaSwag / PIQA / BoolQ / WinoGrande / ARC-E): MHA 24.20 / 28.95 / 58.49 / 61.47 / 52.01 / 35.44; DS(8) 25.20 / 27.68 / 58.71 / 61.10 / 52.49 / 34.18; LiSA_plus(8) 26.20 / 27.92 / 58.65 / 62.02 / 50.20 / 35.14. Authors call both "lossless".

## Smallest scale tested, and whether the result held there
164M parameters, 12 layers, trained from scratch on 10B tokens. Direct sharing (no alignment module, no low-rank term) in 8 of 12 layers matched the baseline on six benchmarks. Caveat: those scores are near chance (4-way OBQA ~25, HellaSwag ~28, WinoGrande ~52), so the table cannot detect moderate losses; the eval-loss curves are in Fig. 15, which I could not read numerically. Nothing below 12 layers.

## Training recipe and gotchas
- Retrofit: base frozen, 0.46-1.64% of parameters trained; 4.2B tokens (RedPajama subsets), batch 128, lr 3e-4, 40,000 steps, 8x A800.
- Head alignment is a retrofit problem only: when trained from scratch with sharing, heads are aligned by construction (Sec. 5.5).
- Layer 1 always computes its own attention; the from-scratch pattern is compute, share x3, compute, share x3, compute, share x2, compute.
- Damage concentrates in multi-step reasoning and exact recall, not multiple-choice tasks.

## Mapping onto the GBA design
My estimates, not the paper's. With sign/ternary keys and top-8 mixing, a shared layer would skip its query projection, the 256-position scan and top-8 selection, and its key ring (256 x 64 bits = 2 KB per head, which matters against 32 KB fast RAM). Reused state is 8 (position, score) pairs per head. The paper's dense per-pair FFN is not affordable on all 256 positions (2h x h multiplies each at 7.5 cycles), and unnecessary when training from scratch. The LiSA_plus analogue fits: rescore only the 8 inherited candidates with a short per-layer key (r ~ 8-16 bits per head, 8 lookups/popcounts), add to the inherited score, renormalize with a ROM exp table. Nothing context-dependent is precomputable into ROM. With 4-6 layers the from-scratch pattern suggests at most: layer 1 own scan, layer 2 shares, layer 3 own scan, layer 4 shares, last layer own scan.

## Verdict
Test. This is the only from-scratch, GPT-2-small-scale evidence found that direct sharing in two-thirds of layers costs nothing measurable, but the measurement is weak and the model is 2-3x deeper than ours. Experiment: 6-layer QAT baseline versus (a) layers 2 and 4 reuse top-8 indices and weights, (b) same plus low-rank rescoring, with saved cycles re-spent on MLP width; compare held-out perplexity and a copy/recall probe (names, numbers from earlier turns), since exact recall is where LiSA lost most.

## Could not verify
- Fig. 15 eval-loss values; all heatmap/line-plot values beyond those quoted in prose.
- The appendix top-5-token table (did not render); the repository (not opened).
- Whether from-scratch sharing holds below 12 layers or with sparse/top-k or quantized attention: not tested in the paper.
- TACL year/volume (acknowledgements thank a TACL action editor; not checked further).

## Links worth following
- https://github.com/takagi97/lisa (official code)
- Xiao et al. 2019, "Sharing Attention Weights for Fast Transformer" (SAN): from-scratch sharing in small NMT models; closest to our regime.
- EchoAtt, https://arxiv.org/abs/2409.14595: sharing with distillation on TinyLlama-1.1B.
