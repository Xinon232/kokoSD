# Four Kilobytes of Memory: A 10M-Parameter Fixed-State Recurrent LM for BabyLM Strict-Small

Paper: https://aclanthology.org/2026.babylm-main.40.pdf (Sam Furr, BabyLM Workshop, Oct 2026, pp. 514-524). Model + PyTorch source: https://huggingface.co/haybales/statehead-10m (MIT). Both read in primary form (full PDF, `config.json`, `modeling_statehead.py`).

## What it is
StateHead is an attention-free, position-embedding-free causal LM: each block is a gated diagonal linear recurrence (QRNN "ifo" pooling with width-1 convolution) plus a GELU MLP. The 9.97M-parameter model (SH-10M) carries L*d = 2,160 state values per sequence, regardless of context length. Trained 1.1 h on a laptop on 10M words, it beats the 98.4M GPT-2 BabyLM baseline on BLiMP only; it is at or below chance on semantic and entity-tracking tests. The paper is candid and empirical, single seed, no ablations.

## Mechanism
Pre-norm residual block (Eq. 1-2, Fig. 1), x_t in R^d:
```
u = Wg * RMSNorm(x_t) + bg          # Wg: 4d x d, split into a,b,c,o
a = sigmoid(u_a); b = sigmoid(u_b); c = tanh(u_c); o = sigmoid(u_o)
s_t = a .* s_{t-1} + b .* c          # elementwise, s in R^d
y_t = o .* s_t
x = x + Wp*y_t + bp                  # Wp: d x d
x = x + Wdown * gelu(Wup * RMSNorm(x))   # d -> r*d -> d
```
Final RMSNorm, tied embedding/output matrix, no bias on the LM head. Retention bias init 2.0 (sigmoid 0.88), other gate biases 0; learned initial state per layer (init zero). No sqrt(1-a^2) write scaling (flag exists, off). "Heads" (H = 6, d_h = 60) do not change the forward maths; they only group rows for the optimizer.
SH-10M: L=6, d=360, r=2, V=8,192 BPE, dropout 0 (Table 2). SH-98M: L=12, d=744, H=12, r=4, V=16,384.

## Numbers
- Table 1: forward 20.0M FLOPs/token (SH-10M) vs 233.6M (GPT-2); recurrence + gate activations are 0.22% of it. State 4.2 KiB at 2 bytes/value vs 36.0 MiB KV cache at 1,024 tokens. Training 9.7e15 FLOPs, 1.1 h, 39.8k tokens/s on an M5 MacBook Air.
- Table 4 (full sets, July 2026 pipeline), GPT-2 98.4M / SH-10M / SH-98M: BLiMP 65.2 / 68.2 / 69.9; BLiMP Supplement 57.3 / 55.3 / 56.9; EWoK 50.6 / 50.6 / 50.1 (chance 50); entity tracking 19.1 / 18.8 / 18.4 (chance 20); COMPS 51.8 / 50.8 / 50.9; reading time ET/SPR 9.6/1.6 vs 0.2/0.1; SH-10M GLUE mean 56.6 (majority-class behaviour on several tasks).
- Fig. 2 / Table 6 (fast subsets): SH-98M BLiMP peaks 72.6 at 40M exposures, falls to 67.0 at 100M; SH-10M peaks 69.0 at 50M, ends 67.8.
- Table 5: carrying state across training windows raised old entity-tracking from 17.3 to 40.3, but only on items whose answer is "nothing" (15.2 -> 94.9); retained items 18.1 vs 18.0. On the revised set, reset 17.8, carry 18.8.
- Appendix C: training windows of 512 / 1,024 / 2,048 gave retained-item entity accuracy 17.1 / 15.9 / 16.5 (98M config).
- No perplexity, bits-per-token or generation samples are reported anywhere.

## Smallest scale tested, and whether the result held there
9.97M parameters, 10M-word corpus, 161.2M token exposures. The BLiMP result holds there (68.2, single seed, no variance). Nothing smaller, nothing quantized.

## Training recipe and gotchas
- 10 epochs, window 1,024 tokens, 32 accumulated windows = 32,768 tokens/update, 4,920 updates, cosine with 200 warmup steps (Table 3).
- Muon on hidden matrices and initial states: lr 2e-2 -> 2e-3, momentum 0.95 Nesterov, 5 Newton-Schulz iterations, applied per head to 4*d_h x d slices of Wg. AdamW on embeddings, norms, biases: 5e-4 -> 5e-5. Weight decay 0.1.
- bf16 with fp32 scan accumulation, scan chunks of 64.
- SH-10M resets state every window; state carry was dropped as unproven.
- Late-training BLiMP decline, worse for the bigger model. Companion paper 2026.babylm-main.1 (abstract only) reports the same for RWKV-7: -2.57 to -3.85 points by step 18k in all three seeds, mitigated by halving updates / lowering LR earlier. Checkpoint averaging (60M/80M/100M) was used for SH-98M.
- Counter-evidence, 2026.babylm-main.7 (abstract only): in 432 controlled runs autoregressive Transformers were best at every scale; Gated DeltaNet gained on entity tracking but lost on grammar.

## Mapping onto the GBA design
(My arithmetic, not the paper's.)
- Weights touched per token per layer: 4d^2 (gates) + d^2 (Wp) + 2r*d^2 (MLP). At d=360, r=2: 1.17M per layer, 7.0M for six layers, about 11M cycles at 1.6 cycles/weight, before the output head. That is 2-4x over budget as published.
- Fits only if shrunk: d=256, L=4, r=2 is 9d^2*4 = 2.36M weights (about 3.8M cycles) with the MLP dense; with our 1-of-64 routed MLP the dense MLP cost is unchanged per token, so the gate matrix (4d^2, 44% of a block) is the thing to attack: couple a = 1-b and drop o (2d^2), or compute gates at reduced rank.
- Recurrence itself: 3*L*d multiplies (6,480 at SH-10M, about 49K cycles) plus 4*L*d table lookups for sigmoid/tanh. Negligible.
- RAM: L*d 16-bit values = 4.3 KB (2-3 KB at d=256, L=4-6). A typing checkpoint is one memcpy of that. By contrast a 256-token ring of full-width 8-bit values plus sign keys is about 256*(d + d/8)*L bytes, roughly 290 KB at d=256, L=4, which would not fit 256 KB unless values are narrowed.
- Fixed point: an 8-bit retention gate caps the time constant near 256 tokens; keep state 16-bit to avoid truncation leak in a.*s. Untested in the paper.
- ROM: nothing new to precompute; initial states are L*d constants.

## Verdict
Test, as a component, not as the whole mixer. It proves a 4 KiB state buys syntax (BLiMP) cheaply, but it shows no evidence of binding, copying or recall, which a chatbot needs (names, the user's last sentence), and the controlled companion study favours attention. Settling experiment: at fixed cycles/token, train (a) the current top-8 attention design, (b) StateHead mixer in all layers, (c) StateHead in lower layers plus one or two attention layers; compare held-out bits per token, a name-copy probe and BLiMP, all under ternary QAT.

## Could not verify
- The code repository URL (linked in Appendix A, but the hyperlink target was not recoverable from the PDF pages I read).
- Perplexity or generation quality: not reported.
- Companion papers main.1 and main.7 were read as Anthology abstracts only (secondary-level detail).
- Model-card extras (torch.compile 1.5-1.8x) came via a summarised fetch of the card.

## Links worth following
- https://aclanthology.org/2026.babylm-main.1/ (late-training degradation, RWKV-7)
- https://aclanthology.org/2026.babylm-main.7/ (controlled token-mixer study)
- Jelassi et al. 2024, "Repeat after me: Transformers are better than state space models at copying" (cited; the copying limit)
- De et al. 2024, Griffin / RG-LRU, arXiv:2402.19427 (recurrence + local attention hybrid)
