# Approximating Two-Layer Feedforward Networks for Efficient Transformers (https://arxiv.org/abs/2310.10837, EMNLP Findings 2023; Csordás, Irie, Schmidhuber; code https://github.com/robertcsordas/moe)

Tags: [P] read in the arXiv HTML full text (via a fetch tool that summarises, so numbers were cross-checked over three passes); [C] my arithmetic; [U] unresolved. The task title "Neurons in large language models" is a different paper; 2310.10837 is this one.

## What it is
A unified view of three ways to approximate a transformer's 2-layer MLP as a key-value memory: top-K activation, product-key memory (PKM) and mixture of experts (MoE). All are compared at parameter parity with dense Transformer-XL baselines (47M and 262M on WikiText-103, C4, peS2o; 41M on Enwik8). The paper proposes σ-MoE: sigmoid (non-competitive) gating, entropy regularisation, expert dropout and a changed init. It is the best primary evidence at about 50M that an MoE FFN can match dense at 12.5-25% of FFN compute, and it contains a K=1 ablation.

## Mechanism
Dense: u = ReLU(W1 x), y = W2 u [P].
- Top-K: keep the K largest u, zero the rest; y = sum of those K columns of W2.
- PKM: sub-key matrices W_a, W_b of shape sqrt(d_ff) x d_model/2; u[i] = u_b[i div sqrt(d_ff)] + u_a[i mod sqrt(d_ff)]; then top-k. Their fix: ReLU on the scores instead of softmax. 4 heads, no dropout.
- σ-MoE, N_E experts of G hidden units each, K active:
```
s  = sigmoid(W3 x)                 # N_E scores, no competition
E  = argtopk(s, K)
y  = sum_{e in E} W2_e * s[e] * ReLU(W1_e x)
L_reg = gamma * sum_e p[e] log p[e],  p = mean over batch of softmax(W3 x)
train: s = sigmoid(W3 x) * m,  m ~ Bernoulli(1-delta), no rescaling (expert dropout)
init: W1_e ~ N(0, sqrt(2/(d_model*n_layers))), W2_e ~ N(0, sqrt(2/(d_ff*n_layers)))
      # uses full d_ff = N_E*G, not G; W3 rows rescaled to equal norm at init
```
Settings (Tables 8-9) [P]: 47M = 16 layers, d_model 412, d_ff 2053, 10 heads, context 256, batch 64; MoE N_E 16, G 128, K 4, delta none, gamma 0.001. 262M = 18 layers, d 1024, d_ff 4110; N_E 32, G 128, K 4, delta 0.2. WT-S* (238M) = the 47M model with N_E 128, K 4, delta 0.05. Enwik8 41M: 12 layers, d 512, N_E 16, delta 0.05, gamma 0.0001.

## Numbers
All [P]; WikiText-103 perplexity unless noted.
- Main (Table 3): 47M dense 11.81 vs σ-MoE 11.71 at 25% FFN FLOPs; 262M 9.46 vs 9.44 at 12.5%; C4 47M 23.76 vs 23.25; peS2o 47M 14.34 vs 14.12; Enwik8 1.08 vs 1.08 bpc.
- Stored capacity at fixed compute (Table 4): σ-MoE N_E 16 -> 128 with K=4: 11.59 -> 10.37 (-0.111 nats [C]) while FFN compute stays that of 4x128 units (3.1% of the equivalent dense FFN, Table 7). Parameter-matched dense (d_ff 16480): 10.03, so the sparse model is 0.033 nats behind [C].
- K at equal active units (Table 4; WT-S / WT-S* / WT-B): K=4,G=128: 11.59 / 10.37 / 9.44; K=8,G=64: 11.63 / 10.30 / 9.58; K=2,G=256: 11.84 / 10.44 / 9.56; **K=1,G=512: 11.90 / 10.83 / 9.58**. Top-1 costs 0.026 and 0.043 nats [C].
- Routing ablations (same columns): Switch Transformer 12.27 / 11.24 / 9.68; Switch-style K=1,G=512 12.32 / 11.31 / 9.77; S-BASE 13.01 / 10.96 / 10.50; softmax renormalised 11.89 / 11.27 / 9.58; no regularisation 11.83 / 10.41 / 9.51; standard init 11.80 / 10.59 / 9.67; plain dropout instead of expert dropout 12.01 / 10.27 / 9.53.
- Collapse (Figure 3, layer 5 of WT-S*): Switch and renormalised softmax collapse; σ-MoE is as balanced as S-BASE.
- Top-K (Table 1), 47M: dense 11.81; K=64 11.86, 128 11.74, 256 11.74, 512 11.68 of 2053 units. 262M: 9.46 -> 9.26 at K=128.
- PKM (Tables 2, 6), 47M: softmax 13.96, ReLU 12.77 (parameter-matched, 62 sub-keys); 14.11 / 13.32 at 46 sub-keys. Always worse than dense.

## Smallest scale tested, and whether the result held there
47M (16 layers, d 412) and 41M character-level. The result held: σ-MoE equals or slightly beats dense at 25% FFN FLOPs. Nothing below 41M, nothing quantised, and sparsity never below 3.1% (ours: 1 of 64 = 1.6%).

## Training recipe and gotchas
- 100k steps, Adam, lr 2.5e-4 cosine to 0, gradient clip 0.25, Transformer-XL memory.
- Sigmoid beats softmax; the score multiplies the expert output.
- Init by total d_ff, not G: worth 0.02-0.2 ppl.
- Entropy term on batch-mean softmax of router logits, gamma 1e-3. Expert dropout only helps larger N_E (delta 0.05-0.2); none at N_E 16.
- No capacity limits or balanced assignment; S-BASE's train/inference mismatch hurt badly at 47M (C4: 35.48 vs 23.76).
- Comparisons are parameter-equal; the abstract calls this crucial.

## Mapping onto the GBA design
Per layer, d = 256 [C]:
- Router: N_E x d = 64 x 256 = 16K ternary weights (about 26K cycles), then top-K over 64 scores. Sigmoid = 256-entry LUT; s[e] is one scalar multiply per output (or a shift), d multiplies per active expert.
- Expert with G units: 2·G·d weights. K=1,G=512: 262K weights (about 420K cycles); K=4,G=128: the same; K=1,G=256: 131K. With 4-6 layers at 1.3-2M weights per token, G per layer is about 256-512 total active units.
- ROM: N_E·2·G·d = 64 x 262K = 16.8M weights per layer at 512 active units; 6 layers = 100M, inside the 50-120M budget. RAM: 64 scores + G hidden values, under 2 KB.
- Choosing K=4 x G=128 instead of K=1 x G=512 costs no extra matvec work, only 3 more scalar-scaled accumulations, and the paper says it is worth 0.03-0.04 nats.
- Top-K hidden units: W2 could be applied column-wise for the K survivors (K·d adds) instead of the row kernel (G·d/4 lookups); a win only if K < G/4 or so.
- Entropy loss, expert dropout and init are training-only and free on the device.

## Verdict
Adopt the σ-MoE recipe (sigmoid score, entropy regulariser, d_ff-scaled init, no balancing constraints) as the router for the expert MLPs; test K. The paper supports the central bet (stored experts add quality at fixed compute: -0.11 nats for 8x experts) but says top-1 is the worst K at equal active units. Experiment: float then ternary d=256 model, equal active units per layer, (K,G) in {(1,512),(2,256),(4,128)} x N_E in {16,64}, plus Switch-style softmax top-1 as control; log usage entropy per layer. Keep K=1 only if the gap is under 0.02 nats after QAT.

## Could not verify
- Table 3 gives 11.71 and Table 4 gives 11.59 for σ-MoE on WT-S; the text I read does not explain the difference [U].
- Whether top-K activation is applied in training, and the PKM top-k value.
- Wall-clock numbers (figures only), tokenizer vocabulary size, dataset token counts, GPU hours.
- Table 3 does not state N_E per row.
- I did not read the code repository.

## Links worth following
- https://github.com/robertcsordas/moe (CUDA kernel, exact regulariser and init)
- https://arxiv.org/abs/2101.03961 (Switch Transformer, the top-1 baseline)
- https://arxiv.org/abs/2407.04153 (PEER, single-neuron experts)

## Independent verification
Second reader, 2026-11-18. Source read directly: arXiv HTML full text of 2310.10837v3 (downloaded and converted to plain text, tables read cell by cell; no summarising tool). Figures 3 and 7 are not rendered as images in the HTML, so only their captions and the surrounding text were read. Code repository not read. [P] = primary text, [C] = my arithmetic.

1. **Stored experts at fixed active compute (11.59 -> 10.37, 0.033 nats behind dense 10.03): confirmed, with caveats.** [P] Table 4 / Table 10 row "σ-MoE (K=4, G=128)": WT-S 11.59, WT-S* 10.37; Table 9 gives N_E 16 vs 128; Table 7 gives 25.0% vs 3.1% of dense FFN FLOPs; Sec. 6.3 gives the d_ff=16480 dense baseline at 10.03. [C] ln(11.59/10.37)=0.111, ln(10.37/10.03)=0.033. Caveats: (a) the authors read this result negatively, saying the gap "still remains significant (10.37 vs 10.03), unlike with the MoE with moderate sparsity"; (b) the two runs also differ in expert dropout (none vs 0.05); (c) no seeds or variances are reported anywhere in the paper, every number is one run; (d) "47M -> 238M" is total model parameters, not FFN parameters. Extra support the first reader did not list (Table 10): with K=1, G=128 the same N_E 16 -> 128 step gives 12.26 -> 11.30 (0.082 nats [C]), and K=8, G=128 gives 11.58 -> 10.22.

2. **Top-1 worse than top-4 at equal active units (11.90 vs 11.59; 10.83 vs 10.37): confirmed numerically, but the comparison is confounded.** [P] Table 4 / Table 10 rows "K=1,G=512" under σ-MoE: 11.90 / 10.83 / 9.58 against 11.59 / 10.37 / 9.44. [C] 0.026 and 0.043 nats; the 262M column gives 0.015 nats, which the claim omits. Missed caveat: parameter count is held constant, so going to K=1, G=512 divides N_E by 4 (App. A.4: N_E' = N_E/m). The 47M top-1 model therefore has only 4 experts and the 238M one has 32; this is not "1 of 64" routing. The paper does conclude "a higher K is beneficial", and K=8, G=64 is mixed (11.63 / 10.30 / 9.58). A cleaner reading of K at fixed N_E and G (Table 10, G=128, compute not equal): K=1/2/4/8 = 12.26 / 11.90 / 11.59 / 11.58 at 47M and 11.30 / 10.66 / 10.37 / 10.22 at 238M. Section 6.3 also states that G<128 "generally degrades performance" and that G·K needs to be above roughly 256-512 active channels.

3. **Sigmoid + entropy regulariser beats Switch-style softmax top-1; Switch worse than dense at 47M and collapses at 128 experts: partly.** [P] Numbers correct: Switch 12.27 / 11.24 / 9.68 (Table 4), dense 11.81 (Table 3); Sec. 6.3 text says Switch and the softmax-renormalised variant "severely suffer from the expert collapse problem" (Figure 3, WT-S* models, validation set). What differs:
   - The 12.27 row is Switch with standard dropout in the experts; the "no dropout" Switch row is 11.88 at 47M, only 0.07 ppl from dense (but 11.10 at 238M and 9.77 at 262M, still worse than σ-MoE).
   - This is not the original Switch Transformer: footnote 6 says their implementation "does not enforce capacity factor-based hard balancing". So the result is about the Switch selection function and auxiliary loss, without its hard balancing.
   - The Switch row is K=1, G=512, so by parameter matching it has 4 experts at 47M and 32 at 238M, not 128. The text says the Figure 3 models have "128 experts"; that holds for the K=4, G=128 rows. I could not see the figure to resolve which Switch variant is plotted, so "collapses at 128 experts" is not established for the top-1 Switch model. Table 10 does have Switch at K=4, G=128 (N_E 128 in WT-S*): 12.05 / 11.37, also poor.
   - "No balancing constraints" overstates it: the entropy term is itself a soft load-balancing loss (Table 9 calls γ the scaler for "the load balancing term"). And the ablation shows the regulariser is not what prevents collapse: "no regularization" (γ=0, δ=0) scores 11.83 / 10.41 / 9.51 and the text says it "does not entail catastrophic collapse even with N_E=128". The clean gating ablation is σ-MoE vs softmax at the same K and G: 11.59 vs 11.89 (renormalised) / 12.05 (not renormalised) at 47M, 10.37 vs 11.27 / 10.54 at 238M.

4. **σ-MoE matches parameter-equal dense at 25% of FFN compute at 47M (11.71 vs 11.81; C4 23.25 vs 23.76): confirmed.** [P] Table 3 rows WikiText-103 47M and C4 47M, "% FLOPs" 25.0%; Table 5 repeats the C4 numbers. Caveats: the 25% is FLOPs of the feedforward block only and excludes the router projection (stated in Sec. 6.3) and all attention cost; the configuration is K=4 of N_E=16, i.e. 512 active units of 2048, far from our 1-of-64. The unexplained discrepancy noted earlier stands: the same configuration is 11.71 in Table 3 and 11.59 in Table 4/10, and I found no explanation in the text (no statement of validation vs test for either table; Sec. 6.3 calls the Table 4 value 10.37 a "test perplexity"). That 0.12 ppl spread is as large as several of the ablation gaps, so differences below about 0.01 nats in this paper should not be relied on.

Overall: the table numbers were transcribed correctly, and the arithmetic in nats is right. The conclusions drawn for our design are weaker than the claims suggest: the K=1 evidence comes from models with 4 or 32 experts, all runs are single-seed float Transformer-XL models of 47M+ parameters with at least 512 active FFN units per layer at d_model 412, and the paper's own reading is that pushing sparsity to 3.1% leaves a significant gap to dense.
