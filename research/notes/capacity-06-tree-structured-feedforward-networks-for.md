# Dynamic sparsity in tree-structured feed-forward layers at scale  (https://arxiv.org/abs/2604.08565, v1 18 Mar 2026; Sedghi, Schiewer, Subramoney, Kappel; Bielefeld / Royal Holloway)

Listed in our queue as "Tree-structured feedforward networks for autoregressive language models"; the arXiv ID is right, the real title is above. Read in full from the arXiv HTML (v1, the only version). [P] = read in the paper; [I] = my inference.

## What it is
Fast-feedforward (FFF, Belcak & Wattenhofer) layers used as drop-in replacements for every MLP block in GPT-style (OPT) models, trained from scratch at 125M and 1.3B parameters, with loss curves only at 2.7B and 6.7B [P]. Each token walks one root-to-leaf path in each of P binary trees; only the visited nodes compute. Routing is hard, has no router network, no auxiliary loss and no straight-through estimator [P]. The paper also analyses why path usage becomes heavy-tailed ("auto-pruning") and how to undo it [P]. All comparisons hold total parameters fixed, not compute [P].

## Mechanism
P perfect binary trees of depth D. Every node n has a routing row w_in[n] (d_in), bias b_in[n], and an output vector w_out[n] (d_out) [P].

```
y = b_out
for p in 1..P:
    s = 0
    for l in 0..D:
        n = node(p, l, s)
        z = dot(x, w_in[n]) + b_in[n]
        y += GELU(z) * w_out[n]
        if l < D: s = 2*s + (z >= 0)      # hard, stop-gradient
```
- Every visited node both routes and contributes output; a node is an ordinary hidden neuron whose sign also picks the child [P].
- Training form ("masked", GPU-friendly, exactly equivalent): Z = X·W_in for all P·N nodes, M = BTS(sg(Z)) a 0/1 path mask, Y = (M ⊙ GELU(Z))·W_out + b_out. No gradient through M; routing rows learn only through their own GELU(z)·w_out term [P].
- FFF-post variant: Y = GELU((M ⊙ Z)·W_out + b_out); balances the trees but scores worse [P].
- P and nodes-per-tree are not stated. Sparsity figures fit 1 − (D+1)/(2^(D+1) − 1), and total nodes presumably equal the dense hidden width [I].

## Numbers
OPT-125M from scratch, SlimPajama, 2.5B tokens, 3 seeds, perplexity (Table 6) [P]:

| Layer | 75% sparse | 90% | 97% |
|---|---|---|---|
| FFF (D=3/5/7) | 25.91 | 27.66 | 29.37 |
| FFF-post | 29.68 | 29.81 | 30.16 |
| MoE top-2 (8/21/106 experts) | 25.83 | 26.70 | 29.31 |

Dense FF: 23.78. So FFF costs +9% / +16% / +24% perplexity against a dense layer with the same parameter count, and ties or slightly trails MoE.

OPT-1.3B, 26B tokens, SlimPajama / WikiText perplexity (Table 1) [P]: dense 15.83 / 17.2; FFF D=4 (83% sparse) 16.61 / 18.10; FFF D=6 (94%) 17.57 / 20.01. "FT" rows (attention initialised from the public 300B-token OPT, FFF random): 13.37 / 14.55 and 14.11 / 15.02, against 13.11 / 14.3 for the public model; these inherit 300B tokens and are not comparable with the 26B dense baseline [I].

Downstream, 1.3B (Tables 2-3) [P]: zero-shot average dense 43.1, FFF D=4 43.0, D=6 40.6; MMLU 5-shot 25.87 / 25.51 / 25.8 (near chance).

Usage (Sec. 5) [P]: leftmost path takes over 50% of visits; dead leaves 1% / 5% / 20% at D=3/5/7. D=7 trees tolerate removing up to 40% of least-used nodes (Fig. 10, 125M).

Speed (Table 5, one layer, A100) [P]: dense 3.03 ms; D=4 1.08 ms; D=6 0.35 ms (8.7x); D=13 0.57 ms. Authors call these "theoretical and simulated potential".

## Smallest scale tested, and whether the result held there
125M parameters, 768 wide, FP16. The claim "matches dense" does not hold there: +9% to +24% perplexity. What holds is "equals MoE at the same active fraction, with no router or balancing loss". The gap to dense shrinks with scale (+5% at D=4, 1.3B), which is the wrong direction for a 256-wide model [I].

## Training recipe and gotchas
- AdamW, lr 1e-4, betas 0.9/0.95, weight decay 0.01, cosine, 3% warmup, clip 1.0, one epoch, ~5e5 tokens per step, sequence 1024, 32 A100s [P].
- Training costs as much as dense: all node logits are computed, then masked [P].
- Keep GELU before the output projection; moving it after (FFF-post) cost 2-4 perplexity points at 125M [P].
- Imbalance is not collapse: the balanced variant was worse, so do not add balancing losses by reflex [P]. Nonlinearity placement and balance are confounded in that comparison [I].
- The "fewer than 5% of units" headline matches only D=7 (+24% perplexity) and the D=13 timing run; the 1.3B models activate about 16% and 6% [I].
- Internal inconsistencies: active nodes are given as P×D and as a sum over D+1 levels; Table 4 FLOPs fit D/(2^D − 1) while the sparsity figures fit D+1 [I].

## Mapping onto the GBA design
All [I]. An FFF layer with K = P(D+1) active nodes touches 2·K·d ternary weights, exactly what a dense MLP with K hidden units costs. Example at d=256, K=288: 147K weights, about 236K cycles per layer at 1.6 cycles/weight; six layers use about 0.9M of the 1.3-2M budget.
- Input side: one ternary row per node, read sequentially from ROM, using the per-layer partial-sum tables we already build. Full kernel speed, plus one branch and one non-sequential ROM seek per node.
- Output side: group each path's nodes in fours and store w_out per sub-path, so one ROM byte selects one partial sum of 4 activations. Accumulators live in RAM, not registers, so expect roughly 2 cycles/weight unless several groups are interleaved.
- GELU: 256-entry table. RAM: a d-wide accumulator plus small transient tables.
- ROM: nodes = P(2^(D+1) − 1). K=288 gives 1.5M weights per layer at (P=48, D=5), 4.7M at (36, 7), about 50M at (24, 11). Depth 9-10 is the ceiling for six layers.
- Against our 1-of-64 experts: no router cost, 2^(D·P) possible active sets, and the P root neurons act as an always-on shared trunk.
- Unknown: ternary weights with sign routing under quantization-aware training; deep nodes see about 1/2^l of the data.

## Verdict
Test. It is the only primary evidence that hard-routed conditional MLPs train stably for autoregressive models without auxiliary losses, but it never compares against a dense layer of equal compute, which is the only comparison that matters here. Settling experiment: in our ternary 256-wide model, at equal weights touched, compare dense (h=K), 1-of-64 experts, and FFF with (P, D) = (72,3), (48,5), (36,7); adopt if FFF beats dense by over 3% perplexity and at least ties experts. Log dead-leaf fraction and path entropy.

## Could not verify
- No code release found (arXiv page and one web search).
- P, nodes per tree, MoE expert width, and 1.3B hyperparameters are not in the paper.
- No compute-matched dense baseline, no model under 125M, no quantized run.
- Figures 2-10 were read as captions and prose only.
- Table 2 has odd entries (OpenBookQA 15-17%, below 25% chance) that I could not explain.

## Links worth following
- https://arxiv.org/abs/2308.14711 (original FFF; soft training, hardening loss)
- https://arxiv.org/abs/2311.10770 (UltraFastBERT; 12 of 4095 neurons, abstract checked)
- https://arxiv.org/abs/2407.04153 (PEER, the lookup-routed alternative)
