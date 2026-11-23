# Understanding Weight-magnitude Hyperparameters in Training Binary Networks (https://arxiv.org/abs/2303.02452, ICLR 2023, Joris Quist, Yunqiang Li, Jan van Gemert; TU Delft / Axelera AI)

Read in primary form (arXiv HTML v1, full text incl. appendix). Figures are images: curve shapes below are from the captions and prose, not read off the plots. Code: https://github.com/jorisquist/Understanding-WM-HP-in-BNNs (not opened).

## What it is
An analysis paper, not an accuracy paper. For sign-quantised weights the latent magnitude never reaches the forward pass, so learning rate, weight decay, init scale and LR decay cannot mean what they mean for real weights. The authors show that SGD + momentum + weight decay on latent weights is exactly a 2nd-order low-pass (IIR) filter on the gradient, whose sign is the weight. Seven hyperparameters collapse to three: alpha (= lr x weight decay), alpha-decay, gamma (momentum). Accuracy matches, never beats, the tuned baseline; the gain is a cheaper search.

## Mechanism
Latent form (Eq 1-5): `m_i = (1-γ)m_{i-1} + γ∇`, `w_i = w_{i-1} − ε(m_i + λ w_{i-1})`, `θ = sign(w)`, sign(0) = random ±1.
With g = −w, unrolled (Eq 7): `g_i = ε Σ_r (1−ελ)^{i−r} m_r`.
- The outer ε only scales g, so it cannot change the sign (Eq 8). LR matters only through `α = ελ` and through its ratio to the init magnitude.
- Filter form (Eq 11-13): `m_i = (1−γ)m_{i−1} + γ∇`; `g_i = (1−α)g_{i−1} + αm_i`; `θ_i = −sign(g_i)`. One-line form (Eq 17): `g_i = αγ∇ − (α+γ−2)g_{i−1} − (α−1)(γ−1)g_{i−2}`.
- Weight decay = EMA discount: window ~1/α steps. LR decay = α-decay = widening window, which drives the flip rate to zero.
- Init: `g_0 = 0`, no bias correction (they report it hurt: early gradients are large). Scaling ε by s is identical to dividing init by s (Fig 3).
- Requires "magnitude independent" training: no latent clipping, no gradient clipping by latent magnitude, no channel-mean scale factor.
- Diagnostic: flip ratio FF = fraction of weights changing sign per update (Eq 18).

## Numbers
- CIFAR-10, Bi-RealNet-20, 1/1 bit, 400 epochs (Table 2): FP 91.7; Bi-Real one-step (their rerun) 85.0; Bi-Real + filtering optimizer 86.5; Bi-Real two-step (rerun) 86.7; IR-Net 86.5. Settings: α = 1e-3 cosine-decayed, γ = 1e-1; baseline SGD lr 0.1, momentum 0.9, wd 1e-4, batch 256.
- ImageNet, ReActNet-A (Table 3): filtering optimizer, one step, 69.7 top-1 / 88.9 top-5; two-step ReActNet-A 69.4 / 88.6; ReActNet-A-AdamBNN two-step 70.5 / 89.1. α = 1e-5 linear decay, γ = 1e-1, 600K iterations, batch 510, Adam lr 0.0025 for real-valued params, teacher-student, 3x A40 for about ten days.
- Fig 4 (CIFAR-10): standard clipped/scaled setting has a single LR optimum; magnitude-independent with nonzero init fails at small LR; zero init makes accuracy independent of LR.
- Fig 5: accuracy is strongly sensitive to α; too large = too many flips, too small = "converges too quickly".
- Fig 6: without α-decay flipping never stops and accuracy plateaus.
- Fig 8 (audio digits, 4-layer FC net, 25 Bayesian-opt trials x 20 runs): same final accuracy; the latent view (6 HPs) needs "around 10 more trials" than the filter view (2 HPs).

## Smallest scale tested, and whether the result held there
A 4-layer fully connected binary net on an audio classification set (first layer not binarised), and Bi-RealNet-20 on CIFAR-10. The equivalence held at both (Fig 7, Fig 8). No language model, no transformer, no ternary weights, no Adam on the quantised weights.

## Training recipe and gotchas
- The derivation is for SGD and binary sign only. The authors say explicitly that Adam's normalisation is not covered and is future work.
- It forbids latent/gradient clipping; they argue α-decay replaces them.
- Tune α on a log grid and always decay it; γ = 0.1 worked in both experiments. α and γ are interchangeable without decay.
- Weight decay exactly 0 = infinite window (pure accumulation); the latent only freezes if LR also decays.
- My inference, not in the paper: for absmean ternary (threshold proportional to mean|w|) the quantiser is also scale-invariant, so the same argument applies per tensor: what matters is step size relative to mean|w|. With AdamW each step is about lr in size, so flip rate ~ lr / mean|w|; decoupled decay gives α = lr x wd and a steady-state latent scale that shrinks as wd grows. "x10 LR still improving" then reads as "init magnitude too large for the LR" (Fig 4b regime), and "wd off late" as "stop forgetting old gradients, let flips die out".

## Mapping onto the GBA design
Zero inference cost: no change to lookups, adds, multiplies, RAM or ROM. Training-side only:
- Log the per-tensor flip ratio (ternary: fraction of weights changing state per step) for attention keys, expert MLPs, embeddings and head. It is the scale-free quantity to sweep on.
- Sweep lr/mean|w_init| and lr x wd, not lr and wd separately; shrinking latent init is a substitute for raising LR.
- Hard-routed experts and hashed tables see sparse gradients: an EMA window counted in global steps is far shorter in "visits" for a rarely routed expert, so α probably needs to be per-visit or smaller there (my inference).
- Zero-init with random sign does not transfer directly to ternary (zero is a legal state); small-magnitude init is the analogue.

## Verdict
Test, as a training-diagnostics rule; skip the optimizer itself. It gives a principled sweep axis and a flip-rate monitor at no cost, but every number is for binary CNNs under SGD. Settling experiment: on the small ternary LM, run a 3x3 grid of (latent init scale, lr) at fixed lr x wd with AdamW; if loss and flip-ratio curves line up along constant lr/init, adopt the reduced sweep; also compare cosine-decayed lr x wd against the two-stage wd-off schedule.

## Could not verify
- Numeric values inside Figures 3-8 (only captions/prose read).
- Any ternary, transformer, LM or Adam result: none in the paper.
- The repository contents were not read.
- The ternary/AdamW paragraphs above are my extrapolation, not claims of the source.

## Links worth following
- Helwegen et al. 2019, "Latent Weights Do Not Exist" (Bop): the parent idea, with a flip threshold.
- Liu et al. 2021, "How Do Adam and Training Strategies Help BNNs Optimization?" (ICML): the Adam side and two-step weight-decay rule this paper leaves open.
- https://github.com/jorisquist/Understanding-WM-HP-in-BNNs
