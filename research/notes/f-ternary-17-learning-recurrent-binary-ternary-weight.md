# Learning Recurrent Binary/Ternary Weights (https://arxiv.org/abs/1809.11086, ICLR 2019; Ardakani, Ji, Smithson, Meyer, Gross, McGill)

Read in primary form (ar5iv and arxiv.org/html/1809.11086v2, via a summarising fetch tool, so numbers are from the tables but wording is paraphrased). Code: https://github.com/arashardakani/Learning-Recurrent-Binary-Ternary-Weights (Theano/Blocks, GPL-2.0; README only read).

## What it is
A training recipe for LSTMs/GRUs whose recurrent and input matrices are binary {-1,+1} or ternary {-1,0,+1}, trained from scratch. The two ingredients are (1) stochastic binarisation/ternarisation of latent float weights in the forward pass and (2) batch normalisation applied to every matrix-vector product, which fixes the gate saturation that makes plain BinaryConnect LSTMs fail. On small language-modelling benchmarks the ternary models match or beat the float baseline. The paper also gives a 65 nm ASIC design (not relevant to us).

## Mechanism
- Latent float W; one fixed scale alpha per matrix (Glorot init value). w_N = W/alpha, clipped to [-1,1].
- Binary: P(+1) = (w_N+1)/2; w_B = 2*Bernoulli(P) - 1.
- Ternary: P(nonzero) = |w_N|; w_T = Bernoulli(|w_N|) * sign(w_N). So zero-fraction is whatever the latent magnitudes give; it is not controlled.
- Gradient: straight-through, dL/dW := dL/dW_T; update the float W (Algorithm 1).
- LSTM gates: f,i,o,g = act( BN(W_h^T h_{t-1}; phi_h) + BN(W_x^T x_t; phi_x) + b ), BN(x) = gamma + phi*(x-E[x])/sqrt(Var[x]+eps), gamma = 0 for the gate BNs; optionally h_t = o * tanh(BN(c_t)). phi learned.
- Kept full precision: biases, BN parameters, softmax classifier, and (by my inference from the size column) the word embedding. Activations are float in training; hardware section assumes 12-bit fixed point.
- Quantised: all W_x and W_h of all gates.

## Numbers (all test-set; "size" = quantised recurrent weights only)
Table 3, word-level PTB (10K vocab, ~1M training tokens), perplexity:
- Small (300 units; 2880 KB float = 720K weights): float 91.5, binary 92.2, ternary 90.7, BinaryConnect 125.9, Alternating 2-bit 103.1 / 3-bit 93.8 / 4-bit 91.4.
- Medium (650 units; 27040 KB = 6.76M weights): float 87.6, binary 87.2, ternary 86.1, BinaryConnect 108.4.
- Large (2x1500; 144000 KB = 36M weights): float 78.5, binary 76.5, ternary 76.3, BinaryConnect 128.5.

Table 1, character-level BPC (1 layer; 1000 units PTB, 512 units others): PTB float 1.39 / binary 1.43 / ternary 1.39; War & Peace 1.72 / 1.78 / 1.72; Linux kernel 1.73 / 1.79 / 1.75. Competitors on char-PTB: TWN 1.51, TTQ 1.49, LAQ ternary 1.46, LAB binary 1.56, BinaryConnect 2.51.

Table 2, text8 (90M chars train, 1 layer of 2000 units, ~16M weights): float 1.46, ternary 1.51, binary 1.54, BinaryConnect 2.45.

Table 4, sequential MNIST (100 units): float 98.9%, ternary 98.8%, binary 98.6%, BinaryConnect 68.3%.
Table 5, CNN QA Attentive Reader: float 59.81%, ternary 60.03%, binary 59.22%, BinaryConnect 5.34%.
Table 6, GRU char BPC: PTB float 1.40 / binary 1.46 / ternary 1.41; War & Peace 1.75 / 1.92 / 1.82; Linux 1.82 / 1.90 / 1.81.

## Smallest scale tested, and whether the result held there
Smallest LM: 720K quantised weights (small word-PTB): ternary 90.7 vs float 91.5, so yes, it held. Smallest overall: 100-unit sequential-MNIST LSTM (~40K weights), 0.1 point loss. So the third-party figure (90.7 vs 91.5) is confirmed from Table 3.

Important reading of this: the gap is zero only in the data-limited regime. PTB has ~1M tokens and the float baselines run with dropout 0.5-0.65; stochastic ternarisation is an extra regulariser and the ternary model wins by more as the model gets larger (large: 76.3 vs 78.5). On text8, the only benchmark here where the model is closer to capacity-limited, the gap reappears: +0.05 BPC ternary, +0.08 BPC binary (about 3.4% and 5.5% more bits). Koko (large synthetic corpus, compute-bound model) is in the text8 regime, not the PTB regime. This paper therefore does not refute a capacity penalty for ternary; it shows the penalty is hidden when overfitting dominates.

## Training recipe and gotchas
- Word PTB: SGD lr 20, divide by 4 when validation ppl rises, grad-norm clip 0.25, BPTT 35, dropout 0.5 (small) / 0.65 (medium, large). Char PTB: Adam 2e-3, batch 64, seq 100. Text8: Adam 1e-3, batch 128, seq 180. Epoch counts not given.
- BN is the essential part: same quantiser without BN (= BinaryConnect) collapses (125.9 ppl; gates saturate, input/output gates stuck open, candidate blocked).
- Accuracy of the quantised model improves with batch size (Fig. 3); BN cannot train at batch 1. Training is slower per step.
- Converges faster than float early, slower later.
- Trained at length 100, tested to length 1000 without degradation.
- Authors say variance from stochastic sampling at test is negligible; they do not state how the final fixed weight set is chosen, or whether BN running statistics are used at test.

## Mapping onto the GBA design
- Inference cost: nothing new. BN after a ternary matvec folds to one per-row affine: y_i = m_i * (row_i . x) + c_i, i.e. one 16x16 multiply and one add per output row (256-384 per matrix, ~10 cycles each), negligible beside the 1.6 cycle/weight kernel. m_i, c_i go in ROM (2-4 bytes per row). The single alpha is absorbed into m_i. No RAM cost.
- So what transfers is a design rule: every ternary matrix should be followed by a learned per-output-channel scale/shift (or a norm) before any saturating nonlinearity or hard decision. In Koko this applies to the attention key scores (top-8 selection), the expert router logits and the two-level head, which are the analogues of LSTM gates.
- Batch statistics do not fit a transformer directly; use RMSNorm/SubLN before the ternary layer (BitNet practice) plus a foldable per-channel gain after it.
- Stochastic ternarisation gives uncontrolled sparsity; our 32-entry table needs exactly one zero per group of 4. Would need a constrained sampler/projection (e.g. zero the smallest-|w| of each 4); not covered here.
- Embeddings and softmax were float here; our ternary/hashed embeddings and class head get no support from this paper.
- An LSTM itself is a poor fit: 8 matvecs of size H per token; at H=512 that is 2.1M weights per layer per token.

## Verdict
Test (low cost), do not adopt as architecture. Use as (a) evidence that with normalisation ternary can match float at 0.7M-36M weights when data-limited, and (b) the text8 gap as a caution. Settling experiment: at our scale train float vs deterministic-absmean ternary vs stochastic ternary, each with and without a post-matvec per-channel affine, on the real corpus, same step budget; compare validation loss. If stochastic sampling helps only when the train/val gap is large, drop it.

## Could not verify
- Whether embedding/softmax were quantised or counted in "size": inferred from arithmetic (720K x 4 B = 2880 KB), not stated.
- Medium model: text says one 650 layer, size column implies two.
- Test-time weight selection and BN statistics: not stated in paper or README; source code not read.
- Total parameter counts including embeddings: not given.
- Paper text says 32x memory saving for ternary in one place; tables show 16x; abstract says 12x.
- All text reached through a summarising fetch; equations reconstructed from that, not copied.

## Links worth following
- https://github.com/arashardakani/Learning-Recurrent-Binary-Ternary-Weights (PTB_word script: test-time quantiser, BN stats)
- Recurrent Batch Normalization, Cooijmans et al. 2016, https://arxiv.org/abs/1603.09025 (the BN-LSTM this builds on)
- Alternating Multi-bit Quantization for RNNs, Xu et al. 2018, https://arxiv.org/abs/1802.00150 (the 2-4 bit baseline in Table 3)
