# N-Grammer: Augmenting Transformers with latent n-grams  (https://arxiv.org/abs/2207.06366, v1 13 Jul 2022; Roy, Anil, Lai, Lee, Zhao, et al.; Google Research, Brain Team)

Tags: [P] arXiv v1 PDF, all 11 pages read; [R] lucidrains/n-grammer-pytorch README (secondary); [E] my estimate.

## What it is
One layer between the token embedding and the first transformer block. Each token embedding is product-quantized to a cluster id per head; the ids at positions i and i-1 form a "latent bigram" id, hashed into a trainable table; the retrieved row is layer-normalised and concatenated to the unigram embedding. Only bigrams were tried. The claim: a 16-layer model with this layer matches a 20-layer one at higher throughput. Evidence is one dataset, single runs, about 3% perplexity.

## Mechanism
[P §3] Input x in R^{l x h x d} (positions, heads, dims per head).
1. Codebook c in R^{k x h x d}, k = 4096 or 8192, learned by mini-batch k-means during training. z[i,j] = argmin_n ||x[i,j] - c[n,j]||.
2. b[i,j] = z[i,j] if i = 0, else z[i,j] + k * z[i-1,j].
3. y[i,j] = B[((r_j * b[i,j] + s_j) mod p_j) mod v, j]; B in R^{v x h x d_b}; p_j a random prime > k^2; r_j, s_j random per head; v = 196K or 393K.
4. Output w = [LN(x), LN(y)], width d + d_b per head.

Serving [P §7]: token -> z is fixed after training, so cache it for the whole vocabulary; inference is then lookups only. Baseline variant: skip step 1 and hash raw token-id bigrams. Best placement is before layer 1 [P §6].

## Numbers
C4, 32K vocabulary, 16 layers, 8 heads, width 1024, sequence 1024, batch 256, 1M steps; test perplexity and TPU-v3 examples/s [P Table 1].
- Transformer 234M: 15.32 (402.00). Transformer-L, 20 layers, 284M: 14.70 (331.12).
- Primer 234M: 15.10. Primer-L 284M: 15.01 (284.40).
- Raw-id bigrams, 196K rows, bigram dim 12.5/25/50%: 15.36 / 15.27 / 15.50.
- 4K clusters, 196K rows: 15.26 / 15.07 / 15.01 (251M / 263M / 288M).
- 8K clusters, 196K rows: 15.58 / 15.44 / 15.01.
- 8K clusters, 393K rows: 15.32 / 15.97 / 14.79 (267M / 292M / 343M; best runs at 356.94).
- Best gain 15.32 -> 14.79: 3.5%, 0.035 nats [E], for +109M parameters and 11% lower throughput. Five of twelve N-Grammer rows are no better than baseline; no seeds.
- Position [P Table 2]: before layer 1 14.79; after layer 1 14.92; middle 15.13; end 15.17.
- Reaches Primer's 15.10 at 465K steps, 90 vs 180 TPU-hours [P §8].
- SuperGLUE average [P Table 3]: Transformer 52.13, Transformer-L 55.03, Primer 53.81, N-Grammer 54.80.
- Clusters are topical, e.g. one holds baseball, football, hockey, Soccer [P Table 4].

## Smallest scale tested, and whether the result held there
234M is the only backbone. Nothing small, low-data or quantized. Unknown at our scale.

## Training recipe and gotchas
- Base: Adam, lr 1e-3, betas 0.9/0.99, epsilon 1e-10, clip 5.0, no weight decay, rotary positions [P §5.1].
- Table: Adagrad, lr 0.1, same schedule; 10x higher or lower "leads to unstable training" [P §5.2].
- Codebook: k-means lr 1e-3, no EMA (it did not help) [P §5.2]. The lucidrains port uses EMA 0.999 and cites 1e-2 as the paper's lr, which the PDF does not say [R].
- Layer norm on both branches aids stability [P §3.4]. Cluster ids drift while rows train; not discussed [E].

## Mapping onto the GBA design
Per token and head: read z from ROM (2 bytes), shift-add to form b, hash, read one row. The paper's hash uses two modulos, which the ARM7TDMI lacks; use multiply-shift into a power-of-two table (one hardware multiply). Total: tens to low hundreds of cycles for 4-8 heads, under 0.01% of the budget [E]. RAM: the previous token's h ids.

Both layer norms fold offline (each depends only on a token or a row), so ROM holds final integer rows. ROM [E]: 196K rows x 4 heads x 16 dims at 8 bits = 12.5 MB, about 2.5 MB ternary; the token -> cluster map is 128 KB at 16K x 4 x 2 bytes.

Changes: (a) concatenation widens layer 1; at width 256-384 reserve part of the width or add instead (untested here; Engram adds with a gate). (b) The negative raw-id result squeezed 32K^2 pairs into 196K rows; with 16K tokens, simple English and millions of slots it may not transfer. Cluster pairs remain a back-off for unseen pairs. (c) Trigrams and deeper injection are untested here.

## Verdict
Test. The mechanism is nearly free on our hardware and fully precomputable, but the paper shows only 3.5% at 234M with noisy ablations, and a deeper plain transformer still wins. For us that deeper model is unaffordable in cycles, so 0.03 nats for free is worth having. Settling experiment: on the 16K backbone compare no table, raw-id hashed bigram, cluster-pair bigram (k = 256, 1024, 4096), and both, at equal ROM bytes, 3 seeds; also add vs concatenate. Treat Engram's gated variant as the stronger arm.

## Could not verify
- Table shapes: the text says bigram dimension "128 to 512", but Table 1 parameter deltas imply about 64/128/256 floats per row (343M - 234M - 8.4M codebook = 100.6M = 393K x 256) [E]. Whether total width grows is not stated.
- Exact v (196K, 393K are rounded; the port's example uses 196,608 [R]).
- Official Lingvo JAX code not read. No small-scale or quantized replication found (not searched for in depth).
- The cached-serving speed is argued, not measured.

## Links worth following
- https://github.com/tensorflow/lingvo/tree/master/lingvo/jax (official code, paper footnote)
- https://github.com/lucidrains/n-grammer-pytorch
- https://arxiv.org/abs/2601.07372 (Engram; see capacity-04 notes)
- https://arxiv.org/abs/2104.04552 (Huang et al., lookup-table recurrent LMs)
