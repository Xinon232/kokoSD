Lookup-based capacity for a 16 MHz, 1M-active-weight LM: findings

Tags: [P] = I read it in the primary source (paper text, repo file or model card); [S] = search snippet or secondary summary only.

1. Product-key memory (PKM) / Memory Layers at Scale
- Mechanism [P]: a query net (linear + BatchNorm) maps the hidden state to a query, which is split in half. Each half is scored against √N sub-keys; take top-k per half, form k² candidates, take the final top-k, softmax the scores, and output the weighted sum of the k value rows. Several heads share one value table. Lample: 4 heads, k=32, 512² slots, replaces the FFN in 1-2 layers. Meta's Memory+: 1024 half-keys (1M values), value dim = model dim, key dim = half the model dim, one pool shared across 3 layers, output = (y ⊙ silu(xW1))W2. The k/heads Meta used I did not find.
- Gains [P]: Lample: 12 layers + 1 memory reaches ppl 15.62 vs 16.02 for a 24-layer baseline (dim 1024). Meta, 134M base: NQ 0.91 → 3.16, TQA 7.7 → 18.77; at 1.3B base, Memory+ 13.68/42.89 vs MoE 8.14/31.46 vs PEER 12.33/42.46.
- Smallest scale [P]: 134M base (Meta), 6-layer ablation (Lample). Nothing near 1M active weights.
- Gotchas [P]: without query BatchNorm, slot usage at 1M slots was 25.8% vs 80.3% (ppl 19.79 vs 17.98). Lample used Adam at 1e-3 for values vs 2.5e-4 elsewhere. Meta needed qk-normalization for stability with big memories on small bases. Too many memory layers hurts.
- Cost for you: about √N·d for key scoring, so 2×1024×128 ≈ 260K ops per head per layer, which is a large slice of your budget. Use 256 half-keys or fewer.
- Follow-ups [S]: Fast-weight PKM (arXiv 2601.00671), MIDUS (2026).
- URLs: https://arxiv.org/abs/2412.09764, https://arxiv.org/abs/1907.05242

2. Per-Layer Embeddings (Gemma 3n)
- Mechanism [P, HF `modeling_gemma3n.py`]: one table of shape vocab × (n_layers × 256), indexed by the current token id. Per-layer input = (RMSNorm(Linear(token_embed)·d^-0.5) + table_row·√256)/√2. At the end of each layer: x += RMSNorm(W_proj(act(W_gate·x) ⊙ ple_i)). Defaults: 35 layers, hidden 2048, vocab 262,144.
- Training: ordinary end-to-end embedding. Google publishes no ablation [S, docs]: https://ai.google.dev/gemma/docs/gemma-3n
- Cost per layer: two small matvecs (d×ple_dim each) plus one row read.

3. slvDev/esp32-ai [P: README, RESULTS.md, `src/model.py`, HF card]
- Architecture: d_model 96, 6 layers, 4 heads, FFN hidden 66, ple_dim 128, context 256, input vocab 32,768.
- Parameters: 28,867,968 stored = 25,165,824 PLE table (87%) + 3,145,728 tied embedding/head + 556,416 dense core. Per token it reads 6 table rows (~450 B); the output head is still fully scanned.
- Quantization: int4 PTQ, group 128, fp16 scales, no QAT; head staged to int8 with int8 activations (+0.0003 nats).
- Quality (TinyStories): ppl 12.58 → 11.41 (+0.098 nats, 2 seeds) vs a same-core baseline. The same table used as a wide input embedding gives only +0.052. The gain is retained at 4-bit.
- Caveats from its own ablations: at vocab 4096 the gain is only +0.025 nats, while doubling the core gives +0.170. Widening rows saturates around a 6M table; adding rows (bigger vocab) does not.
- Time: 94.9 ms/token, of which head 59.4, attention 20.5, PLE 6.4, FFN 6.5. The output head dominates, so budget for it (factorised or smaller head).
- URLs: https://github.com/slvDev/esp32-ai, https://huggingface.co/slvDev/esp32-ai-tinystories

4. Fast feedforward networks / UltraFastBERT
- Mechanism [P]: neurons form a binary tree. Compute the node's dot product, go left or right on its sign (next = 2n+1+(logit>0)), and the output is Σ act(logit_d)·w_out over the D nodes on the path. UltraFastBERT-1x11 uses 12 of 4095 neurons.
- Training [P]: the 2308.14711 version trains with soft sigmoid mixtures plus an entropy "hardening" loss (h = 3-10), then rounds at inference. Known problems: overfragmentation, shrinking per-leaf batch, localized overfitting.
- Quality [P]: GLUE average 79.3 → 77.3 (1x11); nearly all the loss is CoLA (45.0 → 37.1). Encoder only, crammedBERT scale. CPU speedup 78× for the FFN only.
- Generative use: arXiv 2604.08565 (2026) applies tree FFNs to autoregressive LMs above 1B with under 5% of units active, using hard routing and no auxiliary loss, and claims to match dense [P abstract]. "Within 10% perplexity" is [S].
- URLs: https://arxiv.org/abs/2311.10770, https://arxiv.org/abs/2308.14711, https://arxiv.org/abs/2604.08565

5. N-Grammer and hashed n-gram embeddings
- N-Grammer [P]: product-quantize the input embeddings per head (k = 4096-8192 clusters, k-means); bigram id = z_i + k·z_{i-1}; hash ((r·b+s) mod p) mod v into a 196K-393K-row table; LayerNorm, then concatenate with the unigram embedding. The cluster id per token is static after training, so inference is pure lookup.
- N-Grammer gains are weak [P]: C4 ppl 15.32 (234M Transformer) vs 15.01 (288M) and 14.79 (343M); many configurations are worse than baseline. Table trained with Adagrad at lr 0.1; 10× either way was unstable. Smallest tested 234M.
- Engram (DeepSeek, arXiv 2601.07372) [P]: hash suffix 2- and 3-grams of normalised token ids with K=8 heads into prime-sized tables, concatenate, then gate with α = σ(RMSNorm(h)·RMSNorm(W_K e)/√d) and add α·W_V e at layers 2 and 6/15. Table uses Adam at 5× lr, no weight decay. Val loss 1.808 → 1.768 on a 3B MoE backbone; gating and tokenizer compression matter most. Smallest tested 0.56B activated.
- Scone (arXiv 2502.01637) [P]: the embedding of the longest frequent n-gram (up to 5) is produced by a side transformer in training and cached to a table. WikiText ppl 26.1 → 23.4. Smallest 76M.
- Also BLT hashed byte n-grams (arXiv 2412.09871) [S]. I did not research infini-gram/kNN-LM baking.

6. Many-expert MoE / PEER (arXiv 2407.04153) [P]
- Mechanism: PKM retrieval where each retrieved item is a one-neuron expert σ(uᵀx)·v. Default 1024² experts, 8 heads, k=16, query BatchNorm.
- Gains: C4 ppl at 6e18 FLOPs: dense 23.84, MoE-128 21.41, PKM 21.92, PEER 20.63. Usage ~100% even at 1M experts; BatchNorm reduces imbalance (KL 1.52 → 1.06). No small-model results.
- Top-1 hard routing under 50M: I found no primary study. [S]: at GPT-2 scale, without a balance loss one expert took 73.6% of tokens by step 500, and top-2 barely beat top-1 (arXiv 2402.13089 and an HF project). Meta's MoE baseline trains with expert-choice routing and evaluates top-1 [P].

Ranked shortlist for ~1M active / ~50M stored
1. Context-hashed n-gram tables injected per layer with a PLE/Engram-style gate. It costs about 2·d·ple_dim per layer plus row reads. It has the only measured win at your scale (+0.098 nats at 556K active weights), and bigram/trigram keys escape the "only helps with big vocab" limit by multiplying the row count. The combination is my inference; each half is [P].
2. Token-id PLE. It is the simplest version of the same idea and is verified on a microcontroller, but the gain depends on vocab size and you already pay for a big vocab in the output head.
3. Small product-key memory (or PEER) replacing one or two middle FFNs, with at most 256 half-keys, k of 8-16, query BatchNorm and a higher value lr. The gains are strong and concentrated on factual recall, but only shown at 134M and above, and key scoring is real compute.

Your 1-of-64 hard-routed MLP is the least-evidenced item: no primary result at this scale, and collapse is the known risk. Tree (FFF-style) routing is the documented alternative if you keep it.