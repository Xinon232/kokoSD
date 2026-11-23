# Fast Feedforward Networks  (https://arxiv.org/abs/2308.14711, v2, 2023; Peter Belcak, Roger Wattenhofer, ETH Zurich)

Read from the arXiv HTML (v2) and the ar5iv mirror, plus the README of github.com/pbelcak/fastfeedforward. [P] = in the paper or README; [I] = my inference. Figures were not readable as text.

## What it is
A replacement for a feedforward (MLP) layer in which a balanced binary tree of tiny "node" networks picks one of 2^d small "leaf" MLPs per input [P]. Training evaluates every leaf and mixes them with soft sigmoid decisions; inference follows hard decisions and runs one leaf, so cost is O(d·n + ℓ) instead of O(2^d·ℓ) [P]. The paper tests only small image classifiers and a 4-layer ViT; there is no language experiment [P]. It is the origin of the tree-MLP line continued by UltraFastBERT (2311.10770) and 2604.08565.

## Mechanism
- Node N(m,i): a <dim_in, n, 1> network with sigmoid output; "n=1 to suffice in all our experiments", i.e. one hyperplane per node [P]. Leaf: an ordinary <dim_in, ℓ, dim_out> MLP [P]; the README default activation is ReLU [P].
- Sizes: 2^d − 1 nodes, 2^d leaves. Training width 2^d·ℓ; inference width ℓ; inference size d·n + ℓ [P].

```
train(x, N):   if leaf: return N(x)
               c = sigmoid(N(x)); return c*train(x, right) + (1-c)*train(x, left)
infer(x, N):   if leaf: return N(x)
               return infer(x, right if N(x) >= 0 else left)     # c >= 1/2
```
- Loss: L = L_pred + h·L_harden, where L_harden is the sum over nodes and batch items of the Bernoulli entropy of c [P]. h = 3.0 (MLP experiments), 5 / 10 / "∞" (ViT), 0 (inference-size comparison) [P].
- Hardening check: rounding decisions whose entropy is below 0.10 gives "only very modest deviations" from the soft model [P].
- README sizing rule: leaf_width = ceil(hidden_width / 2^depth); to match the dense layer choose 2^depth·leaf_width > hidden_width. Options: train_hardened, region_leak (0-0.5, for very deep trees) [P].

## Numbers
All are accuracy (%), best of 10 runs. MA = training-set ("memorization"), GA = test.

Table 1, single hidden layer, training width 128, flattened images [P]:

| Dataset | dense 128 MA/GA | ℓ=8, d=4 | ℓ=1, d=7 |
|---|---|---|---|
| USPS | 100.0 / 94.2 | 99.5 / 92.1 (2.56x) | 81.4 / 77.8 (2.12x) |
| MNIST | 100.0 / 98.1 | 99.3 / 94.9 (3.39x) | 94.1 / 92.0 (3.93x) |
| FashionMNIST | 99.3 / 89.6 | 90.5 / 86.1 (3.78x) | 78.7 / 77.7 (2.92x) |

Against a dense layer of the same inference cost the tree wins: MNIST dense width 16 gives 98.0 / 95.2, while ℓ=8, d=1 (also 16 trained neurons) gives 94.6 / 93.1 and ℓ=8, d=4 gives 99.3 / 94.9, so depth adds memorization but not generalization [P, I].

Table 2, CIFAR10, inference width 32 for both sparse models (MoE: 16-wide experts, top-2; FFF: ℓ=32), value / epochs to train [P]:

| Train width | dense MA / GA | MoE MA / GA | FFF MA / GA |
|---|---|---|---|
| 64 | 87.2 / 49.3 | 57.8 / 29.4 | 85.8 / 45.9 |
| 256 | 99.9 / 52.0 | 62.4 / 33.9 | 91.2 / 44.4 |
| 1024 | 99.9 / 53.0 | 65.3 / 35.2 | 96.0 / 41.3 |

FFF GA falls as leaves multiply (45.9 → 41.3) while MA rises: more leaves memorize more and generalize less. FFF trains in 9-22 epochs to best GA versus 315-4880 for MoE [P].

Table 3, 4-layer ViT, hidden 128, CIFAR10, FFF training width 128 [P]: dense GA 84.7; ℓ=32, d=2: 83.6 (2.44x); ℓ=8, d=4: 82.8 (3.29x); ℓ=1, d=7: 79.8 (3.93x, inference size 8 = 6% of 128). The abstract's "1% of neurons, 94.2%" is 79.8 / 84.7 with inference width 1 [I].

## Smallest scale tested, and whether the result held there
Everything is small: 16-128 hidden neurons, input 256-3072 pixels, and a 128-wide 4-layer ViT. At that scale the claim "one leaf approaches the full layer" holds only for larger leaves (ℓ ≥ 8: −1 to −3.5 points GA); ℓ=1 loses 5-16 points on MLPs. No text, no autoregressive model, no quantization.

## Training recipe and gotchas
- Table 1: plain SGD, lr 0.2, batch 256, h=3.0. Table 2: Adam 1e-3, halved on plateaus, batch 4096, up to 7000 epochs. ViT: Adam 4e-4, halved on 50-epoch plateaus [P].
- Training costs the full 2^d·ℓ width: every leaf is run for every input [P].
- Overfragmentation: deep trees with small leaves give each leaf few samples ("shrinking batch"), causing localized overfitting. Remedies named but not tested: random child transposition, bigger batches, gradient accumulation, lower lr [P].
- Hardening can stall in wide layers without the entropy loss [P].
- Results are best-of-10, not means [P].

## Mapping onto the GBA design
All [I]. Our "1 of 64 hard-routed experts" is an FFF with d=6 and a wide leaf. What this paper adds is the router:
- Tree router with n=1: 6 ternary rows of width 256 = 1,536 weights, about 2.5K cycles, using the 4-input partial-sum tables the layer already builds. A flat 64-way router costs 16,384 weights, about 26K cycles. With a leaf of 128 hidden units (2·128·256 = 65.5K weights, about 105K cycles) the saving is about 20% of the MLP cost.
- ROM: 63 rows of 256 trits per layer for routing (about 4 KB); 64 leaves × 65.5K weights = 4.2M weights, about 1 MB per layer at 4 weights per byte. No extra RAM beyond one node index.
- Each node needs one non-sequential ROM seek; leaves are contiguous.
- Token-id or hash-keyed routing is cheaper still (zero dot products), so the tree is only worth it if content-based routing measurably beats id-based routing.
- Training cost is the problem: soft mixing runs all 64 leaves, i.e. a 64x wider MLP per layer.

## Verdict
Test, as the router for the existing expert MLP, not as a new design. The paper supports "tree routing beats gated MoE at equal inference width" (CIFAR10 only) and warns that more leaves raise memorization while lowering generalization. Settling experiment: at fixed leaf width, compare (a) token-id routing, (b) depth-6 tree with soft training plus entropy loss, (c) depth-6 tree with hard routing as in 2604.08565, (d) flat 64-way argmax; report validation perplexity and leaf-usage histogram.

## Could not verify
- "Up to 220x faster than FF" and "6x faster than MoE": abstract only; Figures 3-4 (768-wide layer, ℓ=32, depths 1-15, batch 256, A100) carry the data and I could not read them.
- SVHN / CIFAR10 / CIFAR100 MLP results (Figure 2) not read.
- Leaf activation is not stated in the paper; ReLU comes from the README default.
- Transposition probability, run counts for Tables 2-3: not given.
- Appendix tables not checked against Table 1.

## Links worth following
- https://github.com/pbelcak/fastfeedforward (reference PyTorch code, MIT)
- https://arxiv.org/abs/2311.10770 (UltraFastBERT: ℓ=1 trees in a language encoder)
- https://arxiv.org/abs/2604.08565 (hard-routed trees in autoregressive LMs; already in notes as capacity-06)
