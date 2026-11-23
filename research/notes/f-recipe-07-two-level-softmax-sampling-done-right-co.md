# Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion

https://arxiv.org/abs/2610.10483 (v1, 2026; W. Bendada, G. Salha-Galvan). Code: https://github.com/bendadaw/neurips-2ls (link taken from the paper; repository not opened).
Read in primary form via https://arxiv.org/html/2610.10483, through a fetch tool that summarises; the equations and tables below were extracted from the paper text, not from a secondary write-up.

## What it is

A sampling paper from the retrieval/recommender setting, not a language-model training paper. Items have fixed embeddings x_i; a query q defines the exact softmax p(i|q) ∝ exp(qᵀx_i/τ). "Two-level softmax" (2LS) approximates sampling from it post hoc: k-means the items, sample a cluster from a softmax over CENTROID scores, then sample an item exactly within that cluster. The paper proves that this misweights clusters by a factor depending on cluster size and within-cluster score spread, and gives two corrections to the cluster-level logits. Nothing is trained; the partition and corrections are applied to a frozen embedding table.

## Mechanism

Definitions: clusters C_k, k=1..K; centroid mu_k = mean of x_i in C_k; pi_k = cluster share.

- 2LS (Eq. 2-3): p(k|q) ∝ exp(qᵀmu_k/τ); p(i|k,q) ∝ exp(qᵀx_i/τ) over i in C_k.
- Bias (Prop. 1-3, Eq. 5-6): ratio p_2LS(i|q)/p(i|q) depends only on i's cluster, and tends to c(q)·exp(qᵀmu_k/τ) / (pi_k · E_{x in k}[exp(qᵀx/τ)]); under a Gaussian approximation ∝ 1/(pi_k · exp(sigma_k²(q)/2)). So plain 2LS undersamples large clusters and high-spread clusters.
- S-2LS (Eq. 7): cluster logit = qᵀmu_k/τ + log|C_k|. Member step unchanged. Same cost as 2LS.
- SD-2LS (Eq. 10): cluster logit = qᵀmu_k/τ + log|C_k| + ½·sigma_k²(q), with sigma_k²(q) = Var_{x in C_k}(qᵀx/τ). The paper evaluates this as a quadratic form in the d×d intra-cluster covariance Sigma_k; sigma_k²(q) = qᵀSigma_k q/τ² is my inference, the paper does not print it. Cost O(d²K + d|C_k|). Under the Gaussian assumption the ratio tends to 1 (Prop. 6).

Pseudocode (mine; the paper has none):

    for k in 0..K-1:
        s[k] = dot(q, mu[k]) / tau + log_size[k]              # S-2LS
        s[k] += 0.5 * quad(q, Sigma[k]) / tau^2               # SD-2LS only
    k* = sample(softmax(s))
    i* = sample(softmax(dot(q, x[i]) / tau for i in C[k*]))

Note that the log|C_k| term is NOT divided by τ, and the variance term scales as 1/τ².

## Numbers

Setup: K = 1024 k-means clusters, 1,000 queries per dataset, τ in {0.05, 0.1, 0.2}. Datasets: VK-LSVD (N = 19.6M, d = 64), YAMBDA (N = 7.7M, d = 128), GloVe-100 (N = 1.2M, d = 100), two synthetic Gaussian mixtures (N = 10^6, d = 100).

Table 1, KL(approx || exact), mean over queries:

| τ | Dataset | 2LS | S-2LS | SD-2LS | Top-k (1000) | Hier. |
|---|---|---|---|---|---|---|
| 0.05 | GloVe-100 | 0.0961 | 0.0711 | 0.0288 | 1.9137 | 0.5231 |
| 0.05 | VK-LSVD | 0.0541 | 0.0551 | 0.0129 | 2.4902 | 0.7610 |
| 0.05 | YAMBDA | 0.0485 | 0.0551 | 0.0177 | 1.7466 | 1.4484 |
| 0.05 | Synth-bal | 0.5968 | 0.3929 | 0.6586 | 0.0003 | 9.4151 |
| 0.05 | Synth-unbal | 0.8959 | 0.7096 | 0.1740 | 0.0090 | 10.3033 |
| 0.1 | GloVe-100 | 0.0615 | 0.0022 | 0.0001 | 3.9907 | 0.2897 |
| 0.1 | VK-LSVD | 0.0297 | 0.0051 | 0.0002 | 5.2351 | 0.3425 |
| 0.1 | YAMBDA | 0.0300 | 0.0098 | 0.0005 | 3.5266 | 0.6049 |
| 0.2 | GloVe-100 | 0.0677 | 0.0001 | 1.2e-6 | 5.4103 | 0.3007 |
| 0.2 | VK-LSVD | 0.0337 | 0.0003 | 2.7e-6 | 7.3333 | 0.3153 |
| 0.2 | YAMBDA | 0.0388 | 0.0008 | 5.3e-6 | 5.6988 | 0.3263 |

The gain is strongly temperature-dependent: at τ = 0.05 (sharp distributions) S-2LS is no better than 2LS on VK-LSVD and worse on YAMBDA, and SD-2LS is worse than 2LS on Synth-balanced. So the abstract-level "consistently improves at no cost" does not hold in every cell.

Table 2, single-thread CPU latency per query (ms), averaged over τ: GloVe-100 exact 10.30, 2LS 0.235, S-2LS 0.245, SD-2LS 0.722; VK-LSVD 173.0 / 2.564 / 2.662 / 2.906; YAMBDA 79.28 / 1.265 / 1.660 / 3.171. S-2LS is cost-neutral; SD-2LS is roughly 1.1-3x 2LS.

## Smallest scale tested, and whether the result held there

Smallest is N = 10^6 items (synthetic) and N = 1.2M (GloVe-100), K = 1024, i.e. about 1,000 items per cluster. That is 60-75x Koko's 16K vocabulary. Nothing was tested at vocabulary scale, with a learned class head, or with a language model. The theory is asymptotic in cluster population; with only 64-250 members per class the Gaussian approximation is weaker.

## Training recipe and gotchas

There is no training. Gotchas for anyone reusing it:
- It only applies when cluster logits are CENTROID scores. A class-then-member head trained with the exact factorised loss -log p(class) - log p(member|class), with its own class weights and biases, is already a valid distribution and has no such bias; the class bias learns log-size by itself.
- The size term must stay outside the temperature division.
- The dispersion term grows as 1/τ², exactly where the second-order approximation is least accurate (the τ = 0.05 rows).
- The paper flags skewed or heavy-tailed embeddings as a failure case for SD-2LS.

## Mapping onto the GBA design

Koko's head is trained as class-then-member, so as designed the paper's bias does not arise and nothing needs to change. It becomes relevant in three cases:

1. Tied/derived class vectors. If we drop the separate class matrix and use member-embedding centroids (saves K×d ROM weights, a negligible saving against 50-120M), then add log|C_k| per class: one 16-bit ROM constant and one add per class, about 128-256 adds per token, effectively free. Not worth doing just for ROM.
2. Initialisation. Initialising class weights to member centroids and class biases to log|C_k| (or log unigram class mass) is a free, sensible start for QAT. This is my suggestion, not tested in the paper.
3. Temperature sampling. With a trained factorised head, applying temperature T separately at both levels does not equal sampling from the tempered flat softmax; the exact class weight would be logsumexp of member logits/T, which needs every class's members (far too expensive). The paper's lesson transfers as a caution, not a recipe: keep one fixed sampling temperature and tune it at both levels empirically, or use top-k within the top 1-2 classes.

SD-2LS as published is unaffordable: d²K = 256²×128 is about 8.4M multiplies per token, more than the whole budget. A diagonal-covariance variant (d×K squares, about 33K multiply-accumulates, about 250K cycles) is my extrapolation and is not evaluated in the paper.

Predictive typing: top-3 suggestions need class probabilities to be comparable across classes of different size; a trained class bias handles this, a centroid shortcut would not without the log-size term.

## Verdict

Skip as a design component; keep the log|C_k| bias init as a zero-cost detail. Settling experiment (offline, PC): on the trained head, compare token-level KL between the factorised distribution and a flat softmax over the same 16K logits at T in {0.7, 1.0}, with and without frequency-balanced classes. If KL is below about 0.01 nats, nothing here matters.

## Could not verify

- Hardware used for Table 2, and any ablation over K or cheaper covariance forms: not found in the text read.
- Figures 1-19 (bias plots) were read as descriptions only, no underlying numbers.
- The code repository was not opened.
- Exact printed form of sigma_k² in terms of Sigma_k is inferred.
- The page was read through a summarising fetch tool; table values were transcribed by it and not checked against the PDF.

## Links worth following

- https://github.com/bendadaw/neurips-2ls (authors' code)
- Goodman 2001, "Classes for Fast Maximum Entropy Training" and Morin & Bengio 2005 (cited as the trained class/hierarchical softmax lineage, which is what Koko actually uses)
- Adaptive softmax (Grave et al. 2017), mentioned in the paper as the training-time alternative; more relevant to Koko's head than this paper.
