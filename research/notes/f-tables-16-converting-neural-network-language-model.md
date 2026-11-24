# Converting Neural Network Language Models into Back-off Language Models for Efficient Decoding in ASR

URL: https://www.research.ibm.com/publications/converting-neural-network-language-models-into-back-off-language-models-for-efficient-decoding-in-automatic-speech-recognition
IEEE/ACM TASLP 22(1):184-192, 2014 (DOI 10.1109/TASLP.2013.2286919); ICASSP 2013 version pp. 8242-8246. Ebru Arisoy, Stanley F. Chen, Bhuvana Ramabhadran, Abhinav Sethy (IBM).

**Access status:** the TASLP and ICASSP PDFs are paywalled and I could not read them. Everything marked [patent] comes from the description of the same authors' patents US9514744 / US9934778 (same method, same experiment as the ICASSP paper), read through a page-summarising fetch tool, so equations are reconstructions. [abstract] = IBM page summary. [Wang13] and [Krishnan23] were read as PDFs.

## What it is
A recipe for baking a feed-forward neural LM (NNLM) into a static ARPA back-off n-gram table so a speech decoder can use it in the first pass with plain lookups. The neural model's probabilities overwrite the n-gram entries; a conventional count-based n-gram ("background LM", BLM) supplies everything the network does not cover and decides which histories exist. Conversion is hierarchical: 2-gram, then 3-gram, then 4-gram, pruning after each order, so the pruned lower-order table limits which higher-order n-grams are ever enumerated. The converted table keeps most, not all, of the neural gain [abstract].

## Mechanism
[patent] Notation: V_o = NNLM output shortlist; h = history; h' = h minus its oldest word.

- Smoothed NNLM (Eq. 1-2): P(w|h) = beta(h) * P_NN(w|h) if w in V_o, else P_BLM(w|h); beta(h) = sum over v in V_o of P_BLM(v|h). The network only redistributes the mass the BLM gives the shortlist.
- Converted model (Eq. 6), three cases: (1) w in V_o: beta(h) P_NN(w|h), explicit entry; (2) w not in V_o and (h,w) explicit in BLM: P_BLM(w|h); (3) otherwise alpha(h) P_BLM(w|h'), ordinary back-off.
- Tractability: only histories h that exist in the BLM get NNLM entries, giving k * |V_o| candidate n-grams (k = unique histories) instead of |V|^n. Unseen histories back off.
- Optional count switch (Eq. 5): if estimated count C_est(h,w) > K, keep the BLM probability (frequent events are already well estimated); else use the NNLM.

Pseudocode for order N=4:
```
train NNLM_2, NNLM_3, NNLM_4 and count LMs CLM_2..4 on the same text
M = prune(convert(NNLM_2, BLM=CLM_2))            # Stolcke entropy pruning
for n in 3,4:
    BLM_n = M + n-grams of CLM_n; renormalise (n-1)-gram back-off weights
    M = prune(convert(NNLM_n, BLM_n))            # histories limited to those in BLM_n
final = interpolate(M, baseline 4-gram CLM)      # weight tuned for held-out perplexity
```
Pruning: relative-entropy (Stolcke 1998), one threshold per order, set so the final n-gram count equals the baseline LM's. The result may contain n-grams never seen in training text. A separate NNLM per order is trained (120-dim word features, 800 tanh units, 20K shortlist).

## Numbers
[patent, Table 1] English Broadcast News, rt04 test, 55M training words, 80K vocabulary, 430 h acoustic model; held-out set 49K words.

| System | WER |
|---|---|
| 4-gram baseline, decoding | 14.7% |
| + rescoring with 4-gram NNLM | 13.3% |
| + rescoring with 6-gram NNLM | 13.2% |
| converted 4-gram back-off NNLM (same n-gram count), decoding | 13.7% |
| its lattices rescored with 4-gram NNLM | 13.0% |
| its lattices rescored with 6-gram NNLM | 12.8% |

So the table recovers 1.0 of the 1.4 points the real 4-gram NNLM gives (about 70%), at zero added size. Table plus neural pass beats neural pass alone by 0.3-0.4.

[Wang13, Table 1] Same rewrite idea without the hierarchy (the authors call the methods identical), Chinese-English patent MT, 5-gram, CSLM 256-dim projection / 384 hidden / 8192 shortlist: BLEU 31.60 -> 32.58 (42M-word LM), 32.83 -> 33.22 (746M-word LM rewritten by a 42M-word network).

[Krishnan23, Table 3] GPT-2 124M fine-tuned, trigram targets, tiny domain corpora: probability conversion (PBA) is much worse than sampling 100x text and counting (RS-KN3). HUB4: GPT-2 189.91, sampled 369.90, PBA 632.92. ATCO2: 25.04 / 47.04 / 80.04. Both lose at least 2x perplexity to the teacher in 6 of 8 domains.

## Smallest scale tested, and whether the result held there
The source itself: only 55M words, 80K vocabulary, 4-gram; no smaller. Nearest small-scale evidence is [Krishnan23] (27K-735K training words): there probability conversion did not hold up, sampling did. Perplexities for the Arisoy model were not available to me.

## Training recipe and gotchas
- Needs one teacher per order (or a teacher that can be queried with truncated context); a long-context teacher asked for a 3-token-context probability is out of distribution. [Krishnan23] reports GPT-2 over-estimating lower-order n-gram probabilities under conversion.
- Cost is k * |V_o| teacher outputs, one forward pass per history; embarrassingly parallel.
- Prune after every order or the candidate set explodes.
- Always interpolate with the count model; every reported gain is post-interpolation.
- Gain is evaluated on WER, a forgiving metric; no evidence of preserved generation quality.

## Mapping onto the GBA design
Runtime is pure lookup: hash the last 1-3 token ids, one ROM probe per order, back off by adding a stored 8-bit log back-off weight. No multiplies if entries are quantised log-probs. Per history the entry is a sorted successor list (2-byte token id + 1-byte logit = 3 bytes), read sequentially: tens to a few hundred cycles for a top-k list, negligible against 3-5M. RAM: a 16K-entry logit scratch is not needed if the table only biases the candidates the two-level head already scores; a few hundred bytes suffice.

ROM: about 3-4 bytes per n-gram, so 10M n-grams is 30-40 MB uncompressed; realistic share is 2-4M n-grams (8-12 MB), competing with ternary weights. This is our estimate, not the paper's.

Use for us:
1. Fill the hashed n-gram tables with teacher-conditional logits for teacher-chosen histories instead of counts, then prune by relative entropy against the lower order. This is the paper's direct lesson and addresses the tokens-per-table-parameter limit.
2. Use it as the candidate generator for the output head and for predictive typing: top-8 successors per history are exactly what the keyboard needs and skip the transformer when the user is typing.
3. Train the transformer as a residual on top of the frozen table (the "table lattices + neural rescoring" row is the best one).

Changes needed: word-level becomes 16K subword; replace separate NNLMs per order with one teacher marginalised by sampling; quantise log-probs to 8 bits.

## Verdict
**Test.** Cheap at run time and a good fit, but the evidence is 2013-era, WER-only, and the one modern replication favours sampling over probability rewriting. Settling experiment: on our corpus build three 4-gram tables of equal size (2M entries): (a) KN counts on real data, (b) KN counts on real + 20-100x teacher-sampled text, (c) teacher-probability rewrite with hierarchical entropy pruning. Compare held-out perplexity alone and as a logit prior under the same small transformer. Adopt whichever beats (a) by more than about 5% once the transformer is on top.

## Could not verify
- The TASLP article text: its perplexity tables, n-gram counts, comparison with Deoras et al.'s variational sampling ("improves over existing methods" is abstract-only), conversion time, and any journal-only additions.
- Exact equation forms (patent page rendering garbled; reconstructed by the fetch tool).
- The value of K, pruning thresholds, interpolation weights.
- Adel et al. 2014 (RNN conversion comparison): not accessible.

## Links worth following
- Patent with full method text: https://patents.google.com/patent/US9514744
- Wang et al., EMNLP 2013: https://aclanthology.org/www.mt-archive.info/10/EMNLP-2013-Wang.pdf
- Krishnan, Alabi, Klakow 2023, "On the N-gram Approximation of Pre-trained Language Models": https://arxiv.org/abs/2306.06892
- Stolcke, entropy-based pruning: https://arxiv.org/abs/cs/0006025
- Deoras et al., ICASSP 2011, variational approximation of long-span LMs (not read).
