# DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation

https://arxiv.org/abs/1911.00536 — 2019 (ACL 2020 demo), Zhang, Sun, Galley, Chen, Brockett, Gao, Gao, Liu, Dolan (Microsoft). Read in full via the ar5iv HTML, including all tables and the appendix [P]. Items marked [R] come from the official repo README / `LSP_train.py` / Hugging Face model card.

## What it is
GPT-2 (117M / 345M / 762M) trained further, or from scratch, on 147M Reddit reply chains so that it produces a reply instead of continuing prose. There is no architecture change: a dialogue is one flat token string and the loss is plain next-token prediction. The paper reports reference-overlap and diversity metrics on two Reddit test sets and pairwise human preference for the 345M model. MIT-licensed [R]; it is the natural chat baseline for Koko. The repo is now unmaintained, superseded by GODEL [R].

## Mechanism
- Sequence: `turn_1 <|endoftext|> turn_2 <|endoftext|> ... turn_K <|endoftext|>`. No speaker or role tokens; the end-of-text id is the only separator [P section 3.1; R model-card code].
- Loss: p(T_K..T_2 | T_1) = product of next-token probabilities, so one sequence trains every (history, reply) pair at once (Eq. 1).
- Shapes (Table 1): 12 layers / 768 wide, 24 / 1024, 36 / 1280; vocabulary 50,257.
- MMI reranking (section 3.2, 4.4): draw 16 replies with top-K sampling (K = 10); score each with a separate backward model P(source | reply) (345M, trained on reversed pairs); keep the lowest backward loss.
- Data filters (section 2): drop pairs with a URL; a reply with a repeated run of 3+ words; a reply containing none of the 50 most frequent English words; "[" or "]"; source+target over 200 words; blocklisted phrases and subreddits; replies where 90% of trigrams occur more than 1000 times in the corpus (about 1% of data, the "bland" filter).

## Numbers
Training set: 147,116,725 instances, 1.8B words, Reddit 2005-2017.

Table 3, 6K-example multi-reference Reddit test (greedy unless noted):

| Model | NIST-4 | BLEU-4 | METEOR | Ent-4 | Dist-2 | Len |
|---|---|---|---|---|---|---|
| 117M from scratch | 1.37 | 1.77% | 6.17% | 7.11 | 15.9% | 9.41 |
| 117M from GPT-2 | 2.41 | 1.55% | 7.53% | 10.77 | 39.9% | 12.82 |
| 345M from scratch | 3.08 | 4.59% | 9.34% | 9.03 | 25.6% | 11.16 |
| 345M from GPT-2 | 3.06 | 4.56% | 9.81% | 9.12 | 26.3% | 12.19 |
| 345M beam 10 | 3.5 | 7.92% | 10.74% | 10.48 | 48.74% | 11.34 |
| 345M MMI | 3.33 | 3.94% | 11.23% | 11.25 | 45.55% | 17.21 |
| Human | 4.25 | 7.48% | 10.64% | 10.99 | 63.0% | 13.10 |

Table 2, DSTC-7 (2208 examples, 5 references): 117M NIST-4 1.60, BLEU-4 2.02%, METEOR 7.17%, Ent-4 6.94, Dist-2 18.94%; GPT-2 345M without dialogue training 1.79 / 1.06% / 6.38%; DialoGPT 345M 2.82 / 2.31% / 8.51%; 345M beam 2.97 / 6.05% / 9.29%; human 2.65 / 3.13% / 8.31%.

Table 7, human pairwise preference (2000 sources, 3 judges each, 345M only): versus PersonalityChat 72% / 19% relevance; versus its MMI variant 40% / 52%; versus beam 50% / 40%; plain 345M versus human 45% / 47% relevance, 45% / 51% informative, 45% / 50% human-like; MMI versus human 48% / 43% relevance.

Table 8: 345M versus human is not significant for relevance only (p = 0.0552); human wins on informativeness and human-likeness (p <= 0.0001). 345M versus 762M: not significant (p 0.71-0.85).

## Smallest scale tested, and whether the result held there
117M is the smallest, and the headline result does not hold there. It was never human-evaluated; its BLEU-4 (1.55%) is a third of the 345M model's and a fifth of human. Pre-training mattered most at this size: from scratch NIST-4 1.37 versus 2.41 from GPT-2, while at 345M scratch and fine-tuned are equal (3.08 versus 3.06). "Near human" is a 345M, single-turn, Reddit-judged claim.

## Training recipe and gotchas
- 16 V100s; batch per GPU 128 / 64 / 32; Noam schedule, 16,000 warm-up steps; up to 5 epochs (3 for 762M); stop on validation loss. Learning rate not stated in the paper; script default 1e-5, README fine-tune example 1e-4, `max_seq_length` 128, fp16 [R].
- Speed [R]: one V100 does about 10.8K tokens/s, 118 h per epoch on 50M instances.
- Policy-gradient training on the MMI reward collapsed into parroting the source.
- Automatic scores above human are an artifact: a "most likely" reply sits near the centre of the references (section 4.2).
- No perplexity is reported anywhere.

## Mapping onto the GBA design
- Chat format: one reserved separator token id, nothing else. Zero run-time cost; the 256-token ring buffer simply contains separators. Turn-start is a free signal for hashed lookups (position-since-separator, speaker parity) if we want role information DialoGPT lacks.
- MMI on the cartridge: 16 samples plus a second backward model is at least 16x the per-token budget and a second set of ROM weights. Not feasible live. Offline use: backward-model scoring and the trigram blandness filter on our generated training conversations.
- Beam 10: 10x compute and 10 copies of state in RAM; skip. Greedy or top-K with small K fits.
- Predictive typing: the joint objective already trains the model to predict the user's turns, so suggestions come from the same forward pass on the user side at no extra ROM.
- From-scratch at small size was the weak row here; our model has no GPT-2 to start from, so distillation and general-text pre-training before dialogue data carry that role.

## Verdict
Adopt the format, the data filters and DialoGPT-small as the second baseline; skip MMI and beam at run time; test backward-model filtering offline. Settling experiment: run GPT-2 small, DialoGPT-small and Koko on one held-out simple-English dialogue set, reporting bits per byte of the reply, Dist-1/2, Entropy-4, average length, and blind pairwise preference with 3 judgments per pair and bootstrap intervals as in Appendix A. DialoGPT is Reddit-register; report that caveat.

## Could not verify
- Whether Table 2's "DialoGPT (117M)" row is from scratch or from GPT-2: not stated.
- Any perplexity or bits-per-byte for any size: not in paper or README.
- Any human evaluation of the 117M model: none exists in the paper.
- Optimizer, learning rate, total steps, training context length: paper silent; script defaults only [R].
- The paper's prose says 345M versus human is "not statistically significant"; Table 8 supports that for relevance only.

## Links worth following
- https://github.com/microsoft/DialoGPT (training script, DSTC and 6K evaluation scripts)
- https://huggingface.co/microsoft/DialoGPT-small (baseline checkpoint, MIT)
- GODEL, https://arxiv.org/abs/2206.11309 (successor; has small-model dialogue numbers)
- Li et al. 2016, "A Diversity-Promoting Objective Function for Neural Conversation Models" (MMI, Dist-n definitions)
