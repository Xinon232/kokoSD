# Adapting Large Language Models for Character-based Augmentative and Alternative Communication
https://arxiv.org/abs/2501.10582 (v3; 2025; Dylan Gaines, Keith Vertanen, Michigan Tech; Findings of EMNLP 2025 per ACL Anthology listing [S]). Read in primary form: arXiv HTML v3 in full incl. appendices, plus the released `search_algorithm.py` [P]. [M] = my arithmetic or inference.

## What it is
A study of how to get next-character distributions and word predictions out of subword LMs for letter-by-letter text entry (AAC). It gives a beam search that marginalises token probabilities into character probabilities, compares it with a byte LM, a classification head and 12-gram models, and tests domain adaptation. Evaluation is per-character perplexity and simulated keystroke savings (KS) on five test sets of short conversational sentences. It is the closest published protocol to Koko's predictive-typing eval.

## Mechanism
**Char search (Sec. 3, App. C, code)** [P]
```
startup: vocab[prefix] = {token ids whose lowercased text starts with prefix},
         only tokens made entirely of symbols {a-z, ', space}
predict(context):
  cut = last space in context; tokens = BOS + encode(context[:cut])   # drop the whole in-progress word
  hyps = [(logp=0, tokens, len=cut)]
  while hyps and completed < 32000:
    sort hyps best-first; one batched forward; lp = log_softmax(last logits) + hyp.logp
    for hyp, for token in descending lp:
      rem = lower(context)[hyp.len:]
      allowed = all valid tokens if rem=="" else vocab[rem]  U  {single tokens equal to a proper prefix of rem}
      if token not allowed: continue
      if hyp.len + len(token) > len(context): char_lp[token[len(context)-hyp.len]].append(lp); completed++
      else: push into next beam (min-heap, width 8)
    hyps = next beam
  P(c) = softmax_c( logsumexp(char_lp[c]) )
```
Case is ignored when matching; context is lowercased except auto-capitalised sentence start and I/I'm/I'll/I've/I'd. No KV cache; fp16 inference.

**Why back off to the word start**: for prefix "yeste", marginalising first characters without removal gives P(r)=0.0047; with removal 0.9999 [P].

**Keystroke savings (Sec. 7)** [P]: KS = (ka − kp)/ka × 100. Up to 5 word slots; shown before the first letter of each word and after every prefix; error-free user; no removal of previously rejected suggestions; a hit costs 1 keypress and supplies the following space. Text: lowercase, only a-z, apostrophe, space; isolated sentences; no end token.

**Baselines** [P]: 12-gram character models, Witten-Bell, KenLM; "AAC n-gram" 70M params (21B chars); C4+Sub mixture (weights 0.84/0.16) 5B params/30 GB, entropy-pruned to 71M/488 MB.

## Numbers
Dev per-char perplexity, avg written+spoken (Table 1) [P]: AAC n-gram 2.55 (0.1 ms CPU); distilgpt2 82M 2.71; gpt2 124M 2.54; opt-125m 2.39; opt-350m (331M) 2.31; opt-6.7b 2.20 (best); Llama-3.2-1B 2.37. Search time 55-120 ms/char on an A100.

Adaptation of opt-350m, dev (Table 2) [P]: base 2.30; BOLT+Daily one step (2.3M words) 2.07; curricula 2.09-2.10; all merged 2.19; LoRA 2.09-2.11. Random C4 instead of classifier-selected: 2.32 vs 2.22 (Table 8).

Test, 5-set average (Tables 3, 4) [P]:

| model | ppl | KS |
|---|---|---|
| AAC n-gram 70M | 2.54 | 59.7% |
| C4+Sub n-gram 71M / full 5B | 2.58 / 2.53 | 59.6% / 60.5% |
| opt-350m base, search | 2.28 | 63.2% |
| opt-350m BOLT+Daily, search | 2.11 | 65.2% |
| ByGPT5 289M adapted, direct | 2.21 (11.6 ms) | 64.5% |
| opt-350m + char classifier head | 2.52 (14.6 ms) | not given |

Per-set KS spread for the best model: 62.7% (AAC user) to 67.2% (spoken). Mixing the n-gram into the fine-tuned LLM gave no gain (2.10 both; n-gram weight 0.03-0.04) (B.3).

## Smallest scale tested, and whether the result held there
Smallest neural models: distilgpt2 (82M) and gpt2 (124M), perplexity only, not fine-tuned. At that scale the LLM advantage is gone: distilgpt2 loses to the 70M n-gram (2.71 vs 2.55) and gpt2-124M ties it (2.54). KS was reported only for 289-331M models. Nothing near Koko's effective compute was tested. An adapted opt-125m is released without numbers.

## Training recipe and gotchas
- Final recipe: full fine-tune on BOLT+Daily only; dropout 0.483, warmup 905, WSD stable fraction 0.009, LR 1.73e-5, weight decay 0.323, batch 6720, 9 epochs, 8-bit AdamW; Optuna (Table 7).
- A small well-matched set beat billions of classifier-selected words for the pretrained subword LM; the byte model needed the big data first (2.14 vs 2.27).
- A character-classification head trained on truncated sentences was worse than the unadapted base (2.52 vs 2.28): do not train on word fragments.
- Written/spoken test sets are private; ALS bank and COMM2 are public and possibly in pretraining data.

## Mapping onto the GBA design
- The full search costs (rounds × up to 8) forward passes with no cache: at 3-5M cycles per pass that is tens of seconds per keystroke on a GBA. Not portable as is [M].
- What transfers is the rule: never feed partial-word tokens; condition on context up to the last space. Koko then needs one forward pass per word boundary, which it already does. Cache the top-K (token id, score) from the class/member head (K=64: about 200 bytes RAM) and re-filter that list by typed prefix on each keystroke: zero multiplies, a few hundred compares [M].
- Prefix filter in ROM: token ids sorted by lowercased string (16K × 2 bytes = 32 KB) so a prefix is a contiguous range found by binary search [M].
- Multi-token words need extension passes. Design consequence: make word-initial tokens cover whole simple-English words so that one pass suffices, and allow at most one beam-1 extension on the best partial candidate [M].
- The class/member head must expose member scores for several top classes, not only the sampled one, to fill 5 slots [M].
- A hashed n-gram table is a credible typing engine alone: 70M-parameter 12-grams reach 59.7% KS at 0.1 ms.

## Verdict
Adopt the KS protocol and text normalisation exactly; adopt the back-off-to-word-start rule; test the cheap cached-top-K variant instead of beam 8. Settling experiment: on COMM2 and ALS bank, compute KS for gpt2-124M with full search, Koko with cached top-K and one extension, and a ROM-sized n-gram. Expect a target band of roughly 55-60% for Koko [M]; the paper gives no gpt2 KS number.

## Could not verify
- The word-prediction algorithm behind Table 4 is not described (credited to Soufia Bahmani in acknowledgments); the released `search_algorithm.py` contains only character `predict`, though its docstring mentions `predict_word`.
- No KS for gpt2/distilgpt2/opt-125m; no CPU timings for the search; no numbers for opt-125m-aac.
- osf.io/ajm7t now says "moved"; materials are at Zenodo record 24266154 (read via its API).

## Links worth following
- https://zenodo.org/records/24266154 (code, models, classified C4/Subtitle sets)
- https://osf.io/c6mnz (AAC character n-gram models)
- COMM2 (Vertanen 2013) and ALS voice-banking (Costello 2014) public test sets; URLs not checked
