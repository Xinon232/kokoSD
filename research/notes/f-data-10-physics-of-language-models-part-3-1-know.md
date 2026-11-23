# Physics of Language Models: Part 3.1, Knowledge Storage and Extraction  (https://arxiv.org/abs/2309.14316, 2023, Zeyuan Allen-Zhu & Yuanzhi Li, Meta FAIR / MBZUAI)

Read in primary form: arXiv HTML (https://arxiv.org/html/2309.14316), all 134K characters, through a summarising fetch tool in two parts. Figures are images and were not seen; every number below comes from the paper's running text.

## What it is
A controlled study of when a fact that a language model has memorised can be pulled out by a question. The authors build synthetic biographies of 100,000 fictional people (6 attributes each), train GPT-2-style models from scratch, and test question answering on people whose questions were never seen in training. The finding: memorising a biography word for word does not make it answerable. Extraction needs either questions mixed into pretraining, or "knowledge augmentation" of the pretraining text (several wordings, shuffled sentence order, repeated full name). Linear probes show the cause: without augmentation the fact is stored against the preceding words of the sentence, not against the person's name.

## Mechanism
Data
- bioS: N = 100,000 people; attributes birth date, birth city (200), university (300), major (100), employer (263), work city. 46-52 sentence templates per attribute; about 73 GPT-2 tokens per biography. Example: "Anya Briar Forger was born on October 2, 1996." followed by five pronoun sentences.
- bioR: the same people, biographies written by Llama-30B, rejection-sampled until all 6 attributes appear; about 72 tokens.
- QA: 6 questions per person ("What is the birth date of Anya Briar Forger?"), scored by exact match. People are split in half: P_train (questions seen) and P_test (questions held out).

Augmentations (applied to pretraining text only)
- multiM: M independently worded biographies per person.
- permuteP: the six sentences shuffled P times (P entries per person). "+permute" on multiM shuffles each entry.
- fullname: every pronoun replaced by the full name.

Three training regimes
1. Mixed: one run on biographies of everyone plus questions of P_train. Each sampled entry is a QA with probability QA_r = 0.8.
2. Pretrain on biographies, then finetune on P_train questions (full or LoRA: rank 8/16 on query/value, rank 128 on embeddings).
3. Celebrity: add 100,000 extra "celebrity" people with multi5(+permute) biographies; the original "minority" people keep one biography. Finetune on celebrity questions only, test on minority.

Probes: P-probing = frozen model, linear classifier at positions before each attribute, predicting attributes not yet mentioned. Q-probing = frozen model fed the name only, low-rank embedding update plus linear classifier at the last token.

## Numbers
All at GPT-2 scale, P_test accuracy, from the text.
- Mixed training (Result 1, Fig. 1): 86.6% on bioS single, 77.7% on bioR single. Learning order: P_train questions first, then biographies, P_test questions last.
- Pretrain then finetune, no augmentation (Result 2, Fig. 2): near 0% for bioS single and bioR single in all finetune settings, despite 99%+ biography memorisation. Also fails at 682M.
- Augmentation (Result 3, Fig. 3): bioS single 9.7% -> 96.6% with 5 diverse entries. Birth date alone reaches 33% without augmentation.
- Section 5.1 text: multiplicity accounts for 9.7% -> 41%; permutation for 4.4% (single+permute1) -> 70% (single+permute5). One shuffle with no multiplicity is worse than no shuffle (4.4% vs 9.7%).
- Footnote 15: adding a French translation gives about 40%.
- Celebrity (Result 6, Fig. 8): minority accuracy 4.4% -> 86.8% (bioS), 10.0% -> 76.3% (bioR); bioR single+fullname+CEL 82.2%. Mixing in WikiBook text instead of celebrities does not help.
- Probing (Results 4-5): on bioS single, the employer is linearly readable at about 2% until the token just before it, then 100%; with multi5+permute it is near 100% right after the name. Q-probe accuracy tracks QA accuracy.
- BERT-style masked model (Result 7): extracts only birth date and major well.

## Smallest scale tested, and whether the result held there
124M (12 layers, 12 heads, 768 wide) on bioS; 302M on bioR; 682M for one negative result. Nothing below 124M. Koko's 1-30M range is untested here. The storage mechanism described is about what the name tokens bind to, which is not obviously size-dependent, but that is my inference.

## Training recipe and gotchas
- Pretraining: AdamW, weight decay 0.1, eps 1e-6, peak lr 1e-3, 1,000 warmup steps, cosine to 1e-4, batch 96, context 512, 80,000 steps (bioS) or 150,000 (bioR). That is 540-1,000 passes per person: every fact is heavily over-exposed.
- Finetune: lr 1e-4 to 1e-3, batch 48, 50,000 steps; LoRA generally beat full finetune.
- Celebrity data sampled with probability 1/6.
- Higher QA_r gave better held-out QA (Remark 3.1); only 0.2 and 0.8 are named.
- Shuffling without rewording can hurt.
- A general corpus (WikiBook) does not substitute for augmented facts of the same kind.

## Mapping onto the GBA design
Run-time cost: none. No lookups, adds, multiplies, RAM or ROM change; this shapes the corpus only.
- For each fact Koko should answer, generate at least 5 differently worded passages with the fact order shuffled, and name the subject in every sentence rather than using "it/he/she". Koko's 256-token ring buffer and top-8 attention make short, name-anchored sentences doubly useful.
- Put question-answer chat turns into training from step one. Do not plan a "language first, chat finetune later" schedule for facts.
- Not every entity needs full treatment: a heavily augmented, question-covered subset taught extraction for the rest in this paper. Token budget can go to a "celebrity" half.
- Hashed n-gram embeddings from ROM key on local token context, which is exactly the "stored against the preceding words" failure the probes show. They will happily memorise fixed phrasing. Varied wording in training is the defence; an entity-keyed table (hash of the subject name) would match the good storage pattern, but the paper does not test this.
- Combined with Part 3.3 (100-1,000 exposures per fact), 5 wordings is a floor: the paper's own runs resample templates across hundreds of passes.

## Verdict
Adopt as corpus rules (they are free); test at Koko's size. Settling experiment: bioS with N = 10-20K on the Koko architecture, ternary QAT, four corpora (single; multi5; multi5+permute+fullname; single for 80% of people plus augmented 20%), mixed QA from the start, report P_test accuracy. A few GPU-hours each.

## Could not verify
- Per-variant values in Fig. 3, the QA_r curve in Fig. 10 and Fig. 9's BERT numbers: figures not readable.
- Whether 96.6% is multi5 with or without sentence shuffling: Result 3's sentence says "5 diverse biography entries"; Section 5.1 attributes only 41% to multiplicity, so 96.6% most likely includes shuffling, but I could not confirm against the figure.
- "15 knowledge augmentations" is stated in the appendix; I recovered about eight.
- Any result under 124M parameters, with quantized weights, or on natural (non-synthetic) facts: absent.
- Total celebrity-to-minority ratio other than 100K:100K: not tested as far as I read.

## Links worth following
- https://arxiv.org/abs/2309.14402 Part 3.2, Knowledge Manipulation (same authors; cited as the companion; not read here).
- https://arxiv.org/abs/2404.05405 Part 3.3, capacity scaling laws (notes in f-data-09).
