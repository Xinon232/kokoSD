# How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data (FinePhrase)

https://arxiv.org/abs/2604.13977 (v1 15 Apr 2026, v2 30 Jul 2026). Niklaus, Yamaguchi, Štefánik, Penedo, Kydlíček, Bakouch, Tunstall, Beeching, Frere, Raffel, von Werra, Wolf (Hugging Face; Sheffield; NII). Dataset: https://huggingface.co/datasets/HuggingFaceFW/finephrase

Reading method: arXiv HTML v2 (main text and appendices) and the dataset card, both read through a fetch tool that returns extracted text, so numbers are primary but prompts are paraphrased, not verbatim.

## What it is
A controlled study of LLM-rephrased web text as pretraining data, varying (a) the rephrasing prompt, (b) the generator model and its size, (c) the source documents and the real data mixed in. More than 1 trillion tokens were generated; each arm trains the same 1.2B student for 21B tokens and is scored on 12 benchmarks. The product is FinePhrase: 486B tokens of FineWeb-Edu rewritten by SmolLM2-1.7B-Instruct in four formats. It is not about dialogue, small students, or simple English.

## Mechanism
Pipeline (DataTrove v0.6.0 + vLLM v0.8.4; training Nanotron v0.4; eval LightEval v0.9.1dev0, Appendix C):

```
for doc in source (FineWeb-Edu, classifier score 4-5 = "HQ"):
    out = generator(prompt_k + doc)      # no system prompt
corpus = 50% out + 50% original real docs (default; 60-80% synthetic is peak)
```

Generation settings (dataset card): SmolLM2-1.7B-Instruct, temperature 1.0, top_p 1.0, top_k 50, max 2,048 output tokens, 8,192 context, suffix-32 speculative decoding. Average completion 273-437 tokens.

Four new "pedagogical" prompts (Appendix B, opening lines roughly):
- math: rewrite the document as a multi-step word problem with worked solution, based on numbers/relations in the text.
- faq: rewrite as a standalone FAQ ordered from foundational to advanced.
- table: organise key facts as a markdown table, then add one question-answer pair.
- tutorial: rewrite as a numbered step-by-step guide.

Compared against Nemotron-CC prompts (Diverse QA Pairs, Extract Knowledge, Distill, Wikipedia, Knowledge List), REWIRE Guided Rewrite, BeyondWeb Continue/Summarize.

Student: Qwen2-style 1.2B, 28 layers, d=2048, 128,256 vocab; AdamW lr 5e-4, betas 0.9/0.95, wd 0.1, clip 1.0; batch 512 x 4096, 10k steps; 64 H100s. Eval: 3-shot cloze, macro average over 6 categories.

## Numbers
All macro averages, 1.2B student, 21B tokens, unless stated.
- Baselines (Table 2): DCLM 13.77, FineWeb-HQ 11.82, FineWeb-LQ 8.83, Nemotron-HQ-Synth 13.54, REWIRE 13.49, Cosmopedia 10.33, SYNTH 10.03.
- Older prompts (Table 3): Diverse QA 14.58, Continue 13.73, Guided Rewrite 13.72, Summarize 13.01, Extract Knowledge 11.81.
- New prompts, Gemma-3-1B generator (Table 4): math 15.31, table 14.83, faq 14.45, tutorial 14.30.
- Generator size, Gemma 3, math prompt (Table 5): 270M 13.80, 1B 15.31, 4B 15.06, 12B 14.68, 27B 14.76. Exception: Guided Rewrite gains from 1B to 4B (13.72 to 14.58; Table 20 appears to be this prompt).
- SmolLM2 size, tutorial prompt (Figure 2): 135M 12.69, 360M 14.45, 1.7B 15.88.
- Families at ~1-2B, mean of four prompts (Table 6): SmolLM2 16.55, Falcon 3 15.54, Granite 3.1 14.87, Llama 3.2 14.79, Gemma 3 14.72, Qwen 3 14.49.
- Synthetic-only vs 50/50 (Table 7): math 15.20 vs 15.31, faq 13.12 vs 14.45, table 13.74 vs 14.83, tutorial 12.32 vs 14.30. Loss is mostly on NLU/commonsense tasks.
- Mixing ratio (Table 22, SmolLM2-1.7B): 10% synthetic 15.1, 30% 15.8, 50% 16.4, 60% 16.8, 80% 16.8, 90% 16.7; DCLM 13.8.
- Mix-in quality matters more than source quality: rephrased LQ source + HQ mix-in 12.99 vs FWHQ alone 11.82; LQ source + LQ mix-in 9.63 (Tables 8-9).
- Seeds (Table 21, FWHQ vs FWHQ+table): 0.5B student 9.59±0.44 to 14.30±0.20; 1.7B 12.09 to 17.25; 6.2B 13.38 to 18.61.
- Diversity (Section 5.2, App. E.3): of 10,000 math outputs, 7,619 from Qwen 3 start with "Problem:"; SmolLM2 has 1,897 distinct openings and only 68% "complete" solutions, yet trains the best student. Format consistency correlates negatively with downstream score (rho = -0.29, p < 0.01).
- Format has side effects: table format scores 7.66 on HellaSwag vs 9.25 for DCLM (Section 5.1).
- Cost (Section 6, Table 10): about 9,200 tokens/s per H100, about 33.1M tokens per GPU-hour, 612 GPU-days for 486B tokens. 12B-27B generators cost 5-10x more GPU time.
- FinePhrase-table: 17.18, +3.41 over DCLM.

## Smallest scale tested, and whether the result held there
Smallest student is 0.5B parameters (gain +4.71, as large as at 6.2B). Smallest generator is SmolLM2-135M (clearly worse: 12.69) and Gemma-3-270M (13.80, about equal to DCLM). Nothing near Koko's 50-120M ternary weights, and no perplexity or fluency metric, only knowledge/reasoning benchmarks.

## Training recipe and gotchas
- Sample at temperature 1.0; do not constrain output to a rigid template. Template collapse (identical openers) hurts more than incomplete answers.
- Always mix real text in; 40-80% synthetic is a flat plateau.
- Pick the generator by instruction-following/rewrite training (authors credit SmolLM2's Smol-Rewrite data), not by size. Below about 1B quality drops fast.
- A single format skews skills (tables help SQuAD-like reading, hurt commonsense continuation), so use several formats.
- 10-gram contamination is not increased by rephrasing (735 vs 841 flagged docs per billion tokens, App. E.4).

## Mapping onto the GBA design
Zero run-time cost: no lookups, adds, RAM or ROM change. This is purely a corpus-generation recipe.
- Throughput: at the paper's H100 rate, 33M tokens per GPU-hour; a 5090 rate is not measured in the paper. Even at a fraction of that, 20-30 GPU-hours gives several hundred million to about a billion tokens, more than a Koko-sized student can use, so generation is not the bottleneck and a 27B+ generator is not needed for rephrasing.
- Adaptation needed: Koko prompts would be "rewrite as a simple-English chat between user and Koko", "as simple Q and A", "as a short explanation for a child", plus a vocabulary constraint. These are untested by the paper.
- Keep 40-60% real simple-English text (Simple Wikipedia, children's text) in the mix.
- Monitor opener diversity (count distinct first-8-token prefixes per 10k samples) as a cheap collapse detector.

## Verdict
Test, leaning adopt for the generation pipeline. The claims that transfer safely: prompt format beats generator size, mix real with synthetic, diversity beats format obedience. What does not transfer automatically: students 10-20x smaller, dialogue output, and a simple-vocabulary constraint (a 1.7B generator may be much worse than a 12-30B one at multi-turn conversation and at holding a vocabulary limit). Settling experiment: generate 100M tokens of Koko-format data with SmolLM2-1.7B and with a 12-27B model from the same sources and prompts; train the same small float proxy (about 20-30M parameters) on each, 50/50 with real text; compare held-out real-dialogue perplexity and a judged chat sample.

## Could not verify
- Exact prompt texts were only seen as opening lines and paraphrase.
- The paper's code repository URL and the blog/Space URL were not captured; the dataset card cites "The Synthetic Data Playbook: Generating Trillions of the Finest Tokens" (2026) and DataTrove.
- Table 20's prompt is inferred to be Guided Rewrite from matching numbers.
- A claim that with a 6.2B student the 270M-to-1B generator jump is +2.46 came through the extraction without a table number.
- No 5090 throughput figure exists in the source.

## Links worth following
- https://huggingface.co/datasets/HuggingFaceFW/finephrase (card with generation settings)
- https://github.com/huggingface/datatrove (generation pipeline)
- Nemotron-CC, REWIRE and BeyondWeb papers (the compared prompt sets; URLs not fetched)
