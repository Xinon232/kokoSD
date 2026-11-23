# WildChat: 1M ChatGPT Interaction Logs in the Wild  (https://arxiv.org/abs/2405.01470, ICLR 2024; Zhao, Ren, Hessel, Cardie, Choi, Deng; dataset owner: Allen Institute for AI)

Read in primary form: arXiv HTML of the paper, and the raw dataset cards of `allenai/WildChat-1M`, `allenai/WildChat` and `allenai/WildChat-4.8M` on Hugging Face.

## What it is
A corpus of real, consented conversations between anonymous web users and GPT-3.5/GPT-4, collected through two free chatbot Spaces on Hugging Face from 9 April 2023 to 1 May 2024. It is a data source, not a method: the user turns are what people really type to a chatbot (typos, ambiguity, topic switches, pasted text). The paper compares it with Alpaca, Open Assistant, Dolly, ShareGPT and LMSYS-Chat-1M, and fine-tunes Llama-2 7B on it as a sanity check. A later release (WildChat-4.8M, August 2025) extends it to 3.2M non-toxic conversations.

## Mechanism
Nothing to implement in the model; what matters is the schema and a filter pipeline.

Schema (WildChat-1M card): `conversation` = list of `{role, content, language, toxic, redacted, turn_identifier, ...}`; user turns also carry `hashed_ip`, `country`, `state`, `header`. Conversation-level: `model`, `turn` (rounds), `language` (majority language), `toxic`, `redacted`, `openai_moderation`, `detoxify_moderation`, `hashed_ip`.

Proposed extraction for Koko (my design, not from the paper):
```
for conv in WildChat-1M (public non-toxic version):
    if conv.language != "English": skip
    for turn in conv.conversation where role == "user":
        if turn.language != "English" or turn.redacted: skip
        if content empty: skip                    # card: empty inputs exist
        if len(tokens) > 48: skip                 # drops pasted documents, jailbreak templates
        if out-of-vocabulary rate vs Koko 16K vocab > threshold: skip
        dedupe on normalised text, cap per hashed_ip   # power users, repeated prompts
        emit (preceding assistant turn truncated, user turn)
```
Use: (a) train the user-turn side of the model for typing suggestions; (b) seed prompts for the open-weight LLM that writes simple-English replies, discarding the GPT replies.

## Numbers
All from the paper unless marked.
- Table 1 (Llama-2 tokenizer, mean ± std): WildChat 1,039,785 conversations, 204,736 unique IPs, 2.54 turns, user 295.58 ± 1609.18 tokens, chatbot 441.34 ± 410.91, 68 languages. LMSYS-Chat-1M: 1,000,000 conversations, 210,479 users, 2.02 turns, user 69.83 ± 143.49, chatbot 215.71 ± 1858.09, 65 languages. ShareGPT: 94,145 conversations, 3.51 turns, user 94.46 ± 626.39.
- Section 3 text: about 41% of conversations are multi-turn; 3.7% exceed 10 turns. (The text says 1,009,245 conversations and 2.52 turns; Table 1 says 1,039,785 and 2.54. The paper is internally inconsistent.)
- Table 2: GPT-3.5 about 76%, GPT-4 about 24%.
- Table 3 (IP country): US 21.60%, Russia 15.55%, China 10.02%, Hong Kong 4.62%, UK 3.79%.
- Table 4 (first turn, English, 1,000 sampled conversations, DeBERTa classifier distilled from GPT-4): assisting/creative writing 61.9%, analysis/decision explanation 13.6%, coding 6.7%, factual info 6.3%, math reasoning 6.1%. WildChat only; there is no LMSYS column.
- Table 5 (turn-level language): English 52.94%, Chinese 13.38%, Russian 11.61%. LMSYS is 78.00% English.
- Table 6 (turn-level toxicity, either detector): user 10.46%, chatbot 6.58%.
- Table 9 (MT-bench average): WildLlama 6.35, Llama-2 Chat 6.26, Vicuna 6.13.
- Dataset card: public WildChat-1M has 837,989 conversations after toxic removal, 3.36 GB download, 6.84 GB in memory, 25.53% GPT-4. WildChat-4.8M public: 3,199,860 conversations.
- My estimate, not from the source: 838K x about 53% English x about 2.5 turns gives roughly 1M English user turns before length filtering.

## Smallest scale tested, and whether the result held there
The only model trained is Llama-2 7B (WildLlama). Nothing at or near our scale (a few million to 100M ternary weights). No evidence either way that real user turns help a tiny model.

## Training recipe and gotchas
- WildLlama: Llama-2 7B, data up to 16 July 2023, 3 epochs, learning rate 2e-5, effective batch 128 conversations, max length 2048, four A100 80 GB. Not relevant to us.
- Licence: the paper says AI2 ImpACT; the dataset card says it was changed to ODC-BY on 2024-06-26, retroactively. ODC-BY needs attribution only. The public version is not gated (HF API: `gated: false`); the toxic "Full" version is gated.
- Mean user length of 296 tokens with std 1609 means a heavy tail; the paper gives no median and does not explain it. Length filtering is mandatory.
- 62% of English first turns are writing requests, so the user distribution is "task for an assistant", not small talk. Users skew towards IT people (limitations section).
- Empty user inputs exist. PII is replaced by placeholders (`redacted` flag). Jailbreak templates recur thousands of times (Table 8: Narotica 3,903 occurrences).
- Assistant turns are GPT output: far too complex for Koko, and not what we want it to imitate.

## Mapping onto the GBA design
Zero run-time cost: this changes training data only. Possible ROM uses: a table of the most frequent user openers (first 1-3 tokens of filtered user turns) as a prior for typing suggestions at an empty input line, costing a few KB of ROM and one lookup; and user-turn n-gram statistics folded into the hashed n-gram tables. If the model needs to know whose turn it is, that is one role token or one extra embedding row, not extra compute. Vocabulary: the 16K tokenizer should be trained on a mix that includes filtered user turns so that common typed forms (lower-case starts, contractions, "pls", "u") are single tokens.

## Verdict
Test. It is free, openly licensed, and the only large source of real typed user turns, but its distribution (long, task-oriented, multilingual) is far from simple-English chat, so it needs heavy filtering and may leave little. Settling experiment: run the filter above, count surviving turns and vocabulary coverage; then train two small models, one with synthetic user turns only and one with WildChat user turns mixed in, and compare keystroke savings / top-3 next-word accuracy on a held-out set of real short user turns (held-out WildChat plus LMSYS).

## Could not verify
- No median or histogram of user-turn length in the paper; the surviving count after a short-turn filter is unknown until we run it.
- The task brief expected a first-turn category comparison against LMSYS; Table 4 has WildChat only.
- Figure 3 coverage values are in a heatmap image I could not read.
- Did not read the ODC-BY legal text itself, nor the LMSYS-Chat-1M licence (I believe it is gated and more restrictive, from memory, unverified).
- WildChat-4.8M language and length statistics: not on its card.

## Links worth following
- https://huggingface.co/datasets/allenai/WildChat-1M (public, ODC-BY)
- https://huggingface.co/datasets/allenai/WildChat-4.8M and processing script https://github.com/da03/wildchat
- https://arxiv.org/abs/2409.03753 (WildVis, search tool for inspecting the data)
- https://arxiv.org/abs/2406.04770 (WildBench, built from reserved WildChat conversations)
