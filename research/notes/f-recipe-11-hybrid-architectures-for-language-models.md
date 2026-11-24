# Hybrid Architectures for Language Models: Systematic Analysis and Design Insights (https://arxiv.org/abs/2510.04800, 2025; v3 revised Apr 2026; Bae, Acun, Lin, Habeeb, Kim, Luo, Wang, Wu — FAIR at Meta / KAIST)

Read in primary form: arXiv HTML v3, main text and appendices up to part-way through Table 17. The page was read through an extraction tool, so numbers below are as extracted from the tables; figure-only values were not readable.

## What it is
A controlled comparison of Llama-style Transformer, Mamba-2, sliding-window attention (SWA) and two kinds of attention+Mamba hybrid, all trained on DCLM-Baseline at matched data or matched FLOPs. "Inter-layer" hybrids stack whole attention blocks and whole Mamba blocks; "intra-layer" hybrids run a half-size attention and a half-size Mamba in parallel inside one block. It ablates the attention:Mamba block ratio, where the attention blocks sit, and how the parallel branches are fused. Main scale is 1B (also 350M; a scaling sweep covers 100M-3B).

## Mechanism
- Inter-H: N blocks, ratio a:m (attention:Mamba) from {1:0, 1:1, 1:3, 1:5, 1:12, 0:1}. Attention blocks spread evenly through the middle, never at layer 1. 1B, 1:5 = 2 attention + 11 Mamba, attention at layers {4,8} of 13. 1:3 = layers {3,7,11}.
- Intra-H block: heads split in two groups (e.g. 16+16). Attention branch uses half-size query/key (Differential-Transformer style), full-size value; Mamba branch has reduced d_ssm. Each branch output is group-normalised, then fused. Best variant (Table 6): GroupNorm, no learned scalar, fusion by subtraction, two separate output projections. Concatenation is about as good. A learned gate was worse.
- Configs (Table 8): 100M = 8 attn layers, d 1024, ffn 3072, 16 heads / 4 KV heads, or 6 Mamba layers, d_ssm 2048, d_state 128, conv kernel 4. 350M = 14 layers d 1536, or 11 Mamba. 1B = 16 layers d 2048, or 13 Mamba. SWA baseline = 3 global + 13 windowed layers, window 512 + 64 sink tokens.
- Learned-routing check: a learned mixing weight alpha "favor[s] Mamba in early layers and the Transformer in the middle".

## Numbers
DCLM NLL / PG19 NLL / mean few-shot accuracy, 60B tokens, 8K context.
- Table 2, 1B: Llama 2.750 / 2.875 / 52.0; SWA 2.741 / 2.867 / 52.7; Mamba 2.758 / 2.891 / 52.3; Inter-H 1:5 2.735 / 2.861 / 53.3; Intra-H 2.728 / 2.853 / 54.7. KV cache at 8K: 256 / 63 / 13 / 43 / 38 MiB.
- Table 5, 1B ratio: 1:1 2.725; 1:3 2.733; 1:5 2.735; 1:12 (1 attention block) 2.741.
- Table 5, 350M: 1:0 2.882 / 3.015 / 48.7; 0:1 2.880 / 3.024 / 48.7; 1:1 2.850; 1:3 2.858 (acc 50.2); 1:5 2.860 / 2.999 / 49.4; 1:12 2.864 / 3.003 / 49.3.
- Table 17, 1B placement: 1:5 Front {1,8} 2.750 vs Middle {4,8} 2.735; 1:3 Front {1,7,11} 2.750, Middle {3,7,11} 2.733, End {3,7,13} 2.735; 1:1 Front 2.733 vs End 2.725. Moving one attention block to layer 1 erases the whole hybrid gain.
- Table 6 fusion (1B, all blocks hybrid): Add+scalar, no norm 2.740; GroupNorm+Add 2.721; GroupNorm+gate 2.751; GroupNorm+Conc 2.715; GroupNorm+Diff+2 outputs 2.712.
- Table 4, MoE (1 shared + top-1 of 8 routed experts): Llama 2.750 -> 2.656; Inter-H 2.735 -> 2.653; Intra-H 2.728 -> 2.648. Gains from routing are additive with the hybrid gain.
- Table 13, generation throughput at 2K -> 32K (units as in table): Llama 373 -> 30; Mamba 2528 -> 2917; Inter-H 969 -> 213; Intra-H 1325 -> 487.
- Needle retrieval (Fig. 4, text only): Transformer near zero beyond the 8K training length; hybrids hold to about 1.5x.

## Smallest scale tested, and whether the result held there
100M non-embedding parameters (Tables 14-15, matched FLOPs, DCLM NLL only). At 4e19 FLOPs: Llama 3.190, Inter-H 3.159, Intra-H 3.155, Mamba 3.134. At 2e19: Llama 3.237, Inter-H 3.192, Intra-H 3.186, Mamba 3.173. At 8e19: Llama 3.157, Inter-H 3.136, Intra-H 3.131 (no Mamba run). So hybrid > Transformer held at 100M, but pure Mamba was the best of all at 100M, whereas at 350M+ the hybrids lead (4e20: 2.874 / 2.855 / 2.842 / 2.838). Ratio and placement ablations were only run at 350M and 1B. 100M is still d=1024 and 20-50x Koko's active size.

## Training recipe and gotchas
AdamW, gradient clipping, trapezoid schedule (about 25% warmup, about 20% cooldown), batch 2M tokens, LR 6e-3 (100M), 3e-3 (350M), 6e-4 (1B); Llama 3.2 tokenizer (128K), tied embeddings. Mamba's A init changed to [0.01, 2.0] from [1, 16]. Branch normalisation is required for parallel fusion: without it the fused model is worse (2.740 vs 2.721). Gotcha: Appendix Table 11 (global-attention placement inside the SWA model) shows differences of only 0.001-0.002 NLL, i.e. noise; the strong placement evidence is the hybrid Table 17 (0.015). No seeds or error bars reported.

## Mapping onto the GBA design
- What transfers is the layout rule, not Mamba. Koko has 4-6 layers; the rule says layer 1 (and probably 2) should be a local/recurrent mixer, with ring-buffer attention in one or two middle-to-late layers (e.g. layers 3 and 5 of 6).
- RAM is the real win. Each attention layer needs its own 256-slot key/value ring buffer: 256 x 256-384 value bytes = 64-96 KB plus keys. Two attention layers already fill most of the 256 KB slow RAM; six cannot fit. A recurrent layer needs only O(d) state bytes.
- Compute: Koko's top-8 attention is already cheap (key scan is table lookups), so the cycle saving is small. A faithful Mamba-2 layer is not cheap: input-dependent decay and state update are true multiplies (about 7.5 cycles each) over d_ssm x d_state. Replace it with a diagonal gated recurrence (minGRU-like or fixed-decay EMA bank): about d multiplies (under 3K cycles) plus ternary projections.
- Routed-expert MLPs compose with hybrid layouts (Table 4), which supports Koko's 1-of-64 experts.
- GroupNorm before fusion would need an integer approximation (shift-based scale); nothing here is precomputable into ROM.

## Verdict
Test. The placement rule is cheap to adopt as a default (no attention in layer 1), but all evidence is at 100M+ with full-precision Mamba. Settling experiment: at Koko scale with ternary QAT, train 6-layer models with (a) attention in all layers, (b) attention at {3,5} plus recurrent mixers elsewhere, (c) attention at {1,4}, (d) no attention; compare validation NLL and a copy/recall probe at equal cycles.

## Could not verify
- Figure-only values: inter-layer 1:12 placement plots, intra-layer dimension-ratio and scatter/sandwich/cluster placement, needle accuracies, training-dynamics curves.
- Rest of Table 17 (1:5 End, 1:12, 350M rows) and later tables: page text was truncated.
- Weight-decay value; any limitations section; nothing on quantisation or sub-100M models was found in the text read.
- Table 15's 0.09B Inter-H row extracted as "5 attention, 1 Mamba", which contradicts the 1:5 ratio; likely 1 attention + 5 Mamba, not confirmed.
- No code release was found in the text read.

## Links worth following
- Mamba-2 (Dao & Gu): https://arxiv.org/abs/2405.21060
- Hymba (parallel attention+SSM heads in a small LM): https://arxiv.org/abs/2411.13676
- Differential Transformer (subtractive fusion): https://arxiv.org/abs/2410.05258
