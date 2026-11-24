# gbc-transformer: TinyStories-260K on a stock Game Boy Color
https://github.com/maddiedreese/gbc-transformer (May-Jul 2026, owner maddiedreese, MIT; 6 commits, 6 open perf PRs by Scottcjn).
Tags: [C] repo code/README read raw; [PR] PR text via a summarising fetch tool; [S] secondary; [I] my inference/arithmetic.

## What it is
A GBDK-2020 ROM (SDCC C, no assembly) that runs Karpathy's pretrained TinyStories-260K on the GBC's SM83 CPU: on-screen keyboard, on-device BPE, prefill, greedy generation [C]. Weights are post-training row-wise int8 in bank-switched ROM; activations are Q12 in int32; the KV cache sits in cartridge SRAM. No training, no QAT, no quality metric. It is the nearest prior art to Koko, about three orders of magnitude slower than our target.

## Mechanism
Model [C]: dim 64, hidden 172, 5 layers, 8 heads, 4 KV heads (kv_dim 32, 2x grouped-query), vocab 512, tied head, RoPE, RMSNorm, SwiGLU. 260,032 weights [I].

Quantizer [C]: per row, `scale = max|w|/127`, `q = round(w/scale)`, `scale_q16 = round(scale*65536)`; applied to every tensor, including norm gains and RoPE cos/sin. 282 KB in 8 KB banked chunks [I].

Forward (`src/gbc_q8_runtime.c`) [C]:
```
x = emb[tok]                          # q*scale >> 4 -> Q12 int32
per layer:
  rmsnorm: ss = mean((v*v)>>4)+1; inv = 2^22/isqrt(ss)
           out = ((x*inv)>>12 * gain)>>12
  matmul:  acc += (int32)w_i8 * clamp_i16(x_j)
           out = mul_scale_q16(acc, row_scale)   # 16 shift-adds
  RoPE on q,k; k,v -> SRAM (int32)
  head: score = (sum(q*k)>>12) * 1448 >> 12      # 1/sqrt(8)
        p = exp_neg(max-s), normalised by a divide
        exp_neg(x) = 2^24/(4096 + x + x2/2 + x3/6 + x4/24)
  SwiGLU: silu(w1 x) * (w3 x), sigmoid from exp_neg
argmax over 512 dot products with emb rows
```
RAM [I]: 2.7 KB of work vectors; KV cache 20,480 B of SRAM. Context capped at 16. Tokenizer: pair merge with a linear scan of 512 tokens per lookup [C].

## Numbers
- Real GBC, prompt `a`: 16 forward passes in "about 45 minutes", 0.0059 tok/s (2 min 49 s per token); output "res was a big, farmer. He was very happy" [C, README].
- Emulator (PyBoy and SameBoy agree): 87,000 frames for 3 tokens on main; 405,000 for 16 tokens after PR #2 [PR]. At 59.73 fps that is 424-485 s/token [I], 2.5-2.9x slower than the README. Unreconciled.
- Work per token: 259,328 weight MACs (45,312 per layer x 5 + 32,768 head, 12.6%) [I]; all-in 2,700 T-cycles per MAC (README) or 6,900 (emulator) [I].
- Open PRs, emulator only, tokens byte-identical [PR]: pointer walk instead of per-byte bank check 1.18x (#2); CGB double speed 2.01x (#3); software 32-bit multiply replaced by an 8x8 byte-split assembly MAC 3.4x, after which the MAC is "roughly half" of runtime (#4); helper strength reduction 5.2% (#6); the KV path is 2.2% of time (#7). Stack: 10.09x, about 498 -> 50 s/token [S, Hackaday].
- Range [PR #7]: keys peak at |k| = 110,944 in Q12 (27.1); an int16 KV cache overflows 260-283 times per 16-token run.
- Base model: validation loss 1.2968 after about 10 minutes on an A100 [model card].

## Smallest scale tested, and whether the result held there
One scale only: 260K parameters, 16-token context. Desktop int8 greedy text is coherent [C]; hardware text is degraded. No perplexity or FP32-vs-int8 gap is reported.

## Training recipe and gotchas
No training here; the base model's recipe is on its model card.
- SDCC `int` is 16 bits: `int8 * int16` wrapped and gave gibberish on hardware; fixed by a cast (PR #1). The Python simulator had not modelled it [PR].
- A 3-token golden check passed two broken builds that failed at positions 5 and 12 [PR #4, #7].
- `exp_neg` overflows int32 for x >= 20,643 (5.04) [I; confirmed in PR #6].
- The simulator is not bit-exact (truncating shifts, float sqrt, unbounded ints) [I].
- The port omits llama2.c's dummy-prefix space and the keyboard is lower-case only, so prompts are off-distribution [C, I].
- Prefill runs the output head and discards it [C].

## Mapping onto the GBA design
- Baseline [I]: this model on our kernels costs 259K x 7.5 = 1.9M cycles (int8, 8.6 tok/s) or 0.41M (ternary). Koko's 1.3-2M ternary weights per token is 5-8x its active compute with 200-400x its stored weights.
- Nothing architectural to adopt; its costs are C-on-SM83 artefacts.
- Replace every divide with ROM tables: exp for the top-8 softmax, reciprocal and inverse-sqrt for RMSNorm. Fold row scales into the next norm gain.
- GQA halves KV state. Check [I]: 6 layers x 256 positions x 256 B of 8-bit values = 393 KB, over the 256 KB slow RAM; kv_dim <= 128 or fewer cached layers is required.
- At 16K vocabulary the linear-scan tokenizer and a flat head (4.1M MACs at width 256) are impossible: ROM hash/trie plus the two-level head.
- Skip the head during prefill; prefill while the user types, which also yields suggestions.
- Ranges must be trained in (QAT with clipping): PTQ here needed 32-bit activations.

## Verdict
Skip as a method; keep as the baseline to cite and as a test checklist. Adopt a bit-exact integer reference simulator, golden-token gates covering a full ring-buffer wrap, and per-tensor range logging. Settling experiment: run stories260K through the GBA int8 and ternary kernels as a day-one smoke test; confirm cycles per MAC and greedy text against llama2.c.

## Could not verify
- Which speed is right: README hardware (169 s/token) or emulators (424-485 s/token).
- PR numbers are emulator-only and unmerged; read through a summarising tool, not raw diffs.
- Hackaday figures are secondary; its author states no port is verified on silicon.

## Links worth following
- https://hackaday.io/project/206401-transformers-on-retro-game-consoles (ternary vs int8 cycles per MAC on NES/SNES/Genesis/N64; a 6.36M-parameter ternary N64 model at 1.23-2.19 tok/s; a retracted top-K attention result)
- https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K
