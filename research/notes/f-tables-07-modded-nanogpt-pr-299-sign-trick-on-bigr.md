# modded-nanogpt PR #299 "Sign trick on bigram embed" and PR #201 "Bigram Hash Embedding"

https://github.com/KellerJordan/modded-nanogpt/pull/299 (trianxy, merged Jun 2026) and /pull/201 (ClassicLarry, merged Jan 2026). Repo owner: Keller Jordan.

## What it is
Two speedrun records for training a GPT-2-small-class model (768 wide, FineWeb, target val loss 3.28, 8xH100). PR 201 adds a hashed bigram embedding table summed into the residual stream at every layer. PR 299 multiplies each fetched row by a fixed pseudo-random +/-1 vector chosen by a second hash of the same bigram (Chen et al. 2015 hashing trick), and reshapes the table to more rows at lower width. Evidence is "same loss in less wall-clock", not perplexity at fixed compute.

## Mechanism
Read from the diffs (primary).

PR 201 row hash (CPU, in the dataloader), table = 5 x 50304 = 251,520 rows x 768, zero-init:
```
mod = rows - 1
idx[0]  = mod                                  # reserved row
idx[t]  = ((36313 * x[t]) ^ (27191 * x[t-1])) % mod   # int32 wraparound
x0_bigram = bigram_embed[idx]
x = x + x0_lambda[i]*x0 + bigram_lambda[i]*x0_bigram  # before every block i
```
`bigram_lambda` = one learned scalar per layer, init 0.1.

PR 299 changes:
```
rows = 15 * 50304 = 754,560 ; bigram_dim = 192
sign_table = sign(randn(8192, 192))            # fixed buffer, not trained
sign_idx[t] = (x[t-1] ^ x[t]) % 8192           # independent of row hash
x0_bigram = bigram_embed[idx] * sign_table[sign_idx]
x[..., :192] += bigram_lambda[i] * x0_bigram   # only the first 192 residual dims
```
No up-projection: the 192 dims are added straight into a slice of the 768-wide residual. Parameters drop from 193.2M to 144.9M (my arithmetic). Code comment: sign-table sizes "~500-15000 gave similar results". Author's stated intuition (via page fetch): (g_j*s_j)*s_j = g_j while cross terms s_i*s_j for colliding bigrams roughly cancel.

## Numbers
- PR 299 `statistics.md` (primary): baseline n=10, val 3.27937 (std 0.00135), 85,394 ms; PR n=20, val 3.27917 (std 0.00115), 84,430 ms. Delta -965 ms (1.1%) at equal loss, 1480 steps. Loss difference itself is far inside noise.
- PR 299 description (secondary, summarising fetch): about half the gain attributed to the sign trick, half to the reshape; ~28M unique bigrams in the data, so roughly 37 distinct bigrams per row at 754,560 rows (my division). Maintainer retest: merged as -1.3 s, later "-1.5 s".
- PR 201 README (primary): 6 runs, val 3.2776 (std 0.0012), 98.43 s versus 104.06 s retimed; bigram table credited -5.1 s (about 5%); steps 1765 -> 1600.
- PR 201 comment (secondary): HellaSwag 32.5%.

## Smallest scale tested, and whether the result held there
Only one scale: 768-wide, 50K vocab, 1480 training steps (token count not checked). Nothing smaller. The sign-trick gain is about 0.5 s of 85 s, real but tiny, and not separated from the reshape in the logged runs.

## Training recipe and gotchas
- Table optimiser (PR 201 diff): Adam betas (0.75, 0.95), lr multiplier 75, weight-decay multiplier 5, zero init.
- Negative results in PR 201 README: gating, trigrams and multiple hashes lost to plain addition "by a decent margin". PR 299 (secondary): trigrams reliably lowered loss but cost too much time; keeping only the top 100K/1M bigrams, private slots for the top 16K bigrams, and fewer rows all failed; Sylvester-Hadamard signs were "similar".
- Later reversal: current master (`track_1_short/ngram_table.py`, read in primary form, credited to record #360) uses one 84.6M-row x 768 table with a bigram half and a trigram half, each row signed from the 8192-row pool with independent hashes, summed. So trigrams do pay once the table is large, and width went back to 768.
- Maintainer's caveat (secondary): two colliding bigrams still share the same degrees of freedom; signs only make their interference look like noise.

## Mapping onto the GBA design
- Per token: 2 multiplies + xor + mask for the row index (use a power-of-two table, not `%`: no divide instruction); one sequential ROM row read; a second cheap hash for the sign.
- Sign costs zero multiplies: with ternary rows, sign flip keeps values ternary, so quantisation-aware training is unaffected. Store the sign pool as bits: 8192 x 192 bits = 197 KB of ROM, or generate bits from the hash with an xorshift. Apply as conditional negate while unpacking the row (one extra XOR per packed group if the unpack table is indexed with the sign bits).
- Slice injection means a 64-192-dim row touches only that many adds per layer, with no projection matrix: about d adds per layer; fold each layer's scalar into a per-layer shift or a 3-entry lookup rather than a multiply.
- ROM: 1M rows x 96 ternary dims at 1.6 bits is about 19 MB; "more rows, narrower" is exactly the trade ROM favours.
- RAM: one row buffer (under 200 bytes).

## Verdict
Adopt the PR 201 structure (already in the design) and the slice-injection, narrow-row layout; test the sign trick, since it is free at run time but its measured gain is about 0.5% of training time at one scale. Settling experiment: at fixed ROM bytes, a 2x2 of {sign, no sign} x {few wide rows, many narrow rows} on our corpus, 3 seeds, at a collision load matching ours, plus bigram+trigram halves as in master.

## Could not verify
- PR descriptions and comments were read through a summarising fetch (GitHub API blocked); only the diffs, README and statistics files were read verbatim.
- No ablation run logs isolating sign trick from reshape; the 0.5 s / 0.5 s split is the author's statement.
- Did not read PR #360 itself, only its code on master.
- No evidence at small width, small vocab, or with quantised tables.

## Links worth following
- https://github.com/KellerJordan/modded-nanogpt/pull/360 (84.6M-row bigram+trigram table)
- https://arxiv.org/abs/1504.04788 (Chen et al., hashing trick)
- https://arxiv.org/abs/1709.03933 (hash embeddings, cited by PR 201)
