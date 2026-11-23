# Check 1: fast output step applied after training (20 Nov 2026)

Koko (h_koko_L8) was trained scoring all 16,384 output rows. The cost model's 4.3 tok/s assumed 256 rows.
Measured on 12,000 validation positions: does a reduced set of rows still contain the token the full
model ranks first ("top-1 kept"), and how much of the sampling distribution (temperature 0.7, top 40)?

| Method | Rows scored | Top-1 kept | Sampling mass kept |
|---|---|---|---|
| All rows | 16,384 | 100% | 100% |
| 2,048 most frequent + k-means groups of rows (8 probed) | 2,824 | 98.0% | 97.7% |
| 256 most frequent + context tokens only | 301 | 78.8% | 75.6% |
| + previous token's 64 usual successors | 332 | 92.2% | 89.8% |
| + 2-bit router over 256 groups, 8 groups probed | 961 | 98.5% | 97.7% |
| same, 16 groups probed | 1,396 | 99.2% | 98.7% |
| same, 24 groups probed | 1,858 | 99.5% | 99.2% |

- Router: a 256-row 2-bit matrix trained on the frozen model's hidden states (240,000 training positions,
  under a minute on CPU) to predict where the model's own probability mass in the rare tokens sits.
  Groups are k-means clusters of output rows compared on sampled hidden states.
- Successor lists cost ROM: 64 per token = 2.0 MB (32 per token = 1.0 MB, about 0.3 points worse).
- At 1,396 rows the cost model gives about 3.8 tok/s (body 3.76M cycles + 0.65M for the output step).
- Not yet measured: effect on reply quality by reading, and the engine's real cycle counts.
Scripts: study/head_study.py, head_study2.py, root_study.py, root_study_tern.py, collect.py, bigram_lists.py.
