# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs (https://arxiv.org/abs/2503.16870, ACL 2025, Anshumann, Zaidi, Kedia, Ahn, Kwon, Lee, Lee, Lee — Samsung Research)

Read in primary form (arXiv HTML incl. appendices). Some table columns were garbled in extraction; those are marked "(inferred)".

## What it is
A method for offline (cached-teacher) distillation of LLMs. It shows that caching the teacher's Top-K probabilities gives a biased target that makes the student over-confident, and can be worse than plain cross-entropy (CE) at small K. The fix: instead of the K largest probabilities, store N token ids *sampled* from the teacher distribution (with replacement) and train against the empirical histogram. This is unbiased, preserves the full-KD gradient in expectation, and needs about 12 stored entries per position instead of 100-300. Training speed is close to CE.

## Mechanism
- Per position, teacher distribution t over V (they use V = 100,000). Proposal q ∝ t^τ, τ = 1 in all main runs.
- Draw N = 50 ids with replacement (`torch.multinomial(probs, N, replacement=True)`); each draw gets weight (t_i/q_i)/N = 1/N at τ=1. Sparse target t^s_i = c_i / N (c_i = count). N = 50 gives ~12 unique tokens on average (App. C; unique count vs N is roughly a power law, Fig. 5).
- Loss: forward KL, sum_i t^s_i log(t^s_i / p_i), with p = student softmax over the FULL vocabulary; equals CE against the histogram. Gradient wrt logit x_i is p_i - t^s_i, and E[t^s_i] = t_i, so it equals the full-KD gradient p_i - t_i in expectation (App. A.6-A.7).
- Top-K instead gives gradient (sum_{j in K} t_j) * p_i - t_i (Eq. 2): an up-scaled target, tail always 0.
- Storage: 3 bytes per unique entry = 17-bit token id + 7-bit value. With N = 50 the value is the integer count c_i, so 7 bits is exact (App. D.1).
- Optional "+" variant (Sec. 6.3): CE mixed in at weight 0.1, and 2.0x learning rate on "hard" tokens.

## Numbers
All LLaMA-style models, Fineweb-edu, 3B teacher -> 300M student, 10B tokens unless noted.
- Top-K (Table 1): CE 2.80668; Top-12 2.86562 (worse than CE); Top-50 2.80378; Top-300 2.76044; FullKD 2.74698.
- Random Sampling (Table 6; unique-token column inferred): ~2.4 unique 2.76939; ~12 2.74621 (ECE 0.783, 0-shot 43.0); ~57 2.744. FullKD ECE 0.73; CE ECE 0.38, 0-shot 40.4.
- Gradient vs FullKD, one batch (Table 3): Top-12 angle 58.0 deg, norm ratio 2.43; Sampling-12 3.94 deg, 1.0027.
- Throughput relative to FullKD (Table 4, 8x H100): CE 2.9x, Sampling 2.6x at 300M (about 10% slower than CE).
- Storage for 100B tokens (Table 5): Top-300 90 TB; Sampling-12 3.6 TB (36 bytes/position); CE 0.3 TB.
- 100B tokens, 3B->300M (Table 7): CE 2.48029 / ECE 0.662 / 0-shot 45.0; Ours 2.47658 / 0.30 / 46.2; FullKD 2.48342 / 0.428 / 46.2.
- 8B->3B, 100B tokens (Table 8): CE 2.37093, 0-shot 55.6; Top-12 2.49963, ECE 4.66; Ours-12 2.35005, ECE 0.195, 57.5; FullKD 2.34446, 57.5.
- Temperature (Table 12): τ 0.8 / 1.0 / 1.2 give 2.74034 / 2.74621 / 2.73956; τ = 0 (uniform) diverges.
- Loss ablation (Table 15): forward KL 2.75; reverse KL 3.37; MSE 5.38.

## Smallest scale tested, and whether the result held there
300M parameters is the smallest with tabulated numbers; the method matched FullKD there. A 100M student appears only in the size sweep (Fig. 4, 8B teacher, 30x-model-size tokens; no numbers in text); the paper states the gain over CE *grows* with student size, so it shrinks towards our end. Nothing near 1-30M parameters, quantized students or 16K vocabularies. At 300M/100B tokens the loss gain is only 0.004 nats; the benefit is calibration and 0-shot (+1.2 points).

## Training recipe and gotchas
- Adam, betas 0.9/0.95, clip 1.0, no dropout; 300M: batch 1024 x seq 1024, LR 4e-4 -> 4e-5 cosine, 4% warmup (Table 17).
- Teacher and student input sequences must be exactly aligned (App. D.3): misalignment had "a surprisingly large effect".
- An unadapted teacher helped little (8B->300M: CE 2.99378, KD 2.9821, KD after teacher adaptation 2.95965; Table 13).

## Mapping onto the GBA design
Training-time only: zero cycles, RAM or ROM on the cartridge.
- Text that our generator samples at temperature 1 is already an N = 1 instance of this scheme; plain CE on it is unbiased KD with high variance. Logging 50 extra draws per position (from the *unmodified* τ=1 distribution, independent of whatever top-p/temperature was used to pick the emitted token) cuts the variance for the cost of ~12 x 3 bytes = 36 bytes/position: 300M tokens ~ 11 GB. With a 16K vocabulary an id fits 14 bits, so 14 + 7 bits still fits 3 bytes.
- The two-level head is compatible: class target = summed counts per class; member loss only inside classes with non-zero count, weighted by class count.
- Main obstacle: tokenizer mismatch. Counts are over the teacher's tokens; our 16K vocabulary differs. Either re-tokenize and score with the teacher by marginalising over multi-token continuations (expensive), or use an intermediate teacher trained on our vocabulary (a two-stage chain), which is the clean route.
- With V = 16K, dense FullKD is affordable if the teacher runs online, so the speed case is weaker than at V = 100K. The durable lesson: never cache renormalised Top-K.
- Better calibration (ECE) matters for predictive-typing suggestions and for any sampling done on-device.

## Verdict
TEST (cheap). If we distil at all from cached outputs, use sampled counts, not Top-K. Settling experiment: train the same small student three ways on a 50-100M-token slice (CE on generated text; sampled-50 counts; dense FullKD from an own-vocabulary teacher) and compare held-out loss and ECE. Expect a small gain given the student-size trend.

## Could not verify
- Table 6's rounds/unique-token column and Table 1's "~57 tokens" row label (garbled in HTML).
- Numbers behind Fig. 4 (100M and 1B students).
- The hard/easy percentile cutoff; no code repository found in the text.
- No evidence for sub-100M, ternary or small-vocabulary students.

## Links worth following
- Peng et al. and Raman et al. (prior sparse-KD baselines, Table 11; both lose to this method).
- Notes data-05 and data-06 in this folder (capacity gap, distillation scaling laws).
