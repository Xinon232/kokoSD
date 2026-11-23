# Fine-tune of Koko (20 Nov 2026)

Start: h_koko_L8 (2.188). Data: mix2 = mix1 with capitals corrected in 7,406 of Koko's training conversations
("koko" -> "Koko", "game boy advance" -> "Game Boy Advance", the user's name). Learning rate 5e-4 peak, cosine.
Coach = h_koko_L8_fp (2.068), loss (1 - w) * cross-entropy + w * T^2 * KL(coach || student).
Coached runs: batch 96 x 36,000 steps (shortened to fit the time available); control: batch 128 x 45,000.
Overall = koko .4 / chat .2 / facts .2 / stories .2, nats per token, lower is better. One seed each.

| Run | Overall | Koko | Chat | Facts | Stories | Hours |
|---|---|---|---|---|---|---|
| h_koko_L8 (before) | 2.188 | 2.105 | 1.620 | 3.505 | 1.607 | – |
| i_kd: coached, w 0.5, T 1 | 2.179 | 2.097 | 1.619 | 3.475 | 1.607 | 1.96 |
| i_plain: control, no coach | 2.184 | 2.099 | 1.616 | 3.501 | 1.604 | 2.06 |
| i_kd_lr1: coached, lr 1e-3 | 2.192 | 2.111 | 1.633 | 3.493 | 1.615 | 1.96 |
| i_kd_t2: coached, w 0.7, T 2 | 2.270 | 2.190 | 1.690 | 3.626 | 1.652 | 1.95 |

- The best run gains 0.009, which is the size of seed noise measured earlier (about 0.01). The fine-tune did not
  meaningfully change the score. The coached run saw 60% of the control's tokens and ended slightly ahead of it.
- Softer targets with more weight on the coach were clearly worse (+0.08).
- What did change: in 120 sampled identity replies the new model writes "Koko" 67 times and "koko" once, and
  "Game Boy" 46 times with no lowercase.
- i_kd is now the model on the cartridge. Router retrained (top token kept 99.2%, 1,390 rows on average).
  Integer-maths loss 1.9389 vs float 1.9393 (1,024 positions). All engine, game and browser tests pass.
- GPU cost: about 2.2 hours of a 4x RTX 5090 at $2.30/hr.
