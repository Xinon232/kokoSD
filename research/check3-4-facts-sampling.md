# Checks 3 and 4: fact recall and sampling (20 Nov 2026), model h_koko_L8, full output step, CPU

## Drilled facts (plainest wording, greedy reply, table value must appear)
| Kind | Score |
|---|---|
| Country capitals | 182/183 (the miss: "capital of Georgia" answered Atlanta, the US state) |
| Continents | 192/192 |
| Languages | 186/186 |
| Currencies | 169/169 |
| US state capitals | 50/50 |
| Calendar and planets | 27/27 |
| Times tables (0-12) | 80/80 |
| Addition (0-50) | 21/80 |
| Subtraction (0-60) | 13/80 |

Addition and subtraction answers are near misses ("34 plus 6 is 42"). Each sum appears about 15 times in
the drills against about 220 for each times-table entry. Remedy to test: compute sums in the game.

## Sampling (10 prompts x 5 settings, research/check4-samples.txt)
- Greedy is the most factual ("The Moon is a big rock in space that goes around Earth") but can loop
  (the joke prompt repeated one phrase until cut off).
- Temperature 0.5, top 40: no loops, chat and jokes fine, facts mostly hold.
- Temperature 0.7: chat fine, facts drift ("The Moon is a big, angry bird in the sky").
- Temperature 0.9: stories change subject, facts invented.
- Repetition penalty 1.15 changed little at 0.7.
- Undrilled general knowledge is weak at every setting ("who was george washington" -> "a baseball player").
Suggested default: temperature 0.5, top 40, plus a rule that blocks repeating the same phrase.
