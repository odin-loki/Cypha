# T1b: tuned, but not pinned (GENERATION_PLAN §13.43)

2026-11-12 · Odin Loch

**Odin:** "After you have tuned the experts set them to that but dont pin them to it. Set them an optimal amount back and let the model teter to and fro from optimal to noise so that it learns and we get beneficial stochastics."

Branch `wip/tune-adapt`, from T1's branch `wip/tune-predict`. New header `native/include/cypha/online/hyper_adapt.hpp`; hooks in the mixers (`mix_core.hpp`, `full_mixer.hpp`, `meta_mixer.hpp`), the n-gram (`online_ngram.hpp`), the similar-context expert (`similar_context.hpp`, a temperature setter) and `context8k` (`--hyper-adapt`). Off by default: without the flag nothing is made and every user reads its fixed value.

## The method as built (registered 2026-11-12, before any run)

§13.43's method, arms, rule and predictions stand. These are the details it left open.

**Which settings move.** The continuous settings T1 tuned or searched:
- learning rates, by **hypergradient**: `lr1`, `lr2` (every bitwise mixer that reads them, times its rate scale), `lrb` (the geometric mixers), `m1lr1`, `m1lr2` (M1 and its rate copies), `--ngram-lr`;
- the others, by **one-sided finite difference**: `apm` (the SSE rate), `m2eta`, `m2share`, `m2geta`, `m2gshare` (the masters' exponents and shares), `--ngram-cap` (rounded to a whole count), the similar-context temperature.
- Fixed: table sizes, orders, windows, `eps` and `floor` (clamps, not rates), `decay`, M3's settings (not in the headline), and the rates with no flag (the LSTM's, the chain's).

**How each moves** (in log units, x = log θ; one step a byte):
- Start: θ\*'s log moved a fraction *b* toward the old default's.
- Noise (A2, A3): x += κ (c − x) + σ √(κ(2 − κ)) N(0, 1), so x wobbles about c by σ. κ = 1/1024 a byte. The noise never anneals.
- Centre (A1, A3): every block of 1,024 bytes, c −= β g / rms(g), rms over past blocks (decay 0.99). The centre stays within θ\* ×/÷ 8. A2's centre is θ\*, fixed.
- Hypergradient g: the one-step truncated hypergradient. Each weight set keeps the gradient of its last update; a step adds −rate (g_t · g_last) to d loss / d log θ. Each mixer uses its own loss, as it learns from it.
- Finite difference g: the settings take turns. A base block at c, then a probe block at c + 0.2; g = (L_probe − L_base) / 0.2, L the mean bits a byte of the block.
- Whose loss: each thread steps its own settings with its own loss, so a run is fixed by its seed. The mixers' settings use the headline arm's bits (M1Gs); the n-gram's its own bits; the temperature the similar-context expert's own bits.
- σ per setting, from T1's sensitivity table (`bench/tune/t1_sensitivity.tsv`): q = the mean of Δbpc(×0.7) / ln(0.7)² and Δbpc(×1.4) / ln(1.4)²; σ = √(0.0002 / q), clipped to [0.05, 0.5]; σ = 0.5 when q ≤ 0. So each setting's wobble alone costs about 0.0002 bits per byte on the static curve.

**Choices on the 1 MB tuning slice** (small runs, seed 1; T1's tuned set as the base):
1. β from {0.003, 0.01, 0.03}, with A1 at b = 0.25;
2. then b from {0, 0.1, 0.25, 0.5}, with A3 at that β.
The same β and b serve A1, A2 and A3.

**Exactness first.** Without `--hyper-adapt`, the new binary's stdout must `cmp` equal to T1's binary's, on the small slice with T1's tuned set. If not, T1b stops to report.

**Full runs** (full default `context8k` with T1's tuned set, one turn of `~/.cypha_full.lock` each, 4 threads):
- A0: no `--hyper-adapt`. Its stdout must also `cmp` equal to T1's tuned run.
- A1: `mode=adapt`. No randomness, so one run: the seeds would give the same output. Its spread is 0.
- A2 (`mode=noise`) and A3 (`mode=both`): seeds 1, 2 and 3.
- Order: A0, A1, then A2 and A3 alternating by seed.
- Time: CPU seconds (user + system, `/usr/bin/time`), less touched by the shared PC's load than wall clock; wall clock beside it.
- `qa` never passes the flag; its dev rows must `cmp` equal with the new binary.

**Rule** (§13.43, unchanged): an arm is kept if its mean `context8k_bpc` is below A0's by more than its seeds' spread, and time rises at most 10%. A3 must also beat A1 by the same margin for the noise to count. Answers must not get worse.
