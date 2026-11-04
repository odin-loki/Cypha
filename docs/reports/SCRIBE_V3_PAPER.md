# SCRIBE v3 paper: the meta-evaluation re-run with v3 and v3.1

**Odin Loch** · 2026-11-04

**Odin:** "Do a Scribe v3 paper to update the scribe v2 paper on Zenodo. Go through everything, the challenges we had and what measures what and why v3 is better."

The SCRIBE v2 paper (`paper/scribe/SCRIBE_paper.md`, version 1.0 on Zenodo) evaluated v2 on control ladders, held-out corruptions, model ladders, human judgments, attacks, stability and cost. v3 ([SCRIBE_V3.md](SCRIBE_V3.md)) and v3.1 ([SCRIBE_V31.md](SCRIBE_V31.md)) were tested only on Wikipedia ladders, attack batteries and K5's samples. This report re-runs every part of the v2 paper's meta-evaluation that can be re-run, with v3 and v3.1 next to v2, for the paper's version 2.0.

**Everything below the line "Registered" was written and committed before any of these reads was run.** Results are added under "Results", misses included.

## Ground rules

- **The metric is frozen.** Profiles v1, v2, v3 and v31 are not changed. If a defect turns up, it is reported. It is fixed only with a byte-for-byte check of the other profiles' outputs, and the paper says so.
- **No decision is made on these reads.** They are report-only. No setting of any profile is chosen on them.
- **Fresh regions already used are not used for any new decision:** enwik8 [72, 95.37) MiB (v2's and v3's fresh region) and enwik9 [100, 124) MiB (v3.1's). Some of the re-runs below read texts from enwik8 [72, 95.37) MiB, as the v2 paper did. That is a report, not a decision. Note that v3's results on that region's attack battery motivated v3.1's design, so v3.1 is not blind to that region in the strict sense: its attack types were designed after v3's read of it. The corruption families, n-gram texts and v2's attack set were never used in any design.
- **One run, three metrics.** Every score file is written by `scribe … --profile v31`, which also writes v2's and v3's fields. So v2, v3 and v3.1 are always read on the same texts.

## Setup (this container, 2026-11-04)

- Build: `cmake -S native -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --target scribe`. `ppmi_svd` has no CMake target; it is built as the v2 code README says, `g++ -O2 -std=c++20 native/tools/semantics/ppmi_svd.cpp -o ppmi_svd`. `scribe selftest`: 119 checks, 0 failed.
- Data: enwik8 and enwik9 from mattmahoney.net (enwik9's first 10⁸ bytes equal enwik8, checked with `cmp`). Every other dataset from `scripts/scribe_eval/data.py all` (manifest with SHA-256 in the data root).
- Word vectors: vec32 from enwik8 [16, 48) MiB (35 s, 467 MB). The web, books, WritingPrompts, ROC and Dolly packs each get their own vectors, wrapped as `<page><revision><text>…` as the v2 code README says.

**Reproduction checks (made before registering; v2 profile only, or numbers already published).**

| Check | Published | Here | |
|---|---|---|---|
| dev A ladder, `--profile v31` (enwik8 [56, 64), seed 20261027, `--chain-rng`): first pair v2 / SURPRISE / MATCH / v3 / v3.1 | 0.974 / 0.952 / 0.953 / 0.982 / 0.959 | the same | exact |
| v2 fresh ladder (enwik8 [72, 95.37), seed 1123581321), `--profile v2` | 0.975, 0.999, 0.998, 0.998, 1.000; human 0.916 | the same | exact |
| HANNA, v2, story Kendall with RE, CH, EM, SU, EG, CX (no human stories) | 0.014, 0.194, 0.124, 0.143, 0.168, 0.228 | 0.017, 0.193, 0.127, 0.144, 0.170, 0.227 | close |
| Dolly ladder, v2p (`raw:` pack and reference): correct > other / echo / shuffled | 0.961 / 0.957 / 0.863 | 0.961 / 0.957 / 0.864 | close |
| web ladder, v2: human mean, first pair | 0.923, 0.966 | seeds 1123581321, 20261028, 20261027: 0.908–0.919, 0.955–0.968 | not the same texts |
| GPT-2 output dataset, v2, 1,000 `scribe samples` per system (seed 1123581321) | human 0.925, xl 0.826, … top-k 0.968–0.972 | human 0.911, xl 0.837, … top-k 0.958–0.976 | not the same texts |

- Runs on enwik ranges with a recorded seed reproduce exactly.
- The sets from other genres give numbers within a few thousandths. Their word vectors are rebuilt here, so they differ slightly.
- The v2 study did not record the seeds of its web ladder, books ladder, GPT-2 samples, corruptions, n-gram prompts and attack set. Those texts cannot be rebuilt byte for byte. **They are drawn again here with one registered seed, 20261104, and v2 is scored on the new draws next to v3 and v3.1.** The v2 paper's numbers are quoted next to them.

## Registered

### The reads

All commands are in `scripts/scribe_eval/runs/v3_paper.sh` (committed with this registration), one step per read. Wikipedia pack enwik8 [16, 48) MiB, reference [48, 56) MiB, vectors vec32 throughout, unless a read says otherwise.

| # | Step | What | Texts |
|---|---|---|---|
| R1 | `fresh` | **A new fresh read** of the Wikipedia ladder, the 9 registered attacks and the 8 held-out attacks | enwik9 [900, 924) MiB, never read by any SCRIBE (the v2 plan's spare region starts at 900), `--dedup` (it drops 1 of 4,115 documents, counted with `scribe dedup`, which scores nothing), seed 20261104, 200 prompt pairs |
| R2 | `genre` | Web and books ladders with their own pack and reference; web text against the Wikipedia pack and reference | WebText test; books fresh split; seed 20261104 |
| R3 | `gpt2` | The GPT-2 output dataset | 1,000 `scribe samples` per system, seed 20261104 |
| R4 | `dose` | The nine held-out corruption families × 6 strengths | 200 human samples of enwik8 [72, 95.37) MiB, seed 20261104; `corrupt.py` with its default seed (7) |
| R5 | `human` | HANNA (1,056 stories; WritingPrompts pack) | open (prompt removed) and with the prompt; `--window` |
| R6 | `human` | The MAUVE human evaluation (WebText pack) | open and with the prompt; `--window` |
| R7 | `human` | OpenMEVA MANS ROC and WP | open and with the prompt (ROC: `--ref-whole`) |
| R8 | `models` | KN word n-grams, orders 1–5, trained on enwik8 [8, 16) MiB | 250 prompts of enwik8 [72, 95.37) MiB, seed 20261104; scored open |
| R9 | `qa` | The Dolly question-answering ladder (`qa_ladder.py`, its default seed) | scored with the prompt |
| R10 | `attacks` | v2's attack set (`attacks.py`) | 250 prompts of enwik8 [72, 95.37) MiB, seed 20261104; scored with the prompt |
| R11 | `stability` | Five seeds; reference sizes 50–1,000 (3 draws each); packs of 8–128 MiB of enwik9 from 128 MiB (vectors rebuilt each time); length | the v2 paper's commands, profile v31 |
| R12 | `k5` | K5's saved GEN-A dev continuations (`bench/k5_scale/…/reads/`): E8, N7, N30, N100, N300, NALL (old K5) and U8, C1 (upgraded K5), every saved seed | the GEN-A dev ladder (enwik8 [56, 64), seed 20261027, `--gen-trie`) |
| R13 | `cost` | Time and peak memory of `scribe score`, profiles v2, v3 and v31 | the R11 base ladder's texts (Wikipedia pack) and the R3 samples (WebText pack) |

**Statistics.** `scripts/scribe_eval/meta.py`, as in the v2 paper: 1,000 bootstrap resamples, seed 20261028; a ladder pair passes with accuracy ≥ 0.95 and disjoint 95% intervals; a corruption family is followed when Spearman over the six strength means is ≤ −0.9; HANNA and OpenMEVA by Kendall tau-b (story level, machine stories only, and system level); the MAUVE human evaluation by per-pair agreement and system Spearman against Bradley–Terry scores, with the 25-second answer-time filter. Metrics: v2 (`scribe_v2`), v3 (`scribe_v3`), v3.1 (`scribe_v31`), and the judges SURPRISE, MATCH, SURPRISE 3.1, MATCH 3.1 and `min3`. v3.1 is the headline.

**The combination ablation (R1, report only).** From R1's exported scores, v3.1 with one judge left out (the geometric mean of the other two), the plain mean of v3.1's three judges, and the minimum of the three, on the ladder and on all 17 attacks.

### Predictions

Each is scored hit or miss under "Results". Probabilities are mine.

| # | Prediction |
|---|---|
| Q1 | **R1 ladder:** v2, v3 and v3.1 each pass 5 of 5 pairs. v3.1's first pair (human > sentences shuffled) is the thinnest of the three, 0.95–0.98 (about 70% that v3.1 passes all five). MATCH 3.1 alone fails at least one pair. |
| Q2 | **R1 registered attacks:** v3.1 passes at least 8 of the 9 at 0.95 (all 9: about 50%). v3's own rule passes 4–6 of 9. |
| Q3 | **R1 held-out attacks:** v3.1 passes 1–3 of 8. `ho_echo_scatter` passes; `ho_stuff7_rare`, `ho_title_swap`, `ho_list_related` and `ho_template` fail. |
| Q4 | **R1 human mean** under v3.1 is within 0.05 of the earlier fresh reads (0.904): in [0.854, 0.954]. |
| Q5 | **R1 ablation:** dropping v2 from v3.1 fails at least one ladder pair. Dropping MATCH 3.1 loses `echo_copy` and `ho_echo_scatter` (< 0.95). The plain mean of the v3.1 judges passes fewer registered attacks than the geometric mean. |
| Q6 | **R2 web:** v2 passes 5 of 5 (as in the v2 paper). v3 passes 5 of 5. v3.1 passes 4 or 5; its first pair is its weakest. v3.1's human mean on web text is 0.85–0.92, lower than v2's: web text has more lists and boilerplate, which the F and B groups charge. |
| Q7 | **R2 books:** v2 fails "human > sentences shuffled" again (below 0.9). v3 and v3.1 do better on that pair than v2: sentences from other novels bring other names, and the new judges' cohesion counts repeated rare words and strings. At least one of v3 and v3.1 passes 4 of 5 (about 50%). |
| Q8 | **R2 web against the Wikipedia reference:** v3.1 keeps less of its human score than v2 does (v2 kept 0.89 in the v2 paper). |
| Q9 | **R3 GPT-2, top-k 40 (v2's "too typical" blind spot):** v2 again rates every top-k system above WebText. SURPRISE and SURPRISE 3.1 rate WebText above every top-k system. v3.1 puts WebText first (above all four top-k systems), and its Spearman with the expected order over the five systems is ≥ 0.6. v3 puts WebText above at least 3 of the 4. |
| Q10 | **R3 GPT-2, pure sampling:** v3.1's Spearman with the expected order is ≥ 0.8; medium may still fall below small, as for v2. |
| Q11 | **R4 corruptions:** v3 and v3.1 each follow 7 or 8 of the 9 families (Spearman of the means ≤ −0.9). Sentence reordering is not followed by any of the three. Truncated sentences: v3.1's word fit sees the sentence end, so v3.1 falls more with truncation than v2 does (Spearman of the means more negative than v2's), but it does not reach −0.9. |
| Q12 | **R5 HANNA:** v3.1's story-level Kendall with coherence (machine stories) is 0.17–0.26, and at least one of v3 and v3.1 is above v2's re-run value. System-level Kendall for coherence ≥ 0.5. With the prompt, relevance (RE) rises above the open profile's and above 0.05. |
| Q13 | **R6 MAUVE human evaluation:** per-pair agreement of v3.1 is 0.51–0.57 on each question. Its system Spearman is ≥ 0.6 on q3 and is not above v2's on at least 2 of the 3 questions. |
| Q14 | **R7 OpenMEVA:** v3.1's story Pearson is 0.08–0.25 on ROC and below 0.15 on WP. |
| Q15 | **R8 n-grams:** every n-gram order scores below human text under v3.1 (adjacent accuracy human > 5-gram ≥ 0.95). v3 orders orders 3–5 better than v2 (higher adjacent accuracies), because SURPRISE and MATCH do not saturate as hard; v3.1 ties them like v2. Spearman over the six systems ≥ 0.9 for v3. |
| Q16 | **R9 Dolly (with the prompt):** v3.1 ranks the correct answer above the echoed question and above the shuffled answer at ≥ 0.95, and above another question's answer at 0.80–0.95 (below v2p's 0.961). |
| Q17 | **R10 v2's attacks:** v3.1 ranks human text above copied pack text, the echoed prompt, keyword stuffing (every third word), the function-word loop and the generic paragraph at ≥ 0.95. The generic paragraph with the prompt's words fails (< 0.95). Random numbers and inserted negations stay at 0.45–0.55 for all three metrics. |
| Q18 | **R11 seeds:** v3.1 passes 5 of 5 on at least 4 of the 5 seeds. **Reference size:** v3.1 passes 5 of 5 in every draw at 500 and 1,000 samples and fails at least one pair in every draw at 50. **Pack size:** v3.1 passes 5 of 5 at 16, 32, 64 and 128 MiB. **Length:** \|Pearson\| of v3.1 with bytes ≤ 0.15 in each bucket. |
| Q19 | **R12 K5:** the published gate figures reproduce (old K5 E8 → upgraded U8, seed 20261101: v3.1 0.435 → 0.483). Old K5 from 7 MB to 330 MB (E8 → NALL, 4-seed means): v3.1 changes by less than v2 (+0.046); between 0 and +0.03. Upgraded K5 (U8 → C1, 4 seeds): v3.1 changes by −0.03 to 0. |
| Q20 | **R13 cost:** v3.1 scores 350–550 texts a second on one core with the Wikipedia pack, under 600 MB, and is 2–4 times slower than v2. With the larger web pack it is slower (200–450 texts a second) and under 1 GB. |

## Results

(To be added after the runs.)
