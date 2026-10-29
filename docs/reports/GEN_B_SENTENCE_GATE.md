# GEN-B: a sentence gate for open-ended generation

**Odin Loch** · 2026-10-29 · generation track step 2 · section "GEN-B" of the track ([`GEN_TRACK.md`](GEN_TRACK.md))

(The design, the reads, the rules and the predictions below were written and committed before any run of the gate. Results are added under them.)

**Outcome (2026-10-29).** The gate keeps the best of 4 CyphaLM sentences by CyphaLM's own fluency minus a markup penalty (`--gate 4,wm=5`, chosen on dev). On GEN-A's frozen test split it raises SCRIBE v2 from 0.072 [0.059, 0.085] to 0.090 [0.076, 0.106], at every temperature. The intervals overlap, so by the registered rule it is **not adopted**, and `--gate` stays a flag. It improves sentence form (D_S −8%), surface S (+0.015, disjoint) and word fluency outside SCRIBE (+0.11 bits a word). It does not keep the subject: coherence distance rises 10%. The copy share rises 0.017, inside its 0.02 bound. It costs 4.1 times the CPU per byte.

**Question.** CyphaLM's samples drift. A few words after the prompt they change subject, and their sentences break. Can a gate that only chooses help? CyphaLM writes K candidates for the next sentence. The gate keeps one. It never writes a byte itself.

**Rule 10.** PLAN rule 10 lets a tool gate. Every candidate is a sample from CyphaLM's own mixture with `sample`'s own sampler. The gate's features come from CyphaLM too. No string search, retrieval or hand-written text is involved. The gate is in `sample` only, behind a flag; `qa` and chat answers do not use it.

## The gate (`native/tools/cyphalm_trie.cpp`)

- **Flag.** `cyphalm_trie sample … --gate K[,wf=..][,wp=..][,wm=..][,min=..][,max=..]`. Without `--gate`, `sample` is unchanged.
- **Candidates.** For each sentence, K candidates. Each is drawn from today's mixture (trie, then similar-context, then copy) with `sample`'s sampler (11-byte no-repeat ban, temperature, nucleus 0.9). Candidate k of sentence j has its own random stream, so no candidate depends on another.
- **Where a candidate ends.** After a line break; after ". ", "! " or "? " once it has `min` bytes (default 24); at `max` bytes (default 240); or at the byte budget. So the text is exactly as long as without the gate.
- **Features.** All three are CyphaLM's own:
  - **fluency:** the mean log2 probability of the candidate's bytes, under the mixture that wrote them (before temperature and the ban);
  - **pmi:** fluency minus the same mean with the prompt replaced by a line break. It measures what the prompt adds, so a candidate that stays with the prompt's subject scores higher;
  - **markup:** the share of the candidate's bytes that are wiki markup: `[]{}|=<>&*#`, and the second quote of each `''`.
- **Score.** wf × fluency + wp × pmi − wm × markup. The highest score is kept; ties keep the lower k. Defaults: wf 1, wp 0, wm 0.
- **Cost.** About K times `sample`'s cost per byte, and twice that again when wp is not 0 (pmi reads each candidate a second time).
- **Code.** `GateSpec` and `parse_gate` (the flag), `gate_candidate` (one candidate and its features), `sample_gated` (the loop). `sample_with_class`'s step is split into `mix_step` and `pick_byte`, which the gate shares; `sample_with_class` does the same as before. (Since the merge with main, `mix_step` is main's `sample_mix`; see amendment 1.)

## Independence from SCRIBE (the gaming risk)

SCRIBE v2 (`native/include/cypha/eval/scribe.hpp`) scores text from word statistics: a lexicon and a Kneser-Ney word trigram from enwik8 [16, 48) MiB, word vectors (`/root/work/scribe/vec32`), sentence and bracket checks, repetition and verbatim 8-gram copies. The gate reads none of these. It does not tokenize words, call SCRIBE, or read SCRIBE's vectors. Its fluency is CyphaLM's byte probability, trained on [0, 7) MiB. Its pmi compares two CyphaLM contexts. Its markup share counts bytes before any cleaning.

A gate can still please SCRIBE without making better text. So the reads below include judges the gate was not tuned on: SCRIBE's axes one by one, surface S, a word fluency that is not SCRIBE's, and two copy shares. GEN-C found that keeping the topic broke sentence form (`docs/reports/GEN_C_TOPIC_CACHE.md` on main). So D_S, SCRIBE's sentence-form distance, is reported for every arm.

## The reads (fixed before any run)

1. **Parity.** Without `--gate` and with A4d's flags (`--sim default,online=1,probe=128`, today's default), `sample` on the first 20 dev prompts gives the same bytes as A4d's dev file (`/root/work/a4d/runs/SO_dev.txt`) up to that point. The same holds with the model served from a saved file (`--save` then `--model`). If it does, every arm uses the saved model.
2. **Dev arms.** `sample` file 0, its first 100 prompts (`--prompts 100`), T 0.6, 0.8 and 1.0, 600 bytes each: 300 samples an arm. Prompt ids and seeds are those of A4d's dev file, so the baseline G0 is that file's first 100 prompts.

   | Arm | Gate |
   |---|---|
   | G0 | none (A4d's dev file, first 100 prompts) |
   | G1 | `--gate 4` (fluency) |
   | G2 | `--gate 4,wm=5` (fluency, markup) |
   | G3 | `--gate 4,wp=1,wm=5` (fluency, pmi, markup) |
   | G4 | `--gate 8,…` with the weights of the best of G1–G3 by dev SCRIBE |

3. **SCRIBE.** A4d's ladder command on each file: `scribe ladder --pack enwik:enwik8:16:48 --ref enwik:enwik8:48:56 --test enwik:enwik8:56:64 --vectors /root/work/scribe/vec32 --profile v2 --seed 20261027`. Reported: SCRIBE v2 pooled with its 95% bootstrap interval, the three temperatures, surface S, the distances D_L, D_S, D_C, D_I, and the share of samples with D_I > 0 (SCRIBE's copy axis; GEN-C's copy share).
4. **Judges outside SCRIBE** (`/root/work/genb/runs/judges.py`, sha256 prefix `048fec84f190ddf8`). Over all 300 samples of each arm, with 95% bootstrap intervals and the paired difference against G0 (same prompt and temperature):
   - **word fluency:** mean log2 probability per word under a Witten-Bell word bigram trained on enwik8 [64, 72) MiB. That text is outside the trie's and SCRIBE's. Words seen once are `<unk>`; the `<unk>` share is reported too;
   - **copy8:** the share of words inside a word 8-gram that occurs verbatim in the trie's training text [0, 7) MiB;
   - **markup:** the gate's own feature, to show what it did.
5. **Cost.** Microseconds per generated byte, as `sample` prints it, for each arm. Wall time on a shared machine, so it is also given as a ratio to G0's time in the same conditions (the parity run).
6. **Full dev read.** The chosen arm (rule 1) runs on all 250 dev prompts. Its SCRIBE is reported beside A4d's 0.063 [0.051, 0.078]. No choice is made on it.
7. **Test read, once.** The chosen arm against the ungated default on GEN-A's frozen benchmark (`scripts/gen_eval.sh`, `/root/work/gena/HARNESS_READY`), if it is ready when the dev choice is made. If not, dev results are reported and the test read is left to the integrator, under rule 2.

## Rules (fixed before any run)

1. **Dev choice.** Among G1–G4, the arm with the highest pooled dev SCRIBE, if it is above G0's on the same 100 prompts. It must also hold three bounds against G0: the share with D_I > 0 rises by at most 0.02; copy8 rises by at most 0.02; surface S falls by at most 0.01. If G4 (K 8) is within 0.005 SCRIBE of the K 4 arm with the same weights, K 4 is chosen (half the cost). If no arm qualifies, nothing is adopted and there is no test read.
2. **Adoption as `sample`'s default.** On the test read, all three must hold:
   - SCRIBE rises over the ungated default, with disjoint 95% intervals;
   - neither copy share (D_I > 0, copy8) rises by more than 0.02;
   - surface S does not fall (its mean is not below the default's).

   Then the chosen spec becomes `sample`'s default gate. Otherwise `--gate` stays off by default.
3. **Temperature.** `sample` writes T 0.6, 0.8 and 1.0, and the rules use the pooled score. The temperature where the gated text scores best is reported as the one to use for generation. It is not a rule.
4. Nothing is changed after the test read.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| B1 | Read 1: without `--gate`, the first 20 dev prompts are byte-identical to A4d's file, both trained and served from a saved model |
| B2 | G1 raises dev SCRIBE over G0 by +0.005 to +0.04; the intervals overlap |
| B3 | G1 lowers D_S (sentence form) by at least 5% against G0 |
| B4 | G2 against G1: the kept markup share falls by at least 30%, and SCRIBE does not fall |
| B5 | G3 against G2: D_C (coherence) falls by at least 5%; SCRIBE within ±0.01 |
| B6 | K 8 (G4) adds less over its K 4 arm than that arm adds over G0 |
| B7 | The gate's gain over G0 is largest at T 1.0 and smallest at T 0.6 |
| B8 | Every arm: the share with D_I > 0 rises by less than 0.02. G1's copy8 rises (fluency favours the copy expert), but by less than 0.02 |
| B9 | G1 raises surface S by at least 0.005 |
| B10 | G1 raises the bigram word fluency, with the paired interval above 0 |
| B11 | Cost per byte: K times G0's for wp 0, 2K times for wp not 0, each within ±30% |
| B12 | The test read, if made, does not meet rule 2 (the intervals overlap) |

## Amendment 1 (registered 2026-10-29, while G1 ran, before any gated result was seen)

**What changed.** GEN-A's harness became ready at 12:12 UTC, after this report was registered (5362610). Main (e50dd4e) now has `scripts/gen_eval.sh` and `sample --prompts-file`. It is merged into this branch (3c76cf4). The merge takes main's `sample_mix` in place of `mix_step` (the same body). `sample_with_class` draws through `pick_byte`, the same sampler moved into a function. `--gate` refuses `--doc`, because the gate's candidates are drawn without GEN-C's document experts.

**The reads now.**

- **Dev choices stay on A4d's dev prompts**, as registered. G1 already runs there, so all arms stay comparable.
- **Read 6 (all 250 A4d dev prompts) is dropped.** The machine runs at load 25–30 on 4 cores, and each run costs 1–3 hours. The test read gives the full-size number.
- **Read 7, the test read,** runs once if rule 1 chooses an arm. It is `scripts/gen_eval.sh test` with the merged binary, twice:
  - **T0:** today's default, no `--gate`;
  - **T1:** the chosen arm's `--gate` spec.

  GEN_TRACK's Reads table is empty, so T0 is the first test read of the default. Both rows go in the table.
- **Rule 2's copy shares.** The harness also prints `copy32`, the share of generated bytes in 32-byte spans of the training bytes. Rule 2 now needs all three copy shares to rise by at most 0.02: D_I > 0, copy8 (`judges.py` on the harness's files), and pooled copy32. This only adds a bound.
- **Merge parity.** The merged binary must write the same gated text as the binary the dev arms use.

| # | Prediction |
|---|---|
| B13 | With `--gate 4`, the merged binary's first 4 A4d dev prompts are byte-identical to G1's |

## Amendment 2 (registered 2026-10-29 at 16:20 UTC, after the dev choice, before the test read was seen)

Main's Reads table now holds a test read of today's default: B0 (Tokens T4), reused by T4b and GEN-E (E0). B0 and E0 are byte-identical; they came from two runs. So rule 2 compares T1 with B0. T0 still runs, after T1, for two reasons:

- **Parity on the test split.** T0 must be byte-identical to B0.
- **Cost.** T0 and T1 print CPU time per generated byte (b49ff0b), which the machine's load does not change much.

## Results

Tool: `cyphalm_trie` at 5362610 for the dev arms (copied to `/root/work/genb/bin`); the merged binary (3c76cf4, then b49ff0b) for the merge parity and the test read. Every arm runs today's default experts (`--sim default,online=1,probe=128`, then copy), served from one saved model (`--save`, then `--model`). The machine ran 12 to 25 other agents on 4 cores, and its load moved between 3 and 30.

### Read 1: parity

- **Trained (P0):** without `--gate`, the first 20 dev prompts (81 lines, MODEL line included) are byte-identical to A4d's dev file.
- **Served from the saved model (P1):** byte-identical to P0.
- **Merge parity (B13):** with `--gate 4`, the merged binary's first 4 dev prompts are byte-identical to G1's.

### Reads 2–4: dev arms

A4d's first 100 dev prompts × T 0.6, 0.8, 1.0 (300 samples an arm). SCRIBE keeps samples of 100 bytes or more after its cleaning. G0 is A4d's own file, cut to the same prompts.

| Arm | Gate | Kept | SCRIBE v2 [95%] | T 0.6 | T 0.8 | T 1.0 | Surface S [95%] | D_L | D_S | D_C | Share D_I > 0 |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| G0 | none | 251 | 0.078 [0.055, 0.105] | 0.095 | 0.095 | 0.038 | 0.865 [0.855, 0.874] | 0.14 | 18.04 | 4.68 | 0% |
| G1 | `4` | 231 | 0.087 [0.063, 0.113] | 0.154 | 0.063 | 0.045 | 0.865 [0.856, 0.875] | 0.14 | 17.18 | 3.86 | 0% |
| **G2** | **`4,wm=5`** | 246 | **0.100 [0.075, 0.127]** | 0.147 | 0.093 | 0.059 | **0.876 [0.867, 0.885]** | 0.06 | 16.91 | 4.91 | 0% |
| G3 | `4,wp=1,wm=5` | 253 | 0.096 [0.073, 0.125] | 0.137 | 0.110 | 0.041 | 0.872 [0.863, 0.880] | 0.03 | 17.32 | 4.92 | 0% |
| G4 | `8,wm=5` | 246 | 0.104 [0.078, 0.130] | 0.148 | 0.090 | 0.076 | 0.873 [0.865, 0.880] | 0.05 | 15.73 | 4.27 | 0% |

D_I (SCRIBE's integrity distance) is 0.00 in every arm.

**Judges outside SCRIBE** (`judges.py`; all 300 samples; paired difference against G0 with its 95% interval). G0's own values: word fluency −8.968 bits a word, `<unk>` 0.091, copy8 0.0011, markup 0.0645.

| Arm | Word fluency (bits a word) | `<unk>` share | copy8 | Markup share |
|---|---|---|---|---|
| G1 | **−0.093 [−0.182, −0.010]** | +0.009 [+0.002, +0.016] | +0.000 [−0.002, +0.002] | +0.022 [+0.018, +0.027] |
| G2 | **+0.103 [+0.029, +0.178]** | −0.017 [−0.023, −0.011] | +0.000 [−0.001, +0.002] | −0.018 [−0.021, −0.015] |
| G3 | +0.047 [−0.031, +0.130] | −0.014 [−0.020, −0.009] | +0.000 [−0.001, +0.002] | −0.016 [−0.019, −0.013] |
| G4 | +0.082 [+0.001, +0.163] | −0.018 [−0.024, −0.012] | −0.000 [−0.002, +0.001] | −0.022 [−0.025, −0.019] |

**What the gate kept** (`sample`'s `gate` lines, T 0.6 / 0.8 / 1.0):

| Arm | Sentences | Not candidate 0 | Markup share of kept sentences |
|---|---|---|---|
| G1 | 1,065 / 1,074 / 1,160 | 0.66–0.68 | 0.153 / 0.141 / 0.130 |
| G2 | 645 / 699 / 802 | 0.67 | 0.067 / 0.071 / 0.059 |
| G3 | 653 / 742 / 829 | 0.67–0.68 | 0.071 / 0.074 / 0.061 |
| G4 | 684 / 784 / 925 | 0.78–0.80 | 0.069 / 0.062 / 0.058 |

**What the dev arms show.**

- **Fluency alone games itself (G1).** CyphaLM finds wiki markup easy to predict: link brackets, list lines, link names. So G1 keeps short, markup-heavy lines (twice the markup of G0's raw text). SCRIBE rises a little (+0.009), all of it at T 0.6. But the word fluency outside SCRIBE falls, the `<unk>` share rises, and fewer samples survive SCRIBE's cleaning (231 against 251).
- **The markup term fixes that (G2).** Every judge moves the same way: SCRIBE +0.022, surface S +0.011, D_S −6%, word fluency +0.10 bits a word (interval above 0), `<unk>` −0.017. Copying does not change.
- **pmi does not keep the subject (G3).** D_C (coherence) does not fall (4.92 against G2's 4.91), and pmi doubles the cost.
- **K 8 adds little (G4).** +0.004 SCRIBE over G2, within rule 1's 0.005. G4 has the lowest D_S (15.73).
- **By temperature,** G2's gain is at T 0.6 (+0.052) and T 1.0 (+0.021), not at T 0.8 (−0.002). Every interval overlaps G0's.
- **No arm copies.** D_I is 0 and copy8 moves by at most 0.0003.
- **The gate does not make the text coherent.** D_C rises a little in G2 (4.68 → 4.91). The samples below are still broken word chains.

**Cost (wall time).** Microseconds per generated byte: ungated 3,037 (P0) and 2,941 (P1) at load about 25; G1 9,885 at load 22–25 (3.3×); G4 17,562 at load 17–23 (5.9×). G2 (3,188) ran while the load fell to 3, and G3 (10,602) while it rose from 7 to 19, so neither is comparable. CPU time comes from the test read. Peak RSS stays at 79 MB in every arm (the ungated run's too).

### Decision on dev (rule 1)

**G2, `--gate 4,wm=5`.** Its dev SCRIBE (0.100) is above G0's (0.078). It holds all three bounds: the share with D_I > 0 stays at 0%; copy8 moves +0.0002; surface S rises 0.011. G4 is higher (0.104) but within 0.005, so K 4 is chosen at half the cost. GEN-A's harness was ready, so the test read follows (amendments 1 and 2).

### Samples, before and after (dev, same prompt, seed and temperature)

Shown as `sample` writes them (`\n` is a line break). The first 200 bytes.

**Prompt 0, T 0.6:** "Metternich's opposition to this, illogical, but natural from the"

- **G0:** " [[Arabic language|Greek]] word ''animal'', meaning &quot;lizard, Marchistoteliable that the subjects and has a history of the story with the [[Emancipation Church in 1998, and the [[Aramaic tribute of a single and when"
- **G2:** " album salt by the express of and in the [[United States]] [[Adult of stated the of best a great non-governmental discrimination, Rand's theory. This can action found it is possible the sub&lt;/sup&gt; to the [[poor]]"

**Prompt 2, T 0.6:** "No blood was shed this time; hunger was the weapon on which Alar"

- **G0:** "menian anthropologists the Church of the Communical Afghanist of the Computer state the [[German [[Parliament of the [[Middle the American Association for the complete of an [[electron]] [[cation]] in the first person"
- **G2:** "m include the of extension. The most influential of the [[Alaska of the [[Art of the [[American Civil War]]\n*[[1654]] - [[Aleutian Island in American Chemistry|Bell, Alaska Regional of the articles of the area between"

**Prompt 3, T 0.8:** "The requirement is that a person be the subject of the attack or"

- **G0:** " &quot;save&quot; percentages, which in considerable [[Algiers]], [[Strained the continent defeat all be real in arithmetic which were subsequent formed and the [[Alaska of and the [[line-the-the-42 Seasoning and the [[Emp"
- **G2:** " end &quot; - Lt. Col. Walter name=Amazine = Almond\n| regnum in ''Syntagma much stable and from was continued to the sea. The of American football (d. [[1926]])\n*[[1945]] - [[Adventire wenty-four all are relatively"

The gate picks the better of four broken continuations. It cannot write a better one.

### Read 7: the test read (GEN-A's frozen test split, once)

```
CYPHALM_TRIE=<this branch's cyphalm_trie> SCRIBE=/root/work/gena/runs/scribe_gena \
  scripts/gen_eval.sh test genb_T1 OUT --model genb.model --gate 4,wm=5
```

B0 is main's logged read of today's default (`scripts/gen_eval.sh test B0`). T0, the same command without `--gate`, is byte-identical to B0: the generated file and the continuation bits.

| Read | SCRIBE v2 [95%] | T 0.6 | T 0.8 | T 1.0 | Surface S [95%] | D_L | D_S | D_C | Share D_I > 0 | Copy32 | Copy8 | Kept |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| B0 (default) | 0.072 [0.059, 0.085] | 0.106 | 0.064 | 0.046 | 0.861 [0.855, 0.866] | 0.10 | 18.23 | 4.52 | 0% | 0.0386 | 0.0001 | 617 |
| T1 (`--gate 4,wm=5`) | **0.090 [0.076, 0.106]** | 0.131 | 0.081 | 0.059 | **0.876 [0.871, 0.881]** | 0.07 | 16.81 | 4.97 | 0% | 0.0557 | 0.0015 | 613 |

- **Paired, T1 − B0** (`judges.py`, 750 samples):
  - word fluency +0.107 bits a word [+0.055, +0.155];
  - `<unk>` −0.014 [−0.017, −0.011];
  - copy8 +0.0013 [+0.0003, +0.0025];
  - markup −0.021 [−0.023, −0.019].
- **Copy32 by temperature:** 0.0561 / 0.0369 / 0.0227 → 0.0674 / 0.0609 / 0.0388.
- **Continuation bits a byte** are 2.1296 in both reads. The gate does not change the mixture, only which sample is kept.
- **Cost:** CPU 501 µs per generated byte ungated (T0) and 2,055 µs gated (T1), 4.1 times. Wall time was 687 and 5,035 µs at loads of about 13 and 22.

### Decision (rule 2)

**Not adopted.** SCRIBE rises by 0.018 at every temperature, but the intervals overlap (T1's lower end, 0.076, is below B0's upper end, 0.085). The other two conditions hold:

- **Copy shares.** D_I > 0 stays at 0%. Copy8 rises by 0.0013. Copy32 rises by 0.017, the closest to its 0.02 bound.
- **Surface S** rises by 0.015, with disjoint intervals.

`--gate` stays a flag, off by default. `sample`'s output without it is unchanged. Nothing was changed after the read.

**What the gate does and does not do.** It keeps sentences that are better formed and use more real words: D_S −8%, word fluency +0.11 bits a word, less markup. It does not keep the subject: D_C rises 10% on test (4.52 → 4.97), as on dev. Its fluency score also likes text the copy expert and the trie predict well, so copy32 rises. A stronger gate, or more candidates, would meet that bound first.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| B1 | Ungated parity, trained and served | **Hit:** 81 lines byte-identical, both ways; T0 is byte-identical to B0 on test too |
| B2 | G1 +0.005 to +0.04 SCRIBE; intervals overlap | **Hit:** +0.009 (0.087 against 0.078), overlap |
| B3 | G1 lowers D_S by at least 5% | **Miss, narrowly:** −4.8% (18.04 → 17.18) |
| B4 | G2 against G1: kept markup −30% or more; SCRIBE does not fall | **Hit:** kept markup −53% (0.13–0.15 → 0.06–0.07); SCRIBE 0.087 → 0.100 |
| B5 | G3 against G2: D_C −5% or more; SCRIBE within ±0.01 | **Partial:** SCRIBE 0.096 (within); D_C 4.91 → 4.92 (no fall) |
| B6 | K 8 adds less than K 4 did | **Hit:** +0.004 against +0.022 |
| B7 | Gain largest at T 1.0, smallest at T 0.6 | **Miss:** G2's gain is largest at T 0.6 (+0.052) and nil at T 0.8 (−0.002); on test +0.025 / +0.017 / +0.013 |
| B8 | D_I > 0 rises < 0.02 everywhere; G1's copy8 rises, by < 0.02 | **Partial:** D_I > 0 stays 0% everywhere; G1's copy8 does not rise (+0.0000) |
| B9 | G1 raises surface S by 0.005 or more | **Miss:** +0.000 (G2: +0.011) |
| B10 | G1 raises the bigram word fluency (interval above 0) | **Miss, the other way:** −0.093 [−0.182, −0.010]. G2 raises it: +0.103 [+0.029, +0.178] |
| B11 | Cost K times (wp 0), 2K times (wp not 0), ±30% | **Partial:** K 4, wp 0: 4.1 times in CPU time (test). The wp case was measured only in wall time under a changing load (not scorable) |
| B12 | The test read does not meet rule 2 | **Hit:** the intervals overlap |
| B13 | Merge parity of the gated text | **Hit:** 4 prompts byte-identical |

### Checks

- **Selftest.** `cyphalm_trie selftest` passes, with the gate's checks on the tiny model:
  - the spec parser accepts and refuses the right specs;
  - the text is exactly n bytes and the same for the same seed;
  - on 8 prompts and 3 specs, the first sentence kept is the best-scoring candidate, ties to the lower k;
  - with the markup score alone, it is the lowest-markup candidate;
  - after the prompt "\n", every pmi is exactly 0.
- **Planted bugs.** Four fail the selftest: ties kept by the higher k; the prompt left in pmi's context; the budget one byte too long; the markup term's sign flipped.
- **Unchanged without the flag.** Without `--gate`, `sample` is byte-identical to A4d's dev file (P0, P1) and, on the merged tree, to B0 on the test split (T0). The gate is not reachable from `qa` or chat.
- **CTests.** The nine CTests of HANDOFF §4.4 pass on the branch, after each of the three merges with main.
- **Merges with main.** The branch merges main three times: e50dd4e (GEN-A, GEN-C), 84c0547 (GEN-E's `--plan`, BKN's `--bkn`) and c756ee3 (BKN2: byte Kneser-Ney is `sample`'s default). After each merge, the gated text of the first dev prompts is byte-identical to G1's, and the ungated text to A4d's. On c756ee3 this holds with `--no-bkn`.
- **The gate under BKN2.** Since c756ee3, the gate's candidates come from `sample`'s whole mixture, byte Kneser-Ney included. `--gate` refuses only `--doc` and `--plan`, whose experts are not in `sample_mix`. With the BKN default, the gate runs, and its text differs from the `--no-bkn` text. The reads above were all made before BKN2, so a gate over BKN is a new change. It needs its own dev choice and test read.
- **Independence.** The gate reads none of SCRIBE's features, vectors or text (section above). The judges it was not tuned on agree with SCRIBE on dev and test. Two copy measures go against the gate, both inside their bounds: copy32 +0.017 and copy8 +0.0013.

### How to reproduce

```bash
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target cyphalm_trie scribe
./build/cyphalm_trie sample enwik8 G2.txt --prompts 100 --sim default,online=1,probe=128 --gate 4,wm=5 2> G2.err
grep -E '^(gate|generated)' G2.err   # what the gate kept, and the cost
scribe ladder --pack enwik:enwik8:16:48 --ref enwik:enwik8:48:56 --test enwik:enwik8:56:64 \
  --vectors /root/work/scribe/vec32 --profile v2 --seed 20261027 --gen-trie G2.txt --label G2 > G2.md
python3 -I judges.py enwik8 G0.txt G2.txt   # the judges outside SCRIBE (/root/work/genb/runs/judges.py)
CYPHALM_TRIE=build/cyphalm_trie SCRIBE=build/scribe scripts/gen_eval.sh test genb_T1 OUT --gate 4,wm=5
```

### Next

- **Gate the new default.** BKN, now `sample`'s default (PLAN ledger 89), raised its SCRIBE from 0.063 to 0.103 on A4d's prompts. The gate draws from that mixture now, so `--gate 4,wm=5` over BKN is one dev read away. It needs its own registration, with the copy32 bound watched.
- **A subject criterion that works.** pmi against a prompt-free context did not lower D_C. GEN-E's content plan could be a gate criterion instead: keep the candidate that realises a planned fact.
- **Tokens.** At the token level a candidate sentence is a few dozen choices, not hundreds. The same gate would cost less there.

