# S9: possible worlds, consequence tests and fragility

**Odin Loch** · 2026-10-29 · DISCOVERY §4.4, §7.2 (T2), §7.6, §12 (I9), §13 S9 · follows [S7_HYPOTHESIS_TESTS.md](S7_HYPOTHESIS_TESTS.md) (S7, S7b) and [S8A_RULES_AND_CONTRADICTIONS.md](S8A_RULES_AND_CONTRADICTIONS.md)

(The design, the sets, the decision rule and the predictions below were written before any code ran. Results and decisions are added under them.)

**Question.** Cypha needs a sandbox: take the store, add or retract a few facts, let the rules run, and see what follows. Three uses need it here:
- a consequence test (T2) that can say no: S8a brought T2 back, but it was never negative, so S7's rule cut it;
- fragility: which beliefs rest on a single source;
- the hypothetical card: a record of what a world assumed and what followed.

**Gates (DISCOVERY §13 S9).**
- **I9:** sandbox worlds are never written back. A CTest checks it.
- **Consequence tests pass.** This is read as the decision rule below.

**Rule 10.** Worlds, cards and fragility are Cypha's bookkeeping about its own store. Nothing here answers a question. A card is data: training text for CyphaLM later, through the documented paths and their leak checks, or a record for Odin to read.
- DISCOVERY §4.4 lists a first use, "Odin's explicit what-if questions, answered as labelled hypotheticals". That would have Cypha answer a question itself, which rule 10 forbids unless CyphaLM writes the answer. **No path that answers is built.** It is flagged for Odin.

## Design (fixed before the runs)

### Worlds

- A world is a copy-on-write overlay of a store: the store itself, read only, plus assumptions, each one added or retracted.
- The store is held by a const reference. A world has no way to change it, and `World::write_back` always refuses (I9).
- **Forward chaining.** The kept G2 rules (S8a's language: constant, same, inverse, path) chain inside the world up to depth 3, starting from the assumptions.
  - Each derived fact keeps its rule, its premises and its depth.
  - Its confidence is the product of the rules' PCA confidences along its chain (assumptions and stored facts count 1).
  - A derived fact that the world already holds (stored or assumed) is a confirmation, not a new consequence.
- **The result lists:**
  - new consequences;
  - consequences that disappeared: facts the retracted assumptions supported in the store that the world can no longer derive within depth 3 and does not hold;
  - contradictions.
- **Reductio.** A contradiction refutes the assumption set. The minimal conflicting subsets are found by trying subsets in order of size and dropping supersets of any subset already found.
- **Taint.** Everything a world derives or assumes is exported with the hypothetical flag (S5's taint). It is never answerable, and its taint survives any derivation from it.

### What counts as a contradiction with Known facts

A derived fact with chain confidence ≥ 0.8 (S8a's T2 threshold) contradicts the world's held facts (stored or assumed, not derived) when it:
- **(a) functional slot:** puts a value into a functional slot (φ ≥ 0.9; φ from the store, S7b's choice) that holds only other values, compared with `same_value`;
- **(b) date checks:** gives a date slot another year when the slot is functional, or gives a death before a birth, or a life over 125 years;
- **(c) type check:** has an object of the wrong type: a place relation whose object is a person page and not a place, or a person relation whose object is a place or country page and not a person.

These are S7's T1 checks (`t1_phi`), applied to consequences instead of the hypothesis itself.
- Two derived values in one functional slot also count as a contradiction.
- An assumption's own conflict with the store is T1's result and is not counted in T2.

### T2 through worlds

The world is the store plus the hypothesis h.
- **Negative** when the chain reaches a contradiction.
- **Positive** when there is no contradiction and some consequence of h is already held by the store (any chain confidence, as in S8a).
- **Inconclusive** otherwise.

**Variants**, chosen on dev by T2's mutual information with the truth (S7's choice statistic); a tie keeps the registered default:
- depth 1, 2 or 3;
- with or without the type check (c).
- **Registered default:** depth 3, with the type check.
- S8a's own T2 (depth 1, the hand-set φ, functional slots only) is printed beside it on the same items.

### Sets and reads

- **Dev: S7's dev set.** 1,479 pairs, the dev half by bit 11 of FNV-1a(subject).
  - It runs on S7's reduced store: every S7 hold-out removed, 22,783 → the reduced store.
  - The rules are mined on that store with S8a's defaults (support ≥ 10, PCA ≥ 0.5, BH at q 0.1).
  - The set is rebuilt with S7's code and seeds; its pairs are checked against S7's own items file.
- **Fresh confirm: S15 dev's choose half.** These are the subjects with bit 17 of FNV-1a(subject) = 0: 5,754 pairs, 2,618 of them store-reachable.
  - It runs on S7b's S15-reduced store (13,390 facts), with rules mined on that store.
  - It uses the φ table chosen on dev, as is (S7b's amendment).
  - It is read once, for the chosen variant. No choice is made on it.
  - This half has been read before: by S7b, only to check its adapter, and by S8a and S8b during their development. None of those reads involved T2 through worlds.
- **Not read:** S15's test half, S7b's confirm half and S7's test half.
- **Likelihoods** are three-way with add-one (S7b's form): P(outcome | true) and P(outcome | false).
  - Each proportion gets a Wilson 95% interval.
  - Each outcome's likelihood ratio gets a pair-bootstrap 95% interval (2,000 resamples, seed 20261029).

### Decision rule: "consequence tests pass"

All three must hold:
1. The CTest's known-answer checks for chaining, contradictions, reductio and T2 pass.
2. On dev, T2 (the chosen variant) earns its place by S7's rule: applicable ≥ 0.05 and the lower bound of Youden's J above 0.
3. On the confirm set, the likelihood ratios point the way dev's do: LR(positive) > 1 and LR(negative) < 1, for each outcome seen at least 5 times. An outcome seen fewer than 5 times is reported and not judged.

### Fragility (§7.6)

**The store.** The v2 store: every seed row (category, infobox, table, Wikidata) plus the admitted text facts (`facts_f2.tsv`).

**Evidence model, fixed now:**
- **Sources.** One evidence item per (source instance, triple). The source instances are:
  - `category|page`, `infobox|page`, `table|page`: the page holding the row's byte offset;
  - `text|page`: an admitted fact's source page;
  - `wikidata`: Wikidata counts as one source.
- **Trust.** τ is Beta(20, 1)'s mean for text seeds and text facts (pattern_learn's seed and text trust) and Beta(50, 1)'s mean for Wikidata. π = θ = ω = 1.
- **Derivations.** Items from the kept rules mined on the v2 store, depth 1, from other stored facts. Each has mass PCA × the product of its premises' direct P. A premise's direct P is its opinion from its sources, its rivals and its constraints, without derivations.
- **Competing values.** In a functional slot (store φ ≥ 0.9), every other triple whose object is not `same_value` adds its source mass as a competing value, weighted by φ.
- **Constraints.** Lifespan violations (death before birth, a life over 125 years) are hard violations.
- **Fusion.** W = 2 and a = 0.5 (PLAN §7 defaults). The state comes from `classify`, with the strongest rival's opinion. Two triples whose objects are `same_value` are neither rivals nor support for each other.

**Held facts.** A fact is held when it is Known (Grade High) or Believed.

**Worlds for each held fact:**
- (i) its top source retracted: the source item with the largest mass;
- (ii) its weakest premise retracted: the premise with the lowest direct P across its derivations, when it has any.

**Definitions:**
- A fact **drops** when its state class falls: Known → anything else; Believed → Contested, Ignorant or Vacuous.
- **Fragility** = drops / applicable worlds.
- A **single point of failure** has fragility 1. These are written out as certainty gaps for S3's gap map.
- A source instance's **blast radius** is the number of held facts that drop when all its rows are retracted, with one level of cascade through derivation premises.

**Known is out of reach, by arithmetic.** Grade High needs the 5% quantile of Beta(r + 1, s + 1) to be at least 0.9. With s = 0 that needs r ≥ 27.4 units of positive mass. No fact in this store has more than a few sources, so prediction W1 expects no Known fact. Fragility is therefore read on held facts.

### Cards

- Hypothetical cards for the T2 worlds that reach a contradiction (dev, the first 20 in item order) and for the source instances with the largest blast radius.
- Each card holds the assumptions, the new consequences, the consequences that disappeared, the contradictions, the minimal conflicting subsets and its taint.
- They are written as data (`OUT.cards.txt`): labelled HYPOTHETICAL and ending "Nothing here is stored." (I8).

### I9 in the run

- Each base store's fingerprint (FNV-1a 64 of its sorted triples) is taken before and after every world of the run.
- A write-back attempt is made on purpose and must be refused.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| W1 | No stored fact is Known (Grade High) under PLAN §7's constants |
| W2 | T2 (the registered default) is negative on at least 1% of dev's false hypotheses, and its negative rate on false hypotheses is at least twice that on true ones |
| W3 | T2 is applicable (positive or negative) on 8–25% of dev hypotheses |
| W4 | Depths 2 and 3 add less than 20% more applicable hypotheses than depth 1 |
| W5 | The type check raises T2's mutual information on dev |
| W6 | T2 earns its place on dev by S7's rule |
| W7 | On the confirm set the likelihood ratios point the way dev's do |
| W8 | At least 95% of held facts are Believed at grade Low; fewer than 2% of stored facts are Contested |
| W9 | Single points of failure are 80–92% of held facts |
| W10 | Retracting Wikidata drops 60–80% of held facts; no other source instance has a blast radius above 60 |
| W11 | Weakest-premise worlds apply to fewer than 10% of held facts, and drop fewer than 5% of those |
| W12 | I9 holds: every base store's fingerprint is unchanged after every world, and the write-back attempt is refused |

**Clarifications (written after the dev run started and before the confirm read).**
- **The judged confirm set.** As in S7b, the confirm rule is judged on the reachable pairs: both members' objects occur as objects of the relation in the S15-reduced store. All 5,754 pairs are printed beside it.
- **Which contradictions need the type check.** The type check is tried after the functional and date checks. So a contradiction of kind "type" is one that no other check found, and it counts only in the variants with the type check.
- **Ties between variants.** The registration says a tie keeps the default. If the default is not among the best, the tied variant closest to it wins: the default's type setting first, then the deeper. (Added after the first dev run, when depth 1 with and without the type check tied.)
- **The card's wording.** epistemic.hpp's card says "contradiction with a Known fact". In this store no fact is Known, so on these cards it means a held fact (stored or assumed).

## Results (run 2026-10-29)

Tool: `pattern_learn --worlds` (commit a60f4ae, committed before the confirm read). The final run (`--read-confirm`) took 390 s at a 287 MB peak. A dev-only run takes 120–170 s at 277 MB. The dev lines of the final run are identical to the dev-only run's.

### The sets

- **Dev.** S7's dev set rebuilt with S7's code and seeds: 1,479 pairs (U 371, C 1,108) on the reduced store (19,735 of 22,783 facts). Its 2,958 items equal S7's own items file row for row (pair, set, label, Wikidata flag, subject, relation, object).
- **Confirm.** S15 dev's choose half: 5,754 pairs, 2,618 reachable, on the S15-reduced store (13,390 facts). Both counts equal S7b's. Read once.
- **Rules.** On the dev store, 657 candidates and 183 kept: 177 constant, 4 same, 2 inverse, **no path rule**. On the S15-reduced store, 112 kept. Most kept rules link one occupation or genre to another (`occupation(X, Novelist) => occupation(X, Writer)`).

### T2 through worlds on dev

| Variant | Applicable | true pos / neg | false pos / neg | MI, bits |
|---|---:|---|---|---:|
| depth 1, with or without the type check | 0.051 | 117 / 0 | 34 / 0 | 0.0123 |
| depth 2, either | 0.054 | 118 / 0 | 41 / 0 | 0.0099 |
| depth 3, either (registered default) | 0.054 | 118 / 0 | 42 / 0 | 0.0096 |
| S8a's T2 (depth 1, hand φ) | 0.051 | 117 / 0 | 34 / 0 | 0.0123 |
| T1 (φ_store), for reference | 0.092 | 111 / 24 | 12 / 123 | 0.0403 |

- **T2 is never negative.** No world on dev or confirm reached a contradiction. The type check changes nothing, at any depth.
- **Why.** A contradiction needs a confident rule (PCA ≥ 0.8) that puts a value into a functional slot (φ ≥ 0.9). No kept rule does. Exploratory check (dev, no choice made on it): the only slot where a derived fact ever met other held values is `occupation` (φ_store 0.28; 7 true and 47 false items). So a softer contradiction (any φ ≥ 0.5) would change nothing either.
- **Depth 1 of the world equals S8a's T2** on every dev and confirm item. The sandbox reproduces S8a's T2, and deeper chains add a few positives, mostly on false items.
- **The choice.** Depth 1 has the highest MI. With and without the type check it ties (identical outcomes), and the tie went to the type check, the default's setting. Chosen: depth 1 with the type check.
- **Earn its place: cut.** Applicable 0.051 (≥ 0.05), but Youden's J is +0.000 [−0.032, +0.102]. A test that is never negative has c = 0, so J can never be above 0.
- **T2 is applicable only on C** (subjects with other stored facts). On U (one-fact subjects) there is nothing for a consequence to meet.
- **T2 and T1 do not overlap.** All 151 T2 positives are T1-inconclusive. T1 decides 270 other items.
- **Surface variants.** 36 of the 151 positives are case variants of a value the store still holds (for example, `Isaac Newton occupation mathematician` against a stored `Mathematician`). Three of them are "false" members, so those three labels are wrong. Without the 36, positives are 84 true and 34 false.

**Three-way likelihoods, chosen variant (dev, add-one; Wilson 95% on the raw proportions; pair bootstrap 95% on LR):**

| Outcome | Seen | P(· \| true) | P(· \| false) | LR |
|---|---:|---|---|---|
| positive | 151 | 0.0791 [0.0664, 0.0940] | 0.0231 [0.0166, 0.0321] | **3.35 [2.50, 4.72]** |
| negative | 0 | 0 [0, 0.0026] | 0 [0, 0.0026] | 0.995 (add-one only) |
| inconclusive | 2,799 | 0.921 [0.906, 0.934] | 0.977 [0.968, 0.983] | 0.943 [0.928, 0.956] |

The registered default (depth 3 with the type check) gives LR(positive) 2.75 [2.08, 3.78].

**A bug found on dev, fixed before the confirm read.** In the first dev run, a consequence that matched the hypothesis itself by `same_value` (for example, `poet` derived `Poet`) counted as a confirmation: 130 true and 68 false positives at depth 1. The registration says a confirmation is a consequence "already held by the store". `chain` now marks which confirmations the store holds, `t2_world` counts only those, and a CTest checks it.

### The confirm read (S15 dev's choose half)

| Set | Applicable | true pos | false pos | LR(positive) |
|---|---:|---:|---:|---|
| **reachable (judged), 2,618 pairs** | 0.004 | 20 | 1 | **10.5 [4.25, 28.0]** |
| all, 5,754 pairs | 0.002 | 25 | 1 | 13.0 [5.0, 33.0] |

- P(positive | true) is 0.0076 [0.0050, 0.0118] and P(positive | false) is 0.0004 [0.0001, 0.0022] on the reachable set.
- The positive LR points the way dev's does. Negative was never seen, so it is not judged. **Rule 3 holds.**
- T2 fires on under 1% of S15's pairs: the S15-reduced store has lost 9,393 facts, and with them most of what a consequence could meet.

### Decision: "consequence tests pass"

1. The CTest known-answer checks pass: `native_epistemic_test` runs 881 checks, 0 failed (100 of them S9's).
2. **On dev, T2 does not earn its place** by S7's rule (J's lower bound −0.032).
3. On the confirm set, the likelihood ratio points the way dev's does (LR(positive) 10.5).

**Not met: rule 2 fails.** T2 stays out of S7's loop. The sandbox works and I9 holds, but on this store a world cannot reach a contradiction, because the rules that survive mining do not reach functional slots. T2's positive result carries evidence (LR 3.35 on dev, 10.5 on fresh data), but S7's rule measures a test by J, and J needs negatives. Whether a confirm-only test may enter the loop through S7b's three-way likelihoods is a ruling for Odin.

### Fragility (§7.6) of the v2 store

- **The store:** 23,366 distinct facts from 25,390 seed rows and 1,394 admitted text facts, 3,077 source instances (Wikidata as one).
- **Rules on the v2 store:** 255 kept (240 constant, 8 path, 4 same, 3 inverse). They give 5,462 depth-1 derivations.
- **Rivals:** 16,078 rival pairs in functional slots. **Lifespan violations:** 32.

| State | Facts |
|---|---:|
| Known | 0 |
| Believed | 22,418 (grade Low 22,350, Moderate 68) |
| Contested | 571 (2.44%) |
| Ignorant | 377 |

**Held facts: 22,418.** Their median P is 0.664. With one source and no rival, P = (τ + 1)/(τ + 2): 0.661 for a text source, 0.664 for Wikidata.

| Measure | Count | Share of held |
|---|---:|---:|
| Drop when their top source is retracted | 19,114 | 85.3% |
| **Single points of failure (fragility 1)** | **17,234** | **76.9%** |
| Have a weakest-premise world | 2,385 | 10.6% |
| ... and drop in it | 0 | 0% of those |

- **The 1,880 facts** that drop when their top source goes but are not single points of failure have a premise world in which they do not drop. Their fragility is 0.5.
- **By the number of sources:** 19,576 held facts have one source, 2,826 have two and 16 have three or more. Of the two-source facts, 6 are still single points of failure: birth dates and years with two rival values in their slot, so P is barely above 0.5 (0.512–0.519) and losing Wikidata drops them.
- **By top source:** text 519 of 519 single points of failure (100%), table 11 of 11, category 89.4%, infobox 89.2%, Wikidata 72.6%.
- **A derivation alone never holds a fact up.** A depth-1 derivation adds PCA × P(premise), about 0.3–0.7 of mass, so u stays at 0.75 or more. And no fact drops when its weakest premise goes, because its own source still holds it.
- **The list:** the single points of failure are written as certainty gaps in S3's columns (`OUT.spof.tsv`: gap id, kind `certainty`, entity, relation, object, P, top source). They are not merged into S3's gap map, so S3's output is unchanged.

**Blast radius (one level of cascade).**

| Source instance | Facts | Held facts that drop |
|---|---:|---:|
| **Wikidata** | 17,904 | **13,936 (62.2% of held)** |
| infobox, United Kingdom general election, 2001 | 43 | 43 |
| infobox, Arthur Meighen | 21 | 21 |
| infobox, Andrew Jackson | 16 | 16 |
| infobox, Andrew Johnson | 21 | 15 |

- **Mean radius by kind:** infobox 2.39 (815 instances), category 1.78 (1,527), table 1.00 (11), text 0.72 (723).
- **The largest instance other than Wikidata** drops 43 facts.

### Cards

`OUT.cards.txt` holds 10 cards:
- **Dev contradictions:** none, because no world reached one.
- **Five dev worlds where T2 is positive,** as illustrations.
- **The five source instances with the largest blast radius,** each as a world that retracts the facts left with no source. For Wikidata, 15,047 facts lose their only source, 7,135 derived facts (depth ≤ 3) can no longer be derived, and 13,936 held facts drop.

Every card starts with the HYPOTHETICAL label, ends with "Nothing here is stored.", and carries a "Tainted:" line.

### I9

- 43,409 worlds were run over three base stores (S7's reduced store, the S15-reduced store and the v2 store).
- Each store's fingerprint was taken after every world. None changed.
- The deliberate write-back attempt was refused.
- The CTest also checks that:
  - write-back is refused;
  - the store is byte-identical after every world;
  - a world's exported items stay out of answer mode even when their own taint flag is dropped (their premises carry it).

### Other checks

- **CTests:** all nine of HANDOFF §4.4 pass.
- **S3's `--gaps` run is byte-identical** with this branch's `pattern_learn`: `gap_map.tsv`, `.cells.tsv` and `.demand.tsv` have the same MD5 as `/root/work/s4/runs/test/`.

## Predictions scored

| # | Prediction | Result | |
|---|---|---|---|
| W1 | No stored fact is Known | 0 Known | hit |
| W2 | T2 negative on ≥ 1% of dev false, ≥ 2× the true rate | never negative | miss |
| W3 | T2 applicable on 8–25% of dev | 5.4% (default) | miss |
| W4 | Depths 2–3 add < 20% applicable over depth 1 | +6% (160 vs 151) | hit |
| W5 | The type check raises MI | no change: no type clash | miss |
| W6 | T2 earns its place on dev | J lower bound −0.032 | miss |
| W7 | Confirm LRs point dev's way | LR(positive) 10.5; negative not seen | hit |
| W8 | ≥ 95% of held Believed Low; < 2% Contested | 99.7%; but 2.44% Contested | miss |
| W9 | Single points of failure 80–92% of held | 76.9% | miss |
| W10 | Wikidata drops 60–80%; no other instance > 60 | 62.2%; largest other 43 | hit |
| W11 | Premise worlds < 10% of held; drop < 5% of those | 10.6%; 0% | miss |
| W12 | I9 holds | holds | hit |

**5 of 12.** The misses on T2 (W2, W3, W5, W6) share one cause. I expected S8a's path rules (birth place and country to nationality) to survive on the reduced store and reach functional slots. None did.

## For Odin

1. **Rule 10 and DISCOVERY §4.4 use 1.** "Odin's explicit what-if questions, answered as labelled hypotheticals" would have Cypha answer a question itself. No path that answers is built. Options: drop use 1, or keep worlds as data (cards as training text for CyphaLM, through the documented paths and their leak checks) and let CyphaLM write any answer.
2. **Confirm-only tests.** S7's earn-its-place rule (Youden's J above 0) can never keep a test that is never negative. T2's positive result carries evidence: LR 3.35 [2.50, 4.72] on dev and 10.5 [4.25, 28.0] on fresh data. Should a test enter the loop when its three-way likelihood interval excludes 1 (S7b's form), even with no negatives? T2 is a store test, so it is never independent evidence (I4) either way.
3. **Wikidata as one source.** Retracting it drops 62% of held facts. Counting it per claim, or per reference, would change every fragility number. This report counts it as one, as registered.
4. **The single points of failure (17,234)** are written as certainty gaps but not merged into S3's gap map, which would change S3's output.

## Reproduce

```
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
./pattern_learn --worlds enwik8 /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv \
  /root/work/f2/v2/aug/facts_f2.tsv bench/eval/temporal_holdout_dev.tsv OUT [--read-confirm]
```
Outputs: `OUT.dev_items.tsv`, `OUT.dev_rules.tsv`, `OUT.confirm_items.tsv` (with `--read-confirm`), `OUT.fragility.tsv`, `OUT.spof.tsv`, `OUT.blast.tsv` and `OUT.cards.txt`. Runs are in `/root/work/s9/runs/` (`final.*`).

## S9b: S9's questions settled by measurement (2026-10-29)

**Odin Loch** · 2026-10-29 · HANDOFF §5.14, open to Odin's veto (PLAN ledger 55) · T2 in the loop and the loop's part of Wikidata are in [S7_HYPOTHESIS_TESTS.md](S7_HYPOTHESIS_TESTS.md), "S7d"

(The reads, the rules and the predictions below were written and committed before any S9b run. Results are added under them.)

### §4.4 use 1 against rule 10

DISCOVERY §4.4's first use, "Odin's explicit what-if questions, answered as labelled hypotheticals", would have Cypha answer a question itself. **Rule 10 wins.** Worlds, cards and fragility stay Cypha's data: cards may become training text for CyphaLM through the documented paths and their leak checks, or a record for Odin. No path that answers is built.

### Wikidata as one source or per claim (fragility's part)

- `--worlds --s9b` runs S9's fragility twice on the v2 store: Wikidata as one source instance (`wikidata`, as S9 ran) and per claim (`wikidata|subject|relation|object`). It reports states, held facts, single points of failure and the blast radius both ways.
- The choice is S7d's rule 2: one source unless no Known promotion rests on Wikidata alone.

### Single points of failure in S3's gap map

- **The flag.** `pattern_learn … --gaps FILE --gaps-spof SPOF.tsv` adds S9's single points of failure (`OUT.spof.tsv`) to S3's map as a new gap kind, `fragile` (DISCOVERY §13 S3 names "contested and fragile lists"). Without the flag S3's output is byte-identical.
  - One gap per (entity, relation). A slot that already has a contradiction or certainty gap gets none: those already ask about it.
  - Its p is the fact's P, its demand 0, its status open. VOI as for every gap (S3's placeholder test).
- **What it changes.** Gaps by kind; the share of the top 100 and top 1,000 by VOI that are fragile; the old gaps' mean rank shift.
- **Does fragility point at errors?** Among held facts with no Wikidata source, whose slot Wikidata covers: the share that disagrees with Wikidata, for single points of failure against facts with two or more sources. The difference gets a bootstrap 95% interval over subjects (2,000 resamples).
- **Rule.** The fragile kind joins S3's documented command (with `--gaps-spof`) when the difference's interval lies above 0: then fragility points at errors, and asking about these slots has value. Otherwise it stays a flag. S3's parity reference stays the run without it either way.

### Predictions (registered before the runs)

| # | Prediction |
|---|---|
| F1 | Per claim, single points of failure stay within 1 point of S9's 76.9% of held, and no source instance's blast radius exceeds 100 |
| F2 | The fragile kind adds 12,000–17,000 gaps after the slots already asked about are left out |
| F3 | Fragile gaps take more than half of the top 1,000 by VOI |
| F4 | Single points of failure disagree with Wikidata more often than multi-source facts, but the interval includes 0: the kind stays a flag |

### Results (run 2026-10-29)

Tools: `pattern_learn --worlds … --s9b` and `pattern_learn … --gaps … --gaps-spof` (commit e4afd6f). The `--s9b` run takes about 400 s on a loaded machine at 280 MB. Without the flags, every output of `--worlds` and S3's run is byte-identical to main's.

#### Wikidata per claim

| | One source (S9) | Per claim |
|---|---|---|
| States | — | equal for all 23,366 facts |
| Held | 22,418 | 22,418 |
| Single points of failure | 17,234 (76.9%) | 17,256 (77.0%) |
| Largest blast radius | Wikidata 13,936 (62.2% of held) | an infobox page, 43 |
| Wikidata's instances | 1, radius 13,936 | 17,904, radius at most 1 (mean 0.73) |

- Per claim, nothing about each fact changes. Only the blast radius does: Wikidata stops being one thing that can fail.
- That is exactly the dependence the one-source count is there to show. A wrong or vandalised provider fails many claims at once. **One source, as S9 ran, by S7d's rule 2.**

#### Single points of failure in S3's gap map

- S9's 17,234 single points of failure give **10,454 fragile gaps**: one per slot, and 738 slots already had a certainty or contradiction gap.
- The fragile gaps' median p is 0.664 (one source, no rival), their median VOI 0.48.

| | Without the flag | With `--gaps-spof` |
|---|---|---|
| Gaps | 27,842 | 38,296 |
| Fragile in the top 100 / 1,000 / 5,000 | — | 22 / 239 / 807 |
| The old top 100 still in the top 100 | 100 | 78 |

- The old gaps keep their order among themselves. They move down by 8,912 places on average.

**Does fragility point at errors?** These are held facts with no Wikidata source whose slot Wikidata covers:

| | Facts | Disagree with Wikidata |
|---|---:|---|
| Single points of failure | 2,215 | 875 (0.395 [0.375, 0.416]) |
| Two or more sources, not single points of failure | 5 | 1 (0.200 [0.036, 0.625]) |

- The difference is +0.195 [−0.283, +0.414] (bootstrap over 1,106 subjects). **The interval includes 0, so the fragile kind stays a flag.**
- The comparison has almost nothing on one side. Facts without a Wikidata source almost never have a second source. So this store cannot say whether fragility points at errors.
- What it does say: 39.5% of the text-only single-source facts that Wikidata covers disagree with it. That is the size of the certainty problem these gaps would ask about.

### Predictions scored (S9b)

| # | Prediction | Result |
|---|---|---|
| F1 | Per claim, single points of failure within 1 point of 76.9%; no instance's blast radius above 100 | **Hit:** 77.0%; largest 43 |
| F2 | The fragile kind adds 12,000–17,000 gaps | **Miss:** 10,454 (one per slot: several facts often share a slot) |
| F3 | Fragile gaps take more than half of the top 1,000 | **Miss:** 239 |
| F4 | Single points of failure disagree more, but the interval includes 0; the kind stays a flag | **Hit**, though the comparison group is only 5 facts |

2 hits, 2 misses.

### Decisions (S9b), open to Odin's veto

1. **§4.4 use 1:** rule 10 wins. Worlds and cards stay data.
2. **Wikidata counts as one source** (S7d's rule 2), as S9 ran.
3. **The fragile kind stays behind `--gaps-spof`.** Its value cannot be shown on this store. S3's documented command and parity reference are unchanged.

### Checks

- `native_epistemic_test` passes 1,310 checks, 17 of them new. The new checks cover:
  - fragile gaps (one per slot; none where a certainty or contradiction gap asks; their place in ties);
  - `supporting_independent_sources`;
  - a planted case where K4's stated null likelihoods understate how often false hypotheses pass. There K4 over-promotes while K3 keeps its bar.
- Planted bugs, each caught: the certainty gap ignored (2 checks fail); negative results counted as support (1); the fragile kind not last (1); a second fragile gap per slot (2).
- S3's `--gaps` run is byte-identical to main's, all 10 files (gap map `a77678d8ef7dfdef`).
- `--worlds` without `--s9b`: every output file is byte-identical to main's, and stdout matches apart from timings.

### Reproduce (S9b)

```bash
./pattern_learn --worlds enwik8 seeds.tsv wd_values.tsv facts_f2.tsv bench/eval/temporal_holdout_dev.tsv s9 --s9b
./pattern_learn enwik8 OUT --cv --wikidata wd_values.tsv --dev DEV --gaps OUT/gap_map.tsv --demand ROWS --parse PARSE --gaps-spof s9.spof.tsv
```
Runs are in `/root/work/s9b/runs/` (`s9n.*`, `gspof`).
