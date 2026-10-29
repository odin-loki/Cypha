# A1d and A2b: answering variants, measured

**Odin Loch** · 2026-10-28 · follows [A1C_ENTITY_FAMILIARITY_AND_COPY.md](A1C_ENTITY_FAMILIARITY_AND_COPY.md), [A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md](A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md) and [B4_CLASS_CONTEXT_EXPERT.md](B4_CLASS_CONTEXT_EXPERT.md) · steps A1–A3 of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4

Odin's instruction for this work: try a lot of things, profile and record everything, then decide from the facts. Eight open questions were measured on dev, and the final configurations were read once on the frozen test.

**Questions.**
1. **Entity in the token.** With E1's entity in the gate-side token and the copy expert on, does CyphaLM name the question's entity in its "I don't know" message and tag?
2. **Which entity.** E1 picks a wrong entity for 37 in-slice dev questions. Does choosing among the question's spans by CyphaLM's own counts do better?
3. **Which statistic.** Title count, bold-lead count, or both?
4. **Token training.** Statistic on every training turn, the gate's decision (outcome), or a statistic cut matched to the dev base rate?
5. **Shard form.** More copies, more phrasings, or the fact sentences as plain text?
6. **Combined.** The best of these with the class expert and copy. And does class + copy keep the answers (B4B rule 3)?
7. **Gate.** The Cypha-classifier gate against `qa`'s own threshold gate.
8. **R3 with raw-byte frames** (F2R).

**Answers.**
1. **No.** 0 of 65 tags in four variants. InfiniGram's long match keys on the entity's last bytes and writes a training entity that ends the same way ("1 Ceres" → *Antares*). Copy's weight cannot beat it.
2. **Yes.** Taking the span of the question with the highest title count raises entity accuracy from 63 to **76** of 100 in-slice questions. The token's dev precision rises from 0.435 to **0.600** at recall 1.00, and CyphaLM follows it exactly.
3. **The title count.** On span candidates its AUROC is 0.900 (0.862–0.940) against the bold count's 0.869, within each other's intervals. But as the token the bold count halves recall (0.50 against 1.00, intervals apart), so the title count stays.
4. **Outcome labels.** It is the only labelling every model follows. A third variant, a token only on `low` turns, keeps that and also keeps A2's gated answers.
5. **None beats the plain shard.** Every form writes the same 7 right answers; the gate keeps 2–3 out of fold in all of them.
6. **The low-only token with the shard is the best combination** (dev: precision 0.455–0.600, recall 1.00, 2 right and 0 wrong at 2% wrong). **Class + copy fails rule 3 with the shard:** it echoes the question (*"Avicenna born?"*), 6 raw right answers fall to 2.
7. **`qa`'s threshold gate stays.** Over ten configurations the Cypha classifier keeps more right answers (21.7 against 15) but 3.6 times as many wrong ones (18.0 against 5), breaks the 2% target in 6 of 30 seed runs, and varies with the seed.
8. **Yes, better than A3.** Raw frames with the raw-form shard: 5 raw right and 2 right / 1 wrong out of fold, against 3 and 0 / 1.

PLAN rule 10 holds. CyphaLM writes every answer byte. The candidate spans and the familiarity token are gate-side inputs from the question parse and CyphaLM's own counts (Odin's ruling of 2026-10-28). Strings are compared only to grade, to build training text, and for leakage checks.

---

## Predictions (written before each batch of runs)

The full wording, with the reason for each, is in the run log (`/root/work/qa2/notes/predictions.md`). In short:

| Id | Prediction |
|---|---|
| Q1 | P0 (A1c run D′ rerun) reproduces every A1c number |
| Q2 | With the entity in the token, every model still abstains on exactly the 69 `low` and `mid` questions |
| Q3 | Frame `entity="{E}"`: 0–10% of tags name the entity (InfiniGram's long match on the message's wrong entity wins at the tag) |
| Q4 | The same with the entity last in the tag: 0–15% |
| Q5 | Frames `about {E}` and `entity="{E}"`: the message names the entity in 40–80% of IG + copy abstentions, the tag in 30–70% |
| Q6 | Frame `ent="{E}"`: 0–10% |
| Q7 | Holdout BPC (trie + IG) within 0.003 of P0 in every frame run |
| Q8 | E1's entity equals the in-slice source title for 40–60 of 100 |
| Q9 | Best-by-title candidates add +2–8 (NP chunks), +5–15 (NP sub-spans), +8–20 (all spans); the longest capitalised run stays within ±8 |
| Q10 | `ent_title` AUROC 0.80–0.90 with NP sub-spans or spans, the best; token precision 0.45–0.65 at recall 0.90–1.00 |
| Q11 | `ent_tb` within ±0.03 of `ent_title`; `ent_bold` 0.00–0.05 below, the paired interval including 0, for every candidate set |
| Q12 | Shard ×2: covered raw right 5–8; out of fold 2–4 right, 0 wrong; BPC +0.001 to +0.003 |
| Q13 | Shard ×4: covered raw right 6–9; out of fold 2–5 right, 0–1 wrong; BPC +0.002 to +0.006 |
| Q14 | More phrasings: covered raw right 6–9; out of fold 2–5 right, 0 wrong; BPC within +0.003 |
| Q15 | Sentences as plain text: within ±1 of ×1 on raw and gated; BPC within ±0.002 |
| Q16 | No shard variant answers a not-covered question at 2% wrong |
| Q17 | Class + copy (m 8): BPC 1.90–1.97; raw right 1–3; nothing right at 2% wrong |
| Q18 | Class + copy (m 8) + shard: covered raw right 4–7; trie + IG keeps 1–3 right, 0 wrong; the copy model 1–3 right |
| Q19 | R3 raw frames + raw shard: runs on 15–30 questions; raw right 2–6; out of fold 0–2 right, 0–1 wrong; at least A3's raw right |
| Q20 | Span + title token: every model abstains on exactly the 50 `low`: precision 0.600, recall 1.00 |
| Q21 | Span + bold token: exactly the 18 `low`: 0.833 / 0.50 |
| Q22 | Statistic on every turn: Dolly `low` falls from 97% to 60–85%; the trie abstains on nearly all (precision 0.23–0.35); trie + IG 0.25–0.45 |
| Q23 | Matched cut: trie precision 0.30–0.55, recall 0.30–0.70, below the outcome labels on both |
| Q24 | Shard + copy (m 8), no class expert: covered raw right ≥ 4; out of fold 0–2 right |
| Q25 | C1 (token + shard + copy): abstention as Q20; trie + IG out of fold 2–3 right, 0 wrong |
| Q26 | C2 / C3 (+ class shape or tags): raw and gated within ±1 of C1; BPC + class 1.98–2.00, + copy 1.90–1.93 |
| Q27 | No tag names the entity in any combined run |
| Q28 | `cypha_gate` never keeps more right answers than `qa`'s gate, and lets through at least as many wrong |
| Q29 | Low-only token (C1L): the trie writes the token itself, precision 0.23–0.45; trie + IG 0.45–0.60 at recall 0.8–1.0; out of fold 1–2 right, 0 wrong |
| TQ1 | Frozen test, C1L: recall 0.90–1.00; precision 0.45–0.65 (trie), 0.35–0.55 (trie + IG); the dev gate keeps 0–2 right, 0–1 wrong |
| TQ2 | Frozen test, R3raw: R3 runs on 15–25 questions; raw right 1–4; the dev gate keeps 0–2 right, 0–1 wrong |

---

## Setup

As A1c and A2 ([A1c "Setup"](A1C_ENTITY_FAMILIARITY_AND_COPY.md)): enwik8's first 8 MiB (train `[0, 7340032)`, holdout `[7340032, 8388608)`), Dolly 15,015 records, `cyphalm_trie qa`, dev 130 questions (100 in-slice, 30 not-in-slice), the frozen test 200 read once at the end. Every run is under `nice -n 5`, one at a time, single-threaded, on 4 cores shared with up to three other jobs (wall times below vary with that load). g++ 13.3 `-O2 -std=c++20`, clean under `-Wall -Wextra`.

- **The don't-know configuration** (A1c's run D′, the starting point): 20% don't-know turns from the mention file (6,583 rows), the familiarity token after `Assistant: `, `ent_title` with A1c's training-side cut (count ≥ 1), `--fam-train outcome`.
- **Abstention** is CyphaLM writing the message. Precision and recall are on the 30 not-in-slice questions (base rate 0.231).
- **Holdout BPC "trie / + IG"** in the tables is the trie mixture over the whole holdout (A0: 2.344571) and trie + InfiniGram over its second half (A0: 2.071574), as in A2. Every copy and class figure is on the second half.
- **Coverage at 2% wrong** is out of fold: 5-fold CV inside dev (fold = row mod 5). The test uses the gate chosen on all of dev.
- **The fact shard** is D4's `/root/work/f2/shard_facts.tsv` (1,488 rows after the BPC-holdout drop), the file A2 used. The later `/root/work/f2/v2/shard_facts.tsv` (6 Huxley year rows fixed) was not used, for comparability.
- **Parity.** P0 reruns A1c's run D′ with the new code: the rows file, the `ABSTAIN`, `FAMSTAT` and BPC lines are byte-identical (Q1 hit). After the dead code was removed at the end, PAR_C1L reran C1L with the final source: its stdout and rows file are byte-identical to C1L's.

---

## Step 1: the entity in the token

**The idea.** A1c found that no tag names the question's entity: nothing points the copy expert at it. So E1's entity goes into the gate-side token, in the training don't-know turns and at answer time, and copy carries it into the message and the tag.

**One design point first.** The level (`low`, `high`) must stay the last bytes before the answer, or the trie (4 bytes of context) cannot follow it. So the entity goes in its own token before the level: `<|entity="Napoleon"|><|fam=low|>`. A don't-know turn carries its own entity; a Dolly, dev or test question carries E1's.

| Run | Frames before `<\|fam=…\|>` | Tag field order | Why |
|---|---|---|---|
| P0 | none | DISCOVERY §3.5 | A1c run D′, for parity |
| E1a | `entity="{E}"` | §3.5 (`entity`, `relation`, `gap`) | The tag's own bytes `entity="` precede the entity in the token, so copy matches 8 bytes there |
| E1b | `entity="{E}"` | entity last | After the entity the token continues `"\|>`; with the entity last, so does the tag |
| E1c | `about {E}` and `entity="{E}"` | entity last | Adds a frame for the message, "I have nothing about X" |
| E1d | `ent="{E}"` | entity last | The brief's example key; it shares only `t="` with the tag |

All: 20% don't-know turns, `ent_title`, outcome labels, `--copy`; dev, chat form. Tags are counted from the trie + InfiniGram models; the trie alone writes "I don't known reason=…" with no `<|unknown`, so its tags never count.

| Run | Abstained (all 4 models) | Precision / recall | Tags: IG / IG + copy | Complete | Tag names the entity | Message names the entity | Holdout BPC trie + IG (Δ vs P0) |
|---|---:|---|---|---|---:|---:|---|
| P0 | 69 | 0.435 / 1.00 | 68 / 68 | 29 / 29 | 0 | 0 | 2.072703 |
| E1a | 67 | 0.448 / 1.00 | 65 / 65 | 30 / 30 | 0 | 0 | 2.074278 (+0.0016) |
| E1b | 67 | 0.448 / 1.00 | 65 / 65 | 30 / 31 | 0 | 0 | 2.074263 (+0.0016) |
| E1c | 67 | 0.448 / 1.00 | 65 / 53 | 29 / 19 | 0 | 0 | 2.076028 (+0.0033) |
| E1d | 67 | 0.448 / 1.00 | 65 / 65 | 30 / 31 | 0 | 0 | 2.074299 (+0.0016) |

- **No tag and no message names the question's entity, in any variant.** Q3, Q4 and Q6 hit; Q5 missed.
- **Why: InfiniGram keys on the entity's last bytes.** With the entity just before the level token, InfiniGram's longest match runs back from the answer through `…es"|><|fam=low|>I don't know. I have nothing about ` into a training don't-know turn whose entity ends the same way, and writes that turn's entity: "1 Ceres" → *Antares*, "first asteroid" → *Groupoid*, "alan parsons project" → *Eden Project*, "character Spider-Man" → *Transport on the Isle of Man*. Copy's lambda at a 6–11-byte match is 0.29–0.36; InfiniGram's at a long match is far higher. The tag then follows the message's entity, again by InfiniGram's long match.
- **Where copy does win, it overshoots.** In E1c it copied the entity into the message and went on to the token's `|>`, which ends the answer: *"I have nothing about character Spider-Man|>"*. A loose count (the words after "nothing about" start with a span of the question) finds 8 of 67 such messages in E1c with copy, 0 elsewhere.
- **Without InfiniGram, copy reaches the question.** In C3 (below), the trie + tag-class + copy model starts 25 of its 50 messages with a span of the question (*"I have nothing about the second Prime Minister of Canada?"*). It copies the question's tail rather than the entity, and the trie still cannot finish a tag.
- **Two questions are lost.** The 2 dev questions with no E1 entity get a bare `<|fam=mid|>`, never seen alone in training once every training token has a frame. They are answered, not abstained (67 against 69). Q2 missed.
- **BPC.** One frame adds 0.0016 on trie + InfiniGram, two frames 0.0033. Q7 hit for one frame, missed for two.

**What this says.** Copying the entity needs a pointer whose weight depends on where the model is (inside the message or the tag), not only on the match length. That is a model change (a context-gated copy weight, or copy lambdas fit on chat text). The frames were deleted (see "Code").

---

## Steps 2 and 3: which entity, and which statistic

**Candidates.** E1's chunker (`pos_tagger tag --chunks`) tags every question, and a short script (Reproduce) writes candidate spans into the parse file's entity column as `a|b|c`, E1's own entity first. `qa` takes, per statistic, the best candidate's value; the token's entity is the candidate with the highest `--fam-stat` value (ties go to the earlier one, so E1's entity wins a tie). Every candidate is a span of the question, and the choice is made by CyphaLM's own counts over its own training bytes. The source title is read only to grade the choice.

| Set | Candidates | Per dev question |
|---|---|---:|
| e1 | E1's entity (A1c) | 0.98 |
| np | E1's entity and every NP chunk | 1.92 |
| npsub | E1's entity and every 1–6-word span inside an NP chunk | 3.76 |
| cap | the longest run of capitalised words ("of", "and", "de" and the like allowed inside) | 0.98 |
| capall | E1's entity and every capitalised run | 1.37 |
| span | E1's entity and every 1–6-word span of the question that neither starts nor ends with a function word | 7.65 |

**Results** (familiarity-only runs, 80–96 s each). Dev AUROC (in-slice above not-in-slice; 2,000-draw bootstrap 95%); in brackets, the paired bootstrap interval of its difference from `ent_bold`; entity accuracy (the chosen candidate equals the in-slice source title, of 100); and the token each statistic gives at A1c's training-side cut (count ≥ 1), read as "abstain on `low` and `mid`":

| Set | `ent_title` AUROC [− bold] | `ent_bold` | `ent_tb` = max(title, bold) [− bold] | Entity = title (best by title; any candidate) | Token `ent_title`: precision / recall | Token `ent_bold` | Token `ent_tb` |
|---|---|---|---|---|---|---|---|
| e1 | 0.795 (0.745–0.847) [−0.010, 0.067] | 0.765 (0.714–0.818) | 0.820 (0.770–0.871) [0.026, 0.087] | 63; 63 | 0.435 / 1.00 | 0.400 / 1.00 | 0.469 / 1.00 |
| np | 0.810 (0.763–0.862) [−0.011, 0.146] | 0.748 (0.661–0.825) | 0.788 (0.703–0.863) [0.016, 0.067] | 66; 66 | 0.455 / 1.00 | 0.429 / 0.80 | 0.500 / 0.80 |
| npsub | 0.845 (0.793–0.894) [−0.081, 0.071] | 0.854 (0.780–0.918) | 0.863 (0.789–0.926) [0.001, 0.021] | 68; 73 | 0.508 / 1.00 | 0.714 / 0.67 | 0.741 / 0.67 |
| cap | 0.840 (0.789–0.891) [0.005, 0.088] | 0.795 (0.742–0.848) | 0.870 (0.821–0.919) [0.041, 0.111] | 74; 74 | 0.500 / 1.00 | 0.435 / 1.00 | 0.556 / 1.00 |
| capall | 0.840 (0.789–0.891) [−0.034, 0.062] | 0.825 (0.772–0.875) | 0.895 (0.850–0.941) [0.037, 0.105] | 72; 74 | 0.500 / 1.00 | 0.476 / 1.00 | 0.612 / 1.00 |
| **span** | **0.900 (0.862–0.940)** [−0.037, 0.104] | 0.869 (0.800–0.926) | 0.876 (0.809–0.931) [0.000, 0.017] | **76**; 82 | **0.600 (0.462–0.724) / 1.00 (0.886–1.00)** | 0.833 (0.608–0.942) / 0.50 (0.332–0.668) | 0.833 / 0.50 |

- **Spans win on both counts.** The best-by-title span equals the source title for 76 in-slice questions (Wilson 0.668–0.833) against E1's 63 (0.532–0.718), and `ent_title`'s AUROC rises from 0.795 to 0.900.
- **More candidates never lifted a not-in-slice question to `high` under the title count** (0 of 30 on every set). Under the bold count they did: with span candidates 15 of 30 not-in-slice questions have some span with a bold lead in the A slice ("War", "capital", "Napoleon"), so its recall halves.
- **What `low` still holds (span, title): 20 in-slice questions.** Most do not name their article's subject: the second and ninth Prime Ministers of Canada, the smallest ocean, the largest island off Ireland, the second city of Denmark, the 45th Vice President, the hero of the Iliad, *Earth in the Balance*. The rest are surface forms: "albatrosses" (title "Albatross"), "Spider-Man" (title "The Amazing Spider-Man"), "Bronson Alcott", and "the Ashes": the title lookup drops a leading "The" from the candidate, so `<title>The Ashes</title>` is missed.
- `ent_lp` (the entity's trie log-probability) falls as candidates grow (span: 0.611): the best-scoring span is usually a common word.

**Confirmed with full runs** (span candidates, outcome labels, copy; dev, chat form). CyphaLM follows the token exactly in all four models:

| Run | Statistic | Dev `low` (in / not) | Abstained | Precision | Recall | Tags (IG) / complete / reason right / names the entity |
|---|---|---|---:|---|---|---|
| T_title | `ent_title` | 20 / 30 | 50 | **0.600** (0.462–0.724) | **1.00** (0.886–1.00) | 49 / 21 / 30 / 0 |
| T_bold | `ent_bold` | 3 / 15 | 18 | 0.833 (0.608–0.942) | 0.50 (0.332–0.668) | 18 / 6 / 15 / 0 |

Holdout BPC is identical to P0's in both (outcome labels make the training text the same; only the dev tokens differ). Q20 and Q21 hit.

---

## Step 4: how the training turns get their token

Three labellings, all with span candidates and `ent_title`. Dev and test always get the statistic's token.

- **stat:** the statistic on every training turn (A1c run D).
- **outcome:** answered turns `high`, don't-know turns `low` (A1c run D′): the token is the gate's decision.
- **matched:** the statistic on every turn, its cut put where the dev base rate of not-in-slice questions (30/130 = 0.231, one number) is the share of training turns below it. Most turns have a title count of 0, so ties were broken by the entity's trie log-probability.

| Run | Training turns: Dolly low / high; don't-know low / high | Dev `low` (in / not) | Trie: precision / recall | Trie + IG: precision / recall | Holdout BPC trie / + IG |
|---|---|---|---|---|---|
| T_title (outcome) | 0 / 15,015; 6,583 / 0 | 20 / 30 | **0.600 / 1.00** | **0.600 / 1.00** | 2.348246 / 2.072703 |
| T_stat | 14,014 / 999 (2 mid); 6,405 / 178 | 20 / 30 | 0.600 / 1.00 | 0.313 (0.199–0.453) / 0.50 | 2.348002 / 2.072572 |
| T_matched | 3,221 / 11,792 (2 mid); 1,917 / 4,666 | 0 / 6 | 1.00 (0.610–1.00) / 0.20 | 0.268 (0.170–0.396) / 0.50 | 2.348238 / 2.072699 |

- **Outcome is the only labelling every model follows.** With the statistic on every turn, 93% of Dolly turns still get `low`. The trie now follows the token anyway, because `high` precedes almost only answers. InfiniGram reads the question too; `low` precedes mostly ordinary answers, so it abstains on half the not-in-slice questions and on 33 in-slice ones.
- **Matching the base rate puts the wrong turns in `low`.** Among subjects with no article, the tie-break by log-probability marks rare-looking Dolly subjects `low` (3,221) and leaves most don't-know entities `high` (4,666 of 6,583). On dev only 6 questions fall below the cut.
- **BPC is the same to 0.0003** in all three.
- Q22 missed (Dolly `low` 93%; the trie follows the token). Q23 missed (trie precision 1.00, recall 0.20).

---

## Step 5: the fact shard's form

Variants of D4's shard, all training text only. `qa` drops BPC-holdout rows at load and exits if any question equals a dev or test question (0 in every variant). No token, no copy; trie + InfiniGram, chat form.

| Run | Shard text (rows kept) | Holdout BPC trie / + IG (Δ vs ×1) | Raw right (covered of 11) | Out of fold at 2% wrong: right / wrong (covered; not covered) | Full-dev gate keeps |
|---|---|---|---|---|---|
| A2 F2 (×1) | 1,488 question rows, 3 phrasings a fact | 2.345941 / 2.072393 | 7 (5) | 2 / 0 (2/0; 0/0) | 3 right |
| S_x2 | the same rows twice: 2,976 | 2.347923 / 2.073715 (+0.0020 / +0.0013) | 7 (5) | **3 / 0** (3/0; 0/0) | 4 right |
| S_x4 | four times: 5,952 | 2.351784 / 2.076255 (+0.0058 / +0.0039) | 7 (5) | 3 / 1 (2/0; 1/1) | 4 right |
| S_more | + 3 more phrasings a fact (1,839 new rows, 0 equal to dev or test): 2,844 | 2.347605 / 2.073461 (+0.0017 / +0.0011) | 7 (5) | 3 / 2 (3/0; 0/2) | 4 right |
| S_sent | + 504 declarative sentences as plain text: 1,992 | 2.346258 / 2.072546 (+0.0003 / +0.0002) | 7 (5) | 2 / 0 (2/0; 0/0) | 3 right |

- **Every variant writes the same 7 right answers.** Two are A0's luck (Alexis Carrel, Montgomery); five come from the shard: Avicenna 980, Al Gore 1948, Alan Kay 1940, Alcuin 804, Acapulco → Guerrero.
- **The variants change only the confidence.** More copies raise the right answers' minimum log-probability, so the gate keeps one more (×2). ×4 and the extra phrasings also lift wrong answers over the threshold.
- **The best differs from ×1 by one question:** 3 of 130 (Wilson 0.8–6.6%) against 2 of 130 (0.4–5.4%). No variant passes LCI.
- **The raw-form shard of F2R** (run R3raw, step 8) gives the same 7 raw right answers in the chat form, but its gate keeps 0 right and 1 wrong.
- Q12 and Q15 hit; Q13, Q14 and Q16 missed (covered raw right stayed 5; ×4 keeps one not-covered answer, Acapulco, whose relation has no template).

---

## Step 6: combined runs

### Class expert and copy together

`qa` used to refuse `--cls` with `--copy`. Now the class expert goes first and copy second, each lambda set fit on the holdout's first half over the distribution it is mixed into. One run prints all four combinations, and `cls_given_copy_lci` gives the class expert's gain with copy present. `--copy-min N` (B4B's request) sets copy's lambdas for matches shorter than N bytes to 0 after fitting; the default 0 leaves every earlier output byte-identical.

**BPC, holdout second half (524,288 bytes):**

| Model | CM8 (no don't-know, no shard; copy m = 8) trie / + IG | C2 (C1 + class shape:6; copy m = 0) trie / + IG | C3 (C1 + class tags:6; copy m = 0) trie / + IG |
|---|---|---|---|
| alone | 2.319989 / 2.071574 | 2.323552 / 2.073536 | 2.323552 / 2.073536 |
| + class | 2.133872 / 1.995847 | 2.135822 / 1.996821 | 2.097948 / 1.974071 |
| + copy | 2.188441 / 1.985451 | 2.119183 / 1.929405 | 2.119183 / 1.929405 |
| + class + copy | 2.029724 / **1.914210** | 1.970792 / **1.863340** | 1.939169 / **1.843656** |
| copy's gain given class (IG; 90%) | 0.0815 [0.0713, 0.0928] | 0.1333 [0.1201, 0.1474] | 0.1302 [0.1171, 0.1442] |
| class's gain given copy (IG; 90%) | 0.0712 [0.0688, 0.0737] | 0.0660 [0.0638, 0.0683] | 0.0857 [0.0832, 0.0882] |

Every gain has P(gain > 0.005) = 1.000 on 256 blocks. The two experts add almost fully. Copy with m = 8 keeps 59% of full copy's gain (0.086 of 0.145 on trie + IG). The tag classes beat shape classes by 0.020 with copy present. The class table costs 46 MB (shape) or 106 MB (tags); peak RSS 714–718 MB (CM8), 478 MB (C2), 576 MB (C3).

**Answers for B4B's rule 3** ("answers do not get worse against B0 / B"). Chat form, dev; in a `--cls` run every model includes the class expert.

| Run | Model | Raw right | Out of fold: right / wrong | Reference (no class expert) |
|---|---|---:|---|---|
| CM8 | trie + IG + class | 2 | 0 / 0 | B0 (A0): 2; 0 / 0 |
| CM8 | trie + IG + class + copy (m 8) | 1 | 0 / 1 | B (A0 + copy, A1c run C1): 1; 0 / 2 |
| CM8F (+ D4 shard) | trie + IG + class | 6 | 2 / 1 | A2 F2: 7; 2 / 0 |
| CM8F | trie + IG + class + copy (m 8) | **2** | 0 / 0 | A2 F2 copy (m 0): 6; 0 / 0 |
| CM8F_nocls (shard + copy m 8) | trie + IG + copy (m 8) | 6 | 0 / 0 | A2 F2 copy (m 0): 6; 0 / 0 |

- **Without the shard, rule 3 holds.** Class + copy (m 8) matches B: 1 raw right, and its gate lets 1 wrong through against B's 2.
- **With the shard, rule 3 fails for class + copy.** Raw right falls from 6 to 2. The model echoes the question: *"Avicenna born?"*, *"Al Gore born?"*, *"Alan Kay born?"*, where A2's copy model wrote *"Avicenna was born in 980."*. After *"Avicenna"*, copy matches 9 bytes of the question (lambda 0.355) and predicts *" born?"*. Without the class expert, InfiniGram's *"was born in"* from the shard keeps enough weight to win; the class expert takes a share of it, and copy wins.
- **m = 8 does not bring back A2's gated answers.** Copy (m 8) alone writes the same 6 right answers as A2's full copy, but its gate still keeps none (it picks `ig_match0` in four folds, as A2's did). Restricting copy to long matches neither causes nor cures that; the class expert causes the echo.
- The same echo shows in C2 and C3 with full copy: the class + copy model writes 4 (shape) and 5 (tags) right answers against C1's 6. Q17 hit; Q18 missed; Q24 hit.

### The token with the shard

| Run | Token training | Class | Abstained: trie / IG models | Precision (trie / IG) | Recall | Raw right (trie + IG / + copy) | Out of fold, trie + IG: right / wrong | Holdout BPC trie / + IG |
|---|---|---|---|---|---|---|---|---|
| C1 | outcome | none | 50 / 50 | 0.600 / 0.600 | 1.00 | 7 / 6 | **0 / 0** | 2.349574 / 2.073536 |
| C2 | outcome | shape:6 | 50 / 50 | 0.600 / 0.600 | 1.00 | 7 / 4 | 0 / 0 | as C1 (class above) |
| C3 | outcome | tags:6 | 50 / 54 | 0.600 / 0.556 | 1.00 | 6 / 5 | 0 / 0 | as C1 (class above) |
| **C1L** | **low only** | none | 50 / 66 | 0.600 / 0.455 (0.340–0.574) | 1.00 | 7 / 6 | **2 / 0** | 2.348543 / 2.072881 |

All: span candidates, `ent_title`, 20% don't-know turns, D4's shard (×1), `--copy`.

- **The token cost the gate its answers (C1).** The shard's 7 right answers are still written, but nothing survives the 2% gate out of fold (A2 F2: 2 / 0). The cause: a `<|fam=high|>` at the end of the prompt lengthens InfiniGram's exact match into Dolly turns, which all end their prompt with the same token. Whole copied Dolly answers become the most confident: the six answers with the highest minimum log-probability are all wrong (*"Who was Ahab's father?"* → *"Gendry is an unacknowledged son of King Robert Baratheon."*, −0.448). Without the token the top three were right (Avicenna −0.726, Al Gore −0.796, Alcuin −0.895). Q25 missed.
- **A token only on `low` fixes it (C1L).** Don't-know turns and `low` questions get `<|fam=low|>`; answered turns and `high` questions get no token, so their prompts are A2's. The trie still abstains on exactly the 50 `low` questions. Trie + InfiniGram abstains on all 30 not-in-slice questions and on 36 in-slice ones (precision 0.455). The gate keeps **2 right, 0 wrong** out of fold again (`min_lp`, all five folds; the full-dev gate `min_lp ≥ −0.8954` keeps 3 right). Q29: the trie did not write the token itself (missed), the rest hit.
- **The class experts add no answers.** C2 and C3 keep C1's abstention, and the tag classes make the InfiniGram models abstain on 4 more in-slice questions. Q26 missed on the copy model (4 against 6) and on BPC (1.863, better than 1.90–1.93). Q27 hit: no tag names the entity.

---

## Step 7: the Cypha-classifier gate against qa's own gate

`cypha_gate` (`/root/cb_ecs/cypha_gate`) learns "right" against "wrong" from five of CyphaLM's confidence features (`mean_lp`, `min_lp`, `entropy0`, `ig_n0`, `question_lp`) with the same 5-fold CV and the same 2% target; `qa`'s gate is one threshold on one feature. Chat form, out of fold, right / wrong (the rest abstained); three seeds:

| Run, model | `qa` gate | `cypha_gate` seed 1 / 2 / 3 |
|---|---|---|
| S_x2, trie + IG | 3 / 0 | 4 / 0, 2 / 1, 4 / 1 |
| S_x4, trie + IG | 3 / 1 | 4 / 2, 2 / 0, 4 / 3 |
| S_more, trie + IG | 3 / 2 | 1 / 0, 0 / 0, 2 / 2 |
| S_sent, trie + IG | 2 / 0 | 1 / 2, 1 / 1, 3 / 2 |
| CM8F, trie + IG + class | 2 / 1 | 3 / 4, 2 / 3, 3 / 4 |
| R3raw (chat), trie + IG | 0 / 1 | 1 / 2, 1 / 2, 3 / 3 |
| C1, trie + IG | 0 / 0 | 3 / 3, 2 / 1, 3 / 2 |
| C2, trie + IG + class | 0 / 0 | 4 / 2, 2 / 1, 3 / 1 |
| C3, trie + IG + class | 0 / 0 | 1 / 2, 0 / 2, 1 / 2 |
| C1L, trie + IG | 2 / 0 | 1 / 2, 1 / 2, 3 / 2 |
| P0, CM8 and the copy models of CM8, CM8F, R3raw, C1–C3, C1L | 0 / 0–1 | 0 / 0–2 |

- **Summed over the ten configurations above:** `qa`'s gate keeps 15 right and 5 wrong; `cypha_gate` keeps 21.7 right and 18.0 wrong (mean of three seeds). Its wrong rate out of fold is 1.4% of all questions against `qa`'s 0.4%, and it breaks the 2% target (3 or more wrong of 130) in 6 of 30 seed runs; `qa`'s gate never does here.
- **Where one feature cannot separate, the classifier helps.** On C1, C2 and C3 (the token confounds `min_lp`), `qa`'s gate keeps nothing and `cypha_gate` keeps 1–4 right with 1–3 wrong.
- **It depends on the seed.** S_more: 0 to 2 right; R3raw: 1 to 3.
- Q28 missed: `cypha_gate` sometimes keeps more right answers (S_x2 seed 1: 4 / 0 against 3 / 0). But it lets through 3.6 times as many wrong ones.

---

## Step 8: R3 with F2R's raw-byte frames

F2R ([report](F2R_RAW_BYTE_PATTERNS.md)) learned the patterns on the raw bytes CyphaLM reads, and wrote each one's frame in a 12th column, markup kept: `'''[S]''' (born [[[O]]]`, `[S] was born in [[[O]]]`. Its raw-form shard writes each fact in its frame's bytes ("Albert Schweitzer was born in [[Kaysersberg]].").

**Two loader changes.** `--r3` now reads column 12 when it is there and fills `[S]` and `[O]` in place (it split column 2 at spaces, which cannot fill `'''[S]'''` or `[[[O]]]`). A frame continuation's candidate also stops at `]]` when the frame closes its object with `]]`. With cleaned frames both are no-ops (the selftest checks the new stop).

**Run R3raw:** model B (A0 + copy), the raw frames (`/root/work/f2raw/patterns_raw.tsv`), the raw-form shard (`/root/work/f2raw/shard_facts_raw.tsv`: 1,494 rows kept, 516 holdout rows dropped, leakage 0), E1's parse. Against A2/A3's run F2 (D4's cleaned frames and D4's shard):

| | A2/A3 run F2 | R3raw |
|---|---|---|
| R3 runs on | 33 dev questions | 29 (author 2, birth date 6, birth place 8, capital 7, death date 1, director 2, inventor 2, language 1) |
| Raw right / wrong | 3 / 28 | **5 / 24** (selective accuracy 0.172, Wilson 0.076–0.345) |
| Right answers | Anguilla → The Valley, Schweitzer → Kaysersberg, Alcuin → 804 | Angola → Portuguese language, Schweitzer → Kaysersberg, Markov → Ryazan, Al Gore → 1948, Alcuin → 804 |
| Out of fold at 2% wrong | 0 right, 1 wrong | **2 right, 1 wrong** (coverage 2.3%; wrong 0.8%, Wilson upper 4.2%) |
| Gate on all of dev | `min_lp ≥ −0.632`, 2 right | `ig_match_at_object ≥ 21`, 4 right |
| Holdout BPC trie / + IG | 2.345941 / 2.072393 | 2.345416 / 2.071965 |

- **The raw frames meet CyphaLM's bytes.** The right answers come with long InfiniGram matches at the object (13–73 bytes): the frame's prefix with the subject occurs in the training text or the raw shard, and CyphaLM continues it. The CV gate picks that feature (`ig_match_at_object`) in four folds.
- **Junk frames remain**, such as `capital: [S] are [[[O]]]` and `author: '''''[S]''''' is a [[science fiction]] novel by [[[O]]]`. They are F2's data.
- Q19 hit.

---

## Frozen test (200 questions, read once)

Two configurations are recommended as defaults, and each was read once on the frozen test. The test questions were parsed (E1 and its span candidates) the same way as dev; the gates are the ones chosen on all of dev. Both `--test` runs re-print every dev line, and they are identical to the dev-only runs C1L and R3raw.

**C1L: the don't-know default with A2's shard** (span candidates, `ent_title`, the low-only token, 20% don't-know turns, D4's shard, copy models alongside):

| Model (chat form) | Raw R / A / W | Abstained on not-in-slice | Precision (Wilson 95%) | Recall (Wilson 95%) | In-slice abstained | Gated (2% wrong) R / A / W |
|---|---|---:|---|---|---:|---|
| trie | 0 / 74 / 126 | 47 / 50 | **0.635** (0.521–0.736) | **0.94** (0.838–0.979) | 27 / 150 | 0 / 200 / 0 |
| trie + copy | 1 / 74 / 125 | 47 / 50 | 0.635 | 0.94 | 27 / 150 | 0 / 200 / 0 |
| trie + IG | 2 / 97 / 101 | 48 / 50 | 0.495 (0.397–0.593) | 0.96 (0.865–0.989) | 49 / 150 | 0 / 200 / 0 |
| trie + IG + copy | 2 / 97 / 101 | 48 / 50 | 0.495 | 0.96 | 49 / 150 | 0 / 200 / 0 |

- **The token transfers, and better than A1c's.** A1c's test: 132 abstentions, precision 0.371, recall 0.98. Now 74 abstentions, precision 0.635, recall 0.94. The span candidates take 56 in-slice test questions out of `low` (83 → 27) and lose 2 not-in-slice ones (49 → 47 for the trie).
- **Tags** (trie + IG): 72 written, 28 complete, reason right on 45, none naming the entity.
- **No answer survives the gate** (wrong rate 0, Wilson upper 1.9%). The raw right answers are A0's two (Alaska, a stored Dolly answer; Apollo 8, luck) and one echo with copy (*"…the ampere named after?"*). The shard covers only 3 test questions.
- TQ1 hit.

**R3raw: R3 with F2R's raw frames and raw-form shard** (model B):

| System | Raw R / A / W | Gated R / A / W | Coverage | Selective acc. (Wilson lo) | Wrong rate (Wilson hi) |
|---|---|---|---:|---|---|
| R3 (A2/A3's test, cleaned frames + D4 shard) | 2 / 178 / 20 | 1 / 199 / 0 | 0.5% | 1 / 1 (0.21) | 0.0% (1.9%) |
| **R3raw** | 2 / 180 / 18 | **1 / 199 / 0** | 0.5% | 1 / 1 (0.21) | 0.0% (1.9%) |
| chat / trie + IG (raw-form shard) | 2 / 0 / 198 | 0 / 200 / 0 | 0.0% | n/a | 0.0% (1.9%) |

- R3 runs on 20 test questions (capital 7, author 6, inventor 3, birth place 2, birth date 1, director 1). Its two right answers are A2/A3's: Warhol → *Pennsylvania* (kept by the gate; the state, not the town) and Kurosawa → *1910* (not kept).
- **The test cannot tell the two frame sets apart:** one gated answer each. TQ2 hit.

---

## Decisions

| Question | Decision | Evidence |
|---|---|---|
| **Copy as a default** | **Not for answering.** It stays an option (`--copy`, `--copy-min`). For the language model alone it is B4B's call | Every BPC gain passes LCI (copy alone −0.14 on trie + IG, m = 8 −0.086, on top of the class expert −0.08 to −0.13). But in answering, the copy models write the same or fewer right answers, and with the shard their gate keeps none out of fold in every run (A2 F2, CM8F_nocls, C1, C1L). With the class expert, copy echoes the question and loses 2–4 answers (CM8F, C2, C3). It never names the entity in a tag (step 1) |
| **Entity in the token** | **No.** `--fam-ent` deleted | 0 of 65 tags and 0 messages name the entity in four frame variants. InfiniGram's long match keys on the entity's last bytes and writes a training entity; copy's weight loses. It also costs the 2 no-entity questions (67 against 69 abstained) and up to +0.0033 BPC |
| **Entity choice** | **The span of the question with the highest title count** (E1's entity first, so it wins ties) | Entity = source title for 76 of 100 in-slice dev questions against E1's 63. `ent_title` AUROC 0.900 (0.862–0.940) against 0.795. Token precision at recall 1.00: dev 0.600 against 0.435; test 0.635 (recall 0.94) against A1c's 0.371 (0.98) |
| **Statistic** | **`ent_title`** | On span candidates the AUROCs are within each other's intervals (title 0.900, bold 0.869; paired difference [−0.037, 0.104]). By the tie rule the XML-free `ent_bold` would be preferred. But as the token, at the training-side cut, `ent_bold` halves recall: 0.50 (0.332–0.668) against 1.00 (0.886–1.00), intervals apart; its precision is higher, 0.833 against 0.600, intervals overlapping. The token is what is used, so the title count stays. `ent_tb` is not better than either on span candidates. Flagged for Odin |
| **Token training** | **Outcome labels, with the token on `low` only (`--fam-train low`)** | Outcome is the only labelling every model follows (precision 0.600, recall 1.00 in all four). The statistic on every turn: trie + IG 0.313 / 0.50. The matched cut: trie 1.00 / 0.20, trie + IG 0.268 / 0.50. With the shard, a `high` token makes copied Dolly answers the most confident and the gate keeps nothing (C1: 0 / 0). The low-only token keeps the trie's 0.600 / 1.00 and the gate's 2 right, 0 wrong (C1L); trie + IG's precision falls to 0.455. Flagged for Odin |
| **Shard form** | **D4's shard as it is (×1)** | All four variants write the same 7 right answers. Out of fold: ×1 2 / 0, ×2 3 / 0, ×4 3 / 1, more phrasings 3 / 2, sentences 2 / 0. The best differs by one question, and nothing passes LCI. ×2 costs +0.0013 BPC |
| **Gate** | **`qa`'s threshold gate stays.** The Cypha classifier stays a measurement (DISCOVERY §8a) | Over ten configurations: `qa` 15 right, 5 wrong; `cypha_gate` 21.7 right, 18.0 wrong (mean of 3 seeds), over the 2% target in 6 of 30 seed runs, and seed-dependent. It does help where one feature cannot separate (C1–C3: 1–4 right where `qa` keeps none) |
| **R3 frames** | **F2R's raw frames with the raw-form shard, when R3 runs.** R3 stays an option | Dev: 5 raw right and 2 right / 1 wrong out of fold, against 3 and 0 / 1. Test: 1 right, 0 wrong for both. Its raw-form shard is worse in the chat form (gate 0 / 1 against 2 / 0), so A2's chat answering keeps D4's shard |
| **Class expert in answering** | **Not a default for answering** | It adds no right answer (C2, C3, CM8F). The tag classes make the InfiniGram models abstain on 4 more in-slice questions. With copy, it causes the echo. Its BPC gain is large and passes LCI (with copy: −0.066 shape, −0.086 tags), which is B4B's decision |

---

## Predictions, scored

| Prediction | Result | Score |
|---|---|---|
| Q1 P0 reproduces A1c | identical rows, `ABSTAIN`, `FAMSTAT` and BPC lines | **Hit** |
| Q2 69 abstained with the entity in the token | 67: the 2 no-entity questions are answered | **Miss** |
| Q3 E1a tags naming the entity 0–10% | 0 of 65 | **Hit** |
| Q4 E1b 0–15% | 0 of 65; completeness 30–31 (E1a 30) | **Hit** |
| Q5 E1c message 40–80%, tag 30–70% | 0 and 0 (loose message count 8 of 67) | **Miss** |
| Q6 E1d 0–10% | 0 | **Hit** |
| Q7 BPC within 0.003 of P0 | +0.0016 (one frame); +0.0033 (E1c, two frames) | **Hit** for one frame, **miss** for E1c |
| Q8 E1 = title for 40–60 | 63 | **Miss** |
| Q9 +2–8 / +5–15 / +8–20; cap within ±8 | +3 / +5 / +13; cap +11 | **Hit** ×3, **miss** on cap |
| Q10 AUROC 0.80–0.90 best; token 0.45–0.65 at 0.90–1.00 | span 0.900, npsub 0.845; 0.600 / 1.00 and 0.508 / 1.00. But no not-in-slice question moved to `high` under the title count | **Hit** on the numbers |
| Q11 `ent_tb` ±0.03; bold 0.00–0.05 below, interval through 0, every set | `ent_tb` within ±0.03 in 5 of 6 (capall +0.055); bold above title on npsub; cap's interval excludes 0 | **Miss** |
| Q12 ×2 | covered raw 5, out of fold 3 / 0, BPC +0.0013 | **Hit** |
| Q13 ×4 | covered raw 5 (below 6–9) | **Miss** |
| Q14 more phrasings | covered raw 5; out of fold 3 / 2 | **Miss** |
| Q15 sentences | 7 raw, 2 / 0, BPC +0.0002 | **Hit** |
| Q16 no not-covered answer gated | ×4 keeps Acapulco (and one wrong) | **Miss** |
| Q17 CM8 BPC 1.90–1.97, raw 1–3, nothing right gated | 1.914; raw 2 and 1; 0 right (the copy model lets 1 wrong through) | **Hit** |
| Q18 CM8F covered raw 4–7; gates 1–3 right, 0 wrong | copy model raw 2; trie + IG 2 right 1 wrong; copy model 0 | **Miss** |
| Q19 R3raw | 29 questions; 5 raw right; 2 / 1 out of fold | **Hit** |
| Q20 span + title token 0.600 / 1.00 | exactly | **Hit** |
| Q21 span + bold token 0.833 / 0.50 | exactly | **Hit** |
| Q22 statistic on every turn | Dolly `low` 93%; trie 0.600 / 1.00; trie + IG 0.313 | **Miss** |
| Q23 matched cut | trie 1.00 / 0.20 | **Miss** |
| Q24 copy (m 8) without class: covered raw ≥ 4, gate 0–2 | 5 covered; 0 gated | **Hit** |
| Q25 C1 gate 2–3 right | 0 | **Miss** |
| Q26 C2 / C3 within ±1; BPC ranges | copy model 4 against 6; BPC 1.863 / 1.844 below 1.90–1.93 | **Miss** |
| Q27 no tag names the entity | 0 | **Hit** |
| Q28 `cypha_gate` never keeps more right | it does on some seeds (S_x2: 4 / 0) | **Miss** |
| Q29 C1L: trie 0.23–0.45; IG 0.45–0.60; gate 1–2 right, 0 wrong | trie 0.600; IG 0.455; gate 2 / 0 | **Miss** on the trie, **hit** on the rest |
| TQ1 C1L test | recall 0.94 / 0.96; precision 0.635 / 0.495; gate 0 / 0 | **Hit** |
| TQ2 R3raw test | 20 questions; 2 raw right; gate 1 / 0 | **Hit** |
| CYPHALM_ANSWERING §4: abstain precision ≥ 0.8 on not-in-slice | dev 0.600, test 0.635 (Wilson upper 0.736) | **Miss** |

15 hits, 14 misses, 3 split. The misses that matter: no variant puts the entity in the tag (Q5); the token interferes with the gate unless it is on `low` only (Q25); and the class expert makes copy echo the question (Q18, Q26).

---

## Flags for Odin

1. **The low-only token.** Odin allowed the familiarity token gate-side, after `Assistant: `. The recommended form now puts a token only on `low`: answered turns and `high` questions carry none. It is the same gate decision written into the prompt, with less in it. It is what lets the token and A2's gate work together (C1 against C1L).
2. **`ent_title` reads XML `<title>` tags.** By AUROC it ties with the XML-free `ent_bold`, so the agreed tie rule would pick `ent_bold`. I kept `ent_title` because, as the token, `ent_bold` halves recall (0.50 against 1.00 on dev). `ent_bold` on span candidates meets D2 on dev (precision 0.833 ≥ 0.8, recall 0.50 ≥ 0.5); `ent_title` does not (0.600). Odin decides which side of that trade he wants.
3. **The entity in the tag needs a model change.** Copy cannot beat InfiniGram's long match inside a learned template. A copy weight that depends on the context (inside the message or the tag), or copy lambdas fit on chat text, would be the next try. That changes the mixture, so it needs Odin's decision.
4. **Class + copy and the shard (B4B).** With the fact shard, class + copy (m = 8) writes 2 right answers against 6, by echoing the question. Without the shard, rule 3 holds. If both become language-model defaults, A2's answers need the no-class model, or the echo needs a fix.
5. **The title lookup drops a leading "The".** "the Ashes" is looked up as `<title>ashes</title>`, so *The Ashes* (2 dev questions) stays `low`. Counting both forms is a one-line change but a new dev choice; not made here.
6. **D2 still misses.** Abstain precision on not-in-slice is 0.600 (dev) and 0.635 (test) at recall 0.94–1.00. The rest are in-slice questions that do not name their article's subject ("the second Prime Minister of Canada"), which no entity choice can fix.

---

## Code

`native/tools/cyphalm_trie.cpp`: 2,868 → 3,093 lines (43 of them from the SCRIBE merge in `cls`; this work +182). `packed_trie.hpp`, `class_context.hpp` and `infinigram.cpp` unchanged.

**Kept:**
- entity candidates in the parse's entity column (`a|b|c`): per-statistic best, and the token's entity by `--fam-stat`;
- the `ent_tb` statistic; `FAMSTAT`'s paired difference from `ent_bold`; the `ENTITY` and `FAMTOK` lines;
- `--fam-train low`;
- `--cls` with `--copy` (class first, then copy; `cls_given_copy_lci`) and `--copy-min N`;
- `--r3` reading F2R's `frame` column, and the `]]` stop for raw frames;
- the `ABSTAIN` line shared by dev and test, with `entity_eq_title` and `message_entity_in_question`;
- selftest checks: `copy_clip`, `msg_entity`, `names_in`, the `]]` stop (still about 0.1 s; the golden BPC is unchanged).

**Deleted, because the results make them dead** (the code is in snapshot commit 833b0f5):
- `--fam-ent FRAMES` (the entity in the token): 0 of 65 tags in four variants (step 1);
- `--fam-train matched`: worse than outcome on every model (step 4);
- plain-text shard rows (a row with no question): no change on any number (step 5).

---

## Costs

| Item | Value |
|---|---|
| Code | `cyphalm_trie.cpp` +182 lines net (see "Code"); selftest about 0.1 s, golden BPC unchanged |
| `qa` run, no class expert | 411–1,069 s wall time (the spread is the machine's load: 1–3 other jobs), peak RSS 355–380 MB |
| `qa` run with the class expert | 441–768 s; peak RSS 478–718 MB (the tag table is 106 MB, shape 46 MB) |
| `qa --fam-only` | 80–96 s, 210–215 MB |
| Frozen test runs | 501 s (C1L), 461 s (R3raw) |
| Span candidates | E1 chunking of 21,928 questions about 2 s; the script under 1 s; 7.65 candidates per dev question, 14.2 over all questions |
| Familiarity with candidates | the statistics are computed per candidate (suffix-array lookups and one trie pass each); not timed separately, inside the run times above |
| `cypha_gate` | under 1 s per configuration and seed |
| External data | enwik8, Dolly, E1's tagger and model, D4's shard and patterns, F2R's raw frames and raw-form shard |
| Machine | 4 cores shared; the container restarted once (E1c was rerun from scratch) |

---

## Reproduce

Build and selftest as in [A1C_ENTITY_FAMILIARITY_AND_COPY.md](A1C_ENTITY_FAMILIARITY_AND_COPY.md) "Reproduce". `pos_tagger` and `pos_model.bin` are E1's; `$A` holds A1c/A2's files (E1 parses, `dk_mention.tsv`, `a0_rows.tsv` and the question lists); D4's shard and F2R's frames and raw-form shard are where D4 and F2R put them; the tag table is B4's pipeline over wiki train + Dolly.

```bash
E=/root/cdata/enwik8; D=/root/cdata/dolly.jsonl; DEV=bench/eval/t_qa_wiki_dev.tsv; TEST=bench/eval/t_qa_wiki.tsv
A=/root/work/answer; S=/root/work/f2/shard_facts.tsv
P=/root/work/f2raw/patterns_raw.tsv; SR=/root/work/f2raw/shard_facts_raw.tsv; TAGS=/root/work/coord/tags_lex_all.tsv
# Entity candidates (step 2): E1's chunks of every question the runs read, then cands.py (below).
cat $A/dev_q.txt $A/dolly_instr.txt $A/dkm_q.txt > q_devside.txt
pos_tagger tag pos_model.bin --chunks < q_devside.txt > chunks_devside.txt
for m in np npsub cap capall span; do python3 cands.py $m q_devside.txt chunks_devside.txt $A/parse_dev.tsv > parse_dev_$m.tsv; done
# The don't-know rows with the entity last in the tag (E1b-E1d).
sed 's/ entity="\([^"]*\)" relation=\([^ ]*\) gap=\(G-[0-9A-F]*\)|>$/ relation=\2 gap=\3 entity="\1"|>/' $A/dk_mention.tsv > dk_mention_elast.tsv
# Shard variants (step 5): shards.py (below).
python3 shards.py rep 2 $S > shard_x2.tsv; python3 shards.py rep 4 $S > shard_x4.tsv
python3 shards.py more $S $DEV $TEST > shard_more.tsv
python3 shards.py sent $S /root/work/f2/shard_facts_sentences.txt > shard_sent.tsv

Q="./cyphalm_trie qa $E $D --dev $DEV --leak $TEST"
TOK="--dk $A/dk_mention.tsv --dk-ratio 0.20 --fam after --copy --base $A/a0_rows.tsv"
$Q --parse $A/parse_dev.tsv $TOK --fam-stat ent_title --fam-train outcome                 # P0 (= A1c run D')
for m in np npsub cap capall span; do $Q --parse parse_dev_$m.tsv --dk $A/dk_mention.tsv --fam-only; done
$Q --parse $A/parse_dev.tsv --dk $A/dk_mention.tsv --fam-only                              # FO_e1
$Q --parse parse_dev_span.tsv $TOK --fam-stat ent_title --fam-train outcome               # T_title
$Q --parse parse_dev_span.tsv $TOK --fam-stat ent_bold --fam-train outcome                # T_bold
$Q --parse parse_dev_span.tsv $TOK --fam-stat ent_title --fam-train stat                  # T_stat
for v in x2 x4 more; do $Q --parse $A/parse_dev.tsv --shard shard_$v.tsv; done           # S_x2, S_x4, S_more
$Q --parse $A/parse_dev.tsv --cls shape:6 --copy --copy-min 8                             # CM8
$Q --parse $A/parse_dev.tsv --cls shape:6 --copy --copy-min 8 --shard $S                  # CM8F
$Q --parse $A/parse_dev.tsv --copy --copy-min 8 --shard $S                                # CM8F_nocls
$Q --parse $A/parse_dev.tsv --copy --r3 $P --shard $SR                                     # R3raw
C="--parse parse_dev_span.tsv $TOK --fam-stat ent_title --shard $S"
$Q $C --fam-train outcome                                                                  # C1
$Q $C --fam-train outcome --cls shape:6                                                    # C2
$Q $C --fam-train outcome --cls $TAGS:6                                                    # C3
$Q $C --fam-train low                                                                      # C1L, and PAR_C1L (final source)
# Step 7, on any run above with --out ROWS.tsv, seeds 1, 2, 3:
cypha_gate ROWS.tsv --features mean_lp,min_lp,entropy0,ig_n0,question_lp --where form=chat,lm=trie_ig --seed 1
python3 msgent.py ROWS.tsv $DEV       # grading only: messages that start with a span of the question
# The frozen test, once each, at the end (the test questions parsed and chunked as dev):
tail -n +2 $TEST | cut -f1 | cat q_devside.txt - > q_all.txt
pos_tagger tag pos_model.bin --chunks < q_all.txt > chunks_all.txt
python3 cands.py span q_all.txt chunks_all.txt $A/parse_all.tsv > parse_all_span.tsv
./cyphalm_trie qa $E $D --dev $DEV --test $TEST --parse parse_all_span.tsv $TOK --fam-stat ent_title --fam-train low --shard $S
./cyphalm_trie qa $E $D --dev $DEV --test $TEST --parse $A/parse_all.tsv --copy --r3 $P --shard $SR
```

The deleted options are in snapshot commit 833b0f5: `--fam-ent` (E1a: `--fam-ent 'entity="{E}"'` with `dk_mention.tsv`; E1b: the same with `dk_mention_elast.tsv`; E1c: `'about {E};entity="{E}"'`; E1d: `'ent="{E}"'`; each with P0's options), `--fam-train matched` (T_matched: T_title's options) and the plain-text shard rows (S_sent: `--shard shard_sent.tsv`).

New stdout lines: `FAMSTAT … minus_ent_bold` (paired difference), `ENTITY` (entity accuracy, grading only), `FAMTOK` (the token each statistic gives, with Wilson intervals), `holdout2_trie[_ig]_cls_copy_bpc`, `copy_lci …_cls` and `cls_given_copy_lci` (class + copy), and `ABSTAIN` / `TEST_ABSTAIN` with `entity_eq_title` and `message_entity_in_question`. Run outputs, rows and logs are in `/root/work/qa2/runs/`.

`cands.py`:

```python
# Entity candidates for `cyphalm_trie qa --parse` (A1d). Reads questions (one per line), E1's chunker output for them
# (`pos_tagger tag --chunks`: a tag line and a chunk line per question) and E1's question parse (9 columns). Writes the
# parse with its entity column replaced by candidates joined with '|', E1's own entity first. A parse, not an answer:
# every candidate is a span of the question.
#   MODE np     E1 + every NP chunk
#   MODE npsub  E1 + every span of 1-6 words inside an NP chunk (leading determiners and wh-words dropped)
#   MODE cap    the longest capitalised run (E1 when there is none)
#   MODE capall E1 + every capitalised run
#   MODE span   E1 + every span of 1-6 words of the question that neither starts nor ends with a function word
import re, sys
mode, qf, cf, pf = sys.argv[1:5]
qs = [l.rstrip('\n') for l in open(qf, encoding='utf-8', errors='replace')]
ch = [l.rstrip('\n') for l in open(cf, encoding='utf-8', errors='replace')]
assert len(ch) == 2 * len(qs), (len(ch), len(qs))
parse = {}
for l in open(pf, encoding='utf-8', errors='replace'):
    f = l.rstrip('\n').split('\t')
    if len(f) >= 9 and f[0] != 'question': parse.setdefault(f[0], f)
FUNC = set('who whom whose what which when where why how is are was were do does did can could will would shall should '
           'may might has have had be been the a an this that these those in on at of for to from by with about into '
           'as and or but if than then there their his her its our your my it he she they we you i me him them us '
           'some any all each every no not many much more most one'.split())
CONNECT = {'of', 'the', 'and', 'de', 'von', 'van', 'na', 'la', 'le', 'du', 'del', 'da', 'der', 'den', 'al', 'el', ','}
def offsets(q, toks):  # each tagger token's (start, end) in the question
    out, p = [], 0
    for t in toks:
        i = q.find(t, p)
        if i < 0: return None
        out.append((i, i + len(t))); p = i + len(t)
    return out
def caps(q, toks, off):
    runs, cur = [], []
    for k, t in enumerate(toks):
        cap = t[:1].isupper() and t.lower() not in FUNC
        if cap: cur.append(k); continue
        if cur and t.lower() in CONNECT and k + 1 < len(toks) and toks[k+1][:1].isupper() and toks[k+1].lower() not in FUNC:
            cur.append(k); continue
        if cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    return [q[off[r[0]][0]:off[r[-1]][1]] for r in runs]
n_out = 0
print('question\twh\twh_phrase\twh_head\taux\tverb\tfocus\tentity\tentity_via')
for i, q in enumerate(qs):
    tagl, chunkl = ch[2*i], ch[2*i+1]
    toks = [w.rsplit('/', 1)[0] for w in tagl.split(' ') if '/' in w]
    off = offsets(q, toks)
    row = parse.get(q, [q] + ['-'] * 8)
    e1 = '' if row[7] == '-' else row[7]
    c = [e1] if e1 else []
    if off is None:
        cand = []
    elif mode in ('np', 'npsub'):
        cand = []
        # chunk words in order, matched to the tag line's tokens (the chunk line repeats them)
        k = 0
        for m in re.finditer(r'\[NP ([^\[\]]+)\]', chunkl):
            words = m.group(1).split(' ')
            # locate this chunk's words in the token list from position k
            for s in range(k, len(toks) - len(words) + 1):
                if toks[s:s+len(words)] == words: break
            else: continue
            k = s + len(words)
            idx = list(range(s, s + len(words)))
            while idx and toks[idx[0]].lower() in FUNC: idx = idx[1:]
            if not idx: continue
            if mode == 'np': cand.append(q[off[idx[0]][0]:off[idx[-1]][1]])
            else:
                for a in range(len(idx)):
                    for b in range(a, min(len(idx), a + 6)):
                        if toks[idx[a]].lower() in FUNC or toks[idx[b]].lower() in FUNC: continue
                        cand.append(q[off[idx[a]][0]:off[idx[b]][1]])
    elif mode == 'span':
        ok = [t.lower() not in FUNC and any(ch.isalnum() for ch in t) for t in toks]
        cand = [q[off[a][0]:off[b][1]] for a in range(len(toks)) for b in range(a, min(len(toks), a + 6))
                if ok[a] and ok[b] and all(any(ch.isalnum() for ch in toks[k]) or toks[k] in ("'s", '-', ',') for k in range(a, b + 1))]
    else:
        cand = caps(q, toks, off)
        if mode == 'cap':
            cand = [max(cand, key=len)] if cand else []
            c = cand if cand else c
            cand = []
    for x in cand:
        if x and x not in c and '|' not in x: c.append(x)
    if not c: row[7] = '-'
    else: row[7] = '|'.join(c); n_out += 1
    row[0] = q
    print('\t'.join(row))
print('questions', len(qs), 'with_entity', n_out, file=sys.stderr)
```

`shards.py`:

```python
# A2 shard variants for `cyphalm_trie qa --shard` (A1d). Training text only; strings are compared only for the
# leakage check (no generated question may equal a dev or test question after normalisation; qa checks again at load).
#   shards.py rep N SHARD                 the rows N times
#   shards.py more SHARD DEV TEST         + 3 more phrasings per fact (templates below), leaking ones dropped
#   shards.py sent SHARD SENTENCES        + each declarative sentence as a plain-text row (empty question); its source
#                                         is its fact's, so qa drops the BPC-holdout ones
import re, sys
def rows(p):
    L = [l.rstrip('\n').split('\t') for l in open(p, encoding='utf-8')]
    return L[0], L[1:]
def norm(s):
    return ' '.join(re.sub(r'[^0-9a-z]+', ' ', s.lower()).split())
MORE = {
    'birth_date': ['What day was {E} born?', 'When is the birthday of {E}?', '{E} was born on what date?'],
    'birth_year': ['In which year was {E} born?', 'What year was {E} born in?', '{E} was born in which year?'],
    'birth_place': ['Where is {E} from?', 'What is the birthplace of {E}?', '{E} was born where?'],
    'death_date': ['What day did {E} die?', '{E} died on what date?', 'When did {E} pass away?'],
    'death_year': ['In which year did {E} die?', 'What year did {E} die in?', '{E} died in which year?'],
    'death_place': ['Where did {E} pass away?', 'What is the place where {E} died?', '{E} died where?'],
    'capital': ['What city is the capital of {E}?', 'Name the capital of {E}.', 'The capital of {E} is which city?'],
    'largest_city': ['Which city is the biggest in {E}?', 'Name the largest city in {E}.', 'What is the biggest city of {E}?'],
    'located_in': ['In what region is {E}?', 'Where is {E} situated?', 'Where can {E} be found?'],
    'occupation': ['What was the job of {E}?', 'What did {E} work as?', 'What profession did {E} have?'],
    'currency': ['Which currency does {E} use?', 'What money is used in {E}?', 'What is the money of {E}?'],
    'class': ['Which class is {E} in?', '{E} belongs to which class?', 'What class is {E} in?'],
    'order': ['Which order is {E} in?', '{E} belongs to which order?', 'What order is {E} in?'],
    'family': ['Which family is {E} in?', '{E} belongs to which family?', 'What family is {E} in?'],
    'genre': ['Which genre is {E}?', '{E} belongs to which genre?', 'What kind of genre is {E}?'],
    'creator': ['Who is the creator of {E}?', '{E} was created by whom?', 'Who invented {E}?'],
    'composer': ['Who wrote the score of {E}?', '{E} was composed by whom?', 'Who is the composer of {E}?'],
    'author': ['Who is the writer of {E}?', '{E} was written by whom?', 'Who authored {E}?'],
    'director': ['Who is the director of {E}?', '{E} was directed by whom?', 'Who made the film {E}?'],
    'founded': ['In which year was {E} founded?', 'What year was {E} established?', '{E} was founded in which year?'],
    'publisher': ['Who is {E} published by?', '{E} was published by whom?', 'Which publisher released {E}?'],
    'developer': ['Who is {E} developed by?', '{E} was developed by whom?', 'Which company made {E}?'],
    'owner': ['Who is {E} owned by now?', '{E} is owned by whom?', 'Which company owns {E}?'],
}
cmd = sys.argv[1]
if cmd == 'rep':
    h, R = rows(sys.argv[3])
    print('\t'.join(h))
    for _ in range(int(sys.argv[2])):
        for r in R: print('\t'.join(r))
elif cmd == 'more':
    h, R = rows(sys.argv[2])
    ev = set()
    for p in sys.argv[3:5]:
        for l in list(open(p, encoding='utf-8'))[1:]: ev.add(norm(l.split('\t')[0]))
    have = {norm(r[0]) for r in R}
    facts, out, leak = [], [], 0
    seen = set()
    for q, e, rel, src, resp in R:
        if (e, rel, resp) in seen: continue
        seen.add((e, rel, resp))
        for t in MORE.get(rel, []):
            nq = t.replace('{E}', e)
            if norm(nq) in ev: leak += 1; continue
            if norm(nq) in have: continue
            have.add(norm(nq)); out.append([nq, e, rel, src, resp])
    print('\t'.join(h))
    for r in R + out: print('\t'.join(r))
    print('more: facts', len(seen), 'added', len(out), 'dropped_equal_to_dev_or_test', leak, file=sys.stderr)
elif cmd == 'sent':
    h, R = rows(sys.argv[2])
    src = {}
    for q, e, rel, s, resp in R:
        if resp not in src or src[resp][2].endswith('_holdout'): src[resp] = (e, rel, s)
    print('\t'.join(h))
    for r in R: print('\t'.join(r))
    n = 0
    for l in open(sys.argv[3], encoding='utf-8'):
        s = l.rstrip('\n')
        if s in src:
            e, rel, so = src[s]; print('\t'.join(['', e, rel, so, s])); n += 1
    print('sent: sentences', n, file=sys.stderr)
```

`msgent.py` (grading only):

```python
# Grading only: share of don't-know answers whose words after "nothing about " (up to " in what", "?" or 6 words)
# start with a span of 2+ letters-words of the question (a leading "the" dropped). Usage: msgent.py ROWS DEV
import re, sys
q = [l.split('\t')[0] for l in list(open(sys.argv[2], encoding='utf-8'))[1:]]
norm = lambda s: ' '.join(re.sub(r'[^0-9a-z]+', ' ', s.lower()).split())
res = {}
for l in open(sys.argv[1], encoding='utf-8'):
    f = l.rstrip('\n').split('\t')
    if f[0] != 'dev' or f[3] != 'chat' or "nothing about " not in f[12]: continue
    a = f[12].split('nothing about ', 1)[1]
    a = re.split(r' in what|\?|\.', a)[0]
    w = norm(a).split()
    if w and w[0] == 'the': w = w[1:]
    w = w[:6]
    nq = ' ' + norm(q[int(f[1])]) + ' '
    k = 0
    while k < len(w) and (' ' + ' '.join(w[:k+1]) + ' ') in nq: k += 1
    n, hit = res.get(f[4], (0, 0))
    res[f[4]] = (n + 1, hit + (k >= 1 and len(' '.join(w[:k])) >= 3))
for m, (n, h) in sorted(res.items()): print(sys.argv[1].split('/')[-1], m, 'messages', n, 'start_with_question_span', h)
```

---

## ANS: three answering questions, decided by measurement (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · HANDOFF §5.2–§5.4 · decided by measurement under PLAN ledger 55, open to Odin's veto

Odin said "Do all". Flags 1–3 above (HANDOFF §5.2–§5.4) are decided here, each by a rule written before its run.

**The configuration (C1L under today's defaults).** Span candidates, `ent_title`, the low-only token, 20% don't-know turns, D4's shard. The default experts (class, similar-context, copy) are on and off inside answers (B4c, A4d). Its dev read on main c74d2a3 (`/root/work/integ6/runs/VQ`):
- trie: abstains on 50, precision 0.600, recall 1.00;
- trie + IG: abstains on 66, precision 0.455, recall 1.00; the gate keeps 2 right, 0 wrong out of fold (full dev: `min_lp ≥ −0.8954`, 3 right);
- trie + IG tags: 49, 21 complete, 30 with the right reason, 0 naming the question's entity;
- holdout BPC with the defaults (trie + IG + class + similar-context + copy): 1.789464.

These equal A1d's C1L lines.

### 1. The low-only token (§5.2)

- **The evidence is A1d's**, registered (Q25, Q29) and read: with a token on every turn (C1) the gate keeps 0 right; with the token on `low` only (C1L) it keeps 2 right, 0 wrong. C1L's frozen test was read once (TQ1).
- **It still holds today.** The class, similar-context and copy experts are off inside answers, so every answer byte comes from the trie and InfiniGram, as in A1d. VQ reproduces A1d's C1L dev lines.
- **Rule 1.** If VQ's `ABSTAIN`, `CV` and `FULLDEV_GATE` lines for both chat models equal A1d's C1L, the low-only token (`--fam-train low`) is the answering configuration's token form. No new run. The flag's default (`stat`) stays, so older commands keep their meaning.

### 2. `ent_title` against `ent_bold` (§5.3)

**What each reads.** Neither reads anything in the question except its spans.
- `ent_title` counts `<title>span</title>` in the lowered training text (wiki train + Dolly). The span must be an article's title.
- `ent_bold` counts `'''span`. The span must open some article's bold lead.
- A real user question has no tags. So the useful question is: how often does each call an ordinary question unfamiliar?

**New reads.**
- `FAMREAD` (report only): for each dev question, the best span under each statistic and its count.
- `FAMDOLLY` (report only): the token each statistic gives the 15,015 Dolly instructions, which are real user questions, with span candidates, at the same training-side cut.
- Dev run **C1L_bold**: C1L with `--fam-stat ent_bold`.

**Rule 2.** Switch to `ent_bold` only if all three hold on dev:
- (a) its token meets D2 (abstain precision ≥ 0.8 with recall ≥ 0.5 on not-in-slice questions, `FAMTOK`) and `ent_title`'s does not;
- (b) the trie + IG model's gate keeps at least as many right answers out of fold (2) and no more wrong ones (0) as C1L's;
- (c) it marks no larger share of Dolly instructions `low` than `ent_title` does.

Then the frozen test is read once with `ent_bold`. Otherwise `ent_title` stays.

### 3. The entity in the tag (§5.4)

**Design.**
- **The entity frame in the low token** (A1d's `--fam-ent`, restored): a low token becomes `<|entity="E"|><|fam=low|>`. E is the don't-know row's own entity in training, and the chosen span at answer time. Answered turns and `high` questions still carry nothing.
- **The tag with the entity last** (A1d's E1b order, `…relation=… gap=… entity="E"|>`). After the entity, the frame and the tag both continue `"|>`.
- **Copy with its own weight inside a don't-know answer** (`--copy-dk`). Once an answer begins "I don't know" (where decode already switches to its longer limit), the copy expert is on in every model. Its lambdas are fit only on don't-know turns held out of training: the rows after the n_dk that training uses.
- **Nothing else changes.** Ordinary answers keep copy off (B4c). Wiki text keeps the default copy lambdas. Inside a don't-know answer, the no-repeat rule counts only the answer's own bytes, as it does for the copy models.
- So copy can follow the frame from ` entity="` into the tag.

Dev run **DKC**.

**Rule 3.** Adopt into the answering configuration only if all five hold on dev, for the trie + IG model (chat form):
- (a) at least half of the tags name an entity that is a span of the question (`entity_in_question`), Wilson lower bound ≥ 0.30;
- (b) the share of complete tags is not below VQ's (21 of 49) by more than 0.05;
- (c) abstention precision and recall of both chat models are within one question of VQ's;
- (d) the gate keeps at least 2 right and 0 wrong out of fold;
- (e) holdout BPC (defaults, trie + IG) is within +0.003 of 1.789464.

Then the frozen test is read once with DKC. Otherwise the code stays behind its flags, and the tag stays as it is.

**The frozen test.** At most one read per adopted change, logged below.

### Predictions (registered with the rules)

| Id | Prediction |
|---|---|
| AN1 | `FAMDOLLY`: `ent_title` marks at least 85% of Dolly instructions `low`; `ent_bold` 30–70% |
| AN2 | C1L_bold: the trie abstains on exactly the 18 `low` questions (precision 0.833, recall 0.50); trie + IG on 20–35, precision 0.45–0.75 |
| AN3 | C1L_bold's gate out of fold: 1–2 right, 0–1 wrong |
| AN4 | Rule 2 fails on (b): one of the 15 not-in-slice questions it now answers passes the gate, or a right answer is lost. `ent_title` stays |
| AN5 | DKC's don't-know copy lambdas: at least 0.8 for matches of 8 bytes or more, at most 0.3 below 6 bytes |
| AN6 | DKC: 50–90% of the trie + IG tags name the question's entity (VQ: 0) |
| AN7 | DKC: abstentions within ±2 of VQ on both chat models; the gate 2 right, 0 wrong |
| AN8 | DKC: holdout BPC within +0.002 of 1.789464 |
| AN9 | DKC: the message names a span of the question in under 10% of abstentions (no frame points copy at "nothing about") |

### Results (2026-10-29)

**Tool.** `cyphalm_trie` at commit 4e59940, which adds `--fam-ent`, `--copy-dk` and `--fam-read`; nothing else changed. Runs are under `/root/work/ans/runs` (scripts `fr.sh`, `par.sh`, `bold.sh`, `dkc.sh`, `tbold.sh`, `tdkc.sh`). They ran under `nice -n 5` on 4 cores shared with about twenty other jobs (load 25–30). Wall times were 40–85 minutes; peak RSS 184.5–184.8 MB in every run.

**Parity.** PAR is today's C1L with the new binary and no new flag.
- Its rows file is byte-identical to main c74d2a3's VQ.
- Its stdout differs only in one timing line (`SIM_LAT`).
- Holdout BPC is unchanged: 2.072881 (trie + IG), 1.789464 (defaults).
- Answers, BPC and every abstention line equal VQ's, so SCRIBE v2 and surface S are untouched too (`sample` is not changed).

#### 1. The low-only token (§5.2)

VQ's `RAW`, `CV`, `FULLDEV_GATE` and `ABSTAIN` lines for both chat models are identical to A1d's C1L (`/root/work/qa2/runs/C1L.out`).
- trie: abstains on 50, precision 0.600, recall 1.00;
- trie + IG: abstains on 66, precision 0.455; the gate keeps 2 right, 0 wrong out of fold.

**Rule 1 holds.** The low-only token (`--fam-train low`) is the answering configuration's token form. No new run was needed.

#### 2. `ent_title` against `ent_bold` (§5.3)

**What each reads (FR, `--fam-only --fam-read`).** Its familiarity lines equal VQ's byte for byte.

| | `ent_title` | `ent_bold` |
|---|---|---|
| What it counts | `<title>span</title>` in the lowered training text | `'''span`: some article's bold lead starts with the span |
| Dolly instructions marked `low` (real user questions, at the training-side cut) | **14,014 of 15,015 = 0.933** [0.929, 0.937] | **4,980 = 0.332** [0.324, 0.339] |
| Typical best span, not-in-slice dev questions | the entity, count 0 ("Mongolia", "Winston Churchill", "Google") | a common word that opens some bold lead: "born", "capital", "film", "tallest", count 1 |
| Typical best span, in-slice | the article title ("Miss Marple", "alan parsons") | a leading word ("Miss", "alan", "1") |

- **A real question has no title tag, but that is not what `ent_title` reads.** Both statistics read only the question's spans, counted in CyphaLM's training text.
- **The difference is what they count.** A span is `high` under `ent_title` only if it is an article's title. So any question whose subject has no article of that exact name is `low`, and that is 93% of Dolly's real instructions. `ent_bold` is `high` for any span that opens a bold lead somewhere. That is often a common word, which is why it misses half of the not-in-slice questions.

**C1L_bold (dev).** C1L with `--fam-stat ent_bold`.

| Model (chat) | Abstained | Precision (not-in-slice) | Recall | Gate out of fold: right / wrong | Raw right |
|---|---:|---|---|---|---:|
| trie, `ent_title` (VQ) | 50 | 0.600 [0.462, 0.724] | 1.00 [0.886, 1.00] | 0 / 0 | 0 |
| trie, `ent_bold` | 18 | **0.833** [0.608, 0.942] | 0.50 [0.332, 0.668] | 0 / 0 | 0 |
| trie + IG, `ent_title` (VQ) | 66 | 0.455 | 1.00 | 2 / 0 | 7 |
| trie + IG, `ent_bold` | 45 | 0.444 [0.309, 0.588] | 0.667 [0.488, 0.808] | **2 / 0** | 7 |

Holdout BPC is identical to VQ's (outcome labels make the training text the same).

**Rule 2 holds on all three counts:**
- (a) the `ent_bold` token meets D2 (0.833 / 0.50); `ent_title`'s does not (0.600 / 1.00);
- (b) the gate keeps 2 right, 0 wrong, the same as C1L;
- (c) it marks 33% of Dolly instructions `low`, against 93%.

So the frozen test was read once with `ent_bold`.

**Frozen test, `ent_bold` (TBOLD; 200 questions, 50 not in the slice).** A1d's C1L test read (`ent_title`) is beside it.

| Model (chat) | Abstained | Precision (not-in-slice) | Recall | Gated R / A / W | Raw right |
|---|---:|---|---|---|---:|
| trie, `ent_title` (A1d) | 74 | 0.635 [0.521, 0.736] | 0.94 [0.838, 0.979] | 0 / 200 / 0 | 0 |
| trie, `ent_bold` | 27 | **0.963** [0.817, 0.993] | **0.52** [0.385, 0.652] | 0 / 200 / 0 | 0 |
| trie + IG, `ent_title` (A1d) | 97 | 0.495 [0.397, 0.593] | 0.96 | 0 / 200 / 0 | 2 |
| trie + IG, `ent_bold` | 63 | 0.540 [0.418, 0.657] | 0.68 [0.542, 0.792] | 0 / 200 / 0 | 2 |

- **D2 is met on the frozen test for the first time** (CyphaLM's own message, trie model). It abstains on 26 of the 50 not-in-slice questions and on 1 in-slice one.
- **The cost is recall.** Half of the questions with no article in the slice are answered (wrongly), not refused. Trie + IG refuses 34 of 50 instead of 48.
- No gated answer changes: 0 right, 0 wrong in both.

#### 3. The entity in the tag (§5.4)

**The don't-know copy lambdas** (fit on 1,000 held-out don't-know turns, 158,244 bytes):

| Copy match | Bytes | Copy right | λ trie | λ trie + IG |
|---|---:|---:|---:|---:|
| 3 bytes | 10,029 | 0.553 | 0.105 | 0.080 |
| 4–5 | 7,899 | 0.551 | 0.195 | 0.140 |
| 6–7 | 4,157 | 0.946 | 0.845 | 0.760 |
| 8–11 | 7,469 | 0.967 | 0.955 | 0.945 |
| 12–15 | 6,521 | 0.963 | 0.935 | 0.920 |
| 16–31 | 11,881 | 0.962 | 0.930 | 0.910 |
| 32+ | 1,031 | 0.971 | 0.935 | 0.900 |

Inside a don't-know answer, a copy match of 6 bytes or more is right 95–97% of the time. The default copy lambdas, fit on wiki text, give such matches far less weight.

**DKC (dev)** against VQ:

| trie + IG (chat) | VQ | DKC |
|---|---|---|
| Abstained (not-in-slice) | 66 (30 / 30) | 65 (30 / 30) |
| Tags written | 49 | 34 |
| Tag names a span of the question | **0** [0, 0.073] | **34 of 34** [0.898, 1.00] |
| Complete tags | 21 | 34 |
| Reason code right | 30 | 25 |
| Message names the question's entity | 0 | 0 |
| Message entity = tag entity | — | **0 of 34** |
| The answer writes the frame token itself (`<\|entity="Antares"\|><\|fam=low\|>I don't know.`) | 0 | **15** |
| Message ends on the question's tail ("nothing about Economy of Canada?") | 0 | 13 |
| Gate out of fold: right / wrong | 2 / 0 | 2 / 0 |
| Holdout BPC, defaults (trie + IG) | 1.789464 | 1.789591 (+0.00013) |

The trie model's lines are unchanged: 50 abstained, 0.600 / 1.00, no tags.

**Rule 3 holds on all five counts:**
- (a) 34 of 34 tags name a span of the question;
- (b) complete tags 34 of 34, against 21 of 49;
- (c) abstentions 50 / 65, against 50 / 66;
- (d) the gate keeps 2 right, 0 wrong;
- (e) BPC +0.00013.

So the frozen test was read once with DKC.

**Frozen test, DKC (TDKC).** Its dev lines are re-printed identical to DKC's.

| trie + IG (chat) | A1d C1L test | DKC test |
|---|---|---|
| Abstained (not-in-slice) | 97 (48 / 50) | 97 (48 / 50) |
| Tags / complete | 72 / 28 | 50 / **47** [0.838, 0.979 of tags] |
| Tag names a span of the question | **0** | **49 of 50** [0.895, 0.996] |
| Reason code right | 45 | 35 |
| Message entity = tag entity | — | 0 of 50 |
| The answer writes the frame token itself | 0 | 23 of 97 [0.164, 0.331] |
| Message ends on the question's tail | 0 | 20 |
| Gated R / A / W | 0 / 200 / 0 | 0 / 200 / 0 |

The trie model's test lines equal A1d's: 74 abstained, 0.635 / 0.94.

- **The tag now names the question's entity.** On the test, 49 of 50 tags do, against 0 of 72. That is what the gap map reads: DISCOVERY §3.5 parses the tag into demand +1 for that entity.
- **The message does not.** The message comes before the tag. InfiniGram's long match still writes a training entity there ("nothing about Henrik Ibsen … entity=\"tungsten\""). So message and tag now disagree in every tagged answer. Before, both named the same wrong entity.
- **Two new faults the rule did not cover.**
  - In 23 of 97 test abstentions CyphaLM writes the frame token itself, with a training entity, before "I don't know.": InfiniGram continues a training don't-know prompt.
  - In 20, copy carries the message on into the question's tail.
- **Rule 3 adopts it, and I recommend Odin's veto.** It helps the tag (the gap map's input). It makes the visible answer worse: a control token in 24% of refusals, and a message that contradicts its own tag.

#### Frozen Q/A test reads (`bench/eval/t_qa_wiki.tsv`)

| Date | Change | Run | Result |
|---|---|---|---|
| 2026-10-28 | C1L (A1d) | TEST_C1L | above (A1d) |
| 2026-10-28 | R3raw (A1d) | TEST_R3raw | above (A1d) |
| 2026-10-29 | `ent_bold` in the token (ANS, rule 2) | TBOLD | trie 0.963 / 0.52; gated 0 / 0 |
| 2026-10-29 | entity frame + `--copy-dk` (ANS, rule 3) | TDKC | tags name the entity 49 of 50; gated 0 / 0 |
| 2026-10-29 | byte Kneser–Ney default (BKN2) | T4 (`/root/work/bkn/runs/bkn2`) | gated 0 right / 3 wrong (was 0 / 0); trie + IG raw 2 right, 96 wrong, 102 abstained |

The two changes were read separately. Their combination (`ent_bold` with DKC) has not been read.

#### Predictions, scored

| Id | Result | Score |
|---|---|---|
| AN1 | `ent_title` 0.933, `ent_bold` 0.332 of Dolly marked `low` | **Hit** |
| AN2 | trie exactly 18 at 0.833 / 0.50; trie + IG 45 abstained, precision 0.444 (predicted 20–35, 0.45–0.75) | **Partial** |
| AN3 | gate 2 right, 0 wrong | **Hit** |
| AN4 | rule 2 holds; `ent_bold` adopted | **Miss** |
| AN5 | λ ≥ 0.90 from 8 bytes, ≤ 0.195 below 6 | **Hit** |
| AN6 | 34 of 34 tags (100%; predicted 50–90%) | **Miss** (above the range) |
| AN7 | abstentions 50 / 65 against 50 / 66; gate 2 / 0 | **Hit** |
| AN8 | +0.00013 | **Hit** |
| AN9 | the registered count (`message_entity_in_question`) 0 of 65; but 13 messages end on the question's tail, which that count does not catch | **Hit** by the registered count |

6 hits, 1 partial, 2 misses.

#### Decisions (by measurement under PLAN ledger 55, open to Odin's veto)

| Question | Decision | Evidence |
|---|---|---|
| §5.2 low-only token | **The answering configuration's token form** (`--fam-train low`) | Rule 1: today's tool reproduces A1d's C1L; the token on every turn leaves the gate nothing (C1: 0 / 0), low-only keeps 2 / 0 |
| §5.3 `ent_title` or `ent_bold` | **`ent_bold`** (rule 2) | Dev meets D2 (0.833 / 0.50), gate unchanged (2 / 0), 33% of real questions `low` against 93%; test 0.963 [0.817, 0.993] / 0.52, D2 met; recall falls from 0.94 to 0.52 |
| §5.4 entity in the tag | **Adopted by rule 3; Odin's veto recommended** | Test: tags name the entity 49 of 50 (was 0), complete 47 of 50 (was 28 of 72); but the message never agrees with the tag (0 of 50), 23 of 97 refusals write the frame token, 20 messages end on the question's tail |

**Next, measurable without a ruling.**
- A message that follows the tag: the frame before "nothing about " as well (A1d's E1c form, `about {E};entity="{E}"`) with `--copy-dk`.
- Stopping the frame-token echo: no frame in training prompts, only in the tag.
- Each is a new registered dev read, then one test read.

#### Checks

- `cyphalm_trie selftest` passes. It now includes the frame and don't-know-context checks.
- The nine CTests of HANDOFF §4.4 pass (9 of 9).
- Without the new flags, rows and stdout are byte-identical to main's VQ (PAR).
- The source compiles clean under `-Wall -Wextra`.
