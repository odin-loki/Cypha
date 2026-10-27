# A1c: entity-level familiarity, and a copy path for the tag

**Odin Loch** · 2026-10-28 · follows [A0_A1_CYPHALM_QA_AND_DONT_KNOW.md](A0_A1_CYPHALM_QA_AND_DONT_KNOW.md) "Next steps" 1, 3 and 4 · step A1 of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4

**Status: in progress.** Predictions are written first; results follow as each run ends.

---

## Predictions (written before any run)

**Familiarity statistics** (dev AUROC, not-in-slice against in-slice; 0.5 = chance; A1's question log-probability gave 0.54–0.60):

| Id | Statistic | Predicted dev AUROC | Why |
|---|---|---|---|
| F1 | `ent_bold`: count of `'''entity` (a bold lead) | 0.70–0.80 | An article bolds its subject in the lead. The famous not-in-slice subjects have no article. E1 entity errors on in-slice questions ("Canada", "the world") cost the rest |
| F2 | `ent_title`: count of `<title>entity</title>` | 0.68–0.78 | Same idea, stricter |
| F3 | `ent_count`: count of the entity anywhere | 0.30–0.55 | Famous subjects are mentioned more often than obscure ones that have an article |
| F4 | `ent_link`: count of `[[entity` | 0.35–0.60 | Same reason |
| F5 | `ent_igfrac`: longest piece found / length | 0.45–0.55 | Almost every word occurs somewhere |
| F6 | `ent_lp`: the trie's mean log2 p of the entity | 0.45–0.65 | Familiar bytes, not familiar subjects (A1's lesson) |

**Token run (A1c-after: 20% don't-know turns, token after `Assistant: ` from the best statistic, trie):**
- **F7.** Dev abstain precision on not-in-slice 0.35–0.55, recall 0.75–0.95. D2 (precision ≥ 0.8 and recall ≥ 0.5) **misses** on precision, because E1 misses the entity on many in-slice questions and those get `low`.
- **F8.** Holdout BPC changes by ≤ 0.005 against A0 (as D3).

**Copy expert (a match over the current context, mixed like InfiniGram):**
- **C1.** Holdout BPC (second half): trie + copy beats trie by 0.05–0.15; trie + InfiniGram + copy beats trie + InfiniGram by 0.02–0.08. LCI P(gain > 0.005) ≥ 0.95 for both.
- **C2.** Tags naming the question's entity: 0–10% (A1: 0). The copy expert can continue an entity once its first bytes match the question, but nothing points it at the entity's first byte.
- **C3.** Abstain precision and recall within ±0.05 of the same model without copy; tag completeness within ±10 points.

---

## Setup

Everything is as in A0/A1 ([report](A0_A1_CYPHALM_QA_AND_DONT_KNOW.md) "Setup"): enwik8's first 8 MiB (train `[0, 7340032)`, holdout `[7340032, 8388608)`), Dolly 15,015 records, `cyphalm_trie qa`, dev 130 questions (100 in-slice, 30 not-in-slice), the frozen test 200 once at the end. Machine: 4 shared cores, every run under `nice -n 5`, one at a time; g++ 13.3 `-O2 -std=c++20`, clean under `-Wall -Wextra`.

**Parity first.** The rebuilt tool reproduces A0 exactly: holdout mixture 2.344571, trie + InfiniGram 2.071574, dev raw right 0 (trie) and 2 (+ InfiniGram). 7.5 min. `dktext` writes a byte-identical `dk.tsv` (3,963 rows).

### The question parse (E1)

`pos_tagger question --tsv` (the E1 tagger, dev UPOS 94.81%) parses every question once, outside the tool: the 130 dev questions, the 15,015 Dolly instructions, the don't-know questions and the shard questions. `qa --parse FILE` reads the entity column. The tool has no parser of its own.
- E1 finds an entity in 128 of 130 dev questions. It is often not the article's subject: "Canada" for "Who was the second Prime Minister of Canada?", "Peace" for "Who wrote War and Peace?", "dwan die" for "when did allan dwan die".

### Candidate statistics (all CyphaLM's own, over its own training bytes)

For a question, `qa` computes, from the E1 entity with a leading article dropped and lowered:

| Statistic | What it counts | Where |
|---|---|---|
| `question_lp` | A1's statistic: the question's mean log2 p | the wiki trie (state after the wiki pass) |
| `ent_count` | log2(1 + occurrences of the entity) | a suffix array over the lowered training bytes (wiki train + Dolly), the InfiniGram index type |
| `ent_link` | the same for `[[entity` (a link) | same |
| `ent_bold` | the same for `'''entity` (a bold lead: how an article names its subject) | same |
| `ent_title` | the same for `<title>entity</title>` | same |
| `ent_igfrac` | the longest piece of the entity found there, as a share of its length | same |
| `ent_lp` | the entity's mean log2 p, read in its question | the wiki trie |

The index holds only the base training bytes: no don't-know text and no shard text.

### Fixing the threshold recipe

A1's don't-know entities were chosen to be absent from the training bytes. So every count statistic is 0 for them, and a cut fit on "don't-know against Dolly" would only learn "count = 0". The famous not-in-slice dev subjects (Napoleon, Google) are mentioned often, so that cut would miss them. Two changes, neither using dev labels:
1. **New don't-know rows for mentioned entities** (`dktext --mention`). Pages after byte 8,388,608 whose title occurs in the training bytes as a link (`[[title`), so CyphaLM has read the name but has no article on it. That is the not-in-slice case. 2,231 such entities join the 3,274 absent ones: 5,505 entities, 6,583 rows, leakage 0, absence audit 0/300 (on the absent ones). Their reason code stays `UNKNOWN_ENTITY`: no article, so no concept links to the mention (DISCOVERY §3.1).
2. **Cuts fit on training-side entities only.** Positives: the titles of the wiki-train pages (they have an article). Negatives: the don't-know entities. Any title that occurs in dev or test text is left out. The cut is the Youden-J point; the token is `low` below it, `high` at or above it, and `mid` when E1 found no entity.

The dev AUROC needs no fitting at all: it is read off dev directly, with a 2,000-draw bootstrap interval over the dev questions.

---

## Result 1: entity familiarity separates the dev questions

**Dev AUROC** (not-in-slice below in-slice; 130 questions; 2,000-draw bootstrap 95% interval; E1 found no entity in 2 questions, which sit below every number):

| Statistic | Dev AUROC | Bootstrap 95% | Prediction | Score |
|---|---:|---|---|---|
| `ent_title` | **0.795** | 0.745–0.847 | F2: 0.68–0.78 | **Miss** (better) |
| `ent_lp` | 0.771 | 0.667–0.864 | F6: 0.45–0.65 | **Miss** (better) |
| `ent_bold` | 0.765 | 0.714–0.818 | F1: 0.70–0.80 | **Hit** |
| `question_lp` (A1) | 0.693 | 0.576–0.798 | — | — |
| `ent_count` | 0.535 | 0.413–0.654 | F3: 0.30–0.55 | **Hit** |
| `ent_link` | 0.517 | 0.399–0.627 | F4: 0.35–0.60 | **Hit** |
| `ent_igfrac` | 0.495 | 0.434–0.562 | F5: 0.45–0.55 | **Hit** |

- **Three statistics clear 0.60 by a wide margin.** `ent_title` and `ent_bold` do so with their whole interval (lower bounds 0.745 and 0.714). `ent_lp`'s interval reaches down to 0.667.
- **Counts and links are at chance**, as predicted: the famous not-in-slice subjects are mentioned and linked as often as the obscure in-slice ones.
- **A note on A1's 0.54–0.60.** A1 read `question_lp` from the A0 rows, under the final chat model. The same statistic under the wiki-only model (the one A1's token used) gives 0.693 here, with a wide interval (0.576–0.798).
- **The choice.** `ent_title` has the best dev AUROC, so it becomes the token's statistic (the rule fixed before the run: "the best statistic"). `ent_bold` is within its interval and reads only article text, not XML markup; it is the fallback if Odin prefers that.

**Training-side calibration** (positives: 302 wiki-train titles not found in dev or test text; negatives: the don't-know entities). The cut is the Youden-J point.

| Statistic | With mentioned entities (5,505 negatives): AUROC / cut / J | Absent entities only (3,274, A1's set): AUROC / cut / J |
|---|---|---|
| `ent_title` | 0.9997 / count ≥ 1 / 0.999 | 1.0000 / count ≥ 1 / 1.000 |
| `ent_bold` | 0.891 / count ≥ 1 / 0.782 | 0.897 / count ≥ 1 / 0.795 |
| `ent_count` | 0.848 / count ≥ 2 / 0.655 | **1.0000 / count ≥ 1 / 1.000** |
| `ent_link` | 0.693 / count ≥ 1 / 0.320 | 0.863 / count ≥ 1 / 0.725 |
| `ent_igfrac` | 0.797 / whole entity found / 0.595 | 0.9998 / whole entity found / 1.000 |
| `ent_lp` | 0.737 / −2.881 / 0.388 | 0.744 / −2.881 / 0.409 |
| `question_lp` | 0.736 / −2.881 / 0.388 | 0.743 / −2.881 / 0.408 |

- **The broken recipe, shown.** With A1's absent-only entities, every count calibrates perfectly on training data (AUROC 1.000, "count = 0 means unknown") and is useless on dev (`ent_count` 0.535). The mentioned entities break that: `ent_count`'s calibration falls to 0.848. The dev result, not the calibration, is what counts.
- **`ent_title`'s cut is the same either way:** `low` exactly when no page in the slice has the entity as its title.

---

## Result 2: the copy expert (BPC first)

**What it is.** A match over the current context (`--copy`). At each byte it finds the longest suffix of the context, 3 to 64 bytes, that also occurred earlier in the last 4,096 bytes, and predicts the byte that followed it (the most recent of the longest). It is mixed like InfiniGram: P = λ[b]·[the predicted byte] + (1 − λ[b])·P, with one λ per match-length bucket (none, 3, 4–5, 6–7, 8–11, 12–15, 16–31, 32–64). The λs are fit on the first half of the holdout, where the context is the holdout stream itself, and scored on the second half. No dev question is used. It is on top of the trie mixture, or on top of trie + InfiniGram. In answering, the context is the prompt plus the answer so far.

One decoding change comes with it: for the copy models, the no-repeat rule bans only repeats of bytes the answer itself wrote. Copying from the question is the expert's point; looping on its own answer is still banned. The models without copy keep the rule as it was.

**Holdout BPC** (A0 model; second half of the holdout, 524,288 bytes; LCI's P_gain is the Bayesian bootstrap over 256 blocks of 2,048 bytes, PLAN §8.2):

| Model | Without copy | With copy | Gain (90% interval) | P(gain > 0.005) | P(gain > 0) |
|---|---:|---:|---|---:|---:|
| trie mixture | 2.319989 | **2.113702** | 0.206 (0.188–0.225) | 1.000 | 1.000 |
| trie + InfiniGram | 2.071574 | **1.926101** | 0.145 (0.132–0.160) | 1.000 | 1.000 |

- The whole-holdout trie number (2.344571) is unchanged: copy is an extra expert, not a change to the trie.
- λ rises with the match length: 0.105 at 3 bytes to 0.865 at 32–64 bytes (trie); 0.085 to 0.785 on top of InfiniGram.
- **C1: miss, on the good side.** I predicted gains of 0.05–0.15 and 0.02–0.08. The gains are about twice that. The trie's own match expert and InfiniGram both point only into the training bytes. Neither can repeat a name, a link or a template that the current article has already used, and Wikipedia repeats those constantly. LCI's P_gain is 1.000 for both, with 256 blocks (n_eff ≥ 200).
- Cost: up to 4,096 byte comparisons a byte. The whole run (A0 model, copy on, four models answering, R3) took 8.0 min against 7.5 min for A0.

---

## Result 3: the token run as specified (A1c-after, statistic on every turn)

**Run D.** 20% don't-know turns (3,754, drawn from the 6,583 absent and mentioned rows), the `ent_title` token after `Assistant: ` on every turn (Dolly, don't-know, dev), cut `count ≥ 1` from the training-side calibration, copy on.

**The token on the training turns is the problem.**

| Token | Dolly turns | Don't-know turns | Dev in-slice | Dev not-in-slice |
|---|---:|---:|---:|---:|
| low | **14,547** | 6,182 | 37 | 30 |
| mid (no entity) | 279 | 342 | 2 | 0 |
| high | 189 | 59 | 61 | 0 |

- **On dev the token is the right shape:** all 30 not-in-slice questions get `low`, and 61 of 100 in-slice ones get `high`. Obeyed exactly, it would give precision 30/67 = 0.448 and recall 30/30 = 1.00.
- **On Dolly it is almost always `low`** (97%): few Dolly subjects have an article in a slice of mostly "A" pages. So `low` mostly precedes ordinary answers, and `high` appears only 248 times.

**Dev results (chat form; abstained = CyphaLM wrote the message):**

| Model | Abstained | Not-in-slice abstained | Precision (base 0.231) | Recall | Tags / complete / reason right / entity right |
|---|---:|---:|---:|---:|---|
| trie | 130 | 30/30 | 0.231 | 1.00 | 0 / 0 / 0 / 0 |
| trie + copy | 130 | 30/30 | 0.231 | 1.00 | 0 / 0 / 0 / 0 |
| trie + InfiniGram | 55 | 15/30 | 0.273 | 0.50 | 53 / 19 / 15 / 0 |
| trie + InfiniGram + copy | 55 | 15/30 | 0.273 | 0.50 | 53 / 19 / 15 / 0 |

- **The trie says "I don't know" after every token.** After `gh|>` it saw 59 don't-know messages against 189 Dolly answers spread over many first bytes, so "I" wins there too.
- **F7: miss.** Precision 0.231 (trie) and 0.273 (trie + InfiniGram), against a predicted 0.35–0.55. Recall 1.00 and 0.50.
- **BPC** (holdout, against A0): trie mixture 2.347970 (+0.0034), trie + InfiniGram 2.072565 (+0.0010); with copy 2.117984 and 1.928607. **F8: hit** (≤ 0.005).
- **Copy and the tag (C2, C3).** No tag names the question's entity, with or without copy: **C2 hit** (0, inside 0–10%). Abstention and tag completeness are identical with and without copy: **C3 hit**. The copy expert fires inside the message only when the first bytes already match the question: two answers (the two Afghanistan questions) copy "…about Economy of Afghanistan?" from the question, question mark included, then the newline that follows it in the prompt, which ends the answer before the tag. Inside the tag, at `entity="Herman of A`, copy predicts the message's "l" (Alaska) with λ 0.36, and InfiniGram's longer training-text match ("Herman of Ame…") wins.

### Predictions for the outcome-token variant (written before its run)

The fix: the token is a gate decision (Odin: allowed gate-side, like the abstain gate). With `--fam-train outcome`, every answered training turn carries `high` and every don't-know turn `low`; dev and test still get the statistic's token. CyphaLM then learns a clean switch, and the statistic alone decides.
- **G1.** The trie abstains on exactly the 67 `low` dev questions (perhaps also the 2 `mid`): precision 0.43–0.45, recall 1.00. D2 misses on precision.
- **G2.** Trie + InfiniGram: precision 0.30–0.45, recall 0.60–1.00.
- **G3.** Holdout BPC within 0.005 of A0.
- **G4.** Tags naming the question's entity: 0–10%, with or without copy.

---

## Result 4: the outcome-token variant (A1c-gate, run D′)

Same as run D, except `--fam-train outcome`: all 15,015 Dolly turns carry `high`, all 3,754 don't-know turns `low`. Dev questions get the statistic's token (`ent_title`, cut count ≥ 1): `low` 37 in-slice + 30 not-in-slice, `mid` 2 in-slice, `high` 61 in-slice.

| Model | Abstained | Not-in-slice abstained | Precision (Wilson 95%) | Recall (Wilson 95%) | Tags / complete / reason right / entity right |
|---|---:|---:|---|---|---|
| trie | 69 | 30/30 | **0.435** (0.324–0.552) | **1.00** (0.886–1.00) | 0 / 0 / 0 / 0 |
| trie + copy | 69 | 30/30 | 0.435 | 1.00 | 0 / 0 / 0 / 0 |
| trie + InfiniGram | 69 | 30/30 | 0.435 | 1.00 | 68 / 29 / 30 / 0 |
| trie + InfiniGram + copy | 69 | 30/30 | 0.435 | 1.00 | 68 / 29 / 30 / 0 |

- **CyphaLM follows the token exactly,** in all four models: it writes the message on the 67 `low` and 2 `mid` questions and on no `high` one. With a clean switch in training, InfiniGram no longer blurs it (A1b-after: 0.280 with InfiniGram).
- **G1 hit, G2 hit** (0.435 / 1.00 for both). **G3 hit:** holdout BPC 2.348246 (+0.0037 against A0), trie + InfiniGram 2.072703 (+0.0011); with copy 2.118240 and 1.928737. **G4 hit:** no tag names the question's entity.
- **D2 misses on precision** (0.435 against ≥ 0.8; recall 1.00 against ≥ 0.5). The whole gap is in-slice questions whose E1 entity is not a page title: "Canada" (Prime Ministers), "the world", "Ireland", "Iowa" (for "Ames, Iowa"), "343 BC", "dwan die". 37 of the 100 in-slice questions. The statistic is right about the entity it is given; E1 gives it the wrong entity.
- **Reason codes:** every tag says `UNKNOWN_ENTITY`, so it is right on the 30 not-in-slice abstentions (30 of 68 tags) and wrong on the in-slice ones, where `NO_EVIDENCE` was expected. Tags are complete in 29 of 68.
- **Against A1b-after** (the A1 final): dev precision 0.500 → 0.435 and recall 0.43 → 1.00 for the trie. A1b's precision came from a 3-way cut that held 26 questions in `low`; this cut holds 67 and finds every not-in-slice question.
- **Answers:** raw right 0 (trie) and 1 (each other model, luck: "Montgomery is the capital…"). At 2% wrong the gate still answers nothing.

### No `NO_EVIDENCE` rows (skipped)

A `NO_EVIDENCE` row needs an entity with an article and a relation the slice never states. The only evidence of "never states" here is F2's store, and its absence is not evidence: F2's report has 451 birth-place seeds against 1,144 birth-year seeds, so most people's birth place is missing from the store whether the article states it or not, and most relations have no kept pattern at all. A row built from it would teach "I have read about X but not where X was born" for facts the article does state. That would be training a false reason. So every don't-know row stays `UNKNOWN_ENTITY`, and in-slice abstentions keep the wrong code.
