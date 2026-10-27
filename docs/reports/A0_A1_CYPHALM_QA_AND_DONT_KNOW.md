# A0 and A1: CyphaLM answers T-QA by itself, then learns to say "I don't know"

**Odin Loch** · 2026-10-28 · steps A0 and A1 of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4 · harness spec [TOOL_PROFILE_PLAN.md](../TOOL_PROFILE_PLAN.md) §2–§3

**Questions.**
1. **A0.** How many T-QA questions does CyphaLM answer by itself, with its own confidence as the abstain gate at a 2% wrong rate?
2. **A1.** Can CyphaLM be trained to write the DISCOVERY §3.5 "I don't know" message and tag itself, and to write it when it should?

**Answers.**
1. **None.**
   - **Dev:** CyphaLM gets 0–2 of 130 right, and the 2 are lucky words inside nonsense. At a 2% wrong rate, coverage is **0.0%** in every configuration.
   - **Frozen test:** raw 0–2 of 200 right. The dev-chosen gate keeps 3 test answers, all wrong (1.5% wrong, Wilson upper 4.3%).
   - **What it writes:** with InfiniGram, fluent Dolly answers to other questions. Without it, the same loop for every question.
2. **It learns to write the message. It does not learn when to write it.**
   - **Naive text (A1a):** CyphaLM writes the message at about its training share, and the choice is indiscriminate. Abstain precision on not-in-slice questions is 0.23–0.27, against a base rate of 0.23.
   - **Familiarity token (A1b):** CyphaLM follows the token perfectly. With the token placed after `Assistant: `, the trie writes the message on exactly the questions marked `low` and on no others. But the token is weak evidence:
     - dev: precision 0.50, recall 0.43;
     - frozen test: precision 1.00, recall 0.30, from a lucky split.
   - **The bottleneck is the familiarity statistic, not learning.** The model's mean log-probability of the question separates not-in-slice from in-slice questions with an AUROC of only 0.54–0.60.
   - **What it never manages:** naming the question's entity in the tag (0 of every tag written), or the right reason code on in-slice questions.

PLAN rule 10 holds. CyphaLM writes every answer byte. Strings are compared only to grade what it wrote, and in `dktext` to prepare training text: absence checks and the leakage check.

---

## Setup

| Item | Value |
|---|---|
| Text | enwik8 first 8 MiB (= enwik9's), train `[0, 7340032)`, holdout `[7340032, 8388608)`. `/root/cdata/enwik8` |
| Chat | Dolly 15k, 15,015 records (`/root/cdata/dolly.jsonl`) |
| Model | `cyphalm_trie chat` exactly: wiki pass, `<\|doc\|>` break, two Dolly passes, Synth mode, frozen weights. A1 adds don't-know turns to the Dolly passes |
| + InfiniGram | As `context8k`: index over wiki train + chat bytes (no holdout), cap 8192, 28 λ buckets fit on the first half of the holdout |
| Questions | `bench/eval/t_qa_wiki_dev.tsv`: 130 (100 in-slice, 30 not-in-slice). `bench/eval/t_qa_wiki.tsv`: 200 (150 / 50), frozen, run once at the end |
| Machine | 2 shared cores, 7 GB. Every run under `nice -n 5`, one at a time. g++ 13.3 `-O2 -std=c++20`, clean under `-Wall -Wextra` |

**Harness (`cyphalm_trie qa`).**
- **Prompt forms.**
  - Chat: `User: <question>\nAssistant: `.
  - Wiki: `<question> `.
- **Decoding.** Greedy, with the no-repeat rule. It stops at a newline, at a sentence end (`.`, `!` or `?` then a space, except after a one-letter initial), or at 80 bytes.
  - **One change:** once an answer starts with "I don't know", decoding runs to the end of the tag (`|>`, a newline, or 240 bytes), so the tag can be read.
- **Confidence (all CyphaLM's own):**
  - the mean and minimum log2 p per answer byte;
  - the entropy at the first answer byte;
  - the InfiniGram match length at the first answer byte (only with InfiniGram);
  - the mean log2 p of the question's own bytes under the answering model (familiarity).
- **Grading.** The T-QA rule: normalise (case, Latin-1 accents and punctuation removed), then
  - right if the answer equals an accepted variant, or either contains the other when at least one is multi-word, word-bounded;
  - abstained if the answer contains "I don't know" or `<|unknown`. An empty answer (a newline as the first byte) is also counted as abstained; that happens only in the wiki form, on 1–5 questions;
  - not-in-slice: any answer is wrong.
- **Abstain gate.**
  - **Rule:** answer if feature ≥ τ. Among (feature, τ) pairs with wrong / n ≤ 2%, take the one with the most right answers; ties go to fewer wrong, then fewer answered. With no right answers anywhere, the gate answers nothing.
  - **Chosen by 5-fold CV inside dev** (fold = row mod 5): chosen on four folds, applied to the fifth. So the dev numbers are out of fold.
  - **For the test:** the gate is chosen once on all of dev and applied unchanged.

---

## A0: CyphaLM alone

### BPC, for the record (the A0 model is the `chat` model)

| Number | A0 | Reference |
|---|---:|---|
| Holdout BPC, trie mixture | **2.344571** | `chat` `wiki_bpc_after_chat` 2.344571 (B1): identical |
| Second-half holdout, trie only | 2.319989 | `context8k` 2.3199 |
| Second-half holdout, trie + InfiniGram (cap 8192) | 2.071574 | `context8k` 2.0715 |
| Weights w4 / wmatch | 0.960948 / 0.038609 | `chat`: identical |

### Dev, 130 questions

**Raw** means the model answers everything it writes (only its own message abstains). **Gated** means the 5-fold CV abstain gate at a 2% wrong rate.

| Form / model | Raw R / A / W | Gated R / A / W | Coverage at 2% wrong | Selective acc. (Wilson lo) | Wrong rate (Wilson hi) |
|---|---|---|---:|---|---|
| chat / trie | 0 / 0 / 130 | 0 / 130 / 0 | 0.0% | n/a | 0.0% (2.9%) |
| chat / trie + IG | 2 / 0 / 128 | 0 / 130 / 0 | 0.0% | n/a | 0.0% (2.9%) |
| wiki / trie | 0 / 1 / 129 | 0 / 130 / 0 | 0.0% | n/a | 0.0% (2.9%) |
| wiki / trie + IG | 0 / 5 / 125 | 0 / 130 / 0 | 0.0% | n/a | 0.0% (2.9%) |

| Form / model | In-slice (100), raw R / A / W | Not-in-slice (30), raw R / A / W | Gated, in-slice / not-in-slice |
|---|---|---|---|
| chat / trie | 0 / 0 / 100 | 0 / 0 / 30 | all abstained; wrong-rate Wilson hi 3.7% / 11.4% |
| chat / trie + IG | 2 / 0 / 98 | 0 / 0 / 30 | all abstained |
| wiki / trie | 0 / 0 / 100 | 0 / 1 / 29 | all abstained |
| wiki / trie + IG | 0 / 4 / 96 | 0 / 1 / 29 | all abstained |

Raw selective accuracy, chat / trie + IG: 2/130 = 1.5% (Wilson lower 0.4%).

**The two "right" answers are luck.** The T-QA rule counts them, but neither answer means anything:
- "In which year did Alexis Carrel win the Nobel Prize?" → *"No, Sir Winston Churchill did not win the North Atlantic Ocean on 15 April **1912**"*.
- "What is the capital of Alabama?" → *"**Montgomery** is the capacity a team has to accomplish the goals or objectives…"*.

**Which gate the CV picked.**
- chat / trie + IG: no gate in one fold, `neg_entropy0` in four.
- Everything else: no gate in all five.
- The one right answer that a gate could keep sits in a single fold. When that fold is held out, the gate learned on the other four has no right answer to keep. So out of fold it keeps nothing.

### Risk–coverage (in-sample, all of dev; answered : right : wrong as the threshold is lowered)

| Form / model | Feature | 1 | 2 | 3 | 5 | 10 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|---|---|
| chat / trie | every feature | 1:0:1 | 2:0:2 | 3:0:3 | 5:0:5 | 10:0:10 | 50:0:50 | 100:0:100 | 130:0:130 |
| chat / trie + IG | mean_lp | 1:0:1 | 2:0:2 | 3:0:3 | 5:0:5 | 10:0:10 | 50:0:50 | 100:1:99 | 130:2:128 |
| chat / trie + IG | min_lp | 1:0:1 | 2:0:2 | 3:0:3 | 5:0:5 | 10:0:10 | 50:0:50 | 100:0:100 | 130:2:128 |
| chat / trie + IG | neg_entropy0 | **1:1:0** | 2:1:1 | 3:1:2 | 5:1:4 | 10:1:9 | 50:1:49 | 100:2:98 | 130:2:128 |
| chat / trie + IG | ig_match0 | **1:1:0** | 2:1:1 | 3:1:2 | 5:1:4 | 10:1:9 | 50:2:48 | 100:2:98 | 130:2:128 |
| chat / trie + IG | question_lp | **1:1:0** | 2:1:1 | 3:1:2 | 5:1:4 | 10:1:9 | 50:2:48 | 100:2:98 | 130:2:128 |
| wiki / trie, wiki / trie + IG | every feature | 1:0:1 | 2:0:2 | 3:0:3 | 5:0:5 | 10:0:10 | 50:0:50 | 100:0:100 | 129 / 125 answered, 0 right |

In-sample, the best possible point at ≤ 2% wrong is 1 right in 1 answered (0.8% coverage). Out of fold it is 0.

### Examples (chat form, first dev rows)

| Question | trie | trie + InfiniGram |
|---|---|---|
| Who created the Amber Diceless Roleplaying Game? | The states and the states are the statest in the states, and the streated to the | The Evergreen Game is a famous chess game won by Adolf Anderssen against Jean Du |
| Who designed Park Guell? | (the same loop, for every question) | The mitochondria is commonly referred to as the powerhouse of the cell as its pr |
| Who discovered the asteroid 1 Ceres? | ″ | Fruits that are typically not found in American grocery stores include jackfruit |
| Which author created Miss Marple? | ″ | The pineapple is a fruit. |
| What is the national anthem of Ireland called? | ″ | A group of cats is called a clowder. |
| Who created the character Spider-Man? | ″ | The cigarette smoking man is a fictional character from the TV show "The X-Files |

This is what the earlier report found ([CYPHALM_QA_AND_CONTEXT_TEST.md](CYPHALM_QA_AND_CONTEXT_TEST.md) §1–§2):
- **The trie sees only `nt: `.** After `Assistant: `, its last four bytes are the same for every question, so it writes the same loop every time.
- **InfiniGram reaches the question's last word or two,** and copies the Dolly answer of a different question that ended the same way.

### Frozen test (200 questions, run once)

The gate is chosen on all of dev:
- chat / trie + IG: `neg_entropy0` ≥ −1.9795, which keeps 1 dev answer;
- every other configuration: no gate (answer nothing).

| Form / model | Raw R / A / W | Gated R / A / W | Coverage | Selective acc. (Wilson lo) | Wrong rate (Wilson hi) | Gated in-slice (150) R / A / W | Gated not-in-slice (50) R / A / W |
|---|---|---|---:|---|---|---|---|
| chat / trie | 0 / 0 / 200 | 0 / 200 / 0 | 0.0% | n/a | 0.0% (1.9%) | 0 / 150 / 0 | 0 / 50 / 0 |
| chat / trie + IG | 2 / 0 / 198 | **0 / 197 / 3** | 1.5% | 0/3 = 0% (0%) | **1.5% (4.3%)** | 0 / 149 / 1 | 0 / 48 / 2 |
| wiki / trie | 0 / 4 / 196 | 0 / 200 / 0 | 0.0% | n/a | 0.0% (1.9%) | 0 / 150 / 0 | 0 / 50 / 0 |
| wiki / trie + IG | 0 / 5 / 195 | 0 / 200 / 0 | 0.0% | n/a | 0.0% (1.9%) | 0 / 150 / 0 | 0 / 50 / 0 |

- **The gate fails on the test.** The entropy gate picked on one lucky dev answer lets through 3 wrong test answers and no right ones. The point estimate is inside 2%; the Wilson upper bound (4.3%) is not.
- **The two raw right answers** (chat / trie + IG):
  - "When did Alaska become part of the United States as a state?" → *"Alaska became a state on January 3, 1959."* This is a stored Dolly answer that InfiniGram's long match reached. The question's wording is close to a Dolly prompt.
  - "How many lunar orbits did Apollo 8 make?" → *"As of March 2023, Quentin Tarantino has made 10 movies."* This one is luck.
- **Parity.** The `--test` run re-prints every dev line, and they are byte-identical to the dev-only run. That run used the binary built before the selftest and comment edits.

---

## A1: training CyphaLM to write the message and tag

### The training text (`cyphalm_trie dktext`)

- **Templates.** T-QA relation templates taken from the dev question forms:

  | Relation | Template | Used when the page's categories say |
  |---|---|---|
  | `birth_date`, `birth_place`, `death_date`, `identity` | When was / Where was {E} born? When did {E} die? Who was {E}? | births or deaths |
  | `author` | Who wrote {E}? | novels, books, plays, poems |
  | `director` | Who directed {E}? | films |
  | `composer` | Who composed {E}? | operas, songs, compositions |
  | `painter` | Who painted {E}? | paintings |
  | `capital`, `language` | What is the capital / the official language of {E}? | countries, states, provinces, territories |
  | `founder` | Who founded {E}? | companies, organisations, bands |
  | `location`, `country` | Where is {E}? In which country is {E}? | cities, towns, rivers, islands, … |
  | `inventor` | Who invented {E}? | inventions, technology |
  | `definition` | What is {E}? | anything else |

  Each entity gets at most two relations, chosen by a hash.
- **Entities.** Page titles from enwik8 after byte 8,388,608 (11,271 pages). Kept only if:
  - not a namespace page or a redirect;
  - 4–40 ASCII bytes, with no brackets or quotes, and not "List of";
  - **absent from the training bytes**: the lowered title never occurs in the lowered wiki-train + Dolly bytes. This is an exact suffix-array occurrence test. A plain `std::string::find` scan of 300 kept entities, spread across the list, found **0**.
- **Leakage check, before training.**
  - Rejected: 0 entities that occur in dev or test text, and 41 that contain a dev or test answer variant of ≥ 4 bytes.
  - On the final rows: **0** questions equal to a dev or test question, and **0** answer variants anywhere in the text.
  - The builder exits non-zero if any of these checks, or the audit, finds anything.
- **Result.** 3,274 entities and 3,963 rows: `definition` 2,392, `birth_date` 320, `identity` 295, `birth_place` 286, `death_date` 261, `location` 87, `country` 87, `author` 74, `founder` 59, `director` 29, `inventor` 24, `capital` 21, `language` 21, `composer` 7. Takes 6 s.
- **Example row:** `Where was Ambrose of Alexandria born?` → `I don't know. I have nothing about Ambrose of Alexandria in what I've read. <|unknown reason=UNKNOWN_ENTITY entity="Ambrose of Alexandria" relation=birth_place gap=G-42A1|>`.
- **Gap ids** are `G-` plus 16 bits of an FNV-1a hash of entity|relation.
- **Reason codes.** Every row is `UNKNOWN_ENTITY`, because every entity is absent from what CyphaLM read. No other code could be built honestly in the time: a `NO_EVIDENCE` row needs an entity that is present and a fact that is absent.

The don't-know turns are spread evenly through both Dolly passes, as `User: <q>\n[token]Assistant: <message and tag>\n\n`.

### Variants

| Variant | Don't-know share of turns | Familiarity token |
|---|---:|---|
| A1a-5 | 5.0% (790 turns) | none |
| A1a-20 | 20.0% (3,754) | none |
| A1b | 20.0% | `<\|fam=…\|>` right **before** `Assistant: ` (as specified) |
| A1b-after | 20.0% | `<\|fam=…\|>` right **after** `Assistant: `, so it ends the prompt |

**The familiarity token (A1b).**
- **The statistic.** CyphaLM's own mean log2 p per byte of the question text after `User: `. It is computed from the model state after the wiki pass, before any chat, with the trie mixture's frozen weights. The match expert cannot fire on a separate buffer, so it is orders 0–4 only.
- **The same computation for everything:** Dolly instructions, don't-know questions, and dev and test questions. So the token means the same thing in training and at answer time.
- **Thresholds, set on the training questions only:**
  - t1 = −3.2795 bits/byte: the cut that best separates the don't-know questions from the Dolly instructions (Youden J 0.435);
  - t2 = −2.6386: the median Dolly familiarity above t1.
  - low < t1 ≤ mid < t2 ≤ high.

| Token | Dolly turns | Don't-know turns (of 3,963) | Dev in-slice | Dev not-in-slice |
|---|---:|---:|---:|---:|
| low | 3,263 | 2,586 | 13 | 13 |
| mid | 5,876 | 932 | 38 | 7 |
| high | 5,876 | 445 | 49 | 10 |

### Results on dev (chat form; abstained = CyphaLM wrote the message)

"Not-in-slice" scores abstention against the 30 not-in-slice questions (base rate 30/130 = 0.231). "In-slice wrong" is the in-slice questions A0 did not get right with the same model (100 for the trie, 98 with InfiniGram).

| Variant | Model | Abstained | Not-in-slice abstained | Precision (base 0.231) | Recall | In-slice wrong abstained | Right (A0: 0 / 2) | Holdout BPC (Δ vs A0) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A1a-5 | trie | 0 | 0/30 | n/a | 0.00 | 0/100 | 0 | 2.344912 (+0.0003) |
| A1a-5 | trie + IG | 31 | 7/30 | **0.226** | 0.23 | 24/98 | 2 | IG 2.071685 (+0.0001) |
| A1a-20 | trie | 130 | 30/30 | **0.231** | 1.00 | 100/100 | 0 | 2.346545 (+0.0020) |
| A1a-20 | trie + IG | 64 | 17/30 | **0.266** | 0.57 | 46/98 | 1 | IG 2.071500 (−0.0001) |
| A1b | trie | 130 | 30/30 | 0.231 | 1.00 | 100/100 | 0 | 2.347728 (+0.0032) |
| A1b | trie + IG | 52 | 13/30 | 0.250 | 0.43 | 39/98 | 1 | IG 2.072346 (+0.0008) |
| A1b-after | trie | 26 | 13/30 | **0.500** | 0.43 | 13/100 | 0 | 2.347882 (+0.0033) |
| A1b-after | trie + IG | 50 | 14/30 | 0.280 | 0.47 | 36/98 | 1 | IG 2.072360 (+0.0008) |

Precision against all should-abstain questions (not-in-slice plus A0's in-slice wrong) is 0.98–1.00 for every variant. The base rate there is 0.98–1.00, because A0 gets almost nothing right, so that measure says nothing.

**Abstention by token** (abstained / questions; dev, chat form):

| Variant / model | low: in-slice | low: not-in-slice | mid: in / not | high: in / not |
|---|---|---|---|---|
| A1b / trie + IG | 11/13 | 7/13 | 17/38, 2/7 | 11/49, 4/10 |
| A1b-after / trie | **13/13** | **13/13** | 0/38, 0/7 | 0/49, 0/10 |
| A1b-after / trie + IG | 11/13 | 8/13 | 15/38, 2/7 | 10/49, 4/10 |

**What the rows show.**
- **The trie cannot see a token placed before `Assistant: `.** That is the 11 bytes between the token and the answer, and the trie reads only 4. So A1b on the trie behaves exactly like A1a-20: it abstains on every question.
- **Placed after `Assistant: `, the token ends the prompt** (`ow|>`, `id|>`, `gh|>`). The trie then follows it perfectly. This is the ceiling for this statistic: "abstain if and only if `low`" gives precision 13/26 = 0.50 and recall 13/30 = 0.43.
- **With InfiniGram the token is one signal among many.** The long match also reaches the question's last words. Abstention leans toward `low` (69%, against 25% at `high`) but stays loose.
- **The statistic is the limit.** How well each of CyphaLM's own numbers separates not-in-slice from in-slice questions on dev (AUROC from the A0 rows; 0.5 = chance):
  - question log2 p: 0.54–0.60;
  - InfiniGram match: 0.44–0.59;
  - first-byte entropy: 0.44–0.59;
  - answer mean log2 p: 0.42–0.50.
- **Why the statistic is weak.** The not-in-slice subjects are famous: Napoleon, Google, War and Peace. Their words are common in the slice, even though the slice has no article on them. Many in-slice subjects are obscure ("Amber Diceless Roleplaying Game", "Amhrán na bhFiann"). So "familiar bytes" and "has an article" disagree.

**Tags and reason codes.**

| Variant / model | Tags written | Reason code right | Tag entity is the question's |
|---|---:|---:|---:|
| A1a-5 / trie + IG | 31 | 7 (0.23) | 0 |
| A1a-20 / trie + IG | 62 | 17 (0.27) | 0 |
| A1b / trie + IG | 50 | 13 (0.26) | 0 |
| A1b-after / trie + IG | 48 | 14 (0.29) | 0 |
| any trie-only variant | 0 | n/a | n/a |

- **Expected code.** `UNKNOWN_ENTITY` for not-in-slice questions: the subject has no article, so no concept links to it. `NO_EVIDENCE` or `LOW_CERTAINTY` for in-slice questions: the article is there, but no path to the answer.
- **Only one code was trained.** CyphaLM always writes `UNKNOWN_ENTITY`, so reason accuracy is just the not-in-slice share of its abstentions.
- **No tag ever names the question's entity.** Neither the trie nor InfiniGram can copy from the prompt: the match expert and the index both point into the training bytes. So the entity slot is filled with a training entity that shares a prefix.
- **The trie-only variants never finish a tag.** After "I don't know" their 4-byte context drifts. Every one of A1a-20's 130 answers is the same string: `I don't known reason=UNKNOWN_ENTITY entity="Communication of the states and the statest…`. It counts as abstained because it starts with the message.

**Examples (A1a-5, trie + InfiniGram).**

| Question (dev) | Slice | CyphaLM wrote |
|---|---|---|
| Who was the second Prime Minister of Canada? | in | I don't know. I have nothing about Economy of Canada in what I've read. <\|unknown reason=UNKNOWN_ENTITY entity="Economy of the Faroe Islands" relation=country gap=G-BCDA\|> |
| When was Andrei Tarkovsky born? | in | I don't know. I have nothing about Bill Joy in what I've read. <\|unknown reason=UNKNOWN_ENTITY entity="Bill Joy" relation=birth_place gap=G-BB8B\|> |
| Which sign of the zodiac is Aquarius? | in | I don't know. I have nothing about Confucius in what I've read. <\|unknown reason=UNKNOWN_ENTITY entity="Confucius" relation=identity gap=G-A260\|> |
| When was Winston Churchill born? | not | I don't know. I have nothing about Bert Bell in what I've read. <\|unknown reason=UNKNOWN_ENTITY entity="Bert Bell" relation=birth_place gap=G-D526\|> |
| What is the capital of Finland? | not | I don't know. I have nothing about Economy of Finland in what I've read. <\|unknown reason=UNKNOWN_ENTITY entity="Economy of the Faroe Islands" relatio… |
| Where was Napoleon born? | not | I don't know. I have nothing about Charles Ancillon in what I've read. <\|unknown reason=UNKNOWN_ENTITY entity="Charles Anchorage a relatively casual, … |

- **The message is well formed. The tag is complete (it ends in `gap=G-XXXX|>`) in only 9 of 31 answers.** With InfiniGram, it is complete in 16/64 for A1a-20, 10/52 for A1b and 10/50 for A1b-after.
- **The entity is a training entity, not the question's.** It is picked by the last bytes of the question: "…of Canada?" gives "Economy of Canada", "…born?" gives a person.
- **Once the long match runs out, the tag drifts into wiki text.**
- **Both groups get the message.** 24 of these 31 abstentions are in-slice questions, which is why precision equals the base rate.

**The wiki form is unchanged** in every variant: the same right / abstained / wrong counts as A0. The don't-know text exists only in the chat form.

**The abstain gate at 2% wrong** still gives 0 right and 0 wrong (0% coverage) in every A1 variant. A1 adds no right answers that a gate could keep.

### Frozen test (200 questions, run once): A1b-after

The final A1 configuration is A1b-after: 20% don't-know turns, with the token after `Assistant: `. It was chosen on dev, as the only variant whose abstention tracks anything (precision 0.50 against 0.23–0.28 for the rest).

| Model | Raw R / A / W | Abstained on not-in-slice | Precision (Wilson lo) | Recall (Wilson 95%) | In-slice abstained | Gated (2% wrong) R / A / W |
|---|---|---:|---|---|---:|---|
| trie | 0 / 15 / 185 | 15/50 | **1.00** (0.80) | **0.30** (0.19–0.44) | 0/150 | 0 / 200 / 0 |
| trie + IG | 1 / 59 / 140 | 25/50 | 0.42 (0.31) | 0.50 | 34/150 | 0 / 200 / 0 |

Test abstention by token (trie, chat form):

| Token | In-slice | Not-in-slice |
|---|---|---|
| low | 0 (no question got `low`) | 15/15 abstained |
| mid | 0/48 | 0/19 |
| high | 0/102 | 0/16 |

- **The trie again abstains exactly on `low`.** On the test, no in-slice question happened to fall in `low`, so precision is 1.00.
- **Dev and test disagree.** On dev, `low` held 13 in-slice and 13 not-in-slice questions. Dev precision 0.50 has a Wilson interval of 0.32–0.68, so the two sets are consistent only at the edges. The test result is a lucky split of a weak statistic, not a better one.
- **Recall stays low:** 35 of the 50 not-in-slice test questions did not get `low`.
- **Tags (trie + IG):** 56 written, 12 complete, reason right on 24 (0.43), entity from the question 0.
- **Right answers: −1 against A0** (trie + IG).
  - "When did Alaska become part of the United States as a state?" was right in A0: *"Alaska became a state on January 3, 1959."*
  - With the token at the end of the prompt, it becomes *"Massachusetts became a state on February 6, 1788."* The token changes the prompt's last bytes, so InfiniGram's long match no longer reaches the stored Dolly answer.
  - The remaining right answer (Apollo 8 → "10 movies") is luck.
- **Parity.** The dev lines of this run are byte-identical to the dev-only A1b-after run.

---

## Predictions, scored

| Prediction | Result | Score |
|---|---|---|
| **D1.** A1a's abstention is nearly indiscriminate: precision within ±0.1 of the base rate of should-abstain questions | Base rate 0.231 (not-in-slice). Precision: A1a-5 trie+IG 0.226; A1a-20 trie 0.231, trie+IG 0.266. A1a-5 on the trie never abstains, so its precision is undefined; that is also no discrimination. On the should-abstain definition (base 0.98–1.00), precision is 0.98–1.00 | **Hit** |
| **D2.** A1b reaches abstain precision ≥ 0.8 on not-in-slice questions, with recall ≥ 0.5 | Dev:<br>• A1b as specified: 0.231 / 1.00 (trie), 0.250 / 0.43 (trie+IG)<br>• A1b-after: 0.500 / 0.43 (trie), 0.280 / 0.47 (trie+IG)<br>• the best possible with this token on dev is 0.50 / 0.43<br>Frozen test, A1b-after: 1.00 / 0.30 (trie), 0.42 / 0.50 (trie+IG). No configuration meets both criteria on either set | **Miss** |
| **D3.** Neither variant changes holdout BPC by more than 0.01 | Largest change +0.0033 (A1b-after, trie mixture); with InfiniGram ≤ +0.0008 | **Hit** |
| TOOL_PROFILE §2: the baseline is expected near zero | 0–2 right of 130 raw; 0% coverage at 2% wrong | **Hit** |
| CYPHALM_ANSWERING §4: A1 reaches abstain precision ≥ 0.8 on not-in-slice questions | Dev best 0.50. The test gives 1.00 (Wilson lo 0.80) but at recall 0.30, on a split that dev does not reproduce | **Miss** on dev. The test hit does not hold |

---

## Flags for Odin

**Ruled 2026-10-28.** Flag 1: the control token is allowed, on the gate side, like the Cypha-classifier abstain gate. Flag 2: the token goes after `Assistant: `. Odin also asked for an entity-level familiarity to be tried ([CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §5).

1. **The familiarity token is a design choice, and it needs your decision.** CyphaLM's own statistic about a question, computed from an earlier model state (after the wiki pass), is put into the prompt as a control token. That moves part of "deciding to abstain" from the byte model into a hand-placed number:
   - The statistic is CyphaLM's own, and the thresholds come only from training text.
   - But the cut points and the choice of statistic are mine, not learned.
   - It sits close to the Cypha-classifier gate that "counts as CyphaLM for now" (TOOL_PROFILE §1, DISCOVERY §8a).
   - Is a control token from the model's own statistics allowed under rule 10, and on which side of the gate decision does it fall?
2. **Token placement.** The specification puts the token before `Assistant: `. With an order-4 trie that cannot work: the 11 bytes of `Assistant: ` push it out of reach. I also ran the token after `Assistant: ` (the prompt ends with it), and used that as the final A1 configuration. The specified placement works only through InfiniGram, and there only loosely.
3. **Decoding change.** After an answer starts with "I don't know", decoding runs on to the end of the tag (240 bytes at most), so the tag can be scored. Nothing else changes.
4. **The two A0 "right" answers are lucky words in nonsense.** The T-QA containment rule counts them. I left the rule as defined.

---

## Verdict

- **A0.** CyphaLM alone does not answer T-QA.
  - On dev, at a 2% wrong rate, it answers nothing and gets nothing wrong.
  - On the test, the dev gate lets through 3 wrong answers. A gate tuned on one lucky right answer does not transfer.
- **A1.**
  - **The message is learned.** CyphaLM writes the message and tag itself.
  - **The decision is not.** Without a token it says "I don't know" at random, at about its training share (D1).
  - **The token is followed but weak.** With the token where the trie can see it, CyphaLM follows it exactly. But the token carries little information about whether a question is answerable:
    - dev: precision 0.50 at recall 0.43;
    - test: 1.00 at recall 0.30.
    - D2 missed.
  - **The tag's entity and reason are not right.** It cannot copy the entity from the question, and only one reason code was trained.
- **Nothing should become a default yet.** The don't-know text costs at most +0.003 BPC.

---

## Costs

| Item | Value |
|---|---|
| Code | `native/tools/cyphalm_trie.cpp` 779 → 1,765 lines (+986): `qa`, `dktext`, grading, gate, CV, and the selftest checks. `packed_trie.hpp` unchanged |
| `dktext` | 6 s, reads enwik8 (100 MB) and Dolly; output 3,963 rows (`dk.tsv`, about 0.8 MB) |
| `qa` run (train + holdout + InfiniGram + 130 questions × 2 forms × 2 models) | 350–400 s wall time on a shared core. With `--test` (330 questions): 331 s (A0) and 370 s (A1b-after) |
| Peak RSS | 233 MB (A0) to 265 MB (A1b-after) |
| Answer latency | Not timed separately; the whole dev decode of 520 answers plus question scoring is a small part of each run |
| External data | Only enwik8 and Dolly, as `chat` |
| `selftest` | Adds fixed-input checks of grading, tag parsing, Wilson bounds and the gate. Still about 0.1 s, and the golden BPC values are unchanged |

---

## Next steps

1. **A better familiarity statistic, before any more don't-know text.** Score the entity span, not the whole question, using the question parse (E1) to find it. Or use InfiniGram's longest match over the entity, or whether the entity occurs as a title or link in what CyphaLM read. Measure the AUROC on dev first: it must beat 0.60 by a wide margin before it is worth a token.
2. **Rule on flag 1** (the control token) before A1 goes further.
3. **Don't-know rows with other reason codes.**
   - `NO_EVIDENCE`: an entity that is present, with a relation the slice never states. This needs the F2 fact store to show the slot is empty.
   - `AMBIGUOUS_ENTITY`: titles with a "(disambiguation)" page.
   - Without these, reason-code accuracy cannot rise above the not-in-slice share of abstentions.
4. **The entity in the tag** needs a copy path from the prompt: a match expert over the current context, not only over the training bytes. That is a model change; it should be measured on BPC first.
5. **A0 needs a path to answers before a gate can do anything**: A2 (R1 question-form fact text) and A3 (R3 statement scoring). A gate with nothing right to keep will always choose 0% coverage.
6. **CMake.** `native/CMakeLists.txt` belongs to another agent. The B1 block's CTest `native_cyphalm_trie_selftest` already covers the new selftest checks; nothing new needs adding.

---

## Reproduce

```bash
gcc -O2 -c native/third_party/libsais/libsais.c -o libsais.o
g++ -O2 -std=c++20 -I native/include -I native/third_party -I native/third_party/libsais \
    native/tools/cyphalm_trie.cpp native/src/cyphalm/infinigram.cpp libsais.o -o cyphalm_trie
./cyphalm_trie selftest
E=/root/cdata/enwik8; D=/root/cdata/dolly.jsonl; DEV=bench/eval/t_qa_wiki_dev.tsv; TEST=bench/eval/t_qa_wiki.tsv
./cyphalm_trie dktext $E $D --dev $DEV --test $TEST --out dk.tsv
./cyphalm_trie qa $E $D --dev $DEV --out a0_rows.tsv                                              # A0
./cyphalm_trie qa $E $D --dev $DEV --dk dk.tsv --dk-ratio 0.05 --base a0_rows.tsv                 # A1a-5
./cyphalm_trie qa $E $D --dev $DEV --dk dk.tsv --dk-ratio 0.20 --base a0_rows.tsv                 # A1a-20
./cyphalm_trie qa $E $D --dev $DEV --dk dk.tsv --dk-ratio 0.20 --fam before --base a0_rows.tsv    # A1b
./cyphalm_trie qa $E $D --dev $DEV --dk dk.tsv --dk-ratio 0.20 --fam after  --base a0_rows.tsv    # A1b-after
# the frozen test, once, at the end:
./cyphalm_trie qa $E $D --dev $DEV --test $TEST --out a0test_rows.tsv
./cyphalm_trie qa $E $D --dev $DEV --test $TEST --dk dk.tsv --dk-ratio 0.20 --fam after --base a0_rows.tsv
```

stdout lines:
- `RAW` and `CV`: right / abstained / wrong, with Wilson bounds.
- `FULLDEV_GATE`: the gate the test uses.
- `RC`: the risk–coverage curves.
- `ABSTAIN`: A1 precision and recall, tags, reason codes and entities.
- `EX` and `RIGHT`: examples.
- `TEST_*`: the frozen test.

The per-question rows (`--out`) carry every feature and every answer. Run outputs are in the session scratch directory (`a01/`).
