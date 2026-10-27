# E1: an averaged-perceptron tagger reaches 94.81% UPOS on the EWT test split, and a rule chunker finds nominal NPs at F1 83.8

Odin Loch, 2026-10-27

Tool: `native/tools/pos_tagger.cpp` (standalone C++20). Plan rows: Track E, E1 (and the rule part of E2); test T-syntax, POS accuracy (section 9).

## Result

| Measure | Value |
|---|---|
| **UPOS accuracy, EWT test (gate: 94%)** | **94.81%** (23,792 / 25,094). **Gate passed** by 0.81 points. The binomial 95% interval is roughly 94.5 to 95.1; tokens in a sentence are not independent, so the true interval is somewhat wider, but the gate still sits well outside it |
| Known words / unknown words (test) | 96.38% (22,802 tokens) / 79.23% (2,292 tokens, 9.1% of the test) |
| Question sentences (test, 170 sentences, 1,927 tokens) | 93.25% |
| XPOS accuracy (test) | 94.13% |
| NP chunker F1, exact span, predicted UPOS | dev **85.67**, test 83.78 |
| NP chunker F1 with gold UPOS (ceiling of the rules) | dev 90.14, test 88.09 |
| Speed, single thread, token hashing included | about 1.0M tokens/s for UPOS; about 0.5M tokens/s for UPOS and XPOS together |
| Model size | 5.6 MB for UPOS and XPOS (2.8 MB UPOS only); training takes 12 s |

The published reference I was given is that averaged-perceptron taggers score approximately 94 to 96% UPOS on EWT. I did not verify that source. 94.81% sits at the lower end of that range, so this is a competent baseline and not a surprise.

Read this with four cautions.

1. **The test split was run once**, after the tagger code and the chunker rules were frozen. The test numbers above come from that run. Epoch count, tag-dictionary settings and the chunker rules were chosen on dev only.
2. **The chunker rules were adjusted once after I read the dev mismatches** (commas and hyphens inside modifier chains, participles used as adjectives, and trimming stray punctuation from the gold spans). Dev F1 rose from 85.33 to 85.67, so the effect was small. The test F1 is the cleaner number and it is lower: 83.78.
3. **The "gold NPs" are my definition**, read off the dependency trees (below). The F1 measures agreement with that definition, not with a published chunking standard.
4. **The question demonstration is in-sample.** The rules were written while looking at the first three questions Odin gave. Ten examples show behaviour; they are not an accuracy estimate.

## Data

Universal Dependencies English EWT, from `raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/`, fetched 2026-10-27. The repository README header reads **v2.18, 2026-05-15**. I could not read the commit hash (the GitHub API is not reachable from this session), so I recorded checksums instead. The license is CC BY-SA 4.0 (plan section on data).

| Split | Sentences | Tokens (syntactic words) | Question sentences | SHA-256 (first 16 hex) |
|---|---|---|---|---|
| train | 12,544 | 204,578 | 795 | `d68e06122a702464` |
| dev | 2,001 | 25,148 | 170 | `39239e0a60db3ae6` |
| test | 2,077 | 25,094 | 170 | `fa024f43dc5da3c5` |
| total | 16,622 | 254,820 | | |

Tokens are the word lines. Multiword-token range lines (for example "can't" written as `ca` + `n't`) and empty-node lines (37 in train, 4 in dev, 2 in test) are skipped. Tagging is evaluated on the gold tokenisation. A question is a sentence whose `# text` line ends in `?`. A known word is an exact (case-sensitive) form that occurs in train. Files are saved in `/tmp/claude-0/-home-claude/73363f2d-2615-591f-aa38-d54363629bae/scratchpad/ud/`.

## Method

**Tagger.** Averaged perceptron (Collins 2002), greedy left to right, one model per tag set (17 UPOS tags, 49 XPOS tags). Weights are averaged with the timestamp method. Scores start at zero, ties go to the first tag. Features:

| Group | Features |
|---|---|
| Word | word; lowercase word; prefixes and suffixes of 1, 2 and 3 characters (UTF-8 aware); word shape with runs capped at 2 (`Xxx`, `dd`, `x-x`) |
| Flags | has hyphen; has digit; has capital (initial or mid-sentence); all caps |
| Context | previous and next word (lowercase); words at distance 2; suffix of the previous and next word; bigrams previous+current and current+next; shapes of both neighbours |
| Question | if the sentence ends in `?`, the lowercase word again as a separate feature |
| History | previous tag; previous two tags; previous tag with the word; previous tag with the 3-character suffix; previous tag with the next word |

Features are hashed to 64 bits, so the model stores no strings. A tag dictionary covers words seen at least 20 times with one tag at least 97% of the time (685 UPOS words covering 47.1% of train tokens; 605 XPOS words). Those tokens skip the scorer and are not trained on. Beam search was not built.

**Training protocol.** Up to 40 epochs, training order reshuffled each epoch (fixed seeds), patience 6. After each epoch the averaged weights score dev and the best epoch is kept: epoch 13 for UPOS (dev 94.81%), epoch 9 for XPOS (dev 94.29%). The saved file is int16-quantised; `train` reloads it and rescores dev (94.81% and 94.29%, unchanged), so the saved model is the measured one.

**Chunker.** Rules over the predicted UPOS tags:

- **NP:** `DET? DET? (NUM | ADJ | ADV* ADJ | participle)* (NOUN | PROPN)+`. A possessive pronoun (`my`, `his`, `whose` ...) counts as a determiner. A comma between two adjectives and a hyphen inside a modifier chain are allowed. A lone `PRON` is a separate pronoun NP.
- **Verb group:** a run of `AUX`/`VERB`, with adverbs or `not`/`n't` allowed between; `to` starts a new group.
- **PP:** `ADP` followed by an NP.

**Gold NPs from the trees.** For each `NOUN`/`PROPN` that is not a compound or flat part of another noun, take it with its `det`, `det:predet`, `amod`, `nummod`, possessive-pronoun, `compound` and `flat` dependents (and the adverbs of an included adjective). Fill punctuation inside the span, trim it at the edges, and keep the contiguous run that contains the head. Scoring is exact span match over a sentence's set of spans. I report nominal NPs as the main figure and a variant that adds single-token pronoun NPs, because the pronouns are easy and raise F1.

## Tagger results

| Split | UPOS | XPOS | UPOS known | UPOS unknown | UPOS questions |
|---|---|---|---|---|---|
| dev | 94.81% (23,844 / 25,148) | 94.29% | 96.27% | 78.73% (2,087 tokens) | 93.29% (1,953 tokens) |
| **test** | **94.81%** (23,792 / 25,094) | 94.13% | 96.38% | 79.23% (2,292 tokens) | 93.25% (1,927 tokens) |

Dev and test UPOS are both 94.81%. That is a coincidence of the counts, not a copied number (23,844 / 25,148 and 23,792 / 25,094).

The largest test confusions (gold to predicted), 1,302 errors in all:

| Gold to predicted | Count | Gold to predicted | Count |
|---|---|---|---|
| PROPN to NOUN | 194 | NOUN to ADJ | 47 |
| NOUN to PROPN | 186 | ADJ to VERB | 46 |
| ADJ to NOUN | 72 | SCONJ to ADP | 40 |
| VERB to NOUN | 65 | ADV to ADJ | 38 |
| NOUN to VERB | 54 | ADJ to PROPN | 34 |

The PROPN/NOUN pair alone is 380 errors, 29% of all. Unknown words are 9.1% of tokens but 37% of errors (476 of 1,302).

**Dev-only sweep (UPOS, no XPOS)** to pick the tag-dictionary setting and test pruning:

| Setting | Dev UPOS | Model size |
|---|---|---|
| dictionary 20 / 0.97 (kept; 685 words) | 94.81% | 2.77 MB |
| no dictionary | 94.85% | 2.87 MB |
| dictionary 10 / 0.99 (1,390 words) | 94.75% | 2.71 MB |
| dictionary 30 / 0.95 (508 words) | 94.67% | 2.74 MB |
| dictionary 20 / 0.97, drop weights below 0.05 / 0.2 | 94.81% / 94.81% | 2.75 / 2.71 MB |

All settings sit within 0.2 points, which is noise at this dev size (about 0.14 points standard error). The dictionary costs about 0.04 points and buys speed.

**Speed** (single thread, best of 3 runs over the 25,094 test tokens, pre-tokenised, token hashing included; repeat runs vary by about 10%):

| Model | Hashing + UPOS | Hashing + UPOS + XPOS |
|---|---|---|
| with tag dictionary (the shipped model) | 1.07M tokens/s | 0.51M tokens/s |
| trained with `--no-dict` (dev, UPOS 94.85%) | 0.64M tokens/s | 0.28M tokens/s |

The dictionary gives roughly 1.5 times the speed. A model trained with the dictionary cannot be run with it switched off: those words have untrained weights (dev UPOS fell to 80.98% when I tried). For that reason the `eval` command has no such switch, and a no-dictionary model has to be trained as one.

## Chunker results

| Measure | Gold | Predicted | P | R | F1 |
|---|---|---|---|---|---|
| **Dev**, nominal NPs, predicted UPOS | 5,012 | 4,947 | 86.23 | 85.12 | **85.67** |
| Dev, nominal NPs, gold UPOS (rule ceiling) | 5,012 | 4,921 | 90.98 | 89.33 | 90.14 |
| Dev, nominal and pronoun NPs, predicted UPOS | 6,920 | 6,863 | 89.57 | 88.83 | 89.20 |
| Test, nominal NPs, predicted UPOS | 5,002 | 4,895 | 84.70 | 82.89 | 83.78 |
| Test, nominal NPs, gold UPOS | 5,002 | 4,872 | 89.27 | 86.95 | 88.09 |
| Test, nominal and pronoun NPs, predicted UPOS | 6,834 | 6,759 | 88.13 | 87.17 | 87.65 |

Tagging errors cost about 4.5 F1 points (90.14 to 85.67 on dev). The rules themselves lose about 10 points against the tree-derived spans. Reading the dev mismatches, most are conventions of the treebank and not rule bugs: titles and names split into two NPs (`President` / `Bush`, `Sheikh` / `Ahmed Yassin`), appositions (`the Prophet` / `Mohammed`), and quoted or parenthesised pieces inside a name. The rule chunker joins a run of proper nouns into one NP; the trees often do not.

## Question parse examples

`pos_tagger question <model>` prints these fields; `--tsv` prints one tab-separated line per question. The table condenses the tool's output for the shipped model (the full block for question 3 is printed under it). "Focus" is an extra field: the wh head noun if there is one, otherwise the predicate noun after a copula, because for "What is the capital of Algeria?" the wh-phrase itself has no noun.

| # | Question | wh | wh-phrase | wh head | aux | verb | focus | entity (how found) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Who designed the AK-47? | Who | Who | - | - | designed | - | the AK-47 (last NP) | correct; `AK-47` was tagged NOUN, so the fallback rule found it |
| 2 | What is the capital of Algeria? | What | What | - | - | is | capital | Algeria (PROPN run) | correct |
| 3 | In what year did the Anschluss of Austria happen? | what | In what year | year | did | happen | year | Austria (PROPN run) | partly: the entity should be "the Anschluss of Austria"; two single-token PROPN runs tie and the later wins |
| 4 | Where was Napoleon Bonaparte born? | Where | Where | - | was | born | - | Napoleon Bonaparte (PROPN run) | correct |
| 5 | Which river flows through Cairo? | Which | Which | - | - | river | - | Cairo (PROPN run) | **wrong**: `river` was tagged VERB, so wh head is lost and the verb is wrong |
| 6 | How many moons does Mars have? | How | How many moons | moons | does | have | moons | Mars (PROPN run) | correct |
| 7 | When did the Berlin Wall fall? | When | When | - | did | fall | - | the Berlin Wall (PROPN run) | correct |
| 8 | Who was the first president of the United States? | Who | Who | - | - | was | president | the United States (PROPN run) | correct (verb is the copula, aux empty) |
| 9 | What is the highest mountain in South America? | What | What | - | - | is | mountain | South America (PROPN run) | correct |
| 10 | Who wrote the novel Pride and Prejudice? | Who | Who | - | - | wrote | - | Prejudice (PROPN run) | partly: the title is split at "and" and `Pride` was tagged NOUN |

Wh-word, auxiliary and main verb are right on 9 of 10. The wh head noun is right on questions 3 and 6 and missed on question 5. The entity is right on 8 of 10 and partial on 3 and 10. One full output block, as printed by the tool:

```
Q: In what year did the Anschluss of Austria happen?
  tags      : In/ADP what/DET year/NOUN did/AUX the/DET Anschluss/PROPN of/ADP Austria/PROPN happen/VERB ?/PUNCT
  chunks    : [PP In [NP what year]] [VG did] [NP the Anschluss] [PP of [NP Austria]] [VG happen] ?
  wh        : what
  wh_phrase : In what year
  wh_head   : year
  aux       : did
  verb      : happen
  focus     : year
  entity    : Austria  [propn-run]
```

How the helper decides. The wh-word is the first of who, whom, whose, what, which, when, where, why, how. The wh-phrase takes a preceding preposition ("In what year"), a following adjective after "how", and the NP that starts at the wh-word. The auxiliary and main verb are found by scanning right of the wh-phrase until a clause boundary (punctuation, a conjunction, "that", another wh-word): auxiliaries are collected, the first `VERB` is the main verb, and if no `VERB` appears the last auxiliary is the copula and becomes the main verb.

## Limits

- **Gold tokenisation only.** The `tag` and `question` commands use a small rule tokeniser (splits punctuation and clitics such as `n't`, `'s`, `'re`; keeps hyphenated words whole). It was not evaluated. EWT itself sometimes splits hyphenated names (`Al - Zaman`), which this tokeniser would not.
- **Unknown words and proper nouns are the weak spot.** 79% on unknown words and 380 PROPN/NOUN confusions. Sentence-initial and lowercase names (`AK-47`) are the usual failures.
- **Questions are slightly worse** (93.25% against 94.81%). EWT has few questions, and wh-determiners are fragile: in "Which river flows through Cairo?" a noun after `Which` was tagged VERB, and the question helper inherits that.
- **The rule chunker is shallow**: no conjunction handling, no NPs headed by numbers or adjectives, no possessive `'s` merging, bare years like `in 2026` are not NPs, and tree conventions that split names are not reproduced.
- **The question parse is heuristic.** Ties between PROPN runs go to the later run, multiword titles split at function words, and nested relative clauses are only cut at "that" or another wh-word.
- **No Cypha classifier yet.** The plan names this averaged perceptron as the baseline. E1's Cypha version (dense C-vectors, lexicon candidates from the D2 pack) is not built. The section 11 fallback (below 90% after two design rounds) was not triggered.
- **The model file is not in the repository.** It trains in 12 s from the commands below.
- **Not done:** no significance test between settings, and no second shuffle seed.

## Next steps

1. Compare this baseline with the Cypha DIF classifier on the same features plus C1 vectors, to see whether dense features close the gap on unknown words.
2. Add lexicon POS candidates from the D2 pack as features. This is the most direct fix for unknown words and for PROPN/NOUN.
3. Add training questions (the T-QA question sets once they exist, tagged by hand or by a treebank of questions) so that wh-determiners and inverted auxiliaries are learned. Measure on the 170 EWT dev questions.
4. E2: BIO tagging for entities (PERSON, PLACE, ORG, DATE, NUMBER) on the same machinery, replacing the PROPN-run rule for the question entity; add multiword-title handling.
5. Feed the tagger into the S2 tag-plausibility metric (tag 4-gram over UD trees) and into the question classifier (wh-word, focus noun, entity as features).
6. Try beam search of width 3 and a bidirectional second pass; expect a few tenths of a point, not more.

## Reproduce

```
g++ -O2 -std=c++20 native/tools/pos_tagger.cpp -o pos_tagger
UD=https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master
for s in train dev test; do curl -sSfO $UD/en_ewt-ud-$s.conllu; done
./pos_tagger train en_ewt-ud-train.conllu en_ewt-ud-dev.conllu pos_model.bin --epochs 40 --patience 6
./pos_tagger eval pos_model.bin en_ewt-ud-dev.conllu
./pos_tagger eval pos_model.bin en_ewt-ud-test.conllu     # run once
echo "Who designed the AK-47?" | ./pos_tagger question pos_model.bin
echo "The cat sat on the mat." | ./pos_tagger tag pos_model.bin --xpos --chunks
```

Training is deterministic (fixed shuffle seeds). If UD publishes a release after v2.18, `master` will move and the numbers will too; the checksums above identify the files used.
