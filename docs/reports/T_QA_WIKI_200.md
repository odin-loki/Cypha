# T-QA-wiki at 200 questions: a held-out test of the fact-store Q/A tool

> **Removed 2026-10-27.** The question-answering tool measured here did not use CyphaLM, so it broke [PLAN rule 10](../PLAN.md#2-rules-of-the-work) and its code was deleted. The numbers below are not CyphaLM results. See [REMOVED_NON_CYPHALM_QA.md](REMOVED_NON_CYPHALM_QA.md).

**Odin Loch** · 2026-10-27

## Summary

Yesterday `qa_infobox` scored 8/8 on the 8 seed questions. Those patterns were written after reading the seed articles, so that was an in-sample score. This report tests it on 200 new questions, written before I looked at the tool, and also scores the Dolly retrieval index (`chat_index`) on the same questions.

Everything is scored as right, abstained or wrong. Wrong is the failure we care about.

| Tool | In-slice (150): right / abstained / wrong | Not-in-slice (50): right / abstained / wrong |
|---|---|---|
| `qa_infobox` (Wikipedia fact store) | 3 / 147 / 0 | 0 / 50 / 0 |
| `chat_index` (Dolly retrieval) | 2 / 148 / 0 | 0 / 46 / 4 |

- `qa_infobox` is safe but almost silent. It answered 3 of 150 answerable questions (2.0%) and got all 3 right. It never answered a question it should not have.
- The 8/8 does not carry over. The plan target for this suite is at least 70% at 80% coverage. Measured coverage is 2%.
- `chat_index` behaves as expected on Wikipedia questions (it abstains on nearly everything), but it did answer 4 of the 50 questions that have no Wikipedia article. Three of those answers are true facts that Dolly holds. One is nonsense.

## How the questions were made

- **Corpus.** The first 7,340,032 bytes of `/root/cdata/enwik8`, the training range. It holds 915 complete pages, 535 of them not redirects. I listed all titles and read the article text.
- **150 in-slice questions.** They cover 113 different articles, spread across the whole list (questions per tenth of the title list: 27, 18, 14, 24, 26, 11, 7, 9, 10, 4). Types: who, what, where, when, which, how many. Subjects: capitals, authors, dates, places, people, inventions, definitions, numbers. Each has accepted variants.
  - For every question I checked by script that at least one variant occurs in the source article's text. The one exception is the alkane formula, which the page writes with HTML subscripts, so I checked it by hand.
  - Wording is mixed on purpose. Some questions are casual, lower-case, or have no question mark, as people type them.
- **50 not-in-slice questions.** The subject has no `<title>` among the 1,075 titles in the first 8 MiB. I also grepped the slice text for each subject and its answer. I replaced about 20 first-draft questions because the slice does state the answer somewhere. Examples: Uranus and Herschel, the capital of Kenya (a table in Africa), Harry Potter and Rowling, who founded Microsoft, the Berlin Wall date.
  - Two questions still have an entity mentioned in passing, with no answer given: Pride and Prejudice (Huxley wrote the 1940 screenplay) and the Mona Lisa (a theft entry).
- **Blindness.** I did not open `native/tools/qa_infobox.cpp`, `docs/reports/F1_INFOBOX_QA.md` or `bench/eval/t_qa_wiki_seed.tsv` until the 200 questions were saved. A directory listing showed the seed file's name, nothing more.
- **After the blind phase.** The seed covers eight articles: Atlas Shrugged, Algeria, Aristotle, Andorra, An American in Paris, Albania, Alexander the Great and Aardvark. Seven of those articles also appear in my set, but with different facts. Two of my questions ask the same relation as a seed item (see "Flawed questions").

File: `bench/eval/t_qa_wiki.tsv`. Columns are question, answers (`|` separated, empty for not-in-slice), source_title (empty for not-in-slice), in_slice (1/0). There is one header row. Rows are shuffled with a fixed seed.

## How it was scored

- **Match rule.** Case, accents and punctuation are ignored. An answer is right if it equals an accepted variant, or if either one contains the other, when at least one of the two is multi-word.
- **Not-in-slice.** Any answer counts as wrong, because abstaining is the correct behaviour.
- **`qa_infobox`.** I built it as given and ran it once over all 200 questions (a 2-column copy of my file in scratch). I parsed its `ANSWER:` or `ABSTAIN` per question.
  - With the tool's own exact-match scorer it prints 2 correct, 197 abstained, 1 wrong. The "wrong" one is `capital of Alberta?`, where it said "Edmonton, Alberta" and the accepted answer is "Edmonton". I count that as right.
  - Sanity check: this build still scores 8/8 on the seed file.
- **`chat_index`.** `--ask` takes one question per run, and each run re-runs the tool's whole evaluation sweep, about 52 s. 200 runs would take roughly three hours. So I compiled a scratch driver that `#include`s the unmodified `chat_index.cpp`, runs the same sweep once (same seed, same headline, plain / sum), builds the same index, and applies the same score and margin gate to each question. The repo tools are untouched and the driver is not committed. I cross-checked three questions against real `--ask` runs. Two matched to three decimals (Amazon.com and Madame Bovary) and the Don Quixote answer matched. Driver, outputs and scoring script are in the session scratch directory.
- **Intervals.** Wilson 95%. Coverage is answered / total. Selective accuracy is right / answered. Wrong-answer rate is wrong / all questions in the group.

## Results

### `qa_infobox`

| Group | n | Right | Abstained | Wrong | Coverage | Selective accuracy (Wilson lower) | Wrong rate (Wilson upper) |
|---|---|---|---|---|---|---|---|
| In-slice | 150 | 3 | 147 | 0 | 2.0% | 3/3 = 100% (43.8%) | 0/150 = 0.0% (2.5%) |
| Not-in-slice | 50 | 0 | 50 | 0 | 0.0% | n/a (nothing answered) | 0/50 = 0.0% (7.1%) |
| All | 200 | 3 | 197 | 0 | 1.5% | 3/3 = 100% (43.8%) | 0/200 = 0.0% (1.9%) |

- The 95% interval on in-slice coverage is 0.7% to 5.7%.
- Abstention on the not-in-slice set is 50/50, with a Wilson lower bound of 92.9%.
- The three answered questions:
  - `capital of Alberta?` gave "Edmonton, Alberta".
  - `Who wrote Animal Farm` gave "George Orwell".
  - `What is the capital of Austria?` gave "Vienna".
- Why the other 147 in-slice questions got no answer:
  - 142 matched no question template.
  - 5 matched a template, but the subject phrase did not match a title, or no fact was stored. Examples: "the words to the hymn Amazing Grace", "the novel A Clockwork Orange", and "who invented dynamite", where the fact sits in the Alfred Nobel article but no pattern reads it.
- The fact store has 141 facts from 915 pages.
- By a rough keyword count, about 35 to 40 of the 150 questions ask about a relation the store has (capital, author, director, birthplace, inventor, spouse, teacher). It answered 3 of them.

### `chat_index`

| Group | n | Right | Abstained | Wrong | Coverage | Selective accuracy (Wilson lower) | Wrong rate (Wilson upper) |
|---|---|---|---|---|---|---|---|
| In-slice | 150 | 2 | 148 | 0 | 1.3% | 2/2 = 100% (34.2%) | 0/150 = 0.0% (2.5%) |
| Not-in-slice | 50 | 0 | 46 | 4 | 8.0% | 0/4 = 0% (0%) | 4/50 = 8.0% (18.8%) |
| All | 200 | 2 | 194 | 4 | 3.0% | 2/6 = 33.3% (9.7%) | 4/200 = 2.0% (5.0%) |

- Abstention on the not-in-slice set is 46/50 = 92.0%, with a Wilson lower bound of 81.2%.
- The two in-slice answers came from Dolly, not from Wikipedia:
  - `capital of Alberta?` returned the stored "The capital city of the province of Alberta in Canada is Edmonton."
  - The Alaska statehood question returned "Alaska became a state on January 3, 1959."

## Every wrong answer

`qa_infobox`: none.

`chat_index` (all four are on not-in-slice questions, so the expected behaviour was to abstain):

| Question | Expected | Given | Score / margin | Note |
|---|---|---|---|---|
| Who wrote Don Quixote? | abstain | A yes/no list ("El Quijote" - yes, "The Man Who Killed Don Quixote" - no, ...) | 0.738 / 0.224 | Nearest stored question is a different task about Don Quixote titles. Nonsense answer, the real failure. |
| What is the capital of Brazil? | abstain | "Brasília is the capital of Brazil" | 1.000 / 0.102 | True fact; Dolly holds it verbatim. Wrong only under the abstain rule. |
| Who founded Amazon.com? | abstain | "Jeff Bezos" | 0.811 / 0.432 | True fact from Dolly ("Who founded Amazon?"). |
| Who was the first woman to win a Nobel Prize? | abstain | "Marie Curie was the first woman to win a Nobel Prize..." | 0.914 / 0.241 | True fact from Dolly. |

Dolly is a general chat set, so a correct answer to a general-knowledge question is not a hallucination. The real chat_index risk shown here is the one Don Quixote answer, which passed the gate at 0.738 / 0.224.

## Flawed questions

- **Not fully independent of the seed.**
  - "Which philosopher did Aristotle study with?" is the seed's Aristotle teacher relation, rephrased.
  - "Which famous conqueror did Aristotle teach?" is the inverse of seed item 7.
  - The tool abstained on both, although the seed phrasing "Who was Aristotle's teacher?" works. That is a clear example of phrasing fragility.
- **Time-bound answers.** True only as of the 2006 article: the mayor of Amsterdam, the French co-prince of Andorra, Agassi "married since 2001", and Alvin Toffler's wife.
- **Awkward wording.**
  - "Which invention is Alexander Graham Bell widely considered to have invented?" is circular.
  - "After Brazil, which is the second largest country in South America?" gives away half the answer.
- **Alkane formula.** The page stores it with HTML subscripts, so the normaliser matches it only loosely.
- **Short numeric answers.** Answers such as 10, 18, 30 and 95 could match by accident inside long free-text answers. No such accident occurred here.
- **Article count.** The set covers 113 articles, not about 100.

None of these changed a result. Only 3 in-slice questions were answered by `qa_infobox` and 2 by `chat_index`, and I checked all of them by hand.

## What this means

1. The 8/8 was in-sample. On 150 unseen but answerable questions the fact store gets 3 right (2.0%, 95% interval 0.7% to 5.7%). The seed score said nothing about generality.
2. The abstention rule holds: 0 wrong out of 200, with a Wilson upper bound of 1.9%. The cost is that the tool is silent on 98% of answerable questions.
3. The failure is mostly the question parser (142 of 150 got no template), then thin extraction (141 facts from 915 pages), then subject phrasing.
4. For `chat_index`, the abstain gate is good but not tight. 4 of 50 off-topic questions got through, and one of them was nonsense.

## Next steps

1. **Freeze the set and protect it.** Add `t_qa_wiki.tsv` to the frozen suites with a SHA-256 and README entry, as plan section 9 asks. Do not add patterns by reading these failures and then re-scoring on the same 200, because that makes it in-sample again. Split it: develop on the even-numbered rows, and keep the odd-numbered rows locked for the final number. Or have Odin write a fresh 100.
2. **Odin to review** the 200 questions and variants, as the plan says "Odin writes or reviews".
3. **Fix the parser first.** Strip type nouns and filler from subjects ("the novel", "the film", "the words to the hymn"). Add templates for birthplace, spouse, teacher and pupil, "which X did Y", and dates. This is the cheapest gain: 142 of 150 got no template.
4. **Extract more facts.** Read first-sentence definitions and "born ... in" patterns. For example, "who invented dynamite" lives in the Alfred Nobel article's opening sentence.
5. **Add a sentence-retrieval fallback over the article text with an abstain gate**, which is the plan's own mitigation for low pattern coverage. Then re-measure the right / abstained / wrong split. The target is at least 70% selective accuracy at at least 80% coverage, with the wrong-answer rate staying near 0.
6. **Tighten the `chat_index` gate** for near-lexical matches such as Don Quixote, and re-check with this not-in-slice set.
