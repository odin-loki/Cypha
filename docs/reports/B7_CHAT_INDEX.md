# B7: a fuzzy chat index that can abstain

> **Removed 2026-10-27.** The question-answering tool measured here did not use CyphaLM, so it broke [PLAN rule 10](../PLAN.md#2-rules-of-the-work) and its code was deleted. The numbers below are not CyphaLM results. See [REMOVED_NON_CYPHALM_QA.md](REMOVED_NON_CYPHALM_QA.md).

**Odin Loch** · 2026-10-27

**Questions.**
1. Can a stored-answer lookup survive a lowercased first letter, a " Thanks." and a swapped synonym?
2. Can it say "I do not have that" for a question it has never seen?

**Answers.**
1. Yes, at the B7 gate: every changed form keeps at least 95% of verbatim accuracy. With no abstain it is 0.989 or better. With abstain on it is 0.957 to 0.964 on the registered sample, a narrow pass that does not hold on every other sample (see Gate).
2. Mostly. It abstains on 92.6% of held-out Dolly questions. The 7.4% it still answers are mostly wrong or only loosely related. This is retrieval, not understanding.

Tool: [`native/tools/chat_index.cpp`](../../native/tools/chat_index.cpp). Context: [`CYPHALM_QA_AND_CONTEXT_TEST.md`](CYPHALM_QA_AND_CONTEXT_TEST.md) (the old `copied_answer()` fired on 975/975 verbatim prompts and 0/975 with the first letter lowercased). Plan row: B7 in [`../PLAN.md`](../PLAN.md).

---

## What it is

The question text of each stored Dolly record (instruction, plus context when there is one) is normalised and indexed. A query is normalised the same way, scored against every stored question, and the stored response of the best match is returned if the match is good enough. Otherwise it abstains. It stores no model and generates nothing.

| Item | Value |
|---|---|
| Normalisation | Lowercase, Unicode folded to ASCII where trivial (curly quotes, dashes, Latin accents), apostrophes deleted, other punctuation to space, whitespace collapsed. Optional light stemming and optional removal of leading/trailing courtesy words ("please", "thanks") were measured and are off in the headline |
| Scorer `bm25` | BM25, k1 = 1.2, b = 0.75, over word tokens. Score divided by the query's self-score (what an identical stored question would get), so it lies in [0, 1] |
| Scorer `cos3` | TF-IDF cosine over character 3-grams |
| Scorer `sum` | Mean of the two normalised scores. Headline |
| Abstain rule | Answer only if score >= ts and (best - runner-up) >= tm. The runner-up is the best match with a different question text, so an exact duplicate does not zero the margin |
| Data | `dolly.jsonl`, 15,015 records, prompts built as `load_dolly()` does |

## Protocol

| Set | Definition | Use |
|---|---|---|
| H | index % 10 == 3, 1,502 records. 49 removed because their normalised instruction equals an indexed one (near-duplicate leak). **1,453 used.** Never indexed | T-QA-dolly: the correct behaviour is to abstain |
| D | index % 10 == 7, 1,501 records, indexed | Dev only: picks (ts, tm). Positives: the five perturbed forms of each record (6,413 queries). Negatives: the verbatim question with its own entry left out (1,466 queries, those whose instruction is unique) |
| Train | the rest, 12,012 records, indexed | |
| P | 1,000 indexed records not in D, seed 20261027 | T-perturb |

T-perturb forms: verbatim; first letter lowercased (972 records change); " Thanks." appended to the instruction (1,000); punctuation deleted from instruction and context (931 change); one word of the instruction swapped for a synonym from a built-in table of 56 pairs (**only 362 of 1,000 have a replaceable word**; the rest are skipped). Accuracy means the returned response is the target's response.

Two operating points per scorer, both chosen on D only:

- **G (gate, headline).** Every changed form keeps at least 0.96 of the verbatim answered-and-correct count on D (0.95 plus 0.01 for sampling noise), precision on D's perturbed forms is at least 95%, at least 95% of verbatim hits are still answered, then abstain on as many leave-one-out dev questions as possible.
- **B (balanced).** Precision at least 95%, then the best mean of "perturbed answered correctly" and "leave-one-out abstained".

The 95% precision floor alone does not bind: top-1 is about 98% right at any threshold. That is why G adds the ratio and verbatim-kept conditions. The ratio is relative to verbatim at the same operating point, so without the verbatim-kept condition it could be met by abstaining on everything.

## Results (plain normalisation, N = 13,513 indexed questions)

### T-perturb

"always" is top-1 with no abstain. "G" is answered and correct over n at point G, with the ratio to verbatim on the same records in brackets.

| Form | n | bm25 always | bm25 G | cos3 always | cos3 G | sum always | sum G |
|---|---|---|---|---|---|---|---|
| verbatim | 1000 | 98.5% | 93.4% | 98.7% | 94.3% | 98.7% | 93.9% |
| first letter lowercased | 972 | 98.5% (1.000) | 93.3% (1.000) | 98.7% (1.000) | 94.1% (1.000) | 98.7% (1.000) | 93.8% (1.000) |
| " Thanks." | 1000 | 97.5% (0.990) | 90.0% (0.964) | 98.7% (1.000) | 94.1% (0.998) | 98.7% (1.000) | 91.0% (0.969) |
| no punctuation | 931 | 98.1% (0.997) | 92.3% (0.992) | 98.6% (1.000) | 93.3% (0.994) | 98.6% (1.000) | 92.5% (0.990) |
| synonym | 362 | 97.5% (0.989) | 91.4% (0.968) | 97.8% (0.992) | 91.7% (0.962) | 97.8% (0.992) | 91.2% (0.957) |

The lowercase row equals verbatim by construction: the normaliser lowercases. The informative rows are the other three. 18 of the 1,000 targets share their question text with another record that has a different response, so the always-answer ceiling is below 100%.

### Gate (every changed form >= 95% of verbatim)

| Scorer | No abstain | At G | At B |
|---|---|---|---|
| bm25 | PASS (0.989) | PASS (0.964, Thanks) | FAIL (0.917) |
| cos3 | PASS (0.992) | PASS (0.962, synonym) | FAIL (0.806) |
| sum | PASS (0.992) | PASS (0.957, synonym) | FAIL (0.924) |

Verdict: **PASS with no abstain, PASS at G on the registered sample, FAIL at B.** The G pass is narrow. After the headline was fixed I re-drew P with seeds 1, 2 and 3 (not used for any choice). The minimum changed-form ratio at G was:

| Seed | bm25 | cos3 | sum |
|---|---|---|---|
| 20261027 (registered) | 0.964 | 0.962 | 0.957 |
| 1 | 0.965 | 0.952 | 0.966 |
| 2 | 0.956 | 0.939 | 0.942 |
| 3 | 0.975 | 0.967 | 0.964 |

`bm25` passes on all four, `cos3` and `sum` on three of four. The weak form is the synonym swap for `cos3` and `sum` (and " Thanks." for `bm25` on one seed). The synonym form has only about 360 records, so a few records move its ratio by a point. `sum` is the headline because it had the best dev score, not because it is the most stable.

### Held-out H (1,453 questions)

Forced to answer every question, the mean token-F1 against the reference is 0.17 to 0.18 and 5% of answers exceed 0.5.

| Scorer | Point | Abstain rate | Answered | Mean F1 (answered) | Answered with F1 > 0.5 | Answered with F1 > 0.5, of all H |
|---|---|---|---|---|---|---|
| bm25 | G | 88.9% | 161 | 0.248 | 9.9% | 1.1% |
| bm25 | B | 88.2% | 171 | 0.252 | 10.5% | 1.2% |
| cos3 | G | 91.9% | 117 | 0.276 | 12.0% | 1.0% |
| cos3 | B | 95.0% | 72 | 0.258 | 11.1% | 0.6% |
| sum | G | **92.6%** | 107 | 0.276 | 13.1% | 1.0% |
| sum | B | 91.7% | 120 | 0.267 | 11.7% | 1.0% |

Abstaining removes most of the wrong answers and raises the mean F1 of what is left only from 0.175 to 0.276. About 87% of the answers that remain still have F1 of 0.5 or less. On D the leave-one-out abstain rate at G is 93.9% for `sum`, close to the 92.6% on H.

### Risk-coverage (`sum`, the 4,265 T-perturb form queries pooled, margin gate fixed at tm = 0.100)

| ts | Coverage | Accuracy of answered | H abstain | Mean F1 of H answered |
|---|---|---|---|---|
| 0.000 | 94.9% | 98.7% | 82.1% | 0.238 |
| 0.500 | 94.9% | 98.7% | 86.5% | 0.257 |
| 0.700 | 94.2% | 98.8% | 91.9% | 0.276 |
| **0.730 (G)** | 93.8% | 98.8% | 92.6% | 0.276 |
| 0.800 | 91.5% | 98.8% | 95.2% | 0.273 |
| 0.850 | 86.5% | 98.9% | 95.9% | 0.251 |
| 0.900 | 78.3% | 98.8% | 96.7% | 0.270 |
| 0.950 | 71.0% | 98.7% | 97.8% | 0.319 |
| 0.990 | 64.1% | 98.7% | 99.4% | 0.290 |

Accuracy of answered barely moves (98.7% to 98.9%) while coverage falls from 94.9% to 64.1%: on the perturbed forms the threshold discards correct matches about as often as wrong ones, so raising it buys abstention on new questions and costs coverage almost one for one. The curves for `bm25` and `cos3` are printed by the tool.

### Normalisation ablation (`sum`)

| Config | Dev score | Verbatim at G | Min ratio, no abstain | Min ratio at G | Min ratio at B | H abstain at G |
|---|---|---|---|---|---|---|
| plain (headline) | 0.939 | 93.9% | 0.992 | 0.957 | 0.924 | 92.6% |
| + stem | 0.931 | 94.3% | 0.992 | 0.954 | 0.918 | 91.7% |
| + courtesy-word filter | 0.939 | 93.8% | 0.992 | 0.957 | 0.755 | 92.6% |
| + stem + filter | 0.931 | 94.3% | 0.992 | 0.954 | 0.820 | 91.7% |

Stemming lowers the dev score and helps nothing. The filter ties on dev and makes the " Thanks." row exactly 1.000, but it is a rule written with that test form in mind, so I treat it as fitted and left it off.

### Speed and memory (plain, single thread, 2,453 queries: T-perturb verbatim plus H)

| Item | Value |
|---|---|
| Index build | 1.03 s (BM25 postings 0.28 s, 3-gram postings 0.62 s, the rest tokenising) |
| Index size, serialised equivalent | 27.99 MB: BM25 5.32 MB, 3-gram 22.51 MB, document tables 0.16 MB |
| Response store | 4.92 MB |
| Peak RSS, whole process | about 158 MB (includes the parsed Dolly records and, for later configurations, earlier indexes) |

| Scorer | p50 | p99 | p50 / p99, instruction only | p50 / p99, with context |
|---|---|---|---|---|
| bm25 | 0.06 ms | 0.58 ms | 0.05 / 0.15 ms | 0.19 / 0.95 ms |
| cos3 | 0.20 ms | 2.30 ms | 0.15 / 0.69 ms | 0.90 / 3.10 ms |
| sum | 0.34 ms | 3.19 ms | 0.27 / 1.06 ms | 1.35 / 4.01 ms |

Every query scores all 13,513 stored questions; there is no pruning.

## Honest limits

- **This is retrieval.** It returns the stored answer to the nearest stored question. It does not read, understand or compose. If a user asks something that is close to a stored question but differs in the one detail that matters, it can return the stored answer to the wrong question. The margin gate reduces this and cannot remove it.
- **New questions.** On held-out Dolly the correct behaviour is to abstain and it does so 92.6% of the time. Of the answers it still gives, about 87% have F1 of 0.5 or less. Even after the near-duplicate filter, some H items have paraphrase twins in the index, so the F1 above 0.5 cases are partly near-duplicate retrieval, not generalisation.
- **Perturbations are mechanical.** Real paraphrases are harder. The synonym form covers only 36% of sampled records and only a 56-pair table; its ratio is the least stable number here (see seeds). The punctuation form deletes characters in the instruction and context, which differs from how the normaliser treats hyphens and slashes, so it is a real test but not an exhaustive one.
- **No encoder.** The plan's B7 row also lists encoder similarity and an abstain threshold from Cypha OOD. Neither is built. `cos3` is a lexical stand-in, and the threshold comes from dev-set scores. Both are still to do.
- **OpenAssistant is not included.** Hugging Face downloads are blocked here, and I did not find the data on GitHub in the time allowed (the GitHub repository was not reachable from this session, and the data lives on Hugging Face). Dolly only. The plan's claim 3 (two Q/A sets) is therefore still open.
- **Thresholds depend on D.** D is indexed, so its leave-one-out queries slightly overstate how new they are (the term counts still include the held-out entry). H was never used to choose anything.
- **One run per configuration.** No confidence intervals beyond the three extra P samples above. The seeds 1 to 3 were run after the headline was fixed and were not used to choose it.

## Reproduce

```
g++ -O2 -std=c++20 -I native/third_party native/tools/chat_index.cpp -o chat_index
./chat_index /root/cdata/dolly.jsonl                        # all four configurations, about 55 s
./chat_index /root/cdata/dolly.jsonl --configs plain --seed 2
./chat_index /root/cdata/dolly.jsonl --configs plain --ask "why is the sky blue?"
```

The last command answers a lowercased "why is the sky blue?" from Dolly (score 1.000), and abstains on "Who wrote the novel Zorblax and the Moon Pearls?" (score 0.320). The run prints every table in this report.
