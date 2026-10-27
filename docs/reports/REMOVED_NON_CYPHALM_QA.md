# Removed: question answering that did not use CyphaLM

**Odin Loch** · 2026-10-27

**Rule** ([PLAN rule 10](../PLAN.md#2-rules-of-the-work)): every task is done by CyphaLM. A string search, retrieval index or hand-written lookup that answers in CyphaLM's place is cheating. Its output is not a Cypha result.

## What was removed

| Code | What it did | Why it broke the rule |
|---|---|---|
| `copied_answer()` in `native/tools/trie_chat.cpp` (and its use in `trie_qa_probe.cpp`) | `std::string::find` of the whole prompt in the training bytes, returning the bytes after it | A string search, not the model. It produced the original "3 answered questions" |
| `native/tools/qa_infobox.cpp` | Parsed infobox fields and a few sentence patterns into a fact table, matched questions to it with regex templates, and printed the stored value | Hand-written extraction and lookup. CyphaLM was not involved |
| `native/tools/chat_index.cpp` | BM25 and character-3-gram retrieval over Dolly prompts, returning the stored response | A retrieval index. CyphaLM was not involved |

The work-in-progress job that was raising `qa_infobox` coverage was stopped, and its code changes were discarded. The question set it wrote, `bench/eval/t_qa_wiki_dev.tsv` (130 questions), is data, not an answerer, so it is kept as a development set for CyphaLM.

## Results that are not CyphaLM results

These numbers came from the removed tools. They must not be cited as Cypha or CyphaLM results:

- **[F1_INFOBOX_QA.md](F1_INFOBOX_QA.md):** 8/8 on the seed questions.
- **[B7_CHAT_INDEX.md](B7_CHAT_INDEX.md):** about 99% on reworded stored questions; 92.6% abstention on new ones.
- **[T_QA_WIKI_200.md](T_QA_WIKI_200.md):** the tool scores (3 / 197 / 0 and 2 / 194 / 4). The 200 blind questions themselves stay as the frozen test for CyphaLM.
- **[S1_S2_EPISTEMIC.md](S1_S2_EPISTEMIC.md):** the `chat_index` reason codes and fast path. The `epistemic.hpp` library stays.

## What stays

These are CyphaLM results, measurement tools, libraries or data. None of them answers questions in CyphaLM's place.

- **CyphaLM results:** the packed trie and its 8K count-context test ([CYPHALM_QA_AND_CONTEXT_TEST.md](CYPHALM_QA_AND_CONTEXT_TEST.md)). CyphaLM's own Q/A score so far: 0/8 on the Wikipedia facts, 1/30 on held-out Dolly.
- **Libraries:** ECS, LCI, epistemic, SCRIBE.
- **Data and support tools:** the word vectors, the tagger, the question classifier, the pattern learner and the frozen test sets.

## What comes next

CyphaLM has to answer by itself, and its score is reported on `bench/eval/t_qa_wiki.tsv` as right / abstained / wrong. How CyphaLM uses facts, patterns and the tagger to do that is the next design step. It will be agreed with Odin Loch before it is built.
