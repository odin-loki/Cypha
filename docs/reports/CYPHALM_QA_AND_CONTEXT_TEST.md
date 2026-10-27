# CyphaLM question answering and count-context test

> **2026-10-27:** `surface_controls.cpp` was removed to cut code debt; the SCRIBE ladder (`scribe_ladder`) now scores surface S on the controls. It is in git at commit `ab4f9f6` for exact reproduction of section 3. The find-based `copied_answer()` measured in section 1 was removed under PLAN rule 10 ([record](REMOVED_NON_CYPHALM_QA.md)).

**Odin Loch** · 2026-10-26

**Questions.**
1. Does the packed trie answer questions, or does something else?
2. Does a longer count context (order 8192) change that?
3. Does the surface score tell sense from nonsense?

**Answers.**
1. A string search answers them. The trie cannot see the question.
2. No. Context helps BPC up to about 16 bytes and then stops. It does not answer new questions.
3. No.

The forward plan built on these results is [`../PLAN.md`](../PLAN.md).

---

## Setup

| Item | Value |
|---|---|
| Text | First 8 MiB of enwik8, which equals enwik9's first 8 MiB. enwik8 MD5 `a1fa5ffddb56f4953e226637dabbb36a` |
| Split | Train `[0, 7340032)`, holdout `[7340032, 8388608)`, as in `cyphalm_trie chat` (then `trie_chat.cpp`) |
| Chat | Dolly 15k, `databricks-dolly-15k.jsonl` from `databrickslabs/dolly`, 15,015 records loaded. `CURRENT.md` reports 15,011, so this copy differs slightly. The chat model (`cyphalm_trie chat`, then `trie_chat.cpp`) loads Dolly only |
| Model | The packed trie and six-expert mixture exactly as `cyphalm_trie chat` (then `trie_chat.cpp`): wiki pass, `<\|doc\|>` break, two Dolly passes, Synth mode. Reproduced weights: 0.961 on order 4, 0.039 on the match |
| Machine | 2 cores, 7 GB RAM, g++ 13.3, `-O2` |

**Tools.** The test was run with `trie_qa_probe.cpp` and `trie_context_8k.cpp`, which reused `trie_chat.cpp`'s model through `#include`. On 2026-10-27 (PLAN B1) all three became subcommands of `native/tools/cyphalm_trie.cpp` over `native/include/cypha/cyphalm/packed_trie.hpp`, with byte-identical output ([B1 report](B1_TRIE_CONSOLIDATION.md); the old files are in git `9f35798` and `ab4f9f6`).

| Subcommand | What it does |
|---|---|
| `cyphalm_trie probe` | Seen, one-letter-changed and Wikipedia-fact questions; what the mixture writes on its own (greedy, and nucleus 0.9). The copy rate over 975 prompts used `copied_answer()`, which was removed under PLAN rule 10 ([record](REMOVED_NON_CYPHALM_QA.md)) |
| `cyphalm_trie context8k` | `InfiniGram` (order up to 8192) mixed with the trie; BPC at five context caps; matched-length histogram; 50 questions with 30 Dolly turns held out |
| `native/tools/surface_controls.cpp` | Surface score `S` on controls, and on any `GEN` lines piped in. Since retired in favour of `scribe_ladder` (git `ab4f9f6`) |

```bash
gcc -O2 -c native/third_party/libsais/libsais.c -o libsais.o
g++ -O2 -std=c++20 -I native/include -I native/third_party -I native/third_party/libsais \
    native/tools/cyphalm_trie.cpp native/src/cyphalm/infinigram.cpp libsais.o -o cyphalm_trie

./cyphalm_trie probe     enwik8 dolly.jsonl > qa.txt
./cyphalm_trie context8k enwik8 dolly.jsonl > ctx.txt
```

Each of the two model runs takes about 4–6 minutes on one core.

---

## 1. What answers the three questions

`trie_chat.cpp`'s three "answered" questions are `turns[0]`, `turns[n/2]` and `turns[n-1]`, all of them training turns:

- "When did Virgin Australia start operating?"
- "Name some of the bravolebrities from Southern Charm"
- "What is the Masters?"

`show()` prints `copied_answer(data, prompt)`. That function runs `std::string::find` for the whole prompt over the training bytes and returns the bytes that follow it.

| Probe | Result |
|---|---|
| Copy on the three seen prompts | 3/3, byte-identical to the stored answer |
| Copy with the first letter lowercased | 0/3 |
| Copy over 975 random Dolly prompts | **975/975 verbatim, 0/975 with one letter lowercased** |
| The mixture on its own, greedy, after any of the 14 prompts | `The states and the states and …`, the same for all 14 |
| The mixture on its own, nucleus 0.9 at T = 0.8 | `Her 2004 by General launchemic both the locations …`, the same for all 14 |
| Expected answer in the mixture's output | 0/14 |

Every prompt ends in `\nAssistant: `. Orders 1–4 and the 4-byte match see only `nt: `, so the question never reaches the prediction. This agrees with what `CURRENT.md` already says: "The copy is a search for that prompt, not the order-4 model reading the question."

---

## 2. An order-8192 count model on top of the trie

`InfiniGram` takes the last ≤ cap bytes, finds the longest suffix that occurs in the training bytes (wiki train + chat, with no holdout), and counts the byte that followed every occurrence. With cap 8192 it is an order-8192 count model that backs off only when the longer context never occurred.

Mixture: `P = λ·P_ctx + (1 − λ)·P_trie`, with one λ per (match length × count) bucket. The λs are fit on the first half of the holdout and scored on the second half. 30 Dolly turns (index % 500 = 250) were held out of both the trie and the index.

### BPC on the second half of the holdout

| Context cap (bytes) | BPC |
|---:|---:|
| trie only | 2.3199 |
| 4 | 2.2660 |
| 16 | **2.0637** |
| 64 | 2.0711 |
| 512 | 2.0715 |
| 8192 | 2.0715 |

The whole gain arrives by 16 bytes. Past that, the extra buckets cost a little.

### How much of the window matches training text (cap 8192)

| Longest matching suffix | Share of holdout bytes |
|---|---:|
| < 4 | 2.00% |
| 4–7 | 25.13% |
| 8–15 | 54.30% |
| 16–31 | 14.61% |
| 32–63 | 2.84% |
| 64–255 | 1.10% |
| 256–1023 | 0.02% |
| 1024–8192 | 0.0001% (1 byte) |

The median is **10 bytes**. Exact byte strings rarely repeat further than that, so a longer byte window has nothing more to count.

### Questions (greedy, 150 bytes, cap 8192)

| Kind | Correct |
|---|---:|
| Seen verbatim | 3/3 |
| First letter lowercased | 3/3 |
| " Thanks." added before `\nAssistant: ` | 0/3 |
| Hand paraphrases | 0/3 |
| Held-out Dolly | 1/30 (the hit: "is engineering a great career" → "yes") |
| Facts in the Wikipedia slice | 0/8 |

- **The one-letter change now works.** The long suffix match skips over the changed letter and reaches the stored question.
- **A change at the end breaks it**, because the end is where the suffix match starts.
- **New questions get a fluent, copied Dolly answer** to whichever stored question ended in the same few words:
  - "What does an aardvark eat?" → *"Tortoises eat a range of fruit and vegetables…"*
  - "Who composed An American in Paris?" → *"Ten places you should eat at in Paris are: …"*
  - "What is the capital of Algeria?" → *"There are over 500 native languages spoken in Nigeria…"*
  - "Tell me about the Masters golf tournament." → *"Luis Vicente Palomino (born October 9, 1980) is a Peruvian-American mixed martial artist…"*

---

## 3. The surface score against controls

`S = H·D·L` from `surface_score.cpp`, with the same reference (40 windows of the training slice, vocabulary of words of 3–32 letters).

| Text | S |
|---|---:|
| Real Wikipedia window (200 bytes) | 0.933 |
| The same window, words shuffled | **0.928** |
| Words drawn at random from the training slice | **0.918** |
| The trie's nucleus sample | 0.775 |
| One phrase repeated | 0.027 |

| 8K-model output | Mean S |
|---|---:|
| Correct copied answers (seen, first letter changed) | 0.768 |
| Wrong answers: " Thanks." / held-out Dolly / wiki facts / paraphrases | 0.823 / 0.838 / 0.860 / 0.887 |

`S` catches loops and gibberish letters. It scores meaningless word sequences like real text, and scores wrong answers above right ones. A first ad hoc run with a different random draw gave 0.877 for random words; the tool's fixed seed gives 0.918.

---

## 4. Core Cypha's earlier text work (from the repo)

These are from `bench/config/cypha_branch_a_*.json`, `docs/CYPHA_BRANCH_A_EMBEDDINGS.md`, `docs/archive/studies/CYPHA_TESTS_PHASE2.md` and `docs/history/`.

**20 Newsgroups, 2k samples**

| Setup | Accuracy |
|---|---:|
| Hashing fallback | 16–19% |
| TF-IDF/SVD, `W_enc` learning | 34.0% |
| MiniLM, `W_enc` learning | 49.3% |
| MiniLM, `W_enc` frozen | **62.5%** |
| Logistic regression on MiniLM | 60.3% |
| RFF on MiniLM | 4.5% |

**Other text attempts**

- Hebbian against contrastive encoder updates: Hebbian lost 4 of 4 tasks (−0.03 to −0.25).
- Attention memory: +0.20 to +0.27 BPC, killed.
- BPE: 4.15 against 2.87 BPC, killed.
- Word-sized trie expert: weight 0.0005.
- Earlier "semantic" layers (v3, the pattern branch, v7): the demos were exact-string lookups or character-code proximity. No semantic benchmark exists for them.

**Lessons**
- Meaning in Cypha's text results came from outside: MiniLM.
- The classifier is good once it is given meaningful vectors.
- The TF-IDF path never got a fair test, because it ran with `W_enc` learning, the setting that cost 13 points with MiniLM.

---

## 5. What this means

- **The copy is retrieval.** Call it retrieval. Make it tolerant of rewording and able to abstain: PLAN B7.
- **More byte context does not fix it.** A count model, at any order, knows only exact byte strings it has seen. Context has to grow over words, classes and documents instead: PLAN §3.2, B3–B5.
- **Meaning needs structure** (lexicon, concepts, facts, patterns) learned and used by Cypha's counting and classifying machinery: PLAN Tracks C–G.
- **The surface score needs a successor** that passes control ladders: PLAN §6, SCRIBE.
