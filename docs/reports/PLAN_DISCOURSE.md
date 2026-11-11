# Q3: the planner keeps to the subject, drifts a little, and reads the whole paragraph (dev)

**Odin Loch** · 2026-11-06 · branch `wip/plan-discourse` · GENERATION_PLAN §13.42 Q3, on the text-only store of §13.46

**Odin:** "Keep it focussed but let it drift a little and long context."

**Outcome first.** Q3 is `sample --k5 … --k5-plan-discourse on|KEY=VAL,…`. After P1's facts, the planner goes on through linked entities (up to 2 links from the main entity), picks each next fact by a focus weight over the whole paragraph, and skips facts already said. When the facts run out, K5 writes on with a paragraph gate. `--k5-max-words N` allows long samples. With the flags off, the output is byte-identical to main. On dev (text-only store `e9_text`, baseline P1t of §13.46), Q3 states 5.29 facts per planned prompt against 2.32, and 0.648 of its sentences on planned prompts name the main or a linked entity, against 0.316. In 1,500-character samples it states 9.16 facts per planned prompt against 2.32 (0.390 of sentences on the subject against 0.144). SCRIBE v2 rises from 0.511 to 0.606, but SCRIBE rewarded topic words before (§13.34), so it does not decide. The words around the facts are still K5's fill and often false ("Philip Johnson (May 19, 1906 – June 1, 1962) was King of Jerusalem"). The pairwise packets (460 and 1,500 characters, against P1t) are built and committed; the raters and the predictions are left to the lead.

## Design (what was built)

Code: `native/include/cypha/lm/planner/discourse.hpp` (the discourse writer), a few flag lines in `cyphalm_trie.cpp`, one accessor in `writer.hpp`, `scripts/plan_discourse_measures.py`, and `--cut N` in `scripts/blind_rate.py`.

1. **Linked entities.** From the main entity, the store's links:
   - the values of its facts (kind 1);
   - the subjects that have it as a value (kind 2);
   - the subjects that share one of its facts' values under the same relation (kind 3).
   - From those, kinds 1 and 2 again, for a second link. Shared values link only from the main entity, and only if at most 25 subjects share the value under that relation (not "country: United States"). Past the first link, kind 2 has the same limit (not everyone "born in Cleveland").
   - Each candidate has a fit: the link's kind, K5's prompt association of its name, the topic fit of its name, and the document PPMI with the main entity. The best 12 are kept at hop 1, 6 at hop 2. Overview pages ("List of …", "Politics of India") are left out.
2. **The paragraph.** P1's plan comes first, in its order. Then each step takes the unstated fact with the highest focus weight: the entity grid's change over the whole paragraph, the LDA topic fit to the whole paragraph, the prompt association, the PPMI with the main entity and the relation transition, minus 2 per link from the main entity. So the text keeps to the subject and then moves a little.
3. **Long context.** Every sentence written, fact or free, joins the experts' context, so the fill and the gate read the whole paragraph. A fact is not stated twice: not the same subject and relation, not the same subject and value, not one a sentence already holds, and (past P1's plan) not the same relation and value as one already written.
4. **Past the facts.** When the pool is empty, K5 writes on. Its 8 drafts are chosen by the paragraph's topic, its entity grid and the prompt association (z-scored), less markup.
5. **`gap=N`** puts one such K5 sentence after every N fact sentences. The default is 0 (see below).

The search runs on fact IDs, word IDs and class codes; only the winners become text. Prompts with no plan are plain K5, as in P1. The plan log's `D` field gives each written fact's hop and each linked entity's hop, kind, parent and fit.

**Fixed after reading the smoke text** (all in `discourse.hpp`):
- **Prompts stopped after P1's plan.** The repeat ban reads the text's last 11 bytes with the fill's first byte. When the paragraph ends as an earlier sentence ended ("…, California." twice), every word's leading space is banned, so no skeleton can be filled. California had 40 facts in its pool and wrote 3. The ban is now lifted at the sentence edge when nothing can be filled, then wholly, as K5 itself lifts it. On dev, 17 fact steps were filled this way; the whole lift was never needed.
- **Garbled names.** K5 splits words at non-ASCII letters, so names came out as "Lule å" and "Ö sterg ö tland County". Names from the store are now written whole ("Luleå", "Östergötland County"). Hyphens keep K5's spacing, as in the rest of its text.
- **Repeats.** Two sentences per person, one for the birth year and one for the death year, each with a made-up second date ("L. L. Zamenhof, (NH 4, 1859 – July 11, 1174) was a German engineer. L. L. Zamenhof (January 1, 1878 – February 3, 1917) was a Greek philosopher"). Past P1's plan, one of the two is written per person.
- **Off-subject links.** A count cut off after 4,000 store rows let 35 states "located in the United States" through as fewer than 25. Shared values at the second link brought in people who share an occupation with a person (Gustav Radbruch → "Jurist" → Clarence Thomas) or a birthplace. Both are closed as described in step 1.

## Commands

```
rsync -a --delete --exclude .git <worktree>/ ~/q3_src/
cmake -S ~/q3_src/native -B ~/build_q3 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/build_q3 --target cyphalm_trie scribe k5_planner
cd ~/q3_src
export CYPHALM_TRIE=~/build_q3/cyphalm_trie SCRIBE=~/build_q3/scribe WIKI=/mnt/d/cypha_llm/enwik8 \
       ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=4 \
       CYPHA_K5_PLAN_STORE=/mnt/d/cypha_llm/fact_store_e9/e9_text      # = bench/fact_store/e9_text
K5_PLAN_LOG=$OUT/P1t_s0.plan.log scripts/gen_eval.sh dev P1t_s0 $OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt
K5_PLAN_LOG=$OUT/Q3t_s0.plan.log scripts/gen_eval.sh dev Q3t_s0 $OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt \
    --k5-plan-discourse on
# long samples: sample alone, the same seeds, 1,800 bytes
cyphalm_trie sample /mnt/d/cypha_llm/enwik8 $OUT/Q3t_long.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv \
    --plan-drop 75497472:100000000 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --prompts 250 --bytes 1800 --k5-max-words 600 \
    --k5-plan-discourse on          # P1t_long: the same without --k5-plan-discourse
python3 -I scripts/plan_discourse_measures.py --ref $OUT/Q3t_s0.plan.log \
    --arm P1t=$OUT/P1t_s0.gen.txt,$OUT/P1t_s0.plan.log --arm Q3t=$OUT/Q3t_s0.gen.txt,$OUT/Q3t_s0.plan.log [--cut 460]
```

`OUT=/mnt/d/cypha_llm/plan/q3`. Every run used 4 threads; other jobs (O5, T1, Q1) ran on the PC at the same time. Selftests pass: `cyphalm_trie` (with the discourse checks), `scribe` (120/120), `k5_planner`.

## Byte identity (flags off)

- The Q3 binary against a fresh build of main (c10935b), `sample --k5` through the default planner (then `e9_aug`), all 250 dev prompts: the GEN file and stdout are identical under `cmp` (sha256 c05220e2…0ce7). The plan logs are identical once sorted (threads write them in any order).
- The same with `--no-k5-plan` (K5 alone): identical (sha256 2b37f604…0a0a).
- Both also equal the checked-in `bench/k5_planner/P1_e9_s0.gen.txt` and `K5_base_s0.gen.txt` (carriage returns removed; the MODEL line names paths and is left out).
- With `CYPHA_K5_PLAN_STORE=e9_text` and no Q3 flag, the Q3 binary's samples equal the lead's P1t (`bench/k5_planner/P1t_s0.gen.txt`, main 5df1569), bar the MODEL line.
- `blind_rate.py --cut` at its default (460) rebuilds `bench/blind/fill_pair` exactly.

## The gap (a dev choice)

30 dev prompts (12 planned), e9_text:

| Setting | sentences (planned) | name the main entity | name the main or a linked entity | facts stated per planned prompt |
|---|---:|---:|---:|---:|
| P1t | 85 | 0.247 | 0.247 | 1.75 |
| gap 0 | 99 | 0.293 | **0.495** | **4.00** |
| gap 2 | 101 | 0.277 | 0.406 | 3.58 |

**Gap 0 is chosen.** K5's free sentences between the facts are where the text leaves the subject ("These two more than one age.", "From a historical mission of the famous land."). They break the run of facts and push facts out of the cut. The drift Odin asked for comes from the linked entities instead. The same held on the old store e9_aug (a smoke run before the last link fix): 0.643 against 0.495.

## Results (GEN-A dev, seed 0, 250 prompts, store e9_text)

Paths (P1t's plan, unchanged in Q3): main entity 24, co-occurring 53, plain K5 173. Q3 differs from P1t only on the 77 planned prompts.

| Arm | SCRIBE v2 [95%] | v3.1 [95%] | copy32 | 13-word runs | sampling (4 threads) | gen_eval wall | peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1t | 0.511 [0.481, 0.542] | 0.511 [0.491, 0.532] | 0.0388 | 0 | 188 s | 3:49 | 630 MB |
| **Q3t** (gap 0) | 0.606 [0.572, 0.642] | 0.573 [0.549, 0.597] | 0.0385 | 0 | 143 s | 2:58 | 631 MB |
| side: P1 (e9_aug) | 0.532 [0.502, 0.564] | 0.537 [0.514, 0.558] | 0.0376 | 0 | 181 s | 3:39 | 634 MB |
| side: Q3 (e9_aug) | 0.636 [0.601, 0.671] | 0.608 [0.582, 0.636] | 0.0344 | 0 | 142 s | 2:53 | 634 MB |

- Q3 is not slower: it writes fewer of K5's free sentences, which cost 8 drafts each. The runs shared the PC with other jobs, so the times are rough.
- RAM is the same: Q3 adds no model.

**On the subject** (`plan_discourse_measures.py`; the 77 planned prompts, sentences naming the entity by its full or short name; Q3's entity lists for both arms):

| Arm | cut | sentences | name the main entity | name the main or a linked entity | facts stated / planned prompt | / all prompts | repeated facts | repeated sentences | stated by hop 0/1/2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| P1t | 600 B | 566 | 0.311 | 0.316 | 2.32 | 0.72 | 10 of 179 | 0 | — |
| Q3t | 600 B | 659 | 0.326 | **0.648** | **5.29** | 1.63 | 4 of 407 | 1 | 50 / 318 / 39 |
| P1t | 460 ch | 460 | 0.383 | 0.389 | | | | | |
| Q3t | 460 ch | 514 | 0.405 | **0.687** | | | | | |

- "Repeated facts" counts a fact whose subject and value both occur in the prompt as substrings, so "Germany | language | German" counts against a prompt holding "Germany". The writer itself checks whole tokens. The one repeated sentence is "Forrest J." twice, a sentence-split artefact.
- Most stated facts are one link away (318 of 407): the drift is mostly short.

**Long samples** (1,800 bytes, 600 words, all 250 prompts so the seeds match; measured at 1,500 characters):

| Arm | prompts | sentences (planned) | name the main entity | name the main or a linked entity | facts stated / planned prompt | repeated facts | stated by hop 0/1/2 | sampling (4 threads) | peak RAM |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| P1t long | 250 (77 planned) | 1349 | 0.139 | 0.144 | 2.32 | 10 of 179 | — | 499 s | 632 MB |
| **Q3t long** | 250 (77 planned) | 1702 | 0.139 | **0.390** | **9.16** | 4 of 705 | 50 / 496 / 159 | 493 s | 634 MB |
| P1t long | the 25 long-reading prompts (7 planned) | 113 | 0.177 | 0.186 | 2.71 | 1 of 19 | — | | |
| **Q3t long** | the 25 long-reading prompts (7 planned) | 145 | 0.131 | **0.448** | **10.00** | 0 of 70 | 4 / 60 / 6 | | |

- At 1,500 characters Q3 keeps writing facts where P1t has handed over to K5 after its plan: about 9 facts per planned prompt against 2. Then K5 writes the rest with the paragraph gate.
- The drift grows with length: 159 of 705 stated facts are two links away, against 39 of 407 at 600 bytes.
- The share naming the main entity itself is the same in both arms (0.139): Q3's extra sentences name linked entities, and its K5 tail drifts as P1t's does.
- P1t's long samples begin exactly as its dev samples (the same seeds).

## Examples (dev, 460-character cut, the first sentences)

- **Mount Kosciuszko.** P1t: "…bordering the same articles of the New South Wales. The state's son. A collection of the party, and the rest of the former became one of the local ants…" Q3t: "…bordering the same articles of the New South Wales. He died in Canberra, Australia. C. and Australia were written in English. Fitzroy Island National Park was created in 1939. In 1986, Juan made the Budderoo National Park (Qur'an)."
- **Gothenburg** (the prompt is about Gothenburg University). P1t: "He was born in Gothenburg, Sweden.) The central conflict over. It was the de facto)." Q3t: "In 2005, he went to the University of Chalmers University of Technology in Sweden. He also wrote for Chalmers University of Technology in Gothenburg, New York. Björn Ulvaeus (born October 4, 1945) is an American writer and philosopher from Los Angeles, California."
- **California** (drift a little): Q3t goes from California's facts to "Alameda is a city located in the human history of California. It is the second largest city in Colorado, where Denver." P1t turns to K5 after the third sentence.

The facts are on the subject, and the paragraph moves to nearby entities. The words around them are K5's fill and are often false.

## Packets (committed before any rating)

- `bench/blind/q3_pair/`: 50 dev prompts (`bench/blind/llm_rating/prompts.json`), cut 460, Q3t against P1t, seed 44.
- `bench/blind/q3_long/`: 25 dev prompts (`bench/blind/q3_long/prompts.json`, the first 25 of those 50), cut 1,500, Q3t long against P1t long, seed 45.
- Each holds `R1.txt`–`R4.txt`, `RATER_PROMPT.md` and `key.b64`. `KEY.json` is not in git (a copy is in `/mnt/d/cypha_llm/plan/q3/keys/`).
- GEN files: `bench/k5_planner/Q3t_s0.gen.txt` (dev), `Q3t_long_s0.gen.txt` and `P1t_long_s0.gen.txt` (long). P1t's dev file is the lead's `P1t_s0.gen.txt`.

## What is weak

- **Few prompts differ.** On e9_text only 77 of 250 dev prompts have a plan. In the 50 rated prompts, 11 differ; in the 25 long prompts, 7. The rest are the same plain K5 text in both arms and should tie. So even if Q3 won every prompt that differs, its win share could reach only 0.61 on the 460-character reading and 0.64 on the long one. §13.42's prediction for the long reading (above 0.6) needs nearly all of them.
- **The long prompts** (`bench/blind/q3_long/prompts.json`) are the first 25 of the usual 50, fixed before any run. A set drawn from planned prompts would have more power; it was not changed here.
- **The fill.** Q3 writes more fact sentences, so it shows more of the fill's faults: made-up dates and roles around the true fact ("Gene Simmons (August 14, 1949 - April 7, 1858) was an American playwright"). Q2's typed slots address this, not Q3.
- **A list of people.** On the co-occurring path the linked entities are often people who share the main entity's value ("Australian"), so the paragraph becomes a run of "Name (date - date) was a …" sentences. It stays on the subject but reads as a list.
- **Gibberish facts** in the store ("Food writing | nationality | Food") are stated as before; Q1's filter is the fix.
- SCRIBE's rise is not evidence of better reading (§13.34). The raters decide.

## Adopted and made the default (2026-11-06, GENERATION_PLAN §13.42)

- **Blind pairwise against P1t:** 0.590 [0.540, 0.640] at 460 characters, and 0.580 [0.500, 0.680] on the long reading.
- **Now the default:** `sample --k5` uses the discourse writer, and `--k5-plan-discourse off` gives P1t back. Both were checked by `cmp` on the 250 dev prompts.
- **Test read, once:** SCRIBE v2 0.631 [0.595, 0.665], v3.1 0.592 [0.566, 0.619].
