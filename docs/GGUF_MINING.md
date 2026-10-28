# Mining a GGUF language model for Cypha

**Odin Loch** · 2026-10-26 · part of [PLAN.md](PLAN.md) Track H (offline distillation, verified)

This is the full list of what can be statistically extracted from a neural language model stored in GGUF format, and how each item feeds Cypha's data pack and evidence scores. The script is [`scripts/gguf_mine.py`](../scripts/gguf_mine.py).

The teacher model never runs inside Cypha. Everything it gives is data, and every mined item carries the statistics ECS needs ([PLAN §7](PLAN.md#7-ecs-evidence-certainty-score)).

**Status column**
- **script**: the subcommand that does it now.
- **planned**: listed, not built. See section 9 for the route.
- **excluded**: deliberately not done, with the reason.

---

## 0. Ground rules

1. **One model is one witness.**
   - Ten paraphrases agreeing is strong evidence that the *model* believes something. It is not ten independent sources.
   - ECS counts a model as a single source, with an independence weight of `1 + ln n` over its own items ([PLAN §7.1](PLAN.md#71-evidence-items)).
   - A mined fact reaches High grade only when an independent source agrees, such as Wikidata, Wikipedia text or a second model trained on different data.
2. **Calibrate before trusting.**
   - `calibrate` measures the model's accuracy per relation against gold facts and writes the Beta prior that ECS uses as the model's source trust.
   - Uncalibrated, the default prior is Beta(1, 3), as PLAN §7.1 sets for distilled items.
3. **Licence first.**
   - `info` reads `general.license` and warns when it is not on a permissive list.
   - Some licences restrict using outputs to build other models (section 10).
4. **No memorised training text.** Extracting verbatim training data is excluded (section 11).
5. **Provenance on everything.** Every output directory has `provenance.json`:
   - model SHA-256, name, architecture, quantisation and licence;
   - tool and library versions, the command, the seed and the time.

---

## 1. File, metadata and tensor census (static)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 1.1 | Architecture and hyperparameters: layers, width, heads, KV heads, FFN size, context length, RoPE, norm epsilon | GGUF `general.*` and `<arch>.*` keys | Context for every other number; profiling comparisons | script: `info` |
| 1.2 | Licence, name, base model, source URL | `general.license`, `general.name`, … | Licence gate; provenance | script: `info` |
| 1.3 | Chat template | `tokenizer.chat_template` | Whether to prompt as base or chat model | script: `info` |
| 1.4 | Tensor census: name, quantisation type, shape, bytes, parameter count; tied output or not | Reader tensor table | Which analyses apply (fused QKV, MoE experts, tied embeddings) | script: `info` |

## 2. Tokenizer (static)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 2.1 | Vocabulary with surface text | Tokens, decoded from SPM `▁` or GPT-2 byte maps | A ready subword inventory | script: `vocab` |
| 2.2 | Frequency prior | SPM unigram scores (log-probabilities) or BPE merge order (earlier merge = more frequent pair) | Word-frequency ranks without a corpus; a prior for the lexicon (PLAN D2) and the word trie (B3) | script: `vocab` (`freq_rank`) |
| 2.3 | Whole-word candidates | Word-initial alphabetic tokens. Prefix fan-out flags fragments ("con" extends into 200+ tokens). `--dictionary` marks real words | Seed word list, frequency-ordered | script: `vocab` (`words.tsv`) |
| 2.4 | Morphology | Suffix statistics over word tokens (-ing, -ed, -tion, …); frequent continuation pieces | Morphological units for the lexicon and tagger features (E1) | script: `vocab` |
| 2.5 | BPE merge list | Ranked pair merges | A learned subword grammar; pair statistics | script: `vocab` (`merges.tsv`) |
| 2.6 | Script and language coverage | Unicode script of every token | Which languages the pack can cover | script: `vocab` (`vocab_stats.json`) |
| 2.7 | Special and control tokens, byte fallback | Token types | Tokenisation rules | script: `vocab` |

## 3. Embedding geometry (static)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 3.1 | Static word vectors | Rows of `token_embd.weight` for whole-word tokens | Teacher vectors to compare with Cypha's own PPMI-SVD (C1) and to retrofit (C4). An offline table, not a runtime network | script: `embed` (`vectors.npy`) |
| 3.2 | Nearest neighbours | Cosine top-k | Synonym and inflection candidates for the lexicon, to be verified | script: `embed` |
| 3.3 | Clusters | Spherical k-means | Word classes for class-level counts (B4) | script: `embed` |
| 3.4 | Class hierarchy | Average-linkage merge tree over cluster centroids, written as bit paths like Brown clusters | A hierarchy for B4 and SCRIBE E | script: `embed` (`clusters.tsv`, `hierarchy.tsv`) |
| 3.5 | Principal axes | Randomised PCA; the words at each end of each axis | Interpretable directions (frequency, language, topic) | script: `embed` (`pca.tsv`) |
| 3.6 | Norm against frequency | Spearman correlation between vector norm and frequency rank | Diagnostic for vector quality | script: `embed` |
| 3.7 | Relation offsets | Leave-one-out mean offset `b − a` per relation (capital-of, female-of, plural, past tense, comparative, or your `--pairs`), accuracy @1 and @10 | Which relations are linear in the static space: cheap relation vectors | script: `embed` (`relation_offsets.tsv`) |
| 3.8 | Direct-path bigrams | Embedding → final norm → unembedding, skipping every layer (the zero-layer path of the transformer-circuits framework) | A bigram prior per word, approximate | script: `embed` (`direct_bigrams.tsv`) |

## 4. Circuits in the weights (static)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 4.1 | What each FFN neuron **writes** | Project each `ffn_down` column through the final norm and unembedding; list top and bottom tokens and the write entropy (Geva et al. 2022: FFN layers promote concepts in vocabulary space) | A concept dictionary: neurons with low write entropy name a concept; feeds pattern and class discovery | script: `neurons` |
| 4.2 | What each FFN neuron **reads** | Score word embeddings against each `ffn_up` row (key–value view of FFNs) | Trigger words per concept | script: `neurons` |
| 4.3 | MoE experts | The same per expert (`*_exps` tensors) | Expert specialisation map | script: `neurons` |
| 4.4 | Attention OV circuits | Per head: embedding → attention norm → V → O → final norm → unembedding. Copy score: how often a token maps to itself in the top 1 or top 5 (Elhage et al. 2021) | Which heads copy, and which map tokens to related tokens; context for B9 (copy-from-prompt) | script: `heads` |
| 4.5 | QK circuits (which tokens attend to which) | Needs RoPE positions; static analysis is approximate | Attention-pattern priors | planned |
| 4.6 | Weight spectra | Per matrix: singular values, stable rank, effective rank, a power-law α fitted to the eigenvalue tail (MLE with KS-chosen xmin, as in heavy-tailed self-regularisation work), log–log slope of σ_k against k, kurtosis, outliers | Model-quality diagnostics. The σ_k slope tests the harmonic-spectrum claims in the NMP paper on real models | script: `spectra` |

## 5. Probability behaviour (dynamic)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 5.1 | Teacher-forced log-probabilities | Exact per-token scores from logits (`logits_all`), with prefix reuse checked bit-identical against fresh evaluation | The base of every statistic below | script: `Runner` |
| 5.2 | Corpus bits per byte | Sliding-window scoring | Data selection for Track A ("picky" learning): flag junk or out-of-domain text; a fluency reference beside BPC | script: `score` |
| 5.3 | Synthetic corpus | Sampling from prompts (temperature, top-p, seeds recorded) | Training text for counts in Cypha's trie format (distillation by sampling) | script: `sample` |
| 5.4 | Text statistics of that corpus | Unigram and bigram counts, PMI collocations, Zipf slope, type–token ratio | Collocations for the pattern layer; a reference for SCRIBE E | script: `sample` (`text_stats.json`) |
| 5.5 | Next-token distributions on a frozen prompt set | Full distributions | A teacher distribution for comparing Cypha's mixer | planned |

## 6. Knowledge and relations (dynamic)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 6.1 | Factual relations | LAMA-style probing (Petroni et al. 2019) over 52 Wikidata relations, each with 2–3 paraphrased cloze templates plus a question. Candidates come from greedy and sampled answers; each is scored under every template (ParaRel-style consistency, Elazar et al. 2021) | Facts layer (D4): candidate triples with full statistics | script: `facts` |
| 6.2 | Inverse-relation agreement | Score the inverse direction where one exists (capital ↔ capital of, part of ↔ has part) | A consistency feature for ECS | script: `facts --inverse` |
| 6.3 | Knowledge-graph crawl | LMCRAWL (Cohen et al. 2023) and GPTKB (2024) style: list relations for an entity, generate objects, keep an object only when at least two paraphrases agree, queue new entities, dedupe. LMCRAWL reported 82–83% precision for one- and two-hop facts with this rule; GPTKB materialised 101M triples over 2.9M entities from GPT-4o-mini | New entities and relations beyond the fixed relation list | script: `crawl` |
| 6.4 | Arbitrary relations from seed pairs | BertNet (Hao et al. 2023): a relation prompt and a few example pairs; paraphrase the prompt, search entity pairs, rescore | Relations Wikidata lacks | planned (use `facts --relations-file` with your own templates meanwhile) |
| 6.5 | Lexical relations | Synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms, definitions, plurals, past tenses, each with templates and statistics | Lexicon and concept layers (D2, D3); thesaurus candidates to check against WordNet | script: `lexical` |
| 6.6 | Lexical classifications | Distribution over labels by scored options: part of speech, polarity, entity type, concreteness, formality | Lexicon features; gazetteer for E2; polarity lexicon | script: `lexical` |
| 6.7 | Taxonomy chains | Repeated "X is a kind of ___" with cycle detection and a probability cut-off (zero-shot taxonomy induction, Jain & Espinosa-Anke 2022) | Concept hierarchy candidates (D3) | script: `taxonomy` |
| 6.8 | Commonsense relations | ConceptNet relations (UsedFor, CapableOf, HasProperty, AtLocation, MadeOf, HasA, Causes, HasPrerequisite, Desires, CreatedBy, PartOf, MotivatedByGoal), in the spirit of symbolic knowledge distillation (West et al. 2022) | Commonsense facts, low prior until corroborated by ConceptNet | script: `commonsense` |
| 6.9 | Entity linking | Attach IDs to mined strings from a label table (Wikidata labels and aliases from the pack) | Canonical IDs so facts merge with D4 | script: `canon` |

## 7. Language patterns (dynamic)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 7.1 | Question paraphrases | A few-shot rewrite of each relation's question. A paraphrase is kept with its answer-agreement rate: the same answer as the original on test subjects (paraphrase mining as in LPAQA, Jiang et al. 2020) | Pattern variety for the question classifier (E4) and the pattern layer (F2); the "Thanks."/paraphrase failures in the 2026-10-26 test are exactly this gap | script: `paraphrase` |
| 7.2 | Q/A pairs | From mined facts above a confidence threshold, with every verified question pattern | Training data for E4 and B7, marked synthetic | script: `qa` |
| 7.3 | Instruction data | Self-Instruct-style generation | Low priority; low trust | planned |

## 8. Representations through inference (dynamic)

| # | Item | How | Gives Cypha | Status |
|---|---|---|---|---|
| 8.1 | Model-pooled word vectors | Each word through the model in embedding mode, optional PCA. `--pooling mean` (default, as before), `last` (last-token state with the EOS check of section 14, needed for Qwen3-Embedding) or `none` (per-token states averaged over the word's own tokens). A static table, the same idea as Model2Vec's static-embedding distillation | Teacher vectors for C1/C4 comparisons; neighbour lists | script: `vectors` |
| 8.2 | Word senses | Contextual vectors of a word across sentences, spherical k-means with a minimum cluster size, best k by a silhouette-like score | Sense inventory candidates for C5 (Cypha sense classes) | script: `vectors --senses` |

## 9. Needs internal activations (not in the script)

These need per-layer hidden states. llama-cpp-python exposes the final layer only, so there are two routes:
- llama.cpp's eval-callback mechanism (the one `imatrix` uses), from C++;
- loading the GGUF into PyTorch: the transformers library can dequantise GGUF files for supported architectures.

| # | Item | Reference | Gives Cypha |
|---|---|---|---|
| 9.1 | Logit lens / tuned lens: what each layer predicts | nostalgebraist 2020; Belrose et al. 2023 | Where facts resolve; layer choice for 9.2–9.4 |
| 9.2 | Linear relational embeddings: a relation as an affine map on the subject's hidden state | Hernandez et al. 2024 | Relation operators that generalise to unseen subjects |
| 9.3 | Knowledge neurons | Dai et al. 2022 | Neuron-level fact attribution |
| 9.4 | Causal tracing | Meng et al. 2022 (ROME) | Which states carry a fact |
| 9.5 | Sparse-autoencoder dictionary features | Cunningham et al. 2024; Bricken et al. 2023 | A large interpretable concept inventory |
| 9.6 | Contrast-consistent truth directions | Burns et al. 2023 | An unsupervised truth score for ECS |
| 9.7 | Structural probes for syntax | Hewitt & Manning 2019 | Parse supervision for E3 |
| 9.8 | Decoding hidden states into text | Patchscopes, Ghandeharioun et al. 2024 | Explanations of internal concepts |

## 10. Uncertainty statistics attached to every mined item

| Statistic | Meaning | Where |
|---|---|---|
| `choice_p` | Model probability of the candidate among all candidates (whole string), averaged over templates | `facts`, `lexical`, `crawl`, … |
| `choice_p_tok` | The same with per-token normalisation; whole-string probability favours short answers | same |
| `lp_mean`, `lp_mean_sd` | Mean token log-probability and its spread across paraphrases | same |
| `para_agree` | Fraction of templates whose greedy answer is this candidate | same |
| `sample_frac` | Fraction of sampled answers that are this candidate | same |
| `margin` | Log-probability gap to the best other candidate | same |
| `verify` | `log P(Yes) − log P(No)` for "Is it true that …?" | same |
| `sem_entropy` | Entropy over clusters of sampled answers, after Farquhar et al. 2024 (here clustered by folded string; bidirectional entailment is the stronger version) | per (subject, relation) |
| `dk_rate` | Fraction of "don't know" or empty answers | per (subject, relation) |
| `inverse_lp_mean` | Score of the inverse direction | `facts --inverse` |
| `confidence` | Logistic calibration of these features per relation once `calibrate` has run; before that, the heuristic `choice_p × max(para_agree, sample_frac)`, marked `calibrated: false` | all |
| `evidence` | The ECS record: source, method, source-trust Beta prior (from calibration), extraction confidence | all |

`calibrate` also reports accuracy, ECE and Brier per relation, in sample. To measure calibration honestly, hold gold facts out of the fit.

## 11. Excluded

| Item | Why |
|---|---|
| Extracting memorised training text (Carlini et al. 2021; Nasr et al. 2025) | Copyright and personal-data risk; also not useful for structure |
| Personal facts about private individuals | Privacy; the pack covers public entities only |
| Redistributing teacher weights or verbatim outputs where the licence forbids it | Licence terms |

---

## 12. Licences for teacher models

Check the model card for the exact model and size; terms differ within families.

| Family | Terms, as far as checked on 2026-10-26 |
|---|---|
| Qwen2.5 | Apache 2.0 for most sizes. The 3B and 72B use a Qwen licence that needs arrangements for commercial use |
| Llama 3 | The community licence prohibited using outputs to improve other large language models |
| Llama 3.1 | That prohibition was removed. Attribution and naming conditions apply ("Built with …"; derived model names must begin with "Llama") |
| Others (Mistral 7B, OLMo, SmolLM, Phi, Gemma) | Check each model card. Gemma has its own terms with a prohibited-use policy |

Prefer Apache-2.0 or MIT models. `info` and every `provenance.json` record the GGUF's own licence field.

---

## 13. Recipe

```bash
pip install numpy gguf                 # static commands
pip install llama-cpp-python           # dynamic commands
M=path/to/model.gguf; O=mined/$(basename $M .gguf)

python scripts/gguf_mine.py info     --model $M --out $O/info
python scripts/gguf_mine.py vocab    --model $M --out $O/vocab --dictionary pack/words.txt
python scripts/gguf_mine.py embed    --model $M --out $O/embed --dictionary pack/words.txt
python scripts/gguf_mine.py neurons  --model $M --out $O/neurons --max-neurons 2048
python scripts/gguf_mine.py heads    --model $M --out $O/heads
python scripts/gguf_mine.py spectra  --model $M --out $O/spectra --pattern 'blk\.(0|8|16)\.'

# Calibrate on a held-in sample of Wikidata facts, then mine with that calibration.
python scripts/gguf_mine.py calibrate --model $M --out $O/cal --gold pack/wikidata_gold.tsv
python scripts/gguf_mine.py facts    --model $M --out $O/facts --subjects pack/entities.tsv --inverse --calibration $O/cal/calibration.json --resume
python scripts/gguf_mine.py crawl    --model $M --out $O/crawl --seeds pack/seeds.txt --depth 2 --calibration $O/cal/calibration.json
python scripts/gguf_mine.py lexical  --model $M --out $O/lexical --words pack/words.txt --resume
python scripts/gguf_mine.py taxonomy --model $M --out $O/taxonomy --words pack/nouns.txt
python scripts/gguf_mine.py commonsense --model $M --out $O/commonsense --words pack/concepts.txt
python scripts/gguf_mine.py paraphrase --model $M --out $O/para --subjects pack/entities.tsv
python scripts/gguf_mine.py qa       --out $O/qa --facts $O/facts/facts.jsonl --patterns $O/para/patterns.jsonl
python scripts/gguf_mine.py sample   --model $M --out $O/sample --n 10000
python scripts/gguf_mine.py score    --model $M --out $O/score corpus_shard.txt --split
python scripts/gguf_mine.py vectors  --model $M --out $O/vectors --words pack/words.txt --senses pack/ambiguous.txt --corpus corpus_sentences.txt
python scripts/gguf_mine.py canon    --input $O/facts/facts.jsonl --labels pack/wikidata_labels.tsv --output $O/facts/facts_ids.jsonl
```

**Self-test.** This builds two tiny random llama-architecture GGUFs from a vocab-only file (llama.cpp ships them in `models/`). The second has other weights and a Qwen3-style chat template with a `<think>` block. It then runs:
- all 26 subcommands, in 36 steps that include an `embedtexts --resume` rebuild and `wdiff` between the two models;
- checks on the outputs;
- known-answer checks of the geometry maths;
- pooling checks.

```bash
python scripts/gguf_mine.py selftest --vocab llama.cpp/models/ggml-vocab-llama-spm.gguf --out /tmp/gguf_selftest
```

- **Result on 2026-10-26:** all 18 pass with the SPM (Llama) and BPE (Qwen2) vocabularies, on llama-cpp-python 0.3.36 and gguf 0.19.0.
- **Result on 2026-10-28:** all 50 rows pass with both vocabularies (Qwen2 11 min, SPM 6 min, on 2 shared cores).
  - *EOS.* `<|endoftext|>` (151643) is appended for Qwen2 and `</s>` for SPM; neither tokenizer appends one itself.
- **Known-answer checks (section 14):**
  - Spearman against hand-worked cases, with and without ties (0.8 and 8/√95);
  - Mann–Whitney AUC 0.75;
  - CKA(X, 3XR) = 1, in both the feature and the Gram branch;
  - Procrustes recovers a planted rotation;
  - TwoNN on uniform cubes of dimension 2, 5 and 8 is within 25% (1.94, 4.63, 6.82);
  - the temperature fit recovers a planted τ = 0.05 (0.0498);
  - the ridge word weights recover planted weights;
  - randomized SVD matches the full SVD, and a rank-1 ΔW has a top-5% share of 1;
  - a pure offset is fully explained by the offset model, and a rank-1 map by the affine model, with rank 1;
  - alignment is 0 for identical pairs.
- **Output checks:**
  - `compare` on X against XR gives CKA 1, Procrustes residual about 0, Spearman 1, Jaccard 1 and KL about 0;
  - `embedtexts --resume` rebuilds only a deleted part, and the result is identical;
  - token-level mean pooling matches `Llama.embed`;
  - batched decoding matches single decoding;
  - last pooling equals the last token's state;
  - the no-think chat prompt closes `<think>`.
- **Scoring checks:**
  - teacher-forced scores with prefix reuse are bit-identical to fresh evaluation;
  - the first-token score matches the next-token distribution;
  - bits per byte agree within 0.001 across window sizes.
- **Logic test:** a scripted fake model checked crawl (relation listing, "don't know" skipping, the two-paraphrase rule, depth-limited expansion) and candidate ranking.
- **What it does not test:** a random model knows nothing, so the self-test checks plumbing, not knowledge. The first real run should be `calibrate` on a Wikidata sample, so the numbers mean something.

**Cost.**
- Static commands are dominated by dequantising tensors; `neurons` and `spectra` scale with vocabulary × width per layer.
- Dynamic commands cost about `templates × (1 + samples)` generations plus `candidates × templates` scorings per item.
- Time a small run first with `--samples 1` and a few subjects.

---

## 14. Embedding geometry ([EMBEDDING_GEOMETRY.md](EMBEDDING_GEOMETRY.md))

These subcommands dissect an embedding teacher (Qwen3-Embedding) and mine with a generative one (Qwen3.5-9B). The IDs are those of EMBEDDING_GEOMETRY §6. `geometry` and `compare` need no model: they read any `.npy` plus an index TSV, so Cypha's exported vectors work too.

**Last-token pooling and EOS.**
- `--pooling last` reads `tokenizer.ggml.add_eos_token` and `add_sep_token`. It then tokenises a probe with special tokens on and checks whether the tokenizer itself appends an end token.
- If it does not, the script appends `<|endoftext|>` when the vocabulary has it as a single control token, otherwise the GGUF's EOS. `--eos append` forces this and `--eos never` turns it off.
- The decision is logged and written to provenance as `eos.action`: `tokenizer`, `appended` or `none`.
- Embedding runs from tokens through llama.cpp's batch path, several sequences per decode, so the appended EOS survives. llama-cpp-python's `embed()` tokenises the text itself and cannot add it.
- The self-test checks this path three ways:
  - mean pooling matches `embed()`;
  - batched decoding matches one sequence at a time;
  - last pooling equals the last row of per-token states.

| Command | IDs | Needs | Main flags | Output |
|---|---|---|---|---|
| `embedtexts` | QE0, QE2 | embedding model | `--input` (one item per line, or a TSV with `--column`, `--id-column`, `--side-column`); `--pooling` (default `last`); `--eos`; `--instruction`, applied to queries only in the `--template` form (default `Instruct: {instruction}\nQuery:{text}`); `--side`; `--dims N` (MRL prefix: truncate, then renormalise); `--batch`; `--chunk`; `--resume`; `--limit` | `embeddings.npy` (fp16, unit rows); `embeddings_index.tsv` (row, id, side, n_tokens, raw norm, text); `parts/`; items/s and tokens/s in provenance |
| `sensitivity` | B10, C6 | embedding model | `--sentences` (minimal pairs made by rule: negation, role swap, number, entity, shuffle, typo, synonym, padding, plus built-in subject–verb–object templates); `--pairs` (external a, b, kind TSV, e.g. a generated C6 suite); `--kinds`; `--per-kind`; `--typos`; `--role-templates` | `pairs.tsv` (per pair); `summary.tsv` (n, mean, median, IQR per kind, and the median above the unrelated-pair floor); `summary.json` (Q6) |
| `occlusion` | B15, B16 | embedding model | `--sentences`; `--freq` (word, count for the SIF p(w)); `--sif-a`; `--ridge`; `--ridge-map`; `--heldout` | `word_weights.tsv` (per word: occlusions, mean cosine drop, SE, p(w), SIF weight, fitted weight); `additivity.tsv`; `summary.json` (Q5) |
| `instruct` | B8, B9 | embedding model | `--queries`; `--instructions` (default: five Qwen/MTEB-style ones); `--template`; `--heldout`; `--ridge` | `instruct.tsv` (per instruction: query–document cosine, own-document top-1, offset share, affine share, effective rank of A − I); `instruct_items.tsv`; `offset_cos.tsv`; `summary.json` (Q10) |
| `pairs` | C1–C3 | generator, optional embedder | `--docs`; `--types` (keyword, factual, summary, judgement); `--length`; `--difficulty`; `--personas`; `--n`; `--negatives` (topic, negation, role_swap, entity); `--chat-template auto\|on\|off`; `--think`, `--think-tokens`; `--embedder`, `--instruction`, `--min-cos 0.7`, `--neg-margin`; `--min-verify`; `--audit`; `--resume` | `pairs.jsonl` (every row with its controls, `verify`, `p_yes`, `kept`, method, seed and provenance; cosine fields with `--embedder`); `audit.jsonl`; `summary.json` (mask precision with a Wilson interval) |
| `geometry` | B1–B7, B13, B17 | vectors only | `--vectors`, `--index`, `--key`; `--max-items`; `--pairs` (anchor, positive); `--batch-size`; `--labelled-pairs` (a, b, class); `--lengths` (a file, or `index` for the n_tokens column); `--prefixes`; `--knn`, `--knn-max`, `--no-knn-graph` | `geometry.json` (with Q1–Q3 flags); `floor.tsv`; `spectrum.tsv`; `mrl.tsv`; `hubs.tsv`; `classes.tsv`; `class_auc.tsv`; `knn_graph.tsv` |
| `compare` | §7.1 | vectors only | `--a` (teacher), `--b`; `--index-a`, `--index-b`, `--key` (align on an index column); `--procrustes-dim`; `--tau`, `--tau-student`; `--pairs` (fits the teacher τ) | `compare.json`; `compare_strata.tsv` |
| `wdiff` | A4 | two GGUFs | `--model` (the child), `--base` (the parent); `--pattern`; `--max-full`; `--top-frac`; `--iters`; `--max-rows` | `wdiff.tsv` (per tensor); `wdiff_layers.tsv`; `wdiff_summary.json` (Q4) |

**Method notes.**
- **`geometry`.**
  - *Floor and anisotropy.* The floor is the cosine distribution of random pairs, raw and after centring. Anisotropy is the mean vector's norm and the exact mean cosine over all pairs. The five dimensions that contribute most to the mean cosine are listed as rogue dimensions.
  - *Temperature.* τ is the maximum-likelihood estimate of the positive among in-batch candidates. It is found by a golden-section search over log τ; the NLL is convex in 1/τ.
  - *TwoNN.* This is Facco et al.'s line through the origin, fitted with the largest 10% of μ dropped.
  - *Hubness.* This is the skewness of the k-occurrence at k = 10, before and after CSLS.
  - *MRL.* Each prefix is truncated and renormalised, then compared with the full vector: pair Spearman, kNN@10 Jaccard, TwoNN, and the PCA dimensions needed for 95%.
- **Pair Spearman.** In both `geometry` and `compare`, pairs are stratified by the first set's neighbour rank: near (top 10), mid (rank 10 to n/10) and random. Each stratum is also reported on its own.
- **`occlusion`.** There are four fits:
  - (a) the uniform mean and (b) the SIF mean, each with one global scale fitted on the training sentences;
  - (c) per-word weights with ridge towards 1, solved exactly by conjugate gradients on the sparse normal equations;
  - (d) (c) followed by a ridge linear map.

  R² is measured against the held-out mean vector, which is a strong baseline when the vectors are anisotropic. Pair Spearman is the Q5 number.
- **`instruct`.** The offset share is 1 − ‖Δ − t‖²/‖Δ‖² on held-out queries, where t is the mean shift on the training queries. The affine map is fitted by dual ridge. The effective rank of A − I comes from its singular values.
- **`compare` force match.** This is KL(p_T‖p_S) of each anchor's softmax over the other items. It is reported at the teacher's τ (given, fitted from `--pairs`, or 0.05) and at the student τ that minimises it.
- **`wdiff`.**
  - ΔW = W_model − W_base, and the relative change is ΔW measured against ‖W_base‖.
  - The spectrum is exact up to `--max-full`. Above that, a randomized SVD (Halko et al. 2011) gives the top ⌈5%⌉ of the singular values. It is slightly low, so the Q4 share is conservative.
  - For matrices taller than `--max-rows` (the vocabulary matrices), rows are sampled; this is noted on each row of the output.
- **`pairs`.**
  - *Thinking.* It is off by default. The GGUF's own jinja chat template is rendered with llama-cpp-python's `Jinja2ChatFormatter` with `enable_thinking=False`. If a template has a `<think>` block but ignores that flag, the script appends the empty `<think>\n\n</think>` itself.
  - *Template fallback.* With no template, `--chat-template` is a no-op. If the template cannot be rendered, the script falls back to plain prompts and logs a warning; every row records `prompt_mode` and `template_warning`.
  - *`--think`* is used for audit judging only (C3). A thought is generated up to `--think-tokens`, closed, and then the yes/no answer is scored. It needs a template with a `<think>` block.
  - *Embedder.* It is loaded only after the generator has been freed.
  - *Keep rules.* A positive is kept when `verify` > `--min-verify` and its cosine is above `--min-cos`. A hard negative is kept when `verify` < −`--min-verify` and its cosine is within `--neg-margin` of its positive's (C2). C1's third check, that Cypha's evidence does not contradict the pair, runs on the Cypha side.
- **Sizing.** `embedtexts` logs items/s and tokens/s (decode time only) and writes them to provenance. Time 1,000 probe sentences first, then size the full jobs; the full probe set is about 3M tokens (EMBEDDING_GEOMETRY §9.1). `--ctx` (default 512) truncates longer texts, so raise it for paragraphs.

```bash
E=Qwen3-Embedding-0.6B-Q8_0.gguf; G=Qwen3.5-9B-Q8_0.gguf; O=mined/qwen3emb
python scripts/gguf_mine.py info        --model $E --out $O/info     # A1: add_eos_token, chat template
python scripts/gguf_mine.py embedtexts  --model $E --out $O/timing --input probe.txt --limit 1000   # size jobs from this
python scripts/gguf_mine.py embedtexts  --model $E --out $O/probe --input probe.txt --ctx 2048 --resume
python scripts/gguf_mine.py geometry    --vectors $O/probe/embeddings.npy --out $O/geometry --pairs pos_pairs.tsv \
                                        --labelled-pairs classes.tsv --lengths index
python scripts/gguf_mine.py compare     --a $O/probe/embeddings.npy --b cypha_vectors.npy --index-b cypha_index.tsv --key text --out $O/vs_cypha
python scripts/gguf_mine.py sensitivity --model $E --out $O/sensitivity --sentences held_out.txt
python scripts/gguf_mine.py occlusion   --model $E --out $O/occlusion --sentences held_out.txt --freq pack/word_counts.tsv
python scripts/gguf_mine.py instruct    --model $E --out $O/instruct --queries questions.txt
python scripts/gguf_mine.py wdiff       --model $E --base Qwen3-0.6B-Q8_0.gguf --out $O/wdiff --pattern 'blk\.'
python scripts/gguf_mine.py pairs       --model $G --out $O/pairs --docs docs.txt --personas personas.txt --embedder $E --resume
python scripts/gguf_mine.py pairs       --model $G --out $O/audit --audit masked_candidates.tsv --think
```

**Checked on real models (2026-10-28, [QWEN3_EMBEDDING_DISSECTION.md](reports/QWEN3_EMBEDDING_DISSECTION.md)).** llama-cpp-python 0.3.36 (vendored llama.cpp, ggml `0c1e570`), gguf 0.19.0, 2 threads.
- **Qwen3.5-9B support.** It loads (`general.architecture = qwen35`, unsloth Q4_K_M). Greedy answer to "What is the capital of France?": "The capital of France is Paris." `llama-bench` at 2 threads: 10.6 tokens/s prompt, 1.91 tokens/s generation.
- **Qwen3.5's template.** It renders through `PromptFormat` with no warning. With thinking off the prompt ends in an empty `<think>\n\n</think>\n\n` block, from the template itself. The yes/no judge scores +4.64 for a relevant document and −7.65 for an irrelevant one.
- **Memory with `logits_all`.** The generative `Runner` keeps logits for every position: n_ctx × 248,320 floats for Qwen3.5 (1.0 GB at n_ctx 1,024). Peak RSS was 9.2 GB with the 5.7 GB model. Keep `--ctx` small for `pairs`.
- **EOS detection.** Qwen3-Embedding-0.6B's GGUF sets `add_eos_token = true`, EOS 151643 `<|endoftext|>`, and `pooling_type = 3` (last). The tokenizer appends the EOS itself, so the plan is `tokenizer`; `llama-embedding` appends the same token.
- **QE0 gate.** `embedtexts --pooling last` against `llama-embedding --pooling last` on 20 sentences: every cosine ≥ 0.9999999 with flash attention off (llama-cpp-python's default). `llama-embedding`'s default turns flash attention on; then one sentence falls to 0.99883, the same gap `llama-embedding` shows between its own two modes.
- **Fixed: `--threads` ignored for batches.** `Runner` set `n_threads` but not `n_threads_batch`, which llama-cpp-python defaults to `cpu_count()`. Every embedding and prompt decode ran on all cores. Fixed; on 4 shared cores this made `embedtexts` 3.1× faster (19.0 → 58.3 tokens/s on 1,000 probe items). The vectors are bit-identical.
- **Fixed: `wdiff` and padded vocabularies.** Qwen3-Embedding's `token_embd` has 151,669 rows and Qwen3-0.6B-Base's 151,936. `wdiff` now compares the shared rows when both token lists agree on them, instead of reporting a shape mismatch.
- **Fixed: `spectra` memory.** It copied a whole tensor to float64 before sampling rows (5 GB for the 8B vocabulary matrix). It now samples first; the same rows are drawn.
- **Cost in embedding mode.** llama.cpp marks every token as an output and still runs the (tied) output head on each one, about 35% extra compute for the 0.6B model and about 9% for the 8B. This is llama.cpp's, not the script's.
- **Self-test after these fixes:** all 50 rows pass with both vocabularies (Qwen2 657 s, SPM 99 s, 2 cores).
- **Static commands on real models.** `info`, `vocab`, `embed`, `spectra`, `heads`, `neurons` and `wdiff` ran on Qwen3-Embedding-0.6B and its base, and on the 8B embedder (`wdiff` against Qwen3-8B-Base). `heads` and `neurons` cost vocabulary × width per token or neuron: use `--tokens 300` and `--max-neurons 1024` on 0.6B, and `--layers` subsets on 8B.

**Still to run, on Odin's PC (no inference in the cloud container from 2026-10-28).** The exact commands are in [QWEN3_EMBEDDING_DISSECTION.md](reports/QWEN3_EMBEDDING_DISSECTION.md) §9. The sizes are in Qwen3 tokens, with this container's rates at 2 threads: 0.6B 58.3 tokens/s; 8B about 4.6 (estimated at 12.7× the work per token); Qwen3.5-9B 10.6 tokens/s prompt and 1.91 generation.

| Step | Command | Tokens | Here, 0.6B / 8B |
|---|---|---|---|
| QE0 on 8B | `embedtexts` on `mined/qwen3emb/qe0/gate20.txt` and `llama-embedding -fa off` | 298 | — / 1 min |
| Timing | `embedtexts --limit 1000` (0.6B), `--limit 200` (8B) | 18,352 / 3,981 | 5 min / 14 min |
| QE2 probe vectors | `embedtexts --input probe.txt --batch 256 --chunk 2000 --resume` | 2,455,468 | 11.7 h / 6.2 days |
| QE3 geometry | `geometry --key text --pairs pos_pairs.tsv --labelled-pairs classes.tsv --lengths index` | none (vectors only) | minutes |
| B10 (Q6) | `sensitivity --sentences held_out.txt` | about 86,000 | 25 min / 5 h |
| B15–B16 (Q5) | `occlusion --sentences held_out.txt --freq word_counts.tsv` (2,000 sentences) | 888,387 | 4.2 h / 54 h |
| B8–B9 (Q10) | `instruct --queries questions.txt` (2,000 × 5 instructions) | 310,874 | 1.5 h / 19 h |
| 0.6B against 8B | `compare --key text` | none | minutes |
| QE4 (next wave) | `pairs` with Qwen3.5-9B, thinking off | about 3,000–4,300 prompt + 160–520 generated per one-sentence document | 6–11 min per document |

The nested subsets in [mined/probe/README.md](../mined/probe/README.md) (746k and 82k tokens) are for machines where the full set is too slow.

---

## Sources

- [GGUF format (ggml docs)](https://github.com/ggerganov/ggml/blob/9c2adc4962a3a5d259f10db2171e0df5c83e4b05/docs/gguf.md)
- [llama-cpp-python issue on logprobs/logits_all](https://github.com/abetlen/llama-cpp-python/issues/1983)
- [Petroni et al. 2019, Language Models as Knowledge Bases?](https://arxiv.org/pdf/1909.01066)
- [Elazar et al. 2021, Measuring and Improving Consistency in Pretrained Language Models (ParaRel)](https://transacl.org/ojs/index.php/tacl/article/view/3015)
- [Jiang et al. 2020, How Can We Know What Language Models Know?](https://www.arxiv.org/pdf/1911.12543)
- [Jiang et al. 2021, How Can We Know When Language Models Know? (calibration)](https://arxiv.org/pdf/2012.00955)
- [Cohen et al. 2023, Crawling the Internal Knowledge-Base of Language Models](https://arxiv.org/pdf/2301.12810)
- [GPTKB: Enabling LLM Knowledge Analysis via Extensive Materialization](https://www.arxiv.org/abs/2411.04920)
- [Hao et al. 2023, BertNet](https://arxiv.org/abs/2206.14268v3)
- [Jain & Espinosa-Anke 2022, Distilling Hypernymy Relations from Language Models](https://arxiv.org/pdf/2202.04876)
- [West et al. 2022, Symbolic Knowledge Distillation](https://aclanthology.org/2022.naacl-main.341)
- [Wang et al. 2023, Self-Instruct](https://arxiv.org/pdf/2212.10560)
- [Geva et al. 2022, Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space](https://arxiv.org/abs/2203.14680)
- [Elhage et al. 2021, A Mathematical Framework for Transformer Circuits (explainer)](https://www.lesswrong.com/posts/CJsxd8ofLjGFxkmAP/explaining-the-transformer-circuits-framework-by-example)
- [Belrose et al. 2023, Tuned Lens](https://arxiv.org/pdf/2303.08112)
- [Hernandez et al. 2024, Linearity of Relation Decoding in Transformer LMs](https://arxiv.org/pdf/2308.09124)
- [Dai et al. 2022, Knowledge Neurons in Pretrained Transformers](https://arxiv.org/abs/2104.08696v1)
- [Meng et al. 2022, Locating and Editing Factual Associations in GPT](https://arxiv.org/pdf/2202.05262)
- [Cunningham et al. 2024, Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/pdf/2309.08600)
- [Burns et al. 2023, Discovering Latent Knowledge Without Supervision](https://arxiv.org/pdf/2212.03827)
- [Farquhar et al. 2024, Detecting hallucinations using semantic entropy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750)
- [Model2Vec: static embeddings distilled from a sentence transformer](https://huggingface.co/blog/Pringled/model2vec)
- [WeightWatcher (heavy-tailed self-regularisation)](https://pypi.org/project/weightwatcher/0.5.5)
- [Carlini et al. 2021, Extracting Training Data from Large Language Models](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting)
- [Nasr et al. 2025, Scalable Extraction of Training Data from Aligned, Production Language Models](https://iclr.cc/virtual/2025/poster/27894)
- [Llama 3 vs Llama 3.1 licence terms](https://writing.kemitchell.com/2024/07/24/Llama-3-versus-Llama-3-1-License)
- [Qwen 2.5 release and licences](https://www.deeplearning.ai/the-batch/alibaba-releases-qwen-2-5-models-raising-the-bar-for-open-weight-llms)
- [LLM-empowered knowledge graph construction: a survey](https://arxiv.org/html/2510.20345v1)
