# Cypha's modules: where the native code lives

**Odin Loch** · 2026-11-03 · the layout of `native/` by module

Odin asked for Cypha to be reorganised into modules: online learning, classification, the language model and so on.
The native code is now grouped by what it does. Each module keeps its headers in `native/include/cypha/<module>/`, its
sources in `native/src/<module>/` and its tools in `native/tools/<module>/`.

The behaviour did not change. The files were moved with `git mv`, so `git log --follow` shows each file's history.
The build, the tests and a full `context8k` run give the same results before and after. The libraries are the same
as before (`cypha_core` and `cypha_bench_native`). The modules are folders, not separate libraries. The C++
namespaces did not change with the folders. The language model is still `cypha::cyphalm`, the classifier is still in
`cypha`, and the discovery-track headers now in `knowledge/` are still in `cypha::eval`. The one new namespace is
`cypha::online`, for the content mixer.

---

## 1. The map

| Module | What it does | Main headers (`include/cypha/<module>/`) | Main tools (`tools/<module>/`) |
|--------|--------------|------------------------------------------|--------------------------------|
| **`core`** | Shared plumbing with no model in it: environment variables, random numbers, portable helpers, row parallelism, the accel backend (CPU threads or CUDA), the model registry, CSV input, SQLite experiments, NPZ and ONNX writers. | `env`, `mt19937_rng`, `portable_shuffle`, `parallel_rows`, `accel_backend`, `registry`, `csv_ingest`, `npz_util`, `onnx_min_writer` | `cuda_smoke`, `registry_register` |
| **`lm`** | CyphaLM, the byte language model with no neural network: the streaming and packed tries, byte Kneser–Ney, InfiniGram and its FM index, tokenizers and token models, the class-context, similar-context, document and copy experts, the content plan, hp (the gate24 comparison), generation and samplers, the arithmetic coder. | `packed_trie`, `stream_trie`, `byte_kn`, `infinigram`, `fm_infinigram`, `class_context`, `similar_context`, `doc_cache`, `copy_expert`, `tokenizer`, `token_model`, `token_bridge`, `cyphalm_model`, `hp_backend`, `arithmetic_coder` | `cyphalm_trie`, `cyphalm_tok`, `cyphalm_tok_gen`, `trie_ckpt`, `trie_stream_train_v2`, `cyphalm_generate`, `cyphalm_lm_quality`, `cyphalm_train` |
| **`online`** | Online learning. The content mixer: each expert mixed into the chain by a lambda per bucket, fitted or learned online, and the online final layer over every stage. Also the adaptive predictor mixer, elastic weight consolidation (EWC), priority replay and the curriculum. | `content_mixer`, `adaptive_predictor_mixer`, `ewc_regularizer`, `replay_buffer`, `curriculum` | `ewc_smoke`, `ewc_d16b_sweep`, `curriculum_smoke` |
| **`classify`** | Classification. Core Cypha's classifier and its regressors (the DIF): inference, training, the class memory, the NIG field, kernels, encoders, preprocessing, retrieval, the latent sampler, save and load. Also the classifiers built on it: class GMMs, multilabel, the Branch A router, topic routing, the part-of-speech taggers, federated merging. | `infer_cpu`, `memory_train`, `train_step_vector`, `load_cypha`, `nig_field`, `kernel_memory`, `class_gmm`, `regression`, `multilabel_dif`, `branch_a_router`, `text_features` | `cypha_one_smoke`, `cypha_save_load_test`, `cypha_ood_test`, `cypha_tagger`, `pos_tagger`, `topic_bench`, `cypha_onnx_export` |
| **`eval`** | Evaluation. SCRIBE, ECS and LCI, the Wikidata grader, the bench suite (domains, encoders, reports), the report card, the baseline lock and the MC/MG curves. | `scribe`, `ecs`, `lci`, `wd_grader`, `bench_*`, `text_algebraic_fingerprint` | `scribe`, `ecs_lci_test`, `cypha_report_card`, `cypha_bench_run`, `cypha_baseline_lock`, `surface_score` |
| **`knowledge`** | Knowing what Cypha knows (Track S and thought). The epistemic layer, the hypothesis store and its generators, the gap map, possible worlds, curiosity, evidence sources and cards, unknown-unknown signals, the thought interpreter and its learners, the intelligence profiler and its LM hooks, and `pattern_learn`. | `epistemic`, `hypothesis`, `gap_map`, `gen_*`, `worlds`, `curiosity`, `sources`, `evidence`, `interpreter`, `bindings`, `intelligence_profiler` | `pattern_learn`, `epistemic_test`, `thought_test`, `thought_k5`, `thought_k6` |
| **`forecast`** | Event forecasting (unchanged folder). | `forecast_pipeline`, `node_estimator`, `rollout_tree` | `cypha_forecast_run`, `forecast_smoke` |
| **`semantics`** | Meaning from counts: word vectors, word classes, embedding geometry. | `geometry` | `cypha_word_classes`, `ppmi_svd`, `geometry_test` |
| **`som`** | Self-organising maps and growing neural gas (unchanged folder; no tools). | `som_encoder`, `gng_expert` | — |
| **`serve`** | The REST server `cypha_rest`, its routes and the Studio web UI. | `dif_rest`, `cyphalm_rest`, `branch_a_rest`, `intelligence_rest` | `cypha_rest`, `embed_static_ui`; the UI in `tools/serve/static/` |

`native/include/cypha/cypha.hpp` and `native/src/cypha.cpp` sit above the modules. `cypha::Cypha` is the one public
type, and it joins classify, lm, online and knowledge.

## 2. Building one module's tools

Every module with tools has a build target named `cypha_<module>_tools`. It builds every executable whose main source
is in that module's `tools/` folder.

```bash
cmake -S native -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cypha_lm_tools        # every CyphaLM tool
cmake --build build --target cypha_online_tools    # the online-learning tools
cmake --build build --target cyphalm_trie          # one tool
```

The targets are `cypha_core_tools`, `cypha_lm_tools`, `cypha_online_tools`, `cypha_classify_tools`,
`cypha_eval_tools`, `cypha_knowledge_tools`, `cypha_forecast_tools`, `cypha_semantics_tools` and `cypha_serve_tools`.
The goldens in `tests/regression/` belong to no module target. `ctest` runs them as before.

Some tools are built by hand, not by CMake: `pattern_learn`, `pos_tagger`, `ppmi_svd`, `cypha_word_classes`,
`trie_stream_profile`, `surface_score`, `stats_byte_lm`, `vdj_combo_bpc`, `hash_structure`, `hp_min_smoke` and
`cypha_routed_lm`. The first four give their build command in the file's header. `scripts/build_1tb_tools.sh` builds
the 1 TB tools, `ppmi_svd` and `trie_stream_profile` among them.

## 3. Adding an expert to the content mixer

The content mixer is `include/cypha/online/content_mixer.hpp`. It is header-only, like the rest of the trie stack, so
`cyphalm_trie` still builds from the same three files.

- **A chain stage.** An expert gives, for each holdout byte k, its probability of the true byte `pe[k]` and a bucket
  `bk[k]` in `[0, nb)`. `mix_into(pe, bk, half, nb, cur)` mixes it into the chain `cur`. It fits one lambda a bucket
  on the first half and freezes it. With `g_online_mix` (`--online-mix`) the lambdas learn online instead.
  `mix_bkn_into`, `mix_cls_into`, `mix_copy_into` and `mix_doc_into` are the existing stages, written this way.
- **The final layer.** `OnlineFinal<Inputs>` mixes every expert and every stage once more (`--online-final`). Its
  inputs are probabilities of the true byte, with the chain last. A new expert is one more input: `OnlineFinal<25>`.
  The caller picks the weight set (in `context8k`, InfiniGram's bucket and the previous byte's class).

## 4. What stayed where, and why

- **`native/tools/trie_stream_train.cpp`** stays, byte for byte. Odin's standing rule: it is not to be moved or
  changed. `scripts/run_train.sh` and `scripts/build_1tb_tools.sh` build it by that path. It includes only standard
  headers. One of its comments still names `cypha/cyphalm/stream_trie.hpp`, which is now `cypha/lm/stream_trie.hpp`.
- **The hybrid-era code stays where it was.** That is `include/cypha/cyphalm/` (33 headers), `src/cyphalm/` (27
  sources) and 13 tools left in `native/tools/`: GRIA, the LSTM, the SSM and CellAI cells, the Hebbian layers, the NIG
  cells and the PGM cell. No target builds its sources. Its 12 tools build only with `CYPHA_BUILD_LEGACY_CYPHALM`,
  which is off, and `kernel_llm_h04_smoke` has no build at all. So there is no `research` module. Nine of its headers
  are still compiled, because `lm/cyphalm_model.hpp` includes `cyphalm/cyphalm_ewc_regularizer.hpp`, which brings in
  the LSTM, GRIA, CellAI and Hebbian headers it names. None of its sources is. If these models are built again, they
  can move to a `research` module.
- **`tests/`** stays: the goldens, their driver `cypha_golden_run` (it came from `apps/`) and `rest_schema_contract`.
- **`qt/`** (the Studio shell) stays.
- **`forecast/`, `som/` and `semantics/`** were already modules. They keep their folders.
- **`native/apps/` is gone.** `cypha_rest` went to `tools/serve`, its route files to `src/serve`, the bench, tune and
  diagnostics runners to `tools/eval`, `cyphalm_train` to `tools/lm`, and `cypha_golden_run` to `tests/regression`.
- **`bench/` and `scripts/`** (Python and shell) keep their own layout.
- **`paper/`** is not changed. The arXiv bundle and the SCRIBE v1.0 code bundle describe the layout they were
  published with. `paper/scribe/bundle.sh` now copies from the new paths into that same bundle layout.

## 5. Where the map differs from the first sketch

The first sketch was core, lm, online, classify, eval, knowledge, forecast, semantics, som, serve and research. After
reading the code, six things changed.

- **Word classes and `ppmi_svd` are in semantics, not classify.** `cypha_word_classes` calls itself the single
  semantics tool. It learns what words mean and exports word vectors and word classes. `ppmi_svd` makes the word
  vectors that SCRIBE reads.
- **The discovery track left eval for knowledge.** `epistemic`, `hypothesis`, `gap_map`, the `gen_*` generators,
  `curiosity`, `worlds`, `sources`, `unknown_signals` and `abc_closed` propose and test hypotheses. They do not score
  anything. eval keeps the scores (SCRIBE, ECS, LCI) and the Wikidata grader. `epistemic_test` tests mostly the
  discovery track, so it is in `tools/knowledge`.
- **Core Cypha's regressors are in classify.** They share the classifier's memory, inference and training code.
- **The copy expert is in lm** (`lm/copy_expert.hpp`), not in online. The samplers use it without the mixer.
- **`text_algebraic_fingerprint` is in eval.** It is the MS2 metric on text.
- **There is no research module** (see section 4).

## 6. What moved with the paths

A few places compute a path instead of naming one. Each was changed so it finds the same file as before.

- `cyphalm_memorization_canary`, `cyphalm_algebraic_fingerprint` and `cyphalm_lossy_bench` find the repo from their own
  source path. They are one folder deeper now, so each climbs one more level.
- `pattern_learn` finds `data/f2c/wd_seed_types.tsv` from its own source path. It now looks for
  `native/tools/knowledge/pattern_learn.cpp`.
- The bench domains d42, d43, d56 and d57 check that some source files exist. They check the new paths, and their
  JSON keys name the new paths.
- `cypha_rest` looks for the web UI in `tools/serve/static` when it runs from the source tree.
- Usage lines and comments that name a source file name its new path.
