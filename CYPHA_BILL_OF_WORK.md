# Cypha — open work

Living checklist only. Completed July 2026 continuum / optimality / RPSM / cell-sweep closeout lives in the archive — do not treat those files as current pins.

| Kind | Where |
|------|-------|
| Research journal | [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) |
| Forward path | [`docs/FUTURE.md`](docs/FUTURE.md) |
| Docs hub | [`docs/README.md`](docs/README.md) |
| Competition card | [`MODEL_CARD.md`](MODEL_CARD.md) |
| Historical BoW (2026-07-18) | [`docs/archive/reports/CYPHA_BILL_OF_WORK_2026-07-18.md`](docs/archive/reports/CYPHA_BILL_OF_WORK_2026-07-18.md) |
| Historical optimality plan | [`docs/archive/plans/CYPHA_OPTIMALITY_PLAN_2026-07-18.md`](docs/archive/plans/CYPHA_OPTIMALITY_PLAN_2026-07-18.md) |
| Historical forecasting design | [`docs/archive/plans/CYPHA_FORECASTING_PLAN_2026-08-08.md`](docs/archive/plans/CYPHA_FORECASTING_PLAN_2026-08-08.md) |

**D17 bench lock:** Hybrid L2 + Wave2 BPTT **2.664 BPC** @ 300k (`bench/BASELINE_LOCK.json`). That is the overnight classifier/sequence pin, not the language model. The language model being taken forward is the packed trie. hp gate24 is the comparison baseline. [`docs/CURRENT.md`](docs/CURRENT.md). Prior L1 2.873 / SGD L2 2.816 are historical.

**Latest release:** **[v2.5.0](https://github.com/odin-loki/Cypha/releases/tag/v2.5.0)** (2026-09-20). Prior notes: [`docs/reports/RELEASE_V2_4_0_2026-08-16.md`](docs/reports/RELEASE_V2_4_0_2026-08-16.md).

## Open

**Default plan: [`docs/PLAN.md`](docs/PLAN.md)** (2026-10-26). Order of work: its section 12. First items:

- [ ] **Track A**: 1 TB packed-trie run; score each checkpoint against predictions P-A1 to P-A5.
- [x] **Q1–Q3 / B1** (done 2026-10-28, [report](docs/reports/B1_TRIE_CONSOLIDATION.md)): one packed-trie library (`native/include/cypha/cyphalm/packed_trie.hpp`) and one tool (`cyphalm_trie`: `bpc`, `chat`, `probe`, `context8k`, `sample`, `selftest`) replace ten tools with exact parity; 9,574 lines became 2,072. Paths are arguments. The CMake target and the `native_cyphalm_trie_selftest` CTest are in (2026-10-28). `trie_stream_train.cpp` keeps its own copy until the 1 TB run ends.
- [ ] **M1–M4**: frozen suites in `bench/eval/`; SCRIBE with its control ladders; `cypha_report_card`.
  - **SCRIBE v0 built (2026-10-27):** `native/include/cypha/eval/scribe.hpp`, `native/tools/scribe_ladder.cpp`, CTest `native_scribe_selftest`, [report](docs/reports/SCRIBE_V0.md).
  - **It fails its own ladder:** 1 of 6 adjacent pairs pass, the same as surface S. So it is not used to judge anything yet (PLAN §6.4).
  - **SCRIBE v1 (2026-10-27,** [report](docs/reports/SCRIBE_V1.md)**):**
    - Added real trie samples (`trie_sample.cpp`, now `cyphalm_trie sample` in `native/tools/cyphalm_trie.cpp`), SIF sentence vectors from PPMI-SVD, an I gate and length parity.
    - On a fresh, untuned ladder it passes 3 of 6 adjacent pairs; surface S passes 2.
    - Human beats trie output 0.993 of the time.
    - Still failing: trie output against shuffled sentences (the trie writes word salad, so the ladder order is wrong there), and the S-gate conflicts in the report. The ECS and LCI library is done (2026-10-27): `native/include/cypha/eval/`, CTest `native_ecs_lci_test` (242 checks), [report](docs/reports/M1_ECS_LCI_LIBRARY.md).
- [ ] **F1**: REMOVED 2026-10-27. `qa_infobox` answered without CyphaLM ([record](docs/reports/REMOVED_NON_CYPHALM_QA.md)). Fact Q/A is to be redone through CyphaLM (PLAN rule 10).
- [ ] **B7**: REMOVED 2026-10-27. `chat_index` answered without CyphaLM ([record](docs/reports/REMOVED_NON_CYPHALM_QA.md)). Fuzzy question handling is to be redone in CyphaLM (PLAN rule 10).
- [ ] **C0–C1**: fair TF-IDF baselines with `W_enc` frozen, then PPMI-SVD and random-indexing vectors.
  - **C1 vectors done (2026-10-27):** `native/tools/ppmi_svd.cpp`, [report](docs/reports/C1_PPMI_SVD_VECTORS.md). PPMI-SVD on enwik8 (11.2M tokens, 152 s, 640 MB) scores WordSim-353 0.672, SimLex-999 0.255 and MEN 0.637. WS353 is inside the published range; SimLex is below it. Random indexing scores 0.404 / 0.131 / 0.374.
  - **Still open:** the C0 baselines and the 20 Newsgroups gate.
- [ ] **F2**: pattern learning by distant supervision ([report](docs/reports/F2_PATTERN_LEARNING.md), `native/tools/pattern_learn.cpp`).
  - On all of enwik8 the fact store grows from 8,148 to 8,980 facts. The new ones are almost all birth and death dates and birth places.
  - Precision is 49/49 verifiable in a 50-fact sample (Wilson lower bound 0.927), and 35/37 for birth places.
  - [x] **D4 Wikidata seeds** (2026-10-28, [report](docs/reports/D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md)): 98.7% of enwik8 titles map to Wikidata; its values are distant supervision only. Kept patterns 44 → 70, admitted facts 832 → 1,216 (store +16.2%), agreement with Wikidata 0.951 [0.909, 0.974]. A seed bug is fixed: 664 citation seeds dropped, text seeds now 7,484, facts unchanged.
  - [x] **TOOL_PROFILE P1 adapter:** `pattern_learn --shard` writes question-form fact text with a built-in leakage check: 2,001 rows from 675 facts, 0 questions equal to a dev or test question, dev reach 16/100.
  - **Still open:** capital, author and director get no kept pattern even with Wikidata seeds; the 2006 text states them in too many wordings ("is the capital of [[X]]" occurs 12 times in all of enwik8). Next: pattern classes in the gaps (E1) and cross-article evidence. Clean the seed relations left out of the shard (country, location, language, nationality, party, predecessor, successor, spouse).
  - **Waiting for Odin:** the per-relation base rate (it admits 1,216 facts; the plan's 0.5 admits 0) and which leakage rule governs fact shards (TOOL_PROFILE §4 says no test answer string in an added shard; a fact shard must contain answers).
- [ ] **E4**: question classifier ([report](docs/reports/E4_QUESTION_CLASSIFIER.md); code removed 2026-10-27, in git at `707629a`).
  - On TREC, the averaged perceptron scores 89.0–91.6 coarse and 84.6 fine; it is the classifier to use.
  - The Cypha classifier scores 75.0 coarse and 62–67 fine, with badly low confidence and an OOD flag that never fired.
  - **Open research:** close the roughly 10-point gap that batch LDA (84.8 / 74.4) attributes to Cypha's online update noise; recalibrate Cypha's confidence on text; and look at the sparse-feature projection.
- [ ] **E1**: Cypha part-of-speech tagger on UD EWT (gate ≥ 94%).
  - **Perceptron baseline passes (2026-10-27):** `native/tools/pos_tagger.cpp`, [report](docs/reports/E1_POS_TAGGER.md). Test UPOS 94.81% (known words 96.4%, unknown 79.2%, questions 93.3%); about 1M tokens/s, 5.6 MB.
  - Noun-phrase chunker F1 83.8 on test, plus a question-parse helper.
  - **Still open:** the Cypha-classifier tagger, and entities (E2).
- [ ] **S1–S3, S5** ([`docs/DISCOVERY.md`](docs/DISCOVERY.md)): reason codes and evidence cards on every answer and every "I don't know"; epistemic states; the hypothesis store and firewall; the gap map.
  - **Done 2026-10-27** ([report](docs/reports/S1_S2_EPISTEMIC.md)):
    - **S1 for `chat_index`:** removed with `chat_index` on 2026-10-27 (PLAN rule 10).
    - **S2:** `native/include/cypha/eval/epistemic.hpp`, with CTest `native_epistemic_test` (195 checks): states, degree of conflict, the two kinds of uncertainty, reason codes, unseen estimators and cards.
  - **Still open:** S1 on CyphaLM's answers, S3 and S5.
- [ ] **K2–K5** ([`docs/THOUGHT.md`](docs/THOUGHT.md)):
  - K1 (144 processes including AI, data science and teacher mining, validator) is done.
  - **K2 and K3 done 2026-10-27** ([report](docs/reports/K2_K3_THOUGHT_INTERPRETER.md)): the interpreter and the standard learners in `native/include/cypha/thought/`, with CTest `native_thought_test` (5,534 checks). half_split takes 6 tests against 31.3 for random probing; SPRT errors are 0.028 / 0.038 at α = β = 0.05; selection regret falls from 243 to 92.5; ECE 0.015; replay is bit-identical.
  - **Still open:** K4 (primitives bound to CyphaLM and the existing libraries), K5 (first executable processes on their benchmarks), and DIF applicability in place of the per-(process, task) Beta.
- [ ] **"Don't know" message and tag** ([DISCOVERY §3.5](docs/DISCOVERY.md#35-the-dont-know-message-and-tag)). **First run 2026-10-28** ([report](docs/reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md)): CyphaLM writes the message itself (`cyphalm_trie dktext` builds the text), but its familiarity signal is weak (AUROC 0.54–0.60), and no tag names the entity. Odin ruled the control token allowed on the gate side (2026-10-28). **A1c done 2026-10-28** ([report](docs/reports/A1C_ENTITY_FAMILIARITY_AND_COPY.md)): entity-title familiarity reaches dev AUROC 0.795; as a gate token, precision 0.435 / 0.371 (dev / test) at recall ≈ 1.00; the copy expert cuts BPC (1.926 with InfiniGram) but no tag names the entity. **Next:** better entity linking than E1 for the token; a pointer from the question's entity to the tag; NO_EVIDENCE rows once the store can show a slot is empty. Original items:
  - build training text pairing unanswerable questions with the message and the `<|unknown ...|>` tag;
  - train CyphaLM on it;
  - parse emitted tags into the gap map;
  - score abstain recall and precision, and reason-code accuracy.
- [ ] **Semantics S-1 to S-6** ([`docs/SEMANTICS.md`](docs/SEMANTICS.md)): Core Cypha as the meaning layer.
  - **S-1 done 2026-10-28, gate failed** ([report](docs/reports/C5_WORDS_AS_CLASSES.md)): Cypha's word means reach count level (WordSim 0.240) but not PPMI-SVD (0.723); no is-a direction from variance; P1–P3 missed. PPMI-SVD stays the vector source.
  - **Next:** learned context vectors or PMI-weighted inputs, and the EMBEDDING_GEOMETRY §4.2 rule, in the same tool.
- [ ] **How CyphaLM answers by itself** ([`docs/CYPHALM_ANSWERING.md`](docs/CYPHALM_ANSWERING.md), proposal): three routes (question-form training text, generalising experts, statement checking), an answer loop and steps A0–A5. **Agreed 2026-10-28** (§5): R1 + R3 first, teacher text as a separate low-trust shard, frames from the parse and learned patterns. **A2 and A3 run 2026-10-28** ([report](docs/reports/A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md)): the fact shard gives 2 right / 0 wrong out of fold on dev (coverage 1.5%) and nothing on the test; R3 statement scoring gives 0% at 2% wrong. Teacher-shard runs are for Odin's PC. **Next:** A4 (R2 experts) after S-1b; F2 patterns learned on raw bytes so R3 frames match what CyphaLM read.
- [ ] **Embedding geometry QE0–QE9** ([`docs/EMBEDDING_GEOMETRY.md`](docs/EMBEDDING_GEOMETRY.md)): copy Qwen3-Embedding-8B's geometry into Cypha.
  - Needs Odin's machine: the GGUF downloads (Qwen3-Embedding 0.6B and 8B, Qwen3-8B, Qwen3.5-9B) and the teacher passes.
  - **Built 2026-10-28:** `native/include/cypha/semantics/geometry.hpp` with CTest `native_geometry_test` ([report](docs/reports/QE_GEOMETRY_LIBRARY.md)). On a planted teacher, distillation reaches pair Spearman 0.943 and kNN@10 0.889, against 0.367 / 0.429 for InfoNCE alone. With the mask, paraphrases end at cosine 0.750, against 0.640 without. CSLS cuts hub skew from 3.18 to 0.32.
  - **Built 2026-10-28:** the `gguf_mine.py` subcommands `embedtexts`, `geometry`, `compare`, `sensitivity`, `occlusion`, `instruct`, `wdiff` and `pairs`, plus `vectors --pooling last` with an EOS check ([GGUF_MINING.md](docs/GGUF_MINING.md) §14). The selftest passes with the Qwen2 and llama-spm vocabularies, with known-answer checks for Spearman, CKA, Procrustes, TwoNN, τ fitting and randomized SVD.
  - **Still to do on Odin's machine:** the QE0 pooling gate on real models, and the first real render of Qwen3.5-9B's chat template.
- [ ] **Tool profile** ([`docs/TOOL_PROFILE_PLAN.md`](docs/TOOL_PROFILE_PLAN.md)):
  1. Build the CyphaLM Q/A baseline harness. **Done 2026-10-28** (`cyphalm_trie qa`; baseline 0.0% coverage at 2% wrong).
  2. Write adapters for the tagger, the pattern learner, word classes, teacher text and the abstain gate.
     - **Done 2026-10-28:** the fact-text shard (`pattern_learn --shard`, D4) and the abstain gate (`native/tools/cypha_gate.cpp`, CTest `native_cypha_gate_selftest`).
     - **Done 2026-10-28:** the class-context expert `native/include/cypha/cyphalm/class_context.hpp`, shared by the tagger and the word classes ([B4 report](docs/reports/B4_CLASS_CONTEXT_EXPERT.md)): −0.078 BPC on fresh text with shape classes alone, −0.091 with E1 tags; surface S up. In `cyphalm_trie` as `cls` and `qa --cls`.
     - **Still to do:** the teacher-text shard (Qwen3.5-9B).
  3. Run the B / B+T / F / F−T grid with a leakage check.
  4. Keep only the tools that pass LCI and their lines-per-gain bar; delete the rest.
- [ ] **H1–H3**: pick a permissively licensed GGUF teacher. Run `scripts/gguf_mine.py calibrate` on a Wikidata gold sample, then `facts`, `lexical` and `crawl` with that calibration. List: [`docs/GGUF_MINING.md`](docs/GGUF_MINING.md).

- [x] **Cell sweep wrap-up** — B2 and H06 rerun 2026-08-16 @ 300k / eval 2k (isolated `bench/results/cell_sweep_rerun/`). Both **3.681 BPC** with math-integration (same as H14; not a promote vs Hybrid 2.664). Archived at `data/archive/cell_sweep/variant_B2.json` + `variant_H06.json`. Best new row remains U03 **2.822**.
- [x] **Paper PDF body** — regenerated 2026-08-16; living pin **2.664** in `paper/CyphaLM_paper.md` + `paper/arxiv_bundle/` (HTML/PDF). L1 2.873 kept as historical.
- [ ] **arXiv / venue upload** — human; bundle at `paper/arxiv_bundle/` and on the v2.5.0 GitHub Release (zip refreshed with the living PDF). See `paper/arxiv_bundle/SUBMIT_CHECKLIST.md`.
- [x] **Numerical audit — R1–R4 and N1–N4, L5–L9 fixed; L11–L13 recorded** — full report: [`docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md`](docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md). Every finding is hand-verified; L10 did not survive that pass. **Inference gate:** **R1** `world_gate` applied twice on the kernel-LLR path (45.8% confidence understatement) → `native_kernel_gate_invariant`; **R2** the OOD anomaly score divided a latent-variance-scaled `r_eff` by a dimensionless `mahal_ema`, so on any model whose latent variance sat below `mahal_ema` the flag never fired — now `max(0, r_eff/r_base − 1)`, dimensionless and scale-invariant; **R3** ECE binning dropped confidence exactly `1.0` and scored an all-NaN evaluation as a perfect `0.0`, which won the temperature search outright since it minimises ECE; **R4** `adapt_temperature_ece` never scored the incumbent, so it could install a worse temperature and report success. R2/R3/R4 → `native_ported_defects_pinned`, verified red-then-green against the shipped function. The Python-parity objection to fixing R3/R4 was void: Python was decommissioned at P7 (`CHANGELOG.md:55`). **Bessel/GIG kernels:** N1–N4 and L5–L9 turned out to be one defect seen nine ways — `K₂/K₁` was the primitive and `K₀/K₁` derived from it, when `K₂/K₁` carries a `2/x` pole and `K₀/K₁` is bounded in `[0,1)`. Inverting that via the exact recurrence `K₂/K₁ = 2/x + K₀/K₁` took the worst error from **183,084% to 1.9e-6**, put `K₀/K₁` back inside `[0,1)` (it was **negative on 44.1%** of the score-match domain), and let all four wrong GIG-moment limit branches be deleted rather than corrected. → `native_bessel_gig_limits`. One golden fixture was regenerated: it pinned Python-parity values carrying the same error, and the new values are 8–15× closer to `scipy`. The CUDA device path carried the same four defects and now mirrors the host (verified bit-identical by transcribing the device functions to host C++; **not compile-tested — no CUDA toolchain in this environment**), with a text-level drift guard that needs no GPU. Fixing the kernels also exposed that the **Phase 7 acceptance gate was inverted**: `native_gate_score_match_p7_smoke` asserted only `loglik_score_match >= loglik_lut` with no reference value, so it rewarded whichever backend overstated the likelihood — it had been passing by a margin of +0.159 while score-match sat **8.5× further from the `scipy` truth** than the LUT. It now pins the exact held-out log-likelihood and asserts both backends track it. Constants and the previously-unreproducible Bessel table both have provenance now: [`scripts/gen_gig_k0k1_fit.py`](scripts/gen_gig_k0k1_fit.py). **Recorded, not fixed:** **L11**'s NaN asymmetry (CUDA `fmax(NaN, 0) = 0` vs the CPU's `std::max(NaN, 0) = NaN`, which needs a design call on what a NaN Mahalanobis should mean; its `pool_ensure` stale-capacity half **is** fixed), **L12** (`field_diag_a` allocates before its `fd <= 0` guard), **L13** (`infer_at_h` hardcodes `gh_chi = gh_psi = 1.0`, silently discarding caller options on the non-GH path).

## Future (not blocking)

- **CPU SIMD via [xsimd](https://github.com/xtensor-stack/xsimd)** — portable AVX2/AVX-512/NEON kernels for `score_matrix` / matvec / softmax. Training stays on CPU; this is the speed path that is actually worth it. Spec: [`docs/FUTURE.md`](docs/FUTURE.md) §1b.

## Closed (do not reopen)

Optimality P0–P9, RPSM Option B, 28-variant July cell sweep, v2.3.24 / v2.3.25 publish, D10A 90.11%, XOR latent RFF default, forecasting Phases 1–9, **GPU training** (CUDA infer is optional; device training is slower than CPU on this workload — not a gap). Details: archive BoW + [`docs/archive/README.md`](docs/archive/README.md).
