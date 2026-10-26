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
- [ ] **Q1–Q3 / B1**: one packed-trie library replacing the 7 forked `trie_*.cpp` tools; CMake targets; CTests; paths as arguments.
- [ ] **M1–M4**: frozen suites in `bench/eval/`; SCRIBE with its control ladders; `cypha_report_card`.
  - **SCRIBE v0 built (2026-10-27):** `native/include/cypha/eval/scribe.hpp`, `native/tools/scribe_ladder.cpp`, CTest `native_scribe_selftest`, [report](docs/reports/SCRIBE_V0.md).
  - **It fails its own ladder:** 1 of 6 adjacent pairs pass, the same as surface S. So it is not used to judge anything yet (PLAN §6.4).
  - **Next (the report's list):** real model generations for the model rung; sentence vectors for coherence; a gate on I; a better E null; length parity. The ECS and LCI library is done (2026-10-27): `native/include/cypha/eval/`, CTest `native_ecs_lci_test` (242 checks), [report](docs/reports/M1_ECS_LCI_LIBRARY.md).
- [x] **F1**: infobox facts + question classifier v0 (2026-10-27). The 8 Wikipedia questions are 8/8, up from 0/8, but this is in-sample: the patterns were written after reading those articles. 9 of 14 further questions correct, 5 abstained, 0 wrong. Fact precision is 27/30 with a Wilson lower bound of 0.744, below the F2 bar of 0.90; the facts need human review. `native/tools/qa_infobox.cpp`, `bench/eval/t_qa_wiki_seed.tsv`, [report](docs/reports/F1_INFOBOX_QA.md). Held-out test (2026-10-27, `bench/eval/t_qa_wiki.tsv`, 200 blind questions, [report](docs/reports/T_QA_WIKI_200.md)): 3 right / 147 abstained / 0 wrong on the 150 answerable questions, so coverage is 2%. It abstained on all 50 unanswerable questions, with 0 wrong answers out of 200. Honest, but nowhere near the 70% at 80% coverage target. 142 of the 150 questions matched no question template, so the bottleneck is question parsing and fact extraction (E4, F2).
- [ ] **B7**: normalised, fuzzy chat index that can abstain, reported as retrieval. Dolly part done (2026-10-27): `native/tools/chat_index.cpp`, [report](docs/reports/B7_CHAT_INDEX.md). The gate passes with no abstain and narrowly with abstain on, but not on every resample (the synonym form fails 1 of 4 seeds for two scorers). Open: OpenAssistant (not reachable from here); encoder similarity and Cypha OOD as the abstain signal. On the 200 blind questions it gave 4 wrong answers, all to questions with no stored answer: 3 were true Dolly facts and 1 was nonsense. `--ask` reruns the whole evaluation sweep (about 52 s per question) and needs a fast path.
- [ ] **C0–C1**: fair TF-IDF baselines with `W_enc` frozen, then PPMI-SVD and random-indexing vectors.
  - **C1 vectors done (2026-10-27):** `native/tools/ppmi_svd.cpp`, [report](docs/reports/C1_PPMI_SVD_VECTORS.md). PPMI-SVD on enwik8 (11.2M tokens, 152 s, 640 MB) scores WordSim-353 0.672, SimLex-999 0.255 and MEN 0.637. WS353 is inside the published range; SimLex is below it. Random indexing scores 0.404 / 0.131 / 0.374.
  - **Still open:** the C0 baselines and the 20 Newsgroups gate.
- [ ] **E1**: Cypha part-of-speech tagger on UD EWT (gate ≥ 94%).
- [ ] **S1–S3, S5** ([`docs/DISCOVERY.md`](docs/DISCOVERY.md)): reason codes and evidence cards on every answer and every "I don't know"; epistemic states; the hypothesis store and firewall; the gap map.
  - **Done 2026-10-27** ([report](docs/reports/S1_S2_EPISTEMIC.md)):
    - **S1 for `chat_index`:** reason codes and evidence cards; `--ask`/`--ask-file` fast path, 1.3 s instead of 53 s. On the 200 blind questions the strict reading gives 4 wrong; counting true Dolly facts as evidence gives 1.
    - **S2:** `native/include/cypha/eval/epistemic.hpp`, with CTest `native_epistemic_test` (195 checks): states, degree of conflict, the two kinds of uncertainty, reason codes, unseen estimators and cards.
  - **Still open:** S1 for `qa_infobox` (in progress), S3 and S5.
- [ ] **H1–H3**: pick a permissively licensed GGUF teacher. Run `scripts/gguf_mine.py calibrate` on a Wikidata gold sample, then `facts`, `lexical` and `crawl` with that calibration. List: [`docs/GGUF_MINING.md`](docs/GGUF_MINING.md).

- [x] **Cell sweep wrap-up** — B2 and H06 rerun 2026-08-16 @ 300k / eval 2k (isolated `bench/results/cell_sweep_rerun/`). Both **3.681 BPC** with math-integration (same as H14; not a promote vs Hybrid 2.664). Archived at `data/archive/cell_sweep/variant_B2.json` + `variant_H06.json`. Best new row remains U03 **2.822**.
- [x] **Paper PDF body** — regenerated 2026-08-16; living pin **2.664** in `paper/CyphaLM_paper.md` + `paper/arxiv_bundle/` (HTML/PDF). L1 2.873 kept as historical.
- [ ] **arXiv / venue upload** — human; bundle at `paper/arxiv_bundle/` and on the v2.5.0 GitHub Release (zip refreshed with the living PDF). See `paper/arxiv_bundle/SUBMIT_CHECKLIST.md`.
- [x] **Numerical audit — R1–R4 and N1–N4, L5–L9 fixed; L11–L13 recorded** — full report: [`docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md`](docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md). Every finding is hand-verified; L10 did not survive that pass. **Inference gate:** **R1** `world_gate` applied twice on the kernel-LLR path (45.8% confidence understatement) → `native_kernel_gate_invariant`; **R2** the OOD anomaly score divided a latent-variance-scaled `r_eff` by a dimensionless `mahal_ema`, so on any model whose latent variance sat below `mahal_ema` the flag never fired — now `max(0, r_eff/r_base − 1)`, dimensionless and scale-invariant; **R3** ECE binning dropped confidence exactly `1.0` and scored an all-NaN evaluation as a perfect `0.0`, which won the temperature search outright since it minimises ECE; **R4** `adapt_temperature_ece` never scored the incumbent, so it could install a worse temperature and report success. R2/R3/R4 → `native_ported_defects_pinned`, verified red-then-green against the shipped function. The Python-parity objection to fixing R3/R4 was void: Python was decommissioned at P7 (`CHANGELOG.md:55`). **Bessel/GIG kernels:** N1–N4 and L5–L9 turned out to be one defect seen nine ways — `K₂/K₁` was the primitive and `K₀/K₁` derived from it, when `K₂/K₁` carries a `2/x` pole and `K₀/K₁` is bounded in `[0,1)`. Inverting that via the exact recurrence `K₂/K₁ = 2/x + K₀/K₁` took the worst error from **183,084% to 1.9e-6**, put `K₀/K₁` back inside `[0,1)` (it was **negative on 44.1%** of the score-match domain), and let all four wrong GIG-moment limit branches be deleted rather than corrected. → `native_bessel_gig_limits`. One golden fixture was regenerated: it pinned Python-parity values carrying the same error, and the new values are 8–15× closer to `scipy`. The CUDA device path carried the same four defects and now mirrors the host (verified bit-identical by transcribing the device functions to host C++; **not compile-tested — no CUDA toolchain in this environment**), with a text-level drift guard that needs no GPU. Fixing the kernels also exposed that the **Phase 7 acceptance gate was inverted**: `native_gate_score_match_p7_smoke` asserted only `loglik_score_match >= loglik_lut` with no reference value, so it rewarded whichever backend overstated the likelihood — it had been passing by a margin of +0.159 while score-match sat **8.5× further from the `scipy` truth** than the LUT. It now pins the exact held-out log-likelihood and asserts both backends track it. Constants and the previously-unreproducible Bessel table both have provenance now: [`scripts/gen_gig_k0k1_fit.py`](scripts/gen_gig_k0k1_fit.py). **Recorded, not fixed:** **L11**'s NaN asymmetry (CUDA `fmax(NaN, 0) = 0` vs the CPU's `std::max(NaN, 0) = NaN`, which needs a design call on what a NaN Mahalanobis should mean; its `pool_ensure` stale-capacity half **is** fixed), **L12** (`field_diag_a` allocates before its `fd <= 0` guard), **L13** (`infer_at_h` hardcodes `gh_chi = gh_psi = 1.0`, silently discarding caller options on the non-GH path).

## Future (not blocking)

- **CPU SIMD via [xsimd](https://github.com/xtensor-stack/xsimd)** — portable AVX2/AVX-512/NEON kernels for `score_matrix` / matvec / softmax. Training stays on CPU; this is the speed path that is actually worth it. Spec: [`docs/FUTURE.md`](docs/FUTURE.md) §1b.

## Closed (do not reopen)

Optimality P0–P9, RPSM Option B, 28-variant July cell sweep, v2.3.24 / v2.3.25 publish, D10A 90.11%, XOR latent RFF default, forecasting Phases 1–9, **GPU training** (CUDA infer is optional; device training is slower than CPU on this workload — not a gap). Details: archive BoW + [`docs/archive/README.md`](docs/archive/README.md).
