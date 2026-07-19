# D10 ECG / SSM diagnosis (2026-07-11)

> **Superseded (2026-07-18):** Real-data ECG5000 D10A now **90.11%** accuracy on HEAD — see [`D10_ECG5000_REAL_DATA_2026-07-18.md`](D10_ECG5000_REAL_DATA_2026-07-18.md) and [`CONTINUUM_CLOSEOUT_2026-07-18.md`](CONTINUUM_CLOSEOUT_2026-07-18.md).

**Scope:** Re-investigate the `docs/FUTURE.md` §0c claim that "D10 ECG: 17–20% accuracy on
5-class temporal classification (chance = 20%); CellAI/SSM integration not yet tuned for
this domain," per the recommended instrumentation (state-norm collapse/explosion, τ_fast/τ_slow
decay-rate suitability, output-projection connectivity to the expert routing head).

**Bottom line: the finding is stale. It does not reproduce on current HEAD (`e951f73`).**
D10's scored ECG classification (10A) currently measures **60.67% accuracy**, roughly 3×
chance and far above the "17–20%" figure in the doc. No fix was required; no SSM defect
exists in the path that produces this score, because that path doesn't use the SSM at all.
This report documents why, so the stale doc entry can be retired without further engineering
effort spent on a phantom SSM bug.

---

## 1. Reproduction on current HEAD

- **HEAD:** `e951f73` (`feat(kernel-llr): kernelize D14 Feynman expert-routing discriminant...`), confirmed via `git log --oneline -20`.
- **Build:** fresh `native/build_d10/` (MinGW Makefiles, GCC 13.2.0, `-DCMAKE_BUILD_TYPE=Release`), targets `cypha_bench_run` and `cyphalm_ssm_diagnose`.
- **Command:** `.\native\build_d10\cypha_bench_run.exe --domain 10`
- **Result (run twice, identical both times — the domain uses a hardcoded seed, `kBenchSeed=42`, so it's fully deterministic):**

| Experiment | Task | Accuracy | Chance | Notes |
|---|---|---|---|---|
| **10A_ecg_classification** | ECG 5-class temporal classification (the exact task the doc references) | **0.6067** | 0.20 | 3.03× chance |
| 10B_ecg_sliding_window | ECG 5-class, sliding-window features | 0.2946 | 0.20 | modestly above chance |
| 10C_ecg_ood_detection | OOD (normal vs. abnormal) AUROC | 0.5000 (AUROC) | 0.50 | chance — but this experiment isn't the one the doc flagged (it's OOD detection, not classification, and its own bench design allows a chance-level baseline) |
| 10D_financial_return_sign | financial return-sign — unrelated to ECG, doc explicitly doesn't claim this one is broken (`note: near_chance_expected` in its own output) | 0.5304 | 0.50 | as expected per its own annotation |

Raw output captured at `bench/results/d10_repro_2026-07-11.log` and `bench/results/d10_repro_run2_2026-07-11.log`; full per-domain JSON in `bench/BASELINE_REPORT.md` §D10 (generated fresh by this run) and `bench/report/summary.json`.

**Conclusion for step 1 of the task:** the specific claim in `docs/FUTURE.md` §0c — 17–20% (chance-level) accuracy on D10's 5-class ECG classification — **no longer holds**. Current accuracy is 60.67%, a ~3× improvement over the documented figure. This was evidently fixed as a side effect of unrelated work landed since the doc entry was written (the doc entry predates this session's many DIF/routing/causal-graph fixes — see `git log --oneline -20`, e.g. `e31b8ac fix(intelligence): thread persistent CausalGraphMonitor into report path, fixing causal_fidelity=0.0 degeneracy`, `1805c9b feat(intelligence): wire causal-graph fidelity into kappa/criticality_score`, and earlier DIF/expert-routing fixes not shown in the last 20 commits). No commit in the visible history specifically targets "D10" or "ECG," confirming the improvement was incidental, not a targeted fix.

Per the task's explicit instruction — *"If it's no longer chance-level ... say so clearly, document that finding, and stop — don't manufacture work on an already-resolved problem"* — **no fix was implemented**, and none was needed.

---

## 2. Why the doc's proposed diagnosis (SSM state norms / decay rates / routing-head connectivity) would have been the wrong lens even if the score were still bad

This is the more important finding for correcting the doc, independent of the accuracy number:

**D10's scored classification experiments (10A–10D) never construct or call the CellAI SSM.** Traced the full call chain in `native/src/bench/bench_domains.cpp`:

```
run_d10_experiment_a()                     bench_domains.cpp:2553
  └─ load_ecg5000(42)                        (real ECG5000 files absent → synthetic fallback, 500 train / 450 test, 140 timesteps, 5 balanced classes)
  └─ TimeSeriesEncoder(win=32, n_fft=16)      → 24-dim hand-engineered features (8 stats + 16 FFT bins)
  └─ make_online_classifier() → OnlineClassifier { CyphaInferModel, CyphaDifMemoryState }
  └─ train_classifier_online(passes=8)  → dif_train_step_vector()   (cypha_core DIF expert routing — NOT CyphaLM/SSM)
  └─ clf_metrics_native()  → batch_llr_from_x()
```

There is **no `CellAISSM`, no `CyphaLMModel`, no `ssm_step()`, no GRIA routing head** anywhere in this path. The classifier backend is `cypha_core`'s DIF (Distributed Inference Field) expert routing, a completely different mechanism from the CyphaLM SSM stack the doc's §0c section is actually about.

The **only** place D10 touches `CellAISSM` is an *optional*, *forward-only*, *non-scored* diagnostic experiment:

```
run_d10()  [only if --ssm-diagnose / CYPHA_SSM_DIAGNOSE=1]   bench_domains.cpp:2666-2677
  └─ run_d10_ssm_diagnose()  → "10E_ssm_diagnose"             bench_domains.cpp:1287
       └─ constructs a bare CellAISSM(cfg), feeds it encoded ECG features, calls diagnose_cellai_sequence()
       → no training, no classification, no connection to the routing head, does not affect 10A's score in any way
```

**Conclusion:** the doc's §0c framing conflates two unrelated subsystems under one bullet point. D10's ECG accuracy is (and always was) a property of the DIF expert-routing classifier + hand-engineered time-series features, not of the CellAI SSM. Any low-accuracy result on D10 should have been diagnosed by looking at DIF training (learning rate, pass count, feature representation) — not SSM decay rates or state norms. This report retires the SSM framing for D10 specifically; §0c's D17 CyphaLM claims are unaffected (that domain genuinely uses the SSM/GRIA stack) and out of scope here.

---

## 3. Supplementary SSM diagnostic (run anyway, for completeness against the doc's 3 hypotheses)

Even though the SSM isn't on D10's scored critical path, the tool the doc asked for (`cyphalm_ssm_diagnose` / `--ssm-diagnose`) **already exists** (`native/tools/cyphalm_ssm_diagnose.cpp`, `native/src/cyphalm/ssm_diagnose.cpp`, CTest `native_cyphalm_ssm_diagnose_d10_smoke`) — it did not need to be built. Ran it against D10's ECG-encoded sequence for completeness, addressing the doc's three explicit hypotheses:

**Command:** `.\native\build_d10\cyphalm_ssm_diagnose.exe --domain d10 --steps 512 --seed 42` (output: `bench/results/d10_ssm_diagnose_2026-07-11.log`)

| Hypothesis (from doc) | Result | Verdict |
|---|---|---|
| (a) State norms collapse/explode over long sequences | `fast`: min 0.46 → max 8.33 (final 8.32); `slow`: min 0.05 → max 9.00 (final 9.00); `context`: min 0.46 → max 15.50 (final 15.50). `checks_passed: true`; `collapsed: false`; `exploded: false` for all three. Norms grow smoothly and plateau — no collapse, no blow-up over 512 steps. | **Ruled out.** |
| (b) Multi-scale decay rates (τ_fast/τ_slow) mismatched for the domain | D10's SSM-diagnose config uses `tau_fast=10.0, tau_slow=100.0` (hardcoded at `bench_domains.cpp:1293-1294` and `cyphalm_ssm_diagnose.cpp`), **not** the doc's cited 1.0/20.0 (those are `CyphaLMConfig`'s defaults for D17/CyphaLM, a different config struct entirely — `cyphalm_config.hpp:39-40`). The diagnostic tool itself flags one automatic recommendation: `ssm_slow_decay_too_fast` (severity: medium) — worth noting for future SSM-domain tuning, but moot for D10 accuracy since 10A doesn't route through this SSM. | **Not applicable to D10's score** (SSM isn't on the classification path); tau values are correctly domain-scoped where they *are* used, though the tool's own heuristic flags room for tuning if CellAISSM is ever wired into an ECG-scored path in the future. |
| (c) Output projections properly connected to expert routing head | N/A for D10 — `run_d10_ssm_diagnose()` only calls `diagnose_cellai_sequence()` (forward dynamics only), which does not exercise `project_field()`/`proj_ssm_`/GRIA routing at all. That connectivity check (`ssm_projection_rms`, `connected_to_routing`) only runs via `diagnose_model_tokens()` for CyphaLM/D17-style domains that actually have a `CyphaLMModel` with a routing head. D10 has no routing head to disconnect from. | **Not applicable** — there is no SSM→routing-head wiring in D10 to be broken. |

None of the three hypotheses explain D10's (now-resolved) low accuracy, because the SSM was never in the loop for D10's classification score.

---

## 4. D10's actual config (checked per the task's step 3 — is this a data/training-scale issue?)

From `native/src/bench/bench_encoder_timeseries.cpp` and `native/src/bench/bench_domains.cpp` (all hardcoded, no `bench/config/*.json` entry exists for `d10`):

| Parameter | Value |
|---|---|
| Data source | Synthetic (UCR `ECG5000_{TRAIN,TEST}.txt` files absent from repo at `bench/data/ecg5000/`; falls back to a synthetic sinusoidal generator) |
| n_train / n_test | 500 / 450 |
| Sequence length | 140 timesteps |
| Classes | 5, perfectly balanced (`i % 5`) |
| Feature representation | `TimeSeriesEncoder(window=32, n_fft=16)` → 24-dim (8 summary stats + 16 FFT magnitude bins) per series |
| Model | `OnlineClassifier` (DIF-backed), `enc_lr` from profile (default 0.002) |
| Training passes | 8 (10A), 4 (10B), 3 (10C) |
| Seed | 42 (`kBenchSeed`), fully deterministic |

500 balanced synthetic training series with a reasonable hand-engineered feature representation is not a starved-data regime for a 5-class problem with a linear/shallow classifier — consistent with 60.67% being a believable, non-degenerate DIF classification result rather than an artifact of too little data. (It is *not* state-of-the-art ECG classification — a dedicated CNN/RNN on real ECG5000 typically scores >90% — but that's a "needs better feature front-end / real data" ceiling question, not a chance-level bug, and is out of scope per the task's guidance not to chase structural redesigns.)

---

## 5. Final recommendation

- **Status: resolved (stale doc entry), no fix needed.** The 17–20% chance-level figure in `docs/FUTURE.md` §0c does not reproduce on current HEAD; D10 10A now scores 60.67%, ~3× chance.
- **Doc hygiene recommendation:** `docs/FUTURE.md` §0c should be corrected to stop conflating D10's DIF-based ECG classification with the CellAI SSM/CyphaLM stack (they are unrelated subsystems in this domain), following the same pattern already used for the retired "D04 33.2 bpc" bullet in the same section.
- **No further engineering pass is warranted for D10 accuracy right now.** If someone later wants D10 to approach real-ECG-classification-quality accuracy (>90%), that is a genuine, larger-scope future item — e.g. real UCR ECG5000 data (`bench/data/ecg5000/`), a richer feature front-end (e.g. spectrogram/wavelet, or per-class raw-signal templates) or a purpose-built sequence model — not a bug fix, and should be scoped as its own dedicated future pass rather than attempted opportunistically here, consistent with how RPSM/multi-view/D14 investigations handled genuine structural limits today.
- **No code changes were made.** `native/build_d10` was created solely for reproduction/diagnosis and can be deleted or left as a scratch build dir; no CTest regressions are possible since no source was modified. (`native_cyphalm_ssm_diagnose_d10_smoke` was run as a sanity check and passes.)
