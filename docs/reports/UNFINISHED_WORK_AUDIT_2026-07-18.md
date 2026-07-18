> **Note (2026-07-18):** Draft research/audit note. Living status is [CONTINUUM_CLOSEOUT_2026-07-18.md](CONTINUUM_CLOSEOUT_2026-07-18.md) + [CYPHA_BILL_OF_WORK.md](../../CYPHA_BILL_OF_WORK.md). Release is **v2.3.24** / CI is **windows_msvc** (not MinGW). Treat numbers below as proposals unless cross-checked against those sources.
# Cypha — Unfinished Work & Polish Audit

**Prepared for:** Odin Loch
**Basis:** Full sweep of `Cypha-main` @ post-v2.3.24 lock (`a552aee`): all 88 reports in `docs/reports/`, the Bill of Work and Optimality Plan status tables, `BACKLOG_EXECUTION_PLAN_2026-07-18.md`, `FUTURE.md`, CHANGELOG, README, CI/release scripts, and a source-level scan for markers (the C++ itself is remarkably clean — essentially zero TODO/FIXME debt; the unfinished work lives almost entirely in *docs, flags, and decisions*).

**Framing:** Nothing here is broken. What's accumulated is three kinds of debt: (1) **documentation drift** — the repo tells four different stories about its own test count and two about its version, which is precisely what a due-diligence reviewer greps for; (2) **decision debt** — seven shipped-but-default-OFF feature systems with no promote-or-remove verdict; (3) **mid-flight research threads** with their next action already written down in your own plans but not executed. Each item below cites where the loose end lives.

---

## 1. Documentation drift & repo hygiene (highest polish-per-hour; directly M&A-relevant)

**1.1 — Test-count contradiction, four ways.** README's headline paragraph says "**53 CTest** cases" with old `*_parity` names; README's own quickstart says "**118** CTests when d40 merged, else 117, else 116"; CHANGELOG's v2.3.24-era entry says **116 total**; `FUTURE.md` says "**160 CTests**". Pick the pattern FUTURE.md already uses — "see `cypha_native_validate_all.ps1` for the authoritative count" — and make every other document defer to it, with the headline stating no number at all.

**1.2 — Phase 0's own acceptance test is failed.** Optimality Plan Phase 0 is marked **Done**, with done-when: *"`grep -ri parity` returns only historical CHANGELOG entries."* It doesn't. README still says "validated by CTest parity fixtures," lists `cypha_parity` / `memory_train_parity` / `quantile_dif_train_parity` in the headline and in the test table (~lines 186–192, referencing tests that were renamed to `*_golden`), `docs/port/PORT_CONTRACT.md` is still titled "the parity contract," `docs/verify/VERIFICATION_STATUS.md` still frames results as "parity results," and the fixtures blurb still says "parity assets." One editing pass closes Phase 0 for real.

**1.3 — Version drift.** README badge and quickstart say "latest **v2.2.8**"; `CMakeLists.txt` is `project(cypha_native VERSION 2.2.8)`; the actual latest tag is **v2.3.24** (RELEASE_V2_3_24 report). CHANGELOG's compare links at the bottom stop at 2.2.8, and the 2.3.x line exists only as milestone bullets without release headers. Bump the CMake project version, fix the README badge/link, add proper `[2.3.24]` (and intermediate) headers + compare links, and add a release-time check that greps these three locations for the tag being published.

**1.4 — Mojibake / encoding corruption.** `docs/FUTURE.md` has **58** replacement characters (§ symbols mangled to `�` — "�0 �", "Nystr�m", "median-?"), and `CONTINUUM_CLOSEOUT_2026-07-18.md` has `?` where em-dashes/±/→ should be. These files were saved through a wrong codepage at some point. Sweep: a small script that flags any tracked `.md` containing U+FFFD or suspicious `?`-runs, fix the two known files from context, add the check to CI. Cheap, and corrupted internal docs read badly in diligence.

**1.5 — Line-ending inconsistency in `native/src`.** Some sources are CRLF (`infer_cpu.cpp`, headers under `include/cypha/`), others LF (`accel_backend.cpp`, `rff_features.cpp`). `.gitattributes` exists — verify it actually normalizes `*.cpp`/`*.hpp` and run one `git add --renormalize .` pass so diffs stop being polluted.

**1.6 — Broken relative links out of the repo.** README links twice to `../Compression Algorithms/NMP_neural_compression_research_paper.md` — a path *outside* the repository, dead on GitHub. Either vendor the paper (or an abstract stub) under `docs/research/`, or make it an absolute link to wherever it publicly lives.

**1.7 — Report sprawl (88 files) with no index of record.** `BASELINE_PIN_CANONICAL` exists precisely because stale numbers kept leaking between reports, and the same risk now applies to statuses (e.g., "d53–d58 pending_production" in the pin report vs "production" in POST_LOCK_STATUS after `a552aee`). Two fixes: (a) a generated `docs/reports/INDEX.md` (title, date, one-line verdict, superseded-by column); (b) a single hand-maintained `STATUS.md` dashboard — current pin, current release, gate states, open decisions — that every report links to instead of restating numbers. Your BoW disclaims being "a live status dashboard"; nothing currently is one.

**1.8 — Vestigial Python-era artifacts.** `bessel_ratios.npz` sits at repo root and is featured in README's source-documents table, but the native runtime uses the compiled `bessel_table_data.cpp` — post-P7 the npz is (at most) a regeneration input; move it under `docs/research/` or `native/tools/data/` with a one-line provenance note, and de-feature it from the README table. Same pass: confirm every one of the ~40 `fixtures/` dirs still has a consuming `*_golden` test (the parity retirement makes orphaned fixture dirs likely).

**1.9 — SECURITY.md is 1.2 KB.** For a product aimed at defence-adjacent buyers this is the first file some reviewers open. Expand to: supported versions table, disclosure contact + expected response window, scope (REST server, Qt shell, file-format parsing — `.cypha` loading is an attack surface worth one sentence), and a note on the SQLite amalgamation / cpp-httplib vendored-dependency update policy.

---

## 2. Release & CI loose ends

**2.1 — `gh auth` is still broken (stored PAT missing `read:org`)**, and the Actions release path has a documented workaround because `softprops/action-gh-release` can't update a PAT-created release ("Resource not accessible by integration" — RELEASE_V2_3_24). The pipeline works but is brittle and half-manual. Fix the PAT scopes once, then decide one canonical publish path (CLI `publish_release.ps1` *or* Actions, not both) and delete the workaround notes from the living docs into an archive.

**2.2 — MinGW→MSVC packaging transition is half-reflected in docs.** Packaging retired MinGW for MSVC (RELEASE_V2_3_24: "MinGW retired"), but README/quickstart and most build reports remain MinGW-centric, and the CUDA blocker analysis was written against the MinGW-only machine state. Now that MSVC exists in the packaging path, two things unblock: update NATIVE_QUICKSTART/README to present MSVC as the primary Windows toolchain, and revisit the CUDA-on-Windows blocker (the perf roadmap's §5.2) which may simply be gone.

**2.3 — Federated TLS smoke is skip-by-default everywhere.** OpenSSL is present on your host, the smoke passes when enabled, but no CI leg builds `-DCYPHA_ENABLE_OPENSSL=ON` — so TLS regressions are invisible until someone manually runs it. Add one optional Linux CI job with OpenSSL; it's a `ci.yml` matrix entry.

**2.4 — Production-gate consolidation is explicitly deferred.** Four overlapping domains (d53/d54/d57/d58) validate the same 300k pin; the pin report's own follow-up says: after overnight fills production sections, "merge the four gates into one production certificate domain and retire redundant smoke CTests." The lock is now production (`a552aee`, d38 `overnight_certificate_ready`) — the precondition is met; the consolidation is now actionable. Also confirm A1's deliverable (`MATH_INTEGRATION_PRODUCTION_2026-07-18.md` + the BoW §0-bis row update) actually landed; I found the plan for it, not the report itself.

**2.5 — `release.yml` is 27 KB.** After 2.1/2.2 settle, an audit pass on that workflow (dead steps from the MinGW era, the PAT workaround) will likely halve it.

---

## 3. Decision debt: shipped-but-OFF machinery (promote, or REJECT-and-prune)

This is the biggest *structural* unfinished-work category. Seven systems are fully built, tested, documented — and default OFF with no final verdict. Each one is ongoing maintenance surface (format keys, dead-work audits, golden interactions) for zero default-path benefit. Every entry below needs one of three explicit outcomes: **promote** (with golden regen), **keep as documented product option** (opt-in is the *intended* end state — say so), or **REJECT and remove** (archive the report, delete the code path, shrink the format).

| System | Where it stands | The missing decision |
|---|---|---|
| **P3 per-class GMM** | Opt-in; XOR gate (≥75%) failed at ~50%; your own promotion path (batch-EM warm-start from replay) is written in the Phase-3 report and Backlog C2 | Run C2 once with the warm-start; promote or REJECT. Note my quality roadmap's view: the *encoder* is the wall — consider gating this decision on the nonlinear-encoder experiment so GMM isn't rejected for an upstream deficit twice |
| **P4 analytic NIG BMA** | Opt-in; judged (implicitly) on accuracy | Re-judge on ECE/Brier/NLL — BMA's expected win is calibration. If it doesn't move those either, REJECT |
| **P6 variational IB encoder** | Opt-in, fixed β | One β-annealing run (deterministic warmup) before verdict; fixed-β evaluations of VIB are known-unfavourable |
| **P7 score-matching path** | Opt-in; Bessel LUT kept — **two parallel implementations of the GH posterior** now live forever | Decide the LUT is canonical and demote score-match to a research branch, or the reverse. Carrying both indefinitely is the worst option |
| **P5 leverage-Nyström + SORF** | Opt-in | Fold into the "kernel features default-on" decision (quality roadmap §4.1a) — one bench pass decides all three feature paths together |
| **SOM/GNG/GRIA-controller/Hebbian/temporal-SOM** (`native/src/som/`, 6 upgrades) | All flags OFF since 2026-05-26; report verdicts were REVERT/KEEP-as-neutral at 3 seeds on one synthetic task; native code retained "for CyphaLM config parity only" | Either archive the native SOM tree behind a build flag defaulting OFF-and-excluded, or schedule one honest re-evaluation (≥5 seeds, real datasets) — the current state is code kept alive by an archival Python-era report |
| **Math-integration preset** | Helpful @5k (−0.117), harmful @300k (+0.209); "recipe redesign only"; eigenvalue `D_eff` joint-fail, do-not-promote — but the estimator code remains | The scale-dependent sign flip is a genuinely interesting open research question (§4.4 below); the *eigenvalue estimator*, however, has a final verdict — remove it or fence it as research-only so it can't be re-enabled by config accident |

**GGUF export** belongs here too: honestly labelled *partial* (structural GGUF v3, `cypha-dif` arch tag, not llama.cpp-loadable). Decide the end state: (a) it's a container format for *your* loaders — then write the loader-side doc and call it done; (b) it should interop with llama.cpp tooling — then that's a real work item; (c) it was an experiment — archive. Right now it's a shipped tool whose purpose statement is a caveat.

---

## 4. Open research threads with their next action already written

These are mid-flight per your own Backlog (2026-07-18) and BoW — listed with the exact next step so nothing needs re-derivation:

**4.1 — D16 continual-learning fork (Backlog B2–B4) — the biggest open product decision.** EWC D16B improved 0.135→0.108 forgetting @ λ=2.0 but "shared-model CL still open." The B4 decision doc — **ship isolation-only** vs **EWC+replay overlay** vs **routing redesign spike** — is not yet written, and B2 (the 16I replay-interleave experiment, RR vs replay_ratio ∈ {0.22, 0.5}) is its stated input. This choice defines what the product *claims* about continual learning, which feeds the pitch directly. Run B2 (FAST gate, no overnight needed), write B4.

**4.2 — RPSM disposition after the failed capacity gate.** Small-tier gate ran and **failed the stop rule** (gap widened; "architectural ceiling stands"; BPTT-depth already falsified §14). Properly closed as research — but the consequence is undrawn: `rpsm_sequence_layer.cpp` (933 lines) plus the Small-tier plumbing now exist with no promotion path, while `rpsm_score_matrix_batched` remains the *default* DIF batch-LLR engine. Write the one-page disposition: RPSM = production LLR kernel, sequence-model ambitions retired (Izaac-VRF / GMM-world-model follow-ons explicitly cancelled per the C1 stop), Small-tier knobs kept or removed. Otherwise the next agent session re-opens it.

**4.3 — Overnight sweep stragglers.** Optimality Plan header still lists "overnight H16/19/25" open; H15's post-NaN-fix 300k re-row produced 5.262 BPC (recorded, not a promote — effectively closed). Either schedule the three re-rows on the next overnight or mark them SKIP-with-reason in the sweep aggregate so the 25/25 table stops having asterisks.

**4.4 — The math-integration scale-dependent sign flip.** Characterized (worse @500/2k, better @5k/20k, worse @300k) and parked as "recipe redesign only." This is the one parked item I'd argue is *underrated*: a regularizer that helps at small n and hurts at large n is behaving exactly like an informative prior being outweighed by data — which suggests an n-dependent schedule (anneal κ-targeting/`D_eff` weight ∝ 1/n) rather than a redesign. One sweep with annealed weight would settle whether the whole math-integration program has a production role or is research-only.

**4.5 — Paper (Phase F).** Figures regenerated from native sources, 14 references, bundle + attestation built — genuinely close. Remaining: the narrative pass for the 300k trade-off and D16F isolation caveat (Phase F text), then two human-gated steps: arXiv upload from `paper/arxiv_bundle/`, venue submit 2027 Q1. Suggest doing the narrative pass *after* B4 (4.1) so the continual-learning claims in the paper match the product decision.

**4.6 — GPU Phase D.** D1 (document infer-only truth) done; **D2 — wire `accel::batch_encode` into one real offline/bulk path (tool or REST bulk, n>1) — is the open chunk**, explicitly scoped as PR-sized. With MSVC now in the toolchain (2.2), the optional local CUDA smoke stops being blocked.

---

## 5. Product-surface polish (bounded, visible)

**5.1 — Qt hardening checklist has never been executed.** `QT_HARDENING_CHECKLIST_2026-07-18.md` exists with an empty "Pass?" column on every row. One manual session fills it; any crash found is exactly the kind of thing you want found before a demo to Quinn's people rather than during.

**5.2 — Studio Web UI.** The 2026-07-17 polish waves were deliberately bounded slices (chat empty state, generate flow) with the BoW §5 Web UI still marked Partial. Worth one deliberate pass with a product eye: error surfaces on REST failures, model-load feedback, and making the `/intelligence` monitor visible in the UI (it's a differentiator that currently only exists as a REST endpoint).

**5.3 — Demo path.** `demo_cypha_capabilities.ps1` exists — verify it runs clean against a release install (not a dev build dir) on both platforms; that script is plausibly the first thing a technical evaluator runs. Same for `loadtest_ab_predict_example.*` — after the global-lock fix from the perf roadmap, re-run and quote the numbers in the product docs.

**5.4 — Installer/packaging niceties.** AppImage icon dependency (`packaging/cypha.png` — present, good); add an uninstall note to `install_release_windows.ps1`; verify the Windows zip's MSVC runtime story (static vs redist) is stated in packaging/README.

**5.5 — README restructure.** Beyond the fixes in §1: the headline paragraph is a 250-word single bold block — split into a 3-sentence claim + a "validated by" line that defers counts to the script (1.1). Given the Quinn pitch, consider a short "Evaluation" section pointing at the strong-baseline table once the quality roadmap's §5.5 exists.

---

## 6. Suggested closure order

| # | Item | Effort | Why this order |
|---|------|--------|----------------|
| 1 | §1.1–1.6 doc drift sweep (counts, parity language, version, mojibake, EOL, dead links) | half a day | Highest polish-per-hour; everything else reads better after |
| 2 | §2.1–2.2 PAT scopes + one canonical publish path + MSVC-first docs | half a day | Un-bricks releases; unlocks CUDA revisit |
| 3 | §4.1 run B2, write B4 decision doc | 1–2 days | Biggest open product decision; feeds paper + pitch |
| 4 | §3 decision-debt table: one verdict per row (P3/P4/P6/P7/P5/SOM/GGUF) | 2–3 days of gated runs | Converts seven ambiguities into a smaller, honest surface |
| 5 | §4.2 RPSM disposition + §4.3 sweep stragglers + §2.4 gate consolidation | 1 day | Closes the "open" column of the Optimality Plan entirely |
| 6 | §5.1–5.3 Qt QA pass + Web UI slice + demo verification | 1–2 days | Demo-readiness for the M&A conversation |
| 7 | §1.7 STATUS.md dashboard + reports index | half a day | Prevents the drift from §1 recurring |
| 8 | §4.4 annealed math-integration sweep; §4.5 paper narrative pass | as scheduled | Research + calendar-gated |

Total to a genuinely closed state on everything except the research bets: roughly **one focused week**. The repo's engineering discipline is unusually high — the loose ends are almost all *verdicts not yet written down*, and the writing is the cheap part.

