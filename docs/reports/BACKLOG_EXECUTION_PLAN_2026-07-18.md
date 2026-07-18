# Backlog execution plan — 2026-07-18

**Goal:** Close every open BoW backlog item that is actionable without waiting on interactive `gh auth` or inventing free product-polish waves.
**Anchor:** Overnight lock `a552aee`, hybrid pin **2.873**, d38 `overnight_certificate_ready`, free polish STOP’d.
**Rule:** Prefer FAST/CPU gates before any 300k overnight. Do not re-open RPSM BPTT depth (falsified §14). Do not wire `task_block_shuffle` into production profiles.

---

## Track map

| Phase | Track | Acceptance | Overnight? |
|-------|-------|------------|------------|
| **A** | Closeout hygiene | Docs + FAST repro + paper/Qt scaffolding | No |
| **B** | D16 DIF-V3 + EWC decision | 16I experiment + λ grid + written fork | No (non-FAST optional) |
| **C** | Capacity / GMM research | RPSM Small-tier 5k gate; P3 GMM XOR ≥75% or REJECT | No for gate |
| **D** | GPU training gap | Bulk CUDA encode path wired + documented | Optional profile |
| **E** | Release | Dry-run green; live publish needs human `gh auth` | No |
| **F** | Paper submit (2027 Q1) | Native figures + real refs; submit later | No |

---

## Phase A — Hygiene (day 1)

### A1. Math integration production status
- Re-run d53–d58 validate hooks against lock `a552aee`.
- Write `MATH_INTEGRATION_PRODUCTION_2026-07-18.md`: 300k BPC trade-off (κ↑, BPC +0.21), flat-ablation / sign-flip / κ-transfer still open as research.
- **Done when:** BoW §0-bis “pending_production” row updated with real gate statuses.

### A2. Cell-sweep H15 `bpc:null`
- FAST repro `H15` / axiom activation NaN path.
- Fix numerics if confirmed; regenerate `variant_H15.json` at FAST; note 300k re-run as optional follow-up.
- **Done when:** aggregator includes H15 or documents permanent SKIP with reason.

### A3. Paper readiness chunk
- Regenerate 3 native figure payloads from lock / `BASELINE_REPORT` (D17 BPC, α spectrum, D16 forgetting).
- Expand References from named citations in prose.
- **Done when:** `paper/figures/README.md` lists native sources; submit still deferred to 2027 Q1.

### A4. Qt hardening chunk
- Manual QA checklist under `docs/reports/QT_HARDENING_CHECKLIST_2026-07-18.md`.
- Experiment compare → CSV/JSON export (bounded feature).
- **Done when:** checklist + export path exist; packaging smoke optional.

### A5. Release dry-run
- `verify_release_publish.ps1 -AllowPending` (no `gh`).
- Leave live `publish_release.ps1` for human after `gh auth login`.

---

## Phase B — Continual learning / multi-view (days 1–3)

### B1. Close index-reorder for DIF
- Document Phase 2.2 **FAILED** for LM-style reorder (D03 pilot + 16G).
- Keep 16G as harness negative control only.

### B2. DIF-V3 = 16I replay-interleave
- New experiment in `bench_domains.cpp`: RR vs RR+`replay_ratio`∈{0.22,0.5} vs replay bursts after task A.
- FAST gate: forgetting reduction ≥5pp **or** mean-acc hold within 2pp of RR.

### B3. DIF-V2 curriculum + DIF-V1 class-block (D03)
- Measure `CYPHA_CURRICULUM_WINDOW=8`.
- Add opt-in class-block schedule; ≥1pp accuracy or close track.

### B4. EWC extension + P5 fork
- λ grid {0.5,1,2,5,10} × optional real Fisher × best replay from B2.
- Decision doc: **ship D16F isolation-only** vs **EWC+replay overlay** vs **routing redesign spike**.
- Default product recommendation unless ≤0.08 forgetting with B/C acc held: **isolation-only**.

---

## Phase C — Research capacity (days 2–5)

### C1. RPSM Small-tier plumbing (not more BPTT)
- Expose Small-tier + memory knobs; remove Tiny hardcode in `apply_bench_mode(Rpsm)`.
- 5k / 50k gate vs hybrid. Stop if gap does not shrink ≥10% relative.
- Deprioritize Izaac VRF / GMM world-model until C1 passes.

### C2. Optimality P3 GMM XOR
- Batch EM warm-start per class; target `xor_linear_mean_on ≥ 0.75`.
- Keep default OFF until gate passes; else document REJECT and keep RFF path.

### C3. Math scale-law (optional overnight)
- Sweep n_train ∈ {500, 5k, 50k, 300k} for math-integration preset.
- Only after A1 doc lands; do not retune lock casually.

---

## Phase D — GPU training gap (days 3–6)

1. Document current infer-only CUDA truth in VERIFICATION_STATUS.
2. Wire `cypha::accel::batch_encode` into one offline/bulk path (`n>1`).
3. Explicit non-goal this wave: full online `train_step` CUDA (n=1 overhead).
4. Optional: local CUDA smoke with `-DCYPHA_ENABLE_CUDA=ON`.

---

## Phase E — Release (human-gated)

1. Agent: dry-run + notes generation.
2. Human: `gh auth login`.
3. Human/agent: `publish_release.ps1 -Tag <tag> -Draft` then promote.

---

## Phase F — Paper submit (async → 2027 Q1)

After A3: narrative pass for math-integration 300k trade-off, D16F isolation caveat, figure polish. Submission is calendar-gated, not code-gated.

---

## Explicit non-goals this wave

- Free D17 train/infer micro-opt polish (STOP).
- Re-enabling `bptt_window>1` as default (negative @ 5k).
- Promoting `task_block_shuffle` to everyday/REST.
- Hand-editing `BASELINE_LOCK.json`.
- Auto-push / force-publish without auth.

---

## Execution order (follow this)

```
A1 → A2 → A5 (parallel OK with A3/A4)
     ↓
B1 → B2 → B3 → B4
     ↓
C1 → C2  (C3 overnight only if scheduled)
     ↓
D1 → D2
     ↓
E (blocked on gh) + F (async)
```

## Status log

| Item | Status | Evidence |
|------|--------|----------|
| Plan authored | DONE | this file |
| A1 math production | DONE | `MATH_INTEGRATION_PRODUCTION_2026-07-18.md`; d53 `preset_ship_production_wiring_ready`, `lock_joint_ok=false` |
| A2 H15 | DONE | axiom control-gate clamp; FAST BPC **4.51** finite — `H15_AXIOM_NAN_FIX_2026-07-18.md` |
| A3 paper figures | DONE | `paper/figures/native_fig_*.json` |
| A4 Qt | DONE | export CSV/JSON + `QT_HARDENING_CHECKLIST_2026-07-18.md` |
| A5 release dry-run | PARTIAL | needs `native/build` tree; still blocked on `gh auth` for live publish |
| B1–B4 | DONE | `P4_P5_CONTINUAL_LEARNING_DECISION_2026-07-18.md` — isolation product; curriculum opt-in win; 16I/class-block fail |
| C1 RPSM Small | DONE — STOP | Tiny 4.748 vs Small worse — `RPSM_SMALL_TIER_GATE_2026-07-18.md` |
| C2 GMM warm-start | DONE — REJECT | warm=0.505 — `OPTIMALITY_P3_GMM_WARMSTART_2026-07-18.md` |
| D GPU gap | DOCUMENTED | `GPU_TRAINING_GAP_2026-07-18.md` (bulk encode PR next) |
| E publish | blocked | `gh` not logged in |
| F submit | deferred | 2027 Q1; native figure JSONs ready |
