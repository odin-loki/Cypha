# Product adjust STOP — free polish waves (2026-07-17)

**Author:** Odin Loch (agent audit)  
**Scope:** Read-only audit of bounded product/profile adjust work after [wave 2](PRODUCT_ADJUST_WAVE2_2026-07-17.md). Classify all remaining items as overnight-blocked or multi-day research.  
**Repo HEAD @ audit:** `aa396cc`  
**Overnight:** H18 @ 21/25 in flight — [`OVERNIGHT_HEALTH_2026-07-17.md`](OVERNIGHT_HEALTH_2026-07-17.md) §8. **Do not finalize or kill.**

---

## Verdict

**STOP / DONE for free polish waves.**

No bounded product/profile adjust items remain that are **not** overnight-blocked and **not** multi-day research. Waves 1 ([closeout](PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md)) and 2 ([wave 2](PRODUCT_ADJUST_WAVE2_2026-07-17.md)) shipped the full Addendum-2 metrics set, Web/Qt polish, RFF latent promote, cell-sweep summary tool, federated TLS status, finalize prep, and D17 perf floor documentation.

**Do not invent more micro-opts until H22 finalize completes.** Further D17 train gains require structural/CUDA paths (compute-bound floor at ~252 chars/sec MSVC). Post-H22 work is finalize → lock → production gates, not profile polish.

---

## Audit checklist (all shipped or blocked)

| Area | Status | Evidence |
|------|--------|----------|
| **Addendum 2 MC1–MC5** | [x] Shipped | MC1 `7a84b4b`; MC2/MS1 closeout; MC3 `2f3c6f1`; MC4 `b61543f`; MC5/MG5 [`SAMPLE_EFFICIENCY_CURVE_2026-07-17.md`](SAMPLE_EFFICIENCY_CURVE_2026-07-17.md) |
| **Addendum 2 MR1–MR3** | [x] Shipped | MR1/MR2 closeout; MR3 `b61543f` — wired in `bench_metrics.hpp` / `bench_domains.cpp` |
| **Addendum 2 MS1–MS2** | [x] Shipped | MS1 closeout; MS2 `0f39fe4` — [`GENERAL_METRICS_MS2_2026-07-17.md`](GENERAL_METRICS_MS2_2026-07-17.md) |
| **Addendum 2 MG3–MG5** | [x] Shipped | MG3 + warm-up `852d6e5`; MG4 closeout; MG5 = MC5 curve format |
| **Web polish** | [x] Shipped | Chat empty state + LM readiness `436808f` — [`WEB_UI_POLISH_2026-07-17.md`](WEB_UI_POLISH_2026-07-17.md) |
| **Qt polish** | [x] Shipped | Compare stats `7578f13`; shell hints — [`QT_SHELL_POLISH_2026-07-17.md`](QT_SHELL_POLISH_2026-07-17.md) |
| **RFF latent promote** | [x] Shipped | Exploratory default `beacef3`; prod `xor_pair` unchanged — [`RFF_LATENT_PROMOTE_2026-07-17.md`](RFF_LATENT_PROMOTE_2026-07-17.md) |
| **Cell-sweep summary tool** | [x] Script shipped | `aggregate_cell_sweep_summary.ps1` `a04af20`; **run blocked** until H22 flush |
| **Federated TLS status** | [x] Documented | Golden merge blocking; TLS optional — [`FEDERATED_TLS_STATUS_2026-07-17.md`](FEDERATED_TLS_STATUS_2026-07-17.md) |
| **Finalize prep** | [x] Documented | Exact post-H22 command — [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md) |
| **D17 perf floor** | [x] Reached | Parts 1–7; compute-bound — [`PERFORMANCE_PROFILE_2026-07-12.md`](PERFORMANCE_PROFILE_2026-07-12.md), [`DEAD_WORK_AUDIT_2026-07-17.md`](DEAD_WORK_AUDIT_2026-07-17.md) |

**Free bounded items found:** none.

---

## Remaining — overnight-only

| Item | State @ stop audit | Unblocks |
|------|-------------------|----------|
| **300k cell-sweep H19–H22** | H18 @ 21/25; 4 variants remain | H22 child exit + `write_overnight_artifacts` |
| **`poll_and_finalize_overnight.ps1 -AutoCommit`** | Waiting on sweep | After H22 — [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md) |
| **Baseline lock refresh** | Hand-edit OFF-LIMITS | Successful finalize + commit |
| **d27–d38 production gates** | `pending_production` | Lock lands |
| **d38 certificate (115→116 CTests)** | Blocked on 0.1–0.2 | Lock + finalize chain |
| **Cell-sweep `summary.csv` full matrix** | H15–H22 JSON not flushed | Sweep completion → `aggregate_cell_sweep_summary.ps1` |
| **Math-integration production certificate (d53–d58)** | Needs completed 300k math-integration overnight | Same chain |

---

## Remaining — multi-day research

| Area | Why blocked | Next step |
|------|-------------|-----------|
| **P3 class GMM default-on / XOR ≥75%** | XOR ~51% at FAST latent GMM; RFF closes kernel gap not GMM path | Re-evaluate after RFF latent adoption in research configs |
| **RPSM zero-BPTT training gap** | Cheap hypotheses exhausted (§13–§14) | BPTT in training loop |
| **D17 < 2.873 via RPSM** | Not met at any tier tried | Depends on BPTT fix |
| **Hidden=512 @ 300k D_eff (Phase 3)** | Contention with overnight + ~16h wall | Schedule after sweep — [`HIDDEN_DIM_SCALE_PLAN.md`](HIDDEN_DIM_SCALE_PLAN.md) |
| **EWC / shared-model CL (P5)** | Best 0.135→0.108 @ λ=2.0 | Routing redesign or accept isolation-only |
| **Multi-view CyphaDIF (P4 Step 7)** | D16 16G regression documented | DIF-V3 replay-interleave |
| **κ-targeting / math-integration ablations** | Flat at FAST/5k | Production-scale grid post-overnight |
| **Federated TLS + coordinator HTTP** | TLS optional without OpenSSL build | Optional CI enable only |
| **Qt / Web further hardening** | Compare hints + chat readiness shipped | Manual backlog only |
| **Paper submit (2027 Q1)** | Results pending production lock | After overnight + lock |
| **D17 structural perf (CUDA / fused kernels)** | Micro-opt floor reached | New architecture path, not profile polish |

---

## Suggested order (unchanged)

1. **Wait** for H19–H22 (do not restart, do not kill).
2. **`poll_and_finalize_overnight.ps1 -AutoCommit -BuildDir native/build_math`** — [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md).
3. **`aggregate_cell_sweep_summary.ps1`** on flushed variant JSON.
4. Merge **d38**; run production validate env hooks.
5. Re-evaluate **P3 XOR GMM** with RFF latent profile (`latent_rff_auto_gamma.json`).
6. Schedule **hidden=512 @ 300k** on uncontended machine.
7. RPSM **BPTT-in-training** research track in parallel with sweep analysis.

---

## Cross-links

- Wave 1 closeout: [`PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md`](PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md)
- Wave 2 landings: [`PRODUCT_ADJUST_WAVE2_2026-07-17.md`](PRODUCT_ADJUST_WAVE2_2026-07-17.md)
- Finalize prep (post-H22): [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md)
- Overnight health: [`OVERNIGHT_HEALTH_2026-07-17.md`](OVERNIGHT_HEALTH_2026-07-17.md)
- Master task list: [`CYPHA_BILL_OF_WORK.md`](../../CYPHA_BILL_OF_WORK.md)
