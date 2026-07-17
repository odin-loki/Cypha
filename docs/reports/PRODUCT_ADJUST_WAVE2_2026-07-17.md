# Product adjust wave 2 — post-closeout landings (2026-07-17)

**Author:** Odin Loch (agent docs)  
**Scope:** What landed **after** [`PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md`](PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md) (`aad3ce7`) vs what remains blocked on overnight / multi-day research.  
**Repo HEAD @ wave 2:** `0f502e7`  
**Overnight:** H18 @ 21/25 in flight — [`OVERNIGHT_HEALTH_2026-07-17.md`](OVERNIGHT_HEALTH_2026-07-17.md) §8. **Do not finalize or kill.**

---

## Executive summary

**Verdict: bounded product/adjust wave 2 — DONE.**

Wave 2 (post-[closeout](PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md) `aad3ce7`) shipped **read-only overnight health for H18**, **RFF latent exploratory default promotion**, **cell-sweep summary aggregation**, **MC4/MR3 bench metrics**, **Studio Web chat polish**, and **BoW / federated status doc refreshes** — all without touching `build_math`, `build_deff`, `BASELINE_*`, or live overnight processes.

**Remaining work is only:**

1. **Overnight** — H19–H22 cell-sweep completion, then `poll_and_finalize_overnight.ps1 -AutoCommit` + lock commit + d27–d38/d38 certificate ([`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md)).
2. **Multi-day research** — RPSM zero-BPTT gap, hidden=512 @ 300k Phase 3, P3 XOR GMM default-on, EWC/CL, multi-view DIF, κ ablations, federated TLS infra, Qt/Web hardening backlog, paper submit.

No further bounded product/profile adjust items are queued for this wave.

---

## Landed after closeout

### Overnight health (H18)

| Item | Commit | Notes |
|------|--------|-------|
| H18 progress refresh (21/25) | `4a7fb49` | H17 completed; H18 child PID 47108 active; poll `processes=4 lock_n_train=300000` |
| Health doc §8 | `4a7fb49` | ETA for H19–H22 + finalize window documented |

### RFF latent promote

| Item | Commit | Notes |
|------|--------|-------|
| Exploratory default for generalizable `latent` mode | `beacef3` | RFF auto-γ `rff_dim=4096` → **76.3%** (~2.7 pp to sklearn); profile `bench/config/latent_rff_auto_gamma.json` |
| Closeout report | `beacef3` | [`RFF_LATENT_PROMOTE_2026-07-17.md`](RFF_LATENT_PROMOTE_2026-07-17.md) |
| **Unchanged** | — | Production `xor_pair` + Nyström M=512 default for `d03_xor` |

### Cell-sweep summary tool

| Item | Commit | Notes |
|------|--------|-------|
| `scripts/aggregate_cell_sweep_summary.ps1` | `a04af20` | Read-only; writes `bench/results/cell_sweep/summary.csv` with vs_B0/B1/B2 columns |
| Report | `a04af20` | [`CELL_SWEEP_SUMMARY_TOOL_2026-07-17.md`](CELL_SWEEP_SUMMARY_TOOL_2026-07-17.md) |
| **Run after** | — | H22 artifact flush + finalize (partial sweeps omit missing variant rows) |

### Metrics — MC4 / MR3

| ID | Metric | Commit | Report |
|----|--------|--------|--------|
| MC4 | Margin distribution (mean / p50 / p10) | `b61543f` | [`GENERAL_METRICS_MC4_2026-07-17.md`](GENERAL_METRICS_MC4_2026-07-17.md) |
| MR3 | Residual autocorr lag-1 + spectral flatness | `b61543f` | [`GENERAL_METRICS_MR3_2026-07-17.md`](GENERAL_METRICS_MR3_2026-07-17.md) |

Wired in `clf_metrics_native` / regression paths via `bench_metrics.hpp`.

### Web polish

| Item | Commit | Notes |
|------|--------|-------|
| CyphaLM chat empty state + LM readiness label | `436808f` | `#chat-log` placeholder; `#chat-status` from `GET /metrics` → `lm_loaded` |
| Report | `436808f` | [`WEB_UI_POLISH_2026-07-17.md`](WEB_UI_POLISH_2026-07-17.md) |
| BoW §5 mark | `527195d` | Studio Web partial → chat polish noted |

Prior chat pane: `b706647` — [`WEB_UI_GENERATE_2026-07-17.md`](WEB_UI_GENERATE_2026-07-17.md).

### Docs / BoW commits (wave 2)

| Commit | Summary |
|--------|---------|
| `e175da0` | BoW refresh for H18, MC4/MR3, summary tool, RFF promote |
| `527195d` | BoW Studio Web chat polish mark |
| `0f502e7` | Federated TLS status — golden merge blocking; TLS optional without OpenSSL |

### Qt + finalize prep (wave 2 close)

| Item | Notes |
|------|-------|
| Qt compare empty-state / selection hints | [`QT_SHELL_POLISH_2026-07-17.md`](QT_SHELL_POLISH_2026-07-17.md) |
| Post-H22 finalize command + checklist | [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md) |

### Other post-closeout code (same evening, adjacent scope)

| Item | Commit | Notes |
|------|--------|-------|
| MC1 macro-F1 + balanced accuracy | `7a84b4b` / `5562891` | Addendum 2 continuation |
| MC3 FGSM robustness curve | `2f3c6f1` | Adversarial epsilon sweep |
| MG3 needle warm-up lift | `852d6e5` | Recall 0.05→0.35 |
| MS2 algebraic text fingerprint | `0f39fe4` | CyphaLM output fingerprint vector |

---

## Still blocked — overnight

| Blocker | State @ wave 2 | Unblocks |
|---------|----------------|----------|
| **300k cell-sweep H19–H22** | H18 @ 21/25 (`11:45:54Z`); 4 variants remain | H22 child exit + `write_overnight_artifacts` |
| **`poll_and_finalize_overnight.ps1 -AutoCommit`** | Waiting on sweep | After H22 — see [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md) |
| **Baseline lock refresh** | Hand-edit OFF-LIMITS; finalize updates | Successful finalize + commit |
| **d27–d38 production gates** | `pending_production` | Lock lands |
| **d38 certificate (115→116 CTests)** | Blocked on 0.1–0.2 | Lock + finalize chain |
| **Cell-sweep `summary.csv` full matrix** | H15–H22 JSON not flushed yet | Sweep completion → `aggregate_cell_sweep_summary.ps1` |
| **Math-integration production certificate (d53–d58)** | Needs completed 300k math-integration overnight | Same chain |

---

## Still blocked — multi-day research

| Area | Why blocked | Next step |
|------|-------------|-----------|
| **P3 class GMM default-on / XOR ≥75%** | XOR ~51% at FAST latent GMM; RFF closes kernel gap not GMM path | Re-evaluate P3 after RFF latent adoption in research configs |
| **RPSM zero-BPTT training gap** | Cheap hypotheses exhausted (§13–§14) | BPTT in training loop — multi-day |
| **D17 < 2.873 via RPSM** | Not met at any tier tried | Depends on BPTT fix |
| **Hidden=512 @ 300k D_eff (Phase 3)** | Contention with overnight + ~16h wall | Schedule after sweep on uncontended machine — [`HIDDEN_DIM_SCALE_PLAN.md`](HIDDEN_DIM_SCALE_PLAN.md) |
| **EWC / shared-model CL (P5)** | Best 0.135→0.108 @ λ=2.0 | Routing redesign or accept isolation-only |
| **Multi-view CyphaDIF (P4 Step 7)** | D16 16G regression documented | DIF-V3 replay-interleave |
| **κ-targeting / math-integration ablations** | Flat at FAST/5k | Production-scale grid post-overnight |
| **Federated TLS + coordinator HTTP** | Status documented `0f502e7`; TLS optional with OpenSSL | Optional CI enable only |
| **Qt / Web further polish** | Compare hints + chat readiness shipped this wave | Manual hardening backlog only |
| **Paper submit (2027 Q1)** | Results pending production lock | After overnight + lock |

---

## Suggested order (updated)

1. **Wait** for H19–H22 (do not restart, do not kill).
2. **`poll_and_finalize_overnight.ps1 -AutoCommit -BuildDir native/build_math`** — exact command in [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md).
3. **`aggregate_cell_sweep_summary.ps1`** on flushed variant JSON.
4. Merge **d38**; run production validate env hooks.
5. Re-evaluate **P3 XOR GMM** using RFF latent profile (`latent_rff_auto_gamma.json`) — production `xor_pair` default unchanged.
6. Schedule **hidden=512 @ 300k** on uncontended machine.
7. RPSM **BPTT-in-training** research track in parallel with sweep analysis.

---

## Cross-links

- Closeout (wave 1): [`PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md`](PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md)
- Finalize prep: [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md)
- Master task list: [`CYPHA_BILL_OF_WORK.md`](../../CYPHA_BILL_OF_WORK.md)
