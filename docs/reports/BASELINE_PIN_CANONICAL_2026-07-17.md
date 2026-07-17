# D17 hybrid 300k BPC — canonical pin reconciliation (2026-07-17)

**Status:** Documented (Bill of Work §0.5).  
**Do not edit:** `bench/BASELINE_LOCK.json` numeric values (overnight owns refresh).

## Canonical pin

| Field | Value |
|-------|-------|
| **BPC** | **2.873** |
| **Source** | `bench/BASELINE_LOCK.json` → `d17_hybrid_baseline.bpc` |
| **Provenance** | Phase 1c 17A Python reference (`bench/config/d17_phase1c_17a.json`, `cyphalm_bpc: 2.8732`) |
| **Profile** | D17, `hybrid_gria_lstm`, WikiText-2 official train/valid, 300k train / 2000 eval |
| **Validator** | `scripts/validate_baseline_lock.ps1 -Production`, `baseline_lock_validate --production`, d27/d53–d58 gates — all compare native BPC within **±0.05** of this pin |

This is the single authoritative regression reference. Any doc that calls a different number “the” baseline lock without labeling it as historical is wrong.

## Historical / sweep variants (not alternate locks)

Three slightly different 300k hybrid figures had been floating across docs. They are **run artifacts**, not competing lock values:

| BPC | Label | Where it came from |
|-----|-------|-------------------|
| **2.873** | **Canonical pin** | `d17_hybrid_baseline` in `bench/BASELINE_LOCK.json` |
| **2.892** | Native build6 sweep (2026-06-10) | `docs/native/CYPHALM_NATIVE_BENCH_RESULTS.jsonl` — hybrid **2.891897…** rounded; post LSTM in-place fix (`cyphalm_hybrid300k_v2.exe`) |
| **2.897** | Early v2.5 framework release note | `CHANGELOG.md` [2.5.0] — release-time native figure before build6 sweep reconciliation |
| **2.864** | Latest recorded overnight native | `bench/BASELINE_LOCK.json` → `overnight_results.bpc` (production tier, same config envelope) |

**Rule:** cite **2.873** when stating the locked baseline, production gate, or RPSM stop/go target. Cite 2.892 / 2.897 only when describing a specific historical native sweep or release note, explicitly labeled as such.

## Docs updated (2026-07-17)

- `docs/native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md` — was presenting **2.897** as “the” 300k lock
- `docs/native/CYPHALM_NATIVE_UPGRADE_MASTER.md` — was presenting **2.892** as parity target without pin distinction
- `docs/FUTURE.md`, `docs/RESEARCH_STATUS.md`, `paper/CyphaLM_paper.md`, `CHANGELOG.md` — cross-linked or labeled where needed

**Out of scope (other agent):** `CYPHA_BILL_OF_WORK.md`, `CYPHA_OPTIMALITY_PLAN.md` — link this report from those when §0.5 is closed.

## Follow-up: overlapping production gates (Addendum 3)

Four bench domains currently validate overlapping production-lock criteria against the same 300k hybrid pin:

| Domain | Phase | Role |
|--------|-------|------|
| **d53** | 40 | Production preset ship lock |
| **d54** | 41 | Production math certificate (unifies d42+d53 pattern) |
| **d57** | 43 | Production cell sweep math certificate (unifies d56+d54) |
| **d58** | 44 | Unified production overnight math complete (unifies d54+d57) |

**Pending:** maintainer overnight must populate real 300k production-tier results before these gates report `production` (not `pending_production`). **Do not delete** the corresponding CTests or domain runners today — overnight finalize scripts (`scripts/finalize_production_overnight.ps1`, `scripts/cypha_native_validate_all.ps1`) depend on them.

**Consolidation (future):** after overnight fills `bench/BASELINE_LOCK.json` production sections, merge the four gates into one production certificate domain and retire redundant smoke CTests. Track in Bill of Work Addendum 3 / §0.5 follow-on; no code change in this docs-only pass.

## Quick reference

```powershell
# Read canonical pin (do not hand-edit)
Get-Content bench/BASELINE_LOCK.json | ConvertFrom-Json | Select-Object -ExpandProperty d17_hybrid_baseline

# Validate against pin
powershell -File scripts/validate_baseline_lock.ps1 -Production
```
