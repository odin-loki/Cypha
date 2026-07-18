# Post-lock status — 2026-07-18

**Lock commit:** `a552aee` — *bench: lock production overnight results (n_train=300000)*  
**Report time:** 2026-07-18 (UTC+10)

---

## 1. Lock snapshot (`bench/BASELINE_LOCK.json`)

| Field | Value |
|-------|-------|
| `overnight_results.n_train` | **300000** |
| `overnight_results.status` | **production** |
| `overnight_results.bpc` | **2.864125** (~2.864; within production pin tolerance vs canonical **2.873**) |
| `overnight_results.run_at` | 2026-07-12T01:52:47Z |
| `cell_sweep_results.status` | **production** (25 variants) |
| `d17_hybrid_baseline.bpc` | **2.873** (canonical pin) |

---

## 2. Production validation

**Command:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate_baseline_lock.ps1 -Production
```

**Verdict:** **PASS**

```
  (-Production: n_train=300000 status=production bpc pin OK)
validate_baseline_lock: OK bench\BASELINE_LOCK.json
```

Optional extended hook (not run here): `CYPHA_VALIDATE_PRODUCTION=1` on `scripts/cypha_native_validate_all.ps1`.

---

## 3. d38 — “Merge d38” (BoW §0.4)

**What it means:** Ship the **Phase 24 production overnight completion certificate** — bench domain **d38** validates full 300k overnight + RPSM + cell-sweep alignment and **≥ 28** variants. Adds CTest **`native_d38_overnight_certificate_smoke`** (gate count **115 → 116**).

**Current repo state (grep / BoW):**

| Item | Status |
|------|--------|
| Domain runner `run_d38_overnight_certificate_validation` | Present (`bench_domains.cpp`) |
| Profile `bench/config/d38_overnight_certificate_profile.json` | Present |
| CTest `native_d38_overnight_certificate_smoke` | Present (`native/CMakeLists.txt`) |
| Validate hook `CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE=1` | Present (`cypha_native_validate_all.ps1`) |
| BoW §0.4 | **Open** — run d38 smoke + production validate env hooks now that lock landed |

**d27–d38 `pending_production` gates:** Pre-lock, domains reported `pending_production` when `overnight_results.n_train < 300000`. With lock at **300k / production**, re-running d27–d38 (or full validate-all) should promote to production-tier pass / `overnight_certificate_ready` (d38) rather than `pending_production`.

**Next step:** After §0.2 finalize chain confirmed, run `ctest -R native_d38` (or `cypha_bench_run --domain-tag d38`) and mark BoW §0.4 done when smoke passes at production tier.

---

## 4. GitHub auth / release

```text
gh auth status → NOT logged in
```

**Release blocked** on **`gh auth login`** before `scripts/publish_release.ps1` (BoW §0.3). No login attempted in this pass.

---

## 5. Maintainer checklist (post-`a552aee`)

| # | Task | Status |
|---|------|--------|
| 0.1 | 300k production overnight | **Done** — lock `a552aee` |
| 0.2 | `poll_and_finalize_overnight.ps1 -AutoCommit` | **Done** (finalize exit=0 per [`OVERNIGHT_COMPLETE_2026-07-18.md`](OVERNIGHT_COMPLETE_2026-07-18.md)) |
| 0.3 | `gh auth login` + `publish_release.ps1` | **Blocked** — gh not authenticated |
| 0.4 | Merge / validate **d38** | **Next** — domain + CTest in tree; run production hooks |

---

## 6. References

- [`CYPHA_BILL_OF_WORK.md`](../../CYPHA_BILL_OF_WORK.md) §0.2–0.4
- [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md)
- [`INTELLIGENCE_STATS_IMPLEMENTATION.md`](INTELLIGENCE_STATS_IMPLEMENTATION.md) — d38 / Phase 24
