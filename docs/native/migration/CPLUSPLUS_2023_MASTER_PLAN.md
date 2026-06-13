# C++2023 migration — Python decommission master plan

**Goal:** Cypha runtime = **native C++ only** (C++20/23). Python retained only for fixture generation and one-off research until parity is frozen.

**Current baseline (v2.2.8):** 52 CTest, ~274 pytest, four blocking CI jobs, GitHub Release installers.

---

## Phase map

| Phase | Scope | Status |
|-------|-------|--------|
| **P0** | Core DIF infer/train, `.cypha` v3, REST, Qt shell, bench d01–d17 | ✅ SHIPPED |
| **P1** | Intelligence Profiler (`cypha/intelligence/`) | 🔄 **Started** — smoke + CTest |
| **P2** | XOR / kernel LLR end-to-end (train + batch + persist) | 🔄 score_matrix blend + benchmark script |
| **P3** | Archive failed experiments (`cypha_som` → docs/archive) | ✅ Documented |
| **P4** | CyphaLM native-only inference path (drop Python `cypha_lm/` at runtime) | Partial — `cyphalm_bench_native` |
| **P5** | Studio: Qt shell replaces PySide6 + FastAPI reference | Partial — `cypha_qt_shell` |
| **P6** | Bench/tune/diagnostics: native-only (`cypha_bench_run`, `cypha_tune_run`) | ✅ Native runners exist |
| **P7** | Remove `Cypha.py` from CI gate; pytest → CTest-only | Planned |
| **P8** | Bump `CMAKE_CXX_STANDARD` to 20 (then 23 when toolchains stable) | Planned |

---

## Python packages — decommission order

| Package | Native mirror | Decommission when |
|---------|---------------|-------------------|
| `cypha_som/` | `native/src/som/` (smoke only) | **Now** — flags OFF, archived |
| `cypha_accel/` | `native/src/accel_cuda.cu` | After CUDA bench parity |
| `cypha_views/` | — | Port or drop |
| `cypha_diagnostics/` | `cypha_diagnostics_run` | After Phase 4 inline checks |
| `cypha_bench/` | `cypha_bench_native` | After report PNG parity lock |
| `cypha_lm/` | `cypha_lm_native` | After D17 300k parity lock |
| `cypha_studio/` | `cypha_qt_shell` + `cypha_rest` | After GUI feature parity |
| `Cypha.py` | `cypha_core` | Last — golden reference until P7 |

---

## C++ standard

- **Today:** C++17 (`native/CMakeLists.txt`) for MSVC/GCC/MinGW/CUDA compatibility.
- **Target:** C++20 modules optional; C++23 when GitHub Actions MSVC + GCC both green with `-std=c++23`.

---

## Validation gate (unchanged)

```powershell
powershell -File scripts\cypha_native_validate_all.ps1
```

Add after P1/P2:
```powershell
ctest -R native_intelligence_profiler_smoke
python scripts/benchmark_xor_kernel_llr.py -o artifacts/profiles/xor_kernel_llr.json
```

---

## Intelligence Stats → C++

Papers live in [`docs/research/intelligence_stats/`](../../research/intelligence_stats/README.md).

Priority implementation order:
1. Profiler + measurers (P1) ✅
2. Test bench domain `d_profile` / landscape CSV export
3. Self-correcting loop (Paper IV)
4. Soft world (Paper V)
