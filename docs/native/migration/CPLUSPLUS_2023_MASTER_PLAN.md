# C++2023 migration — Python decommission master plan



**Goal:** Cypha runtime = **native C++ only** (C++20/23). Python packages removed from the product path.



**Current baseline:** ~**160 CTest**, two blocking CI jobs (`build_and_test`, `windows_msvc`), GitHub Release installers (**v2.3.24**). See **`scripts/cypha_native_validate_all.ps1`** for authoritative tally.



---



## Phase map



| Phase | Scope | Status |

|-------|-------|--------|

| **P0** | Core DIF infer/train, `.cypha` v3, REST, Qt shell, bench d01–d17 | ✅ SHIPPED |

| **P1** | Intelligence Profiler (`cypha/intelligence/`) | 🔄 **Started** — smoke + CTest |

| **P2** | XOR / kernel LLR end-to-end (train + batch + persist + bench d03_xor) | ✅ Nyström + train wire + `xor_kernel_bench` + CTests + `cypha_bench_run --domain-tag d03_xor` + validate smoke |

| **P3** | Archive failed experiments (`cypha_som` removed) | ✅ Done |

| **P4** | CyphaLM native-only inference path | ✅ **`cyphalm_bench_native`**, REST `/generate` |

| **P5** | Studio: Qt shell replaces PySide6 + FastAPI reference | ✅ **`cypha_qt_shell`** + **`cypha_rest`** |

| **P6** | Bench/tune/diagnostics: native-only (`cypha_bench_run`, `cypha_tune_run`) | ✅ Native runners exist |

| **P7** | Remove Python runtime packages; CI gate = CTest-only | ✅ **Complete** — `cypha_core`, ``, `bench/`, `cypha_lm/` decommissioned |

| **P8** | Bump `CMAKE_CXX_STANDARD` to 23 | ✅ Done |



---



## Python packages — decommission status



| Package | Native mirror | Status |

|---------|---------------|--------|

| `cypha_som/` | `native/src/som/` (smoke only) | ✅ **Removed** — archived |

| `cypha_accel/` | `native/src/accel_cuda.cu` | ✅ Native CUDA/CPU accel |

| `cypha_views/` | — | Dropped |

| `cypha_diagnostics/` | `cypha_diagnostics_run` | ✅ Native orchestrator |

| `bench/` | `cypha_bench_run`, `cypha_bench_report` | ✅ **Removed** |

| `cypha_lm/` | `cypha_lm_native`, `cyphalm_bench_native` | ✅ **Removed** |

| `` | `cypha_qt_shell` + `cypha_rest` | ✅ **Removed** |

| `cypha_core` | `cypha_core` | ✅ **Removed** — `cypha_core` is authoritative |



---



## C++ standard



- **Today:** C++23 (`native/CMakeLists.txt`) for MSVC/GCC/MinGW; CUDA standard 23 when enabled.



---



## Validation gate



```powershell

powershell -File scripts\cypha_native_validate_all.ps1

```



```bash

bash scripts/ci_native_linux.sh

ctest --test-dir native/build -R native_ --output-on-failure

```



P1/P2 smoke targets:

```powershell

ctest -R native_intelligence_profiler_smoke

ctest -R native_xor_kernel_bench_smoke

ctest -R native_kernel_llr

native/build/Release/cypha_bench_run.exe --domain-tag d03_xor

native/build/Release/xor_kernel_bench.exe --seeds 3 --passes 8 --kernel-blend 1.0

```



---



## Intelligence Stats → C++



Papers live in [`docs/research/intelligence_stats/`](../../research/intelligence_stats/README.md).



Priority implementation order:

1. Profiler + measurers (P1) ✅

2. Test bench domain `d_profile` / landscape CSV export

3. Self-correcting loop (Paper IV)

4. Soft world (Paper V)

