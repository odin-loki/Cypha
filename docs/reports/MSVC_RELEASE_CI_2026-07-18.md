# Windows MSVC CI + release — 2026-07-18

## Change

| Before | After |
|--------|-------|
| CI `mingw_cross` (Ubuntu → MinGW PE) | CI `windows_msvc` (`windows-latest`, MSVC + Ninja Release) |
| Release job MinGW cross on Ubuntu | Release job native MSVC on `windows-latest` |
| `package_release_windows.sh` MinGW layout + stale `*_parity` names | `package_release_windows.ps1` MSVC layout + current `*_golden` / lock CLIs |

## Package contents (Windows zip)

**bin/:** `cypha_rest`, `cypha_bench_run`, `cypha_bench_report`, `cypha_diagnostics_run`, `cypha_tune_run`, `cyphalm_bench_native`, `cypha_baseline_lock`, `baseline_lock_validate`, `registry_register`, `create_model_smoke`

**bin/dev/:** `score_batch_golden`, `multilabel_dif_golden`, `merge_from_golden`, `similarity_index_golden`, `embed_table_golden`, `retrieval_golden`, `som_golden`, `kernel_llr_golden`, `gh_infer_deliberation_golden`, `cyphalm_checkpoint_golden`

## Related compile fixes (Linux Release Qt)

- `RpsmGlobalMemory::slots()` → `slot_data()` (Qt `slots` macro clash)
- `experiment_export_compare` uses `QString(key).replace(...)` (const QString)

## CI follow-up

First `windows_msvc` run failed on `em_step.hpp` (`std::max` vs Windows `max` macro). Fixed with global MSVC `NOMINMAX` / `WIN32_LEAN_AND_MEAN` in `native/CMakeLists.txt` and `#include <algorithm>` in `em_step.hpp` / `kernel_memory.hpp`.
