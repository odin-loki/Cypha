#!/usr/bin/env bash
# Python regression checks: test_cypha.py + pytest bundle (native parity modules under tests/, schema/registry helpers) + optional full fast suite.
# Usage: bash scripts/run_all_regressions.sh [--full]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

PY="${PY:-python3}"
if [[ -x "$ROOT/.venv-wsl/bin/python" ]] \
  && "$ROOT/.venv-wsl/bin/python" -c "import pytest" 2>/dev/null; then
  PY="$ROOT/.venv-wsl/bin/python"
fi

echo "== test_cypha.py =="
"$PY" test_cypha.py

echo "== pytest regression bundle =="
"$PY" -m pytest \
  tests/test_regression_mixture_contract.py \
  tests/test_regression_m4_native_parity.py \
  tests/test_regression_rff_native_parity.py \
  tests/test_trainer_regression_fit.py \
  tests/test_export_regression_head_script.py \
  tests/test_parity_fixtures.py \
  tests/test_studio_data_registry.py \
  tests/test_experiment_foreign_key.py \
  tests/test_experiment_schema_contract.py \
  tests/test_experiment_db_paging.py \
  tests/test_experiment_native_seed.py \
  tests/test_api_contract.py \
  tests/test_cypha_rest_smoke.py \
  tests/test_native_ctest_pytest_registry.py \
  tests/test_preprocessor_fit_native_parity.py \
  tests/test_csv_ingest_native_parity.py \
  tests/test_dif_regressor_train_step_native_parity.py \
  tests/test_cypha_binary_buffer_api.py \
  tests/test_qt_stub_native.py \
  tests/test_qt_shell_native.py \
  tests/test_studio_trainer_classify_hotpath_native_parity.py \
  tests/test_studio_trainer_gh_classify_hotpath_native_parity.py \
  tests/test_studio_trainer_preprocess_classify_hotpath_native_parity.py \
  tests/test_studio_trainer_preprocess_gh_classify_hotpath_native_parity.py \
  tests/test_batch_llr_native_parity.py \
  tests/test_cypha_parity_native.py \
  tests/test_csv_preprocess_classify_hotpath_native_parity.py \
  tests/test_dif_train_replay_native_parity.py \
  tests/test_experiment_db_crud_native_parity.py \
  tests/test_experiment_db_smoke_native_parity.py \
  tests/test_memory_train_native_parity.py \
  tests/test_memory_train_roundtrip_native.py \
  tests/test_mke_train_step_native_parity.py \
  tests/test_nig_adapt_native_parity.py \
  tests/test_preprocessor_native_parity.py \
  tests/test_quantile_dif_train_native_parity.py \
  tests/test_registry_register_native_parity.py \
  tests/test_train_step_vector_native_parity.py \
  tests/test_two_stage_e2e_ridge_native_parity.py \
  tests/test_two_stage_pipeline_native_parity.py \
  tests/test_two_stage_ridge_fit_native_parity.py \
  tests/test_accel_cross_gemm.py \
  tests/test_accel_cypha_wired.py \
  tests/test_csv_chunked_parity.py \
  tests/test_cypha_load_state_context.py \
  tests/test_env_config.py \
  tests/test_fused_encode_score.py \
  tests/test_gig_vectorized.py \
  tests/test_inference_engine.py \
  tests/test_memory_train_fixture.py \
  tests/test_native_parity_sidecar.py \
  tests/test_preprocessor_fixture.py \
  tests/test_score_matrix_field_modes.py \
  tests/test_trainer_registry_eval_wiring.py \
  tests/test_training_plot_compress.py \
  -q --tb=short

if [[ "${1:-}" == "--full" ]]; then
  echo "== pytest tests/ -m 'not slow' =="
  "$PY" -m pytest tests/ -m "not slow" -q --tb=short
fi

echo "Regression bundle OK. CTest↔pytest registry, all native subprocess parity modules under tests/ (many skip without binary), .cypha buffer API, optional Qt stub. cypha_rest: native/build-mingw-w64/cypha_rest.exe or CYPHA_REST_BIN. See native/README.md."
