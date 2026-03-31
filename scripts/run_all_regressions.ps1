# Run Python regression checks: Cypha.py suite + pytest bundle (schema/registry/API + native subprocess parity modules — many skip without ELF/exe) + optional full fast suite.
# Usage (repo root): powershell -ExecutionPolicy Bypass -File scripts/run_all_regressions.ps1 [-Full]
param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $env:QT_QPA_PLATFORM) {
    $env:QT_QPA_PLATFORM = "offscreen"
}

# Prefer UTF-8 for console I/O where supported (reduces Windows encoding surprises).
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$VenvWin = Join-Path $Root ".venv-win\Scripts\python.exe"
$Py = $null
if (Test-Path $VenvWin) {
    & $VenvWin -c "import pytest" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $Py = $VenvWin }
}
if (-not $Py) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $Py = "py"
    } else {
        $Py = "python"
    }
}

Write-Host "== test_cypha.py (Cypha.py regressors + save/load) =="
if ($Py -eq "py") {
    & py -3 (Join-Path $Root "test_cypha.py")
} else {
    & $Py (Join-Path $Root "test_cypha.py")
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$regTests = @(
    "tests/test_regression_mixture_contract.py",
    "tests/test_regression_mixture_native_parity.py",
    "tests/test_regression_m4_native_parity.py",
    "tests/test_regression_rff_native_parity.py",
    "tests/test_trainer_regression_fit.py",
    "tests/test_export_regression_head_script.py",
    "tests/test_parity_fixtures.py",
    "tests/test_studio_data_registry.py",
    "tests/test_experiment_foreign_key.py",
    "tests/test_experiment_schema_contract.py",
    "tests/test_experiment_db_paging.py",
    "tests/test_experiment_native_seed.py",
    "tests/test_api_contract.py",
    "tests/test_cypha_rest_smoke.py",
    "tests/test_native_ctest_pytest_registry.py",
    "tests/test_preprocessor_fit_native_parity.py",
    "tests/test_csv_ingest_native_parity.py",
    "tests/test_dif_regressor_train_step_native_parity.py",
    "tests/test_cypha_binary_buffer_api.py",
    "tests/test_qt_stub_native.py",
    "tests/test_qt_shell_native.py",
    "tests/test_studio_trainer_classify_hotpath_native_parity.py",
    "tests/test_studio_trainer_gh_classify_hotpath_native_parity.py",
    "tests/test_studio_trainer_preprocess_classify_hotpath_native_parity.py",
    "tests/test_studio_trainer_preprocess_gh_classify_hotpath_native_parity.py",
    "tests/test_batch_llr_native_parity.py",
    "tests/test_cypha_parity_native.py",
    "tests/test_csv_preprocess_classify_hotpath_native_parity.py",
    "tests/test_dif_train_replay_native_parity.py",
    "tests/test_experiment_db_crud_native_parity.py",
    "tests/test_experiment_db_smoke_native_parity.py",
    "tests/test_memory_train_native_parity.py",
    "tests/test_memory_train_roundtrip_native.py",
    "tests/test_mke_train_step_native_parity.py",
    "tests/test_nig_adapt_native_parity.py",
    "tests/test_preprocessor_native_parity.py",
    "tests/test_quantile_dif_train_native_parity.py",
    "tests/test_registry_register_native_parity.py",
    "tests/test_train_step_vector_native_parity.py",
    "tests/test_two_stage_e2e_ridge_native_parity.py",
    "tests/test_two_stage_pipeline_native_parity.py",
    "tests/test_two_stage_ridge_fit_native_parity.py",
    "tests/test_accel_cross_gemm.py",
    "tests/test_accel_cypha_wired.py",
    "tests/test_csv_chunked_parity.py",
    "tests/test_cypha_load_state_context.py",
    "tests/test_env_config.py",
    "tests/test_fused_encode_score.py",
    "tests/test_gig_vectorized.py",
    "tests/test_inference_engine.py",
    "tests/test_memory_train_fixture.py",
    "tests/test_native_parity_sidecar.py",
    "tests/test_preprocessor_fixture.py",
    "tests/test_score_matrix_field_modes.py",
    "tests/test_trainer_registry_eval_wiring.py",
    "tests/test_training_plot_compress.py"
)
$paths = $regTests | ForEach-Object { Join-Path $Root $_ }

Write-Host "== pytest regression bundle =="
if ($Py -eq "py") {
    & py -3 -m pytest @paths -q --tb=short
} else {
    & $Py -m pytest @paths -q --tb=short
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Full) {
    Write-Host "== pytest tests/ -m 'not slow' =="
    if ($Py -eq "py") {
        & py -3 -m pytest (Join-Path $Root "tests") -m "not slow" -q --tb=short
    } else {
        & $Py -m pytest (Join-Path $Root "tests") -m "not slow" -q --tb=short
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Regression bundle OK (CTest registry, all native subprocess parity modules in tests/ — many skip without ELF/exe, .cypha buffer API, Qt stub if built). cypha_rest: WSL MinGW -> native\build-mingw-w64\cypha_rest.exe (powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1) or CYPHA_REST_BIN. Other native tools: native/README.md."
