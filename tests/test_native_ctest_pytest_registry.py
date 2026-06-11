"""
Guardrail: every ``add_test`` ``NAME native_*`` in ``native/CMakeLists.txt`` must have subprocess pytest coverage.

When you add a CTest, extend ``_NATIVE_CTEST_TO_PYTEST`` and add the pytest module.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CMAKE = _ROOT / "native" / "CMakeLists.txt"

# CTest name -> one or more pytest files (relative to repo root) that exercise the same native binary / scenario.
_NATIVE_CTEST_TO_PYTEST: dict[str, list[str]] = {
    "native_parity": ["tests/test_cypha_parity_native.py"],
    "native_batch_llr": ["tests/test_batch_llr_native_parity.py"],
    "native_memory_train": ["tests/test_memory_train_native_parity.py"],
    "native_multilabel_dif": ["tests/test_multilabel_dif_native_parity.py"],
    "native_merge_from": ["tests/test_merge_from_native_parity.py"],
    "native_similarity_index": ["tests/test_similarity_index_native_parity.py"],
    "native_memory_train_roundtrip": ["tests/test_memory_train_roundtrip_native.py"],
    "native_preprocessor": ["tests/test_preprocessor_native_parity.py"],
    "native_preprocessor_fit": ["tests/test_preprocessor_fit_native_parity.py"],
    "native_csv_ingest": ["tests/test_csv_ingest_native_parity.py"],
    "native_studio_trainer_preprocess_classify_hotpath": [
        "tests/test_studio_trainer_preprocess_classify_hotpath_native_parity.py"
    ],
    "native_studio_trainer_preprocess_gh_classify_hotpath": [
        "tests/test_studio_trainer_preprocess_gh_classify_hotpath_native_parity.py"
    ],
    "native_csv_preprocess_classify_hotpath": ["tests/test_csv_preprocess_classify_hotpath_native_parity.py"],
    "native_nig_adapt": ["tests/test_nig_adapt_native_parity.py"],
    "native_train_step_vector": ["tests/test_train_step_vector_native_parity.py"],
    "native_dif_regressor_train_step": ["tests/test_dif_regressor_train_step_native_parity.py"],
    "native_regression_mixture": ["tests/test_regression_mixture_native_parity.py"],
    "native_regression_m4": ["tests/test_regression_m4_native_parity.py"],
    "native_regression_rff": ["tests/test_regression_rff_native_parity.py"],
    "native_regression_two_stage_pipeline": ["tests/test_two_stage_pipeline_native_parity.py"],
    "native_regression_two_stage_ridge_fit": ["tests/test_two_stage_ridge_fit_native_parity.py"],
    "native_regression_two_stage_e2e_ridge": ["tests/test_two_stage_e2e_ridge_native_parity.py"],
    "native_quantile_dif_train": ["tests/test_quantile_dif_train_native_parity.py"],
    "native_dif_train_replay": ["tests/test_dif_train_replay_native_parity.py"],
    "native_studio_trainer_classify_hotpath": ["tests/test_studio_trainer_classify_hotpath_native_parity.py"],
    "native_studio_trainer_gh_classify_hotpath": ["tests/test_studio_trainer_gh_classify_hotpath_native_parity.py"],
    "native_mke_train_step": ["tests/test_mke_train_step_native_parity.py"],
    "native_mke_train_extended": ["tests/test_mke_train_step_native_parity.py"],
    "native_generation": ["tests/test_generation_native_parity.py"],
    "native_create_model": ["tests/test_create_model_native.py"],
    "native_cuda_smoke": ["tests/test_cuda_smoke_native.py"],
    "native_cuda_bench": ["tests/test_cuda_smoke_native.py"],
    "native_registry_register": ["tests/test_registry_register_native_parity.py"],
    "native_experiment_db_smoke": ["tests/test_experiment_db_smoke_native_parity.py"],
    "native_experiment_db_file": ["tests/test_experiment_db_smoke_native_parity.py"],
    "native_experiment_db_crud": ["tests/test_experiment_db_crud_native_parity.py"],
    "native_qt_stub_load_reference": ["tests/test_qt_stub_native.py"],
    "native_qt_shell_smoke": ["tests/test_qt_shell_native.py"],
    "native_score_batch": ["tests/test_score_batch_native_parity.py"],
    "native_kernel_llr": ["tests/test_kernel_llr_native_parity.py"],
    "native_embed_table": ["tests/test_embed_table_native_parity.py"],
    "native_retrieval": ["tests/test_retrieval_native_parity.py"],
    "native_gh_infer_deliberation": ["tests/test_gh_infer_deliberation_native_parity.py"],
    "native_bench_run_list_domains": ["tests/test_native_bench_cli_smoke.py"],
    "native_tune_run_smoke": ["tests/test_native_bench_cli_smoke.py"],
    "native_diagnostics_run": ["tests/test_native_bench_cli_smoke.py"],
    "native_cyphalm_char_lstm": ["tests/test_cyphalm_native_parity.py"],
    "native_cyphalm_model_parity": ["tests/test_cyphalm_native_parity.py"],
    "native_cyphalm_checkpoint_parity": ["tests/test_cyphalm_native_parity.py"],
    "native_cyphalm_hebbian": ["tests/test_cyphalm_native_parity.py"],
    "native_cyphalm_ssm": ["tests/test_cyphalm_native_parity.py"],
    "native_cyphalm_ssm_fixture": ["tests/test_cyphalm_native_parity.py"],
    "native_cyphalm_parity_suite": ["tests/test_cyphalm_native_parity.py"],
    "native_som_parity": ["tests/test_som_native_parity.py"],
}


def _ctest_names_from_cmake() -> set[str]:
    assert _CMAKE.is_file(), "native/CMakeLists.txt missing"
    txt = _CMAKE.read_text(encoding="utf-8")
    return set(re.findall(r"NAME\s+(native_\w+)", txt))


def test_every_native_ctest_has_pytest_mapping():
    from_cmake = _ctest_names_from_cmake()
    mapped = set(_NATIVE_CTEST_TO_PYTEST)
    missing_map = sorted(from_cmake - mapped)
    assert not missing_map, (
        "Add these CTest names to _NATIVE_CTEST_TO_PYTEST in test_native_ctest_pytest_registry.py: " + ", ".join(missing_map)
    )
    stale = sorted(mapped - from_cmake)
    assert not stale, (
        "Remove stale CTest keys from _NATIVE_CTEST_TO_PYTEST (not in CMakeLists.txt): " + ", ".join(stale)
    )


def test_mapped_pytest_files_exist():
    for paths in _NATIVE_CTEST_TO_PYTEST.values():
        for rel in paths:
            p = _ROOT / rel
            assert p.is_file(), f"missing pytest module for native CTest registry: {rel}"
