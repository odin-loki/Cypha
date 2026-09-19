# CTest labels for CI tiers.
#
# cypha_slow — maintainer / optional nightly tier (excluded from PR hp smoke gate).
# PR gate (Linux + macOS): scripts/ci_native_hp_smoke.sh (hp/CyphaLM allowlist only).
# Broad maintainer gate: scripts/ci_native_fast.sh (-LE cypha_slow) or ci_native_linux.sh (all).

set(CYPHA_CTEST_LABEL_SLOW "cypha_slow")

function(cypha_ctest_mark_slow test_name)
  if(NOT TEST "${test_name}")
    return()
  endif()
  get_property(_labels TEST "${test_name}" PROPERTY LABELS)
  if(_labels AND "${CYPHA_CTEST_LABEL_SLOW}" IN_LIST _labels)
    return()
  endif()
  set_property(TEST "${test_name}" APPEND PROPERTY LABELS "${CYPHA_CTEST_LABEL_SLOW}")
endfunction()

function(cypha_ctest_name_is_slow test_name out_var)
  set(_slow FALSE)
  # Bench / lock / forecast / train / tune / overnight orchestration.
  if("${test_name}" MATCHES "^native_tune_run_smoke$"
      OR "${test_name}" MATCHES "^native_cyphalm_train_smoke$"
      OR "${test_name}" MATCHES "^native_one_cypha_smoke$"
      OR "${test_name}" MATCHES "^native_baseline_lock"
      OR "${test_name}" MATCHES "^native_cyphalm_bench"
      OR "${test_name}" MATCHES "^native_forecast_smoke$"
      OR "${test_name}" MATCHES "^native_cell_hypothesis"
      OR "${test_name}" MATCHES "^native_overnight"
      OR "${test_name}" MATCHES "^native_corpus_smoke$"
      OR "${test_name}" MATCHES "^native_diagnostics_run$"
      OR "${test_name}" MATCHES "^native_intelligence_bench"
      OR "${test_name}" MATCHES "^native_predictive_codec"
      OR "${test_name}" MATCHES "^native_sample_efficiency"
      OR "${test_name}" MATCHES "^native_robustness_curve"
      OR "${test_name}" MATCHES "^native_needle_haystack"
      OR "${test_name}" MATCHES "^native_memorization_canary"
      OR "${test_name}" MATCHES "^native_algebraic_fingerprint"
      OR "${test_name}" MATCHES "^native_xor_kernel_bench"
      OR "${test_name}" MATCHES "^native_class_gmm_p3"
      OR "${test_name}" MATCHES "^native_encoder_ib_p6"
      OR "${test_name}" MATCHES "^native_nig_bma_p4"
      OR "${test_name}" MATCHES "^native_orf_encoder_bench"
      OR "${test_name}" MATCHES "^native_views_leaderboard"
      OR "${test_name}" MATCHES "^native_tau_forget_gate"
      OR "${test_name}" MATCHES "^native_lm_self_correct"
      OR "${test_name}" MATCHES "^native_quality_recipe_wave"
      OR "${test_name}" MATCHES "^native_throughput_lock")
    set(_slow TRUE)
  endif()
  # Domain smokes d21–d76 (bench grids, overnight tiers, math-integration sweeps).
  if("${test_name}" MATCHES "^native_d(2[1-9]|[3-7][0-9])_")
    set(_slow TRUE)
  endif()
  # d17 wikitext / hybrid bench smokes (hp path uses hybrid alias but still heavy).
  if("${test_name}" MATCHES "^native_d17_")
    set(_slow TRUE)
  endif()
  set(${out_var} ${_slow} PARENT_SCOPE)
endfunction()

function(cypha_apply_ctest_slow_labels)
  get_property(_all_tests DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY TESTS)
  set(_n_slow 0)
  foreach(_t IN LISTS _all_tests)
    cypha_ctest_name_is_slow("${_t}" _is_slow)
    if(_is_slow)
      cypha_ctest_mark_slow("${_t}")
      math(EXPR _n_slow "${_n_slow} + 1")
    endif()
  endforeach()
  message(STATUS "Cypha CTest: ${_n_slow} tests labeled cypha_slow (PR gate uses ci_native_hp_smoke.sh)")
endfunction()
