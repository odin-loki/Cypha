# CTest labels for CI tiers.
#
# cypha_slow — excluded from default PR CI (scripts/ci_native_fast.sh uses -LE cypha_slow).
# Maintainer full gate: ctest -R native_ (no -LE) or scripts/ci_native_linux.sh.

set(CYPHA_CTEST_LABEL_SLOW "cypha_slow")

# Tests known slow without a high TIMEOUT property (hp train/tune/bench orchestration).
set(_CYPHA_CTEST_SLOW_EXPLICIT
  native_tune_run_smoke
  native_cyphalm_train_smoke
  native_rpsm_sequence_smoke
  native_sample_efficiency_curve_smoke
  native_robustness_curve_smoke
  native_needle_haystack_smoke
  native_memorization_canary_smoke
  native_algebraic_fingerprint_smoke
  native_cyphalm_bench_intelligence_profile
  native_d17_wikitext_smoke
  native_d17_wikitext_overnight_smoke
  native_overnight_mini_smoke
  native_cell_hypothesis_overnight_smoke
  native_cell_hypothesis_sweep_smoke
  native_cell_hypothesis_tier3_smoke
  native_corpus_smoke
  native_diagnostics_run
  native_intelligence_bench_smoke
  native_predictive_codec_smoke
  native_predictive_codec_bench_smoke
  native_xor_kernel_bench_smoke
  native_class_gmm_p3_smoke
  native_encoder_ib_p6_smoke
  native_nig_bma_p4_smoke
  native_d26_medium_overnight_smoke
  native_d39_intelligence_monitor_smoke
  native_d40_math_integration_smoke
)

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

function(cypha_apply_ctest_slow_labels)
  foreach(_t IN LISTS _CYPHA_CTEST_SLOW_EXPLICIT)
    cypha_ctest_mark_slow("${_t}")
  endforeach()

  get_property(_all_tests DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY TESTS)
  foreach(_t IN LISTS _all_tests)
    # Joint grid / math-integration domain smokes (d41–d76): maintainer tier.
    if(_t MATCHES "^native_d(4[1-9]|[5-7][0-9])_")
      cypha_ctest_mark_slow("${_t}")
    endif()

  endforeach()

  list(LENGTH _CYPHA_CTEST_SLOW_EXPLICIT _n_explicit)
  message(STATUS "Cypha CTest: cypha_slow labels applied (explicit=${_n_explicit}, d41–d76 domain smokes)")
endfunction()
