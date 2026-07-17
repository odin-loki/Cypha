# CyphaRegression.cmake — native golden regression executables (sources under tests/regression/).

set(CYPHA_REGRESSION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression")

function(cypha_add_golden_exe target)
  cmake_parse_arguments(_PAR "" "LINK;SOURCE" "" ${ARGN})
  if(NOT _PAR_LINK)
    set(_PAR_LINK cypha_core)
  endif()
  if(NOT _PAR_SOURCE)
    set(_PAR_SOURCE "${CYPHA_REGRESSION_DIR}/${target}.cpp")
  endif()
  add_executable("${target}" "${_PAR_SOURCE}")
  target_include_directories("${target}" PRIVATE "${CYPHA_REGRESSION_DIR}")
  target_link_libraries("${target}" PRIVATE ${_PAR_LINK})
endfunction()

# --- cypha_core golden regression tools ---
cypha_add_golden_exe(cypha_golden SOURCE "${CYPHA_REGRESSION_DIR}/golden_main.cpp")
cypha_add_golden_exe(batch_llr_golden)
cypha_add_golden_exe(rpsm_batched_llr_smoke)
cypha_add_golden_exe(em_step_smoke)
cypha_add_golden_exe(rpsm_sequence_smoke LINK cypha_lm_native)
cypha_add_golden_exe(rpsm_hierarchy_smoke)
cypha_add_golden_exe(rpsm_train_smoke)
cypha_add_golden_exe(rpsm_train_multiclass_smoke)
cypha_add_golden_exe(rpsm_embed_grad_finite_diff LINK cypha_lm_native)
cypha_add_golden_exe(rpsm_bptt_grad_finite_diff)
cypha_add_golden_exe(rpsm_spectral_alpha_smoke)
cypha_add_golden_exe(rpsm_normalized_eta_smoke)
cypha_add_golden_exe(rpsm_world_stats_smoke)
cypha_add_golden_exe(score_batch_golden)
cypha_add_golden_exe(kernel_llr_golden)
cypha_add_golden_exe(memory_train_golden)
cypha_add_golden_exe(multilabel_dif_golden)
cypha_add_golden_exe(merge_from_golden)
cypha_add_golden_exe(similarity_index_golden)
cypha_add_golden_exe(preprocessor_golden)
cypha_add_golden_exe(preprocessor_fit_golden)
cypha_add_golden_exe(csv_ingest_golden)
cypha_add_golden_exe(preprocess_train_classify_golden)
cypha_add_golden_exe(nig_adapt_golden)
cypha_add_golden_exe(train_step_vector_golden)
cypha_add_golden_exe(dif_regressor_train_step_golden)
cypha_add_golden_exe(regression_mixture_golden)
cypha_add_golden_exe(regression_m4_golden)
cypha_add_golden_exe(regression_rff_golden)
cypha_add_golden_exe(regression_two_stage_pipeline_golden)
cypha_add_golden_exe(regression_two_stage_ridge_fit_golden)
cypha_add_golden_exe(quantile_dif_train_golden)
cypha_add_golden_exe(mke_train_step_golden)
cypha_add_golden_exe(generation_golden)
cypha_add_golden_exe(gh_infer_deliberation_golden)
cypha_add_golden_exe(retrieval_golden)

# --- cypha_lm_native golden regression tools ---
cypha_add_golden_exe(cyphalm_model_golden LINK cypha_lm_native)
cypha_add_golden_exe(cyphalm_ssm_golden LINK cypha_lm_native)
cypha_add_golden_exe(embed_table_golden LINK cypha_lm_native)
cypha_add_golden_exe(cyphalm_hebbian_golden LINK cypha_lm_native)
cypha_add_golden_exe(som_golden LINK cypha_lm_native)
cypha_add_golden_exe(cyphalm_char_lstm_golden LINK cypha_lm_native)
cypha_add_golden_exe(cyphalm_checkpoint_golden LINK cypha_lm_native)
cypha_add_golden_exe(cyphalm_golden LINK cypha_lm_native)

add_executable(cypha_golden_run "${CMAKE_CURRENT_SOURCE_DIR}/apps/cypha_golden_run.cpp")
target_link_libraries(cypha_golden_run PRIVATE cypha_core)

set(
  CYPHA_GOLDEN_EXE_TARGETS
  cyphalm_model_golden
  cyphalm_ssm_golden
  cypha_golden
  gh_infer_deliberation_golden
  embed_table_golden
  retrieval_golden
  batch_llr_golden
  rpsm_batched_llr_smoke
  em_step_smoke
  rpsm_sequence_smoke
  rpsm_hierarchy_smoke
  rpsm_train_smoke
  rpsm_train_multiclass_smoke
  rpsm_embed_grad_finite_diff
  rpsm_bptt_grad_finite_diff
  rpsm_spectral_alpha_smoke
  rpsm_normalized_eta_smoke
  rpsm_world_stats_smoke
  score_batch_golden
  kernel_llr_golden
  memory_train_golden
  multilabel_dif_golden
  merge_from_golden
  similarity_index_golden
  preprocessor_golden
  preprocessor_fit_golden
  csv_ingest_golden
  preprocess_train_classify_golden
  nig_adapt_golden
  train_step_vector_golden
  dif_regressor_train_step_golden
  regression_mixture_golden
  regression_m4_golden
  regression_rff_golden
  regression_two_stage_pipeline_golden
  regression_two_stage_ridge_fit_golden
  quantile_dif_train_golden
  mke_train_step_golden
  generation_golden
  cyphalm_hebbian_golden
  som_golden
  cyphalm_char_lstm_golden
  cyphalm_checkpoint_golden
  cyphalm_golden
  cypha_golden_run
)
