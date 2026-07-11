# CyphaParity.cmake — Python-vs-native parity executables (sources under tests/parity/).
# Binary names are stable for CTest and cypha_diagnostics_run; only source paths moved in Phase B.

set(CYPHA_PARITY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/parity")

function(cypha_add_parity_exe target)
  cmake_parse_arguments(_PAR "" "LINK;SOURCE" "" ${ARGN})
  if(NOT _PAR_LINK)
    set(_PAR_LINK cypha_core)
  endif()
  if(NOT _PAR_SOURCE)
    set(_PAR_SOURCE "${CYPHA_PARITY_DIR}/${target}.cpp")
  endif()
  add_executable("${target}" "${_PAR_SOURCE}")
  target_include_directories("${target}" PRIVATE "${CYPHA_PARITY_DIR}")
  target_link_libraries("${target}" PRIVATE ${_PAR_LINK})
endfunction()

# --- cypha_core parity tools ---
cypha_add_parity_exe(cypha_parity SOURCE "${CYPHA_PARITY_DIR}/parity_main.cpp")
cypha_add_parity_exe(batch_llr_parity)
cypha_add_parity_exe(rpsm_batched_llr_smoke)
cypha_add_parity_exe(rpsm_sequence_smoke LINK cypha_lm_native)
cypha_add_parity_exe(rpsm_hierarchy_smoke)
cypha_add_parity_exe(rpsm_train_smoke)
cypha_add_parity_exe(rpsm_train_multiclass_smoke)
cypha_add_parity_exe(rpsm_embed_grad_finite_diff LINK cypha_lm_native)
cypha_add_parity_exe(rpsm_bptt_grad_finite_diff)
cypha_add_parity_exe(rpsm_spectral_alpha_smoke)
cypha_add_parity_exe(rpsm_normalized_eta_smoke)
cypha_add_parity_exe(score_batch_parity)
cypha_add_parity_exe(kernel_llr_parity)
cypha_add_parity_exe(memory_train_parity)
cypha_add_parity_exe(multilabel_dif_parity)
cypha_add_parity_exe(merge_from_parity)
cypha_add_parity_exe(similarity_index_parity)
cypha_add_parity_exe(preprocessor_parity)
cypha_add_parity_exe(preprocessor_fit_parity)
cypha_add_parity_exe(csv_ingest_parity)
cypha_add_parity_exe(preprocess_train_classify_parity)
cypha_add_parity_exe(nig_adapt_parity)
cypha_add_parity_exe(train_step_vector_parity)
cypha_add_parity_exe(dif_regressor_train_step_parity)
cypha_add_parity_exe(regression_mixture_parity)
cypha_add_parity_exe(regression_m4_parity)
cypha_add_parity_exe(regression_rff_parity)
cypha_add_parity_exe(regression_two_stage_pipeline_parity)
cypha_add_parity_exe(regression_two_stage_ridge_fit_parity)
cypha_add_parity_exe(quantile_dif_train_parity)
cypha_add_parity_exe(mke_train_step_parity)
cypha_add_parity_exe(generation_parity)
cypha_add_parity_exe(gh_infer_deliberation_parity)
cypha_add_parity_exe(retrieval_parity)

# --- cypha_lm_native parity tools ---
cypha_add_parity_exe(cyphalm_model_parity LINK cypha_lm_native)
cypha_add_parity_exe(cyphalm_ssm_parity LINK cypha_lm_native)
cypha_add_parity_exe(embed_table_parity LINK cypha_lm_native)
cypha_add_parity_exe(cyphalm_hebbian_parity LINK cypha_lm_native)
cypha_add_parity_exe(som_parity LINK cypha_lm_native)
cypha_add_parity_exe(cyphalm_char_lstm_parity LINK cypha_lm_native)
cypha_add_parity_exe(cyphalm_checkpoint_parity LINK cypha_lm_native)
cypha_add_parity_exe(cyphalm_parity LINK cypha_lm_native)

add_executable(cypha_parity_run "${CMAKE_CURRENT_SOURCE_DIR}/apps/cypha_parity_run.cpp")
target_link_libraries(cypha_parity_run PRIVATE cypha_core)

set(
  CYPHA_PARITY_EXE_TARGETS
  cyphalm_model_parity
  cyphalm_ssm_parity
  cypha_parity
  gh_infer_deliberation_parity
  embed_table_parity
  retrieval_parity
  batch_llr_parity
  rpsm_batched_llr_smoke
  rpsm_sequence_smoke
  rpsm_hierarchy_smoke
  rpsm_train_smoke
  rpsm_train_multiclass_smoke
  rpsm_embed_grad_finite_diff
  rpsm_bptt_grad_finite_diff
  rpsm_spectral_alpha_smoke
  rpsm_normalized_eta_smoke
  score_batch_parity
  kernel_llr_parity
  memory_train_parity
  multilabel_dif_parity
  merge_from_parity
  similarity_index_parity
  preprocessor_parity
  preprocessor_fit_parity
  csv_ingest_parity
  preprocess_train_classify_parity
  nig_adapt_parity
  train_step_vector_parity
  dif_regressor_train_step_parity
  regression_mixture_parity
  regression_m4_parity
  regression_rff_parity
  regression_two_stage_pipeline_parity
  regression_two_stage_ridge_fit_parity
  quantile_dif_train_parity
  mke_train_step_parity
  generation_parity
  cyphalm_hebbian_parity
  som_parity
  cyphalm_char_lstm_parity
  cyphalm_checkpoint_parity
  cyphalm_parity
  cypha_parity_run
)
