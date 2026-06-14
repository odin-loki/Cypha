cmake_minimum_required(VERSION 3.19)
project(cypha_native VERSION 2.2.8 LANGUAGES CXX C)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CyphaMinGW.cmake")

option(CYPHA_BUILD_EXPERIMENT_DB "Build experiment_db_smoke / experiment_db_crud_parity (requires SQLite3)" OFF)
option(CYPHA_BUILD_QT "Build optional Qt targets under qt/: cypha_qt_stub (Core), cypha_qt_shell (Widgets)" OFF)
option(CYPHA_QT_CHARTS "When CYPHA_BUILD_QT is ON, link Qt Charts for cypha_qt_shell if found (default OFF for qt6-base-only CI)" OFF)
option(CYPHA_ENABLE_CUDA "Build CUDA GPU path in accel backend (NVIDIA toolkit + driver); CPU uses std::thread" OFF)
option(CYPHA_ENABLE_OPENSSL "Enable CPPHTTPLIB_OPENSSL_SUPPORT for federated coordinator/worker TLS" OFF)
set(CYPHA_ACCEL_GPU_MIN_BATCH_ROWS "1" CACHE STRING "Minimum batch row count (n) to use CUDA in cypha::accel (1 = always batch on GPU when available)")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CYPHA_ENABLE_CUDA)
  if(MINGW)
    message(FATAL_ERROR "CYPHA_ENABLE_CUDA is not supported for MinGW targets (use native MSVC or Linux + nvcc).")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 23)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
endif()

add_library(cypha_core STATIC
  src/load_cypha.cpp
  src/save_cypha.cpp
  src/infer_cpu.cpp
  src/retrieval.cpp
  src/nig_gig_math.cpp
  src/bessel_table_data.cpp
  src/preprocessor.cpp
  src/preprocessor_fit.cpp
  src/mt19937_rng.cpp
  src/csv_ingest.cpp
  src/memory_train.cpp
  src/ewc_regularizer.cpp
  src/curriculum.cpp
  src/federated_aggregate.cpp
  src/registry.cpp
  src/sync_infer.cpp
  src/encoder_contrastive.cpp
  src/replay_buffer.cpp
  src/train_step_vector.cpp
  src/nig_field.cpp
  src/regression_stub.cpp
  src/mke_scalar_train_step.cpp
  src/generation.cpp
  src/accel_backend.cpp
  src/create_model.cpp
  src/kernel_memory.cpp
  src/branch_a_router.cpp
  src/som/som_encoder.cpp
  src/som/temporal_som.cpp
  src/som/gng_expert.cpp
  src/som/gria_controller.cpp
  src/som/discriminative_feedback.cpp
  src/multilabel_dif.cpp
  src/similarity_index.cpp
  src/intelligence/nig_statistic_state.cpp
  src/intelligence/measurers.cpp
  src/intelligence/intelligence_profiler.cpp
  src/intelligence/intelligence_profile_json.cpp
  src/intelligence/epistemic_threshold.cpp
  src/intelligence/soft_world_monitor.cpp
  src/intelligence/self_correcting_infer.cpp
  src/intelligence/profile_from_model.cpp
  src/intelligence/causal_graph.cpp
  src/intelligence/profile_guided_loss.cpp
  src/rpsm/psi_matrices.cpp
  src/rpsm/rpsm_sequence_layer.cpp
  src/bench/bench_paths.cpp
)

if(CYPHA_ENABLE_CUDA)
  target_sources(cypha_core PRIVATE src/accel_cuda.cu)
  target_compile_definitions(cypha_core PUBLIC CYPHA_ENABLE_CUDA)
  target_link_libraries(cypha_core PUBLIC CUDA::cudart)
  set_target_properties(cypha_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

file(GLOB CYPHALM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/*.cpp")
if(NOT CYPHALM_SOURCES)
  message(FATAL_ERROR "cypha_lm_native: no sources under native/src/cyphalm/")
endif()
add_library(cypha_lm_native STATIC ${CYPHALM_SOURCES})

target_include_directories(cypha_lm_native PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_link_libraries(cypha_lm_native PUBLIC cypha_core)

add_library(cypha_bench_native STATIC
  src/bench/bench_profile.cpp
  src/bench/bench_metrics.cpp
  src/bench/bench_baselines.cpp
  src/bench/bench_report_json.cpp
  src/bench/bench_encoder_text.cpp
  src/bench/bench_encoder_image.cpp
  src/bench/bench_encoder_chess.cpp
  src/bench/bench_encoder_go.cpp
  src/bench/bench_encoder_poker.cpp
  src/bench/bench_encoder_document.cpp
  src/bench/bench_encoder_timeseries.cpp
  src/bench/bench_cross_domain.cpp
  src/bench/bench_report.cpp
  src/bench/bench_figures.cpp
  src/bench/bench_figure_render.cpp
  src/bench/bench_tune.cpp
  src/bench/bench_domains.cpp
)

target_include_directories(cypha_bench_native PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_link_libraries(cypha_bench_native PUBLIC cypha_core cypha_lm_native)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cypha_lm_native PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "cypha_lm_native: OpenMP enabled")
else()
  message(STATUS "cypha_lm_native: OpenMP not found; batch uses std::thread")
endif()

if(MSVC)
  target_compile_options(cypha_lm_native PRIVATE /W4 /permissive-)
else()
  target_compile_options(cypha_lm_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Phase B: parity harness sources live under tests/parity/ (binary names unchanged).
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CyphaParity.cmake")

add_executable(cyphalm_bench_native tools/cyphalm_bench_native.cpp)
target_link_libraries(cyphalm_bench_native PRIVATE cypha_lm_native)
cypha_mingw_apply_exe_link_options(cyphalm_bench_native)

add_executable(cyphalm_ssm_diagnose tools/cyphalm_ssm_diagnose.cpp)
target_link_libraries(cyphalm_ssm_diagnose PRIVATE cypha_bench_native)
cypha_mingw_apply_exe_link_options(cyphalm_ssm_diagnose)

# Phase B: production entrypoints under apps/ (see cmake/CyphaApps.cmake).
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CyphaApps.cmake")

target_include_directories(cypha_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_compile_definitions(cypha_core PRIVATE
  CYPHA_ACCEL_GPU_MIN_BATCH_ROWS=${CYPHA_ACCEL_GPU_MIN_BATCH_ROWS}
)

add_executable(cypha_fixture_gen tools/fixture_generate.cpp)
target_link_libraries(cypha_fixture_gen PRIVATE cypha_core cypha_lm_native)

add_executable(xor_kernel_bench tools/xor_kernel_bench.cpp)
target_link_libraries(xor_kernel_bench PRIVATE cypha_core)

add_executable(cypha_kernel_tune tools/cypha_kernel_tune.cpp)
target_link_libraries(cypha_kernel_tune PRIVATE cypha_core)

add_executable(kernel_snapshot_roundtrip tools/kernel_snapshot_roundtrip.cpp)
target_link_libraries(kernel_snapshot_roundtrip PRIVATE cypha_core)

add_executable(kernel_cypha_roundtrip tools/kernel_cypha_roundtrip.cpp)
target_link_libraries(kernel_cypha_roundtrip PRIVATE cypha_core)

add_executable(memory_train_roundtrip tools/memory_train_roundtrip.cpp)
target_link_libraries(memory_train_roundtrip PRIVATE cypha_core)

add_executable(preprocessor_rff_gamma_cv_test tools/preprocessor_rff_gamma_cv_test.cpp)
target_link_libraries(preprocessor_rff_gamma_cv_test PRIVATE cypha_core)

add_executable(registry_register tools/registry_register.cpp)
target_link_libraries(registry_register PRIVATE cypha_core)

add_executable(rest_schema_contract tests/rest_schema_contract.cpp)
target_include_directories(rest_schema_contract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
if(WIN32)
  target_link_libraries(rest_schema_contract PRIVATE ws2_32)
endif()

add_executable(cuda_smoke tools/cuda_smoke.cpp)
target_link_libraries(cuda_smoke PRIVATE cypha_core)

add_executable(create_model_smoke tools/create_model_smoke.cpp)
target_link_libraries(create_model_smoke PRIVATE cypha_core)

add_executable(cypha_federated_merge tools/cypha_federated_merge.cpp)
target_link_libraries(cypha_federated_merge PRIVATE cypha_core)

add_executable(cypha_federated_coordinator tools/cypha_federated_coordinator.cpp)
target_include_directories(cypha_federated_coordinator PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(cypha_federated_coordinator PRIVATE cypha_core)
if(WIN32)
  target_link_libraries(cypha_federated_coordinator PRIVATE ws2_32)
endif()

add_executable(cypha_federated_worker tools/cypha_federated_worker.cpp)
target_include_directories(cypha_federated_worker PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(cypha_federated_worker PRIVATE cypha_core)
if(WIN32)
  target_link_libraries(cypha_federated_worker PRIVATE ws2_32)
endif()

add_executable(ewc_d16b_smoke tools/ewc_d16b_smoke.cpp)
target_link_libraries(ewc_d16b_smoke PRIVATE cypha_core)

add_executable(ewc_cyphalm_smoke tools/ewc_cyphalm_smoke.cpp)
target_link_libraries(ewc_cyphalm_smoke PRIVATE cypha_lm_native)

add_executable(ewc_hybrid_smoke tools/ewc_hybrid_smoke.cpp)
target_link_libraries(ewc_hybrid_smoke PRIVATE cypha_lm_native)

add_executable(ewc_weights_smoke tools/ewc_weights_smoke.cpp)
target_link_libraries(ewc_weights_smoke PRIVATE cypha_lm_native)

add_executable(native_sr_gate_laws_smoke tools/native_sr_gate_laws_smoke.cpp)
target_link_libraries(native_sr_gate_laws_smoke PRIVATE cypha_lm_native)

add_executable(federated_worker_smoke tools/federated_worker_smoke.cpp)
target_include_directories(federated_worker_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(federated_worker_smoke PRIVATE cypha_core)
if(WIN32)
  target_link_libraries(federated_worker_smoke PRIVATE ws2_32)
endif()

add_executable(federated_tls_smoke tools/federated_tls_smoke.cpp)
target_include_directories(federated_tls_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(federated_tls_smoke PRIVATE cypha_core)
if(WIN32)
  target_link_libraries(federated_tls_smoke PRIVATE ws2_32)
endif()

if(CYPHA_ENABLE_OPENSSL)
  find_package(OpenSSL REQUIRED)
  foreach(_cypha_fed_tls_target cypha_federated_coordinator cypha_federated_worker federated_tls_smoke)
    target_compile_definitions(${_cypha_fed_tls_target} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(${_cypha_fed_tls_target} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    if(WIN32)
      target_link_libraries(${_cypha_fed_tls_target} PRIVATE crypt32)
    endif()
  endforeach()
  message(STATUS "Federated TLS: OpenSSL enabled (CPPHTTPLIB_OPENSSL_SUPPORT)")
else()
  message(STATUS "Federated TLS: OpenSSL disabled (default); pass -DCYPHA_ENABLE_OPENSSL=ON to enable HTTPS")
endif()

add_executable(cypha_onnx_export tools/cypha_onnx_export.cpp)
target_link_libraries(cypha_onnx_export PRIVATE cypha_core)

add_executable(cypha_export_gguf tools/cypha_export_gguf.cpp)
target_link_libraries(cypha_export_gguf PRIVATE cypha_core)

add_executable(intelligence_profiler_smoke tools/intelligence_profiler_smoke.cpp)
target_link_libraries(intelligence_profiler_smoke PRIVATE cypha_core)

add_executable(ewc_smoke tools/ewc_smoke.cpp)
target_link_libraries(ewc_smoke PRIVATE cypha_core)

add_executable(curriculum_smoke tools/curriculum_smoke.cpp)
target_link_libraries(curriculum_smoke PRIVATE cypha_core)

add_executable(corpus_smoke tools/corpus_smoke.cpp)
target_link_libraries(corpus_smoke PRIVATE cypha_lm_native)

add_executable(intelligence_profiler_papers tools/intelligence_profiler_papers.cpp)
target_link_libraries(intelligence_profiler_papers PRIVATE cypha_core)

add_executable(cypha_intelligence_bench tools/cypha_intelligence_bench.cpp)
target_link_libraries(cypha_intelligence_bench PRIVATE cypha_core cypha_bench_native)

add_executable(cypha_cell_hypothesis_sweep tools/cypha_cell_hypothesis_sweep.cpp)
target_link_libraries(cypha_cell_hypothesis_sweep PRIVATE cypha_lm_native)
cypha_mingw_apply_exe_link_options(cypha_cell_hypothesis_sweep)

add_executable(cypha_baseline_lock tools/cypha_baseline_lock.cpp)
target_link_libraries(cypha_baseline_lock PRIVATE cypha_bench_native)
cypha_mingw_apply_exe_link_options(cypha_baseline_lock)

add_executable(baseline_lock_validate tools/baseline_lock_smoke.cpp)
target_include_directories(baseline_lock_validate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
cypha_mingw_apply_exe_link_options(baseline_lock_validate)

set(
  _cypha_mingw_exe_targets
  ${CYPHA_PARITY_EXE_TARGETS}
  cypha_fixture_gen
  xor_kernel_bench
  cypha_kernel_tune
  kernel_snapshot_roundtrip
  kernel_cypha_roundtrip
  memory_train_roundtrip
  preprocessor_rff_gamma_cv_test
  ${CYPHA_APP_EXE_TARGETS}
  registry_register
  rest_schema_contract
  cuda_smoke
  create_model_smoke
  cypha_federated_merge
  cypha_federated_coordinator
  cypha_federated_worker
  cypha_onnx_export
  cypha_export_gguf
  intelligence_profiler_smoke
  ewc_smoke
  ewc_d16b_smoke
  ewc_cyphalm_smoke
  ewc_hybrid_smoke
  ewc_weights_smoke
  native_sr_gate_laws_smoke
  federated_worker_smoke
  federated_tls_smoke
  curriculum_smoke
  corpus_smoke
  intelligence_profiler_papers
  cyphalm_bench_native
  cyphalm_ssm_diagnose
  cypha_cell_hypothesis_sweep
  cypha_baseline_lock
  baseline_lock_validate
)
foreach(_cypha_t IN LISTS _cypha_mingw_exe_targets)
  cypha_mingw_apply_exe_link_options("${_cypha_t}")
endforeach()

# M6: SQLite for experiment_db_smoke — prefer system SQLite3; else optional official amalgamation (no apt/vcpkg).
find_package(SQLite3 QUIET)
option(
  CYPHA_FETCH_SQLITE3_AMALGAMATION
  "If SQLite3 is not found, download the official amalgamation for experiment_db_smoke (requires network at configure)"
  ON
)

set(_cypha_exp_sqlite_link "")
if(SQLite3_FOUND)
  set(_cypha_exp_sqlite_link SQLite::SQLite3)
elseif(CYPHA_FETCH_SQLITE3_AMALGAMATION)
  include(FetchContent)
  FetchContent_Declare(
    sqlite_amalgamation
    URL https://sqlite.org/2024/sqlite-amalgamation-3470200.zip
    URL_HASH SHA256=aa73d8748095808471deaa8e6f34aa700e37f2f787f4425744f53fdd15a89c40
    SOURCE_SUBDIR sqlite-amalgamation-3470200
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(sqlite_amalgamation)
  add_library(cypha_sqlite3_amalg STATIC "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c")
  target_include_directories(cypha_sqlite3_amalg PUBLIC "${sqlite_amalgamation_SOURCE_DIR}")
  if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(cypha_sqlite3_amalg PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
  endif()
  set(_cypha_exp_sqlite_link cypha_sqlite3_amalg)
  message(STATUS "experiment_db_smoke: using fetched SQLite amalgamation 3.47.2")
endif()

set(_cypha_experiment_ddl_sql "${CMAKE_CURRENT_SOURCE_DIR}/sql/experiment_schema.sql")
if(NOT EXISTS "${_cypha_experiment_ddl_sql}")
  message(FATAL_ERROR "Missing canonical experiment DDL: ${_cypha_experiment_ddl_sql}")
endif()

if(CYPHA_BUILD_EXPERIMENT_DB AND _cypha_exp_sqlite_link)
  add_executable(experiment_db_smoke tools/experiment_db_smoke.cpp src/experiment_db.cpp)
  target_link_libraries(experiment_db_smoke PRIVATE ${_cypha_exp_sqlite_link})
  target_include_directories(experiment_db_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  cypha_mingw_apply_exe_link_options(experiment_db_smoke)
  if(MSVC)
    target_compile_options(experiment_db_smoke PRIVATE /W4 /permissive-)
  else()
    target_compile_options(experiment_db_smoke PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_executable(
    experiment_db_crud_parity
    tests/parity/experiment_db_crud_parity.cpp
    src/experiment_db.cpp
    src/experiment_db_crud.cpp
  )
  target_link_libraries(experiment_db_crud_parity PRIVATE ${_cypha_exp_sqlite_link})
  target_include_directories(experiment_db_crud_parity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  cypha_mingw_apply_exe_link_options(experiment_db_crud_parity)
  if(MSVC)
    target_compile_options(experiment_db_crud_parity PRIVATE /W4 /permissive-)
  else()
    target_compile_options(experiment_db_crud_parity PRIVATE -Wall -Wextra -Wpedantic)
  endif()
elseif(CYPHA_BUILD_EXPERIMENT_DB AND NOT _cypha_exp_sqlite_link)
  message(
    STATUS
    "experiment_db_smoke: skipped (no SQLite3: install dev package, or leave CYPHA_FETCH_SQLITE3_AMALGAMATION=ON)"
  )
endif()

if(MSVC)
  target_compile_options(cypha_core PRIVATE /W4 /permissive-)
  target_compile_options(cypha_parity PRIVATE /W4 /permissive-)
  target_compile_options(cypha_parity_run PRIVATE /W4 /permissive-)
  target_compile_options(batch_llr_parity PRIVATE /W4 /permissive-)
  target_compile_options(rpsm_batched_llr_smoke PRIVATE /W4 /permissive-)
  target_compile_options(rpsm_hierarchy_smoke PRIVATE /W4 /permissive-)
  target_compile_options(rpsm_train_smoke PRIVATE /W4 /permissive-)
  target_compile_options(cypha_fixture_gen PRIVATE /W4 /permissive-)
  target_compile_options(score_batch_parity PRIVATE /W4 /permissive-)
  target_compile_options(kernel_llr_parity PRIVATE /W4 /permissive-)
  target_compile_options(xor_kernel_bench PRIVATE /W4 /permissive-)
  target_compile_options(cypha_kernel_tune PRIVATE /W4 /permissive-)
  target_compile_options(kernel_snapshot_roundtrip PRIVATE /W4 /permissive-)
  target_compile_options(kernel_cypha_roundtrip PRIVATE /W4 /permissive-)
  target_compile_options(memory_train_parity PRIVATE /W4 /permissive-)
  target_compile_options(multilabel_dif_parity PRIVATE /W4 /permissive-)
  target_compile_options(merge_from_parity PRIVATE /W4 /permissive-)
  target_compile_options(similarity_index_parity PRIVATE /W4 /permissive-)
  target_compile_options(memory_train_roundtrip PRIVATE /W4 /permissive-)
  target_compile_options(preprocessor_parity PRIVATE /W4 /permissive-)
  target_compile_options(preprocessor_fit_parity PRIVATE /W4 /permissive-)
  target_compile_options(preprocessor_rff_gamma_cv_test PRIVATE /W4 /permissive-)
  target_compile_options(csv_ingest_parity PRIVATE /W4 /permissive-)
  target_compile_options(preprocess_train_classify_parity PRIVATE /W4 /permissive-)
  target_compile_options(cypha_rest PRIVATE /W4 /permissive-)
  target_compile_options(registry_register PRIVATE /W4 /permissive-)
  target_compile_options(nig_adapt_parity PRIVATE /W4 /permissive-)
  target_compile_options(train_step_vector_parity PRIVATE /W4 /permissive-)
  target_compile_options(dif_regressor_train_step_parity PRIVATE /W4 /permissive-)
  target_compile_options(regression_mixture_parity PRIVATE /W4 /permissive-)
  target_compile_options(regression_m4_parity PRIVATE /W4 /permissive-)
  target_compile_options(regression_rff_parity PRIVATE /W4 /permissive-)
  target_compile_options(regression_two_stage_pipeline_parity PRIVATE /W4 /permissive-)
  target_compile_options(regression_two_stage_ridge_fit_parity PRIVATE /W4 /permissive-)
  target_compile_options(quantile_dif_train_parity PRIVATE /W4 /permissive-)
  target_compile_options(mke_train_step_parity PRIVATE /W4 /permissive-)
  target_compile_options(generation_parity PRIVATE /W4 /permissive-)
  target_compile_options(gh_infer_deliberation_parity PRIVATE /W4 /permissive-)
  target_compile_options(embed_table_parity PRIVATE /W4 /permissive-)
  target_compile_options(retrieval_parity PRIVATE /W4 /permissive-)
  target_compile_options(cuda_smoke PRIVATE /W4 /permissive-)
  target_compile_options(cypha_lm_native PRIVATE /W4 /permissive-)
  target_compile_options(cypha_bench_native PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_ssm_diagnose PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_hebbian_parity PRIVATE /W4 /permissive-)
  target_compile_options(som_parity PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_model_parity PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_char_lstm_parity PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_checkpoint_parity PRIVATE /W4 /permissive-)
  target_compile_options(cypha_bench_run PRIVATE /W4 /permissive-)
  target_compile_options(cypha_tune_run PRIVATE /W4 /permissive-)
  target_compile_options(cypha_diagnostics_run PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_train PRIVATE /W4 /permissive-)
  target_compile_options(cypha_onnx_export PRIVATE /W4 /permissive-)
  target_compile_options(cypha_export_gguf PRIVATE /W4 /permissive-)
else()
  target_compile_options(cypha_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_lm_native PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_bench_native PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_hebbian_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(som_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_model_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_bench_native PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_ssm_diagnose PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_bench_run PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_tune_run PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_diagnostics_run PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_train PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_parity_run PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(batch_llr_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(rpsm_batched_llr_smoke PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(rpsm_hierarchy_smoke PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(rpsm_train_smoke PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_fixture_gen PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(score_batch_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(kernel_llr_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(xor_kernel_bench PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_kernel_tune PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(kernel_snapshot_roundtrip PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(kernel_cypha_roundtrip PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(memory_train_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(multilabel_dif_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(merge_from_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(similarity_index_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(memory_train_roundtrip PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(preprocessor_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(preprocessor_fit_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(preprocessor_rff_gamma_cv_test PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(csv_ingest_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(preprocess_train_classify_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_rest PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(registry_register PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(nig_adapt_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(train_step_vector_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(dif_regressor_train_step_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(regression_mixture_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(regression_m4_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(regression_rff_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(regression_two_stage_pipeline_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(regression_two_stage_ridge_fit_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(quantile_dif_train_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(mke_train_step_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(generation_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(gh_infer_deliberation_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(embed_table_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(retrieval_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cuda_smoke PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_onnx_export PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_export_gguf PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_char_lstm_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_checkpoint_parity PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Parity fixture paths: MinGW .exe run on Windows cannot open /mnt/c/... (WSL cross-build).
set(_parity_fix "${CMAKE_SOURCE_DIR}/../fixtures")
cmake_path(ABSOLUTE_PATH _parity_fix NORMALIZE)
cypha_mingw_fix_parity_path_for_cross_host(_parity_fix)

enable_testing()
add_test(
  NAME native_parity
  COMMAND "$<TARGET_FILE:cypha_parity>"
    "${_parity_fix}/reference.cypha"
    "${_parity_fix}/native_parity.bin"
)
add_test(
  NAME native_parity_run_list
  COMMAND "$<TARGET_FILE:cypha_parity_run>" --list
)
add_test(
  NAME native_gh_infer_deliberation
  COMMAND "$<TARGET_FILE:gh_infer_deliberation_parity>"
    "${_parity_fix}/gh_infer_deliberation"
)
add_test(
  NAME native_batch_llr
  COMMAND "$<TARGET_FILE:batch_llr_parity>"
    "${_parity_fix}/batch_llr/sidecar.json"
)
add_test(
  NAME native_rpsm_batched_llr_smoke
  COMMAND "$<TARGET_FILE:rpsm_batched_llr_smoke>"
    "${_parity_fix}/reference.cypha"
    "${_parity_fix}/native_parity.bin"
)
add_test(
  NAME native_rpsm_sequence_smoke
  COMMAND "$<TARGET_FILE:rpsm_sequence_smoke>"
)
add_test(
  NAME native_rpsm_hierarchy_smoke
  COMMAND "$<TARGET_FILE:rpsm_hierarchy_smoke>"
)
add_test(
  NAME native_rpsm_train_smoke
  COMMAND "$<TARGET_FILE:rpsm_train_smoke>"
)
add_test(
  NAME native_fixture_gen_list
  COMMAND "$<TARGET_FILE:cypha_fixture_gen>" --list
)
set_tests_properties(
  native_fixture_gen_list
  PROPERTIES
    PASS_REGULAR_EXPRESSION
      "batch_llr[\t ]+;memory_train[\t ]+;preprocessor[\t ]+;train_step_vector[\t ]+;quantile_dif_train[\t ]+;regression_m4[\t ]+;score_batch[\t ]+;kernel_llr[\t ]+;gh_infer_deliberation[\t ]+;generation[\t ]+;embed_table[\t ]+;retrieval[\t ]+;multilabel_dif[\t ]+;merge_from[\t ]+;similarity_index[\t ]+;dif_regressor_train_step[\t ]+;mke_train_step[\t ]+;mke_train_extended[\t ]+;dif_train_replay[\t ]+;studio_trainer_classify_hotpath[\t ]+;studio_trainer_gh_classify_hotpath[\t ]+;rff_regression[\t ]+;two_stage_pipeline[\t ]+;two_stage_ridge_fit[\t ]+;two_stage_e2e_ridge[\t ]+;csv_preprocess_classify_hotpath[\t ]+;studio_trainer_preprocess_classify_hotpath[\t ]+;studio_trainer_preprocess_gh_classify_hotpath[\t ]+;cyphalm_char_lstm[\t ]+;cyphalm_checkpoint_char_lstm[\t ]+;cyphalm_checkpoint_hybrid[\t ]+;cyphalm_ssm[\t ]+"
)
add_test(
  NAME native_score_batch
  COMMAND "$<TARGET_FILE:score_batch_parity>"
    "${_parity_fix}/score_batch/sidecar.json"
)
add_test(
  NAME native_kernel_llr
  COMMAND "$<TARGET_FILE:kernel_llr_parity>"
    "${_parity_fix}/kernel_llr/sidecar.json"
)
add_test(
  NAME native_xor_kernel_bench_smoke
  COMMAND "$<TARGET_FILE:xor_kernel_bench>" --seeds 1 --passes 2 --kernel-blend 1.0
    --kernel-m 256 --gamma-scale 2.0 --kernel-lr-scale 2.0 --kernel-xor-features
)
set_tests_properties(native_xor_kernel_bench_smoke PROPERTIES
  PASS_REGULAR_EXPRESSION "\"kernel_mean_acc\": 0\\.[7-9]")
add_test(
  NAME native_kernel_snapshot_roundtrip
  COMMAND "$<TARGET_FILE:kernel_snapshot_roundtrip>"
)
add_test(
  NAME native_kernel_cypha_roundtrip
  COMMAND "$<TARGET_FILE:kernel_cypha_roundtrip>"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_cypha_roundtrip_out.cypha"
)
add_test(
  NAME native_memory_train
  COMMAND "$<TARGET_FILE:memory_train_parity>"
    "${_parity_fix}/memory_train"
)
add_test(
  NAME native_multilabel_dif
  COMMAND "$<TARGET_FILE:multilabel_dif_parity>"
    "${_parity_fix}/multilabel_dif"
)
add_test(
  NAME native_merge_from
  COMMAND "$<TARGET_FILE:merge_from_parity>"
    "${_parity_fix}/merge_from"
)
add_test(
  NAME native_similarity_index
  COMMAND "$<TARGET_FILE:similarity_index_parity>"
    "${_parity_fix}/similarity_index"
)
add_test(
  NAME native_memory_train_roundtrip
  COMMAND "$<TARGET_FILE:memory_train_roundtrip>"
    "${_parity_fix}/memory_train"
    "${CMAKE_CURRENT_BINARY_DIR}/memory_train_roundtrip_out.cypha"
)
add_test(
  NAME native_preprocessor
  COMMAND "$<TARGET_FILE:preprocessor_parity>"
    "${_parity_fix}/preprocessor"
)
add_test(
  NAME native_preprocessor_fit
  COMMAND "$<TARGET_FILE:preprocessor_fit_parity>"
    "${_parity_fix}/preprocessor_fit"
    "${_parity_fix}/preprocessor_fit_no_scale"
    "${_parity_fix}/preprocessor_fit_rff"
)
add_test(NAME native_preprocessor_rff_gamma_cv COMMAND "$<TARGET_FILE:preprocessor_rff_gamma_cv_test>")
add_test(
  NAME native_csv_ingest
  COMMAND "$<TARGET_FILE:csv_ingest_parity>"
    "${_parity_fix}/csv_ingest"
)
add_test(
  NAME native_studio_trainer_preprocess_classify_hotpath
  COMMAND "$<TARGET_FILE:preprocess_train_classify_parity>"
    "${_parity_fix}/studio_trainer_preprocess_classify_hotpath"
)
add_test(
  NAME native_studio_trainer_preprocess_gh_classify_hotpath
  COMMAND "$<TARGET_FILE:preprocess_train_classify_parity>"
    "${_parity_fix}/studio_trainer_preprocess_gh_classify_hotpath"
)
add_test(
  NAME native_csv_preprocess_classify_hotpath
  COMMAND "$<TARGET_FILE:preprocess_train_classify_parity>"
    "${_parity_fix}/csv_preprocess_classify_hotpath"
)
add_test(NAME native_nig_adapt COMMAND "$<TARGET_FILE:nig_adapt_parity>")
add_test(
  NAME native_train_step_vector
  COMMAND "$<TARGET_FILE:train_step_vector_parity>"
    "${_parity_fix}/train_step_vector"
)
add_test(
  NAME native_dif_regressor_train_step
  COMMAND "$<TARGET_FILE:dif_regressor_train_step_parity>"
    "${_parity_fix}/dif_regressor_train_step"
)
add_test(NAME native_regression_mixture COMMAND "$<TARGET_FILE:regression_mixture_parity>")
add_test(
  NAME native_regression_m4
  COMMAND "$<TARGET_FILE:regression_m4_parity>"
    "${_parity_fix}/regression_m4/sidecar.json"
)
add_test(
  NAME native_regression_rff
  COMMAND "$<TARGET_FILE:regression_rff_parity>"
    "${_parity_fix}/rff_regression/sidecar.json"
)
add_test(
  NAME native_regression_two_stage_pipeline
  COMMAND "$<TARGET_FILE:regression_two_stage_pipeline_parity>"
    "${_parity_fix}/two_stage_pipeline/sidecar.json"
)
add_test(
  NAME native_regression_two_stage_ridge_fit
  COMMAND "$<TARGET_FILE:regression_two_stage_ridge_fit_parity>"
    "${_parity_fix}/two_stage_ridge_fit/sidecar.json"
)
add_test(
  NAME native_regression_two_stage_e2e_ridge
  COMMAND "$<TARGET_FILE:regression_two_stage_ridge_fit_parity>"
    "${_parity_fix}/two_stage_e2e_ridge/sidecar.json"
)
add_test(
  NAME native_quantile_dif_train
  COMMAND "$<TARGET_FILE:quantile_dif_train_parity>"
    "${_parity_fix}/quantile_dif_train"
)
add_test(
  NAME native_dif_train_replay
  COMMAND "$<TARGET_FILE:quantile_dif_train_parity>"
    "${_parity_fix}/dif_train_replay"
)
add_test(
  NAME native_studio_trainer_classify_hotpath
  COMMAND "$<TARGET_FILE:quantile_dif_train_parity>"
    "${_parity_fix}/studio_trainer_classify_hotpath"
)
add_test(
  NAME native_studio_trainer_gh_classify_hotpath
  COMMAND "$<TARGET_FILE:quantile_dif_train_parity>"
    "${_parity_fix}/studio_trainer_gh_classify_hotpath"
)
add_test(
  NAME native_mke_train_step
  COMMAND "$<TARGET_FILE:mke_train_step_parity>"
    "${_parity_fix}/mke_train_step"
)
add_test(
  NAME native_mke_train_extended
  COMMAND "$<TARGET_FILE:mke_train_step_parity>"
    "${_parity_fix}/mke_train_extended"
)
add_test(
  NAME native_generation
  COMMAND "$<TARGET_FILE:generation_parity>"
    "${_parity_fix}/generation/sidecar.json"
)

add_test(
  NAME native_embed_table
  COMMAND "$<TARGET_FILE:embed_table_parity>"
    "${_parity_fix}/embed_table/sidecar.json"
)
if(NOT EXISTS "${_parity_fix}/embed_table/sidecar.json")
  set_tests_properties(native_embed_table PROPERTIES DISABLED TRUE)
endif()

add_test(
  NAME native_retrieval
  COMMAND "$<TARGET_FILE:retrieval_parity>"
    "${_parity_fix}/retrieval/sidecar.json"
)
if(NOT EXISTS "${_parity_fix}/retrieval/sidecar.json")
  set_tests_properties(native_retrieval PROPERTIES DISABLED TRUE)
endif()

add_test(
  NAME native_cyphalm_char_lstm
  COMMAND "$<TARGET_FILE:cyphalm_char_lstm_parity>"
    "${_parity_fix}/cyphalm_char_lstm/sidecar.json"
)
if(NOT EXISTS "${_parity_fix}/cyphalm_char_lstm/sidecar.json")
  set_tests_properties(native_cyphalm_char_lstm PROPERTIES DISABLED TRUE)
endif()

add_test(
  NAME native_cyphalm_model_parity
  COMMAND "$<TARGET_FILE:cyphalm_model_parity>"
)

add_test(
  NAME native_cyphalm_checkpoint_parity
  COMMAND "$<TARGET_FILE:cyphalm_checkpoint_parity>"
    "${_parity_fix}/cyphalm_checkpoint/char_lstm/sidecar.json"
)
if(NOT EXISTS "${_parity_fix}/cyphalm_checkpoint/char_lstm/sidecar.json")
  set_tests_properties(native_cyphalm_checkpoint_parity PROPERTIES DISABLED TRUE)
endif()

add_test(
  NAME native_cyphalm_train_smoke
  COMMAND "$<TARGET_FILE:cyphalm_train>"
    --profile d04
    --epochs 1
    --synthetic-tokens 512
    --max-train-steps 128
    --out "${CMAKE_CURRENT_BINARY_DIR}/cyphalm_train_smoke_ckpt"
)

add_test(
  NAME native_bench_run_list_domains
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --list-domains
)

add_test(
  NAME native_cyphalm_ssm_diagnose_smoke
  COMMAND "$<TARGET_FILE:cyphalm_ssm_diagnose>" --domain synthetic --steps 64
)

add_test(
  NAME native_cyphalm_ssm_diagnose_d10_smoke
  COMMAND "$<TARGET_FILE:cyphalm_ssm_diagnose>" --domain d10 --steps 64
)

add_test(
  NAME native_tune_run_smoke
  COMMAND "$<TARGET_FILE:cypha_tune_run>"
    --config "${CMAKE_SOURCE_DIR}/../bench/config/cyphalm_hybrid_lstm_tune_smoke.json"
    --max-cells 2
)
set_tests_properties(native_tune_run_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_create_model
  COMMAND "$<TARGET_FILE:create_model_smoke>"
)

add_test(
  NAME native_onnx_export_help
  COMMAND "$<TARGET_FILE:cypha_onnx_export>" --help
)

add_test(
  NAME native_onnx_export_dry_run
  COMMAND "$<TARGET_FILE:cypha_onnx_export>"
    --cypha "${_parity_fix}/reference.cypha"
    --out "${CMAKE_CURRENT_BINARY_DIR}/onnx_export_smoke.onnx"
    --dry-run
)

add_test(
  NAME native_onnx_export_smoke
  COMMAND "$<TARGET_FILE:cypha_onnx_export>"
    --cypha "${_parity_fix}/reference.cypha"
    --out "${CMAKE_CURRENT_BINARY_DIR}/onnx_export_reference.onnx"
)

add_test(
  NAME native_export_gguf_help
  COMMAND "$<TARGET_FILE:cypha_export_gguf>" --help
)

add_test(
  NAME native_export_gguf_smoke
  COMMAND "$<TARGET_FILE:cypha_export_gguf>"
    --cypha "${_parity_fix}/reference.cypha"
    --out "${CMAKE_CURRENT_BINARY_DIR}/gguf_export_reference.gguf"
)

add_test(
  NAME native_intelligence_profiler_smoke
  COMMAND "$<TARGET_FILE:intelligence_profiler_smoke>"
)

add_test(
  NAME native_ewc_smoke
  COMMAND "$<TARGET_FILE:ewc_smoke>"
)

add_test(
  NAME native_ewc_d16b_smoke
  COMMAND "$<TARGET_FILE:ewc_d16b_smoke>"
)

add_test(
  NAME native_ewc_cyphalm_smoke
  COMMAND "$<TARGET_FILE:ewc_cyphalm_smoke>"
)

add_test(
  NAME native_ewc_hybrid_smoke
  COMMAND "$<TARGET_FILE:ewc_hybrid_smoke>"
)

add_test(
  NAME native_ewc_weights_smoke
  COMMAND "$<TARGET_FILE:ewc_weights_smoke>"
)

add_test(
  NAME native_sr_gate_laws_smoke
  COMMAND "$<TARGET_FILE:native_sr_gate_laws_smoke>"
)

add_test(
  NAME native_curriculum_smoke
  COMMAND "$<TARGET_FILE:curriculum_smoke>"
)

add_test(
  NAME native_corpus_smoke
  COMMAND "$<TARGET_FILE:corpus_smoke>"
)

add_test(
  NAME native_intelligence_profiler_papers
  COMMAND "$<TARGET_FILE:intelligence_profiler_papers>"
)
add_test(
  NAME native_intelligence_bench_smoke
  COMMAND "$<TARGET_FILE:cypha_intelligence_bench>"
)

add_test(
  NAME native_cyphalm_bench_intelligence_profile
  COMMAND "$<TARGET_FILE:cyphalm_bench_native>"
    --profile d17 --mode hybrid --n-train 120 --n-eval 40 --intelligence-profile
)
set_tests_properties(native_cyphalm_bench_intelligence_profile PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_d17_wikitext_smoke
  COMMAND "$<TARGET_FILE:cyphalm_bench_native>"
    --profile d17 --mode hybrid --n-train 512 --n-eval 64
)
set_tests_properties(native_d17_wikitext_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FULL_CORPUS=1;CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_d17_wikitext_overnight_smoke
  COMMAND "$<TARGET_FILE:cyphalm_bench_native>"
    --profile d17 --mode hybrid --overnight --n-train 500 --n-eval 64
)
set_tests_properties(native_d17_wikitext_overnight_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FULL_CORPUS=1;CYPHA_BENCH_OVERNIGHT=1;CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_overnight_mini_smoke
  COMMAND "$<TARGET_FILE:cyphalm_bench_native>"
    --profile d17 --mode hybrid --overnight --n-train 800 --n-eval 64
)
set_tests_properties(native_overnight_mini_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/baseline_lock_smoke.json"
  COPYONLY
)

add_test(
  NAME native_baseline_lock_smoke
  COMMAND "$<TARGET_FILE:cypha_baseline_lock>"
    --run d17 --fast --n-train 200
    --lock-file "${CMAKE_CURRENT_BINARY_DIR}/baseline_lock_smoke.json"
    --exe-dir "${CMAKE_CURRENT_BINARY_DIR}"
)
set_tests_properties(native_baseline_lock_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_baseline_lock_validate_smoke
  COMMAND "$<TARGET_FILE:baseline_lock_validate>"
    --lock-file "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
)

add_test(
  NAME native_baseline_lock_validate_production_status
  COMMAND "$<TARGET_FILE:baseline_lock_validate>"
    --lock-file "${CMAKE_SOURCE_DIR}/../fixtures/baseline_lock_production_status.json"
    --production
)

add_test(
  NAME native_federated_merge_smoke
  COMMAND "$<TARGET_FILE:cypha_federated_merge>"
    --merge
    "${CMAKE_SOURCE_DIR}/../fixtures/federated_merge/worker_a.json"
    "${CMAKE_SOURCE_DIR}/../fixtures/federated_merge/worker_b.json"
    --out "${CMAKE_CURRENT_BINARY_DIR}/federated_merge_smoke.json"
)

add_test(
  NAME native_federated_coordinator_smoke
  COMMAND "$<TARGET_FILE:cypha_federated_coordinator>"
    --watch-dir "${CMAKE_SOURCE_DIR}/../fixtures/federated_merge"
    --min-workers 2
    --once
    --out "${CMAKE_CURRENT_BINARY_DIR}/federated_coordinator_smoke.json"
)

add_test(
  NAME native_federated_worker_smoke
  COMMAND "$<TARGET_FILE:federated_worker_smoke>"
    "${CMAKE_SOURCE_DIR}/../fixtures/federated_merge/worker_a.json"
    "${CMAKE_SOURCE_DIR}/../fixtures/federated_merge/worker_b.json"
    "${CMAKE_CURRENT_BINARY_DIR}/federated_worker_smoke.json"
)

add_test(
  NAME native_federated_tls_smoke
  COMMAND "$<TARGET_FILE:federated_tls_smoke>"
    "${CMAKE_SOURCE_DIR}/../fixtures/federated_merge/worker_a.json"
    "${CMAKE_SOURCE_DIR}/../fixtures/federated_merge/worker_b.json"
    "${CMAKE_CURRENT_BINARY_DIR}/federated_tls_smoke.json"
)
set_tests_properties(native_federated_tls_smoke PROPERTIES SKIP_RETURN_CODE 2)

add_test(
  NAME native_cyphalm_bench_rpsm_smoke
  COMMAND "$<TARGET_FILE:cyphalm_bench_native>"
    --profile d17 --mode rpsm --n-train 100 --n-eval 32
)
set_tests_properties(native_cyphalm_bench_rpsm_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_d21_rpsm_smoke
  COMMAND "$<TARGET_FILE:cyphalm_bench_native>"
    --profile d21 --mode rpsm --overnight --n-train 500 --n-eval 64
)
set_tests_properties(native_d21_rpsm_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FULL_CORPUS=1;CYPHA_BENCH_OVERNIGHT=1;CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_d22_cross_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d22
)
set_tests_properties(native_d22_cross_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d23_overnight_lock_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d23_overnight_lock_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d23
)
set_tests_properties(native_d23_overnight_lock_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 120
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d24_production_lock_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d24_production_lock_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d24
)
set_tests_properties(native_d24_production_lock_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 180
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d26_medium_overnight_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d26_medium_overnight_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d26
)
set_tests_properties(native_d26_medium_overnight_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FULL_CORPUS=1"
  TIMEOUT 300
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d27_production_lock_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d27_production_lock_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d27
)
set_tests_properties(native_d27_production_lock_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 30
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d28_overnight_complete_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d28_overnight_complete_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d28
)
set_tests_properties(native_d28_overnight_complete_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 30
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d29_release_readiness_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d29_release_readiness_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d29
)
set_tests_properties(native_d29_release_readiness_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 60
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d30_artifact_hygiene_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d30_artifact_hygiene_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d30
)
set_tests_properties(native_d30_artifact_hygiene_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 30
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d31_post_overnight_pipeline_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d31_post_overnight_pipeline_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d31
)
set_tests_properties(native_d31_post_overnight_pipeline_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 120
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d32_production_complete_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d32_production_complete_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d32
)
set_tests_properties(native_d32_production_complete_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 60
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../bench/BASELINE_LOCK.json"
  "${CMAKE_CURRENT_BINARY_DIR}/d33_release_publish_smoke.json"
  COPYONLY
)

add_test(
  NAME native_d33_release_publish_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d33
)
set_tests_properties(native_d33_release_publish_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 60
)

add_test(
  NAME native_d25_corpus_smoke
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --domain-tag d25
)
set_tests_properties(native_d25_corpus_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
  TIMEOUT 60
)

add_test(
  NAME native_cell_hypothesis_sweep_smoke
  COMMAND "$<TARGET_FILE:cypha_cell_hypothesis_sweep>" --smoke
)

add_test(
  NAME native_cell_hypothesis_tier2_smoke
  COMMAND "$<TARGET_FILE:cypha_cell_hypothesis_sweep>" --tier2-smoke
)

add_test(
  NAME native_cell_hypothesis_tier3_smoke
  COMMAND "$<TARGET_FILE:cypha_cell_hypothesis_sweep>" --tier3-smoke
)

add_test(
  NAME native_cell_hypothesis_overnight_smoke
  COMMAND "$<TARGET_FILE:cypha_cell_hypothesis_sweep>"
    --overnight-sweep-smoke
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/cell_hypothesis_overnight_smoke"
)
set_tests_properties(native_cell_hypothesis_overnight_smoke PROPERTIES
  ENVIRONMENT "CYPHA_BENCH_FAST=1"
)

add_test(
  NAME native_diagnostics_run
  COMMAND "$<TARGET_FILE:cypha_diagnostics_run>"
    --fixtures "${_parity_fix}"
    --out "${CMAKE_CURRENT_BINARY_DIR}/diagnostics_results"
)

add_test(
  NAME native_cyphalm_hebbian
  COMMAND "$<TARGET_FILE:cyphalm_hebbian_parity>"
)

add_test(
  NAME native_som_parity
  COMMAND "$<TARGET_FILE:som_parity>"
)

add_test(NAME native_cyphalm_ssm COMMAND "$<TARGET_FILE:cyphalm_ssm_parity>")

if(EXISTS "${_parity_fix}/cyphalm_ssm/sidecar.json")
  add_test(
    NAME native_cyphalm_ssm_fixture
    COMMAND "$<TARGET_FILE:cyphalm_parity>" "${_parity_fix}/cyphalm_ssm/sidecar.json"
  )
endif()

add_test(
  NAME native_cyphalm_parity_suite
  COMMAND "$<TARGET_FILE:cyphalm_parity>"
)

add_test(
  NAME native_cuda_smoke
  COMMAND "$<TARGET_FILE:cuda_smoke>"
)

add_test(
  NAME native_cuda_bench
  COMMAND "$<TARGET_FILE:cuda_smoke>" --bench
)
set_tests_properties(native_cuda_bench PROPERTIES SKIP_RETURN_CODE 2)

add_test(
  NAME native_registry_register
  COMMAND "$<TARGET_FILE:registry_register>"
    "${CMAKE_CURRENT_BINARY_DIR}/registry_native_smoke_root"
    "native_reg_smoke"
    "0.0.1"
    "${_parity_fix}/reference.cypha"
    "${_parity_fix}/registry_register/card.json"
    --overwrite
    --and-verify
)

if(WIN32)
  add_test(
    NAME native_rest_multi_model
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
      -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_cypha_rest_multi_model.ps1"
      -BuildDir "${CMAKE_CURRENT_BINARY_DIR}"
  )
  set_tests_properties(native_rest_multi_model PROPERTIES RUN_SERIAL TRUE)
  add_test(
    NAME native_rest_uncertainty_rank
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
      -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_cypha_rest_uncertainty_rank.ps1"
      -BuildDir "${CMAKE_CURRENT_BINARY_DIR}"
  )
  set_tests_properties(native_rest_uncertainty_rank PROPERTIES RUN_SERIAL TRUE)
  add_test(
    NAME native_rest_ui_smoke
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
      -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_cypha_rest_ui_smoke.ps1"
      -BuildDir "${CMAKE_CURRENT_BINARY_DIR}"
  )
  set_tests_properties(native_rest_ui_smoke PROPERTIES RUN_SERIAL TRUE)
endif()

if(WIN32)
  add_test(
    NAME native_rest_schema_contract
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
      -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_rest_schema_contract.ps1"
      -BuildDir "${CMAKE_CURRENT_BINARY_DIR}"
  )
else()
  add_test(
    NAME native_rest_schema_contract
    COMMAND "$<TARGET_FILE:rest_schema_contract>"
      --cypha-rest "$<TARGET_FILE:cypha_rest>"
      --registry-register "$<TARGET_FILE:registry_register>"
      --repo-root "${CMAKE_CURRENT_SOURCE_DIR}/.."
      --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()
set_tests_properties(native_rest_schema_contract PROPERTIES RUN_SERIAL TRUE)

if(TARGET experiment_db_smoke)
  add_test(
    NAME native_experiment_db_smoke
    COMMAND "$<TARGET_FILE:experiment_db_smoke>" "${_cypha_experiment_ddl_sql}"
  )
  add_test(
    NAME native_experiment_db_file
    COMMAND "$<TARGET_FILE:experiment_db_smoke>" "${_cypha_experiment_ddl_sql}"
      "${CMAKE_CURRENT_BINARY_DIR}/experiment_db_native_roundtrip.sqlite"
  )
endif()

if(TARGET experiment_db_crud_parity)
  add_test(
    NAME native_experiment_db_crud
    COMMAND "$<TARGET_FILE:experiment_db_crud_parity>" "${_cypha_experiment_ddl_sql}"
  )
endif()

if(CYPHA_BUILD_QT)
  add_subdirectory(qt)
endif()

if(TARGET cypha_qt_stub)
  add_test(
    NAME native_qt_stub_load_reference
    COMMAND "$<TARGET_FILE:cypha_qt_stub>" "${_parity_fix}/reference.cypha"
  )
  set_tests_properties(
    native_qt_stub_load_reference
    PROPERTIES
      ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
  )
endif()

if(TARGET cypha_qt_shell)
  add_test(
    NAME native_qt_shell_smoke
    COMMAND "$<TARGET_FILE:cypha_qt_shell>" --smoke "${_parity_fix}/reference.cypha"
  )
  set_tests_properties(
    native_qt_shell_smoke
    PROPERTIES
      ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
  )
endif()
