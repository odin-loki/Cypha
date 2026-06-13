cmake_minimum_required(VERSION 3.19)
project(cypha_native VERSION 2.2.8 LANGUAGES CXX C)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CyphaMinGW.cmake")

option(CYPHA_BUILD_EXPERIMENT_DB "Build experiment_db_smoke / experiment_db_crud_parity (requires SQLite3)" OFF)
option(
  CYPHA_CHECK_EXPERIMENT_SCHEMA_DRIFT
  "At configure time, compare native/sql/experiment_schema.sql to scripts/export_experiment_schema_sql.py (requires Python 3)"
  OFF
)
option(CYPHA_BUILD_QT "Build optional Qt targets under qt/: cypha_qt_stub (Core), cypha_qt_shell (Widgets)" OFF)
option(CYPHA_QT_CHARTS "When CYPHA_BUILD_QT is ON, link Qt Charts for cypha_qt_shell if found (default OFF for qt6-base-only CI)" OFF)
option(CYPHA_ENABLE_CUDA "Build CUDA GPU path in accel backend (NVIDIA toolkit + driver); CPU uses std::thread" OFF)
set(CYPHA_ACCEL_GPU_MIN_BATCH_ROWS "16" CACHE STRING "Minimum batch row count (n) to use CUDA in cypha::accel (smaller batches use CPU threads)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CYPHA_ENABLE_CUDA)
  if(MINGW)
    message(FATAL_ERROR "CYPHA_ENABLE_CUDA is not supported for MinGW targets (use native MSVC or Linux + nvcc).")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
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
  src/numpy_default_rng.cpp
  src/csv_ingest.cpp
  src/memory_train.cpp
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
  src/bench/bench_paths.cpp
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

add_executable(cyphalm_model_parity tools/cyphalm_model_parity.cpp)
target_link_libraries(cyphalm_model_parity PRIVATE cypha_lm_native)

add_executable(cyphalm_bench_native tools/cyphalm_bench_native.cpp)
target_link_libraries(cyphalm_bench_native PRIVATE cypha_lm_native)
cypha_mingw_apply_exe_link_options(cyphalm_bench_native)

add_executable(cypha_bench_run tools/cypha_bench_run.cpp)
target_link_libraries(cypha_bench_run PRIVATE cypha_bench_native)
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    cypha_zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.3.1
    GIT_SHALLOW TRUE
  )
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(cypha_zlib)
  if(TARGET zlibstatic)
    add_library(CyphaZlib::Zlib ALIAS zlibstatic)
  elseif(TARGET zlib)
    add_library(CyphaZlib::Zlib ALIAS zlib)
  else()
    message(FATAL_ERROR "FetchContent zlib did not define zlibstatic or zlib")
  endif()
  target_link_libraries(cypha_bench_run PRIVATE CyphaZlib::Zlib)
else()
  target_link_libraries(cypha_bench_run PRIVATE ZLIB::ZLIB)
endif()
cypha_mingw_apply_exe_link_options(cypha_bench_run)

add_executable(cypha_bench_report tools/cypha_bench_report.cpp)
target_link_libraries(cypha_bench_report PRIVATE cypha_bench_native)
cypha_mingw_apply_exe_link_options(cypha_bench_report)

add_executable(cypha_tune_run tools/cypha_tune_run.cpp)
target_link_libraries(cypha_tune_run PRIVATE cypha_bench_native)
cypha_mingw_apply_exe_link_options(cypha_tune_run)

target_include_directories(cypha_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_compile_definitions(cypha_core PRIVATE
  CYPHA_ACCEL_GPU_MIN_BATCH_ROWS=${CYPHA_ACCEL_GPU_MIN_BATCH_ROWS}
)

add_executable(cyphalm_ssm_parity tools/cyphalm_ssm_parity.cpp)
target_link_libraries(cyphalm_ssm_parity PRIVATE cypha_lm_native)

add_executable(cypha_parity tools/parity_main.cpp)
target_link_libraries(cypha_parity PRIVATE cypha_core)

add_executable(batch_llr_parity tools/batch_llr_parity.cpp)
target_link_libraries(batch_llr_parity PRIVATE cypha_core)

add_executable(score_batch_parity tools/score_batch_parity.cpp)
target_link_libraries(score_batch_parity PRIVATE cypha_core)

add_executable(kernel_llr_parity tools/kernel_llr_parity.cpp)
target_link_libraries(kernel_llr_parity PRIVATE cypha_core)

add_executable(memory_train_parity tools/memory_train_parity.cpp)
target_link_libraries(memory_train_parity PRIVATE cypha_core)

add_executable(multilabel_dif_parity tools/multilabel_dif_parity.cpp)
target_link_libraries(multilabel_dif_parity PRIVATE cypha_core)

add_executable(merge_from_parity tools/merge_from_parity.cpp)
target_link_libraries(merge_from_parity PRIVATE cypha_core)

add_executable(similarity_index_parity tools/similarity_index_parity.cpp)
target_link_libraries(similarity_index_parity PRIVATE cypha_core)

add_executable(memory_train_roundtrip tools/memory_train_roundtrip.cpp)
target_link_libraries(memory_train_roundtrip PRIVATE cypha_core)

add_executable(preprocessor_parity tools/preprocessor_parity.cpp)
target_link_libraries(preprocessor_parity PRIVATE cypha_core)

add_executable(preprocessor_fit_parity tools/preprocessor_fit_parity.cpp)
target_link_libraries(preprocessor_fit_parity PRIVATE cypha_core)

add_executable(csv_ingest_parity tools/csv_ingest_parity.cpp)
target_link_libraries(csv_ingest_parity PRIVATE cypha_core)

add_executable(preprocess_train_classify_parity tools/preprocess_train_classify_parity.cpp)
target_link_libraries(preprocess_train_classify_parity PRIVATE cypha_core)

add_executable(cypha_rest tools/cypha_rest.cpp tools/cyphalm_rest_routes.cpp tools/branch_a_rest_routes.cpp tools/dif_rest_routes.cpp)
target_link_libraries(cypha_rest PRIVATE cypha_core cypha_lm_native)

add_executable(registry_register tools/registry_register.cpp)
target_link_libraries(registry_register PRIVATE cypha_core)

add_executable(nig_adapt_parity tools/nig_adapt_parity.cpp)
target_link_libraries(nig_adapt_parity PRIVATE cypha_core)

add_executable(train_step_vector_parity tools/train_step_vector_parity.cpp)
target_link_libraries(train_step_vector_parity PRIVATE cypha_core)

add_executable(dif_regressor_train_step_parity tools/dif_regressor_train_step_parity.cpp)
target_link_libraries(dif_regressor_train_step_parity PRIVATE cypha_core)

add_executable(regression_mixture_parity tools/regression_mixture_parity.cpp)
target_link_libraries(regression_mixture_parity PRIVATE cypha_core)

add_executable(regression_m4_parity tools/regression_m4_parity.cpp)
target_link_libraries(regression_m4_parity PRIVATE cypha_core)

add_executable(regression_rff_parity tools/regression_rff_parity.cpp)
target_link_libraries(regression_rff_parity PRIVATE cypha_core)

add_executable(regression_two_stage_pipeline_parity tools/regression_two_stage_pipeline_parity.cpp)
target_link_libraries(regression_two_stage_pipeline_parity PRIVATE cypha_core)

add_executable(regression_two_stage_ridge_fit_parity tools/regression_two_stage_ridge_fit_parity.cpp)
target_link_libraries(regression_two_stage_ridge_fit_parity PRIVATE cypha_core)

add_executable(quantile_dif_train_parity tools/quantile_dif_train_parity.cpp)
target_link_libraries(quantile_dif_train_parity PRIVATE cypha_core)

add_executable(mke_train_step_parity tools/mke_train_step_parity.cpp)
target_link_libraries(mke_train_step_parity PRIVATE cypha_core)

add_executable(generation_parity tools/generation_parity.cpp)
target_link_libraries(generation_parity PRIVATE cypha_core)

add_executable(gh_infer_deliberation_parity tools/gh_infer_deliberation_parity.cpp)
target_link_libraries(gh_infer_deliberation_parity PRIVATE cypha_core)

add_executable(embed_table_parity tools/embed_table_parity.cpp)
target_link_libraries(embed_table_parity PRIVATE cypha_lm_native)

add_executable(retrieval_parity tools/retrieval_parity.cpp)
target_link_libraries(retrieval_parity PRIVATE cypha_core)

add_executable(cuda_smoke tools/cuda_smoke.cpp)
target_link_libraries(cuda_smoke PRIVATE cypha_core)

add_executable(create_model_smoke tools/create_model_smoke.cpp)
target_link_libraries(create_model_smoke PRIVATE cypha_core)

add_executable(intelligence_profiler_smoke tools/intelligence_profiler_smoke.cpp)
target_link_libraries(intelligence_profiler_smoke PRIVATE cypha_core)

add_executable(cypha_diagnostics_run tools/cypha_diagnostics_run.cpp)
target_link_libraries(cypha_diagnostics_run PRIVATE cypha_core)

add_executable(cyphalm_hebbian_parity tools/cyphalm_hebbian_parity.cpp)
target_link_libraries(cyphalm_hebbian_parity PRIVATE cypha_lm_native)

add_executable(som_parity tools/som_parity.cpp)
target_link_libraries(som_parity PRIVATE cypha_lm_native)

add_executable(cyphalm_char_lstm_parity tools/cyphalm_char_lstm_parity.cpp)
target_link_libraries(cyphalm_char_lstm_parity PRIVATE cypha_lm_native)

add_executable(cyphalm_checkpoint_parity tools/cyphalm_checkpoint_parity.cpp)
target_link_libraries(cyphalm_checkpoint_parity PRIVATE cypha_lm_native)

add_executable(cyphalm_parity tools/cyphalm_parity.cpp)
target_link_libraries(cyphalm_parity PRIVATE cypha_lm_native)

if(WIN32)
  target_link_libraries(cypha_rest PRIVATE ws2_32)
endif()

set(
  _cypha_mingw_exe_targets
  cyphalm_ssm_parity
  cypha_parity
  gh_infer_deliberation_parity
  embed_table_parity
  retrieval_parity
  batch_llr_parity
  score_batch_parity
  kernel_llr_parity
  memory_train_parity
  multilabel_dif_parity
  merge_from_parity
  similarity_index_parity
  memory_train_roundtrip
  preprocessor_parity
  preprocessor_fit_parity
  csv_ingest_parity
  preprocess_train_classify_parity
  cypha_rest
  registry_register
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
  gh_infer_deliberation_parity
  cuda_smoke
  create_model_smoke
  intelligence_profiler_smoke
  cyphalm_hebbian_parity
  som_parity
  cyphalm_char_lstm_parity
  cyphalm_checkpoint_parity
  cyphalm_model_parity
  cyphalm_bench_native
  cyphalm_parity
  cypha_bench_run
  cypha_bench_report
  cypha_tune_run
  cypha_diagnostics_run
)
foreach(_cypha_t IN LISTS _cypha_mingw_exe_targets)
  cypha_mingw_apply_exe_link_options("${_cypha_t}")
endforeach()

# M6: SQLite for experiment_db_smoke — prefer system SQLite3; else optional official amalgamation (no apt/vcpkg).
find_package(Python3 COMPONENTS Interpreter QUIET)
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

if(CYPHA_CHECK_EXPERIMENT_SCHEMA_DRIFT)
  if(NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "CYPHA_CHECK_EXPERIMENT_SCHEMA_DRIFT=ON requires Python3 Interpreter")
  endif()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/../scripts/export_experiment_schema_sql.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    OUTPUT_VARIABLE _cypha_exp_ddl_exported
    RESULT_VARIABLE _cypha_exp_ddl_rc
  )
  if(NOT _cypha_exp_ddl_rc EQUAL 0)
    message(FATAL_ERROR "scripts/export_experiment_schema_sql.py failed (exit ${_cypha_exp_ddl_rc})")
  endif()
  file(READ "${_cypha_experiment_ddl_sql}" _cypha_exp_ddl_committed)
  string(STRIP "${_cypha_exp_ddl_committed}" _cypha_exp_ddl_committed)
  string(STRIP "${_cypha_exp_ddl_exported}" _cypha_exp_ddl_exported)
  if(NOT _cypha_exp_ddl_committed STREQUAL _cypha_exp_ddl_exported)
    message(FATAL_ERROR
      "native/sql/experiment_schema.sql drifted from cypha_studio.core.experiment._SCHEMA; "
      "re-export with: python scripts/export_experiment_schema_sql.py "
      "-o native/sql/experiment_schema.sql"
    )
  endif()
  message(STATUS "experiment_schema.sql: matches Python _SCHEMA (drift check passed)")
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
    experiment_db_crud_parity tools/experiment_db_crud_parity.cpp src/experiment_db.cpp src/experiment_db_crud.cpp
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
  target_compile_options(batch_llr_parity PRIVATE /W4 /permissive-)
  target_compile_options(score_batch_parity PRIVATE /W4 /permissive-)
  target_compile_options(kernel_llr_parity PRIVATE /W4 /permissive-)
  target_compile_options(memory_train_parity PRIVATE /W4 /permissive-)
  target_compile_options(multilabel_dif_parity PRIVATE /W4 /permissive-)
  target_compile_options(merge_from_parity PRIVATE /W4 /permissive-)
  target_compile_options(similarity_index_parity PRIVATE /W4 /permissive-)
  target_compile_options(memory_train_roundtrip PRIVATE /W4 /permissive-)
  target_compile_options(preprocessor_parity PRIVATE /W4 /permissive-)
  target_compile_options(preprocessor_fit_parity PRIVATE /W4 /permissive-)
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
  target_compile_options(cyphalm_hebbian_parity PRIVATE /W4 /permissive-)
  target_compile_options(som_parity PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_model_parity PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_char_lstm_parity PRIVATE /W4 /permissive-)
  target_compile_options(cyphalm_checkpoint_parity PRIVATE /W4 /permissive-)
  target_compile_options(cypha_bench_run PRIVATE /W4 /permissive-)
  target_compile_options(cypha_tune_run PRIVATE /W4 /permissive-)
  target_compile_options(cypha_diagnostics_run PRIVATE /W4 /permissive-)
else()
  target_compile_options(cypha_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_lm_native PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_bench_native PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_hebbian_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(som_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_model_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_bench_native PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_bench_run PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_tune_run PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_diagnostics_run PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(batch_llr_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(score_batch_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(kernel_llr_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(memory_train_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(multilabel_dif_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(merge_from_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(similarity_index_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(memory_train_roundtrip PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(preprocessor_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(preprocessor_fit_parity PRIVATE -Wall -Wextra -Wpedantic)
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
  target_compile_options(cyphalm_char_lstm_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cyphalm_checkpoint_parity PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Parity fixture paths: MinGW .exe run on Windows cannot open /mnt/c/... (WSL cross-build).
set(_parity_fix "${CMAKE_SOURCE_DIR}/../parity_fixtures")
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
  NAME native_bench_run_list_domains
  COMMAND "$<TARGET_FILE:cypha_bench_run>" --list-domains
)

add_test(
  NAME native_tune_run_smoke
  COMMAND "$<TARGET_FILE:cypha_tune_run>"
    --config "${CMAKE_SOURCE_DIR}/../cypha_bench/config/cyphalm_hybrid_lstm_tune_smoke.json"
    --max-cells 2
)

add_test(
  NAME native_create_model
  COMMAND "$<TARGET_FILE:create_model_smoke>"
)

add_test(
  NAME native_intelligence_profiler_smoke
  COMMAND "$<TARGET_FILE:intelligence_profiler_smoke>"
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
