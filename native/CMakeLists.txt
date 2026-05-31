cmake_minimum_required(VERSION 3.19)
project(cypha_native VERSION 0.1.0 LANGUAGES CXX C)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CyphaMinGW.cmake")

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
  src/nig_gig_math.cpp
  src/bessel_table_data.cpp
  src/preprocessor.cpp
  src/preprocessor_fit.cpp
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
)

if(CYPHA_ENABLE_CUDA)
  target_sources(cypha_core PRIVATE src/accel_cuda.cu)
  target_compile_definitions(cypha_core PUBLIC CYPHA_ENABLE_CUDA)
  target_link_libraries(cypha_core PUBLIC CUDA::cudart)
  set_target_properties(cypha_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(cypha_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_compile_definitions(cypha_core PRIVATE
  CYPHA_ACCEL_GPU_MIN_BATCH_ROWS=${CYPHA_ACCEL_GPU_MIN_BATCH_ROWS}
)

add_executable(cypha_parity tools/parity_main.cpp)
target_link_libraries(cypha_parity PRIVATE cypha_core)

add_executable(batch_llr_parity tools/batch_llr_parity.cpp)
target_link_libraries(batch_llr_parity PRIVATE cypha_core)

add_executable(memory_train_parity tools/memory_train_parity.cpp)
target_link_libraries(memory_train_parity PRIVATE cypha_core)

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

add_executable(cypha_rest tools/cypha_rest.cpp)
target_link_libraries(cypha_rest PRIVATE cypha_core)

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

add_executable(cuda_smoke tools/cuda_smoke.cpp)
target_link_libraries(cuda_smoke PRIVATE cypha_core)

add_executable(create_model_smoke tools/create_model_smoke.cpp)
target_link_libraries(create_model_smoke PRIVATE cypha_core)

if(WIN32)
  target_link_libraries(cypha_rest PRIVATE ws2_32)
endif()

set(
  _cypha_mingw_exe_targets
  cypha_parity
  batch_llr_parity
  memory_train_parity
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
  cuda_smoke
  create_model_smoke
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
elseif(Python3_Interpreter_FOUND AND CYPHA_FETCH_SQLITE3_AMALGAMATION)
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

set(_cypha_experiment_ddl_sql "${CMAKE_CURRENT_BINARY_DIR}/experiment_ddl.sql")
if(Python3_Interpreter_FOUND AND _cypha_exp_sqlite_link)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/../scripts/export_experiment_schema_sql.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    OUTPUT_VARIABLE _cypha_exp_ddl_body
    RESULT_VARIABLE _cypha_exp_ddl_rc
  )
  if(NOT _cypha_exp_ddl_rc EQUAL 0)
    message(FATAL_ERROR "scripts/export_experiment_schema_sql.py failed (exit ${_cypha_exp_ddl_rc})")
  endif()
  file(WRITE "${_cypha_experiment_ddl_sql}" "${_cypha_exp_ddl_body}")

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
else()
  if(NOT Python3_Interpreter_FOUND)
    message(STATUS "experiment_db_smoke: skipped (Python3 Interpreter not found)")
  elseif(NOT _cypha_exp_sqlite_link)
    message(
      STATUS
      "experiment_db_smoke: skipped (no SQLite3: install dev package, or leave CYPHA_FETCH_SQLITE3_AMALGAMATION=ON)"
    )
  endif()
endif()

if(MSVC)
  target_compile_options(cypha_core PRIVATE /W4 /permissive-)
  target_compile_options(cypha_parity PRIVATE /W4 /permissive-)
  target_compile_options(batch_llr_parity PRIVATE /W4 /permissive-)
  target_compile_options(memory_train_parity PRIVATE /W4 /permissive-)
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
  target_compile_options(cuda_smoke PRIVATE /W4 /permissive-)
else()
  target_compile_options(cypha_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(batch_llr_parity PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(memory_train_parity PRIVATE -Wall -Wextra -Wpedantic)
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
  target_compile_options(cuda_smoke PRIVATE -Wall -Wextra -Wpedantic)
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
  NAME native_batch_llr
  COMMAND "$<TARGET_FILE:batch_llr_parity>"
    "${_parity_fix}/batch_llr/sidecar.json"
)
add_test(
  NAME native_memory_train
  COMMAND "$<TARGET_FILE:memory_train_parity>"
    "${_parity_fix}/memory_train"
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
  NAME native_create_model
  COMMAND "$<TARGET_FILE:create_model_smoke>"
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
