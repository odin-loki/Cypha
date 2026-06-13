# CyphaApps.cmake — production / orchestration entrypoints (sources under apps/).
# cypha_rest compiles route TU's from the same directory; binary name unchanged.

set(CYPHA_APPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/apps")
set(CYPHA_STATIC_UI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tools/static")

option(CYPHA_EMBED_STATIC_UI "Embed Studio Web UI in cypha_rest (no static/ beside binary)" OFF)

# --- cypha_rest: DIF + CyphaLM + Branch A route translation units ---
add_executable(
  cypha_rest
  "${CYPHA_APPS_DIR}/cypha_rest.cpp"
  "${CYPHA_APPS_DIR}/cyphalm_rest_routes.cpp"
  "${CYPHA_APPS_DIR}/branch_a_rest_routes.cpp"
  "${CYPHA_APPS_DIR}/dif_rest_routes.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/cypha_rest_static_ui.cpp"
)
target_include_directories(cypha_rest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools")
target_link_libraries(cypha_rest PRIVATE cypha_core cypha_lm_native)
if(WIN32)
  target_link_libraries(cypha_rest PRIVATE ws2_32)
endif()

set(_cypha_static_embed_h "${CMAKE_CURRENT_BINARY_DIR}/cypha_rest_static_embed.hpp")
if(CYPHA_EMBED_STATIC_UI)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_custom_command(
    OUTPUT "${_cypha_static_embed_h}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_static_ui.py"
      --static-dir "${CYPHA_STATIC_UI_DIR}"
      --out "${_cypha_static_embed_h}"
    DEPENDS
      "${CYPHA_STATIC_UI_DIR}/index.html"
      "${CYPHA_STATIC_UI_DIR}/app.js"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_static_ui.py"
    COMMENT "Generating embedded Studio Web UI for cypha_rest"
    VERBATIM
  )
  add_custom_target(cypha_rest_static_embed DEPENDS "${_cypha_static_embed_h}")
  add_dependencies(cypha_rest cypha_rest_static_embed)
  target_compile_definitions(cypha_rest PRIVATE CYPHA_EMBED_STATIC_UI)
  target_include_directories(cypha_rest PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
else()
  add_custom_command(
    TARGET cypha_rest
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CYPHA_STATIC_UI_DIR}"
      "$<TARGET_FILE_DIR:cypha_rest>/static"
    COMMENT "Copy Studio Web UI static/ next to cypha_rest"
    VERBATIM
  )
endif()

# --- bench / tune / diagnostics runners ---
add_executable(cypha_bench_run "${CYPHA_APPS_DIR}/cypha_bench_run.cpp")
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

add_executable(cypha_bench_report "${CYPHA_APPS_DIR}/cypha_bench_report.cpp")
target_link_libraries(cypha_bench_report PRIVATE cypha_bench_native)

add_executable(cypha_tune_run "${CYPHA_APPS_DIR}/cypha_tune_run.cpp")
target_link_libraries(cypha_tune_run PRIVATE cypha_bench_native)

add_executable(cypha_diagnostics_run "${CYPHA_APPS_DIR}/cypha_diagnostics_run.cpp")
target_link_libraries(cypha_diagnostics_run PRIVATE cypha_core)

add_executable(cyphalm_train "${CYPHA_APPS_DIR}/cyphalm_train.cpp")
target_link_libraries(cyphalm_train PRIVATE cypha_lm_native)

set(
  CYPHA_APP_EXE_TARGETS
  cypha_rest
  cypha_bench_run
  cypha_bench_report
  cypha_tune_run
  cypha_diagnostics_run
  cyphalm_train
)
