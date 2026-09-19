# hp (odin-loki/CompressionAlgorithm) integration for Cypha LLM path.
# Integer-exact context mixer — no float/double in hp/include or hp/src.
#
# Profiles (see cmake/HpFlags.cmake):
#   -DCYPHA_HP_PROFILE=light   (default) HP_SLOT_MAX=24, grow flags OFF
#   -DCYPHA_HP_PROFILE=champ    v78_flags.ps1 + HP_SLOT_MAX=35

set(CYPHA_HP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hp")

if(NOT EXISTS "${CYPHA_HP_ROOT}/include/hp/predictor.hpp")
  message(FATAL_ERROR "hp tree missing at ${CYPHA_HP_ROOT}")
endif()

target_include_directories(cypha_core PUBLIC
  "${CYPHA_HP_ROOT}/include"
  "${CYPHA_HP_ROOT}/third_party/xsimd/include")

target_compile_definitions(cypha_core PUBLIC CYPHA_LLM_ALGORITHM_HP=1)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HpFlags.cmake")
cypha_apply_hp_compile_flags(cypha_core)

# hp SIMD mixer dots (HP_XSIMD=1) require SSE4.1 + matching xsimd arch support.
# gate24/champ default ON (v82 recipe uses -msse4.1); light stays scalar unless forced.
option(CYPHA_HP_XSIMD "Enable hp xsimd SIMD mixer dots (requires SSE4.1)" OFF)
if(CYPHA_HP_PROFILE STREQUAL "gate24" OR CYPHA_HP_PROFILE STREQUAL "champ")
  if(NOT CYPHA_HP_XSIMD)
    set(CYPHA_HP_XSIMD ON)
    message(STATUS "CYPHA_HP_PROFILE=${CYPHA_HP_PROFILE}: enabling CYPHA_HP_XSIMD (v82 SIMD path)")
  endif()
endif()
if(CYPHA_HP_XSIMD)
  target_compile_definitions(cypha_core PUBLIC HP_XSIMD=1)
  if(NOT MSVC)
    # PUBLIC: consumers include hp headers that pull xsimd SSE4.1 batches.
    target_compile_options(cypha_core PUBLIC -msse4.1)
  endif()
else()
  target_compile_definitions(cypha_core PUBLIC HP_XSIMD=0)
endif()

set(CYPHA_CYPHALM_HP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/hp_backend.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_config.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_checkpoint.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_generation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_alpha_spectrum.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_views.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_corpus.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_parallel.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_math_integration.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cypha_cell_hypothesis.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/predictive_codec.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/adaptive_predictor_mixer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/arithmetic_coder.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/bpe_tokenizer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/npz_util.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/memory_policy.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/cyphalm_intelligence_hook.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cyphalm/text_algebraic_fingerprint.cpp"
)
