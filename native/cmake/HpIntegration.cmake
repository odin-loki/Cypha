# hp (odin-loki/CompressionAlgorithm) integration for Cypha LLM path.
# Integer-exact context mixer — no float/double in hp/include or hp/src.
#
# CyphaLM always builds gate24: v78_flags.ps1 + HP_SLOT_MAX=24 (see cmake/HpFlags.cmake).

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

# hp SIMD mixer dots (HP_XSIMD=1) use SSE4.1 batches in simd_dot.hpp (x86 only).
# gate24 default ON on x86; auto-fallback to scalar on macOS arm64 and other non-SSE hosts.
option(CYPHA_HP_XSIMD "Enable hp xsimd SIMD mixer dots (requires SSE4.1 on x86)" ON)

set(_cypha_hp_xsimd_effective ${CYPHA_HP_XSIMD})
if(_cypha_hp_xsimd_effective AND NOT MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-msse4.1" _CYPHA_HP_HAS_MSSE41)
  if(NOT _CYPHA_HP_HAS_MSSE41)
    set(_cypha_hp_xsimd_effective OFF)
    message(STATUS "CyphaLM hp: -msse4.1 unavailable; CYPHA_HP_XSIMD=OFF (scalar path, gate24 flags unchanged)")
  endif()
endif()

if(_cypha_hp_xsimd_effective)
  target_compile_definitions(cypha_core PUBLIC HP_XSIMD=1)
  if(MSVC)
    # hp/simd_dot.hpp gates on __SSE4_1__; MSVC does not define it unless /arch:SSE4.2+.
    target_compile_options(cypha_core PUBLIC /arch:SSE4.2)
    target_compile_definitions(cypha_core PUBLIC __SSE4_1__=1)
  else()
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
