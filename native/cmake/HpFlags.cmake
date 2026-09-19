# hp compile profile for CyphaLM — gate24 only.
#
# Always applies vendored v78_flags.ps1 + HP_SLOT_MAX=24 (~1.61 BPC class on enwik8MB).
# Lab RSS ~1.5–2 GB @ mem 22. No light/champ SKU matrix.

set(CYPHA_HP_SLOT_MAX 24 CACHE STRING "hp HP_SLOT_MAX compile cap (fixed at 24)")

set(_CYPHA_HP_V78_FEATURE_DEFS "")
set(_CYPHA_V78_PS1
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hp/tools/v78_flags.ps1")
if(NOT EXISTS "${_CYPHA_V78_PS1}")
  message(FATAL_ERROR "v78_flags.ps1 missing at ${_CYPHA_V78_PS1}")
endif()
file(READ "${_CYPHA_V78_PS1}" _CYPHA_V78_CONTENT)
string(REGEX MATCHALL "-DHP_[A-Z0-9_]+=[0-9]+" _CYPHA_V78_MATCHES "${_CYPHA_V78_CONTENT}")
foreach(_match IN LISTS _CYPHA_V78_MATCHES)
  string(REGEX REPLACE "^-D" "" _def "${_match}")
  if(_def MATCHES "^HP_SLOT_MAX=")
    continue()
  endif()
  list(APPEND _CYPHA_HP_V78_FEATURE_DEFS "${_def}")
endforeach()
list(LENGTH _CYPHA_HP_V78_FEATURE_DEFS _CYPHA_V78_N)
message(STATUS "CyphaLM hp: gate24 (v78_flags.ps1 + HP_SLOT_MAX=24, ${_CYPHA_V78_N} feature defs)")

function(cypha_apply_hp_compile_flags target)
  target_compile_definitions(${target} PUBLIC
    CYPHA_HP_GATE24=1
    CYPHA_HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
    HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
  )
  foreach(_def IN LISTS _CYPHA_HP_V78_FEATURE_DEFS)
    target_compile_definitions(${target} PRIVATE ${_def})
  endforeach()
endfunction()
