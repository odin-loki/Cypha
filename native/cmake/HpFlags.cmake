# hp compile-time profiles for CyphaLM.
#
# light (default): HP_SLOT_MAX=24 only; hp/features.hpp defaults (grow flags OFF).
#   Lab RSS ~1.6 GB @ mem 22 (hp/tools/hp_harness.sh RECORD H34). ~1.72 BPC enwik8MB.
#   Label: fast/dev — NOT user champ quality.
# gate24: v78_flags.ps1 feature set + HP_SLOT_MAX=24 (~1.61 BPC class on enwik8MB).
#   PLAN 8MB screen quality tier; RAM between light and champ.
# champ: v78_flags.ps1 feature set + HP_SLOT_MAX=35 (~15 GB @ mem 22, ~1.610 BPC).
#   User quality bar; requires >=32 GiB host for full 8 MB compress.
#   mem-26-class: -DCYPHA_HP_PROFILE=champ -DCYPHA_HP_SLOT_MAX=31

set(CYPHA_HP_PROFILE "light" CACHE STRING
  "hp compile profile: light (default), gate24 (v78+SLOT_MAX=24), or champ (v78+SLOT_MAX=35)")

set(CYPHA_HP_SLOT_MAX 24 CACHE STRING
  "hp HP_SLOT_MAX compile cap (24=light/gate24, 35=champ default, 31=mem-26-class)")

# Back-compat: CYPHA_HP_CHAMP_BUILD=ON forces champ profile.
option(CYPHA_HP_CHAMP_BUILD
  "Deprecated alias for -DCYPHA_HP_PROFILE=champ"
  OFF)

if(CYPHA_HP_CHAMP_BUILD)
  set(CYPHA_HP_PROFILE "champ")
  message(STATUS "CYPHA_HP_CHAMP_BUILD=ON -> CYPHA_HP_PROFILE=champ")
endif()

if(NOT CYPHA_HP_PROFILE STREQUAL "light"
    AND NOT CYPHA_HP_PROFILE STREQUAL "gate24"
    AND NOT CYPHA_HP_PROFILE STREQUAL "champ")
  message(FATAL_ERROR
    "CYPHA_HP_PROFILE must be 'light', 'gate24', or 'champ' (got '${CYPHA_HP_PROFILE}')")
endif()

if(CYPHA_HP_PROFILE STREQUAL "champ")
  if(CYPHA_HP_SLOT_MAX STREQUAL "24")
    set(CYPHA_HP_SLOT_MAX 35)
  endif()
  message(STATUS "CYPHA_HP_PROFILE=champ: HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX} (v78_flags.ps1)")
elseif(CYPHA_HP_PROFILE STREQUAL "gate24")
  set(CYPHA_HP_SLOT_MAX 24)
  message(STATUS "CYPHA_HP_PROFILE=gate24: HP_SLOT_MAX=24 (v78_flags.ps1, quality screen)")
else()
  set(CYPHA_HP_SLOT_MAX 24)
  message(STATUS "CYPHA_HP_PROFILE=light: HP_SLOT_MAX=24 (features.hpp defaults, grow OFF)")
endif()

# Parse HP compile defs from vendored v78_flags.ps1 (gate24 + champ profiles).
set(_CYPHA_HP_V78_FEATURE_DEFS "")
if(CYPHA_HP_PROFILE STREQUAL "champ" OR CYPHA_HP_PROFILE STREQUAL "gate24")
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
  message(STATUS "CYPHA_HP_PROFILE=${CYPHA_HP_PROFILE}: ${_CYPHA_V78_N} v78 feature defs from v78_flags.ps1")
endif()

function(cypha_apply_hp_compile_flags target)
  target_compile_definitions(${target} PUBLIC
    CYPHA_HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
    HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
  )
  if(CYPHA_HP_PROFILE STREQUAL "champ")
    target_compile_definitions(${target} PUBLIC
      CYPHA_HP_PROFILE_CHAMP=1
      CYPHA_HP_CHAMP_BUILD=1)
  elseif(CYPHA_HP_PROFILE STREQUAL "gate24")
    target_compile_definitions(${target} PUBLIC
      CYPHA_HP_PROFILE_GATE24=1
      CYPHA_HP_CHAMP_BUILD=0)
  else()
    target_compile_definitions(${target} PUBLIC
      CYPHA_HP_PROFILE_LIGHT=1
      CYPHA_HP_CHAMP_BUILD=0)
  endif()
  foreach(_def IN LISTS _CYPHA_HP_V78_FEATURE_DEFS)
    target_compile_definitions(${target} PRIVATE ${_def})
  endforeach()
endfunction()
