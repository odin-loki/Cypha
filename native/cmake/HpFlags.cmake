# hp compile profile for CyphaLM — gate24 only.
#
# The vendored hp tree is gate24-only: every ablation flag CyphaLM did not
# ship was resolved out of the source (see docs/reports/CYPHALM_HP_GATE24_STRIP.md),
# so no -DHP_* feature list is needed. HP_SLOT_MAX stays a build knob, fixed at 24.
# Lab RSS ~1.5 GB @ mem 22.

set(CYPHA_HP_SLOT_MAX 24 CACHE STRING "hp HP_SLOT_MAX compile cap (fixed at 24)")

message(STATUS "CyphaLM hp: gate24 (gate24-only hp tree, HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX})")

function(cypha_apply_hp_compile_flags target)
  target_compile_definitions(${target} PUBLIC
    CYPHA_HP_GATE24=1
    CYPHA_HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
    HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
  )
endfunction()
