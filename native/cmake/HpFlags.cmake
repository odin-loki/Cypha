# hp compile-time feature flags (v78 champ set from odin-loki/CompressionAlgorithm).
# Production default: HP_SLOT_MAX=24 (~1.6 GB RSS @ mem 22 per hp/tools/hp_harness.sh RECORD H34).
# Champ/research: CYPHA_HP_CHAMP_BUILD=ON -> HP_SLOT_MAX=35 (~15 GB RSS @ mem 22).

set(CYPHA_HP_SLOT_MAX 24 CACHE STRING
  "hp HP_SLOT_MAX compile cap (24=production RAM-speed, 35=champ/research)")

option(CYPHA_HP_CHAMP_BUILD
  "Compile hp champ profile (HP_SLOT_MAX=35, ~15 GB RSS class). Default OFF."
  OFF)

if(CYPHA_HP_CHAMP_BUILD)
  set(CYPHA_HP_SLOT_MAX 35)
  message(STATUS "CYPHA_HP_CHAMP_BUILD=ON: HP_SLOT_MAX=35 (champ / research)")
else()
  message(STATUS "CYPHA_HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX} (production RAM-speed default)")
endif()

# v78_flags.ps1 feature set (SLOT_MAX applied separately above).
set(_CYPHA_HP_V78_FEATURE_DEFS
  HP_MIXER_SKIP=32
  HP_SPARSE_UTF8=1
  HP_SENWORD=1
  HP_GATE_ARGMAX=1
  HP_SLOT_GROW=1
  HP_GATE_WORDPOS=1
  HP_STATE_TABLE2=1
  HP_SLOT_GROW_EXTRA=1
  HP_SENT_STREAM=1
  HP_MATCH_18=1
  HP_QUOTE_STACK=1
  HP_SLOT_WORD2=1
  HP_SLOT_WORD3=1
  HP_SLOT_WORD4=1
  HP_SLOT_WORD5=1
  HP_SLOT_WORD6=1
  HP_SLOT_WORD7=1
  HP_SLOT_WORD8=1
  HP_SLOT_WORD9=1
  HP_MATCH_01=1
  HP_MATCH_02=1
  HP_MATCH_05=1
  HP_WMATCH_4=1
  HP_SLOT_WSTR2=1
  HP_SLOT_S3=1
  HP_SLOT_S4=1
  HP_SENT_MEM=1
  HP_SENT_GRP_CTX=1
  HP_WSTR_GRP=1
  HP_WBI_GRP=1
  HP_SMEM_GRP=1
  HP_SLOT_O34=1
  HP_SLOT_O34B=1
  HP_SLOT_O34C=1
  HP_SLOT_O34D=1
  HP_SLOT_O34E=1
  HP_SLOT_O34F=1
  HP_SLOT_O6=1
  HP_SLOT_O6B=1
  HP_SLOT_O6C=1
  HP_SLOT_O6D=1
  HP_SLOT_O6E=1
  HP_SLOT_O6F=1
  HP_SLOT_O6G=1
  HP_SLOT_O6H=1
  HP_SLOT_O6I=1
  HP_MATCH_GROW=1
  HP_SENGRP_MOD=1
  HP_SLOT_SGRP=1
  HP_SENGRP_WORD=1
  HP_COL_GRP=1
  HP_TAG_GRP=1
  HP_SENWORD_GRP=1
  HP_BRK_GRP=1
  HP_SP_GRP=1
  HP_SLOT_COL3=1
  HP_NEST_MOD=1
  HP_PARA_MOD=1
  HP_LINE_MOD=1
  HP_STATE_MOD=1
  HP_HASH_CHK=1
  HP_TPLNAME_MOD=1
  HP_DMC_MOD=1
  HP_LZP_MOD=1
  HP_SKIPK_MOD=1
  HP_INFOKEY_MOD=1
  HP_HASH2_O6=1
  HP_SKIP3_MOD=1
  HP_LINKPIPE_MOD=1
  HP_SKIP4_MOD=1
  HP_CAT_MOD=1
  HP_HEADING_MOD=1
  HP_TITLE_MOD=1
  HP_SECTITLE_MOD=1
  HP_WIKISTACK_MOD=1
  HP_CAPMASK_MOD=1
  HP_UPPERGAP_MOD=1
  HP_WORDLEN_MOD=1
)

function(cypha_apply_hp_compile_flags target)
  target_compile_definitions(${target} PUBLIC
    CYPHA_HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
    HP_SLOT_MAX=${CYPHA_HP_SLOT_MAX}
  )
  if(CYPHA_HP_CHAMP_BUILD)
    target_compile_definitions(${target} PUBLIC CYPHA_HP_CHAMP_BUILD=1)
  else()
    target_compile_definitions(${target} PUBLIC CYPHA_HP_CHAMP_BUILD=0)
  endif()
  foreach(_def IN LISTS _CYPHA_HP_V78_FEATURE_DEFS)
    target_compile_definitions(${target} PRIVATE ${_def})
  endforeach()
endfunction()
