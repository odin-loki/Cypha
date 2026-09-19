# Pre-hp Hybrid GRIA+LSTM / SSM / CharLSTM tools and CTests.
# Default OFF: CI and default `cmake --build` only build the hp CyphaLM path.

option(CYPHA_BUILD_LEGACY_CYPHALM
  "Build removed Hybrid GRIA+LSTM/SSM/CharLSTM regression tools and CTests"
  OFF)

if(CYPHA_BUILD_LEGACY_CYPHALM)
  message(STATUS "CYPHA_BUILD_LEGACY_CYPHALM=ON: building legacy GRIA/LSTM/SSM targets")
else()
  message(STATUS "CYPHA_BUILD_LEGACY_CYPHALM=OFF: hp-only CyphaLM (legacy targets skipped)")
endif()

# Executables gated when legacy is OFF (see CMakeLists.txt + CyphaRegression.cmake).
set(CYPHA_LEGACY_CYPHALM_EXE_TARGETS
  stacked_lstm_smoke
  cyphalm_ssm_diagnose
  cypha_fixture_gen
  ewc_cyphalm_smoke
  ewc_hybrid_smoke
  native_sr_gate_laws_smoke
  navigation_loss_hybrid_smoke
  reversible_ssm_cell_smoke
  pgm_cell_smoke
  pgm_checkpoint_roundtrip_smoke
  ewc_weights_smoke
  pgm_cell_bench
  cyphalm_ssm_golden
  embed_table_golden
  cyphalm_hebbian_golden
  cyphalm_char_lstm_golden
)
