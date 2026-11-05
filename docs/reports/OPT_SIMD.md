# O5: the byte LSTM on AVX-512 and the n-gram's lookups (GENERATION_PLAN §13.41)

2026-11-05 · Odin Loch

**Odin:** "Keep looking. Come up with more. Try optimisation techniques … Go crazy on it."

The job (§13.41, O5): make the byte LSTM and the online n-gram expert faster, with exactly the same output. Branch `wip/opt-simd`, from `main` at 6530cb2. Only two files change: `native/include/cypha/online/byte_lstm.hpp` and `native/include/cypha/online/online_ngram.hpp`.

RESULTS_PLACEHOLDER
