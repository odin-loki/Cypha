#pragma once
//
// hp/features.hpp — build switches left after the gate24 strip.
//
// Every model and mixer ablation flag that CyphaLM's gate24 recipe did not
// ship has been removed along with its code; the gate24 set is now the only
// code path. Recover the full ablation lab from odin-loki/CompressionAlgorithm.
//
// NOTHING here is transmitted. Encoder and decoder are the same binary.

#ifndef HP_SLOT_MAX
#define HP_SLOT_MAX 24             // per-model table-bit cap (gate24)
#endif
#ifndef HP_XSIMD
#define HP_XSIMD 1                 // integer mixer dots via xsimd/SSE4.1; bit-identical to scalar
#endif
