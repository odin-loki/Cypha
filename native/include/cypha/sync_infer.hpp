#pragma once

namespace cypha {

struct CyphaInferModel;
struct CyphaDifMemoryState;

/// Copy world + class tensors from training memory into inference snapshot (after `memory_train`).
void sync_infer_model_from_memory(CyphaInferModel& m, const CyphaDifMemoryState& s);

}  // namespace cypha
