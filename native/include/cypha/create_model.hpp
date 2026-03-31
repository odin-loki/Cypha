#pragma once
/// Build a fresh, empty CyphaInferModel / CyphaDifMemoryState from scratch.
///
/// Enables full Python-off operation: the Qt shell can create an empty model
/// (with no pre-existing .cypha) and immediately start training from a CSV.
///
/// The returned ``CNode`` root is a valid v3 Cypha binary-compatible map that
/// ``CyphaInferModel::from_root`` and ``CyphaDifMemoryState::from_cypha_root``
/// both accept.  Optionally save it with ``cypha::save_cypha_file``.

#include <cstdint>
#include <string>
#include <vector>

#include "cypha/load_cypha.hpp"

namespace cypha {

struct FreshModelParams {
    int    input_dim  {8};      ///< Raw feature dimension (also latent dim for VectorEncoder)
    int    field_dim  {24};     ///< Temporal field dimension
    double temperature{1.0};   ///< Initial classifier temperature
    double world_lr   {0.005}; ///< World learning rate (stored for native train UI)
    double delta_lr   {0.01};  ///< Class-delta learning rate
};

/// Build an empty model root.  The result is immediately loadable; no classes
/// are registered yet — they appear automatically when the first training step
/// names a new label.
[[nodiscard]] CNode create_fresh_model_root(const FreshModelParams& p = {});

/// Convenience: create and save to disk in one step.
void create_and_save_fresh_model(const char* path, const FreshModelParams& p = {});

}  // namespace cypha
