#pragma once

#include <string>
#include <vector>

namespace cypha::cyphalm {

class CyphaLMModel;
struct MixingOptions;

/// Save ``{base}.json`` (config metadata) + ``{base}.hpbin`` (HPCP v4 predictor state).
void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path);

/// Load checkpoint from ``{path}.json`` (and sibling ``.hpbin`` when present).
/// Also loads an ensemble manifest (see ``save_cyphalm_ensemble_manifest``):
/// the first member plus the rest attached with ``add_ensemble_member``.
CyphaLMModel load_cyphalm_model(const std::string& json_path);

/// Write a manifest that ``load_cyphalm_model`` loads as a serve-time
/// ensemble of ``member_checkpoints`` (checkpoint .json paths, relative to the
/// manifest or absolute; the first is the primary; equal starting weights).
void save_cyphalm_ensemble_manifest(const std::string& manifest_path,
                                    const std::vector<std::string>& member_checkpoints,
                                    double learning_rate = 0.01);
/// The same, with the flag-gated mixing options (``MixingOptions``) as keys
/// ``final_temperature``, ``final_temperature_lr``, ``infinigram_mode``,
/// ``neural_mix``, ``ensemble_gate``; only those differing from the
/// defaults are written. Loading applies them after every member and stage
/// is attached (the ∞-gram mode before the index); a manifest without them
/// loads today's mixing, and an unknown mode name throws.
void save_cyphalm_ensemble_manifest(const std::string& manifest_path,
                                    const std::vector<std::string>& member_checkpoints,
                                    double learning_rate, const MixingOptions& mixing);

}  // namespace cypha::cyphalm
