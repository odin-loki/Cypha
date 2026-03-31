#pragma once

#include <string>
#include <vector>

namespace cypha {

struct RegistryModelRef {
  std::string name;
  std::string version;
  std::string model_path;
  std::string preprocessor_path;
  std::string card_path;
};

/// Scan `<root>/<name>/<version>/` for `model.cypha` + `card.json` (matches `ModelRegistry`).
std::vector<RegistryModelRef> registry_scan(const char* root);

/// Copy `model.cypha` + `card.json` (+ optional `preprocessor.json`) into `<root>/<name>/<version>/`
/// (same layout as Python `ModelRegistry.register` for pre-built artifacts). Fails if the version dir
/// exists and `overwrite` is false. On failure, writes a short message into `error_out` when non-null.
bool registry_register_bundle(const char* root, const char* name, const char* version,
                              const char* cypha_src_path, const char* card_src_path,
                              const char* preprocessor_src_path, bool overwrite, std::string* error_out);

}  // namespace cypha
