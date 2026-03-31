#include "cypha/registry.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace cypha {

std::vector<RegistryModelRef> registry_scan(const char* root) {
  std::vector<RegistryModelRef> out;
  fs::path r(root);
  if (!fs::exists(r) || !fs::is_directory(r)) {
    return out;
  }
  for (const auto& name_ent : fs::directory_iterator(r)) {
    if (!name_ent.is_directory()) {
      continue;
    }
    std::string name = name_ent.path().filename().string();
    for (const auto& ver_ent : fs::directory_iterator(name_ent.path())) {
      if (!ver_ent.is_directory()) {
        continue;
      }
      std::string version = ver_ent.path().filename().string();
      fs::path model_p = ver_ent.path() / "model.cypha";
      fs::path card_p = ver_ent.path() / "card.json";
      if (!fs::exists(model_p) || !fs::exists(card_p)) {
        continue;
      }
      RegistryModelRef ref;
      ref.name = name;
      ref.version = version;
      ref.model_path = model_p.string();
      ref.card_path = card_p.string();
      fs::path pre_p = ver_ent.path() / "preprocessor.json";
      if (fs::exists(pre_p)) {
        ref.preprocessor_path = pre_p.string();
      }
      out.push_back(std::move(ref));
    }
  }
  return out;
}

bool registry_register_bundle(const char* root, const char* name, const char* version,
                              const char* cypha_src_path, const char* card_src_path,
                              const char* preprocessor_src_path, bool overwrite, std::string* error_out) {
  auto set_err = [&](const char* msg) {
    if (error_out != nullptr) {
      *error_out = msg;
    }
    return false;
  };
  if (root == nullptr || name == nullptr || version == nullptr || cypha_src_path == nullptr ||
      card_src_path == nullptr) {
    return set_err("null argument");
  }
  try {
    fs::path dest = fs::path(root) / name / version;
    if (fs::exists(dest)) {
      if (!overwrite) {
        return set_err("destination exists (use overwrite)");
      }
      fs::remove_all(dest);
    }
    fs::create_directories(dest);
    const fs::path cypha_src{cypha_src_path};
    const fs::path card_src{card_src_path};
    if (!fs::exists(cypha_src) || !fs::is_regular_file(cypha_src)) {
      return set_err("cypha source missing");
    }
    if (!fs::exists(card_src) || !fs::is_regular_file(card_src)) {
      return set_err("card source missing");
    }
    fs::copy_file(cypha_src, dest / "model.cypha", fs::copy_options::overwrite_existing);
    fs::copy_file(card_src, dest / "card.json", fs::copy_options::overwrite_existing);
    if (preprocessor_src_path != nullptr && std::strlen(preprocessor_src_path) > 0) {
      const fs::path pre_src{preprocessor_src_path};
      if (!fs::exists(pre_src) || !fs::is_regular_file(pre_src)) {
        return set_err("preprocessor source missing");
      }
      fs::copy_file(pre_src, dest / "preprocessor.json", fs::copy_options::overwrite_existing);
    }
    return true;
  } catch (const std::exception& e) {
    if (error_out != nullptr) {
      *error_out = e.what();
    }
    return false;
  }
}

}  // namespace cypha
