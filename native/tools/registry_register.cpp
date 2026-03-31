// Register a pre-built bundle into a ModelRegistry-style tree: <root>/<name>/<version>/model.cypha + card.json.
// Usage: registry_register <root> <name> <version> <path/to/model.cypha> <path/to/card.json>
//          [--pre path/to/preprocessor.json] [--overwrite] [--and-verify]
#include <cstring>
#include <iostream>
#include <string>

#include "cypha/registry.hpp"

namespace {

void usage(const char* argv0) {
  std::cerr << "usage: " << (argv0 ? argv0 : "registry_register")
            << " <root> <name> <version> <model.cypha> <card.json> "
               "[--pre preprocessor.json] [--overwrite] [--and-verify]\n";
}

bool verify_scan(const char* root, const char* name, const char* version) {
  auto refs = cypha::registry_scan(root);
  for (const auto& r : refs) {
    if (r.name == name && r.version == version) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    usage(argv[0]);
    return 2;
  }
  const char* root = argv[1];
  const char* name = argv[2];
  const char* ver = argv[3];
  const char* cypha_p = argv[4];
  const char* card_p = argv[5];
  const char* pre_p = nullptr;
  bool overwrite = false;
  bool and_verify = false;
  for (int i = 6; i < argc; ++i) {
    if (std::strcmp(argv[i], "--overwrite") == 0) {
      overwrite = true;
    } else if (std::strcmp(argv[i], "--and-verify") == 0) {
      and_verify = true;
    } else if (std::strcmp(argv[i], "--pre") == 0 && i + 1 < argc) {
      pre_p = argv[++i];
    } else {
      std::cerr << "unknown arg: " << argv[i] << "\n";
      usage(argv[0]);
      return 2;
    }
  }
  std::string err;
  if (!cypha::registry_register_bundle(root, name, ver, cypha_p, card_p, pre_p, overwrite, &err)) {
    std::cerr << "registry_register_bundle failed: " << err << "\n";
    return 1;
  }
  if (and_verify && !verify_scan(root, name, ver)) {
    std::cerr << "--and-verify: model not visible in registry_scan\n";
    return 1;
  }
  return 0;
}
