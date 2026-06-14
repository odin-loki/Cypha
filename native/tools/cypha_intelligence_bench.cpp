// cypha_intelligence_bench — P-space profile export for reference fixture (Paper I–III).
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/intelligence/profile_from_model.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  fs::path repo = cypha::bench::repo_root();
  fs::path cypha_path;
  fs::path out_path;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + k);
      }
      return argv[++i];
    };
    if (k == "--repo") {
      repo = need();
    } else if (k == "--cypha") {
      cypha_path = need();
    } else if (k == "--out") {
      out_path = need();
    } else if (k == "--help" || k == "-h") {
      std::cerr << "usage: cypha_intelligence_bench [--repo DIR] [--cypha PATH] [--out PATH]\n";
      return 0;
    }
  }

  try {
    const auto profiler = cypha::intelligence::profile_from_reference_fixture(repo, cypha_path);
    nlohmann::json report = cypha::intelligence::intelligence_profile_report_json(profiler);
    report["domain"] = "d18_intelligence_profile";
    const std::string text = report.dump(2);
    if (out_path.empty()) {
      std::cout << text << '\n';
    } else {
      std::ofstream out(out_path);
      if (!out) {
        throw std::runtime_error("cannot write " + out_path.string());
      }
      out << text;
      std::cerr << "wrote " << out_path << '\n';
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "cypha_intelligence_bench: " << ex.what() << '\n';
    return 1;
  }
}
