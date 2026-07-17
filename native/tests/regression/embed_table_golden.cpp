// embed_table_golden — Izaac GF(2^n) EmbedTable vs Python IzaacEmbedding fixture.
// Usage: embed_table_golden <fixtures/embed_table/sidecar.json>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/embed_table.hpp"

namespace {

bool near_all(const std::vector<double>& got, const std::vector<double>& exp, double atol) {
  if (got.size() != exp.size()) {
    return false;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::abs(got[i] - exp[i]) > atol) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: embed_table_golden <sidecar.json>\n";
      return 2;
    }
    std::ifstream sf(argv[1]);
    if (!sf) {
      throw std::runtime_error("cannot open sidecar.json");
    }
    std::stringstream sb;
    sb << sf.rdbuf();
    nlohmann::json side = nlohmann::json::parse(sb.str());

    constexpr double kAtol = 1e-12;

    const auto run_case = [&](const nlohmann::json& cfg) {
      const std::uint32_t vocab = cfg.at("vocab_size").get<std::uint32_t>();
      const std::uint32_t d_embed = cfg.at("d_embed").get<std::uint32_t>();
      const std::uint32_t seed = cfg.at("seed").get<std::uint32_t>();
      const double table_sum_exp = cfg.at("table_sum").get<double>();
      const std::string tag = std::to_string(vocab) + "x" + std::to_string(d_embed) + "s" + std::to_string(seed);

      cypha::cyphalm::EmbedTable table(vocab, d_embed, seed);

      double table_sum = 0.0;
      for (double v : table.table()) {
        table_sum += v;
      }
      if (std::abs(table_sum - table_sum_exp) > kAtol) {
        std::cerr << "FAIL " << tag << " table_sum got=" << table_sum << " exp=" << table_sum_exp << "\n";
        return false;
      }
      std::cout << "PASS " << tag << " table_sum=" << table_sum << "\n";

      for (const auto& item : cfg.at("tokens").items()) {
        const std::uint32_t tid = static_cast<std::uint32_t>(std::stoul(item.key()));
        auto exp = item.value().get<std::vector<double>>();
        auto got = table.embed_vec(tid);
        if (!near_all(got, exp, kAtol)) {
          double max_e = 0.0;
          std::size_t max_i = 0;
          for (std::size_t i = 0; i < got.size(); ++i) {
            const double e = std::abs(got[i] - exp[i]);
            if (e > max_e) {
              max_e = e;
              max_i = i;
            }
          }
          std::cerr << "FAIL " << tag << " token " << tid << " max_err=" << max_e << " at i=" << max_i
                    << " got=" << got[max_i] << " exp=" << exp[max_i] << "\n";
          return false;
        }
        std::cout << "PASS " << tag << " token " << tid << "\n";
      }
      return true;
    };

    bool ok = true;
    if (side.contains("cases")) {
      for (const auto& cfg : side.at("cases")) {
        ok = run_case(cfg) && ok;
      }
    } else {
      ok = run_case(side);
    }

    if (!ok) {
      return 1;
    }
    std::cout << "embed_table_golden OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "embed_table_golden error: " << e.what() << "\n";
    return 1;
  }
}
