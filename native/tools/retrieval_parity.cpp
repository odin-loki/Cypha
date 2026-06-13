// retrieval_parity — CyphaDIF.retrieve vs native retrieve_from_x.
// Usage: retrieval_parity <fixtures/retrieval/sidecar.json>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

std::vector<double> load_ff_json(const std::string& path, int d, int fd) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("cannot open f_field.json: " + path);
  }
  std::stringstream b;
  b << f.rdbuf();
  auto j = nlohmann::json::parse(b.str());
  std::vector<double> out;
  for (const auto& row : j) {
    for (const auto& v : row) {
      out.push_back(v.get<double>());
    }
  }
  if (static_cast<int>(out.size()) != d * fd) {
    throw std::runtime_error("f_field.json size mismatch");
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: retrieval_parity <fixture_dir>/sidecar.json\n";
      return 2;
    }
    const std::string side_path = argv[1];
    const std::string dir = side_path.substr(0, side_path.find_last_of("/\\"));
    std::ifstream sf(side_path);
    if (!sf) {
      throw std::runtime_error("missing sidecar.json");
    }
    std::stringstream sb;
    sb << sf.rdbuf();
    nlohmann::json side = nlohmann::json::parse(sb.str());

    const std::string cypha_path = dir + "/" + side.value("reference_cypha", "reference.cypha");
    cypha::CNode root = cypha::load_cypha_file(cypha_path.c_str());
    const cypha::CNode& fh = cypha::map_get_required(root, "field_h");
    const int fd = static_cast<int>(fh.shape[0]);
    const cypha::CNode& enc = cypha::map_get_required(root, "enc_W");
    const int d = static_cast<int>(enc.shape[0]);
    const int input_dim = side.at("input_dim").get<int>();

    std::vector<double> fflat = load_ff_json(dir + "/" + side.value("f_field_json", "f_field.json"), d, fd);
    cypha::CyphaInferModel model = cypha::CyphaInferModel::from_root(root, fflat.data(), fd);

    cypha::CyphaInferOptions opt;
    opt.use_field = side.value("use_field", true);
    opt.deliberation_lo = side.value("deliberation_lo", cypha::kDeliberationLoDefault);
    opt.deliberation_hi = side.value("deliberation_hi", cypha::kDeliberationHiDefault);

    constexpr double kAtol = 1e-10;
    int failures = 0;

    for (const auto& c : side.at("cases")) {
      const std::string name = c.value("name", "?");
      std::vector<double> query = c.at("query_x").get<std::vector<double>>();
      if (static_cast<int>(query.size()) != input_dim) {
        throw std::runtime_error(name + ": query_x dim mismatch");
      }

      std::vector<double> db_flat;
      for (const auto& row : c.at("database_x")) {
        auto rv = row.get<std::vector<double>>();
        if (static_cast<int>(rv.size()) != input_dim) {
          throw std::runtime_error(name + ": database row dim mismatch");
        }
        db_flat.insert(db_flat.end(), rv.begin(), rv.end());
      }
      const int n_db = static_cast<int>(c.at("database_x").size());
      const int top_k = c.at("top_k").get<int>();

      std::optional<std::string> label_opt;
      if (c.contains("label") && !c.at("label").is_null()) {
        label_opt = c.at("label").get<std::string>();
      }

      auto hits = cypha::retrieve_from_x(model, query.data(), db_flat.data(), n_db, input_dim, top_k, opt,
                                         label_opt);

      const auto& exp = c.at("expected");
      if (static_cast<int>(hits.size()) != static_cast<int>(exp.size())) {
        std::cerr << "FAIL " << name << ": hit count got=" << hits.size() << " exp=" << exp.size() << "\n";
        ++failures;
        continue;
      }

      bool ok = true;
      for (std::size_t i = 0; i < hits.size(); ++i) {
        const int exp_idx = exp[i].at("index").get<int>();
        const double exp_ll = exp[i].at("log_likelihood").get<double>();
        const std::string exp_label = exp[i].at("predicted_label").get<std::string>();
        if (hits[i].index != exp_idx || hits[i].predicted_label != exp_label ||
            !near_eq(hits[i].log_likelihood, exp_ll, kAtol)) {
          std::cerr << "FAIL " << name << "[" << i << "]: got=(" << hits[i].index << ", "
                    << hits[i].log_likelihood << ", " << hits[i].predicted_label << ") exp=(" << exp_idx << ", "
                    << exp_ll << ", " << exp_label << ")\n";
          ok = false;
        }
      }
      if (ok) {
        std::cout << "PASS " << name << " (" << hits.size() << " hits)\n";
      } else {
        ++failures;
      }
    }

    if (failures > 0) {
      std::cerr << "retrieval_parity: " << failures << " case(s) failed\n";
      return 1;
    }
    std::cout << "retrieval_parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "retrieval_parity error: " << e.what() << "\n";
    return 1;
  }
}
