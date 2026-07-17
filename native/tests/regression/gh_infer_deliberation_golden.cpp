// Reference parity fixture: gh_infer + deliberation vs native infer_cpu (infer_at_h / gh_infer_at_h).
// Fixture: fixtures/gh_infer_deliberation/sidecar.json
// Generate: python scripts/generate_gh_infer_deliberation_fixture.py

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
      std::cerr << "usage: gh_infer_deliberation_golden <fixture_dir>\n";
      return 2;
    }
    const std::string dir = argv[1];
    const std::string side_path = dir + "/sidecar.json";
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

    std::vector<double> fflat = load_ff_json(dir + "/" + side.value("f_field_json", "f_field.json"), d, fd);
    cypha::CyphaInferModel model = cypha::CyphaInferModel::from_root(root, fflat.data(), fd);

    const double alpha = side.value("nig_alpha", 0.98);
    double chi = 1.0;
    double psi = 1.0;

    constexpr double kAtol = 1e-10;

    for (const auto& c : side.at("cases")) {
      const std::string name = c.value("name", "?");
      std::vector<double> x = c.at("x").get<std::vector<double>>();
      if (static_cast<int>(x.size()) != d) {
        throw std::runtime_error(name + ": x dim mismatch");
      }
      std::vector<double> H;
      cypha::batch_encode(model, x.data(), 1, H);

      const bool use_gh = c.value("use_gh", false);
      std::string got_label;
      double got_conf = 0.0;
      double got_r_eff = 0.0;

      if (use_gh) {
        chi = c.value("chi", chi);
        psi = c.value("psi", psi);
        cypha::GhInferAtHResult gh = cypha::gh_infer_at_h(model, H.data(), chi, psi, alpha);
        got_label = gh.label;
        got_conf = gh.confidence;
        got_r_eff = gh.r_eff;
        if (!near_eq(got_conf, c.at("expected_confidence").get<double>(), kAtol)) {
          std::cerr << name << ": confidence got " << got_conf << " expected "
                    << c.at("expected_confidence").get<double>() << "\n";
          return 1;
        }
        if (!near_eq(got_r_eff, c.at("expected_r_eff").get<double>(), kAtol)) {
          std::cerr << name << ": r_eff got " << got_r_eff << " expected "
                    << c.at("expected_r_eff").get<double>() << "\n";
          return 1;
        }
        if (!near_eq(gh.chi_new, c.at("expected_chi_new").get<double>(), kAtol)) {
          std::cerr << name << ": chi_new mismatch\n";
          return 1;
        }
        if (!near_eq(gh.psi_new, c.at("expected_psi_new").get<double>(), kAtol)) {
          std::cerr << name << ": psi_new mismatch\n";
          return 1;
        }
        chi = gh.chi_new;
        psi = gh.psi_new;
      } else {
        cypha::CyphaInferOptions opt{};
        opt.deliberation_lo = c.value("deliberation_lo", model.deliberation_lo);
        opt.deliberation_hi = c.value("deliberation_hi", model.deliberation_hi);
        opt.use_field = true;
        cypha::InferAtHResult inf = cypha::infer_at_h(model, H.data(), opt);
        got_label = inf.label;
        got_conf = inf.confidence;
        if (!near_eq(got_conf, c.at("expected_confidence").get<double>(), kAtol)) {
          std::cerr << name << ": confidence got " << got_conf << " expected "
                    << c.at("expected_confidence").get<double>() << "\n";
          return 1;
        }
      }

      const std::string exp_label = c.at("expected_label").get<std::string>();
      if (got_label != exp_label) {
        std::cerr << name << ": label got '" << got_label << "' expected '" << exp_label << "'\n";
        return 1;
      }
      (void)got_r_eff;
    }

    std::cout << "gh_infer_deliberation parity OK\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "gh_infer_deliberation_golden: " << ex.what() << "\n";
    return 1;
  }
}
