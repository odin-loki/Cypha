#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"

namespace fs = std::filesystem;

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

void compare_vec(const std::vector<double>& a, const std::vector<double>& b, const char* name, double atol) {
  if (a.size() != b.size()) {
    throw std::runtime_error(std::string("size mismatch: ") + name);
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!near_eq(a[i], b[i], atol)) {
      std::cerr << name << "[" << i << "] got " << a[i] << " expected " << b[i] << "\n";
      throw std::runtime_error(std::string("mismatch: ") + name);
    }
  }
}

std::vector<double> flatten_f_field(const nlohmann::json& j) {
  std::vector<double> o;
  for (const auto& row : j) {
    for (const auto& v : row) {
      o.push_back(v.get<double>());
    }
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: merge_from_parity <parity_fixtures/merge_from>\n";
      return 2;
    }
    fs::path dir = fs::path(argv[1]);
    std::ifstream sf(dir / "sidecar.json");
    if (!sf) {
      throw std::runtime_error("cannot open sidecar.json");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());

    const int field_dim = j.at("field_dim").get<int>();
    std::vector<double> f_self = flatten_f_field(j.at("f_field_self"));
    std::vector<double> f_other = flatten_f_field(j.at("f_field_other"));

    cypha::CNode self_before = cypha::load_cypha_file((dir / "self_before.cypha").string().c_str());
    cypha::CNode other = cypha::load_cypha_file((dir / "other.cypha").string().c_str());
    cypha::CNode after_exp = cypha::load_cypha_file((dir / "self_after.cypha").string().c_str());

    cypha::CyphaDifMemoryState self =
        cypha::CyphaDifMemoryState::from_cypha_root(self_before, f_self.data(), field_dim);
    cypha::CyphaDifMemoryState other_mem =
        cypha::CyphaDifMemoryState::from_cypha_root(other, f_other.data(), field_dim);

    const double w_self = j.value("weight_self", 0.5);
    const double w_other = j.value("weight_other", 0.5);
    auto new_labels = cypha::memory_merge_from(self, other_mem, w_self, w_other);

    std::vector<std::string> exp_new = j.at("expected_new_labels").get<std::vector<std::string>>();
    if (new_labels != exp_new) {
      std::cerr << "new_labels mismatch\n";
      return 1;
    }

    const cypha::CNode& w_exp = cypha::map_get_required(after_exp, "world");
    compare_vec(self.world_mu, cypha::map_get_required(w_exp, "mu").tensor, "world.mu", 1e-9);
    compare_vec(self.world_v, cypha::map_get_required(w_exp, "v").tensor, "world.v", 1e-9);
    const cypha::CNode& n_n = cypha::map_get_required(w_exp, "n");
    std::int64_t n_exp = n_n.kind == cypha::CNode::Int ? n_n.i : static_cast<std::int64_t>(n_n.f);
    if (self.world_n != n_exp) {
      std::cerr << "world.n mismatch\n";
      return 1;
    }

    const cypha::CNode& cl_exp = cypha::map_get_required(after_exp, "classes");
    std::size_t idx = 0;
    for (const auto& pr : cl_exp.map) {
      const std::string& lbl = pr.first;
      if (idx >= self.labels.size() || self.labels[idx] != lbl) {
        throw std::runtime_error("label order mismatch");
      }
      const cypha::CNode& dm = cypha::map_get_required(pr.second, "delta_mu");
      for (int jj = 0; jj < self.d_latent; ++jj) {
        double got = self.D[idx * static_cast<std::size_t>(self.d_latent) + static_cast<std::size_t>(jj)];
        if (!near_eq(got, dm.tensor[static_cast<std::size_t>(jj)], 1e-9)) {
          std::cerr << "delta_mu mismatch " << lbl << "\n";
          return 1;
        }
      }
      const cypha::CNode& no = cypha::map_get_required(pr.second, "n_obs");
      double n_obs_e = no.kind == cypha::CNode::Int ? static_cast<double>(no.i) : no.f;
      if (!near_eq(self.n_obs_buf[idx], n_obs_e, 1e-9)) {
        std::cerr << "n_obs mismatch " << lbl << "\n";
        return 1;
      }
      ++idx;
    }

    std::cout << "merge_from parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "merge_from_parity: " << e.what() << "\n";
    return 1;
  }
}
