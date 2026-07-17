#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

void need_tensor(const cypha::CNode& n, const std::vector<std::uint32_t>& sh) {
  if (n.kind != cypha::CNode::Tensor) {
    throw std::runtime_error("expected tensor");
  }
  if (n.shape != sh) {
    throw std::runtime_error("tensor shape mismatch");
  }
}

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
      std::cerr << "usage: memory_train_golden <fixtures/memory_train_dir>\n";
      return 2;
    }
    std::string dir = argv[1];
    std::string sidepath = dir + "/sidecar.json";
    std::ifstream sf(sidepath);
    if (!sf) {
      throw std::runtime_error("cannot open sidecar");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());

    std::vector<double> h;
    for (const auto& v : j["h"]) {
      h.push_back(v.get<double>());
    }
    std::vector<double> h_field;
    for (const auto& v : j["h_field"]) {
      h_field.push_back(v.get<double>());
    }
    std::string label = j["label"].get<std::string>();
    double temperature = j["temperature"].get<double>();
    double ood_sigma = j["ood_sigma"].get<double>();
    double world_lr = j["world_lr"].get<double>();
    double delta_lr = j["delta_lr"].get<double>();
    double expected_loss = j["expected_loss"].get<double>();
    int field_dim = j["field_dim"].get<int>();
    std::vector<double> f_flat = flatten_f_field(j["f_field"]);
    (void)temperature;
    (void)ood_sigma;

    std::unordered_map<std::string, double> ctx;
    for (auto it = j["context_prior"].begin(); it != j["context_prior"].end(); ++it) {
      ctx[it.key()] = it.value().get<double>();
    }

    cypha::CNode before = cypha::load_cypha_file((dir + "/before.cypha").c_str());
    cypha::CyphaDifMemoryState st =
        cypha::CyphaDifMemoryState::from_cypha_root(before, f_flat.data(), field_dim);

    double loss = st.memory_train(h.data(), label, h_field.data(), ctx, temperature, ood_sigma, world_lr,
                                  delta_lr, nullptr);
    constexpr double kLossTol = 1e-9;
    if (!near_eq(loss, expected_loss, kLossTol)) {
      std::cerr << "loss got " << loss << " expected " << expected_loss << "\n";
      return 1;
    }

    cypha::CNode after = cypha::load_cypha_file((dir + "/after.cypha").c_str());
    const cypha::CNode& w_exp = cypha::map_get_required(after, "world");
    const cypha::CNode& mu_n = cypha::map_get_required(w_exp, "mu");
    const cypha::CNode& v_n = cypha::map_get_required(w_exp, "v");
    const cypha::CNode& n_n = cypha::map_get_required(w_exp, "n");
    const cypha::CNode& dr_n = cypha::map_get_required(w_exp, "drift_ema");
    need_tensor(mu_n, {static_cast<unsigned>(st.d_latent)});
    need_tensor(v_n, {static_cast<unsigned>(st.d_latent)});
    compare_vec(st.world_mu, mu_n.tensor, "world.mu", 1e-9);
    compare_vec(st.world_v, v_n.tensor, "world.v", 1e-9);
    std::int64_t n_exp = 0;
    if (n_n.kind == cypha::CNode::Int) {
      n_exp = n_n.i;
    } else {
      n_exp = static_cast<std::int64_t>(n_n.f);
    }
    if (st.world_n != n_exp) {
      std::cerr << "world.n mismatch\n";
      return 1;
    }
    double dr_e = dr_n.kind == cypha::CNode::Float ? dr_n.f : static_cast<double>(dr_n.i);
    if (!near_eq(st.world_drift_ema, dr_e, 1e-9)) {
      std::cerr << "drift_ema mismatch got " << st.world_drift_ema << " expected " << dr_e << "\n";
      return 1;
    }

    const cypha::CNode& cl_exp = cypha::map_get_required(after, "classes");
    std::size_t idx = 0;
    for (const auto& pr : cl_exp.map) {
      const std::string& lbl = pr.first;
      if (idx >= st.labels.size() || st.labels[idx] != lbl) {
        throw std::runtime_error("label order mismatch vs after.cypha");
      }
      const cypha::CNode& cnode = pr.second;
      const cypha::CNode& dm = cypha::map_get_required(cnode, "delta_mu");
      const cypha::CNode& no = cypha::map_get_required(cnode, "n_obs");
      const cypha::CNode* nc = cypha::map_get(cnode, "n_correct");
      for (int jj = 0; jj < st.d_latent; ++jj) {
        double got = st.D[idx * static_cast<std::size_t>(st.d_latent) + static_cast<std::size_t>(jj)];
        double ex = dm.tensor[static_cast<std::size_t>(jj)];
        if (!near_eq(got, ex, 1e-9)) {
          std::cerr << "delta_mu " << lbl << "[" << jj << "] mismatch\n";
          return 1;
        }
      }
      double n_obs_e = 0;
      if (no.kind == cypha::CNode::Int) {
        n_obs_e = static_cast<double>(no.i);
      } else {
        n_obs_e = no.f;
      }
      if (!near_eq(st.n_obs_buf[idx], n_obs_e, 1e-9)) {
        std::cerr << "n_obs mismatch " << lbl << "\n";
        return 1;
      }
      std::int64_t nce = 0;
      if (nc != nullptr && nc->kind != cypha::CNode::Nil) {
        if (nc->kind == cypha::CNode::Int) {
          nce = nc->i;
        } else {
          nce = static_cast<std::int64_t>(nc->f);
        }
      }
      if (st.n_correct[idx] != nce) {
        std::cerr << "n_correct mismatch " << lbl << "\n";
        return 1;
      }
      ++idx;
    }
    if (idx != st.labels.size()) {
      throw std::runtime_error("class count mismatch");
    }

    std::cout << "memory_train parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "memory_train_golden: " << e.what() << "\n";
    return 1;
  }
}
