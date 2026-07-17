// Golden parity: federated_average_payloads on worker JSON fixtures (world_mu + class stats).
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/federated_aggregate.hpp"

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

nlohmann::json read_json(const fs::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open: " + path.string());
  }
  std::stringstream buf;
  buf << in.rdbuf();
  return nlohmann::json::parse(buf.str());
}

std::vector<double> read_vec1d(const nlohmann::json& j, const char* key) {
  if (!j.contains(key) || !j[key].is_array()) {
    throw std::runtime_error(std::string("missing array: ") + key);
  }
  std::vector<double> out;
  out.reserve(j[key].size());
  for (const auto& v : j[key]) {
    out.push_back(v.get<double>());
  }
  return out;
}

void compare_merged_state(const cypha::CyphaDifMemoryState& merged, const nlohmann::json& exp) {
  constexpr double kTol = 1e-9;
  if (merged.d_latent != exp.at("d_latent").get<int>()) {
    throw std::runtime_error("d_latent mismatch");
  }
  if (merged.field_dim != exp.at("field_dim").get<int>()) {
    throw std::runtime_error("field_dim mismatch");
  }
  if (merged.world_n != exp.at("world_n").get<std::int64_t>()) {
    throw std::runtime_error("world_n mismatch");
  }
  compare_vec(merged.world_mu, read_vec1d(exp, "world_mu"), "world_mu", kTol);
  compare_vec(merged.world_v, read_vec1d(exp, "world_v"), "world_v", kTol);

  const auto& exp_classes = exp.at("classes");
  if (!exp_classes.is_array()) {
    throw std::runtime_error("expected classes array");
  }
  if (exp_classes.size() != merged.labels.size()) {
    throw std::runtime_error("class count mismatch");
  }
  for (std::size_t k = 0; k < merged.labels.size(); ++k) {
    const auto& row = exp_classes[k];
    if (merged.labels[k] != row.at("label").get<std::string>()) {
      throw std::runtime_error("label mismatch at " + std::to_string(k));
    }
    const double n_obs_e = row.at("n_obs").get<double>();
    if (!near_eq(merged.n_obs_buf[k], n_obs_e, kTol)) {
      throw std::runtime_error("n_obs mismatch: " + merged.labels[k]);
    }
    if (merged.n_correct[k] != row.at("n_correct").get<std::int64_t>()) {
      throw std::runtime_error("n_correct mismatch: " + merged.labels[k]);
    }
    const auto dm_exp = read_vec1d(row, "delta_mu");
    std::vector<double> dm_got;
    dm_got.reserve(static_cast<std::size_t>(merged.d_latent));
    for (int j = 0; j < merged.d_latent; ++j) {
      dm_got.push_back(merged.D[k * static_cast<std::size_t>(merged.d_latent) + static_cast<std::size_t>(j)]);
    }
    compare_vec(dm_got, dm_exp, ("delta_mu." + merged.labels[k]).c_str(), kTol);
  }
}

void test_nested_world_format() {
  nlohmann::json nested = {
      {"d_latent", 2},
      {"field_dim", 8},
      {"world", {{"n", 5}, {"mu", {1.0, 2.0}}, {"v", {1.0, 1.0}}}},
      {"classes",
       {{{"label", "x"}, {"n_obs", 1.0}, {"n_correct", 1}, {"delta_mu", {0.5, -0.5}}}}},
  };
  const cypha::FederatedPayload p = cypha::federated_payload_from_json(nested);
  if (p.world_n != 5 || p.world_mu.size() != 2 || p.labels.size() != 1) {
    throw std::runtime_error("nested world parse failed");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: federated_merge_golden <fixtures/federated_merge>\n";
      return 2;
    }
    const fs::path dir = fs::path(argv[1]);
    test_nested_world_format();

    const nlohmann::json exp = read_json(dir / "expected_merged.json");
    const nlohmann::json ja = read_json(dir / "worker_a.json");
    const nlohmann::json jb = read_json(dir / "worker_b.json");

    std::vector<cypha::FederatedPayload> payloads{
        cypha::federated_payload_from_json(ja),
        cypha::federated_payload_from_json(jb),
    };
    const cypha::CyphaDifMemoryState merged = cypha::federated_average_payloads(payloads);
    compare_merged_state(merged, exp);

    std::cout << "federated_merge parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "federated_merge_golden: " << e.what() << "\n";
    return 1;
  }
}
