#include "cypha/federated_aggregate.hpp"

#include <cmath>
#include <stdexcept>

namespace cypha {

namespace {

constexpr double kEps = 1e-12;

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

}  // namespace

FederatedPayload federated_payload_from_json(const nlohmann::json& j) {
  FederatedPayload p;
  if (j.contains("d_latent")) {
    p.d_latent = j["d_latent"].get<int>();
  }
  if (j.contains("field_dim")) {
    p.field_dim = j["field_dim"].get<int>();
  }
  if (j.contains("world_n")) {
    p.world_n = j["world_n"].get<std::int64_t>();
  }
  if (j.contains("world_mu")) {
    p.world_mu = read_vec1d(j, "world_mu");
  } else if (j.contains("world") && j["world"].is_object()) {
    const auto& w = j["world"];
    if (w.contains("n")) {
      p.world_n = w["n"].get<std::int64_t>();
    }
    if (w.contains("mu")) {
      p.world_mu = read_vec1d(w, "mu");
    }
    if (w.contains("v")) {
      p.world_v = read_vec1d(w, "v");
    }
  }
  if (p.world_v.empty() && j.contains("world_v")) {
    p.world_v = read_vec1d(j, "world_v");
  }
  if (p.d_latent <= 0 && !p.world_mu.empty()) {
    p.d_latent = static_cast<int>(p.world_mu.size());
  }
  if (p.field_dim <= 0) {
    p.field_dim = 16;
  }

  const nlohmann::json* classes = nullptr;
  if (j.contains("classes") && j["classes"].is_array()) {
    classes = &j["classes"];
  } else if (j.contains("class_stats") && j["class_stats"].is_array()) {
    classes = &j["class_stats"];
  }
  if (classes != nullptr) {
    for (const auto& row : *classes) {
      p.labels.push_back(row.at("label").get<std::string>());
      p.n_obs.push_back(row.value("n_obs", 0.0));
      p.n_correct.push_back(row.value("n_correct", static_cast<std::int64_t>(0)));
      const auto dm = read_vec1d(row, "delta_mu");
      p.delta_mu.insert(p.delta_mu.end(), dm.begin(), dm.end());
    }
  }
  return p;
}

CyphaDifMemoryState cypha_dif_state_from_federated_payload(const FederatedPayload& p) {
  if (p.d_latent <= 0) {
    throw std::runtime_error("federated payload: d_latent unset");
  }
  CyphaDifMemoryState s;
  s.d_latent = p.d_latent;
  s.field_dim = p.field_dim;
  s.world_n = p.world_n;
  s.world_mu = p.world_mu;
  s.world_v = p.world_v;
  if (s.world_v.empty()) {
    s.world_v.assign(static_cast<std::size_t>(s.d_latent), 1.0);
  }
  s.world_inv_v.resize(static_cast<std::size_t>(s.d_latent));
  for (int j = 0; j < s.d_latent; ++j) {
    s.world_inv_v[static_cast<std::size_t>(j)] =
        1.0 / std::max(s.world_v[static_cast<std::size_t>(j)], kEps);
  }
  s.refresh_world_log_norm_from_v();
  double sum_v = 0.0;
  for (int j = 0; j < s.d_latent; ++j) {
    sum_v += s.world_v[static_cast<std::size_t>(j)];
  }
  s.world_v_mean = sum_v / static_cast<double>(s.d_latent);

  s.labels = p.labels;
  s.n_obs_buf = p.n_obs;
  s.n_correct = p.n_correct;
  s.D = p.delta_mu;
  if (s.D.size() != s.labels.size() * static_cast<std::size_t>(s.d_latent)) {
    s.D.assign(s.labels.size() * static_cast<std::size_t>(s.d_latent), 0.0);
  }
  for (std::size_t k = 0; k < s.labels.size(); ++k) {
    s.label_index[s.labels[k]] = static_cast<int>(k);
  }
  return s;
}

CyphaDifMemoryState federated_average_payloads(const std::vector<FederatedPayload>& payloads) {
  if (payloads.empty()) {
    throw std::runtime_error("federated_average_payloads: no payloads");
  }
  CyphaDifMemoryState acc = cypha_dif_state_from_federated_payload(payloads.front());
  for (std::size_t i = 1; i < payloads.size(); ++i) {
    const CyphaDifMemoryState other = cypha_dif_state_from_federated_payload(payloads[i]);
    memory_merge_from(acc, other, 1.0, 1.0);
  }
  return acc;
}

nlohmann::json federated_payload_to_json(const FederatedPayload& p) {
  nlohmann::json classes = nlohmann::json::array();
  const int d = p.d_latent;
  for (std::size_t k = 0; k < p.labels.size(); ++k) {
    nlohmann::json row;
    row["label"] = p.labels[k];
    row["n_obs"] = k < p.n_obs.size() ? p.n_obs[k] : 0.0;
    row["n_correct"] = k < p.n_correct.size() ? p.n_correct[k] : 0;
    nlohmann::json dm = nlohmann::json::array();
    for (int j = 0; j < d; ++j) {
      dm.push_back(p.delta_mu[k * static_cast<std::size_t>(d) + static_cast<std::size_t>(j)]);
    }
    row["delta_mu"] = dm;
    classes.push_back(row);
  }
  return nlohmann::json{
      {"d_latent", p.d_latent},
      {"field_dim", p.field_dim},
      {"world_n", p.world_n},
      {"world_mu", p.world_mu},
      {"world_v", p.world_v},
      {"classes", classes},
  };
}

}  // namespace cypha
