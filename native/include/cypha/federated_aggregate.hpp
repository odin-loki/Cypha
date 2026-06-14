#pragma once

#include "cypha/memory_train.hpp"

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha {

/// Minimal federated payload: world prior + per-class stats (see docs/FUTURE.md §8).
struct FederatedPayload {
  int d_latent{0};
  int field_dim{0};
  std::int64_t world_n{0};
  std::vector<double> world_mu;
  std::vector<double> world_v;
  std::vector<std::string> labels;
  std::vector<double> n_obs;
  std::vector<std::int64_t> n_correct;
  std::vector<double> delta_mu;
};

/// Parse a worker JSON export (``world`` + ``classes`` arrays).
FederatedPayload federated_payload_from_json(const nlohmann::json& j);

CyphaDifMemoryState cypha_dif_state_from_federated_payload(const FederatedPayload& p);

/// Average ``payloads`` into a single memory state (Fisher–Rao weighted class merge).
CyphaDifMemoryState federated_average_payloads(const std::vector<FederatedPayload>& payloads);

nlohmann::json federated_payload_to_json(const FederatedPayload& p);

}  // namespace cypha
