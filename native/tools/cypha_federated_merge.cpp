// cypha_federated_merge — average federated worker JSON payloads (world_mu + class stats).
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/federated_aggregate.hpp"

namespace {

void usage() {
  std::cerr << "usage: cypha_federated_merge --merge worker1.json [worker2.json ...] [--out merged.json]\n";
}

nlohmann::json read_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open: " + path);
  }
  nlohmann::json j;
  in >> j;
  return j;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> inputs;
    std::string out_path;
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (k == "--merge") {
        while (i + 1 < argc && argv[i + 1][0] != '-') {
          inputs.push_back(argv[++i]);
        }
      } else if (k == "--out") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --out");
        }
        out_path = argv[++i];
      } else if (k == "--help" || k == "-h") {
        usage();
        return 0;
      } else {
        throw std::runtime_error("unknown arg: " + k);
      }
    }
    if (inputs.size() < 2) {
      usage();
      return 2;
    }

    std::vector<cypha::FederatedPayload> payloads;
    payloads.reserve(inputs.size());
    for (const auto& path : inputs) {
      payloads.push_back(cypha::federated_payload_from_json(read_json_file(path)));
    }
    const cypha::CyphaDifMemoryState merged = cypha::federated_average_payloads(payloads);

    nlohmann::json out;
    out["d_latent"] = merged.d_latent;
    out["field_dim"] = merged.field_dim;
    out["world_n"] = merged.world_n;
    out["world_mu"] = merged.world_mu;
    out["world_v"] = merged.world_v;
    nlohmann::json classes = nlohmann::json::array();
    for (std::size_t k = 0; k < merged.labels.size(); ++k) {
      nlohmann::json row;
      row["label"] = merged.labels[k];
      row["n_obs"] = merged.n_obs_buf[k];
      row["n_correct"] = merged.n_correct[k];
      nlohmann::json dm = nlohmann::json::array();
      for (int j = 0; j < merged.d_latent; ++j) {
        dm.push_back(merged.D[k * static_cast<std::size_t>(merged.d_latent) + static_cast<std::size_t>(j)]);
      }
      row["delta_mu"] = dm;
      classes.push_back(row);
    }
    out["classes"] = classes;
    out["n_workers"] = inputs.size();

    const std::string dumped = out.dump(2);
    if (out_path.empty()) {
      std::cout << dumped << std::endl;
    } else {
      std::ofstream fo(out_path);
      if (!fo) {
        throw std::runtime_error("cannot write: " + out_path);
      }
      fo << dumped << std::endl;
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "cypha_federated_merge: " << ex.what() << "\n";
    usage();
    return 1;
  }
}
