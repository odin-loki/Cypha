/// Loopback smoke: in-process coordinator HTTP /submit + worker-style POST.
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/federated_aggregate.hpp"
#include "httplib.h"

namespace fs = std::filesystem;

namespace {

nlohmann::json read_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open: " + path);
  }
  nlohmann::json j;
  in >> j;
  return j;
}

void write_merged(const cypha::CyphaDifMemoryState& merged, const std::string& out_path, int n_workers) {
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
  out["n_workers"] = n_workers;
  out["coordinator"] = "federated_worker_smoke";

  std::ofstream fo(out_path);
  if (!fo) {
    throw std::runtime_error("cannot write: " + out_path);
  }
  fo << out.dump(2) << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::fprintf(stderr, "usage: federated_worker_smoke <worker_a.json> <worker_b.json> <out.json>\n");
      return 2;
    }

    const std::string worker_a = argv[1];
    const std::string worker_b = argv[2];
    const std::string out_path = argv[3];

    constexpr int kPort = 19876;
    httplib::Server svr;
    std::vector<nlohmann::json> payloads;
    std::mutex mu;
    const int min_workers = 2;

    svr.Post("/submit", [&](const httplib::Request& req, httplib::Response& res) {
      const nlohmann::json body = nlohmann::json::parse(req.body);
      std::lock_guard<std::mutex> lk(mu);
      payloads.push_back(body);
      nlohmann::json ack{{"accepted", true}, {"worker_count", payloads.size()}};
      res.set_content(ack.dump(), "application/json");
      if (static_cast<int>(payloads.size()) >= min_workers) {
        std::vector<cypha::FederatedPayload> parsed;
        parsed.reserve(payloads.size());
        for (const auto& j : payloads) {
          parsed.push_back(cypha::federated_payload_from_json(j));
        }
        const auto merged = cypha::federated_average_payloads(parsed);
        write_merged(merged, out_path, static_cast<int>(parsed.size()));
        svr.stop();
      }
    });

    std::thread server_thread([&]() { (void)svr.listen("127.0.0.1", kPort); });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    httplib::Client cli("127.0.0.1", kPort);
    for (const auto& path : {worker_a, worker_b}) {
      const auto body = read_json_file(path).dump();
      const auto res = cli.Post("/submit", body, "application/json");
      assert(res);
      assert(res->status == 200);
    }

    server_thread.join();
    assert(fs::exists(out_path));

    const auto merged = read_json_file(out_path);
    assert(merged.value("n_workers", 0) == 2);
    assert(merged.contains("classes"));

    std::printf("federated_worker_smoke: merged -> %s PASS\n", out_path.c_str());
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "federated_worker_smoke: %s\n", ex.what());
    return 1;
  }
}
