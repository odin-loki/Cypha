// cypha_federated_coordinator — collect worker JSON payloads and merge via federated_average_payloads.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/federated_aggregate.hpp"
#include "httplib.h"

namespace fs = std::filesystem;

namespace {

void usage() {
  std::cerr << "usage: cypha_federated_coordinator\n"
            << "  --watch-dir <dir> [--poll-ms N] [--min-workers N] [--once] --out <merged.json>\n"
            << "  --listen <port> [--min-workers N] [--once] --out <merged.json>\n"
            << "  --merge worker1.json [worker2.json ...] --out <merged.json>\n";
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
  out["coordinator"] = "cypha_federated_coordinator";

  std::ofstream fo(out_path);
  if (!fo) {
    throw std::runtime_error("cannot write: " + out_path);
  }
  fo << out.dump(2) << std::endl;
}

cypha::CyphaDifMemoryState merge_paths(const std::vector<std::string>& paths) {
  std::vector<cypha::FederatedPayload> payloads;
  payloads.reserve(paths.size());
  for (const auto& path : paths) {
    payloads.push_back(cypha::federated_payload_from_json(read_json_file(path)));
  }
  return cypha::federated_average_payloads(payloads);
}

std::vector<std::string> list_json_files(const fs::path& dir) {
  std::vector<std::string> out;
  if (!fs::is_directory(dir)) {
    return out;
  }
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".json") {
      out.push_back(entry.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool run_watch_dir(const fs::path& dir, const std::string& out_path, int min_workers, int poll_ms, bool once) {
  for (;;) {
    const auto files = list_json_files(dir);
    if (static_cast<int>(files.size()) >= min_workers) {
      const auto merged = merge_paths(files);
      write_merged(merged, out_path, static_cast<int>(files.size()));
      std::cout << "merged " << files.size() << " workers -> " << out_path << std::endl;
      if (once) {
        return true;
      }
    }
    if (!once) {
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    } else {
      return false;
    }
  }
}

bool run_http_server(int port, const std::string& out_path, int min_workers, bool once) {
  httplib::Server svr;
  std::vector<nlohmann::json> payloads;
  std::mutex mu;

  svr.Post("/submit", [&](const httplib::Request& req, httplib::Response& res) {
    try {
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
        std::cout << "merged " << parsed.size() << " workers -> " << out_path << std::endl;
        if (once) {
          svr.stop();
        }
      }
    } catch (const std::exception& ex) {
      res.status = 400;
      res.set_content(nlohmann::json{{"detail", ex.what()}}.dump(), "application/json");
    }
  });

  svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(mu);
    res.set_content(nlohmann::json{{"workers", payloads.size()}, {"min_workers", min_workers}}.dump(),
                    "application/json");
  });

  std::cout << "coordinator listening on port " << port << std::endl;
  return svr.listen("127.0.0.1", port);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> merge_inputs;
    std::string out_path;
    std::string watch_dir;
    int listen_port = 0;
    int min_workers = 2;
    int poll_ms = 200;
    bool once = false;

    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (k == "--merge") {
        while (i + 1 < argc && argv[i + 1][0] != '-') {
          merge_inputs.push_back(argv[++i]);
        }
      } else if (k == "--out") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --out");
        }
        out_path = argv[++i];
      } else if (k == "--watch-dir") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --watch-dir");
        }
        watch_dir = argv[++i];
      } else if (k == "--listen") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --listen");
        }
        listen_port = std::stoi(argv[++i]);
      } else if (k == "--min-workers") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --min-workers");
        }
        min_workers = std::stoi(argv[++i]);
      } else if (k == "--poll-ms") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --poll-ms");
        }
        poll_ms = std::stoi(argv[++i]);
      } else if (k == "--once") {
        once = true;
      } else if (k == "--help" || k == "-h") {
        usage();
        return 0;
      } else {
        throw std::runtime_error("unknown arg: " + k);
      }
    }

    if (out_path.empty()) {
      throw std::runtime_error("--out is required");
    }
    if (min_workers < 1) {
      throw std::runtime_error("--min-workers must be >= 1");
    }

    if (!merge_inputs.empty()) {
      if (static_cast<int>(merge_inputs.size()) < min_workers) {
        throw std::runtime_error("need at least --min-workers input files for --merge");
      }
      const auto merged = merge_paths(merge_inputs);
      write_merged(merged, out_path, static_cast<int>(merge_inputs.size()));
      return 0;
    }

    if (!watch_dir.empty()) {
      const bool ok = run_watch_dir(fs::path(watch_dir), out_path, min_workers, poll_ms, once);
      return ok ? 0 : 2;
    }

    if (listen_port > 0) {
      const bool ok = run_http_server(listen_port, out_path, min_workers, once);
      return ok ? 0 : 1;
    }

    usage();
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "cypha_federated_coordinator: " << ex.what() << "\n";
    usage();
    return 1;
  }
}
