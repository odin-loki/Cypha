/// Federated TLS loopback smoke: self-signed cert + HTTPS coordinator/worker POST.
///
/// Exit codes:
///   0  TLS submit/merge succeeded
///   2  skip (OpenSSL not enabled at build time, or openssl CLI unavailable)
///   1  failure
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/federated_aggregate.hpp"
#include "httplib.h"

namespace fs = std::filesystem;

namespace {

#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
int skip_no_openssl_build() {
  std::printf("SKIP: federated_tls_smoke requires -DCYPHA_ENABLE_OPENSSL=ON\n");
  return 2;
}
#else

bool openssl_cli_available() {
#ifdef _WIN32
  return std::system("openssl version >NUL 2>&1") == 0;
#else
  return std::system("openssl version >/dev/null 2>&1") == 0;
#endif
}

bool generate_self_signed_cert(const fs::path& cert_path, const fs::path& key_path) {
  const std::string cmd =
      "openssl req -x509 -newkey rsa:2048 -keyout \"" + key_path.string() + "\" -out \"" +
      cert_path.string() +
      "\" -days 1 -nodes -subj \"/CN=localhost\""
#ifdef _WIN32
      " -config NUL"
      " >NUL 2>&1"
#else
      " >/dev/null 2>&1"
#endif
      ;
  return std::system(cmd.c_str()) == 0;
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
  out["coordinator"] = "federated_tls_smoke";
  out["tls"] = true;

  std::ofstream fo(out_path);
  if (!fo) {
    throw std::runtime_error("cannot write: " + out_path);
  }
  fo << out.dump(2) << std::endl;
}

#endif

}  // namespace

int main(int argc, char** argv) {
#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
  (void)argc;
  (void)argv;
  return skip_no_openssl_build();
#else
  try {
    if (argc < 4) {
      std::fprintf(stderr, "usage: federated_tls_smoke <worker_a.json> <worker_b.json> <out.json>\n");
      return 2;
    }

    if (!openssl_cli_available()) {
      std::printf("SKIP: openssl CLI not found on PATH\n");
      return 2;
    }

    const std::string worker_a = argv[1];
    const std::string worker_b = argv[2];
    const std::string out_path = argv[3];

    const fs::path cert_dir = fs::temp_directory_path() / "cypha_federated_tls_smoke";
    fs::create_directories(cert_dir);
    const fs::path cert_path = cert_dir / "cert.pem";
    const fs::path key_path = cert_dir / "key.pem";
    std::error_code rm_ec;
    fs::remove(cert_path, rm_ec);
    fs::remove(key_path, rm_ec);
    if (!generate_self_signed_cert(cert_path, key_path)) {
      std::printf("SKIP: openssl req failed to generate self-signed cert\n");
      return 2;
    }

    constexpr int kPort = 19877;
    std::vector<nlohmann::json> payloads;
    std::mutex mu;
    const int min_workers = 2;

    httplib::SSLServer svr(cert_path.string().c_str(), key_path.string().c_str());
    if (!svr.is_valid()) {
      throw std::runtime_error("invalid TLS cert/key for federated_tls_smoke");
    }

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
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    for (const auto& path : {worker_a, worker_b}) {
      httplib::SSLClient cli("127.0.0.1", kPort);
      cli.enable_server_certificate_verification(false);
      const auto body = read_json_file(path).dump();
      const auto res = cli.Post("/submit", body, "application/json");
      assert(res);
      assert(res->status == 200);
    }

    server_thread.join();
    assert(fs::exists(out_path));

    const auto merged = read_json_file(out_path);
    assert(merged.value("n_workers", 0) == 2);
    assert(merged.value("tls", false) == true);
    assert(merged.contains("classes"));

    std::printf("federated_tls_smoke: merged -> %s PASS\n", out_path.c_str());
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "federated_tls_smoke: %s\n", ex.what());
    return 1;
  }
#endif
}
