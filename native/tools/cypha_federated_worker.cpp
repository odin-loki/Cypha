// cypha_federated_worker — POST a local federated JSON payload to a coordinator /submit endpoint.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "httplib.h"

namespace {

void usage() {
  std::cerr << "usage: cypha_federated_worker --payload <worker.json> --coordinator <host:port>\n"
            << "  [--path /submit] [--scheme http|https]\n"
            << "\n"
            << "Without CPPHTTPLIB_OPENSSL_SUPPORT, https requires CYPHA_FEDERATED_INSECURE=1.\n";
}

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::pair<std::string, int> parse_host_port(const std::string& host_port) {
  const auto colon = host_port.rfind(':');
  if (colon == std::string::npos || colon + 1 >= host_port.size()) {
    throw std::runtime_error("expected host:port for --coordinator");
  }
  const std::string host = host_port.substr(0, colon);
  const int port = std::stoi(host_port.substr(colon + 1));
  return {host, port};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string payload_path;
    std::string coordinator;
    std::string path = "/submit";
    std::string scheme = "http";

    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (k == "--payload") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --payload");
        }
        payload_path = argv[++i];
      } else if (k == "--coordinator") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --coordinator");
        }
        coordinator = argv[++i];
      } else if (k == "--path") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --path");
        }
        path = argv[++i];
      } else if (k == "--scheme") {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --scheme");
        }
        scheme = argv[++i];
      } else if (k == "--help" || k == "-h") {
        usage();
        return 0;
      } else {
        throw std::runtime_error("unknown arg: " + k);
      }
    }

    if (payload_path.empty() || coordinator.empty()) {
      usage();
      return 2;
    }

    const auto [host, port] = parse_host_port(coordinator);
    const std::string body = read_file(payload_path);

#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    if (scheme == "https") {
      const char* insecure = std::getenv("CYPHA_FEDERATED_INSECURE");
      if (insecure == nullptr || insecure[0] != '1') {
        throw std::runtime_error(
            "https requested but OpenSSL is unavailable; set CYPHA_FEDERATED_INSECURE=1 or use http");
      }
      std::cerr << "warning: https downgraded to http (no OpenSSL)\n";
      scheme = "http";
    }
#endif

    httplib::Result res;
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    if (scheme == "https") {
      httplib::SSLClient cli(host, port);
      cli.enable_server_certificate_verification(false);
      res = cli.Post(path.c_str(), body, "application/json");
    } else
#endif
    {
      httplib::Client cli(host, port);
      res = cli.Post(path.c_str(), body, "application/json");
    }

    if (!res) {
      throw std::runtime_error("POST failed: transport error");
    }
    if (res->status < 200 || res->status >= 300) {
      throw std::runtime_error("POST failed: HTTP " + std::to_string(res->status) + " " + res->body);
    }

    std::cout << res->body << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "cypha_federated_worker: " << ex.what() << "\n";
    usage();
    return 1;
  }
}
