// Validates REST JSON key shapes for cypha_rest against docs/port/PORT_CONTRACT.md §3.
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "httplib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string quote_arg(const std::string& s) {
  if (s.find(' ') == std::string::npos && s.find('"') == std::string::npos) {
    return s;
  }
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') {
      out += "\\\"";
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

#ifdef _WIN32
#include <process.h>
#include <windows.h>

struct ChildProcess {
  intptr_t pid{-1};

  ChildProcess() = default;
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ChildProcess(ChildProcess&& other) noexcept : pid(other.pid) { other.pid = -1; }
  ChildProcess& operator=(ChildProcess&& other) noexcept {
    if (this != &other) {
      reset();
      pid = other.pid;
      other.pid = -1;
    }
    return *this;
  }

  ~ChildProcess() { reset(); }

  void reset() {
    if (pid > 0) {
      HANDLE proc = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
      if (proc != nullptr) {
        TerminateProcess(proc, 1);
        WaitForSingleObject(proc, 5000);
        CloseHandle(proc);
      }
      pid = -1;
    }
  }
};

std::vector<char*> build_argv(const fs::path& exe, const std::vector<std::string>& args,
                              std::vector<std::string>& storage) {
  storage.clear();
  storage.push_back(exe.string());
  storage.insert(storage.end(), args.begin(), args.end());
  std::vector<char*> argv;
  argv.reserve(storage.size() + 1);
  for (auto& s : storage) {
    argv.push_back(s.data());
  }
  argv.push_back(nullptr);
  return argv;
}

ChildProcess spawn_process(const fs::path& exe, const std::vector<std::string>& args,
                           const fs::path& work_dir) {
  const fs::path prev = fs::current_path();
  fs::current_path(work_dir);
  std::vector<std::string> storage;
  std::vector<char*> argv = build_argv(exe, args, storage);
  const intptr_t pid = _spawnv(_P_NOWAIT, exe.string().c_str(), argv.data());
  fs::current_path(prev);
  if (pid == -1) {
    throw std::runtime_error("_spawnv failed for " + exe.string() + " (errno " +
                             std::to_string(errno) + ")");
  }
  ChildProcess child;
  child.pid = pid;
  return child;
}

#else
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct ChildProcess {
  pid_t pid{-1};

  ChildProcess() = default;
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ChildProcess(ChildProcess&& other) noexcept : pid(other.pid) { other.pid = -1; }
  ChildProcess& operator=(ChildProcess&& other) noexcept {
    if (this != &other) {
      reset();
      pid = other.pid;
      other.pid = -1;
    }
    return *this;
  }

  ~ChildProcess() { reset(); }

  void reset() {
    if (pid > 0) {
      kill(pid, SIGTERM);
      waitpid(pid, nullptr, 0);
      pid = -1;
    }
  }
};

ChildProcess spawn_process(const fs::path& exe, const std::vector<std::string>& args,
                           const fs::path& work_dir) {
  pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("fork failed");
  }
  if (pid == 0) {
    if (chdir(work_dir.c_str()) != 0) {
      _exit(127);
    }
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (const auto& a : args) {
      argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);
    execv(exe.c_str(), argv.data());
    _exit(127);
  }
  ChildProcess child;
  child.pid = pid;
  return child;
}
#endif

const std::vector<std::string> kModelCardKeys = {
    "name",           "version",           "description",       "author",
    "date",           "task",              "model_type",        "encoder_type",
    "feat_dim",       "field_dim",         "n_classes",         "class_labels",
    "input_dim",      "dataset_name",      "n_train",           "n_val",
    "train_steps",    "training_time_s",   "val_accuracy",      "val_f1",
    "val_r2",         "val_rmse",          "calibration_error", "ood_auroc",
    "gh_protected",   "stage",             "model_file",        "preprocessor_file",
    "card_file",      "intended_use",      "known_limitations", "training_data_desc",
};

const std::vector<std::string> kHealthKeys = {"status", "model", "uptime", "n_predictions"};

const std::vector<std::string> kPredictKeys = {
    "label", "confidence", "all_scores", "anomaly_score", "is_ood",
    "regression_val", "uncertainty", "explanation", "latency_ms",
};

const std::vector<std::string> kUpdateKeys = {"loss", "n_corrections"};

const std::vector<std::string> kReadyKeys = {"ready", "model_type"};

const std::vector<std::string> kMetricsKeys = {
    "uptime_seconds",  "model_loaded",          "model_type",     "n_predictions",
    "n_corrections",   "registry_model_count",  "loaded_model_count", "active_model",
    "session",         "regression_head_loaded", "lm_loaded",      "branch_a_router",
};

const std::vector<std::string> kSessionSummaryKeys = {
    "n_predictions",  "n_corrections",     "correction_accuracy", "mean_confidence",
    "mean_anomaly",   "n_ood_flagged",     "label_distribution",  "session_duration_s",
};

struct HttpResult {
  int status{0};
  json body;
};

HttpResult http_json(httplib::Client& cli, const char* method, const char* path,
                     const std::string& req_body = {}) {
  httplib::Result res;
  if (std::string(method) == "GET") {
    res = cli.Get(path);
  } else if (std::string(method) == "POST") {
    res = cli.Post(path, req_body, "application/json");
  } else if (std::string(method) == "DELETE") {
    res = cli.Delete(path);
  } else {
    throw std::runtime_error(std::string("unsupported method: ") + method);
  }
  if (!res) {
    throw std::runtime_error(std::string("HTTP failed: ") + path + " (" +
                             httplib::to_string(res.error()) + ")");
  }
  HttpResult out;
  out.status = res->status;
  try {
    out.body = json::parse(res->body);
  } catch (...) {
    throw std::runtime_error("non-JSON response from " + std::string(path) + ": " + res->body);
  }
  return out;
}

bool has_exact_keys(const json& j, const std::vector<std::string>& keys) {
  if (!j.is_object() || j.size() != keys.size()) {
    return false;
  }
  for (const auto& k : keys) {
    if (!j.contains(k)) {
      return false;
    }
  }
  return true;
}

void require_keys(const json& j, const std::vector<std::string>& keys, const char* ctx) {
  for (const auto& k : keys) {
    if (!j.is_object() || !j.contains(k)) {
      throw std::runtime_error(std::string(ctx) + ": missing key '" + k + "'");
    }
  }
}

void require_exact_keys(const json& j, const std::vector<std::string>& keys, const char* ctx) {
  if (!has_exact_keys(j, keys)) {
    std::ostringstream msg;
    msg << ctx << ": expected exactly {";
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (i) {
        msg << ", ";
      }
      msg << keys[i];
    }
    msg << "}, got " << j.dump();
    throw std::runtime_error(msg.str());
  }
}

void wait_for_health(httplib::Client& cli) {
  // Up to ~60s: model preload can be slow under CI load after a long CTest suite.
  for (int i = 0; i < 240; ++i) {
    auto res = cli.Get("/health");
    if (res && res->status == 200) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  throw std::runtime_error("server did not become ready");
}

int run_registry_register(const fs::path& reg_exe, const fs::path& reg_root, const fs::path& cypha,
                          const fs::path& card) {
  std::vector<std::string> args = {
      reg_root.string(), "native_schema_smoke", "0.0.1", cypha.string(), card.string(),
      "--overwrite",     "--and-verify",
  };
#ifdef _WIN32
  std::vector<std::string> storage;
  std::vector<char*> argv = build_argv(reg_exe, args, storage);
  const int rc = _spawnv(_P_WAIT, reg_exe.string().c_str(), argv.data());
  if (rc == -1) {
    throw std::runtime_error("registry_register spawn failed (errno " + std::to_string(errno) + ")");
  }
  return rc;
#else
  ChildProcess child = spawn_process(reg_exe, args, reg_exe.parent_path());
  int status = 0;
  waitpid(child.pid, &status, 0);
  child.pid = -1;
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return 1;
#endif
}

struct Options {
  fs::path cypha_rest;
  fs::path registry_register;
  fs::path repo_root;
  fs::path build_dir;
  std::string base_url;
  bool spawn{true};
};

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--cypha-rest" && i + 1 < argc) {
      opt.cypha_rest = fs::absolute(argv[++i]);
    } else if (a == "--registry-register" && i + 1 < argc) {
      opt.registry_register = fs::absolute(argv[++i]);
    } else if (a == "--repo-root" && i + 1 < argc) {
      opt.repo_root = fs::absolute(argv[++i]);
    } else if (a == "--build-dir" && i + 1 < argc) {
      opt.build_dir = fs::absolute(argv[++i]);
    } else if (a == "--base-url" && i + 1 < argc) {
      opt.base_url = argv[++i];
      opt.spawn = false;
    } else if (a == "--help" || a == "-h") {
      std::fprintf(stderr,
                   "usage: rest_schema_contract --cypha-rest EXE --registry-register EXE "
                   "[--repo-root DIR] [--build-dir DIR]\n"
                   "   or: rest_schema_contract --base-url http://127.0.0.1:PORT/\n");
      std::exit(2);
    }
  }
  if (opt.repo_root.empty()) {
    if (const char* env = std::getenv("CYPHA_REPO_ROOT")) {
      opt.repo_root = fs::absolute(env);
    } else {
      opt.repo_root = fs::absolute(fs::current_path());
    }
  }
  if (opt.build_dir.empty()) {
    opt.build_dir = fs::current_path();
  }
  return opt;
}

void validate_contract(httplib::Client& cli) {
  const HttpResult health = http_json(cli, "GET", "/health");
  if (health.status != 200) {
    throw std::runtime_error("/health status " + std::to_string(health.status));
  }
  require_keys(health.body, kHealthKeys, "/health");

  const HttpResult ready = http_json(cli, "GET", "/ready");
  // The spawned server always has a model preloaded via --cypha, so /ready is expected to
  // report 200/ready:true here; a 503 (no model loaded) shape is also tolerated so this test
  // stays valid if the harness is ever pointed at a server without a preloaded model.
  if (ready.status == 200) {
    require_keys(ready.body, kReadyKeys, "/ready");
  } else if (ready.status == 503) {
    require_keys(ready.body, {"ready", "reason"}, "/ready (no model loaded)");
  } else {
    throw std::runtime_error("/ready unexpected status " + std::to_string(ready.status));
  }

  const HttpResult metrics = http_json(cli, "GET", "/metrics");
  if (metrics.status != 200) {
    throw std::runtime_error("/metrics status " + std::to_string(metrics.status));
  }
  require_keys(metrics.body, kMetricsKeys, "/metrics");

  const std::string predict_body = R"({"input":[0,0,0,0,0,0,0,0],"use_gh":true})";
  const HttpResult predict = http_json(cli, "POST", "/predict", predict_body);
  if (predict.status != 200) {
    throw std::runtime_error("/predict status " + std::to_string(predict.status));
  }
  require_keys(predict.body, kPredictKeys, "/predict");

  const std::string update_body =
      R"({"input":[0,0,0,0,0,0,0,0],"correct_label":"0","use_gh":true})";
  const HttpResult update = http_json(cli, "POST", "/update", update_body);
  if (update.status != 200) {
    throw std::runtime_error("/update status " + std::to_string(update.status));
  }
  require_exact_keys(update.body, kUpdateKeys, "/update");

  const HttpResult classes = http_json(cli, "GET", "/classes");
  // Same "model may or may not be loaded" tolerance as /ready above.
  if (classes.status == 200) {
    if (!classes.body.contains("classes") || !classes.body["classes"].is_object()) {
      throw std::runtime_error("/classes.classes must be object");
    }
    for (const auto& [label, row] : classes.body["classes"].items()) {
      (void)label;
      require_keys(row, {"n_obs"}, "/classes[]");
    }
  } else if (classes.status == 503) {
    require_keys(classes.body, {"detail"}, "/classes (no model loaded)");
  } else {
    throw std::runtime_error("/classes unexpected status " + std::to_string(classes.status));
  }

  const HttpResult models = http_json(cli, "GET", "/models");
  if (models.status != 200) {
    throw std::runtime_error("/models status " + std::to_string(models.status));
  }
  if (!models.body.contains("models") || !models.body["models"].is_array()) {
    throw std::runtime_error("/models.models must be array");
  }
  if (!models.body.contains("active_model")) {
    throw std::runtime_error("/models missing active_model");
  }
  if (!models.body["models"].empty()) {
    const json& row = models.body["models"][0];
    require_keys(row, {"name", "version", "loaded", "active"}, "/models[]");
    for (const auto& k : kModelCardKeys) {
      if (!row.contains(k)) {
        throw std::runtime_error("/models[] missing ModelCard key '" + k + "'");
      }
    }
  }

  const HttpResult models_summary = http_json(cli, "GET", "/models?summary=true");
  if (models_summary.status != 200) {
    throw std::runtime_error("/models?summary=true status " + std::to_string(models_summary.status));
  }
  if (!models_summary.body["models"].empty()) {
    require_keys(models_summary.body["models"][0], {"name", "version", "loaded", "active"},
                 "/models?summary=true[]");
  }

  const std::string load_body = R"({"name":"native_schema_smoke","version":"0.0.1"})";
  const HttpResult load = http_json(cli, "POST", "/load", load_body);
  if (load.status != 200) {
    throw std::runtime_error("/load status " + std::to_string(load.status));
  }
  if (!load.body.contains("loaded") || !load.body["loaded"].is_object()) {
    throw std::runtime_error("/load.loaded must be object");
  }
  for (const auto& k : kModelCardKeys) {
    if (!load.body["loaded"].contains(k)) {
      throw std::runtime_error("/load.loaded missing ModelCard key '" + k + "'");
    }
  }

  // /session only exposes GET (summary) and DELETE (clear) — there is no POST /session
  // route to create a session; a session accumulates implicitly from /predict + /update.
  const HttpResult session = http_json(cli, "GET", "/session");
  if (session.status != 200) {
    throw std::runtime_error("/session status " + std::to_string(session.status));
  }
  require_exact_keys(session.body, kSessionSummaryKeys, "/session");

  const HttpResult session_delete = http_json(cli, "DELETE", "/session");
  if (session_delete.status != 200) {
    throw std::runtime_error("DELETE /session status " + std::to_string(session_delete.status));
  }
  require_keys(session_delete.body, {"cleared"}, "DELETE /session");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);

    std::string host = "127.0.0.1";
    int port = 18103;
    ChildProcess child{};

    if (opt.spawn) {
      if (opt.cypha_rest.empty() || opt.registry_register.empty()) {
        std::fprintf(stderr, "rest_schema_contract: --cypha-rest and --registry-register required\n");
        return 2;
      }

      const fs::path cypha = opt.repo_root / "fixtures/reference.cypha";
      const fs::path ff = opt.repo_root / "fixtures/f_field.json";
      const fs::path card = opt.repo_root / "fixtures/registry_register/card.json";
      const fs::path reg_root = opt.build_dir / "rest_schema_registry";

      if (fs::exists(reg_root)) {
        fs::remove_all(reg_root);
      }
      fs::create_directories(reg_root);

      if (run_registry_register(opt.registry_register, reg_root, cypha, card) != 0) {
        throw std::runtime_error("registry_register failed");
      }
      fs::copy_file(ff, reg_root / "native_schema_smoke/0.0.1/f_field.json",
                    fs::copy_options::overwrite_existing);

      const fs::path model_in_reg = reg_root / "native_schema_smoke/0.0.1/model.cypha";
      std::vector<std::string> args = {
          "--listen", host + ":" + std::to_string(port), "--cypha", model_in_reg.string(),
          "--f-field-json", ff.string(), "--registry", reg_root.string(), "--preload-registry",
      };
      child = spawn_process(opt.cypha_rest, args, opt.build_dir);
    } else {
      const std::string url = opt.base_url;
      const auto scheme_end = url.find("://");
      if (scheme_end == std::string::npos) {
        throw std::runtime_error("invalid --base-url");
      }
      const auto host_start = scheme_end + 3;
      const auto path_start = url.find('/', host_start);
      const std::string hostport =
          path_start == std::string::npos ? url.substr(host_start) : url.substr(host_start, path_start - host_start);
      const auto colon = hostport.rfind(':');
      if (colon == std::string::npos) {
        throw std::runtime_error("base-url must include explicit port");
      }
      host = hostport.substr(0, colon);
      port = std::stoi(hostport.substr(colon + 1));
    }

    httplib::Client cli(host, port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(30, 0);
    wait_for_health(cli);
    validate_contract(cli);

    std::printf("rest_schema_contract: OK\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "rest_schema_contract: FAIL — %s\n", e.what());
    return 1;
  }
}
