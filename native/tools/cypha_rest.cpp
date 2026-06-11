// REST surface aligned with cypha_studio.server.api (subset + registry-aware /load).
// Missing-model errors use {"detail":"No model loaded"} with HTTP 503 (same as FastAPI) on POST /predict, /update, /adapt_temperature and GET /classes.
// Malformed JSON on those POSTs returns HTTP 400 {"detail":"bad json"} (FastAPI uses 422 + validation detail for the same case).
// Build: cypha_rest --listen host:port --cypha model.cypha [--f-field-json ff.json] [--pre preprocessor.json]
//        [--train-hparams path] [--registry <root>]
//        `--train-hparams` JSON may include `align_every` (encoder align period; default 500) and
//        `temp_recalib_every` (0 = off), matching `parity_fixtures/train_hparams.json`.
//        If `world.F_field` is in the .cypha blob, `--f-field-json` is optional. Registry `/load` uses
//        `f_field.json` next to the model when the blob has no embedded `F_field`.
//        With `--registry <root>`, `POST /register` copies `model_cypha` + `card_json` (+ optional `preprocessor_json`)
//        paths into `<root>/<name>/<version>/` (see PORT_CONTRACT §3).
//        Optional `--regression-json regression_head.json`: scalar MoE targets per class label → `/predict`
//        fills `regression_val` and `uncertainty` (mixture of expert EMAs; see PORT_CONTRACT §3).
//        Optional top-level `mke` object in that JSON → RFF + expert RLS + router `dif_train_step_vector` on
//        `POST /update` when the body includes numeric `regression_y` (see PORT_CONTRACT §3).
//        Optional `--cyphalm-checkpoint base` or env `CYPHALM_CHECKPOINT` → auto-load CyphaLM at startup.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/mke_scalar_train_step.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/registry.hpp"
#include "cypha/train_step_vector.hpp"
#include "cypha/branch_a_rest.hpp"
#include "cypha/cyphalm/cyphalm_rest.hpp"
#include "cypha/dif_rest.hpp"

namespace fs = std::filesystem;

namespace {

std::mutex g_mu;
std::unique_ptr<cypha::CyphaInferModel> g_model;
std::unique_ptr<cypha::CyphaDifMemoryState> g_mem;
std::unique_ptr<cypha::PreprocessorState> g_pre;
std::string g_registry_root;
std::vector<cypha::RegistryModelRef> g_registry_cache;

std::chrono::steady_clock::time_point g_started = std::chrono::steady_clock::now();
std::chrono::steady_clock::time_point g_sess_started = std::chrono::steady_clock::now();

int g_predictions{0};
int g_engine_corrections{0};

struct SessPred {
  std::string label;
  double confidence{};
  double anomaly_score{};
  bool is_ood{};
};

std::vector<SessPred> g_sess;

double g_world_lr{0.008};
double g_delta_lr{0.05};
double g_ood_sigma{15.0};
std::vector<double> g_gh_inv_v_clean;
double g_gh_R_base{1.0};
double g_gh_chi{1.0};
double g_gh_psi{1.0};
constexpr double kGhNigAdaptAlpha = 0.98;
/// Python ``InferenceEngine.OOD_THRESHOLD`` (anomaly_score > threshold → is_ood).
constexpr double kOodThreshold = 3.0;

std::unique_ptr<cypha::ReplayBuffer> g_replay;
std::mt19937 g_rng{424242};
int g_enc_updates{0};
cypha::TrainStepParams g_tsp{};
int g_total_steps{0};
double g_llr_ema{0.0};

/// Optional scalar MoE head (`regression_head.json`): aligned with `g_model->labels` order.
std::vector<double> g_reg_mu;
std::vector<double> g_reg_var;

/// Optional `mke` block in the same JSON: scalar ``MKERegressor``-style online step via ``mke_scalar_train_step``.
bool g_mke_active{false};
int g_mke_d_in{0};
std::vector<double> g_mke_W;
std::vector<double> g_mke_b;
double g_mke_temperature{1.0};
double g_mke_forgetting{1.0};
double g_mke_pi_floor{0.02};
std::vector<double> g_mke_gh_scales;
std::unordered_map<std::string, std::vector<double>> g_mke_w;
std::unordered_map<std::string, std::vector<double>> g_mke_p;

void clear_mke_state() {
  g_mke_active = false;
  g_mke_d_in = 0;
  g_mke_W.clear();
  g_mke_b.clear();
  g_mke_temperature = 1.0;
  g_mke_forgetting = 1.0;
  g_mke_pi_floor = 0.02;
  g_mke_gh_scales.clear();
  g_mke_w.clear();
  g_mke_p.clear();
}

void reset_session_counters() {
  g_sess.clear();
  g_predictions = 0;
  g_engine_corrections = 0;
  g_sess_started = std::chrono::steady_clock::now();
}

void apply_default_train_hparams() {
  g_world_lr = 0.008;
  g_delta_lr = 0.05;
  g_ood_sigma = 15.0;
  g_tsp.enc_lr = 0.002;
  g_tsp.replay_ratio = 0.30;
  g_tsp.replay_cap = 10000;
  g_tsp.align_every = 500;
  g_tsp.temp_recalib_every = 0;
}

bool try_load_train_hparams_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    return false;
  }
  std::stringstream b;
  b << f.rdbuf();
  nlohmann::json j = nlohmann::json::parse(b.str());
  if (j.contains("world_lr")) {
    g_world_lr = j["world_lr"].get<double>();
  }
  if (j.contains("delta_lr")) {
    g_delta_lr = j["delta_lr"].get<double>();
  }
  if (j.contains("ood_sigma")) {
    g_ood_sigma = j["ood_sigma"].get<double>();
  }
  if (j.contains("enc_lr")) {
    g_tsp.enc_lr = j["enc_lr"].get<double>();
  }
  if (j.contains("replay_ratio")) {
    g_tsp.replay_ratio = j["replay_ratio"].get<double>();
  }
  if (j.contains("replay_cap")) {
    g_tsp.replay_cap = j["replay_cap"].get<int>();
    if (g_tsp.replay_cap < 8) {
      g_tsp.replay_cap = 8;
    }
  }
  if (j.contains("temp_recalib_every")) {
    g_tsp.temp_recalib_every = j["temp_recalib_every"].get<int>();
    if (g_tsp.temp_recalib_every < 0) {
      g_tsp.temp_recalib_every = 0;
    }
  }
  if (j.contains("align_every")) {
    g_tsp.align_every = j["align_every"].get<int>();
    if (g_tsp.align_every < 0) {
      g_tsp.align_every = 0;
    }
  }
  return true;
}

void snapshot_gh_clean_metric() {
  g_gh_chi = 1.0;
  g_gh_psi = 1.0;
  if (!g_model) {
    g_gh_inv_v_clean.clear();
    g_gh_R_base = 1.0;
    return;
  }
  const int d = g_model->d_latent;
  g_gh_inv_v_clean = g_model->inv_v;
  double mean_inv = 0.0;
  for (int j = 0; j < d; ++j) {
    mean_inv += g_gh_inv_v_clean[static_cast<std::size_t>(j)];
  }
  mean_inv /= static_cast<double>(std::max(d, 1));
  constexpr double kEps = 1e-8;
  g_gh_R_base = 1.0 / (mean_inv + kEps);
}

bool load_ff_json(const std::string& path, int d, int fd, std::vector<double>& fflat) {
  fflat.clear();
  if (path.empty()) {
    return false;
  }
  std::ifstream f(path);
  if (!f) {
    return false;
  }
  std::stringstream b;
  b << f.rdbuf();
  auto j = nlohmann::json::parse(b.str());
  for (const auto& row : j) {
    for (const auto& v : row) {
      fflat.push_back(v.get<double>());
    }
  }
  return static_cast<int>(fflat.size()) == d * fd;
}

bool cypha_has_embedded_world_f_field(const cypha::CNode& root, int d, int fd) {
  const cypha::CNode& world = cypha::map_get_required(root, "world");
  const cypha::CNode* wff = cypha::map_get(world, "F_field");
  const int expected = d * fd;
  return wff != nullptr && wff->kind == cypha::CNode::Tensor && wff->shape.size() == 2 &&
         static_cast<int>(wff->shape[0]) == d && static_cast<int>(wff->shape[1]) == fd &&
         static_cast<int>(wff->tensor.size()) == expected;
}

bool try_load_regression_head_json(const std::string& path, const cypha::CyphaInferModel& model) {
  g_reg_mu.clear();
  g_reg_var.clear();
  clear_mke_state();
  if (path.empty()) {
    return true;
  }
  std::ifstream f(path);
  if (!f) {
    return false;
  }
  std::stringstream b;
  b << f.rdbuf();
  nlohmann::json j = nlohmann::json::parse(b.str());
  if (!j.contains("experts") || !j["experts"].is_object()) {
    return false;
  }
  const auto& ex = j["experts"];
  const int k = static_cast<int>(model.labels.size());
  g_reg_mu.assign(static_cast<std::size_t>(k), 0.0);
  g_reg_var.assign(static_cast<std::size_t>(k), 0.0);
  for (int i = 0; i < k; ++i) {
    const std::string& lbl = model.labels[static_cast<std::size_t>(i)];
    if (!ex.contains(lbl)) {
      continue;
    }
    const auto& row = ex[lbl];
    if (row.contains("mu")) {
      if (row["mu"].is_number()) {
        g_reg_mu[static_cast<std::size_t>(i)] = row["mu"].get<double>();
      } else if (row["mu"].is_array() && !row["mu"].empty()) {
        g_reg_mu[static_cast<std::size_t>(i)] = row["mu"][0].get<double>();
      }
    }
    if (row.contains("var_ema") && row["var_ema"].is_number()) {
      g_reg_var[static_cast<std::size_t>(i)] = row["var_ema"].get<double>();
    }
  }

  if (!j.contains("mke") || j["mke"].is_null()) {
    return true;
  }
  if (!j["mke"].is_object()) {
    return false;
  }
  const auto& mk = j["mke"];
  try {
    g_mke_d_in = mk.at("d_in").get<int>();
    const int d_rff = mk.at("D_rff").get<int>();
    if (d_rff != model.d_latent) {
      return false;
    }
    g_mke_W = mk.at("rff_W_rowmajor").get<std::vector<double>>();
    g_mke_b = mk.at("rff_b").get<std::vector<double>>();
    if (static_cast<int>(g_mke_W.size()) != d_rff * g_mke_d_in ||
        static_cast<int>(g_mke_b.size()) != d_rff) {
      return false;
    }
    g_mke_temperature = mk.value("temperature", 1.0);
    g_mke_forgetting = mk.value("forgetting_factor", 1.0);
    g_mke_pi_floor = mk.value("pi_floor", 0.02);

    const auto& wj = mk.at("w");
    const auto& pj = mk.at("P");
    if (!wj.is_object() || !pj.is_object()) {
      return false;
    }
    const std::size_t p_expect = static_cast<std::size_t>(d_rff) * static_cast<std::size_t>(d_rff);
    g_mke_w.clear();
    g_mke_p.clear();
    for (int i = 0; i < k; ++i) {
      const std::string& lbl = model.labels[static_cast<std::size_t>(i)];
      if (!wj.contains(lbl) || !pj.contains(lbl)) {
        return false;
      }
      auto ww = wj[lbl].get<std::vector<double>>();
      auto pp = pj[lbl].get<std::vector<double>>();
      if (static_cast<int>(ww.size()) != d_rff || pp.size() != p_expect) {
        return false;
      }
      g_mke_w[lbl] = std::move(ww);
      g_mke_p[lbl] = std::move(pp);
    }
    g_mke_gh_scales.clear();
    if (mk.contains("gh_scales") && mk["gh_scales"].is_array()) {
      g_mke_gh_scales = mk["gh_scales"].get<std::vector<double>>();
      if (static_cast<int>(g_mke_gh_scales.size()) != k) {
        return false;
      }
    }
    g_mke_active = true;
  } catch (...) {
    clear_mke_state();
    return false;
  }
  return true;
}

bool load_bundle_paths(const std::string& cypha_path, const std::string& pre_path,
                       const std::string& ff_json_path, const std::string& train_hparams_path_opt,
                       const std::string& regression_json_path_opt) {
  apply_default_train_hparams();
  g_reg_mu.clear();
  g_reg_var.clear();
  clear_mke_state();
  cypha::CNode root = cypha::load_cypha_file(cypha_path.c_str());
  const cypha::CNode& fh = cypha::map_get_required(root, "field_h");
  int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root, "enc_W");
  int d = static_cast<int>(enc.shape[0]);

  std::vector<double> fflat;
  const double* ff_ptr = nullptr;
  if (cypha_has_embedded_world_f_field(root, d, fd)) {
    ff_ptr = nullptr;
  } else if (load_ff_json(ff_json_path, d, fd, fflat)) {
    ff_ptr = fflat.data();
  } else {
    std::cerr << "Provide world.F_field inside .cypha or a valid --f-field-json (expected " << (d * fd)
              << " floats)\n";
    return false;
  }

  g_model.reset(new cypha::CyphaInferModel(cypha::CyphaInferModel::from_root(root, ff_ptr, fd)));
  g_mem.reset(
      new cypha::CyphaDifMemoryState(cypha::CyphaDifMemoryState::from_cypha_root(root, ff_ptr, fd)));

  g_pre.reset();
  if (!pre_path.empty()) {
    g_pre.reset(
        new cypha::PreprocessorState(cypha::PreprocessorState::from_json_file(pre_path.c_str())));
  }

  if (!train_hparams_path_opt.empty()) {
    if (!try_load_train_hparams_file(train_hparams_path_opt)) {
      std::cerr << "warning: could not read --train-hparams " << train_hparams_path_opt << "\n";
    }
  } else {
    fs::path auto_hp = fs::path(cypha_path).parent_path() / "train_hparams.json";
    if (fs::exists(auto_hp)) {
      try_load_train_hparams_file(auto_hp.string());
    }
  }

  snapshot_gh_clean_metric();
  g_enc_updates = 0;
  g_replay = std::make_unique<cypha::ReplayBuffer>(g_tsp.replay_cap);
  g_total_steps = g_model->saved_total_steps;
  g_llr_ema = g_model->llr_ema;
  reset_session_counters();
  if (!regression_json_path_opt.empty()) {
    if (!try_load_regression_head_json(regression_json_path_opt, *g_model)) {
      std::cerr << "failed to read --regression-json " << regression_json_path_opt << "\n";
      return false;
    }
  }
  return true;
}

const cypha::RegistryModelRef* find_registry_ref(const std::string& name, std::string version) {
  if (g_registry_cache.empty()) {
    return nullptr;
  }
  if (version.empty() || version == "latest") {
    std::vector<std::string> vers;
    for (const auto& r : g_registry_cache) {
      if (r.name == name) {
        vers.push_back(r.version);
      }
    }
    std::sort(vers.begin(), vers.end());
    if (vers.empty()) {
      return nullptr;
    }
    version = vers.back();
  }
  for (const auto& r : g_registry_cache) {
    if (r.name == name && r.version == version) {
      return &r;
    }
  }
  return nullptr;
}

void refresh_registry_cache() {
  g_registry_cache.clear();
  if (!g_registry_root.empty()) {
    g_registry_cache = cypha::registry_scan(g_registry_root.c_str());
  }
}

std::string json_predict(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_model || !g_mem) {
    return R"({"detail":"No model loaded"})";
  }
  auto t0 = std::chrono::steady_clock::now();
  std::vector<double> x;
  for (const auto& v : body.at("input")) {
    x.push_back(v.get<double>());
  }
  if (g_pre) {
    x = g_pre->transform_one(x);
  }
  std::vector<double> H;
  if (g_mke_active) {
    if (static_cast<int>(x.size()) != g_mke_d_in) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
    H.resize(static_cast<std::size_t>(g_model->d_latent));
    cypha::regression::rff_encode_batch_rowmajor(x.data(), 1, g_mke_d_in, g_mke_W.data(), g_mke_b.data(),
                                                  g_model->d_latent, H.data());
  } else {
    if (static_cast<int>(x.size()) != g_model->d_latent) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
    cypha::batch_encode(*g_model, x.data(), 1, H);
  }

  const int k = static_cast<int>(g_model->labels.size());
  bool use_gh = body.value("use_gh", true);
  cypha::CyphaInferOptions iopt{};
  iopt.deliberation_lo = body.value("deliberation_lo", g_model->deliberation_lo);
  iopt.deliberation_hi = body.value("deliberation_hi", g_model->deliberation_hi);
  iopt.use_field = true;

  std::string pred_label;
  double conf = 0.0;
  double r_eff = 0.0;
  std::vector<double> llr_for_scores;
  double anomaly = 0.0;
  bool is_ood = false;

  if (g_mke_active) {
    std::vector<double> llr;
    cypha::score_matrix_use_field(*g_model, H.data(), 1, llr);
    double eps = 1e-8;
    double T = g_mke_temperature;
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] = llr[static_cast<std::size_t>(j)] / (T + eps);
    }
    std::vector<double> probs;
    cypha::softmax_batch_like_python(z.data(), 1, k, eps, probs);
    int bi = 0;
    for (int j = 1; j < k; ++j) {
      if (probs[static_cast<std::size_t>(j)] > probs[static_cast<std::size_t>(bi)]) {
        bi = j;
      }
    }
    pred_label = g_model->labels[static_cast<std::size_t>(bi)];
    conf = probs[static_cast<std::size_t>(bi)];
    llr_for_scores = std::move(llr);
  } else if (use_gh) {
    iopt.gh_chi = g_gh_chi;
    iopt.gh_psi = g_gh_psi;
    cypha::GhInferAtHResult gh = cypha::gh_infer_at_h(*g_model, H.data(), g_gh_chi, g_gh_psi, kGhNigAdaptAlpha);
    pred_label = gh.label;
    conf = gh.confidence;
    r_eff = gh.r_eff;
    llr_for_scores = std::move(gh.llrs);
    const double r_base =
        (g_model->has_mahal_ema && g_model->mahal_ema > 0.0) ? g_model->mahal_ema : 1.0;
    anomaly = cypha::gh_infer_anomaly_score(r_eff, r_base);
    is_ood = anomaly > kOodThreshold;
  } else {
    cypha::InferAtHResult inf = cypha::infer_at_h(*g_model, H.data(), iopt);
    pred_label = inf.label;
    conf = inf.confidence;
    llr_for_scores = std::move(inf.llrs);
    anomaly = 0.0;
    is_ood = false;
  }

  nlohmann::json scores = nlohmann::json::object();
  for (int j = 0; j < k; ++j) {
    scores[g_model->labels[static_cast<std::size_t>(j)]] = llr_for_scores[static_cast<std::size_t>(j)];
  }
  double latency = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  g_predictions += 1;
  g_sess.push_back(SessPred{pred_label, conf, anomaly, is_ood});

  nlohmann::json out;
  out["label"] = pred_label;
  out["confidence"] = conf;
  out["all_scores"] = scores;
  out["anomaly_score"] = anomaly;
  out["is_ood"] = is_ood;
  if (g_mke_active) {
    double eps = 1e-8;
    double T = g_mke_temperature;
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] = llr_for_scores[static_cast<std::size_t>(j)] / (T + eps);
    }
    std::vector<double> probs;
    cypha::softmax_batch_like_python(z.data(), 1, k, eps, probs);
    double y_mix = 0.0;
    const int d_rff = g_model->d_latent;
    for (int j = 0; j < k; ++j) {
      const std::string& lbl = g_model->labels[static_cast<std::size_t>(j)];
      auto it = g_mke_w.find(lbl);
      if (it == g_mke_w.end() || static_cast<int>(it->second.size()) != d_rff) {
        continue;
      }
      double dp = 0.0;
      for (int t = 0; t < d_rff; ++t) {
        dp += it->second[static_cast<std::size_t>(t)] * H[static_cast<std::size_t>(t)];
      }
      y_mix += probs[static_cast<std::size_t>(j)] * dp;
    }
    out["regression_val"] = y_mix;
    if (static_cast<int>(g_reg_var.size()) == k) {
      double v_mix = 0.0;
      for (int j = 0; j < k; ++j) {
        v_mix += probs[static_cast<std::size_t>(j)] * g_reg_var[static_cast<std::size_t>(j)];
      }
      out["uncertainty"] = std::sqrt(std::max(v_mix, 0.0));
    } else {
      out["uncertainty"] = 0.0;
    }
  } else if (static_cast<int>(g_reg_mu.size()) == k && static_cast<int>(g_reg_var.size()) == k) {
    double eps = 1e-8;
    double T = g_model->temperature;
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] = llr_for_scores[static_cast<std::size_t>(j)] / (T + eps);
    }
    std::vector<double> probs;
    cypha::softmax_batch_like_python(z.data(), 1, k, eps, probs);
    double y_mix = 0.0;
    double u_mix = 0.0;
    cypha::regression::predict_mixture_scalar(probs.data(), g_reg_mu.data(), g_reg_var.data(),
                                              static_cast<std::size_t>(k), y_mix, u_mix);
    out["regression_val"] = y_mix;
    out["uncertainty"] = u_mix;
  } else {
    out["regression_val"] = nullptr;
    out["uncertainty"] = 0.0;
  }
  const bool want_expl = body.value("return_explanation", false);
  if (want_expl) {
    nlohmann::json expl;
    expl["label"] = out["label"];
    expl["confidence"] = conf;
    expl["all_scores"] = scores;
    expl["anomaly_score"] = anomaly;
    expl["is_ood"] = is_ood;
    expl["r_eff"] = use_gh ? r_eff : 0.0;
    nlohmann::json cdet = nlohmann::json::object();
    const int d = g_model->d_latent;
    for (int ci = 0; ci < k; ++ci) {
      double sumsq = 0.0;
      for (int j = 0; j < d; ++j) {
        double v = g_model->D[static_cast<std::size_t>(ci * d + j)];
        sumsq += v * v;
      }
      nlohmann::json row;
      row["n_obs"] = g_model->n_obs[static_cast<std::size_t>(ci)];
      row["delta_mu_norm"] = std::sqrt(sumsq);
      cdet[g_model->labels[static_cast<std::size_t>(ci)]] = row;
    }
    expl["class_details"] = cdet;
    double wh = 0.0;
    for (int j = 0; j < d; ++j) {
      double t = H[static_cast<std::size_t>(j)] - g_model->mu_world[static_cast<std::size_t>(j)];
      wh += t * t;
    }
    expl["world_mu_distance"] = std::sqrt(wh);
    out["explanation"] = std::move(expl);
  } else {
    out["explanation"] = nullptr;
  }
  out["latency_ms"] = latency;
  return out.dump();
}

std::string json_update(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_model || !g_mem) {
    return R"({"detail":"No model loaded"})";
  }
  std::vector<double> x;
  for (const auto& v : body.at("input")) {
    x.push_back(v.get<double>());
  }
  if (g_pre) {
    x = g_pre->transform_one(x);
  }
  std::string label = body.at("correct_label").get<std::string>();
  bool use_gh = body.value("use_gh", true);
  const int d = g_model->d_latent;

  const bool has_regr_y = body.contains("regression_y") && !body["regression_y"].is_null();
  if (has_regr_y && !body["regression_y"].is_number()) {
    return R"({"detail":"regression_y must be a number"})";
  }
  if (has_regr_y && !g_mke_active) {
    return R"({"detail":"regression_y requires mke block in regression_head.json"})";
  }
  constexpr double kSoftmaxEps = 1e-8;
  const bool want_mke = g_mke_active && has_regr_y;

  if (want_mke) {
    if (static_cast<int>(x.size()) != g_mke_d_in) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
  } else {
    if (static_cast<int>(x.size()) != g_model->d_latent) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
  }

  if (!g_replay) {
    g_replay = std::make_unique<cypha::ReplayBuffer>(g_tsp.replay_cap);
  }
  cypha::TrainStepExtras extras{};
  extras.total_steps = &g_total_steps;
  extras.ood_sigma = &g_ood_sigma;
  extras.llr_ema = &g_llr_ema;
  std::vector<double> replay_u01_storage;
  std::size_t replay_u01_pos = 0;
  if (body.contains("replay_u01") && body["replay_u01"].is_array()) {
    for (const auto& v : body["replay_u01"]) {
      replay_u01_storage.push_back(v.get<double>());
    }
    extras.replay_u01 = replay_u01_storage.data();
    extras.replay_u01_len = replay_u01_storage.size();
    extras.replay_u01_pos = &replay_u01_pos;
  }

  double loss = std::numeric_limits<double>::quiet_NaN();
  if (want_mke) {
    double y = body["regression_y"].get<double>();
    const std::string* router_ov = nullptr;
    std::string router_storage;
    if (body.contains("router_train_label") && body["router_train_label"].is_string()) {
      router_storage = body["router_train_label"].get<std::string>();
      if (!router_storage.empty()) {
        router_ov = &router_storage;
      }
    }
    const double* gh_ptr = nullptr;
    if (use_gh && static_cast<int>(g_mke_gh_scales.size()) == static_cast<int>(g_model->labels.size())) {
      gh_ptr = g_mke_gh_scales.data();
    }
    cypha::regression::MkeScalarTrainStepOutputs step_out{};
    (void)cypha::regression::mke_scalar_train_step(
        *g_model, *g_mem, *g_replay, x.data(), g_mke_d_in, y, g_mke_W.data(), g_mke_b.data(), g_model->d_latent,
        g_mke_w, g_mke_p, gh_ptr, g_mke_temperature, g_mke_forgetting, g_mke_pi_floor, g_tsp, g_world_lr,
        g_delta_lr, g_ood_sigma, g_rng, g_enc_updates, &extras, router_ov, kSoftmaxEps, &step_out);
    loss = step_out.router_loss;
    (void)label;
  } else if (use_gh && static_cast<int>(g_gh_inv_v_clean.size()) == d) {
    cypha::GhTrainStepResult gh = cypha::dif_gh_train_step_vector(
        *g_model, *g_mem, *g_replay, x.data(), d, label, g_gh_inv_v_clean, g_gh_R_base, g_gh_chi, g_gh_psi,
        kGhNigAdaptAlpha, g_world_lr, g_delta_lr, g_ood_sigma, g_tsp, g_rng, g_enc_updates, nullptr, &extras);
    loss = gh.loss;
    g_gh_chi = gh.chi_new;
    g_gh_psi = gh.psi_new;
  } else {
    loss = cypha::dif_train_step_vector(*g_model, *g_mem, *g_replay, x.data(), d, label, g_world_lr, g_delta_lr,
                                        g_world_lr, g_delta_lr, g_ood_sigma, g_tsp, g_rng, g_enc_updates, nullptr,
                                        &extras);
  }
  g_engine_corrections += 1;

  nlohmann::json out;
  out["loss"] = loss;
  out["n_corrections"] = g_engine_corrections;
  return out.dump();
}

std::string json_adapt_temperature(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_model || !g_mem) {
    return R"({"detail":"No model loaded"})";
  }
  const auto& cal = body.at("calibration");
  int n_grid = body.value("n_grid", 20);
  double T_min = body.value("T_min", 0.3);
  double T_max = body.value("T_max", 8.0);
  int n_bins = body.value("n_bins", 10);
  if (n_grid < 1) {
    n_grid = 1;
  }
  if (n_bins < 2) {
    n_bins = 10;
  }
  const int d = g_model->d_latent;
  std::vector<double> h_batch;
  std::vector<int> true_idx;
  for (const auto& row : cal) {
    std::vector<double> xv;
    for (const auto& v : row.at("input")) {
      xv.push_back(v.get<double>());
    }
    if (g_pre) {
      xv = g_pre->transform_one(xv);
    }
    if (static_cast<int>(xv.size()) != d) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
    std::string y = row.at("correct_label").get<std::string>();
    auto it = g_mem->label_index.find(y);
    if (it == g_mem->label_index.end()) {
      continue;
    }
    std::vector<double> oneh;
    cypha::batch_encode(*g_model, xv.data(), 1, oneh);
    h_batch.insert(h_batch.end(), oneh.begin(), oneh.end());
    true_idx.push_back(it->second);
  }
  nlohmann::json out;
  out["n_used"] = static_cast<int>(true_idx.size());
  if (true_idx.empty()) {
    out["temperature"] = g_model->temperature;
    return out.dump();
  }
  double T = cypha::adapt_temperature_ece(*g_model, h_batch.data(), static_cast<int>(true_idx.size()),
                                          true_idx.data(), n_grid, T_min, T_max, n_bins);
  out["temperature"] = T;
  return out.dump();
}

nlohmann::json session_summary_json() {
  nlohmann::json payload;
  if (g_sess.empty()) {
    // Match InferenceSession.summary() when no prediction history.
    double duration =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g_sess_started).count();
    payload["n_predictions"] = 0;
    payload["n_corrections"] = 0;
    payload["correction_accuracy"] = 0.0;
    payload["mean_confidence"] = 0.0;
    payload["mean_anomaly"] = 0.0;
    payload["n_ood_flagged"] = 0;
    payload["label_distribution"] = nlohmann::json::object();
    payload["session_duration_s"] = duration;
    return payload;
  }
  double sum_c = 0.0;
  double sum_a = 0.0;
  int n_ood = 0;
  std::unordered_map<std::string, int> dist;
  for (const auto& p : g_sess) {
    sum_c += p.confidence;
    sum_a += p.anomaly_score;
    if (p.is_ood) {
      n_ood += 1;
    }
    dist[p.label] += 1;
  }
  nlohmann::json ld = nlohmann::json::object();
  for (const auto& pr : dist) {
    ld[pr.first] = pr.second;
  }
  double n = static_cast<double>(g_sess.size());
  payload["n_predictions"] = static_cast<int>(g_sess.size());
  // HTTP /update does not append InferenceSession._corrections (matches FastAPI session summary).
  payload["n_corrections"] = 0;
  payload["correction_accuracy"] = 0.0;
  payload["mean_confidence"] = sum_c / n;
  payload["mean_anomaly"] = sum_a / n;
  payload["n_ood_flagged"] = n_ood;
  payload["label_distribution"] = ld;
  payload["session_duration_s"] =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - g_sess_started).count();
  return payload;
}

std::string json_register(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_registry_root.empty()) {
    return R"({"detail":"No registry configured"})";
  }
  try {
    std::string name = body.at("name").get<std::string>();
    std::string version = body.at("version").get<std::string>();
    std::string cypha_src = body.at("model_cypha").get<std::string>();
    std::string card_src = body.at("card_json").get<std::string>();
    const char* pre_ptr = nullptr;
    std::string pre_storage;
    if (body.contains("preprocessor_json") && !body["preprocessor_json"].is_null()) {
      if (!body["preprocessor_json"].is_string()) {
        return R"({"detail":"preprocessor_json must be a string path or null"})";
      }
      pre_storage = body["preprocessor_json"].get<std::string>();
      if (!pre_storage.empty()) {
        pre_ptr = pre_storage.c_str();
      }
    }
    bool overwrite = body.value("overwrite", false);
    std::string err;
    if (!cypha::registry_register_bundle(g_registry_root.c_str(), name.c_str(), version.c_str(),
                                        cypha_src.c_str(), card_src.c_str(), pre_ptr, overwrite, &err)) {
      nlohmann::json j;
      j["detail"] = err;
      return j.dump();
    }
    refresh_registry_cache();
    fs::path model_dir = fs::absolute(fs::path(g_registry_root) / name / version);
    nlohmann::json ok;
    ok["registered"] = true;
    ok["model_dir"] = model_dir.generic_string();
    return ok.dump();
  } catch (const nlohmann::json::exception&) {
    return R"({"detail":"invalid register request"})";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string listen = "127.0.0.1";
  int port = 8099;
  std::string cypha_path;
  std::string pre_path;
  std::string ff_json;
  std::string train_hparams_path;
  std::string regression_json_path;
  std::string cyphalm_checkpoint_path;
  std::string branch_a_json_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--listen" && i + 1 < argc) {
      std::string hp = argv[++i];
      auto c = hp.find(':');
      if (c != std::string::npos) {
        listen = hp.substr(0, c);
        port = std::stoi(hp.substr(c + 1));
      } else {
        listen = hp;
      }
    } else if (a == "--cypha" && i + 1 < argc) {
      cypha_path = argv[++i];
    } else if (a == "--preprocessor" && i + 1 < argc) {
      pre_path = argv[++i];
    } else if (a == "--f-field-json" && i + 1 < argc) {
      ff_json = argv[++i];
    } else if (a == "--registry" && i + 1 < argc) {
      g_registry_root = argv[++i];
    } else if (a == "--train-hparams" && i + 1 < argc) {
      train_hparams_path = argv[++i];
    } else if (a == "--regression-json" && i + 1 < argc) {
      regression_json_path = argv[++i];
    } else if (a == "--cyphalm-checkpoint" && i + 1 < argc) {
      cyphalm_checkpoint_path = argv[++i];
    } else if (a == "--branch-a-json" && i + 1 < argc) {
      branch_a_json_path = argv[++i];
    }
  }
  if (cyphalm_checkpoint_path.empty()) {
    if (const char* env_ckpt = std::getenv("CYPHALM_CHECKPOINT")) {
      cyphalm_checkpoint_path = env_ckpt;
    }
  }
  if (branch_a_json_path.empty()) {
    if (const char* env_ba = std::getenv("CYPHA_BRANCH_A_CHECKPOINT")) {
      branch_a_json_path = env_ba;
    }
  }

  refresh_registry_cache();

  if (cypha_path.empty()) {
    std::cerr << "usage: cypha_rest --listen host:port --cypha model.cypha [--f-field-json f_field.json] "
                 "[--pre preprocessor.json] [--train-hparams train_hparams.json] "
                 "[--regression-json regression_head.json] [--cyphalm-checkpoint ckpt_base] "
                 "[--branch-a-json branch_a_router.json] [--registry models_root]  "
                 "(POST /register needs --registry)\n";
    return 2;
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!load_bundle_paths(cypha_path, pre_path, ff_json, train_hparams_path, regression_json_path)) {
      return 1;
    }
  }

  httplib::Server svr;

  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    nlohmann::json j;
    j["status"] = "ok";
    j["model"] = (g_model) ? "CyphaDIF" : "none";
    j["lm_loaded"] = cypha::cyphalm::cyphalm_rest_lm_loaded();
    j["uptime"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_started).count();
    j["n_predictions"] = g_predictions;
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/ready", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_model) {
      res.status = 503;
      res.set_content(R"({"ready":false,"reason":"no_model_loaded"})", "application/json");
      return;
    }
    nlohmann::json j;
    j["ready"] = true;
    j["model_type"] = "CyphaDIF";
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    double uptime =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g_started).count();
    nlohmann::json payload;
    payload["uptime_seconds"] = std::round(uptime * 1000.0) / 1000.0;
    payload["model_loaded"] = g_model != nullptr;
    payload["model_type"] = g_model ? "CyphaDIF" : nullptr;
    payload["n_predictions"] = g_predictions;
    payload["n_corrections"] = g_engine_corrections;
    payload["registry_model_count"] = static_cast<int>(g_registry_cache.size());
    if (g_model) {
      payload["gh_chi_session"] = g_gh_chi;
      payload["gh_psi_session"] = g_gh_psi;
      payload["session"] = session_summary_json();
      payload["regression_head_loaded"] = (!g_reg_mu.empty() || g_mke_active);
    } else {
      payload["session"] = nullptr;
      payload["regression_head_loaded"] = false;
    }
    payload["lm_loaded"] = cypha::cyphalm::cyphalm_rest_lm_loaded();
    payload["branch_a_router"] = cypha::branch_a_rest_summary_json();
    res.set_content(payload.dump(), "application/json");
  });

  svr.Post("/predict", [](const httplib::Request& req, httplib::Response& res) {
    try {
      auto body = nlohmann::json::parse(req.body);
      std::string out = json_predict(body);
      if (out.find("No model loaded") != std::string::npos) {
        res.status = 503;
      } else if (out.find("\"detail\"") != std::string::npos) {
        res.status = 400;
      }
      res.set_content(out, "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
    }
  });

  svr.Post("/update", [](const httplib::Request& req, httplib::Response& res) {
    try {
      auto body = nlohmann::json::parse(req.body);
      std::string out = json_update(body);
      if (out.find("No model loaded") != std::string::npos) {
        res.status = 503;
      } else if (out.find("\"detail\"") != std::string::npos) {
        res.status = 400;
      }
      res.set_content(out, "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
    }
  });

  svr.Post("/register", [](const httplib::Request& req, httplib::Response& res) {
    try {
      auto body = nlohmann::json::parse(req.body);
      std::string out = json_register(body);
      if (out.find("No registry configured") != std::string::npos) {
        res.status = 503;
      } else if (out.find("\"detail\"") != std::string::npos) {
        res.status = 400;
      }
      res.set_content(out, "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
    }
  });

  svr.Post("/adapt_temperature", [](const httplib::Request& req, httplib::Response& res) {
    try {
      auto body = nlohmann::json::parse(req.body);
      std::string out = json_adapt_temperature(body);
      if (out.find("No model loaded") != std::string::npos) {
        res.status = 503;
      } else if (out.find("\"detail\"") != std::string::npos) {
        res.status = 400;
      }
      res.set_content(out, "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
    }
  });

  svr.Get("/models", [](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    bool summary = false;
    if (req.has_param("summary")) {
      std::string s = req.get_param_value("summary");
      summary = (s == "1" || s == "true" || s == "True");
    }
    nlohmann::json arr = nlohmann::json::array();
    if (g_registry_cache.empty()) {
      nlohmann::json j;
      j["models"] = arr;
      res.set_content(j.dump(), "application/json");
      return;
    }
    if (summary) {
      for (const auto& r : g_registry_cache) {
        nlohmann::json row;
        row["name"] = r.name;
        row["version"] = r.version;
        arr.push_back(row);
      }
    } else {
      for (const auto& r : g_registry_cache) {
        try {
          std::ifstream f(r.card_path);
          std::stringstream b;
          b << f.rdbuf();
          arr.push_back(nlohmann::json::parse(b.str()));
        } catch (...) {
          nlohmann::json row;
          row["name"] = r.name;
          row["version"] = r.version;
          row["error"] = "card_parse_failed";
          arr.push_back(row);
        }
      }
    }
    nlohmann::json j;
    j["models"] = arr;
    res.set_content(j.dump(), "application/json");
  });

  svr.Post("/load", [](const httplib::Request& req, httplib::Response& res) {
    if (g_registry_root.empty() || g_registry_cache.empty()) {
      res.status = 503;
      res.set_content(R"({"detail":"No registry configured"})", "application/json");
      return;
    }
    try {
      auto body = nlohmann::json::parse(req.body);
      std::string name = body.at("name").get<std::string>();
      std::string version = body.value("version", "latest");
      const cypha::RegistryModelRef* ref = nullptr;
      {
        std::lock_guard<std::mutex> lk(g_mu);
        ref = find_registry_ref(name, version);
      }
      if (ref == nullptr) {
        res.status = 404;
        res.set_content(R"({"detail":"model not found"})", "application/json");
        return;
      }
      fs::path dir = fs::path(ref->model_path).parent_path();
      fs::path ff_path = dir / "f_field.json";
      std::string ff_p = fs::exists(ff_path) ? ff_path.string() : "";
      fs::path reg_path = dir / "regression_head.json";
      std::string reg_p = fs::exists(reg_path) ? reg_path.string() : "";
      std::string pre = ref->preprocessor_path;
      std::lock_guard<std::mutex> lk(g_mu);
      if (!load_bundle_paths(ref->model_path, pre, ff_p, "", reg_p)) {
        res.status = 500;
        res.set_content(R"({"detail":"load failed"})", "application/json");
        return;
      }
      std::ifstream cf(ref->card_path);
      std::stringstream cb;
      cb << cf.rdbuf();
      nlohmann::json card = nlohmann::json::parse(cb.str());
      nlohmann::json wrap;
      wrap["loaded"] = std::move(card);
      res.set_content(wrap.dump(), "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
    }
  });

  svr.Get("/session", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    nlohmann::json s = session_summary_json();
    res.set_content(s.dump(), "application/json");
  });

  svr.Delete("/session", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_sess.clear();
    // Match InferenceSession.clear(): reset session GH NIG state (model weights unchanged).
    g_gh_chi = 1.0;
    g_gh_psi = 1.0;
    nlohmann::json j;
    j["cleared"] = true;
    res.set_content(j.dump(), "application/json");
  });

  // ── /session/rng — deterministic replay: snapshot/restore std::mt19937 state ─────────────────
  // Matches Python FastAPI ``GET /session/rng`` response shape:
  //   {"bit_generator":"MT19937", "state":[624 uint32 values], "pos":int}
  // libstdc++ ``operator<<`` for mersenne_twister_engine writes:
  //   word[0] word[1] ... word[623] pos   (625 space-separated unsigned integers)
  svr.Get("/session/rng", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    std::ostringstream oss;
    oss << g_rng;
    std::istringstream iss(oss.str());
    nlohmann::json state_arr = nlohmann::json::array();
    for (int i = 0; i < 624; ++i) {
      unsigned long v = 0;
      iss >> v;
      state_arr.push_back(static_cast<uint32_t>(v));
    }
    unsigned long pos = 0;
    iss >> pos;
    nlohmann::json j;
    j["bit_generator"] = "MT19937";
    j["state"]         = state_arr;
    j["pos"]           = static_cast<int>(pos);
    res.set_content(j.dump(), "application/json");
  });

  svr.Post("/session/rng", [](const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
      body = nlohmann::json::parse(req.body);
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
      return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    if (body.contains("seed") && body["seed"].is_number_integer()) {
      // Re-seed from scratch — identical to Python np.random.MT19937(seed)
      g_rng = std::mt19937{static_cast<uint32_t>(body["seed"].get<long long>())};
    } else if (body.contains("state") && body["state"].is_array()
               && body["state"].size() == 624) {
      // Full state restore: reconstruct the serialised text representation and feed to operator>>.
      std::ostringstream oss;
      for (const auto& v : body["state"]) {
        oss << v.get<unsigned long>() << ' ';
      }
      oss << body.value("pos", 0);
      std::istringstream iss(oss.str());
      iss >> g_rng;
    } else {
      res.status = 400;
      res.set_content(R"({"detail":"provide seed (int) or state (array of 624 uint32) + pos"})",
                      "application/json");
      return;
    }
    // Return new state
    std::ostringstream oss2;
    oss2 << g_rng;
    std::istringstream iss2(oss2.str());
    nlohmann::json state_arr = nlohmann::json::array();
    for (int i = 0; i < 624; ++i) {
      unsigned long v = 0;
      iss2 >> v;
      state_arr.push_back(static_cast<uint32_t>(v));
    }
    unsigned long pos2 = 0;
    iss2 >> pos2;
    nlohmann::json j;
    j["bit_generator"] = "MT19937";
    j["state"]         = state_arr;
    j["pos"]           = static_cast<int>(pos2);
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/classes", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_model) {
      res.status = 503;
      res.set_content(R"({"detail":"No model loaded"})", "application/json");
      return;
    }
    nlohmann::json classes = nlohmann::json::object();
    const int K = static_cast<int>(g_model->labels.size());
    for (int k = 0; k < K; ++k) {
      nlohmann::json row;
      row["n_obs"] = g_model->n_obs[static_cast<std::size_t>(k)];
      classes[g_model->labels[static_cast<std::size_t>(k)]] = row;
    }
    nlohmann::json out;
    out["classes"] = classes;
    res.set_content(out.dump(), "application/json");
  });

  cypha::dif_rest_configure(&g_mu, &g_model, &g_pre, &g_rng);
  cypha::register_dif_rest_routes(svr);
  cypha::cyphalm::register_cyphalm_rest_routes(svr);
  cypha::branch_a_rest_configure(branch_a_json_path);
  cypha::register_branch_a_rest_routes(svr);

  if (!cyphalm_checkpoint_path.empty()) {
    try {
      cypha::cyphalm::cyphalm_rest_lm_load(cyphalm_checkpoint_path);
      std::cout << "CyphaLM loaded from " << cyphalm_checkpoint_path << "\n";
    } catch (const std::exception& ex) {
      std::cerr << "CyphaLM checkpoint load failed: " << ex.what() << "\n";
      return 1;
    }
  }

  std::cout << "cypha_rest listening on http://" << listen << ":" << port << "\n";
  if (!svr.listen(listen.c_str(), port)) {
    std::cerr << "bind failed\n";
    return 1;
  }
  return 0;
}
