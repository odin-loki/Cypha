// REST surface aligned with cypha_studio.server.api (subset + registry-aware /load).
// Missing-model errors use {"detail":"No model loaded"} with HTTP 503 (same as FastAPI) on POST /predict, /update, /adapt_temperature and GET /classes.
// Malformed JSON on those POSTs returns HTTP 400 {"detail":"bad json"} (FastAPI uses 422 + validation detail for the same case).
// Build: cypha_rest --listen host:port --cypha model.cypha [--f-field-json ff.json] [--pre preprocessor.json]
//        [--train-hparams path] [--registry <root>]
//        `--train-hparams` JSON may include `align_every` (encoder align period; default 500) and
//        `temp_recalib_every` (0 = off), matching `fixtures/train_hparams.json`.
//        If `world.F_field` is in the .cypha blob, `--f-field-json` is optional. Registry `/load` uses
//        `f_field.json` next to the model when the blob has no embedded `F_field`.
//        With `--registry <root>`, `POST /register` copies `model_cypha` + `card_json` (+ optional `preprocessor_json`)
//        paths into `<root>/<name>/<version>/` (see PORT_CONTRACT §3).
//        Multi-model (FUTURE.md §5): `--preload-registry` loads all registry bundles into RAM;
//        `POST /predict` and `POST /update` accept optional `"model":"name/version"`; `GET /models` adds
//        `loaded` / `active`; `POST /load` hot-swaps active model and fills the in-memory map.
//        Optional `--regression-json regression_head.json`: scalar MoE targets per class label → `/predict`
//        fills `regression_val` and `uncertainty` (mixture of expert EMAs; see PORT_CONTRACT §3).
//        Optional top-level `mke` object in that JSON → RFF + expert RLS + router `dif_train_step_vector` on
//        `POST /update` when the body includes numeric `regression_y` (see PORT_CONTRACT §3).
//        Optional `--sequence-checkpoint base` (alias `--cyphalm-checkpoint`) or env
//        `CYPHA_SEQUENCE_CHECKPOINT` / `CYPHA_LM_CHECKPOINT` / `CYPHALM_CHECKPOINT` → auto-load
//        Cypha sequence at startup.

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

#include "cypha/cypha.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/regression.hpp"
#include "cypha/registry.hpp"
#include "cypha/branch_a_rest.hpp"
#include "cypha/cyphalm/cyphalm_rest.hpp"
#include "cypha/dif_rest.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/causal_graph.hpp"
#include "cypha/intelligence/measurers.hpp"
#include "cypha/intelligence/self_correcting_infer.hpp"
#include "cypha/curriculum.hpp"
#include "cypha/intelligence_rest.hpp"
#include "cypha_rest_static_ui.hpp"

namespace fs = std::filesystem;

namespace {

std::mutex g_mu;
/// Primary process model — classify + regress + latent sample (+ optional sequence).
std::unique_ptr<cypha::Cypha> g_cypha = std::make_unique<cypha::Cypha>();
cypha::intelligence::IntelligenceProfiler g_intelligence_profiler;
cypha::intelligence::CausalGraphMonitor g_causal_graph_monitor;
cypha::intelligence::EpistemicThreshold g_epistemic_threshold(0.5, 5.0);
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

constexpr double kGhNigAdaptAlpha = 0.98;
/// Python ``InferenceEngine.OOD_THRESHOLD`` (anomaly_score > threshold → is_ood).
constexpr double kOodThreshold = 3.0;

/// In-memory model slot keyed by registry ``name/version`` (FUTURE.md §5).
struct LoadedModelBundle {
  std::mutex mu;
  std::unique_ptr<cypha::Cypha> cypha = std::make_unique<cypha::Cypha>();
};

std::unordered_map<std::string, LoadedModelBundle> g_models;
std::string g_active_model_key;
bool g_preload_registry{false};

std::string model_registry_key(const std::string& name, const std::string& version) {
  return name + "/" + version;
}

bool parse_model_registry_key(const std::string& s, std::string& name, std::string& version) {
  const auto slash = s.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= s.size()) {
    return false;
  }
  name = s.substr(0, slash);
  version = s.substr(slash + 1);
  return true;
}

/// Non-owning view so predict/update can run on globals or a map slot.
struct ModelView {
  cypha::Cypha* cypha{};
  std::unique_ptr<cypha::CyphaInferModel>* model{};
  std::unique_ptr<cypha::CyphaDifMemoryState>* mem{};
  std::unique_ptr<cypha::KernelMemory>* kernel_mem{};
  bool* use_kernel_llr{};
  double* kernel_blend{};
  std::unique_ptr<cypha::PreprocessorState>* pre{};
  std::unique_ptr<cypha::ReplayBuffer>* replay{};
  int* enc_updates{};
  cypha::TrainStepParams* tsp{};
  int* total_steps{};
  double* llr_ema{};
  double* world_lr{};
  double* delta_lr{};
  double* ood_sigma{};
  std::vector<double>* gh_inv_v_clean{};
  double* gh_R_base{};
  double* gh_chi{};
  double* gh_psi{};
  std::vector<double>* reg_mu{};
  std::vector<double>* reg_var{};
  bool* mke_active{};
  int* mke_d_in{};
  std::vector<double>* mke_W{};
  std::vector<double>* mke_b{};
  double* mke_temperature{};
  double* mke_forgetting{};
  double* mke_pi_floor{};
  std::vector<double>* mke_gh_scales{};
  std::unordered_map<std::string, std::vector<double>>* mke_w{};
  std::unordered_map<std::string, std::vector<double>>* mke_p{};
};

std::string json_update_impl(const nlohmann::json& body, ModelView v);

ModelView view_from_cypha(cypha::Cypha& c) {
  return ModelView{&c,
                   &c.infer_owned(),
                   &c.mem_owned(),
                   &c.kernel_mem_owned(),
                   &c.use_kernel_llr(),
                   &c.kernel_blend(),
                   &c.preprocessor_owned(),
                   &c.replay_owned(),
                   &c.enc_updates(),
                   &c.train_params(),
                   &c.total_steps(),
                   &c.llr_ema(),
                   &c.world_lr(),
                   &c.delta_lr(),
                   &c.ood_sigma(),
                   &c.gh_inv_v_clean(),
                   &c.gh_R_base(),
                   &c.gh_chi(),
                   &c.gh_psi(),
                   &c.reg_mu(),
                   &c.reg_var(),
                   &c.mke_active_ref(),
                   &c.mke_d_in_ref(),
                   &c.mke_W(),
                   &c.mke_b(),
                   &c.mke_temperature(),
                   &c.mke_forgetting(),
                   &c.mke_pi_floor(),
                   &c.mke_gh_scales(),
                   &c.mke_w(),
                   &c.mke_p()};
}

ModelView view_from_globals() { return view_from_cypha(*g_cypha); }

ModelView view_from_bundle(LoadedModelBundle& b) { return view_from_cypha(*b.cypha); }

cypha::Cypha* cypha_for_view(ModelView v) { return v.cypha; }

std::mt19937& rng_for_view(ModelView v) {
  cypha::Cypha* c = cypha_for_view(v);
  return (c != nullptr ? *c : *g_cypha).rng();
}

void apply_default_train_hparams_view(ModelView v);
bool try_load_train_hparams_file_view(const std::string& path, ModelView v);

void clear_mke_state_in(ModelView v) {
  *v.mke_active = false;
  *v.mke_d_in = 0;
  v.mke_W->clear();
  v.mke_b->clear();
  *v.mke_temperature = 1.0;
  *v.mke_forgetting = 1.0;
  *v.mke_pi_floor = 0.02;
  v.mke_gh_scales->clear();
  v.mke_w->clear();
  v.mke_p->clear();
}

void clear_mke_state() { clear_mke_state_in(view_from_globals()); }

void reset_session_counters() {
  g_sess.clear();
  g_predictions = 0;
  g_engine_corrections = 0;
  g_sess_started = std::chrono::steady_clock::now();
}

void apply_default_train_hparams() {
  apply_default_train_hparams_view(view_from_globals());
}

bool try_load_train_hparams_file(const std::string& path) {
  return try_load_train_hparams_file_view(path, view_from_globals());
}

void snapshot_gh_clean_metric_view(ModelView v) {
  *v.gh_chi = 1.0;
  *v.gh_psi = 1.0;
  if (!*v.model) {
    v.gh_inv_v_clean->clear();
    *v.gh_R_base = 1.0;
    return;
  }
  const int d = (*v.model)->d_latent;
  *v.gh_inv_v_clean = (*v.model)->inv_v;
  double mean_inv = 0.0;
  for (int j = 0; j < d; ++j) {
    mean_inv += (*v.gh_inv_v_clean)[static_cast<std::size_t>(j)];
  }
  mean_inv /= static_cast<double>(std::max(d, 1));
  constexpr double kEps = 1e-8;
  *v.gh_R_base = 1.0 / (mean_inv + kEps);
}

void snapshot_gh_clean_metric() {
  snapshot_gh_clean_metric_view(view_from_globals());
}

void load_kernel_from_root_view(ModelView v, const cypha::CNode& root, int d) {
  v.kernel_mem->reset();
  *v.use_kernel_llr = false;
  *v.kernel_blend = 0.5;
  if (d <= 0) {
    return;
  }
  cypha::KernelMemory km(d, 256, 0);
  if (cypha::try_load_kernel_from_root(root, km, *v.use_kernel_llr, *v.kernel_blend)) {
    v.kernel_mem->reset(new cypha::KernelMemory(std::move(km)));
  }
}

void load_kernel_from_root(const cypha::CNode& root, int d) {
  load_kernel_from_root_view(view_from_globals(), root, d);
}

void apply_kernel_to_options_view(ModelView v, cypha::CyphaInferOptions& opt) {
  opt.kernel_mem = v.kernel_mem->get();
  opt.use_kernel_llr = *v.use_kernel_llr && *v.kernel_mem != nullptr;
  opt.kernel_blend = *v.kernel_blend;
}

void apply_global_kernel_to_options(cypha::CyphaInferOptions& opt) {
  apply_kernel_to_options_view(view_from_globals(), opt);
}

void sync_kernel_from_json_view(ModelView v, const nlohmann::json& body) {
  if (!*v.model) {
    return;
  }
  if (body.contains("use_kernel_llr")) {
    *v.use_kernel_llr = body["use_kernel_llr"].get<bool>();
  }
  if (body.contains("kernel_blend") && body["kernel_blend"].is_number()) {
    *v.kernel_blend = body["kernel_blend"].get<double>();
  }
  if (!*v.use_kernel_llr) {
    return;
  }
  const int d = (*v.model)->d_latent;
  if (*v.kernel_mem == nullptr || (*v.kernel_mem)->feat_dim() != d) {
    v.kernel_mem->reset(
        new cypha::KernelMemory(d, 256, static_cast<std::uint64_t>(rng_for_view(v)())));
  }
}

void sync_kernel_from_json(const nlohmann::json& body) {
  sync_kernel_from_json_view(view_from_globals(), body);
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

bool try_load_regression_head_json_view(const std::string& path, ModelView v) {
  v.reg_mu->clear();
  v.reg_var->clear();
  clear_mke_state_in(v);
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
  const int k = static_cast<int>((*v.model)->labels.size());
  v.reg_mu->assign(static_cast<std::size_t>(k), 0.0);
  v.reg_var->assign(static_cast<std::size_t>(k), 0.0);
  for (int i = 0; i < k; ++i) {
    const std::string& lbl = (*v.model)->labels[static_cast<std::size_t>(i)];
    if (!ex.contains(lbl)) {
      continue;
    }
    const auto& row = ex[lbl];
    if (row.contains("mu")) {
      if (row["mu"].is_number()) {
        (*v.reg_mu)[static_cast<std::size_t>(i)] = row["mu"].get<double>();
      } else if (row["mu"].is_array() && !row["mu"].empty()) {
        (*v.reg_mu)[static_cast<std::size_t>(i)] = row["mu"][0].get<double>();
      }
    }
    if (row.contains("var_ema") && row["var_ema"].is_number()) {
      (*v.reg_var)[static_cast<std::size_t>(i)] = row["var_ema"].get<double>();
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
    *v.mke_d_in = mk.at("d_in").get<int>();
    const int d_rff = mk.at("D_rff").get<int>();
    if (d_rff != (*v.model)->d_latent) {
      return false;
    }
    *v.mke_W = mk.at("rff_W_rowmajor").get<std::vector<double>>();
    *v.mke_b = mk.at("rff_b").get<std::vector<double>>();
    if (static_cast<int>(v.mke_W->size()) != d_rff * *v.mke_d_in ||
        static_cast<int>(v.mke_b->size()) != d_rff) {
      return false;
    }
    *v.mke_temperature = mk.value("temperature", 1.0);
    *v.mke_forgetting = mk.value("forgetting_factor", 1.0);
    *v.mke_pi_floor = mk.value("pi_floor", 0.02);

    const auto& wj = mk.at("w");
    const auto& pj = mk.at("P");
    if (!wj.is_object() || !pj.is_object()) {
      return false;
    }
    const std::size_t p_expect = static_cast<std::size_t>(d_rff) * static_cast<std::size_t>(d_rff);
    v.mke_w->clear();
    v.mke_p->clear();
    for (int i = 0; i < k; ++i) {
      const std::string& lbl = (*v.model)->labels[static_cast<std::size_t>(i)];
      if (!wj.contains(lbl) || !pj.contains(lbl)) {
        return false;
      }
      auto ww = wj[lbl].get<std::vector<double>>();
      auto pp = pj[lbl].get<std::vector<double>>();
      if (static_cast<int>(ww.size()) != d_rff || pp.size() != p_expect) {
        return false;
      }
      (*v.mke_w)[lbl] = std::move(ww);
      (*v.mke_p)[lbl] = std::move(pp);
    }
    v.mke_gh_scales->clear();
    if (mk.contains("gh_scales") && mk["gh_scales"].is_array()) {
      *v.mke_gh_scales = mk["gh_scales"].get<std::vector<double>>();
      if (static_cast<int>(v.mke_gh_scales->size()) != k) {
        return false;
      }
    }
    *v.mke_active = true;
  } catch (...) {
    clear_mke_state_in(v);
    return false;
  }
  return true;
}

bool try_load_regression_head_json(const std::string& path, const cypha::CyphaInferModel& model) {
  (void)model;
  return try_load_regression_head_json_view(path, view_from_globals());
}

void apply_default_train_hparams_view(ModelView v) {
  *v.world_lr = 0.008;
  *v.delta_lr = 0.05;
  *v.ood_sigma = 15.0;
  v.tsp->enc_lr = 0.002;
  v.tsp->replay_ratio = 0.30;
  v.tsp->replay_cap = 10000;
  v.tsp->align_every = 500;
  v.tsp->temp_recalib_every = 0;
}

bool try_load_train_hparams_file_view(const std::string& path, ModelView v) {
  std::ifstream f(path);
  if (!f) {
    return false;
  }
  std::stringstream b;
  b << f.rdbuf();
  nlohmann::json j = nlohmann::json::parse(b.str());
  if (j.contains("world_lr")) {
    *v.world_lr = j["world_lr"].get<double>();
  }
  if (j.contains("delta_lr")) {
    *v.delta_lr = j["delta_lr"].get<double>();
  }
  if (j.contains("ood_sigma")) {
    *v.ood_sigma = j["ood_sigma"].get<double>();
  }
  if (j.contains("enc_lr")) {
    v.tsp->enc_lr = j["enc_lr"].get<double>();
  }
  if (j.contains("replay_ratio")) {
    v.tsp->replay_ratio = j["replay_ratio"].get<double>();
  }
  if (j.contains("replay_cap")) {
    v.tsp->replay_cap = j["replay_cap"].get<int>();
    if (v.tsp->replay_cap < 8) {
      v.tsp->replay_cap = 8;
    }
  }
  if (j.contains("temp_recalib_every")) {
    v.tsp->temp_recalib_every = j["temp_recalib_every"].get<int>();
    if (v.tsp->temp_recalib_every < 0) {
      v.tsp->temp_recalib_every = 0;
    }
  }
  if (j.contains("align_every")) {
    v.tsp->align_every = j["align_every"].get<int>();
    if (v.tsp->align_every < 0) {
      v.tsp->align_every = 0;
    }
  }
  return true;
}

bool load_bundle_into(ModelView v, const std::string& cypha_path, const std::string& pre_path,
                      const std::string& ff_json_path, const std::string& train_hparams_path_opt,
                      const std::string& regression_json_path_opt, bool reset_session) {
  cypha::Cypha* c = cypha_for_view(v);
  if (c != nullptr) {
    if (!c->load(cypha_path, pre_path, ff_json_path, regression_json_path_opt,
                 train_hparams_path_opt)) {
      std::cerr << "Provide world.F_field inside .cypha or a valid --f-field-json / failed load\n";
      return false;
    }
    if (reset_session && c == g_cypha.get()) {
      reset_session_counters();
    }
    return true;
  }

  apply_default_train_hparams_view(v);
  v.reg_mu->clear();
  v.reg_var->clear();
  clear_mke_state_in(v);
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

  v.model->reset(new cypha::CyphaInferModel(cypha::CyphaInferModel::from_root(root, ff_ptr, fd)));
  v.mem->reset(
      new cypha::CyphaDifMemoryState(cypha::CyphaDifMemoryState::from_cypha_root(root, ff_ptr, fd)));
  load_kernel_from_root_view(v, root, d);

  v.pre->reset();
  if (!pre_path.empty()) {
    v.pre->reset(
        new cypha::PreprocessorState(cypha::PreprocessorState::from_json_file(pre_path.c_str())));
  }

  if (!train_hparams_path_opt.empty()) {
    if (!try_load_train_hparams_file_view(train_hparams_path_opt, v)) {
      std::cerr << "warning: could not read --train-hparams " << train_hparams_path_opt << "\n";
    }
  } else {
    fs::path auto_hp = fs::path(cypha_path).parent_path() / "train_hparams.json";
    if (fs::exists(auto_hp)) {
      try_load_train_hparams_file_view(auto_hp.string(), v);
    }
  }

  snapshot_gh_clean_metric_view(v);
  *v.enc_updates = 0;
  *v.replay = std::make_unique<cypha::ReplayBuffer>(v.tsp->replay_cap);
  *v.total_steps = (*v.model)->saved_total_steps;
  *v.llr_ema = (*v.model)->llr_ema;
  if (reset_session) {
    reset_session_counters();
  }
  if (!regression_json_path_opt.empty()) {
    if (!try_load_regression_head_json_view(regression_json_path_opt, v)) {
      std::cerr << "failed to read --regression-json " << regression_json_path_opt << "\n";
      return false;
    }
  }
  return true;
}

bool load_registry_ref_into_map(const cypha::RegistryModelRef& ref, bool set_active, bool reset_session) {
  const std::string key = model_registry_key(ref.name, ref.version);
  fs::path dir = fs::path(ref.model_path).parent_path();
  fs::path ff_path = dir / "f_field.json";
  std::string ff_p = fs::exists(ff_path) ? ff_path.string() : "";
  fs::path reg_path = dir / "regression_head.json";
  std::string reg_p = fs::exists(reg_path) ? reg_path.string() : "";

  if (set_active) {
    if (!load_bundle_into(view_from_globals(), ref.model_path, ref.preprocessor_path, ff_p, "", reg_p,
                          reset_session)) {
      return false;
    }
    g_active_model_key = key;
  }

  LoadedModelBundle& slot = g_models[key];
  std::lock_guard<std::mutex> slot_lk(slot.mu);
  return load_bundle_into(view_from_bundle(slot), ref.model_path, ref.preprocessor_path, ff_p, "", reg_p,
                          false);
}

void preload_registry_models() {
  if (g_registry_root.empty() || g_registry_cache.empty()) {
    return;
  }
  for (const auto& ref : g_registry_cache) {
    const std::string key = model_registry_key(ref.name, ref.version);
    if (key == g_active_model_key) {
      continue;
    }
    if (g_models.find(key) != g_models.end()) {
      continue;
    }
    if (!load_registry_ref_into_map(ref, false, false)) {
      std::cerr << "warning: preload failed for " << key << "\n";
    }
  }
}

bool load_bundle_paths(const std::string& cypha_path, const std::string& pre_path,
                       const std::string& ff_json_path, const std::string& train_hparams_path_opt,
                       const std::string& regression_json_path_opt) {
  return load_bundle_into(view_from_globals(), cypha_path, pre_path, ff_json_path,
                          train_hparams_path_opt, regression_json_path_opt, true);
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

/// Resolve ``body["model"]`` (``name/version``) to a loaded slot; default = active globals.
bool resolve_model_view(const nlohmann::json& body, ModelView& out_view, LoadedModelBundle** out_slot,
                        std::string* detail_out, int* status_out = nullptr) {
  std::string req_key;
  if (body.contains("model") && body["model"].is_string()) {
    req_key = body["model"].get<std::string>();
  }
  if (req_key.empty() || req_key == g_active_model_key) {
    out_view = view_from_globals();
    *out_slot = nullptr;
    if (!g_cypha->loaded() || !g_cypha->infer() || !g_cypha->mem()) {
      if (detail_out) {
        *detail_out = R"({"detail":"No model loaded"})";
      }
      if (status_out) {
        *status_out = 503;
      }
      return false;
    }
    return true;
  }
  auto it = g_models.find(req_key);
  if (it == g_models.end()) {
    if (detail_out) {
      *detail_out = R"({"detail":"model not loaded"})";
    }
    if (status_out) {
      *status_out = 404;
    }
    return false;
  }
  *out_slot = &it->second;
  out_view = view_from_bundle(it->second);
  if (!it->second.cypha || !it->second.cypha->loaded() || !it->second.cypha->infer() ||
      !it->second.cypha->mem()) {
    if (detail_out) {
      *detail_out = R"({"detail":"model not loaded"})";
    }
    if (status_out) {
      *status_out = 404;
    }
    return false;
  }
  return true;
}

/// Cypha-owned view → Cypha::predict/update (One Cypha cutover).
bool use_cypha_primary_predict(ModelView v) { return cypha_for_view(v) != nullptr; }

bool use_cypha_primary_update(ModelView v) { return cypha_for_view(v) != nullptr; }

std::string json_predict_via_cypha(cypha::Cypha& cypha, const nlohmann::json& body) {
  cypha::CyphaInferModel& model = *cypha.infer();
  auto t0 = std::chrono::steady_clock::now();
  std::vector<double> x;
  for (const auto& val : body.at("input")) {
    x.push_back(val.get<double>());
  }

  cypha::PredictOpts opts{};
  opts.use_gh = body.value("use_gh", true);
  opts.use_field = body.value("use_field", true);
  opts.deliberation_lo = body.value("deliberation_lo", model.deliberation_lo);
  opts.deliberation_hi = body.value("deliberation_hi", model.deliberation_hi);
  opts.self_correct = body.value("self_correct", false);
  opts.self_correct_max_passes = body.value("self_correct_max_passes", 3);

  const cypha::PredictOut pred = cypha.predict(x.data(), static_cast<int>(x.size()), opts);
  if (!pred.detail.empty()) {
    nlohmann::json err;
    err["detail"] = pred.detail;
    return err.dump();
  }

  const int k = static_cast<int>(pred.labels.size());
  const bool use_gh = opts.use_gh;
  const bool mke_active = cypha.mke_active();
  const bool self_correct = opts.self_correct;
  const double conf = pred.confidence;
  const double anomaly = pred.anomaly_score;
  const bool is_ood = pred.is_ood;
  const double r_eff = pred.r_eff;

  nlohmann::json scores = nlohmann::json::object();
  for (int j = 0; j < k; ++j) {
    scores[pred.labels[static_cast<std::size_t>(j)]] = pred.all_scores[static_cast<std::size_t>(j)];
  }

  std::vector<double> H;
  {
    std::vector<double> x_pp = x;
    if (cypha.preprocessor()) {
      x_pp = cypha.preprocessor()->transform_one(x_pp);
    }
    if (mke_active) {
      if (static_cast<int>(x_pp.size()) == cypha.mke_d_in()) {
        H.resize(static_cast<std::size_t>(model.d_latent));
        cypha::regression::rff_encode_batch_rowmajor(x_pp.data(), 1, cypha.mke_d_in(), cypha.mke_W().data(),
                                                     cypha.mke_b().data(), model.d_latent, H.data());
      }
    } else if (static_cast<int>(x_pp.size()) == model.d_latent) {
      H.resize(static_cast<std::size_t>(model.d_latent));
      cypha::batch_encode(model, x_pp.data(), 1, H);
    }
  }

  {
    cypha::intelligence::ProfileObservation obs;
    const int d = model.d_latent;
    if (d > 0 && static_cast<int>(H.size()) == d) {
      std::vector<double> x_pp = x;
      if (cypha.preprocessor()) {
        x_pp = cypha.preprocessor()->transform_one(x_pp);
      }
      if (!mke_active && static_cast<int>(x_pp.size()) == d) {
        obs.alpha = cypha::intelligence::compute_alpha_gria(x_pp.data(), H.data(), 1, d);
      }
    }

    double epistemic_var = 0.0;
    double aleatoric_var = 0.0;
    bool has_r_eu = false;
    if (pred.self_corrected && pred.r_eu_proxy > 0.0) {
      obs.r_eu = std::clamp(pred.r_eu_proxy, 0.0, 1.0);
      has_r_eu = true;
    } else if (use_gh && !mke_active) {
      epistemic_var = std::max(anomaly, 1e-6);
      aleatoric_var = std::max(1.0 - anomaly, 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
      has_r_eu = true;
    } else if (mke_active && static_cast<int>(cypha.reg_var().size()) == k) {
      const double eps = 1e-8;
      const double T = cypha.mke_temperature();
      std::vector<double> z(static_cast<std::size_t>(k));
      for (int j = 0; j < k; ++j) {
        z[static_cast<std::size_t>(j)] = pred.all_scores[static_cast<std::size_t>(j)] / (T + eps);
      }
      std::vector<double> probs;
      cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
      double v_mix = 0.0;
      for (int j = 0; j < k; ++j) {
        v_mix += probs[static_cast<std::size_t>(j)] * cypha.reg_var()[static_cast<std::size_t>(j)];
      }
      aleatoric_var = std::max(v_mix, 1e-6);
      epistemic_var = std::max((1.0 - conf) * (1.0 - conf), 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
      has_r_eu = true;
    } else if (static_cast<int>(cypha.reg_mu().size()) == k &&
               static_cast<int>(cypha.reg_var().size()) == k) {
      const double eps = 1e-8;
      const double T = model.temperature;
      std::vector<double> z(static_cast<std::size_t>(k));
      for (int j = 0; j < k; ++j) {
        z[static_cast<std::size_t>(j)] = pred.all_scores[static_cast<std::size_t>(j)] / (T + eps);
      }
      std::vector<double> probs;
      cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
      double y_mix = 0.0;
      double u_mix = 0.0;
      cypha::regression::predict_mixture_scalar(probs.data(), cypha.reg_mu().data(),
                                                cypha.reg_var().data(), static_cast<std::size_t>(k),
                                                y_mix, u_mix);
      aleatoric_var = std::max(u_mix * u_mix, 1e-6);
      epistemic_var = std::max((1.0 - conf) * (1.0 - conf), 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
      has_r_eu = true;
    }
    if (!has_r_eu) {
      epistemic_var = std::max((1.0 - conf) * (1.0 - conf), 1e-6);
      if (is_ood) {
        epistemic_var = std::max(epistemic_var, 0.5);
      }
      aleatoric_var = std::max(conf * conf, 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
    }

    if (body.contains("label") && body["label"].is_string()) {
      const std::string gt_label = body["label"].get<std::string>();
      const int correct = pred.label == gt_label ? 1 : 0;
      const double conf_arr[] = {conf};
      const int correct_arr[] = {correct};
      obs.calibration = cypha::intelligence::compute_calibration(conf_arr, correct_arr, 1);
    }

    if (pred.self_corrected) {
      obs.tau = cypha::intelligence::normalize_memory_depth(pred.correction_passes, 8);
      g_engine_corrections += 1;
      const double r_eu_after = pred.r_eu_proxy;
      const double r_eu_before = std::min(1.0, r_eu_after + 0.2);
      const double resolution = std::max(0.0, r_eu_before - r_eu_after);
      g_causal_graph_monitor.simulation_step(r_eu_before, r_eu_after, resolution);
    }

    g_intelligence_profiler.update(obs);
    g_causal_graph_monitor.observe_profile(obs);
  }

  const double latency =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  g_predictions += 1;
  g_sess.push_back(SessPred{pred.label, conf, anomaly, is_ood});

  nlohmann::json out;
  out["label"] = pred.label;
  out["confidence"] = conf;
  out["all_scores"] = scores;
  out["anomaly_score"] = anomaly;
  out["is_ood"] = is_ood;
  if (pred.y.has_value()) {
    out["regression_val"] = *pred.y;
  } else {
    out["regression_val"] = nullptr;
  }
  out["uncertainty"] = pred.uncertainty;
  const bool want_expl = body.value("return_explanation", false);
  if (want_expl && static_cast<int>(H.size()) == model.d_latent) {
    nlohmann::json expl;
    expl["label"] = out["label"];
    expl["confidence"] = conf;
    expl["all_scores"] = scores;
    expl["anomaly_score"] = anomaly;
    expl["is_ood"] = is_ood;
    expl["r_eff"] = use_gh ? r_eff : 0.0;
    nlohmann::json cdet = nlohmann::json::object();
    const int d = model.d_latent;
    for (int ci = 0; ci < k; ++ci) {
      double sumsq = 0.0;
      for (int j = 0; j < d; ++j) {
        double dv = model.D[static_cast<std::size_t>(ci * d + j)];
        sumsq += dv * dv;
      }
      nlohmann::json row;
      row["n_obs"] = model.n_obs[static_cast<std::size_t>(ci)];
      row["delta_mu_norm"] = std::sqrt(sumsq);
      cdet[model.labels[static_cast<std::size_t>(ci)]] = row;
    }
    expl["class_details"] = cdet;
    double wh = 0.0;
    for (int j = 0; j < d; ++j) {
      double t = H[static_cast<std::size_t>(j)] - model.mu_world[static_cast<std::size_t>(j)];
      wh += t * t;
    }
    expl["world_mu_distance"] = std::sqrt(wh);
    out["explanation"] = std::move(expl);
  } else {
    out["explanation"] = nullptr;
  }
  out["latency_ms"] = latency;
  if (self_correct && !mke_active && !use_gh) {
    out["self_corrected"] = pred.self_corrected;
    out["correction_passes"] = pred.correction_passes;
    out["r_eu_proxy"] = pred.r_eu_proxy;
  }
  return out.dump();
}

std::string json_update_via_cypha(cypha::Cypha& cypha, const nlohmann::json& body) {
  sync_kernel_from_json_view(view_from_cypha(cypha), body);

  std::vector<double> x;
  for (const auto& val : body.at("input")) {
    x.push_back(val.get<double>());
  }

  const bool has_regr_y = body.contains("regression_y") && !body["regression_y"].is_null();
  if (has_regr_y && !body["regression_y"].is_number()) {
    return R"({"detail":"regression_y must be a number"})";
  }

  std::vector<double> replay_u01_storage;
  if (body.contains("replay_u01") && body["replay_u01"].is_array()) {
    for (const auto& rv : body["replay_u01"]) {
      replay_u01_storage.push_back(rv.get<double>());
    }
  }

  cypha::UpdateOpts opts{};
  opts.use_gh = body.value("use_gh", true);
  opts.ewc_snapshot = body.value("ewc_snapshot", false);
  if (body.contains("ewc_lambda") && body["ewc_lambda"].is_number()) {
    opts.ewc_lambda = body["ewc_lambda"].get<double>();
  }
  if (!replay_u01_storage.empty()) {
    opts.replay_u01 = replay_u01_storage.data();
    opts.replay_u01_len = replay_u01_storage.size();
  }
  std::string router_storage;
  if (body.contains("router_train_label") && body["router_train_label"].is_string()) {
    router_storage = body["router_train_label"].get<std::string>();
    if (!router_storage.empty()) {
      opts.router_train_label = &router_storage;
    }
  }

  double y_val = 0.0;
  const double* y_ptr = nullptr;
  if (has_regr_y) {
    y_val = body["regression_y"].get<double>();
    y_ptr = &y_val;
  }
  std::string label = body.at("correct_label").get<std::string>();
  const std::string* label_ptr = has_regr_y ? nullptr : &label;

  const cypha::UpdateOut upd =
      cypha.update(x.data(), static_cast<int>(x.size()), label_ptr, y_ptr, opts);
  if (!upd.detail.empty()) {
    nlohmann::json err;
    err["detail"] = upd.detail;
    return err.dump();
  }
  g_engine_corrections += 1;

  nlohmann::json out;
  out["loss"] = upd.loss;
  out["n_corrections"] = g_engine_corrections;
  return out.dump();
}

std::string json_predict_impl(const nlohmann::json& body, ModelView v) {
  if (use_cypha_primary_predict(v)) {
    sync_kernel_from_json_view(v, body);
    return json_predict_via_cypha(*cypha_for_view(v), body);
  }
  if (!*v.model || !*v.mem) {
    return R"({"detail":"No model loaded"})";
  }
  cypha::CyphaInferModel& model = **v.model;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<double> x;
  for (const auto& val : body.at("input")) {
    x.push_back(val.get<double>());
  }
  if (*v.pre) {
    x = (*v.pre)->transform_one(x);
  }
  std::vector<double> H;
  if (*v.mke_active) {
    if (static_cast<int>(x.size()) != *v.mke_d_in) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
    H.resize(static_cast<std::size_t>(model.d_latent));
    cypha::regression::rff_encode_batch_rowmajor(x.data(), 1, *v.mke_d_in, v.mke_W->data(), v.mke_b->data(),
                                                  model.d_latent, H.data());
  } else {
    if (static_cast<int>(x.size()) != model.d_latent) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
    cypha::batch_encode(model, x.data(), 1, H);
  }

  const int k = static_cast<int>(model.labels.size());
  bool use_gh = body.value("use_gh", true);
  sync_kernel_from_json_view(v, body);

  cypha::CyphaInferOptions iopt{};
  iopt.deliberation_lo = body.value("deliberation_lo", model.deliberation_lo);
  iopt.deliberation_hi = body.value("deliberation_hi", model.deliberation_hi);
  iopt.use_field = true;
  apply_kernel_to_options_view(v, iopt);

  std::string pred_label;
  double conf = 0.0;
  double r_eff = 0.0;
  std::vector<double> llr_for_scores;
  double anomaly = 0.0;
  bool is_ood = false;
  bool self_corrected = false;
  int correction_passes = 0;
  double r_eu_proxy = 0.0;

  if (*v.mke_active) {
    std::vector<double> llr;
    cypha::score_matrix_use_field(model, H.data(), 1, llr);
    double eps = 1e-8;
    double T = *v.mke_temperature;
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] = llr[static_cast<std::size_t>(j)] / (T + eps);
    }
    std::vector<double> probs;
    cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
    int bi = 0;
    for (int j = 1; j < k; ++j) {
      if (probs[static_cast<std::size_t>(j)] > probs[static_cast<std::size_t>(bi)]) {
        bi = j;
      }
    }
    pred_label = model.labels[static_cast<std::size_t>(bi)];
    conf = probs[static_cast<std::size_t>(bi)];
    llr_for_scores = std::move(llr);
  } else if (use_gh) {
    iopt.gh_chi = *v.gh_chi;
    iopt.gh_psi = *v.gh_psi;
    cypha::GhInferAtHResult gh =
        cypha::gh_infer_at_h(model, H.data(), *v.gh_chi, *v.gh_psi, kGhNigAdaptAlpha, &iopt);
    pred_label = gh.label;
    conf = gh.confidence;
    r_eff = gh.r_eff;
    llr_for_scores = std::move(gh.llrs);
    const double r_base = (model.has_mahal_ema && model.mahal_ema > 0.0) ? model.mahal_ema : 1.0;
    anomaly = cypha::gh_infer_anomaly_score(r_eff, r_base);
    is_ood = anomaly > kOodThreshold;
  } else {
    const bool self_correct = body.value("self_correct", false);
    const int max_passes = body.value("self_correct_max_passes", 3);
    if (self_correct) {
      const auto scr =
          cypha::intelligence::self_correcting_infer_at_h(model, H.data(), iopt, g_epistemic_threshold,
                                                          max_passes);
      pred_label = scr.infer.label;
      conf = scr.infer.confidence;
      llr_for_scores = std::move(scr.infer.llrs);
      anomaly = 0.0;
      is_ood = false;
      self_corrected = scr.corrected;
      correction_passes = scr.correction_passes;
      r_eu_proxy = scr.r_eu_proxy;
      g_engine_corrections += scr.corrected ? 1 : 0;
      if (scr.corrected) {
        const double r_eu_after = scr.r_eu_proxy;
        const double r_eu_before = std::min(1.0, r_eu_after + 0.2);
        const double resolution = std::max(0.0, r_eu_before - r_eu_after);
        g_causal_graph_monitor.simulation_step(r_eu_before, r_eu_after, resolution);
      }
    } else {
      cypha::InferAtHResult inf = cypha::infer_at_h(model, H.data(), iopt);
      pred_label = inf.label;
      conf = inf.confidence;
      llr_for_scores = std::move(inf.llrs);
      anomaly = 0.0;
      is_ood = false;
    }
  }

  nlohmann::json scores = nlohmann::json::object();
  for (int j = 0; j < k; ++j) {
    scores[model.labels[static_cast<std::size_t>(j)]] = llr_for_scores[static_cast<std::size_t>(j)];
  }
  double latency = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  g_predictions += 1;
  g_sess.push_back(SessPred{pred_label, conf, anomaly, is_ood});

  {
    cypha::intelligence::ProfileObservation obs;
    const int d = model.d_latent;
    if (d > 0 && static_cast<int>(x.size()) == d && static_cast<int>(H.size()) == d) {
      obs.alpha = cypha::intelligence::compute_alpha_gria(x.data(), H.data(), 1, d);
    }

    double epistemic_var = 0.0;
    double aleatoric_var = 0.0;
    bool has_r_eu = false;
    if (self_corrected && r_eu_proxy > 0.0) {
      obs.r_eu = std::clamp(r_eu_proxy, 0.0, 1.0);
      has_r_eu = true;
    } else if (use_gh && !*v.mke_active) {
      epistemic_var = std::max(anomaly, 1e-6);
      aleatoric_var = std::max(1.0 - anomaly, 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
      has_r_eu = true;
    } else if (*v.mke_active && static_cast<int>(v.reg_var->size()) == k) {
      double eps = 1e-8;
      double T = *v.mke_temperature;
      std::vector<double> z(static_cast<std::size_t>(k));
      for (int j = 0; j < k; ++j) {
        z[static_cast<std::size_t>(j)] = llr_for_scores[static_cast<std::size_t>(j)] / (T + eps);
      }
      std::vector<double> probs;
      cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
      double v_mix = 0.0;
      for (int j = 0; j < k; ++j) {
        v_mix += probs[static_cast<std::size_t>(j)] * (*v.reg_var)[static_cast<std::size_t>(j)];
      }
      aleatoric_var = std::max(v_mix, 1e-6);
      epistemic_var = std::max((1.0 - conf) * (1.0 - conf), 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
      has_r_eu = true;
    } else if (static_cast<int>(v.reg_mu->size()) == k && static_cast<int>(v.reg_var->size()) == k) {
      double eps = 1e-8;
      double T = model.temperature;
      std::vector<double> z(static_cast<std::size_t>(k));
      for (int j = 0; j < k; ++j) {
        z[static_cast<std::size_t>(j)] = llr_for_scores[static_cast<std::size_t>(j)] / (T + eps);
      }
      std::vector<double> probs;
      cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
      double y_mix = 0.0;
      double u_mix = 0.0;
      cypha::regression::predict_mixture_scalar(probs.data(), v.reg_mu->data(), v.reg_var->data(),
                                                static_cast<std::size_t>(k), y_mix, u_mix);
      aleatoric_var = std::max(u_mix * u_mix, 1e-6);
      epistemic_var = std::max((1.0 - conf) * (1.0 - conf), 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
      has_r_eu = true;
    }
    if (!has_r_eu) {
      epistemic_var = std::max((1.0 - conf) * (1.0 - conf), 1e-6);
      if (is_ood) {
        epistemic_var = std::max(epistemic_var, 0.5);
      }
      aleatoric_var = std::max(conf * conf, 1e-6);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
    }

    if (body.contains("label") && body["label"].is_string()) {
      const std::string gt_label = body["label"].get<std::string>();
      const int correct = pred_label == gt_label ? 1 : 0;
      const double conf_arr[] = {conf};
      const int correct_arr[] = {correct};
      obs.calibration =
          cypha::intelligence::compute_calibration(conf_arr, correct_arr, 1);
    }

    if (self_corrected) {
      obs.tau = cypha::intelligence::normalize_memory_depth(correction_passes, 8);
    }

    g_intelligence_profiler.update(obs);
    g_causal_graph_monitor.observe_profile(obs);
  }

  nlohmann::json out;
  out["label"] = pred_label;
  out["confidence"] = conf;
  out["all_scores"] = scores;
  out["anomaly_score"] = anomaly;
  out["is_ood"] = is_ood;
  if (*v.mke_active) {
    double eps = 1e-8;
    double T = *v.mke_temperature;
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] = llr_for_scores[static_cast<std::size_t>(j)] / (T + eps);
    }
    std::vector<double> probs;
    cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
    double y_mix = 0.0;
    const int d_rff = model.d_latent;
    for (int j = 0; j < k; ++j) {
      const std::string& lbl = model.labels[static_cast<std::size_t>(j)];
      auto it = v.mke_w->find(lbl);
      if (it == v.mke_w->end() || static_cast<int>(it->second.size()) != d_rff) {
        continue;
      }
      double dp = 0.0;
      for (int t = 0; t < d_rff; ++t) {
        dp += it->second[static_cast<std::size_t>(t)] * H[static_cast<std::size_t>(t)];
      }
      y_mix += probs[static_cast<std::size_t>(j)] * dp;
    }
    out["regression_val"] = y_mix;
    if (static_cast<int>(v.reg_var->size()) == k) {
      double v_mix = 0.0;
      for (int j = 0; j < k; ++j) {
        v_mix += probs[static_cast<std::size_t>(j)] * (*v.reg_var)[static_cast<std::size_t>(j)];
      }
      out["uncertainty"] = std::sqrt(std::max(v_mix, 0.0));
    } else {
      out["uncertainty"] = 0.0;
    }
  } else if (static_cast<int>(v.reg_mu->size()) == k && static_cast<int>(v.reg_var->size()) == k) {
    double eps = 1e-8;
    double T = model.temperature;
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] = llr_for_scores[static_cast<std::size_t>(j)] / (T + eps);
    }
    std::vector<double> probs;
    cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
    double y_mix = 0.0;
    double u_mix = 0.0;
    cypha::regression::predict_mixture_scalar(probs.data(), v.reg_mu->data(), v.reg_var->data(),
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
    const int d = model.d_latent;
    for (int ci = 0; ci < k; ++ci) {
      double sumsq = 0.0;
      for (int j = 0; j < d; ++j) {
        double dv = model.D[static_cast<std::size_t>(ci * d + j)];
        sumsq += dv * dv;
      }
      nlohmann::json row;
      row["n_obs"] = model.n_obs[static_cast<std::size_t>(ci)];
      row["delta_mu_norm"] = std::sqrt(sumsq);
      cdet[model.labels[static_cast<std::size_t>(ci)]] = row;
    }
    expl["class_details"] = cdet;
    double wh = 0.0;
    for (int j = 0; j < d; ++j) {
      double t = H[static_cast<std::size_t>(j)] - model.mu_world[static_cast<std::size_t>(j)];
      wh += t * t;
    }
    expl["world_mu_distance"] = std::sqrt(wh);
    out["explanation"] = std::move(expl);
  } else {
    out["explanation"] = nullptr;
  }
  out["latency_ms"] = latency;
  if (body.value("self_correct", false) && !*v.mke_active && !use_gh) {
    out["self_corrected"] = self_corrected;
    out["correction_passes"] = correction_passes;
    out["r_eu_proxy"] = r_eu_proxy;
  }
  return out.dump();
}

std::string json_predict(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  ModelView view{};
  LoadedModelBundle* slot = nullptr;
  std::string detail;
  if (!resolve_model_view(body, view, &slot, &detail)) {
    return detail;
  }
  if (slot != nullptr) {
    std::lock_guard<std::mutex> slot_lk(slot->mu);
    return json_predict_impl(body, view);
  }
  return json_predict_impl(body, view);
}

// Result of the uncertainty-rank pipeline: the JSON response body plus the HTTP status
// the caller should set. Keeping the status explicit here (rather than having the route
// handler re-derive it by substring-matching the body) avoids fragile string sniffing.
struct UncertaintyRankResult {
  std::string body;
  int status = 200;
};

UncertaintyRankResult json_uncertainty_rank_impl(const nlohmann::json& body, ModelView v) {
  if (!*v.model || !*v.mem) {
    return {R"({"detail":"No model loaded"})", 503};
  }
  if (*v.mke_active) {
    return {R"({"detail":"uncertainty-rank not supported in MKE mode"})", 400};
  }
  if (!body.contains("rows") || !body["rows"].is_array()) {
    return {std::string(R"json({"detail":"rows required (array of feature vectors)"})json"), 400};
  }
  const auto& rows_j = body["rows"];
  const int n = static_cast<int>(rows_j.size());
  if (n == 0) {
    nlohmann::json out;
    out["indices"] = nlohmann::json::array();
    out["entropies"] = nlohmann::json::array();
    out["top_n"] = 0;
    return {out.dump(), 200};
  }
  if (!rows_j[0].is_array() || rows_j[0].empty()) {
    return {R"({"detail":"each row must be a non-empty feature array"})", 400};
  }
  const int d_row = static_cast<int>(rows_j[0].size());
  for (int i = 1; i < n; ++i) {
    if (!rows_j[i].is_array() || static_cast<int>(rows_j[i].size()) != d_row) {
      return {R"({"detail":"rows must be uniform-length feature arrays"})", 400};
    }
  }

  cypha::CyphaInferModel& model = **v.model;
  std::vector<double> x_latent;
  x_latent.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(model.d_latent));
  for (int i = 0; i < n; ++i) {
    std::vector<double> x;
    x.reserve(static_cast<std::size_t>(d_row));
    for (const auto& val : rows_j[i]) {
      x.push_back(val.get<double>());
    }
    if (*v.pre) {
      x = (*v.pre)->transform_one(x);
    }
    if (static_cast<int>(x.size()) != model.d_latent) {
      return {R"({"detail":"input dim mismatch after preprocessor"})", 400};
    }
    x_latent.insert(x_latent.end(), x.begin(), x.end());
  }

  const int k = static_cast<int>(model.labels.size());
  std::vector<double> H(static_cast<std::size_t>(n) * static_cast<std::size_t>(model.d_latent));
  cypha::batch_encode(model, x_latent.data(), n, H);

  std::vector<double> llr(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  cypha::score_matrix_use_field(model, H.data(), n, llr);

  const double eps = 1e-8;
  const double T = body.value("temperature", model.temperature);
  std::vector<double> z(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  for (int i = 0; i < n * k; ++i) {
    z[static_cast<std::size_t>(i)] = llr[static_cast<std::size_t>(i)] / (T + eps);
  }
  std::vector<double> probs(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  cypha::softmax_batch_reference(z.data(), n, k, eps, probs);

  struct RankRow {
    int index;
    double entropy;
    double confidence;
  };
  std::vector<RankRow> ranked;
  ranked.reserve(static_cast<std::size_t>(n));
  const bool curriculum = body.value("curriculum", false);
  for (int i = 0; i < n; ++i) {
    const double* prow = probs.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(k);
    const double entropy = cypha::row_entropy_from_probs(prow, k, eps);
    const double confidence = cypha::row_max_softmax_confidence(prow, k);
    ranked.push_back({i, entropy, confidence});
  }
  if (curriculum) {
    std::sort(ranked.begin(), ranked.end(), [](const RankRow& a, const RankRow& b) {
      if (a.confidence != b.confidence) {
        return a.confidence < b.confidence;
      }
      return a.index < b.index;
    });
  } else {
    std::sort(ranked.begin(), ranked.end(), [](const RankRow& a, const RankRow& b) {
      if (a.entropy != b.entropy) {
        return a.entropy > b.entropy;
      }
      return a.index < b.index;
    });
  }

  int top_n = body.value("top_n", n);
  if (top_n < 0) {
    top_n = 0;
  }
  if (top_n > n) {
    top_n = n;
  }

  nlohmann::json out;
  nlohmann::json indices = nlohmann::json::array();
  nlohmann::json entropies = nlohmann::json::array();
  nlohmann::json confidences = nlohmann::json::array();
  for (int i = 0; i < top_n; ++i) {
    indices.push_back(ranked[static_cast<std::size_t>(i)].index);
    entropies.push_back(ranked[static_cast<std::size_t>(i)].entropy);
    confidences.push_back(ranked[static_cast<std::size_t>(i)].confidence);
  }
  out["indices"] = std::move(indices);
  out["entropies"] = std::move(entropies);
  out["confidences"] = std::move(confidences);
  out["top_n"] = top_n;
  if (curriculum) {
    out["curriculum"] = true;
  }
  return {out.dump(), 200};
}

UncertaintyRankResult json_uncertainty_rank(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  ModelView view{};
  LoadedModelBundle* slot = nullptr;
  std::string detail;
  int status = 400;
  if (!resolve_model_view(body, view, &slot, &detail, &status)) {
    return {detail, status};
  }
  if (slot != nullptr) {
    std::lock_guard<std::mutex> slot_lk(slot->mu);
    return json_uncertainty_rank_impl(body, view);
  }
  return json_uncertainty_rank_impl(body, view);
}

std::string json_update_batch_impl(const nlohmann::json& body, ModelView v) {
  cypha::Cypha* cypha = cypha_for_view(v);
  if (cypha == nullptr || !cypha->infer() || !cypha->mem()) {
    return R"({"detail":"No model loaded"})";
  }
  const auto& batch = body.at("batch");
  const int n = static_cast<int>(batch.size());
  if (n == 0) {
    nlohmann::json out;
    out["losses"] = nlohmann::json::array();
    out["n_corrections"] = g_engine_corrections;
    return out.dump();
  }

  std::vector<int> order(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    order[static_cast<std::size_t>(i)] = i;
  }

  const bool curriculum = body.value("curriculum", false);
  if (curriculum) {
    cypha::CyphaInferModel& model = *cypha->infer();
    const int k = static_cast<int>(model.labels.size());
    const double eps = 1e-8;
    const double T = model.temperature;
    std::vector<double> confidences;
    confidences.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const auto& item = batch[i];
      std::vector<double> x;
      for (const auto& val : item.at("input")) {
        x.push_back(val.get<double>());
      }
      if (cypha->preprocessor()) {
        x = cypha->preprocessor()->transform_one(x);
      }
      if (static_cast<int>(x.size()) != model.d_latent) {
        return R"({"detail":"input dim mismatch after preprocessor"})";
      }
      std::vector<double> H;
      cypha::batch_encode(model, x.data(), 1, H);
      std::vector<double> llr;
      cypha::score_matrix_use_field(model, H.data(), 1, llr);
      if (k > 0) {
        std::vector<double> z(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) {
          z[static_cast<std::size_t>(j)] = llr[static_cast<std::size_t>(j)] / (T + eps);
        }
        std::vector<double> probs;
        cypha::softmax_batch_reference(z.data(), 1, k, eps, probs);
        confidences.push_back(cypha::row_max_softmax_confidence(probs.data(), k));
      } else {
        confidences.push_back(0.0);
      }
    }
    order = cypha::curriculum_order_ascending_confidence(confidences, n);
  }

  nlohmann::json losses = nlohmann::json::array();
  for (int ord : order) {
    nlohmann::json one = batch[ord];
    if (body.contains("use_gh")) {
      one["use_gh"] = body["use_gh"];
    }
    if (body.contains("ewc_lambda")) {
      one["ewc_lambda"] = body["ewc_lambda"];
    }
    if (body.contains("ewc_snapshot")) {
      one["ewc_snapshot"] = body["ewc_snapshot"];
    }
    if (body.contains("replay_u01")) {
      one["replay_u01"] = body["replay_u01"];
    }
    if (body.contains("use_kernel_llr")) {
      one["use_kernel_llr"] = body["use_kernel_llr"];
    }
    if (body.contains("kernel_blend")) {
      one["kernel_blend"] = body["kernel_blend"];
    }
    const std::string sub = json_update_via_cypha(*cypha, one);
    const auto parsed = nlohmann::json::parse(sub, nullptr, false);
    if (!parsed.is_discarded() && parsed.contains("detail")) {
      return sub;
    }
    const auto j = nlohmann::json::parse(sub);
    losses.push_back(j.value("loss", 0.0));
  }

  nlohmann::json out;
  out["losses"] = std::move(losses);
  out["n_corrections"] = g_engine_corrections;
  if (curriculum) {
    out["curriculum"] = true;
  }
  return out.dump();
}

std::string json_update_impl(const nlohmann::json& body, ModelView v) {
  if (body.contains("batch") && body.at("batch").is_array()) {
    return json_update_batch_impl(body, v);
  }
  if (use_cypha_primary_update(v)) {
    return json_update_via_cypha(*cypha_for_view(v), body);
  }
  return R"({"detail":"No model loaded"})";
}

std::string json_update(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  ModelView view{};
  LoadedModelBundle* slot = nullptr;
  std::string detail;
  if (!resolve_model_view(body, view, &slot, &detail)) {
    return detail;
  }
  if (slot != nullptr) {
    std::lock_guard<std::mutex> slot_lk(slot->mu);
    return json_update_impl(body, view);
  }
  return json_update_impl(body, view);
}

std::string json_adapt_temperature(const nlohmann::json& body) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_cypha->loaded() || !g_cypha->infer() || !g_cypha->mem()) {
    return R"({"detail":"No model loaded"})";
  }
  cypha::CyphaInferModel& model = *g_cypha->infer();
  cypha::CyphaDifMemoryState& mem = *g_cypha->mem();
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
  const int d = model.d_latent;
  std::vector<double> h_batch;
  std::vector<int> true_idx;
  for (const auto& row : cal) {
    std::vector<double> xv;
    for (const auto& v : row.at("input")) {
      xv.push_back(v.get<double>());
    }
    if (g_cypha->preprocessor()) {
      xv = g_cypha->preprocessor()->transform_one(xv);
    }
    if (static_cast<int>(xv.size()) != d) {
      return R"({"detail":"input dim mismatch after preprocessor"})";
    }
    std::string y = row.at("correct_label").get<std::string>();
    auto it = mem.label_index.find(y);
    if (it == mem.label_index.end()) {
      continue;
    }
    std::vector<double> oneh;
    cypha::batch_encode(model, xv.data(), 1, oneh);
    h_batch.insert(h_batch.end(), oneh.begin(), oneh.end());
    true_idx.push_back(it->second);
  }
  nlohmann::json out;
  out["n_used"] = static_cast<int>(true_idx.size());
  if (true_idx.empty()) {
    out["temperature"] = model.temperature;
    return out.dump();
  }
  double T = cypha::adapt_temperature_ece(model, h_batch.data(), static_cast<int>(true_idx.size()),
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
    } else if (a == "--preload-registry") {
      g_preload_registry = true;
    } else if (a == "--train-hparams" && i + 1 < argc) {
      train_hparams_path = argv[++i];
    } else if (a == "--regression-json" && i + 1 < argc) {
      regression_json_path = argv[++i];
    } else if (a == "--sequence-checkpoint" && i + 1 < argc) {
      cyphalm_checkpoint_path = argv[++i];
    } else if (a == "--cyphalm-checkpoint" && i + 1 < argc) {
      const char* alias_path = argv[++i];
      if (cyphalm_checkpoint_path.empty()) {
        cyphalm_checkpoint_path = alias_path;
      }
    } else if (a == "--branch-a-json" && i + 1 < argc) {
      branch_a_json_path = argv[++i];
    }
  }
  if (cyphalm_checkpoint_path.empty()) {
    if (const char* env_ckpt = std::getenv("CYPHA_SEQUENCE_CHECKPOINT")) {
      cyphalm_checkpoint_path = env_ckpt;
    } else if (const char* env_ckpt = std::getenv("CYPHA_LM_CHECKPOINT")) {
      cyphalm_checkpoint_path = env_ckpt;
    } else if (const char* env_ckpt = std::getenv("CYPHALM_CHECKPOINT")) {
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
                 "[--regression-json regression_head.json] [--sequence-checkpoint ckpt_base] "
                 "[--branch-a-json branch_a_router.json] [--registry models_root] [--preload-registry] "
                 "(POST /register needs --registry)\n";
    return 2;
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!load_bundle_paths(cypha_path, pre_path, ff_json, train_hparams_path, regression_json_path)) {
      return 1;
    }
    if (!g_registry_root.empty()) {
      const fs::path cypha_abs = fs::absolute(fs::path(cypha_path));
      const fs::path reg_abs = fs::absolute(fs::path(g_registry_root));
      for (const auto& r : g_registry_cache) {
        try {
          if (fs::equivalent(fs::path(r.model_path), cypha_abs)) {
            g_active_model_key = model_registry_key(r.name, r.version);
            load_registry_ref_into_map(r, false, false);
            break;
          }
        } catch (...) {
        }
      }
      if (g_active_model_key.empty()) {
        fs::path rel = cypha_abs.lexically_relative(reg_abs);
        const fs::path parts = rel.parent_path();
        if (parts.has_parent_path()) {
          const std::string name = parts.parent_path().filename().string();
          const std::string version = parts.filename().string();
          if (const cypha::RegistryModelRef* ref = find_registry_ref(name, version)) {
            g_active_model_key = model_registry_key(name, version);
            load_registry_ref_into_map(*ref, false, false);
          }
        }
      }
    }
    if (g_preload_registry) {
      preload_registry_models();
    }
  }

  httplib::Server svr;

  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    nlohmann::json j;
    j["status"] = "ok";
    j["model"] = g_cypha->loaded() ? "Cypha" : "none";
    const bool sequence_loaded = g_cypha->sequence_loaded();
    j["sequence_loaded"] = sequence_loaded;
    j["lm_loaded"] = sequence_loaded;  // alias
    j["uptime"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_started).count();
    j["n_predictions"] = g_predictions;
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/ready", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_cypha->loaded()) {
      res.status = 503;
      res.set_content(R"({"ready":false,"reason":"no_model_loaded"})", "application/json");
      return;
    }
    nlohmann::json j;
    j["ready"] = true;
    j["model_type"] = "Cypha";
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_mu);
    double uptime =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g_started).count();
    nlohmann::json payload;
    payload["uptime_seconds"] = std::round(uptime * 1000.0) / 1000.0;
    payload["model_loaded"] = g_cypha->loaded();
    payload["model_type"] = g_cypha->loaded() ? "Cypha" : nullptr;
    payload["n_predictions"] = g_predictions;
    payload["n_corrections"] = g_engine_corrections;
    payload["registry_model_count"] = static_cast<int>(g_registry_cache.size());
    payload["loaded_model_count"] = static_cast<int>(g_models.size());
    if (g_active_model_key.empty()) {
      payload["active_model"] = nullptr;
    } else {
      payload["active_model"] = g_active_model_key;
    }
    if (g_cypha->loaded()) {
      payload["gh_chi_session"] = g_cypha->gh_chi();
      payload["gh_psi_session"] = g_cypha->gh_psi();
      payload["session"] = session_summary_json();
      payload["regression_head_loaded"] = (!g_cypha->reg_mu().empty() || g_cypha->mke_active());
    } else {
      payload["session"] = nullptr;
      payload["regression_head_loaded"] = false;
    }
    const bool sequence_loaded = g_cypha->sequence_loaded();
    payload["sequence_loaded"] = sequence_loaded;
    payload["lm_loaded"] = sequence_loaded;  // alias
    payload["branch_a_router"] = cypha::branch_a_rest_summary_json();
    res.set_content(payload.dump(), "application/json");
  });

  svr.Post("/predict", [](const httplib::Request& req, httplib::Response& res) {
    try {
      auto body = nlohmann::json::parse(req.body);
      std::string out = json_predict(body);
      if (out.find("No model loaded") != std::string::npos) {
        res.status = 503;
      } else if (out.find("model not loaded") != std::string::npos) {
        res.status = 404;
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
      } else if (out.find("model not loaded") != std::string::npos) {
        res.status = 404;
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
      if (g_active_model_key.empty()) {
        j["active_model"] = nullptr;
      } else {
        j["active_model"] = g_active_model_key;
      }
      res.set_content(j.dump(), "application/json");
      return;
    }
    auto annotate = [](nlohmann::json row, const std::string& name, const std::string& version) {
      const std::string key = model_registry_key(name, version);
      row["loaded"] = g_models.find(key) != g_models.end();
      row["active"] = key == g_active_model_key;
      return row;
    };
    if (summary) {
      for (const auto& r : g_registry_cache) {
        nlohmann::json row;
        row["name"] = r.name;
        row["version"] = r.version;
        arr.push_back(annotate(std::move(row), r.name, r.version));
      }
    } else {
      for (const auto& r : g_registry_cache) {
        try {
          std::ifstream f(r.card_path);
          std::stringstream b;
          b << f.rdbuf();
          nlohmann::json row = nlohmann::json::parse(b.str());
          arr.push_back(annotate(std::move(row), r.name, r.version));
        } catch (...) {
          nlohmann::json row;
          row["name"] = r.name;
          row["version"] = r.version;
          row["error"] = "card_parse_failed";
          arr.push_back(annotate(std::move(row), r.name, r.version));
        }
      }
    }
    nlohmann::json j;
    j["models"] = arr;
    if (g_active_model_key.empty()) {
      j["active_model"] = nullptr;
    } else {
      j["active_model"] = g_active_model_key;
    }
    res.set_content(j.dump(), "application/json");
  });

  svr.Post("/load", [](const httplib::Request& req, httplib::Response& res) {
    if (g_registry_root.empty() || g_registry_cache.empty()) {
      res.status = 503;
      res.set_content(R"({"detail":"No registry configured"})", "application/json");
      return;
    }
    try {
      nlohmann::json body =
          req.body.empty() ? nlohmann::json::object() : nlohmann::json::parse(req.body);

      if (body.empty() || (!body.contains("name") && !body.contains("model"))) {
        std::lock_guard<std::mutex> lk(g_mu);
        nlohmann::json keys = nlohmann::json::array();
        for (const auto& ref : g_registry_cache) {
          if (load_registry_ref_into_map(ref, false, false)) {
            keys.push_back(model_registry_key(ref.name, ref.version));
          }
        }
        nlohmann::json wrap;
        wrap["loaded"] = std::move(keys);
        res.set_content(wrap.dump(), "application/json");
        return;
      }

      std::string name;
      std::string version;
      if (body.contains("model") && body["model"].is_string()) {
        if (!parse_model_registry_key(body["model"].get<std::string>(), name, version)) {
          res.status = 400;
          res.set_content(R"({"detail":"model must be name/version"})", "application/json");
          return;
        }
      } else {
        name = body.at("name").get<std::string>();
        version = body.value("version", "latest");
      }

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
      std::lock_guard<std::mutex> lk(g_mu);
      if (!load_registry_ref_into_map(*ref, true, true)) {
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
      wrap["model"] = model_registry_key(ref->name, ref->version);
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
    g_cypha->gh_chi() = 1.0;
    g_cypha->gh_psi() = 1.0;
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
    oss << g_cypha->rng();
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
      g_cypha->rng() = std::mt19937{static_cast<uint32_t>(body["seed"].get<long long>())};
    } else if (body.contains("state") && body["state"].is_array()
               && body["state"].size() == 624) {
      // Full state restore: reconstruct the serialised text representation and feed to operator>>.
      std::ostringstream oss;
      for (const auto& v : body["state"]) {
        oss << v.get<unsigned long>() << ' ';
      }
      oss << body.value("pos", 0);
      std::istringstream iss(oss.str());
      iss >> g_cypha->rng();
    } else {
      res.status = 400;
      res.set_content(R"({"detail":"provide seed (int) or state (array of 624 uint32) + pos"})",
                      "application/json");
      return;
    }
    // Return new state
    std::ostringstream oss2;
    oss2 << g_cypha->rng();
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
    if (!g_cypha->loaded() || !g_cypha->infer()) {
      res.status = 503;
      res.set_content(R"({"detail":"No model loaded"})", "application/json");
      return;
    }
    nlohmann::json classes = nlohmann::json::object();
    const auto& model = *g_cypha->infer();
    const int K = static_cast<int>(model.labels.size());
    for (int k = 0; k < K; ++k) {
      nlohmann::json row;
      row["n_obs"] = model.n_obs[static_cast<std::size_t>(k)];
      classes[model.labels[static_cast<std::size_t>(k)]] = row;
    }
    nlohmann::json out;
    out["classes"] = classes;
    res.set_content(out.dump(), "application/json");
  });

  svr.Get("/uncertainty-rank", [](const httplib::Request& req, httplib::Response& res) {
    try {
      nlohmann::json body;
      if (req.has_param("payload")) {
        body = nlohmann::json::parse(req.get_param_value("payload"));
      } else if (!req.body.empty()) {
        body = nlohmann::json::parse(req.body);
      } else {
        res.status = 400;
        res.set_content(
            std::string(R"msg({"detail":"JSON body or ?payload=<urlencoded-json> required (GET bodies are not read by the HTTP stack)"})msg"),
            "application/json");
        return;
      }
      const UncertaintyRankResult out = json_uncertainty_rank(body);
      res.status = out.status;
      res.set_content(out.body, "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
    }
  });

  svr.Post("/uncertainty-rank", [](const httplib::Request& req, httplib::Response& res) {
    try {
      if (req.body.empty()) {
        res.status = 400;
        res.set_content(R"({"detail":"JSON body required"})", "application/json");
        return;
      }
      auto body = nlohmann::json::parse(req.body);
      const UncertaintyRankResult out = json_uncertainty_rank(body);
      res.status = out.status;
      res.set_content(out.body, "application/json");
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"detail":"bad json"})", "application/json");
    }
  });

  cypha::dif_rest_configure(&g_mu, g_cypha.get());
  cypha::register_dif_rest_routes(svr);  // registers /sample + /retrieve (former /dif/*)
  cypha::cyphalm::cyphalm_rest_configure(&g_mu, g_cypha.get());
  cypha::cyphalm::register_cyphalm_rest_routes(svr);
  cypha::branch_a_rest_configure(branch_a_json_path);
  cypha::register_branch_a_rest_routes(svr);
  cypha::intelligence_rest_configure(&g_mu, &g_intelligence_profiler, &g_causal_graph_monitor);
  cypha::register_intelligence_rest_routes(svr);

  if (!cyphalm_checkpoint_path.empty()) {
    try {
      cypha::cyphalm::cyphalm_rest_lm_load(cyphalm_checkpoint_path);
      std::cout << "Cypha sequence loaded from " << cyphalm_checkpoint_path << "\n";
    } catch (const std::exception& ex) {
      std::cerr << "Cypha sequence checkpoint load failed: " << ex.what() << "\n";
      return 1;
    }
  }

  cypha_rest_ui::configure_static_ui(svr, (argc > 0) ? argv[0] : nullptr);

  std::cout << "cypha_rest listening on http://" << listen << ":" << port << "\n";
  if (!svr.listen(listen.c_str(), port)) {
    std::cerr << "bind failed\n";
    return 1;
  }
  return 0;
}
