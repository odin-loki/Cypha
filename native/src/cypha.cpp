#include "cypha/cypha.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "cypha/create_model.hpp"
#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/intelligence/self_correcting_infer.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/mke_scalar_train_step.hpp"
#include "cypha/regression.hpp"
#include "cypha/sync_infer.hpp"

namespace fs = std::filesystem;

namespace cypha {
namespace {

constexpr double kSoftmaxEps = 1e-8;

bool cypha_has_embedded_world_f_field(const CNode& root, int d, int fd) {
  const CNode& world = map_get_required(root, "world");
  const CNode* wff = map_get(world, "F_field");
  const int expected = d * fd;
  return wff != nullptr && wff->kind == CNode::Tensor && wff->shape.size() == 2 &&
         static_cast<int>(wff->shape[0]) == d && static_cast<int>(wff->shape[1]) == fd &&
         static_cast<int>(wff->tensor.size()) == expected;
}

// ---- Save format 2 -------------------------------------------------------------------------------------------
// Format 1 (no ``cypha_save_version`` key) kept only the world, classes, encoder, field_h, temperature and a few
// session scalars, and ``load`` ignored most of those scalars. Format 2 also writes the online state ``predict`` and
// ``update`` read (tiered label context, OOD EMAs, LLR scale, field dynamics, GH gate session, hyperparameters,
// replay buffer, RNG, self-correction threshold, EWC, regression head). Keys the older readers know are written
// at the top level with their usual meaning; the rest go in the ``cypha_save_state`` map.
constexpr std::int64_t kCyphaSaveVersion = 2;
constexpr const char* kSaveVersionKey = "cypha_save_version";
constexpr const char* kSaveStateKey = "cypha_save_state";

CNode cn_f64(double v) {
  CNode n;
  n.kind = CNode::Float;
  n.f = v;
  return n;
}

CNode cn_i64(std::int64_t v) {
  CNode n;
  n.kind = CNode::Int;
  n.i = v;
  return n;
}

CNode cn_bool(bool v) {
  CNode n;
  n.kind = CNode::Bool;
  n.b = v;
  return n;
}

CNode cn_str(const std::string& v) {
  CNode n;
  n.kind = CNode::Str;
  n.s = v;
  return n;
}

CNode cn_map() {
  CNode n;
  n.kind = CNode::Map;
  return n;
}

CNode cn_tensor(const std::vector<double>& v, std::vector<std::uint32_t> shape) {
  CNode n;
  n.kind = CNode::Tensor;
  n.shape = std::move(shape);
  n.tensor = v;
  return n;
}

CNode cn_vec(const std::vector<double>& v) {
  return cn_tensor(v, {static_cast<std::uint32_t>(v.size())});
}

/// Replace ``key`` in place, or append it.
void cn_put(CNode& m, const std::string& key, CNode v) {
  for (auto& kv : m.map) {
    if (kv.first == key) {
      kv.second = std::move(v);
      return;
    }
  }
  m.map.emplace_back(key, std::move(v));
}

/// Empty vectors are left out (a zero-length tensor carries nothing and a missing key reads back as empty).
void cn_put_vec(CNode& m, const std::string& key, const std::vector<double>& v) {
  if (!v.empty()) {
    cn_put(m, key, cn_vec(v));
  }
}

double cn_num(const CNode& n, const char* what) {
  if (n.kind == CNode::Float) {
    return n.f;
  }
  if (n.kind == CNode::Int) {
    return static_cast<double>(n.i);
  }
  throw std::runtime_error(std::string("Cypha::load: expected a number for ") + what);
}

std::int64_t cn_int(const CNode& n, const char* what) {
  if (n.kind == CNode::Int) {
    return n.i;
  }
  if (n.kind == CNode::Float) {
    return static_cast<std::int64_t>(n.f);
  }
  throw std::runtime_error(std::string("Cypha::load: expected an integer for ") + what);
}

double get_num(const CNode& m, const char* key) { return cn_num(map_get_required(m, key), key); }

std::int64_t get_int(const CNode& m, const char* key) { return cn_int(map_get_required(m, key), key); }

bool get_bool(const CNode& m, const char* key) {
  const CNode& n = map_get_required(m, key);
  if (n.kind != CNode::Bool) {
    throw std::runtime_error(std::string("Cypha::load: expected a bool for ") + key);
  }
  return n.b;
}

std::string get_str(const CNode& m, const char* key) {
  const CNode& n = map_get_required(m, key);
  if (n.kind != CNode::Str) {
    throw std::runtime_error(std::string("Cypha::load: expected a string for ") + key);
  }
  return n.s;
}

/// Missing key → empty vector (see ``cn_put_vec``).
std::vector<double> get_vec(const CNode& m, const char* key) {
  const CNode* n = map_get(m, key);
  if (n == nullptr) {
    return {};
  }
  if (n->kind != CNode::Tensor) {
    throw std::runtime_error(std::string("Cypha::load: expected a tensor for ") + key);
  }
  return n->tensor;
}

const CNode& get_map(const CNode& m, const char* key) {
  const CNode& n = map_get_required(m, key);
  if (n.kind != CNode::Map) {
    throw std::runtime_error(std::string("Cypha::load: expected a map for ") + key);
  }
  return n;
}

/// Sorted keys so the bytes written do not depend on hash-map iteration order.
CNode cn_dict(const std::unordered_map<std::string, double>& m) {
  std::map<std::string, double> sorted(m.begin(), m.end());
  CNode n = cn_map();
  for (const auto& kv : sorted) {
    n.map.emplace_back(kv.first, cn_f64(kv.second));
  }
  return n;
}

CNode cn_dict2(const std::unordered_map<std::string, std::unordered_map<std::string, double>>& m) {
  std::map<std::string, const std::unordered_map<std::string, double>*> sorted;
  for (const auto& kv : m) {
    sorted[kv.first] = &kv.second;
  }
  CNode n = cn_map();
  for (const auto& kv : sorted) {
    n.map.emplace_back(kv.first, cn_dict(*kv.second));
  }
  return n;
}

std::unordered_map<std::string, double> read_dict(const CNode& n, const char* what) {
  std::unordered_map<std::string, double> out;
  for (const auto& kv : n.map) {
    out[kv.first] = cn_num(kv.second, what);
  }
  return out;
}

CNode cn_vec_dict(const std::unordered_map<std::string, std::vector<double>>& m) {
  std::map<std::string, const std::vector<double>*> sorted;
  for (const auto& kv : m) {
    sorted[kv.first] = &kv.second;
  }
  CNode n = cn_map();
  for (const auto& kv : sorted) {
    n.map.emplace_back(kv.first, cn_vec(*kv.second));
  }
  return n;
}

std::unordered_map<std::string, std::vector<double>> read_vec_dict(const CNode& n, const char* what) {
  std::unordered_map<std::string, std::vector<double>> out;
  for (const auto& kv : n.map) {
    if (kv.second.kind != CNode::Tensor) {
      throw std::runtime_error(std::string("Cypha::load: expected tensors in ") + what);
    }
    out[kv.first] = kv.second.tensor;
  }
  return out;
}

/// ``std::mt19937`` text form (the standard fixes it), written with the classic locale.
std::string rng_to_string(const std::mt19937& r) {
  std::ostringstream o;
  o.imbue(std::locale::classic());
  o << r;
  return o.str();
}

void rng_from_string(const std::string& s, std::mt19937& r) {
  std::istringstream i(s);
  i.imbue(std::locale::classic());
  std::mt19937 t;
  i >> t;
  if (i.fail()) {
    throw std::runtime_error("Cypha::load: bad RNG state in cypha_save_state");
  }
  r = t;
}

void snapshot_gh_clean(CyphaInferModel& model, std::vector<double>& gh_inv_v_clean, double& gh_R_base) {
  const int d = model.d_latent;
  gh_inv_v_clean.assign(static_cast<std::size_t>(d), 1.0);
  if (static_cast<int>(model.inv_v.size()) == d) {
    gh_inv_v_clean = model.inv_v;
  }
  gh_R_base = (model.has_mahal_ema && model.mahal_ema > 0.0) ? model.mahal_ema : 1.0;
}

}  // namespace

Cypha::Cypha() = default;
Cypha::~Cypha() = default;
Cypha::Cypha(Cypha&&) noexcept = default;
Cypha& Cypha::operator=(Cypha&&) noexcept = default;

void Cypha::clear_mke() {
  mke_active_ = false;
  mke_d_in_ = 0;
  mke_W_.clear();
  mke_b_.clear();
  mke_temperature_ = 1.0;
  mke_forgetting_ = 1.0;
  mke_pi_floor_ = 0.02;
  mke_gh_scales_.clear();
  mke_w_.clear();
  mke_p_.clear();
}

void Cypha::apply_default_hparams() {
  world_lr_ = 0.008;
  delta_lr_ = 0.05;
  ood_sigma_ = 15.0;
  tsp_.enc_lr = 0.002;
  tsp_.replay_ratio = 0.30;
  tsp_.replay_cap = 10000;
  tsp_.align_every = 500;
  tsp_.temp_recalib_every = 0;
}

void Cypha::reset_ood_threshold() {
  ood_target_ = 0.05;
  ood_h_.clear();
  ood_count_ = 0;
  ood_head_ = 0;
  ood_tau_ = kOodThreshold;
  ood_dirty_ = false;
}

void Cypha::set_ood_target_rate(double rate) {
  if (!(rate > 0.0 && rate < 1.0)) {
    throw std::invalid_argument("Cypha::set_ood_target_rate: the rate must be in (0, 1)");
  }
  ood_target_ = rate;
  ood_dirty_ = true;
}

void Cypha::record_ood_latent(const std::vector<double>& h) {
  const std::size_t d = h.size();
  for (double v : h) {
    if (!std::isfinite(v)) {
      return;
    }
  }
  if (ood_count_ < kOodWindow) {
    ood_h_.insert(ood_h_.end(), h.begin(), h.end());
    ++ood_count_;
  } else {
    std::copy(h.begin(), h.end(), ood_h_.begin() + static_cast<std::ptrdiff_t>(ood_head_ * d));
    ood_head_ = (ood_head_ + 1) % kOodWindow;
  }
  ood_dirty_ = true;
}

double Cypha::ood_threshold() const {
  if (ood_count_ < kOodMinScores || !infer_) {
    return kOodThreshold;
  }
  if (ood_dirty_ || !(ood_tau_chi_ == gh_chi_) || !(ood_tau_psi_ == gh_psi_)) {
    // Score every held latent with the model as it is now, then take the (1 - target) empirical quantile:
    // s[ceil((1 - target) n) - 1] of the sorted scores, so at most a share ``target`` of them lies above it.
    const std::size_t d = static_cast<std::size_t>(infer_->d_latent);
    std::vector<double> v(ood_count_);
    for (std::size_t r = 0; r < ood_count_; ++r) {
      v[r] = gh_anomaly_score_at_h(*infer_, ood_h_.data() + r * d, gh_chi_, gh_psi_);
    }
    const double pos = std::ceil((1.0 - ood_target_) * static_cast<double>(ood_count_));
    const std::size_t k = std::min(ood_count_ - 1, static_cast<std::size_t>(std::max(pos, 1.0)) - 1);
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    ood_tau_ = v[k];
    ood_tau_chi_ = gh_chi_;
    ood_tau_psi_ = gh_psi_;
    ood_dirty_ = false;
  }
  return ood_tau_;
}

bool Cypha::load_ff_json(const std::string& path, int d, int fd, std::vector<double>& out) const {
  if (path.empty()) {
    return false;
  }
  std::ifstream f(path);
  if (!f) {
    return false;
  }
  std::stringstream b;
  b << f.rdbuf();
  nlohmann::json j = nlohmann::json::parse(b.str());
  if (!j.is_array()) {
    return false;
  }
  out = j.get<std::vector<double>>();
  return static_cast<int>(out.size()) == d * fd;
}

bool Cypha::load_regression_json(const std::string& path) {
  reg_mu_.clear();
  reg_var_.clear();
  clear_mke();
  if (path.empty() || !infer_) {
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
  const int k = static_cast<int>(infer_->labels.size());
  reg_mu_.assign(static_cast<std::size_t>(k), 0.0);
  reg_var_.assign(static_cast<std::size_t>(k), 0.0);
  for (int i = 0; i < k; ++i) {
    const std::string& lbl = infer_->labels[static_cast<std::size_t>(i)];
    if (!ex.contains(lbl)) {
      continue;
    }
    const auto& row = ex[lbl];
    if (row.contains("mu")) {
      if (row["mu"].is_number()) {
        reg_mu_[static_cast<std::size_t>(i)] = row["mu"].get<double>();
      } else if (row["mu"].is_array() && !row["mu"].empty()) {
        reg_mu_[static_cast<std::size_t>(i)] = row["mu"][0].get<double>();
      }
    }
    if (row.contains("var_ema") && row["var_ema"].is_number()) {
      reg_var_[static_cast<std::size_t>(i)] = row["var_ema"].get<double>();
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
    mke_d_in_ = mk.at("d_in").get<int>();
    const int d_rff = mk.at("D_rff").get<int>();
    if (d_rff != infer_->d_latent) {
      return false;
    }
    mke_W_ = mk.at("rff_W_rowmajor").get<std::vector<double>>();
    mke_b_ = mk.at("rff_b").get<std::vector<double>>();
    if (static_cast<int>(mke_W_.size()) != d_rff * mke_d_in_ ||
        static_cast<int>(mke_b_.size()) != d_rff) {
      return false;
    }
    mke_temperature_ = mk.value("temperature", 1.0);
    mke_forgetting_ = mk.value("forgetting_factor", 1.0);
    mke_pi_floor_ = mk.value("pi_floor", 0.02);
    const auto& wj = mk.at("w");
    const auto& pj = mk.at("P");
    if (!wj.is_object() || !pj.is_object()) {
      return false;
    }
    const std::size_t p_expect = static_cast<std::size_t>(d_rff) * static_cast<std::size_t>(d_rff);
    mke_w_.clear();
    mke_p_.clear();
    for (int i = 0; i < k; ++i) {
      const std::string& lbl = infer_->labels[static_cast<std::size_t>(i)];
      if (!wj.contains(lbl) || !pj.contains(lbl)) {
        return false;
      }
      auto ww = wj[lbl].get<std::vector<double>>();
      auto pp = pj[lbl].get<std::vector<double>>();
      if (static_cast<int>(ww.size()) != d_rff || pp.size() != p_expect) {
        return false;
      }
      mke_w_[lbl] = std::move(ww);
      mke_p_[lbl] = std::move(pp);
    }
    mke_gh_scales_.clear();
    if (mk.contains("gh_scales") && mk["gh_scales"].is_array()) {
      mke_gh_scales_ = mk["gh_scales"].get<std::vector<double>>();
      if (static_cast<int>(mke_gh_scales_.size()) != k) {
        return false;
      }
    }
    mke_active_ = true;
  } catch (...) {
    clear_mke();
    return false;
  }
  return true;
}

CNode Cypha::export_save_state(const CyphaInferModel& m) const {
  CNode st = cn_map();

  // Online inference state that the format-1 keys cannot hold exactly (from_root recomputes these totals
  // as plain sums, which differ in the last bits from the running values training keeps).
  CNode inf = cn_map();
  inf.map.emplace_back("mid_freq_total", cn_f64(m.mid_freq_total));
  inf.map.emplace_back("mid_trans_tot", cn_dict(m.mid_trans_tot));
  inf.map.emplace_back("t1_total", cn_f64(m.t1_total));
  inf.map.emplace_back("t1_counts", cn_dict(m.t1_counts));
  st.map.emplace_back("infer", std::move(inf));

  // Training-memory scalars from_cypha_root resets (Welford M2 for the first 20 rows, cached log-norm).
  CNode mem = cn_map();
  cn_put_vec(mem, "world_M2", mem_->world_M2);
  mem.map.emplace_back("world_log_norm", cn_f64(mem_->world_log_norm));
  mem.map.emplace_back("world_log_n_ctr", cn_i64(mem_->world_log_n_ctr));
  mem.map.emplace_back("world_v_mean", cn_f64(mem_->world_v_mean));
  mem.map.emplace_back("task_prefix_protect", cn_str(mem_->task_prefix_protect));
  st.map.emplace_back("memory", std::move(mem));

  // Session: hyperparameters, counters, RNG and the learned self-correction threshold.
  CNode ses = cn_map();
  ses.map.emplace_back("world_lr", cn_f64(world_lr_));
  ses.map.emplace_back("delta_lr", cn_f64(delta_lr_));
  ses.map.emplace_back("enc_lr", cn_f64(tsp_.enc_lr));
  ses.map.emplace_back("replay_ratio", cn_f64(tsp_.replay_ratio));
  ses.map.emplace_back("replay_cap", cn_i64(tsp_.replay_cap));
  ses.map.emplace_back("align_every", cn_i64(tsp_.align_every));
  ses.map.emplace_back("temp_recalib_every", cn_i64(tsp_.temp_recalib_every));
  ses.map.emplace_back("enc_updates", cn_i64(enc_updates_));
  ses.map.emplace_back("ewc_lambda", cn_f64(ewc_lambda_));
  ses.map.emplace_back("last_active_query_score", cn_f64(last_active_query_score_));
  ses.map.emplace_back("rng", cn_str(rng_to_string(rng_)));
  ses.map.emplace_back("ood_target", cn_f64(ood_target_));
  ses.map.emplace_back("ood_count", cn_i64(static_cast<std::int64_t>(ood_count_)));
  ses.map.emplace_back("ood_head", cn_i64(static_cast<std::int64_t>(ood_head_)));
  if (ood_count_ > 0) {
    ses.map.emplace_back("ood_latents", cn_tensor(ood_h_, {static_cast<std::uint32_t>(ood_count_),
                                                           static_cast<std::uint32_t>(m.d_latent)}));
  }
  {
    const intelligence::NigStatisticState::Raw r = epistemic_threshold_.state().raw();
    CNode ep = cn_map();
    ep.map.emplace_back("mu", cn_f64(r.mu));
    ep.map.emplace_back("lambda", cn_f64(r.lambda));
    ep.map.emplace_back("alpha", cn_f64(r.alpha));
    ep.map.emplace_back("beta", cn_f64(r.beta));
    ep.map.emplace_back("running_mean", cn_f64(r.running_mean));
    ep.map.emplace_back("running_m2", cn_f64(r.running_m2));
    ep.map.emplace_back("n_updates", cn_i64(r.n_updates));
    ep.map.emplace_back("n_observations", cn_i64(r.n_observations));
    ses.map.emplace_back("epistemic_threshold", std::move(ep));
  }
  st.map.emplace_back("session", std::move(ses));

  if (replay_) {
    const ReplayBuffer::State r = replay_->export_state();
    CNode rp = cn_map();
    rp.map.emplace_back("cap", cn_i64(r.cap));
    rp.map.emplace_back("len", cn_i64(r.len));
    rp.map.emplace_back("insert_n", cn_i64(r.insert_n));
    rp.map.emplace_back("d", cn_i64(r.d));
    if (r.len > 0 && r.d > 0) {
      rp.map.emplace_back("h", cn_tensor(r.h, {static_cast<std::uint32_t>(r.len), static_cast<std::uint32_t>(r.d)}));
    }
    cn_put_vec(rp, "w", r.w);
    // Labels as a small table plus one index per row.
    std::map<std::string, int> index;
    CNode table = cn_map();
    std::vector<double> label_idx;
    label_idx.reserve(r.labels.size());
    for (const auto& lbl : r.labels) {
      auto it = index.find(lbl);
      if (it == index.end()) {
        const int k = static_cast<int>(index.size());
        it = index.emplace(lbl, k).first;
        table.map.emplace_back(std::to_string(k), cn_str(lbl));
      }
      label_idx.push_back(static_cast<double>(it->second));
    }
    rp.map.emplace_back("label_table", std::move(table));
    cn_put_vec(rp, "label_idx", label_idx);
    st.map.emplace_back("replay", std::move(rp));
  }

  if (ewc_) {
    const EwcRegularizer::State e = ewc_->export_state();
    CNode ew = cn_map();
    ew.map.emplace_back("protect_world_field", cn_bool(e.protect_world_field));
    cn_put_vec(ew, "anchor_D", e.anchor_D);
    cn_put_vec(ew, "anchor_enc_w", e.anchor_enc_w);
    cn_put_vec(ew, "anchor_world_mu", e.anchor_world_mu);
    cn_put_vec(ew, "fisher_D", e.fisher_D);
    cn_put_vec(ew, "fisher_enc_w", e.fisher_enc_w);
    cn_put_vec(ew, "fisher_world_mu", e.fisher_world_mu);
    st.map.emplace_back("ewc", std::move(ew));
  }

  if (kernel_mem_ && use_kernel_llr_) {
    // patch_kernel_into_root writes the basis and weights; these are the parts it leaves out.
    CNode km = cn_map();
    km.map.emplace_back("rng", cn_str(rng_to_string(kernel_mem_->rng_engine())));
    km.map.emplace_back("gamma_scale", cn_f64(kernel_mem_->gamma_scale()));
    km.map.emplace_back("landmark_sampling", cn_i64(static_cast<std::int64_t>(kernel_mem_->landmark_sampling())));
    st.map.emplace_back("kernel", std::move(km));
  }

  if (!reg_mu_.empty() || !reg_var_.empty() || mke_active_) {
    // The regression head learns in update() (mke w / P) but used to live only in regression_head.json.
    CNode rg = cn_map();
    cn_put_vec(rg, "reg_mu", reg_mu_);
    cn_put_vec(rg, "reg_var", reg_var_);
    rg.map.emplace_back("mke_active", cn_bool(mke_active_));
    if (mke_active_) {
      rg.map.emplace_back("mke_d_in", cn_i64(mke_d_in_));
      cn_put_vec(rg, "mke_W", mke_W_);
      cn_put_vec(rg, "mke_b", mke_b_);
      rg.map.emplace_back("mke_temperature", cn_f64(mke_temperature_));
      rg.map.emplace_back("mke_forgetting", cn_f64(mke_forgetting_));
      rg.map.emplace_back("mke_pi_floor", cn_f64(mke_pi_floor_));
      cn_put_vec(rg, "mke_gh_scales", mke_gh_scales_);
      rg.map.emplace_back("mke_w", cn_vec_dict(mke_w_));
      rg.map.emplace_back("mke_P", cn_vec_dict(mke_p_));
    }
    st.map.emplace_back("regression", std::move(rg));
  }
  return st;
}

void Cypha::import_save_hparams(const CNode& st) {
  const CNode& ses = get_map(st, "session");
  world_lr_ = get_num(ses, "world_lr");
  delta_lr_ = get_num(ses, "delta_lr");
  tsp_.enc_lr = get_num(ses, "enc_lr");
  tsp_.replay_ratio = get_num(ses, "replay_ratio");
  tsp_.replay_cap = static_cast<int>(get_int(ses, "replay_cap"));
  tsp_.align_every = static_cast<int>(get_int(ses, "align_every"));
  tsp_.temp_recalib_every = static_cast<int>(get_int(ses, "temp_recalib_every"));
}

void Cypha::import_save_state(const CNode& st) {
  const CNode& inf = get_map(st, "infer");
  infer_->mid_freq_total = get_num(inf, "mid_freq_total");
  infer_->mid_trans_tot = read_dict(get_map(inf, "mid_trans_tot"), "mid_trans_tot");
  infer_->t1_total = get_num(inf, "t1_total");
  infer_->t1_counts = read_dict(get_map(inf, "t1_counts"), "t1_counts");

  const CNode& mem = get_map(st, "memory");
  std::vector<double> m2 = get_vec(mem, "world_M2");
  if (static_cast<int>(m2.size()) == mem_->d_latent) {
    mem_->world_M2 = std::move(m2);
  }
  mem_->world_log_norm = get_num(mem, "world_log_norm");
  mem_->world_log_n_ctr = static_cast<int>(get_int(mem, "world_log_n_ctr"));
  mem_->world_v_mean = get_num(mem, "world_v_mean");
  mem_->task_prefix_protect = get_str(mem, "task_prefix_protect");

  const CNode& ses = get_map(st, "session");
  enc_updates_ = static_cast<int>(get_int(ses, "enc_updates"));
  ewc_lambda_ = get_num(ses, "ewc_lambda");
  last_active_query_score_ = get_num(ses, "last_active_query_score");
  rng_from_string(get_str(ses, "rng"), rng_);
  if (const CNode* ot = map_get(ses, "ood_target"); ot != nullptr && ot->kind != CNode::Nil) {
    const double target = cn_num(*ot, "ood_target");
    const std::int64_t count = get_int(ses, "ood_count");
    const std::int64_t head = get_int(ses, "ood_head");
    std::vector<double> lat = get_vec(ses, "ood_latents");
    const std::size_t d = static_cast<std::size_t>(infer_->d_latent);
    const std::int64_t window = static_cast<std::int64_t>(kOodWindow);
    if (!(target > 0.0 && target < 1.0) || count < 0 || count > window || head < 0 ||
        (count < window && head != 0) || (count == window && head >= window) ||
        lat.size() != static_cast<std::size_t>(count) * d) {
      throw std::runtime_error("Cypha::load: bad OOD threshold state in cypha_save_state");
    }
    ood_target_ = target;
    ood_count_ = static_cast<std::size_t>(count);
    ood_head_ = static_cast<std::size_t>(head);
    ood_h_ = std::move(lat);
    ood_dirty_ = true;
  }
  {
    const CNode& ep = get_map(ses, "epistemic_threshold");
    intelligence::NigStatisticState::Raw r;
    r.mu = get_num(ep, "mu");
    r.lambda = get_num(ep, "lambda");
    r.alpha = get_num(ep, "alpha");
    r.beta = get_num(ep, "beta");
    r.running_mean = get_num(ep, "running_mean");
    r.running_m2 = get_num(ep, "running_m2");
    r.n_updates = static_cast<int>(get_int(ep, "n_updates"));
    r.n_observations = static_cast<int>(get_int(ep, "n_observations"));
    epistemic_threshold_.state().set_raw(r);
  }

  if (const CNode* rp = map_get(st, "replay"); rp != nullptr && rp->kind == CNode::Map) {
    ReplayBuffer::State r;
    r.cap = static_cast<int>(get_int(*rp, "cap"));
    r.len = static_cast<int>(get_int(*rp, "len"));
    r.insert_n = static_cast<int>(get_int(*rp, "insert_n"));
    r.d = static_cast<int>(get_int(*rp, "d"));
    r.h = get_vec(*rp, "h");
    r.w = get_vec(*rp, "w");
    const CNode& table = get_map(*rp, "label_table");
    std::vector<std::string> names(table.map.size());
    for (const auto& kv : table.map) {
      const std::size_t k = static_cast<std::size_t>(std::stoul(kv.first));
      if (k >= names.size() || kv.second.kind != CNode::Str) {
        throw std::runtime_error("Cypha::load: bad replay label table");
      }
      names[k] = kv.second.s;
    }
    for (double v : get_vec(*rp, "label_idx")) {
      const std::size_t k = static_cast<std::size_t>(v);
      if (v < 0.0 || k >= names.size()) {
        throw std::runtime_error("Cypha::load: bad replay label index");
      }
      r.labels.push_back(names[k]);
    }
    auto buf = std::make_unique<ReplayBuffer>(r.cap);
    if (!buf->import_state(r)) {
      throw std::runtime_error("Cypha::load: replay buffer sizes do not agree");
    }
    replay_ = std::move(buf);
  }

  ewc_.reset();
  if (const CNode* ew = map_get(st, "ewc"); ew != nullptr && ew->kind == CNode::Map) {
    EwcRegularizer::State e;
    e.protect_world_field = get_bool(*ew, "protect_world_field");
    e.anchor_D = get_vec(*ew, "anchor_D");
    e.anchor_enc_w = get_vec(*ew, "anchor_enc_w");
    e.anchor_world_mu = get_vec(*ew, "anchor_world_mu");
    e.fisher_D = get_vec(*ew, "fisher_D");
    e.fisher_enc_w = get_vec(*ew, "fisher_enc_w");
    e.fisher_world_mu = get_vec(*ew, "fisher_world_mu");
    ewc_ = std::make_unique<EwcRegularizer>();
    ewc_->import_state(std::move(e));
  }

  if (const CNode* km = map_get(st, "kernel"); km != nullptr && km->kind == CNode::Map && kernel_mem_) {
    rng_from_string(get_str(*km, "rng"), kernel_mem_->rng_engine());
    kernel_mem_->set_gamma_scale(get_num(*km, "gamma_scale"));
    kernel_mem_->set_landmark_sampling(
        static_cast<KernelMemory::LandmarkSamplingKind>(get_int(*km, "landmark_sampling")));
  }
}

void Cypha::import_regression_state(const CNode& rg) {
  reg_mu_ = get_vec(rg, "reg_mu");
  reg_var_ = get_vec(rg, "reg_var");
  clear_mke();
  if (!get_bool(rg, "mke_active")) {
    return;
  }
  mke_d_in_ = static_cast<int>(get_int(rg, "mke_d_in"));
  mke_W_ = get_vec(rg, "mke_W");
  mke_b_ = get_vec(rg, "mke_b");
  mke_temperature_ = get_num(rg, "mke_temperature");
  mke_forgetting_ = get_num(rg, "mke_forgetting");
  mke_pi_floor_ = get_num(rg, "mke_pi_floor");
  mke_gh_scales_ = get_vec(rg, "mke_gh_scales");
  mke_w_ = read_vec_dict(get_map(rg, "mke_w"), "mke_w");
  mke_p_ = read_vec_dict(get_map(rg, "mke_P"), "mke_P");
  const int d_rff = infer_->d_latent;
  if (static_cast<int>(mke_W_.size()) != d_rff * mke_d_in_ || static_cast<int>(mke_b_.size()) != d_rff) {
    clear_mke();
    throw std::runtime_error("Cypha::load: regression head sizes do not agree");
  }
  mke_active_ = true;
}

bool Cypha::load(const std::string& cypha_path, const std::string& preprocessor_path,
                 const std::string& f_field_json_path, const std::string& regression_json_path,
                 const std::string& train_hparams_path) {
  apply_default_hparams();
  reset_ood_threshold();
  reg_mu_.clear();
  reg_var_.clear();
  clear_mke();

  CNode root = load_cypha_file(cypha_path.c_str());
  const CNode& fh = map_get_required(root, "field_h");
  int fd = static_cast<int>(fh.shape[0]);
  const CNode& enc = map_get_required(root, "enc_W");
  int d = static_cast<int>(enc.shape[0]);

  std::vector<double> fflat;
  const double* ff_ptr = nullptr;
  if (cypha_has_embedded_world_f_field(root, d, fd)) {
    ff_ptr = nullptr;
  } else if (load_ff_json(f_field_json_path, d, fd, fflat)) {
    ff_ptr = fflat.data();
  } else {
    return false;
  }

  infer_ = std::make_unique<CyphaInferModel>(CyphaInferModel::from_root(root, ff_ptr, fd));
  mem_ = std::make_unique<CyphaDifMemoryState>(CyphaDifMemoryState::from_cypha_root(root, ff_ptr, fd));
  root_ = std::move(root);

  // Format 2 files carry the rest of the online state; format 1 files (no version key) load as before.
  const CNode* ver_node = map_get(root_, kSaveVersionKey);
  const std::int64_t save_version =
      (ver_node != nullptr && ver_node->kind == CNode::Int) ? ver_node->i : std::int64_t{1};
  const CNode* save_state = nullptr;
  if (save_version >= 2) {
    save_state = map_get(root_, kSaveStateKey);
    if (save_state == nullptr || save_state->kind != CNode::Map) {
      throw std::runtime_error("Cypha::load: " + cypha_path + " says save format " + std::to_string(save_version) +
                               " but has no cypha_save_state map");
    }
  }

  kernel_mem_.reset();
  use_kernel_llr_ = false;
  kernel_blend_ = 0.5;
  KernelMemory km(d, 256, 0);
  bool use_k = false;
  double blend = 0.5;
  if (try_load_kernel_from_root(root_, km, use_k, blend)) {
    kernel_mem_ = std::make_unique<KernelMemory>(std::move(km));
    use_kernel_llr_ = use_k;
    kernel_blend_ = blend;
  }

  pre_.reset();
  if (!preprocessor_path.empty()) {
    pre_ = std::make_unique<PreprocessorState>(
        PreprocessorState::from_json_file(preprocessor_path.c_str()));
  }

  auto apply_hparams_file = [this](const std::string& path) {
    std::ifstream f(path);
    if (!f) {
      return;
    }
    std::stringstream b;
    b << f.rdbuf();
    nlohmann::json j = nlohmann::json::parse(b.str());
    if (j.contains("world_lr")) {
      world_lr_ = j["world_lr"].get<double>();
    }
    if (j.contains("delta_lr")) {
      delta_lr_ = j["delta_lr"].get<double>();
    }
    if (j.contains("ood_sigma")) {
      ood_sigma_ = j["ood_sigma"].get<double>();
    }
    if (j.contains("enc_lr")) {
      tsp_.enc_lr = j["enc_lr"].get<double>();
    }
    if (j.contains("replay_ratio")) {
      tsp_.replay_ratio = j["replay_ratio"].get<double>();
    }
    if (j.contains("replay_cap")) {
      tsp_.replay_cap = j["replay_cap"].get<int>();
    }
  };
  if (save_state != nullptr) {
    // Saved hyperparameters first; a train_hparams.json (explicit or next to the file) still overrides them.
    import_save_hparams(*save_state);
    if (const CNode* oods = map_get(root_, "ood_sigma"); oods != nullptr && oods->kind != CNode::Nil) {
      ood_sigma_ = cn_num(*oods, "ood_sigma");
    }
  }
  if (!train_hparams_path.empty()) {
    apply_hparams_file(train_hparams_path);
  } else {
    fs::path auto_hp = fs::path(cypha_path).parent_path() / "train_hparams.json";
    if (fs::exists(auto_hp)) {
      apply_hparams_file(auto_hp.string());
    }
  }

  snapshot_gh_clean(*infer_, gh_inv_v_clean_, gh_R_base_);
  gh_chi_ = 1.0;
  gh_psi_ = 1.0;
  enc_updates_ = 0;
  replay_ = std::make_unique<ReplayBuffer>(tsp_.replay_cap);
  total_steps_ = infer_->saved_total_steps;
  llr_ema_ = infer_->llr_ema;

  if (save_state != nullptr) {
    // GH gate session (format 1 wrote these keys too but load reset them).
    gh_chi_ = get_num(root_, "gh_chi_session");
    gh_psi_ = get_num(root_, "gh_psi_session");
    gh_R_base_ = get_num(root_, "gh_R_base");
    std::vector<double> inv = get_vec(root_, "gh_inv_v_clean");
    if (static_cast<int>(inv.size()) == d) {
      gh_inv_v_clean_ = std::move(inv);
    }
    // Field power-iteration vector (from_root resets it to 1/sqrt(fd)).
    if (const CNode* sr = map_get(root_, "field_sr_vec");
        sr != nullptr && sr->kind == CNode::Tensor && !infer_->field_w_t.empty() &&
        sr->tensor.size() == infer_->field_sr_vec.size()) {
      for (std::size_t i = 0; i < sr->tensor.size(); ++i) {
        infer_->field_sr_vec[i] = static_cast<float>(sr->tensor[i]);
      }
    }
    import_save_state(*save_state);
  } else if (!infer_->labels.empty()) {
    std::cerr << "note: " << cypha_path
              << " uses Cypha save format 1; GH gate session, replay buffer, RNG and running totals start fresh"
              << " (re-save to keep them)\n";
  }

  const CNode* saved_regression = save_state != nullptr ? map_get(*save_state, "regression") : nullptr;
  if (!regression_json_path.empty()) {
    if (!load_regression_json(regression_json_path)) {
      return false;
    }
  } else if (saved_regression != nullptr && saved_regression->kind == CNode::Map) {
    // The saved head wins over a regression_head.json next to the file, which may predate the saved training.
    import_regression_state(*saved_regression);
  } else {
    fs::path auto_reg = fs::path(cypha_path).parent_path() / "regression_head.json";
    if (fs::exists(auto_reg)) {
      (void)load_regression_json(auto_reg.string());
    }
  }
  return true;
}

bool Cypha::load_sequence(const std::string& json_path) {
  if (json_path.empty()) {
    return false;
  }
  seq_ = std::make_unique<cyphalm::CyphaLMModel>(cyphalm::load_cyphalm_model(json_path));
  return true;
}

bool Cypha::init_default_sequence(int vocab_size, int d_model) {
  // The hp production recipe sets vocab_size to 256 (hp models bytes). A smaller request used to be widened
  // silently, so a caller that asked for 64 symbols got a 256-symbol model that generated tokens >= 64.
  if (vocab_size != 256) {
    throw std::invalid_argument("Cypha::init_default_sequence: CyphaLM's hp model reads bytes, so vocab_size must be "
                                "256 (got " + std::to_string(vocab_size) + ")");
  }
  cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = vocab_size;
  cfg.d_embed = d_model;
  cfg.d_state = std::max(d_model, 64);
  cfg.lstm_hidden = std::max(d_model, 64);
  cfg.ngram_context = 3;
  cyphalm::apply_hybrid_production_recipe(cfg);
  seq_ = std::make_unique<cyphalm::CyphaLMModel>(cfg);
  return true;
}

void Cypha::save(const std::string& cypha_path) const {
  if (!infer_ || !mem_) {
    throw std::runtime_error("Cypha::save: no model loaded");
  }
  if (cypha_path.empty()) {
    throw std::runtime_error("Cypha::save: empty path");
  }
  // Keep infer snapshot aligned with online memory before writing world/classes.
  CyphaInferModel synced = *infer_;
  sync_infer_model_from_memory(synced, *mem_);

  CNode save_root;
  if (root_.kind == CNode::Map) {
    save_root = clone_cnode(root_);
  } else {
    FreshModelParams fmp;
    fmp.input_dim = synced.d_latent;
    fmp.field_dim = synced.field_dim;
    fmp.temperature = synced.temperature;
    save_root = create_fresh_model_root(fmp);
  }
  save_root = CyphaDifMemoryState::merge_state_into_root_for_save(save_root, *mem_);

  const auto d = static_cast<std::uint32_t>(synced.d_latent);
  const auto fd = static_cast<std::uint32_t>(synced.field_dim);
  cn_put(save_root, "enc_W", cn_tensor(synced.enc_w, {d, d}));
  cn_put(save_root, "field_h", cn_tensor(synced.field_h, {fd}));
  cn_put(save_root, "temperature", cn_f64(synced.temperature));
  cn_put(save_root, "llr_ema", cn_f64(llr_ema_));
  cn_put(save_root, "total_steps", cn_i64(static_cast<std::int64_t>(total_steps_)));

  // Online inference state under the keys from_root (and the Python loader) already read. Format 1 left
  // these at their values from the file it was loaded from (a fresh model: zeros).
  cn_put(save_root, "base_temp", cn_f64(synced.base_temp));
  cn_put(save_root, "mahal_ema", synced.has_mahal_ema ? cn_f64(synced.mahal_ema) : CNode{});
  cn_put(save_root, "mahal_std_ema", cn_f64(synced.mahal_std_ema));
  cn_put(save_root, "llr_scale_ema", cn_f64(synced.llr_scale_ema));
  cn_put(save_root, "llr_scale_n", cn_i64(synced.llr_scale_n));
  cn_put(save_root, "llr_scale_baseline", cn_f64(synced.llr_scale_baseline));
  cn_put(save_root, "total_correct", cn_i64(synced.total_correct));
  cn_put(save_root, "deliberation_lo", cn_f64(synced.deliberation_lo));
  cn_put(save_root, "deliberation_hi", cn_f64(synced.deliberation_hi));
  cn_put(save_root, "field_step", cn_i64(synced.field_step));
  if (!synced.field_w_t.empty() && synced.field_w_t.size() == static_cast<std::size_t>(fd) * fd) {
    cn_put(save_root, "field_W_T", cn_tensor(synced.field_w_t, {fd, fd}));
    if (synced.field_a_eff.size() == synced.field_w_t.size()) {
      std::vector<double> a_eff(synced.field_a_eff.begin(), synced.field_a_eff.end());
      cn_put(save_root, "field_a_eff", cn_tensor(a_eff, {fd, fd}));
    }
    if (synced.field_sr_vec.size() == static_cast<std::size_t>(fd)) {
      std::vector<double> sr(synced.field_sr_vec.begin(), synced.field_sr_vec.end());
      cn_put(save_root, "field_sr_vec", cn_tensor(sr, {fd}));
    }
  }

  // Tiered label context (Python TieredContextBuffer keys).
  cn_put(save_root, "mid_n", cn_f64(synced.mid_n));
  {
    CNode mf = cn_map();
    for (const auto& pr : synced.mid_freq) {
      mf.map.emplace_back(pr.first, cn_f64(pr.second));
    }
    cn_put(save_root, "mid_freq", std::move(mf));
  }
  cn_put(save_root, "mid_trans", cn_dict2(synced.mid_trans));
  {
    CNode hist = cn_map();
    for (std::size_t i = 0; i < synced.ctx_history.size(); ++i) {
      CNode e = cn_map();
      e.map.emplace_back("l", cn_str(synced.ctx_history[i].first));
      e.map.emplace_back("c", cn_bool(synced.ctx_history[i].second));
      hist.map.emplace_back(std::to_string(i), std::move(e));
    }
    cn_put(save_root, "ctx_hist_packed", std::move(hist));
  }
  cn_put(save_root, "ctx_cooccur", cn_dict2(synced.cooccur));
  cn_put(save_root, "ctx_cooccur_tot", cn_dict(synced.cooccur_tot));
  cn_put(save_root, "ctx_last_label", cn_str(synced.ctx_last_label));

  // Session scalars (format 1 wrote these too; format 2 also reads them back).
  cn_put(save_root, "ood_sigma", cn_f64(ood_sigma_));
  cn_put(save_root, "gh_chi_session", cn_f64(gh_chi_));
  cn_put(save_root, "gh_psi_session", cn_f64(gh_psi_));
  cn_put(save_root, "gh_R_base", cn_f64(gh_R_base_));
  if (static_cast<int>(gh_inv_v_clean_.size()) == synced.d_latent) {
    cn_put(save_root, "gh_inv_v_clean", cn_vec(gh_inv_v_clean_));
  }

  if (kernel_mem_) {
    patch_kernel_into_root(save_root, *kernel_mem_, use_kernel_llr_, kernel_blend_);
  }
  cn_put(save_root, kSaveVersionKey, cn_i64(kCyphaSaveVersion));
  cn_put(save_root, kSaveStateKey, export_save_state(synced));
  save_cypha_file(cypha_path.c_str(), save_root);
}

void Cypha::save_sequence(const std::string& base_path) const {
  if (!seq_) {
    throw std::runtime_error("Cypha::save_sequence: no sequence model");
  }
  cyphalm::save_cyphalm_model(*seq_, base_path);
}

PredictOut Cypha::predict(const double* x_in, int n, const PredictOpts& o) {
  PredictOut out;
  if (!infer_ || !mem_ || x_in == nullptr || n <= 0) {
    out.detail = "No model loaded";
    return out;
  }
  std::vector<double> x(x_in, x_in + n);
  if (pre_) {
    x = pre_->transform_one(x);
  }

  std::vector<double> H;
  if (mke_active_) {
    if (static_cast<int>(x.size()) != mke_d_in_) {
      out.detail = "input dim mismatch after preprocessor";
      return out;
    }
    H.resize(static_cast<std::size_t>(infer_->d_latent));
    regression::rff_encode_batch_rowmajor(x.data(), 1, mke_d_in_, mke_W_.data(), mke_b_.data(),
                                          infer_->d_latent, H.data());
  } else {
    if (static_cast<int>(x.size()) != infer_->d_latent) {
      out.detail = "input dim mismatch after preprocessor";
      return out;
    }
    batch_encode(*infer_, x.data(), 1, H);
  }

  const int k = static_cast<int>(infer_->labels.size());
  CyphaInferOptions iopt{};
  iopt.deliberation_lo = o.deliberation_lo;
  iopt.deliberation_hi = o.deliberation_hi;
  iopt.use_field = o.use_field;
  if (use_kernel_llr_ && kernel_mem_) {
    iopt.use_kernel_llr = true;
    iopt.kernel_mem = kernel_mem_.get();
    iopt.kernel_blend = kernel_blend_;
  }

  std::vector<double> llr_for_scores;
  if (mke_active_) {
    std::vector<double> llr;
    score_matrix_use_field(*infer_, H.data(), 1, llr);
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] = llr[static_cast<std::size_t>(j)] / (mke_temperature_ + kSoftmaxEps);
    }
    std::vector<double> probs;
    softmax_batch_reference(z.data(), 1, k, kSoftmaxEps, probs);
    int bi = 0;
    for (int j = 1; j < k; ++j) {
      if (probs[static_cast<std::size_t>(j)] > probs[static_cast<std::size_t>(bi)]) {
        bi = j;
      }
    }
    out.label = infer_->labels[static_cast<std::size_t>(bi)];
    out.confidence = probs[static_cast<std::size_t>(bi)];
    llr_for_scores = std::move(llr);
  } else if (o.use_gh) {
    // self_correct uses deliberation deepening on standard infer; GH path skips it.
    iopt.gh_chi = gh_chi_;
    iopt.gh_psi = gh_psi_;
    GhInferAtHResult gh = gh_infer_at_h(*infer_, H.data(), gh_chi_, gh_psi_, kGhNigAdaptAlpha, &iopt);
    out.label = gh.label;
    out.confidence = gh.confidence;
    out.r_eff = gh.r_eff;
    llr_for_scores = std::move(gh.llrs);
    // Pair r_eff with the baseline it was formed against, not with mahal_ema (different units).
    out.anomaly_score = gh_infer_anomaly_score(out.r_eff, gh.r_base);
    out.is_ood = out.anomaly_score > ood_threshold();
  } else if (o.self_correct) {
    const auto scr = intelligence::self_correcting_infer_at_h(*infer_, H.data(), iopt, epistemic_threshold_,
                                                            o.self_correct_max_passes);
    out.label = scr.infer.label;
    out.confidence = scr.infer.confidence;
    llr_for_scores = std::move(scr.infer.llrs);
    out.anomaly_score = 0.0;
    out.is_ood = false;
    out.self_corrected = scr.corrected;
    out.correction_passes = scr.correction_passes;
    out.r_eu_proxy = scr.r_eu_proxy;
  } else {
    InferAtHResult inf = infer_at_h(*infer_, H.data(), iopt);
    out.label = inf.label;
    out.confidence = inf.confidence;
    llr_for_scores = std::move(inf.llrs);
  }

  out.labels = infer_->labels;
  out.all_scores = llr_for_scores;

  if (k > 0) {
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] =
          llr_for_scores[static_cast<std::size_t>(j)] / (infer_->temperature + kSoftmaxEps);
    }
    std::vector<double> probs;
    softmax_batch_reference(z.data(), 1, k, kSoftmaxEps, probs);
    last_active_query_score_ = active_query_score_from_probs(probs.data(), k, kSoftmaxEps);
  } else {
    last_active_query_score_ = 0.0;
  }

  if (mke_active_) {
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] =
          llr_for_scores[static_cast<std::size_t>(j)] / (mke_temperature_ + kSoftmaxEps);
    }
    std::vector<double> probs;
    softmax_batch_reference(z.data(), 1, k, kSoftmaxEps, probs);
    double y_mix = 0.0;
    const int d_rff = infer_->d_latent;
    for (int j = 0; j < k; ++j) {
      const std::string& lbl = infer_->labels[static_cast<std::size_t>(j)];
      auto it = mke_w_.find(lbl);
      if (it == mke_w_.end() || static_cast<int>(it->second.size()) != d_rff) {
        continue;
      }
      double dp = 0.0;
      for (int t = 0; t < d_rff; ++t) {
        dp += it->second[static_cast<std::size_t>(t)] * H[static_cast<std::size_t>(t)];
      }
      y_mix += probs[static_cast<std::size_t>(j)] * dp;
    }
    out.y = y_mix;
    if (static_cast<int>(reg_var_.size()) == k) {
      double v_mix = 0.0;
      for (int j = 0; j < k; ++j) {
        v_mix += probs[static_cast<std::size_t>(j)] * reg_var_[static_cast<std::size_t>(j)];
      }
      out.uncertainty = std::sqrt(std::max(v_mix, 0.0));
    }
  } else if (static_cast<int>(reg_mu_.size()) == k && static_cast<int>(reg_var_.size()) == k) {
    std::vector<double> z(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      z[static_cast<std::size_t>(j)] =
          llr_for_scores[static_cast<std::size_t>(j)] / (infer_->temperature + kSoftmaxEps);
    }
    std::vector<double> probs;
    softmax_batch_reference(z.data(), 1, k, kSoftmaxEps, probs);
    double y_mix = 0.0;
    double u_mix = 0.0;
    regression::predict_mixture_scalar(probs.data(), reg_mu_.data(), reg_var_.data(),
                                       static_cast<std::size_t>(k), y_mix, u_mix);
    out.y = y_mix;
    out.uncertainty = u_mix;
  }
  return out;
}

UpdateOut Cypha::update(const double* x_in, int n, const std::string* label, const double* y,
                        const UpdateOpts& o) {
  UpdateOut out;
  if (!infer_ || !mem_ || x_in == nullptr || n <= 0) {
    out.detail = "No model loaded";
    return out;
  }
  std::vector<double> x(x_in, x_in + n);
  if (pre_) {
    x = pre_->transform_one(x);
  }
  const int d = infer_->d_latent;
  const bool has_y = y != nullptr;
  if (has_y && !mke_active_) {
    out.detail = "regression_y requires mke block in regression_head.json";
    return out;
  }
  if (!has_y && (label == nullptr || label->empty())) {
    out.detail = "correct_label required";
    return out;
  }

  if (has_y) {
    if (static_cast<int>(x.size()) != mke_d_in_) {
      out.detail = "input dim mismatch after preprocessor";
      return out;
    }
  } else if (static_cast<int>(x.size()) != d) {
    out.detail = "input dim mismatch after preprocessor";
    return out;
  }

  if (!replay_) {
    replay_ = std::make_unique<ReplayBuffer>(tsp_.replay_cap);
  }
  TrainStepExtras extras{};
  // Keep the row's latent, encoded before learning it (the training step writes it). The current model's scores on
  // the last kOodWindow such latents set the OOD threshold (docs/reports/CYPHA_OOD_THRESHOLD.md).
  std::vector<double> ood_latent;
  const bool keep_ood_latent = !has_y && !mke_active_ && !infer_->labels.empty();
  if (keep_ood_latent) {
    extras.latent_out = &ood_latent;
  }
  extras.total_steps = &total_steps_;
  extras.ood_sigma = &ood_sigma_;
  extras.llr_ema = &llr_ema_;
  if (use_kernel_llr_ && kernel_mem_) {
    extras.kernel_mem = kernel_mem_.get();
    extras.use_kernel_llr = true;
    extras.kernel_blend = kernel_blend_;
  }

  if (o.ewc_snapshot) {
    if (!ewc_) {
      ewc_ = std::make_unique<EwcRegularizer>();
    }
    ewc_->snapshot(*mem_, *infer_);
  }
  double step_ewc_lambda = ewc_lambda_;
  if (std::isfinite(o.ewc_lambda)) {
    step_ewc_lambda = o.ewc_lambda;
    ewc_lambda_ = o.ewc_lambda;
  }
  if (step_ewc_lambda > 0.0 && ewc_ && ewc_->has_snapshot()) {
    extras.ewc_lambda = step_ewc_lambda;
    extras.ewc = ewc_.get();
  }

  std::size_t replay_u01_pos = 0;
  if (o.replay_u01 != nullptr && o.replay_u01_len > 0) {
    extras.replay_u01 = o.replay_u01;
    extras.replay_u01_len = o.replay_u01_len;
    extras.replay_u01_pos = &replay_u01_pos;
  }

  if (has_y) {
    const double* gh_ptr = nullptr;
    if (o.use_gh && static_cast<int>(mke_gh_scales_.size()) == static_cast<int>(infer_->labels.size())) {
      gh_ptr = mke_gh_scales_.data();
    }
    regression::MkeScalarTrainStepOutputs step_out{};
    (void)regression::mke_scalar_train_step(
        *infer_, *mem_, *replay_, x.data(), mke_d_in_, *y, mke_W_.data(), mke_b_.data(), d, mke_w_,
        mke_p_, gh_ptr, mke_temperature_, mke_forgetting_, mke_pi_floor_, tsp_, world_lr_, delta_lr_,
        ood_sigma_, rng_, enc_updates_, &extras, o.router_train_label, kSoftmaxEps, &step_out);
    out.loss = step_out.router_loss;
  } else if (o.use_gh && static_cast<int>(gh_inv_v_clean_.size()) == d) {
    GhTrainStepResult gh = dif_gh_train_step_vector(
        *infer_, *mem_, *replay_, x.data(), d, *label, gh_inv_v_clean_, gh_R_base_, gh_chi_, gh_psi_,
        kGhNigAdaptAlpha, world_lr_, delta_lr_, ood_sigma_, tsp_, rng_, enc_updates_, nullptr, &extras);
    out.loss = gh.loss;
    gh_chi_ = gh.chi_new;
    gh_psi_ = gh.psi_new;
  } else {
    out.loss = dif_train_step_vector(*infer_, *mem_, *replay_, x.data(), d, *label, world_lr_, delta_lr_,
                                     world_lr_, delta_lr_, ood_sigma_, tsp_, rng_, enc_updates_, nullptr,
                                     &extras);
  }
  if (keep_ood_latent && static_cast<int>(ood_latent.size()) == d) {
    record_ood_latent(ood_latent);
  }
  return out;
}

std::string Cypha::mode_name(SampleMode m) {
  switch (m) {
    case SampleMode::Langevin:
      return "langevin";
    case SampleMode::FromObservation:
      return "from_observation";
    case SampleMode::RetrievalAugmented:
      return "retrieval_augmented";
    case SampleMode::ClassGaussian:
      return "class_gaussian";
    case SampleMode::Conditioned:
      return "conditioned";
  }
  return "unknown";
}

SampleOut Cypha::sample(const SampleOpts& o) {
  SampleOut out;
  out.mode = mode_name(o.mode);
  if (!infer_ || o.x == nullptr || o.x_dim <= 0) {
    out.detail = "No model loaded";
    return out;
  }
  std::vector<double> x(o.x, o.x + o.x_dim);
  if (pre_) {
    x = pre_->transform_one(x);
  }
  if (static_cast<int>(x.size()) != infer_->d_latent && o.mode != SampleMode::RetrievalAugmented) {
    // RAG uses raw input dim; others need latent-sized (or preprocessed) input matching d_latent
  }
  if (o.mode != SampleMode::RetrievalAugmented && static_cast<int>(x.size()) != infer_->d_latent) {
    out.detail = "input dim mismatch after preprocessor";
    return out;
  }

  std::vector<double> h_query;
  batch_encode(*infer_, x.data(), 1, h_query);

  std::string label = o.label;
  CyphaInferOptions opt = o.infer_opt;
  opt.use_field = true;

  auto infer_label = [&]() {
    InferAtHResult inf = infer_at_h(*infer_, h_query.data(), opt);
    return inf.label;
  };

  if (o.mode == SampleMode::Langevin) {
    if (label.empty()) {
      label = infer_label();
    }
    out.h = generate_langevin(*infer_, label, o.n_samples, o.n_steps, o.step_size, o.temperature, &rng_,
                              nullptr, nullptr);
  } else if (o.mode == SampleMode::FromObservation) {
    if (label.empty()) {
      label = infer_label();
    }
    out.h = generate_from_observation(*infer_, h_query.data(), label, o.n_samples, o.temperature, o.n_steps,
                                      &rng_, nullptr);
  } else if (o.mode == SampleMode::RetrievalAugmented) {
    if (o.database_x == nullptr || o.n_db <= 0) {
      out.detail = "database required for retrieval_augmented";
      return out;
    }
    const int input_dim = o.database_input_dim > 0 ? o.database_input_dim : static_cast<int>(x.size());
    if (label.empty()) {
      const auto hits =
          retrieve_from_x(*infer_, x.data(), o.database_x, o.n_db, input_dim, o.k_neighbors, opt);
      label = hits.empty() ? infer_label() : hits.front().predicted_label;
    }
    out.h = generate_retrieval_augmented(*infer_, x.data(), o.database_x, o.n_db, input_dim, o.k_neighbors,
                                         o.n_samples, o.temperature, o.n_steps, opt, &rng_, nullptr, nullptr,
                                         nullptr);
  } else if (o.mode == SampleMode::ClassGaussian) {
    if (label.empty()) {
      label = infer_label();
    }
    out.h = generate_class_gaussian(*infer_, label, o.n_samples, o.temperature, &rng_);
  } else if (o.mode == SampleMode::Conditioned) {
    if (label.empty()) {
      label = infer_label();
    }
    out.h = generate_conditioned(*infer_, label, o.n_samples, o.temperature, &rng_);
  }
  out.label = label;
  return out;
}

RetrieveOut Cypha::retrieve(const double* x, int n, const double* database_x, int n_db, int input_dim,
                            const RetrieveOpts& o) {
  RetrieveOut out;
  if (!infer_ || x == nullptr || database_x == nullptr || n_db <= 0) {
    out.detail = "No model loaded";
    return out;
  }
  std::vector<double> xv(x, x + n);
  if (pre_) {
    xv = pre_->transform_one(xv);
  }
  out.hits = retrieve_from_x(*infer_, xv.data(), database_x, n_db, input_dim, o.top_k, o.infer_opt, o.label);
  return out;
}

TokenOut Cypha::predict_next(std::uint32_t token) {
  TokenOut out;
  if (!seq_) {
    out.detail = "No sequence model loaded";
    return out;
  }
  auto pn = seq_->predict_next(token);
  out.log_probs = std::move(pn.log_probs);
  out.top_k_tokens = std::move(pn.top_k_tokens);
  out.top_k_probs = std::move(pn.top_k_probs);
  return out;
}

double Cypha::train_token(std::uint32_t token, std::uint32_t next) {
  if (!seq_) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return seq_->train_step(token, next).loss;
}

cyphalm::PredictiveCodecResult Cypha::compress_tokens(const std::vector<std::uint32_t>& tokens,
                                                      const cyphalm::PredictiveCodecOptions& opt) {
  cyphalm::PredictiveCodecResult out;
  if (!seq_) {
    out.detail = "No sequence model loaded";
    return out;
  }
  return cyphalm::compress_tokens(*seq_, tokens, opt);
}

std::vector<std::uint32_t> Cypha::decompress_tokens(const std::vector<std::uint8_t>& bytes,
                                                    std::uint32_t seed, std::size_t n_tokens,
                                                    std::string* detail,
                                                    const cyphalm::PredictiveCodecOptions& opt) {
  if (!seq_) {
    if (detail) *detail = "No sequence model loaded";
    return {};
  }
  return cyphalm::decompress_tokens(*seq_, bytes, seed, n_tokens, detail, opt);
}

std::vector<std::uint32_t> Cypha::generate_via_bits(const std::vector<std::uint32_t>& prefix,
                                                    std::size_t n_new, std::uint64_t rng_seed,
                                                    std::string* detail,
                                                    const cyphalm::PredictiveCodecOptions& opt) {
  if (!seq_) {
    if (detail) *detail = "No sequence model loaded";
    return {};
  }
  return cyphalm::generate_via_bits(*seq_, prefix, n_new, rng_seed, detail, opt);
}

std::string Cypha::generate(const std::vector<int>& prompt_ids, const GenerateTokenOpts& o) {
  if (!seq_) {
    return {};
  }
  auto gen = cyphalm::generate_decode(*seq_, prompt_ids, o.max_tokens, o.decode);
  std::vector<std::uint32_t> ids(gen.generated_ids.begin(), gen.generated_ids.end());
  if (seq_->has_bpe_tokenizer()) {
    return seq_->decode_tokens(ids);
  }
  // Char / byte vocabulary fallback (no BPE configured).
  std::string s;
  s.reserve(ids.size());
  for (std::uint32_t id : ids) {
    if (id < 256) {
      s.push_back(static_cast<char>(id));
    }
  }
  return s;
}

double Cypha::drift_score() const {
  if (!mem_) {
    return 0.0;
  }
  return mem_->world_drift_ema;
}

}  // namespace cypha
