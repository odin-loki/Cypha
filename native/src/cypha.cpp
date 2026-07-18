#include "cypha/cypha.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
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

bool Cypha::load(const std::string& cypha_path, const std::string& preprocessor_path,
                 const std::string& f_field_json_path, const std::string& regression_json_path,
                 const std::string& train_hparams_path) {
  apply_default_hparams();
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

  if (!regression_json_path.empty()) {
    if (!load_regression_json(regression_json_path)) {
      return false;
    }
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
  cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = vocab_size;
  cfg.d_embed = d_model;
  cfg.d_state = d_model;
  cyphalm::apply_cell_variant("U06", cfg);
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

  auto upsert = [&](const char* key, CNode val) {
    for (auto& kv : save_root.map) {
      if (kv.first == key) {
        kv.second = std::move(val);
        return;
      }
    }
    save_root.map.emplace_back(key, std::move(val));
  };

  for (auto& kv : save_root.map) {
    if (kv.first == "enc_W") {
      kv.second.kind = CNode::Tensor;
      kv.second.shape = {static_cast<std::uint32_t>(synced.d_latent),
                         static_cast<std::uint32_t>(synced.d_latent)};
      kv.second.tensor = synced.enc_w;
    } else if (kv.first == "field_h") {
      kv.second.kind = CNode::Tensor;
      kv.second.shape = {static_cast<std::uint32_t>(synced.field_dim)};
      kv.second.tensor = synced.field_h;
    } else if (kv.first == "temperature") {
      kv.second.kind = CNode::Float;
      kv.second.f = synced.temperature;
    } else if (kv.first == "llr_ema") {
      kv.second.kind = CNode::Float;
      kv.second.f = llr_ema_;
    } else if (kv.first == "total_steps") {
      kv.second.kind = CNode::Int;
      kv.second.i = static_cast<std::int64_t>(total_steps_);
    }
  }

  {
    CNode ood;
    ood.kind = CNode::Float;
    ood.f = ood_sigma_;
    upsert("ood_sigma", std::move(ood));
    CNode gchi;
    gchi.kind = CNode::Float;
    gchi.f = gh_chi_;
    upsert("gh_chi_session", std::move(gchi));
    CNode gpsi;
    gpsi.kind = CNode::Float;
    gpsi.f = gh_psi_;
    upsert("gh_psi_session", std::move(gpsi));
    CNode grb;
    grb.kind = CNode::Float;
    grb.f = gh_R_base_;
    upsert("gh_R_base", std::move(grb));
    if (static_cast<int>(gh_inv_v_clean_.size()) == synced.d_latent) {
      CNode ginv;
      ginv.kind = CNode::Tensor;
      ginv.shape = {static_cast<std::uint32_t>(synced.d_latent)};
      ginv.tensor = gh_inv_v_clean_;
      upsert("gh_inv_v_clean", std::move(ginv));
    }
  }

  if (kernel_mem_) {
    patch_kernel_into_root(save_root, *kernel_mem_, use_kernel_llr_, kernel_blend_);
  }
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
    const double r_base = (infer_->has_mahal_ema && infer_->mahal_ema > 0.0) ? infer_->mahal_ema : 1.0;
    out.anomaly_score = gh_infer_anomaly_score(out.r_eff, r_base);
    out.is_ood = out.anomaly_score > kOodThreshold;
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

}  // namespace cypha
