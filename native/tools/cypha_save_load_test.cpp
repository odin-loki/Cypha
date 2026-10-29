/// CTest ``native_cypha_save_load_test``: ``Cypha::save`` then ``Cypha::load`` must give back the same model.
///
/// For each configuration the test trains model A, saves it, loads the file into a fresh object B, and
/// requires:
///   1. byte-identical ``predict`` output (label, scores, probabilities, anomaly, y, ...) on fixed probes,
///      under four predict modes (GH gate, plain with field, plain without field, self-correcting);
///   2. byte-identical losses while A and B keep learning on the same rows;
///   3. byte-identical predictions again after that extra learning;
///   4. byte-identical files when A and B are saved again;
///   5. byte-identical predictions from a third object C loaded from B's second save.
/// It also checks that a file without the new save-state block (the old format) still loads.
///
/// Configurations cover the GH gate in training, the tiered label context, field dynamics, custom
/// hyperparameters with replay, EWC, kernel LLR, a regression head and an early save (world n <= 20).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/nig_field.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kD = 10;  // not a power of two, so running means and 1/d products differ in the last bits
constexpr int kFieldDim = 16;
constexpr int kK = 3;
constexpr int kProbes = 40;

struct Config {
  std::string name;
  int n_train{200};
  int n_more{80};
  bool train_gh{false};
  bool field_inject{false};
  bool hparams{false};
  bool ewc{false};
  bool kernel{false};
  bool regression{false};
};

struct Row {
  std::vector<double> x;
  std::string label;
  double y{0.0};
};

/// Three overlapping Gaussian classes with sticky label order (so the tiered context prior matters).
class Data {
 public:
  explicit Data(unsigned seed) : rng_(seed) {
    std::normal_distribution<double> n01(0.0, 1.0);
    for (int k = 0; k < kK; ++k) {
      std::vector<double> m(kD);
      std::vector<double> w(kD);
      for (int j = 0; j < kD; ++j) {
        m[static_cast<std::size_t>(j)] = 1.5 * n01(rng_);
        w[static_cast<std::size_t>(j)] = n01(rng_);
      }
      means_.push_back(m);
      slopes_.push_back(w);
    }
  }

  Row next() {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> n01(0.0, 1.0);
    if (u(rng_) > 0.7) {
      cur_ = std::min(kK - 1, static_cast<int>(u(rng_) * kK));
    }
    Row r;
    r.label = "c" + std::to_string(cur_);
    r.x.resize(kD);
    double y = 0.5 * cur_;
    for (int j = 0; j < kD; ++j) {
      const std::size_t sj = static_cast<std::size_t>(j);
      r.x[sj] = means_[static_cast<std::size_t>(cur_)][sj] + 1.1 * n01(rng_);
      y += 0.2 * slopes_[static_cast<std::size_t>(cur_)][sj] * r.x[sj];
    }
    r.y = y + 0.05 * n01(rng_);
    return r;
  }

 private:
  std::mt19937 rng_;
  std::vector<std::vector<double>> means_;
  std::vector<std::vector<double>> slopes_;
  int cur_{0};
};

std::vector<std::vector<double>> make_probes() {
  Data d(777);
  std::vector<std::vector<double>> out;
  for (int i = 0; i < kProbes; ++i) {
    std::vector<double> x = d.next().x;
    if (i % 7 == 6) {  // a few far-out rows for the OOD / anomaly path
      for (double& v : x) {
        v *= 8.0;
      }
    }
    out.push_back(std::move(x));
  }
  return out;
}

std::vector<cypha::PredictOpts> predict_modes() {
  cypha::PredictOpts gh;
  gh.use_gh = true;
  gh.use_field = true;
  cypha::PredictOpts plain_field;
  plain_field.use_gh = false;
  plain_field.use_field = true;
  cypha::PredictOpts plain_nofield;
  plain_nofield.use_gh = false;
  plain_nofield.use_field = false;
  cypha::PredictOpts self_corr;
  self_corr.use_gh = false;
  self_corr.use_field = true;
  self_corr.self_correct = true;
  return {gh, plain_field, plain_nofield, self_corr};
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  return a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

std::vector<double> probabilities(const cypha::PredictOut& p, double temperature) {
  std::vector<double> z(p.all_scores.size());
  double mx = -1e300;
  for (std::size_t j = 0; j < z.size(); ++j) {
    z[j] = p.all_scores[j] / (temperature + 1e-8);
    mx = std::max(mx, z[j]);
  }
  double s = 0.0;
  for (double& v : z) {
    v = std::exp(v - mx);
    s += v;
  }
  for (double& v : z) {
    v /= s;
  }
  return z;
}

/// Empty string when equal; else the name of the first field that differs.
std::string predict_diff(cypha::Cypha& a, const cypha::PredictOut& pa, cypha::Cypha& b,
                         const cypha::PredictOut& pb) {
  if (pa.detail != pb.detail) return "detail";
  if (pa.label != pb.label) return "label";
  if (!same_bits(pa.confidence, pb.confidence)) return "confidence";
  if (pa.labels != pb.labels) return "labels";
  if (!same_vec(pa.all_scores, pb.all_scores)) return "all_scores";
  if (!same_bits(a.infer()->temperature, b.infer()->temperature)) return "temperature";
  if (!same_vec(probabilities(pa, a.infer()->temperature), probabilities(pb, b.infer()->temperature)))
    return "probabilities";
  if (pa.y.has_value() != pb.y.has_value()) return "y.has_value";
  if (pa.y.has_value() && !same_bits(*pa.y, *pb.y)) return "y";
  if (!same_bits(pa.uncertainty, pb.uncertainty)) return "uncertainty";
  if (!same_bits(pa.anomaly_score, pb.anomaly_score)) return "anomaly_score";
  if (pa.is_ood != pb.is_ood) return "is_ood";
  if (!same_bits(pa.r_eff, pb.r_eff)) return "r_eff";
  if (pa.self_corrected != pb.self_corrected) return "self_corrected";
  if (pa.correction_passes != pb.correction_passes) return "correction_passes";
  if (!same_bits(pa.r_eu_proxy, pb.r_eu_proxy)) return "r_eu_proxy";
  if (!same_bits(a.active_query_score(), b.active_query_score())) return "active_query_score";
  if (!same_bits(a.drift_score(), b.drift_score())) return "drift_score";
  return {};
}

/// Field-by-field state comparison: catches state that the probes and learning steps happen not to touch.
class StateDiff {
 public:
  void d(const char* name, double a, double b) { note(name, same_bits(a, b)); }
  void v(const char* name, const std::vector<double>& a, const std::vector<double>& b) { note(name, same_vec(a, b)); }
  template <class T>
  void eq(const char* name, const T& a, const T& b) { note(name, a == b); }
  void fv(const char* name, const std::vector<float>& a, const std::vector<float>& b) {
    note(name, a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0));
  }
  void dict(const char* name, const std::unordered_map<std::string, double>& a,
            const std::unordered_map<std::string, double>& b) {
    bool same = a.size() == b.size();
    for (const auto& kv : a) {
      auto it = b.find(kv.first);
      same = same && it != b.end() && same_bits(kv.second, it->second);
    }
    note(name, same);
  }
  void dict2(const char* name, const std::unordered_map<std::string, std::unordered_map<std::string, double>>& a,
             const std::unordered_map<std::string, std::unordered_map<std::string, double>>& b) {
    bool same = a.size() == b.size();
    for (const auto& kv : a) {
      auto it = b.find(kv.first);
      if (it == b.end()) {
        same = false;
        continue;
      }
      StateDiff inner;
      inner.dict("x", kv.second, it->second);
      same = same && inner.diffs.empty();
    }
    note(name, same);
  }
  void vdict(const char* name, const std::unordered_map<std::string, std::vector<double>>& a,
             const std::unordered_map<std::string, std::vector<double>>& b) {
    bool same = a.size() == b.size();
    for (const auto& kv : a) {
      auto it = b.find(kv.first);
      same = same && it != b.end() && same_vec(kv.second, it->second);
    }
    note(name, same);
  }
  std::vector<std::string> diffs;

 private:
  void note(const char* name, bool same) {
    if (!same) {
      diffs.emplace_back(name);
    }
  }
};

/// Every piece of state ``predict`` and ``update`` read. Skips ``infer()->saved_total_steps`` and
/// ``infer()->llr_ema`` (snapshots taken at load; the live values are ``total_steps()`` and ``llr_ema()``) and
/// the replay ``f`` rows and ``world_buf`` scratch (never read back).
std::vector<std::string> state_diff(cypha::Cypha& a, cypha::Cypha& b) {
  StateDiff s;
  const cypha::CyphaInferModel& ia = *a.infer();
  const cypha::CyphaInferModel& ib = *b.infer();
  s.eq("infer.d_latent", ia.d_latent, ib.d_latent);
  s.eq("infer.field_dim", ia.field_dim, ib.field_dim);
  s.eq("infer.labels", ia.labels, ib.labels);
  s.v("infer.n_obs", ia.n_obs, ib.n_obs);
  s.v("infer.D", ia.D, ib.D);
  s.v("infer.class_pi", ia.class_pi, ib.class_pi);
  s.eq("infer.class_n_comp", ia.class_n_comp, ib.class_n_comp);
  s.eq("infer.use_class_gmm", ia.use_class_gmm, ib.use_class_gmm);
  s.eq("infer.class_gmm_m", ia.class_gmm_m, ib.class_gmm_m);
  s.eq("infer.use_nig_bma", ia.use_nig_bma, ib.use_nig_bma);
  s.d("infer.nig_bma_z", ia.nig_bma_z, ib.nig_bma_z);
  s.eq("infer.cypha_format", ia.cypha_format, ib.cypha_format);
  s.v("infer.enc_w", ia.enc_w, ib.enc_w);
  s.v("infer.mu_world", ia.mu_world, ib.mu_world);
  s.v("infer.inv_v", ia.inv_v, ib.inv_v);
  s.d("infer.v_mean", ia.v_mean, ib.v_mean);
  s.v("infer.f_field", ia.f_field, ib.f_field);
  s.v("infer.field_h", ia.field_h, ib.field_h);
  s.v("infer.field_w_t", ia.field_w_t, ib.field_w_t);
  s.fv("infer.field_a_eff", ia.field_a_eff, ib.field_a_eff);
  s.fv("infer.field_sr_vec", ia.field_sr_vec, ib.field_sr_vec);
  s.v("infer.w_inject", ia.w_inject, ib.w_inject);
  s.d("infer.temperature", ia.temperature, ib.temperature);
  s.d("infer.mahal_ema", ia.mahal_ema, ib.mahal_ema);
  s.eq("infer.has_mahal_ema", ia.has_mahal_ema, ib.has_mahal_ema);
  s.d("infer.mahal_std_ema", ia.mahal_std_ema, ib.mahal_std_ema);
  s.eq("infer.total_correct", ia.total_correct, ib.total_correct);
  s.d("infer.mid_n", ia.mid_n, ib.mid_n);
  {
    bool same = ia.mid_freq.size() == ib.mid_freq.size();
    for (std::size_t i = 0; same && i < ia.mid_freq.size(); ++i) {
      same = ia.mid_freq[i].first == ib.mid_freq[i].first && same_bits(ia.mid_freq[i].second, ib.mid_freq[i].second);
    }
    s.eq("infer.mid_freq", same, true);
  }
  s.d("infer.mid_freq_total", ia.mid_freq_total, ib.mid_freq_total);
  s.eq("infer.ctx_last_label", ia.ctx_last_label, ib.ctx_last_label);
  s.dict("infer.t1_counts", ia.t1_counts, ib.t1_counts);
  s.d("infer.t1_total", ia.t1_total, ib.t1_total);
  s.eq("infer.ctx_history", ia.ctx_history, ib.ctx_history);
  s.dict2("infer.cooccur", ia.cooccur, ib.cooccur);
  s.dict("infer.cooccur_tot", ia.cooccur_tot, ib.cooccur_tot);
  s.dict2("infer.mid_trans", ia.mid_trans, ib.mid_trans);
  s.dict("infer.mid_trans_tot", ia.mid_trans_tot, ib.mid_trans_tot);
  s.d("infer.llr_scale_ema", ia.llr_scale_ema, ib.llr_scale_ema);
  s.eq("infer.llr_scale_n", ia.llr_scale_n, ib.llr_scale_n);
  s.d("infer.llr_scale_baseline", ia.llr_scale_baseline, ib.llr_scale_baseline);
  s.d("infer.base_temp", ia.base_temp, ib.base_temp);
  s.eq("infer.field_step", ia.field_step, ib.field_step);
  s.d("infer.deliberation_lo", ia.deliberation_lo, ib.deliberation_lo);
  s.d("infer.deliberation_hi", ia.deliberation_hi, ib.deliberation_hi);

  const cypha::CyphaDifMemoryState& ma = *a.mem();
  const cypha::CyphaDifMemoryState& mb = *b.mem();
  s.eq("mem.labels", ma.labels, mb.labels);
  s.v("mem.D", ma.D, mb.D);
  s.v("mem.class_pi", ma.class_pi, mb.class_pi);
  s.eq("mem.class_n_comp", ma.class_n_comp, mb.class_n_comp);
  s.eq("mem.use_class_gmm", ma.use_class_gmm, mb.use_class_gmm);
  s.eq("mem.class_gmm_m", ma.class_gmm_m, mb.class_gmm_m);
  s.eq("mem.task_prefix_protect", ma.task_prefix_protect, mb.task_prefix_protect);
  s.eq("mem.cypha_format", ma.cypha_format, mb.cypha_format);
  s.v("mem.n_obs_buf", ma.n_obs_buf, mb.n_obs_buf);
  s.eq("mem.n_correct", ma.n_correct, mb.n_correct);
  s.eq("mem.label_index", ma.label_index, mb.label_index);
  s.v("mem.f_field", ma.f_field, mb.f_field);
  s.v("mem.world_mu", ma.world_mu, mb.world_mu);
  s.v("mem.world_v", ma.world_v, mb.world_v);
  s.v("mem.world_inv_v", ma.world_inv_v, mb.world_inv_v);
  s.d("mem.world_v_mean", ma.world_v_mean, mb.world_v_mean);
  s.eq("mem.world_n", ma.world_n, mb.world_n);
  s.d("mem.world_drift_ema", ma.world_drift_ema, mb.world_drift_ema);
  s.v("mem.world_M2", ma.world_M2, mb.world_M2);
  s.d("mem.world_log_norm", ma.world_log_norm, mb.world_log_norm);
  s.d("mem.world_D_LOG2PI", ma.world_D_LOG2PI, mb.world_D_LOG2PI);
  s.v("mem.world_inv_d", ma.world_inv_d, mb.world_inv_d);
  s.eq("mem.world_log_n_ctr", ma.world_log_n_ctr, mb.world_log_n_ctr);

  s.d("world_lr", a.world_lr(), b.world_lr());
  s.d("delta_lr", a.delta_lr(), b.delta_lr());
  s.d("ood_sigma", a.ood_sigma(), b.ood_sigma());
  s.d("enc_lr", a.train_params().enc_lr, b.train_params().enc_lr);
  s.d("replay_ratio", a.train_params().replay_ratio, b.train_params().replay_ratio);
  s.eq("replay_cap", a.train_params().replay_cap, b.train_params().replay_cap);
  s.eq("align_every", a.train_params().align_every, b.train_params().align_every);
  s.eq("temp_recalib_every", a.train_params().temp_recalib_every, b.train_params().temp_recalib_every);
  s.eq("enc_updates", a.enc_updates(), b.enc_updates());
  s.eq("total_steps", a.total_steps(), b.total_steps());
  s.d("llr_ema", a.llr_ema(), b.llr_ema());
  s.v("gh_inv_v_clean", a.gh_inv_v_clean(), b.gh_inv_v_clean());
  s.d("gh_R_base", a.gh_R_base(), b.gh_R_base());
  s.d("gh_chi", a.gh_chi(), b.gh_chi());
  s.d("gh_psi", a.gh_psi(), b.gh_psi());
  s.eq("rng", a.rng(), b.rng());
  s.d("active_query_score", a.active_query_score(), b.active_query_score());
  {
    const auto ea = a.epistemic_threshold().state().raw();
    const auto eb = b.epistemic_threshold().state().raw();
    const bool same = same_bits(ea.mu, eb.mu) && same_bits(ea.lambda, eb.lambda) && same_bits(ea.alpha, eb.alpha) &&
                      same_bits(ea.beta, eb.beta) && same_bits(ea.running_mean, eb.running_mean) &&
                      same_bits(ea.running_m2, eb.running_m2) && ea.n_updates == eb.n_updates &&
                      ea.n_observations == eb.n_observations;
    s.eq("epistemic_threshold", same, true);
  }
  s.eq("ewc.present", a.ewc() != nullptr, b.ewc() != nullptr);
  if (a.ewc() != nullptr && b.ewc() != nullptr) {
    const auto ea = a.ewc()->export_state();
    const auto eb = b.ewc()->export_state();
    s.eq("ewc.protect_world_field", ea.protect_world_field, eb.protect_world_field);
    s.v("ewc.anchor_D", ea.anchor_D, eb.anchor_D);
    s.v("ewc.anchor_enc_w", ea.anchor_enc_w, eb.anchor_enc_w);
    s.v("ewc.anchor_world_mu", ea.anchor_world_mu, eb.anchor_world_mu);
    s.v("ewc.fisher_D", ea.fisher_D, eb.fisher_D);
    s.v("ewc.fisher_enc_w", ea.fisher_enc_w, eb.fisher_enc_w);
    s.v("ewc.fisher_world_mu", ea.fisher_world_mu, eb.fisher_world_mu);
  }
  s.d("ewc_lambda", a.ewc_lambda(), b.ewc_lambda());
  s.eq("replay.present", a.replay() != nullptr, b.replay() != nullptr);
  if (a.replay() != nullptr && b.replay() != nullptr) {
    const auto ra = a.replay()->export_state();
    const auto rb = b.replay()->export_state();
    s.eq("replay.cap", ra.cap, rb.cap);
    s.eq("replay.len", ra.len, rb.len);
    s.eq("replay.insert_n", ra.insert_n, rb.insert_n);
    s.eq("replay.d", ra.d, rb.d);
    s.v("replay.h", ra.h, rb.h);
    s.v("replay.w", ra.w, rb.w);
    s.eq("replay.labels", ra.labels, rb.labels);
  }
  s.eq("use_kernel_llr", a.use_kernel_llr(), b.use_kernel_llr());
  s.d("kernel_blend", a.kernel_blend(), b.kernel_blend());
  s.eq("kernel.present", a.kernel_mem() != nullptr, b.kernel_mem() != nullptr);
  if (a.kernel_mem() != nullptr && b.kernel_mem() != nullptr) {
    const auto ka = a.kernel_mem()->export_snapshot();
    const auto kb = b.kernel_mem()->export_snapshot();
    s.eq("kernel.feat_dim", ka.feat_dim, kb.feat_dim);
    s.eq("kernel.M", ka.M, kb.M);
    s.d("kernel.gamma", ka.gamma, kb.gamma);
    s.eq("kernel.n_basis", ka.n_basis, kb.n_basis);
    s.eq("kernel.n_seen", ka.n_seen, kb.n_seen);
    s.v("kernel.basis", ka.basis_rowmajor, kb.basis_rowmajor);
    bool same_w = ka.weights.size() == kb.weights.size();
    for (const auto& kv : ka.weights) {
      auto it = kb.weights.find(kv.first);
      same_w = same_w && it != kb.weights.end() && same_vec(kv.second, it->second);
    }
    s.eq("kernel.weights", same_w, true);
    s.eq("kernel.rng", a.kernel_mem()->rng_engine(), b.kernel_mem()->rng_engine());
    s.d("kernel.gamma_scale", a.kernel_mem()->gamma_scale(), b.kernel_mem()->gamma_scale());
    s.eq("kernel.landmark_sampling", a.kernel_mem()->landmark_sampling(), b.kernel_mem()->landmark_sampling());
  }
  s.v("reg_mu", a.reg_mu(), b.reg_mu());
  s.v("reg_var", a.reg_var(), b.reg_var());
  s.eq("mke_active", a.mke_active(), b.mke_active());
  s.eq("mke_d_in", a.mke_d_in(), b.mke_d_in());
  s.v("mke_W", a.mke_W(), b.mke_W());
  s.v("mke_b", a.mke_b(), b.mke_b());
  s.d("mke_temperature", a.mke_temperature(), b.mke_temperature());
  s.d("mke_forgetting", a.mke_forgetting(), b.mke_forgetting());
  s.d("mke_pi_floor", a.mke_pi_floor(), b.mke_pi_floor());
  s.v("mke_gh_scales", a.mke_gh_scales(), b.mke_gh_scales());
  s.vdict("mke_w", a.mke_w(), b.mke_w());
  s.vdict("mke_p", a.mke_p(), b.mke_p());
  return s.diffs;
}

struct CompareResult {
  int n{0};
  int diff{0};
  int label_diff{0};
  int errors{0};  // predict returned an error on the reference model (would make the check vacuous)
  std::string first;
};

/// Runs the same probes through both models in lockstep (self-correction mutates both the same way).
CompareResult compare_predictions(cypha::Cypha& a, cypha::Cypha& b,
                                  const std::vector<std::vector<double>>& probes) {
  CompareResult r;
  const auto modes = predict_modes();
  for (std::size_t mi = 0; mi < modes.size(); ++mi) {
    for (std::size_t i = 0; i < probes.size(); ++i) {
      const cypha::PredictOut pa = a.predict(probes[i].data(), kD, modes[mi]);
      const cypha::PredictOut pb = b.predict(probes[i].data(), kD, modes[mi]);
      ++r.n;
      if (!pa.detail.empty() || !std::isfinite(pa.confidence)) {
        ++r.errors;
      }
      const std::string d = predict_diff(a, pa, b, pb);
      if (!d.empty()) {
        ++r.diff;
        if (pa.label != pb.label) {
          ++r.label_diff;
        }
        if (r.first.empty()) {
          r.first = "mode " + std::to_string(mi) + " probe " + std::to_string(i) + ": " + d;
        }
      }
    }
  }
  return r;
}

cypha::UpdateOut train_one(cypha::Cypha& m, const Row& row, const Config& c, int step) {
  cypha::UpdateOpts o;
  o.use_gh = c.train_gh;
  if (c.ewc && step == c.n_train / 2) {
    o.ewc_snapshot = true;
    o.ewc_lambda = 0.5;
  }
  if (c.regression) {
    o.router_train_label = &row.label;
    return m.update(row.x.data(), kD, nullptr, &row.y, o);
  }
  return m.update(row.x.data(), kD, &row.label, nullptr, o);
}

std::vector<char> read_bytes(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_regression_json(const fs::path& p) {
  std::mt19937 rng(31);
  std::normal_distribution<double> n01(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.0, 6.283185307179586);
  std::ostringstream o;
  o << std::setprecision(17);
  o << "{\"experts\": {}, \"mke\": {\"d_in\": " << kD << ", \"D_rff\": " << kD << ", \"rff_W_rowmajor\": [";
  for (int i = 0; i < kD * kD; ++i) {
    o << (i ? ", " : "") << 0.5 * n01(rng);
  }
  o << "], \"rff_b\": [";
  for (int i = 0; i < kD; ++i) {
    o << (i ? ", " : "") << u(rng);
  }
  o << "], \"temperature\": 1.0, \"forgetting_factor\": 0.999, \"pi_floor\": 0.02, \"w\": {}, \"P\": {}}}\n";
  std::ofstream f(p);
  f << o.str();
}

bool run_config(const Config& c, const fs::path& base, const std::vector<std::vector<double>>& probes) {
  const fs::path dir = base / c.name;
  fs::create_directories(dir);

  cypha::FreshModelParams fp;
  fp.input_dim = kD;
  fp.field_dim = kFieldDim;
  fp.temperature = 1.0;
  fp.world_lr = 0.008;
  fp.delta_lr = 0.05;
  cypha::CNode root = cypha::create_fresh_model_root(fp);
  if (c.field_inject) {
    // Non-zero injection so the NIG field (field_h, W_T, a_eff, step) moves while training, a random W_T, and a
    // non-zero world F_field so the field shifts class means (else field_h never reaches a prediction).
    std::mt19937 rng(5);
    std::normal_distribution<double> n01(0.0, 1.0);
    for (auto& kv : root.map) {
      if (kv.first == "w_inject") {
        for (double& v : kv.second.tensor) {
          v = 0.4 * n01(rng);
        }
      } else if (kv.first == "field_W_T") {
        // Spectral radius well above 0.85, so the first causal update rescales W_T (and a_eff moves with it).
        for (double& v : kv.second.tensor) {
          v = 0.5 * n01(rng);
        }
      } else if (kv.first == "world") {
        for (auto& wk : kv.second.map) {
          if (wk.first == "F_field") {
            for (double& v : wk.second.tensor) {
              v = 0.3 * n01(rng);
            }
          }
        }
      }
    }
  }
  if (c.field_inject) {
    // Stored a_eff consistent with the new W_T (as a real save would leave it).
    std::vector<double> w_t;
    for (const auto& kv : root.map) {
      if (kv.first == "field_W_T") {
        w_t = kv.second.tensor;
      }
    }
    std::vector<float> a_eff;
    cypha::recompute_field_a_eff(kFieldDim, w_t, a_eff);
    for (auto& kv : root.map) {
      if (kv.first == "field_a_eff") {
        kv.second.tensor.assign(a_eff.begin(), a_eff.end());
      }
    }
  }
  const fs::path fresh = dir / "fresh.cypha";
  cypha::save_cypha_file(fresh.string().c_str(), root);

  std::string reg_json;
  if (c.regression) {
    // Placed next to the saved files on purpose: it holds the untrained head, so a reload that picks it up
    // instead of the saved head would lose the learned experts.
    reg_json = (dir / "regression_head.json").string();
    write_regression_json(reg_json);
  }

  cypha::Cypha a;
  if (!a.load(fresh.string(), "", "", reg_json)) {
    std::cerr << c.name << ": could not load the fresh model\n";
    return false;
  }
  if (c.hparams) {
    a.world_lr() = 0.02;
    a.delta_lr() = 0.1;
    a.ood_sigma() = 9.0;
    a.train_params().enc_lr = 0.01;
    a.train_params().replay_ratio = 0.6;
    a.train_params().replay_cap = 48;
    a.train_params().align_every = 40;
    a.train_params().temp_recalib_every = 25;
    a.replay_owned() = std::make_unique<cypha::ReplayBuffer>(48);
    a.mem()->task_prefix_protect = "c1";  // D16 task-sticky learning: only c1 and the target class move
    // Settings callers can change through infer(); training does not move them, so set them to non-defaults.
    a.infer()->base_temp = 1.3;
    a.infer()->total_correct = 17;
    a.infer()->deliberation_lo = 0.1;
    a.infer()->deliberation_hi = 0.9;
  }
  if (c.field_inject) {
    a.gh_psi() = 1.25;  // GH session ψ (training adapts χ only, so set ψ off its default)
  }
  if (c.kernel) {
    a.kernel_mem_owned() = std::make_unique<cypha::KernelMemory>(kD, 24, 11);
    a.use_kernel_llr() = true;
    a.kernel_blend() = 0.3;
  }

  Data data(1234);
  for (int s = 0; s < c.n_train; ++s) {
    const cypha::UpdateOut u = train_one(a, data.next(), c, s);
    if (!u.detail.empty()) {
      std::cerr << c.name << ": train step " << s << " failed: " << u.detail << "\n";
      return false;
    }
  }

  // The configuration must reach the state it is meant to test, else the round-trip check is vacuous.
  if (c.field_inject) {
    bool moved = a.infer()->field_step > 0;
    bool nonzero = false;
    for (double v : a.infer()->field_h) {
      nonzero = nonzero || v != 0.0;
    }
    if (!moved || !nonzero) {
      std::cerr << c.name << ": the field did not move while training\n";
      return false;
    }
  }
  if (c.kernel && (a.kernel_mem() == nullptr || a.kernel_mem()->n_seen() <= a.kernel_mem()->M())) {
    std::cerr << c.name << ": the kernel reservoir did not fill\n";
    return false;
  }
  if (c.ewc && (a.ewc() == nullptr || !a.ewc()->has_snapshot())) {
    std::cerr << c.name << ": no EWC snapshot\n";
    return false;
  }
  if (c.regression && !a.mke_active()) {
    std::cerr << c.name << ": regression head not active\n";
    return false;
  }

  const fs::path a1 = dir / "a1.cypha";
  a.save(a1.string());
  cypha::Cypha b;
  if (!b.load(a1.string())) {
    std::cerr << c.name << ": could not load the saved model\n";
    return false;
  }

  bool ok = true;
  auto report = [&](const char* what, const CompareResult& r) {
    std::cout << "  " << what << ": " << r.diff << "/" << r.n << " predictions differ (" << r.label_diff
              << " labels)";
    if (!r.first.empty()) {
      std::cout << "; first: " << r.first;
    }
    std::cout << "\n";
    if (r.diff != 0) {
      ok = false;
    }
    if (r.errors != 0) {
      std::cout << "  " << what << ": " << r.errors << " predictions returned an error\n";
      ok = false;
    }
  };

  auto report_state = [&](const char* what, const std::vector<std::string>& diffs) {
    std::cout << "  " << what << ": " << diffs.size() << " state fields differ";
    for (std::size_t i = 0; i < diffs.size() && i < 6; ++i) {
      std::cout << (i ? ", " : ": ") << diffs[i];
    }
    std::cout << "\n";
    if (!diffs.empty()) {
      ok = false;
    }
  };

  std::cout << c.name << ":\n";
  // State first: predict overwrites active_query_score and moves the self-correction threshold.
  report_state("after load", state_diff(a, b));
  report("after load", compare_predictions(a, b, probes));

  int loss_diff = 0;
  std::string first_loss;
  for (int s = 0; s < c.n_more; ++s) {
    const Row row = data.next();
    const cypha::UpdateOut ua = train_one(a, row, c, c.n_train + s);
    const cypha::UpdateOut ub = train_one(b, row, c, c.n_train + s);
    if (!ua.detail.empty() || ua.detail != ub.detail || !same_bits(ua.loss, ub.loss)) {
      ++loss_diff;
      if (first_loss.empty()) {
        std::ostringstream o;
        o << std::setprecision(17) << "step " << s << ": " << ua.loss << " vs " << ub.loss << " " << ua.detail;
        first_loss = o.str();
      }
    }
  }
  std::cout << "  keep learning: " << loss_diff << "/" << c.n_more << " losses differ";
  if (!first_loss.empty()) {
    std::cout << "; first: " << first_loss;
  }
  std::cout << "\n";
  if (loss_diff != 0) {
    ok = false;
  }
  report_state("after more learning", state_diff(a, b));
  report("after more learning", compare_predictions(a, b, probes));

  const fs::path a2 = dir / "a2.cypha";
  const fs::path b2 = dir / "b2.cypha";
  a.save(a2.string());
  b.save(b2.string());
  const bool same_files = read_bytes(a2) == read_bytes(b2);
  std::cout << "  second save: files " << (same_files ? "identical" : "DIFFER") << "\n";
  if (!same_files) {
    ok = false;
  }

  cypha::Cypha cm;
  if (!cm.load(b2.string())) {
    std::cerr << c.name << ": could not load the second save\n";
    return false;
  }
  report_state("load of second save", state_diff(a, cm));
  report("load of second save", compare_predictions(a, cm, probes));

  // Old format: strip the save-state block and version key; the file must still load and predict.
  cypha::CNode old_root = cypha::load_cypha_file(a1.string().c_str());
  std::vector<std::pair<std::string, cypha::CNode>> kept;
  for (auto& kv : old_root.map) {
    if (kv.first != "cypha_save_version" && kv.first != "cypha_save_state") {
      kept.push_back(std::move(kv));
    }
  }
  old_root.map = std::move(kept);
  const fs::path old_path = dir / "old_format" / "model.cypha";
  fs::create_directories(old_path.parent_path());
  cypha::save_cypha_file(old_path.string().c_str(), old_root);
  cypha::Cypha old_model;
  if (!old_model.load(old_path.string())) {
    std::cout << "  old format: load FAILED\n";
    ok = false;
  } else {
    const cypha::PredictOut p = old_model.predict(probes[0].data(), kD, cypha::PredictOpts{});
    const bool good = p.detail.empty() && std::isfinite(p.confidence);
    std::cout << "  old format: loads and predicts " << (good ? "ok" : "FAILED") << "\n";
    if (!good) {
      ok = false;
    }
  }

  std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  fs::path base;
  if (argc > 1) {
    base = argv[1];
  } else {
    base = fs::temp_directory_path() / "cypha_save_load_test";
  }
  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(base);

  std::vector<Config> configs;
  {
    Config c;
    c.name = "plain";
    configs.push_back(c);
  }
  {
    Config c;
    c.name = "gh_gate";
    c.train_gh = true;
    configs.push_back(c);
  }
  {
    Config c;
    c.name = "field_dynamics";
    c.train_gh = true;
    c.field_inject = true;
    configs.push_back(c);
  }
  {
    Config c;
    c.name = "hparams_replay";
    c.hparams = true;
    configs.push_back(c);
  }
  {
    Config c;
    c.name = "ewc";
    c.ewc = true;
    configs.push_back(c);
  }
  {
    Config c;
    c.name = "kernel_llr";
    c.kernel = true;
    configs.push_back(c);
  }
  {
    Config c;
    c.name = "regression_head";
    c.regression = true;
    configs.push_back(c);
  }
  {
    Config c;
    c.name = "early_save";
    c.n_train = 12;
    c.n_more = 40;
    c.train_gh = true;
    configs.push_back(c);
  }

  const auto probes = make_probes();
  int failed = 0;
  for (const auto& c : configs) {
    try {
      if (!run_config(c, base, probes)) {
        ++failed;
      }
    } catch (const std::exception& e) {
      std::cout << c.name << ": exception: " << e.what() << "\n  FAIL\n";
      ++failed;
    }
  }
  fs::remove_all(base, ec);
  if (failed == 0) {
    std::cout << "cypha_save_load_test: all " << configs.size() << " configurations round-trip exactly\n";
  } else {
    std::cout << "cypha_save_load_test: " << failed << " of " << configs.size() << " configurations FAILED\n";
  }
  return failed == 0 ? 0 : 1;
}
