/// CTest ``native_branch_a_router_save_load_test``: ``BranchARouter::save_checkpoint`` then ``load_checkpoint``
/// must give back the same router (docs/reports/ROUTER_SAVE_LOAD_FIX.md).
///
/// Each configuration trains a Cypha model on synthetic rows, saves it with ``Cypha::save`` and writes a Branch A
/// checkpoint around it (``branch_a_router.json`` plus the model file, and a separate ``f_field_json`` where the
/// configuration says so). Router A loads that checkpoint. The test then requires:
///   1. router B, loaded from A's save, routes every probe text byte-identically to A (label, confidence,
///      epistemic value, abstain, action) at four thresholds;
///   2. B's loaded inference model and memory state equal A's, field by field, and their summaries agree;
///   3. B's save is byte-identical to A's (both files), and router C loaded from it routes like A.
/// The router does not learn (it has no update), so there is no "further learning" step.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/branch_a_router.hpp"
#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/nig_field.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kD = 16;          // latent size = the router's n_components
constexpr int kFieldDim = 12;
constexpr int kFeatures = 512;  // the router's hashing features
constexpr int kK = 3;
constexpr int kTrain = 240;

struct Config {
  std::string name;
  bool projection{true};    // a stored n_components x n_features projection (else hash truncation)
  bool ff_json{false};      // world.F_field kept out of the model file, in a separate f_field_json (older layout)
  bool format1{false};      // the model file without Cypha's save-state block (save format 1)
  bool train_seconds{true};
  double threshold{0.5};
};

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

bool same_fvec(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

std::vector<char> read_bytes(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

/// Rows for training: three Gaussian classes in the latent space the router feeds to the model.
struct Rows {
  explicit Rows(unsigned seed) : rng(seed) {
    std::normal_distribution<double> n01(0.0, 1.0);
    for (int k = 0; k < kK; ++k) {
      std::vector<double> m(kD);
      for (double& v : m) v = 1.2 * n01(rng);
      means.push_back(m);
    }
  }
  std::vector<double> next(int& k) {
    std::uniform_int_distribution<int> pick(0, kK - 1);
    std::normal_distribution<double> n01(0.0, 1.0);
    k = pick(rng);
    std::vector<double> x(kD);
    for (int j = 0; j < kD; ++j) x[static_cast<std::size_t>(j)] = means[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] + n01(rng);
    return x;
  }
  std::mt19937 rng;
  std::vector<std::vector<double>> means;
};

/// Probe texts: words from a small vocabulary, plus empty and one-word texts.
std::vector<std::string> probe_texts() {
  const char* words[] = {"river", "bank", "loan", "money", "fish", "water", "court", "judge", "law", "goal",
                         "match", "team", "music", "song", "band", "stone", "city", "king", "war", "peace",
                         "tree", "leaf", "rain", "cloud", "ship", "sea", "bridge", "road", "car", "wheel"};
  std::mt19937 rng(91);
  std::uniform_int_distribution<int> w(0, 29);
  std::uniform_int_distribution<int> len(1, 12);
  std::vector<std::string> out = {"", "a", "river bank", "The judge and the law court."};
  for (int i = 0; i < 56; ++i) {
    std::string s;
    const int n = len(rng);
    for (int t = 0; t < n; ++t) {
      if (t) s += ' ';
      s += words[w(rng)];
    }
    out.push_back(s);
  }
  return out;
}

const std::vector<std::optional<double>>& thresholds() {
  static const std::vector<std::optional<double>> t = {std::nullopt, 0.05, 0.5, 2.0};
  return t;
}

/// Empty when equal, else the first difference.
std::string route_diff(const cypha::BranchARouter& a, const cypha::BranchARouter& b,
                       const std::vector<std::string>& texts) {
  for (const auto& thr : thresholds()) {
    for (const auto& t : texts) {
      const auto ra = a.route(t, thr);
      const auto rb = b.route(t, thr);
      const std::string where = " (text \"" + t.substr(0, 30) + "\", threshold " +
                                (thr ? std::to_string(*thr) : std::string("default")) + ")";
      if (ra.label != rb.label) return "label" + where;
      if (!same_bits(ra.confidence, rb.confidence)) return "confidence" + where;
      if (!same_bits(ra.epistemic_var, rb.epistemic_var)) return "epistemic_var" + where;
      if (ra.abstain != rb.abstain) return "abstain" + where;
      if (ra.action != rb.action) return "action" + where;
      if (ra.embedding_backend != rb.embedding_backend) return "embedding_backend" + where;
    }
  }
  return {};
}

template <class M>
bool same_dict(const M& a, const M& b) {
  if (a.size() != b.size()) return false;
  for (const auto& kv : a) {
    auto it = b.find(kv.first);
    if (it == b.end() || !same_bits(kv.second, it->second)) return false;
  }
  return true;
}

template <class M>
bool same_dict2(const M& a, const M& b) {
  if (a.size() != b.size()) return false;
  for (const auto& kv : a) {
    auto it = b.find(kv.first);
    if (it == b.end() || !same_dict(kv.second, it->second)) return false;
  }
  return true;
}

/// Field-by-field comparison of the loaded inference model and memory state.
std::vector<std::string> state_diff(const cypha::BranchARouter& ra, const cypha::BranchARouter& rb) {
  std::vector<std::string> bad;
  const auto* ia = ra.infer_model();
  const auto* ib = rb.infer_model();
  if (!ia || !ib) return {"infer model missing"};
  auto chk = [&](const char* n, bool ok) {
    if (!ok) bad.push_back(n);
  };
  chk("infer.d_latent", ia->d_latent == ib->d_latent);
  chk("infer.field_dim", ia->field_dim == ib->field_dim);
  chk("infer.labels", ia->labels == ib->labels);
  chk("infer.n_obs", same_vec(ia->n_obs, ib->n_obs));
  chk("infer.D", same_vec(ia->D, ib->D));
  chk("infer.class_pi", same_vec(ia->class_pi, ib->class_pi));
  chk("infer.class_n_comp", ia->class_n_comp == ib->class_n_comp);
  chk("infer.use_class_gmm", ia->use_class_gmm == ib->use_class_gmm);
  chk("infer.cypha_format", ia->cypha_format == ib->cypha_format);
  chk("infer.enc_w", same_vec(ia->enc_w, ib->enc_w));
  chk("infer.mu_world", same_vec(ia->mu_world, ib->mu_world));
  chk("infer.inv_v", same_vec(ia->inv_v, ib->inv_v));
  chk("infer.v_mean", same_bits(ia->v_mean, ib->v_mean));
  chk("infer.f_field", same_vec(ia->f_field, ib->f_field));
  chk("infer.field_h", same_vec(ia->field_h, ib->field_h));
  chk("infer.field_w_t", same_vec(ia->field_w_t, ib->field_w_t));
  chk("infer.field_a_eff", same_fvec(ia->field_a_eff, ib->field_a_eff));
  chk("infer.field_sr_vec", same_fvec(ia->field_sr_vec, ib->field_sr_vec));
  chk("infer.w_inject", same_vec(ia->w_inject, ib->w_inject));
  chk("infer.temperature", same_bits(ia->temperature, ib->temperature));
  chk("infer.mahal_ema", same_bits(ia->mahal_ema, ib->mahal_ema));
  chk("infer.has_mahal_ema", ia->has_mahal_ema == ib->has_mahal_ema);
  chk("infer.mahal_std_ema", same_bits(ia->mahal_std_ema, ib->mahal_std_ema));
  chk("infer.total_correct", ia->total_correct == ib->total_correct);
  chk("infer.mid_n", same_bits(ia->mid_n, ib->mid_n));
  {
    bool same = ia->mid_freq.size() == ib->mid_freq.size();
    for (std::size_t i = 0; same && i < ia->mid_freq.size(); ++i) {
      same = ia->mid_freq[i].first == ib->mid_freq[i].first && same_bits(ia->mid_freq[i].second, ib->mid_freq[i].second);
    }
    chk("infer.mid_freq", same);
  }
  chk("infer.mid_freq_total", same_bits(ia->mid_freq_total, ib->mid_freq_total));
  chk("infer.ctx_last_label", ia->ctx_last_label == ib->ctx_last_label);
  chk("infer.t1_counts", same_dict(ia->t1_counts, ib->t1_counts));
  chk("infer.t1_total", same_bits(ia->t1_total, ib->t1_total));
  chk("infer.ctx_history", ia->ctx_history == ib->ctx_history);
  chk("infer.cooccur", same_dict2(ia->cooccur, ib->cooccur));
  chk("infer.mid_trans", same_dict2(ia->mid_trans, ib->mid_trans));
  chk("infer.llr_scale_ema", same_bits(ia->llr_scale_ema, ib->llr_scale_ema));
  chk("infer.llr_scale_n", ia->llr_scale_n == ib->llr_scale_n);
  chk("infer.llr_scale_baseline", same_bits(ia->llr_scale_baseline, ib->llr_scale_baseline));
  chk("infer.base_temp", same_bits(ia->base_temp, ib->base_temp));
  chk("infer.field_step", ia->field_step == ib->field_step);
  chk("infer.deliberation_lo", same_bits(ia->deliberation_lo, ib->deliberation_lo));
  chk("infer.deliberation_hi", same_bits(ia->deliberation_hi, ib->deliberation_hi));

  const auto* ma = ra.memory_state();
  const auto* mb = rb.memory_state();
  if (!ma || !mb) {
    bad.push_back("memory state missing");
    return bad;
  }
  chk("mem.labels", ma->labels == mb->labels);
  chk("mem.D", same_vec(ma->D, mb->D));
  chk("mem.class_pi", same_vec(ma->class_pi, mb->class_pi));
  chk("mem.class_n_comp", ma->class_n_comp == mb->class_n_comp);
  chk("mem.cypha_format", ma->cypha_format == mb->cypha_format);
  chk("mem.n_obs_buf", same_vec(ma->n_obs_buf, mb->n_obs_buf));
  chk("mem.n_correct", ma->n_correct == mb->n_correct);
  chk("mem.f_field", same_vec(ma->f_field, mb->f_field));
  chk("mem.world_mu", same_vec(ma->world_mu, mb->world_mu));
  chk("mem.world_v", same_vec(ma->world_v, mb->world_v));
  chk("mem.world_inv_v", same_vec(ma->world_inv_v, mb->world_inv_v));
  chk("mem.world_v_mean", same_bits(ma->world_v_mean, mb->world_v_mean));
  chk("mem.world_n", ma->world_n == mb->world_n);
  chk("mem.world_drift_ema", same_bits(ma->world_drift_ema, mb->world_drift_ema));
  chk("mem.world_M2", same_vec(ma->world_M2, mb->world_M2));
  chk("mem.world_buf", same_vec(ma->world_buf, mb->world_buf));
  chk("mem.world_log_norm", same_bits(ma->world_log_norm, mb->world_log_norm));
  chk("mem.world_inv_d", same_vec(ma->world_inv_d, mb->world_inv_d));
  return bad;
}

/// The router's summary without the fields that name where it was loaded from.
nlohmann::json summary_core(const cypha::BranchARouter& r) {
  auto j = r.summary();
  j.erase("checkpoint_base");
  j.erase("checkpoint_exists");
  return j;
}

/// Train a Cypha model and write a Branch A checkpoint around it in ``dir``. Returns the JSON path.
fs::path write_checkpoint(const Config& c, const fs::path& dir, double std_scale) {
  fs::create_directories(dir);
  cypha::FreshModelParams fp;
  fp.input_dim = kD;
  fp.field_dim = kFieldDim;
  fp.temperature = 1.0;
  cypha::CNode root = cypha::create_fresh_model_root(fp);
  // A moving field (as in native_cypha_save_load_test's field configuration), so F_field and field_h reach routing.
  std::mt19937 rng(5);
  std::normal_distribution<double> n01(0.0, 1.0);
  std::vector<double> w_t;
  for (auto& kv : root.map) {
    if (kv.first == "w_inject") {
      for (double& v : kv.second.tensor) v = 0.4 * n01(rng);
    } else if (kv.first == "field_W_T") {
      for (double& v : kv.second.tensor) v = 0.5 * n01(rng);
      w_t = kv.second.tensor;
    } else if (kv.first == "world") {
      for (auto& wk : kv.second.map) {
        if (wk.first == "F_field") {
          for (double& v : wk.second.tensor) v = 0.3 * n01(rng);
        }
      }
    }
  }
  std::vector<float> a_eff;
  cypha::recompute_field_a_eff(kFieldDim, w_t, a_eff);
  for (auto& kv : root.map) {
    if (kv.first == "field_a_eff") kv.second.tensor.assign(a_eff.begin(), a_eff.end());
  }
  const fs::path fresh = dir / "fresh.cypha";
  cypha::save_cypha_file(fresh.string().c_str(), root);

  cypha::Cypha m;
  if (!m.load(fresh.string())) throw std::runtime_error(c.name + ": could not load the fresh model");
  Rows rows(1234);
  for (int s = 0; s < kTrain; ++s) {
    int k = 0;
    const auto x = rows.next(k);
    const std::string label = "topic" + std::to_string(k);
    cypha::UpdateOpts o;
    const auto u = m.update(x.data(), kD, &label, nullptr, o);
    if (!u.detail.empty()) throw std::runtime_error(c.name + ": train step failed: " + u.detail);
  }
  const fs::path model = dir / "model.cypha";
  m.save(model.string());

  nlohmann::json meta;
  meta["version"] = 1;
  meta["format"] = "native";
  meta["model_cypha"] = "model.cypha";

  if (c.ff_json || c.format1) {
    cypha::CNode r = cypha::load_cypha_file(model.string().c_str());
    if (c.format1) {
      std::vector<std::pair<std::string, cypha::CNode>> keep;
      for (auto& kv : r.map) {
        if (kv.first != "cypha_save_version" && kv.first != "cypha_save_state") keep.push_back(std::move(kv));
      }
      r.map = std::move(keep);
    }
    if (c.ff_json) {
      std::vector<double> ff;
      for (auto& kv : r.map) {
        if (kv.first != "world") continue;
        std::vector<std::pair<std::string, cypha::CNode>> keep;
        for (auto& wk : kv.second.map) {
          if (wk.first == "F_field") {
            ff = wk.second.tensor;
          } else {
            keep.push_back(std::move(wk));
          }
        }
        kv.second.map = std::move(keep);
      }
      if (ff.empty()) throw std::runtime_error(c.name + ": the saved model has no world.F_field to move out");
      nlohmann::json ffj;
      ffj["F_field"] = ff;
      std::ofstream(dir / "f_field.json") << ffj.dump() << '\n';
      meta["f_field_json"] = "f_field.json";
    }
    cypha::save_cypha_file(model.string().c_str(), r);
  }

  std::normal_distribution<double> nproj(0.0, 1.5);
  std::uniform_real_distribution<double> ustd(0.5, 1.5);
  std::vector<double> mean(kD), stdv(kD);
  for (int j = 0; j < kD; ++j) {
    // Hashed features without a projection are non-negative (about 0.3 where a token lands): centre them.
    mean[static_cast<std::size_t>(j)] = c.projection ? 0.05 * n01(rng) : 0.15 + 0.02 * n01(rng);
    stdv[static_cast<std::size_t>(j)] = j == 3 ? 0.0 : std_scale * ustd(rng);  // one zero std: load sets it to 1
  }
  meta["mean"] = mean;
  meta["std"] = stdv;
  meta["epistemic_threshold"] = c.threshold;
  meta["embedding_backend"] = "hashing_svd";
  meta["hash_n_features"] = c.projection ? kFeatures : kD;  // no projection: hash into kD buckets, all kept
  meta["hash_n_components"] = kD;
  meta["hash_seed"] = 7;  // not the loader's default, so a dropped key shows
  meta["n_train_samples"] = kTrain;
  meta["n_classes"] = kK;
  if (c.train_seconds) meta["train_seconds"] = 1.25;
  if (c.projection) {
    std::vector<double> proj(static_cast<std::size_t>(kD * kFeatures));
    for (double& v : proj) v = nproj(rng);
    meta["projection"] = proj;
  }
  const fs::path json = dir / "branch_a_router.json";
  std::ofstream(json) << meta.dump(2) << '\n';
  return json;
}

bool run_config(const Config& c, const fs::path& base, const std::vector<std::string>& texts) {
  const fs::path dir = base / c.name;
  fs::remove_all(dir);
  bool ok = true;
  auto fail = [&](const std::string& msg) {
    std::cerr << c.name << ": " << msg << "\n";
    ok = false;
  };

  // The probes must reach more than one label, else a lost class would pass unseen: take the first scale (a fixed
  // list, so the test stays deterministic) at which they do.
  fs::path src_json;
  std::size_t n_labels = 0;
  for (double scale : {0.1, 1.0, 0.3, 3.0, 0.03}) {
    fs::remove_all(dir / "src");
    src_json = write_checkpoint(c, dir / "src", scale);
    try {
      cypha::BranchARouter probe;
      probe.load_checkpoint(src_json.string());
      std::vector<std::string> seen;
      for (const auto& t : texts) {
        const auto r = probe.route(t);
        if (std::find(seen.begin(), seen.end(), r.label) == seen.end()) seen.push_back(r.label);
      }
      n_labels = seen.size();
    } catch (const std::exception&) {
      break;  // a checkpoint that does not load fails below, with its stage
    }
    if (n_labels >= 2) break;
  }

  std::string stage = "loading the source checkpoint";
  try {
  cypha::BranchARouter a;
  a.load_checkpoint(src_json.string());

  if (n_labels < 2) fail("the probes reach only " + std::to_string(n_labels) + " label(s)");

  stage = "saving";
  const std::string j1 = a.save_checkpoint((dir / "s1").string());
  stage = "loading the saved checkpoint";
  cypha::BranchARouter b;
  b.load_checkpoint(j1);
  stage = "comparing";
  if (auto d = route_diff(a, b, texts); !d.empty()) fail("routes differ after a reload: " + d);
  for (const auto& n : state_diff(a, b)) fail("state differs after a reload: " + n);
  if (summary_core(a) != summary_core(b)) fail("summary differs after a reload");

  // The saved JSON keeps every key the loader reads (a zero std is saved as the 1 the loader set).
  {
    nlohmann::json src, sav;
    std::ifstream(src_json) >> src;
    std::ifstream(j1) >> sav;
    auto stdv = src["std"].get<std::vector<double>>();
    for (double& v : stdv) {
      if (std::abs(v) < 1e-8) v = 1.0;
    }
    src["std"] = stdv;
    for (const char* k : {"mean", "std", "epistemic_threshold", "embedding_backend", "hash_n_features",
                          "hash_n_components", "hash_seed", "n_train_samples", "train_seconds", "projection"}) {
      if (src.contains(k) != sav.contains(k) || (src.contains(k) && src[k] != sav[k])) {
        fail(std::string("saved JSON loses or changes ") + k);
      }
    }
  }
  stage = "saving again";
  const std::string j2 = b.save_checkpoint((dir / "s2").string());
  for (const char* f : {"branch_a_router.json", "router_model.cypha"}) {
    if (read_bytes(dir / "s1" / f) != read_bytes(dir / "s2" / f)) fail(std::string("second save differs: ") + f);
  }
  stage = "loading the second save";
  cypha::BranchARouter cr;
  cr.load_checkpoint(j2);
  if (auto d = route_diff(a, cr, texts); !d.empty()) fail("routes differ after a second reload: " + d);
  for (const auto& n : state_diff(a, cr)) fail("state differs after a second reload: " + n);

  } catch (const std::exception& e) {
    fail("exception while " + stage + ": " + e.what());
  }
  std::cout << c.name << ": " << (ok ? "ok" : "FAILED") << "\n";
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const fs::path base = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "branch_a_router_save_load_test";
  const auto texts = probe_texts();
  const std::vector<Config> configs = {
      {"embedded", true, false, false, true, 0.5},
      {"ff_json", true, true, false, true, 0.5},
      {"no_projection", false, false, false, false, 0.9},
      {"format1", true, false, true, true, 0.3},
      {"ff_json_format1", false, true, true, false, 0.7},
  };
  int failed = 0;
  for (const auto& c : configs) {
    try {
      if (!run_config(c, base, texts)) ++failed;
    } catch (const std::exception& e) {
      std::cerr << c.name << ": exception: " << e.what() << "\n";
      std::cout << c.name << ": FAILED\n";
      ++failed;
    }
  }
  std::cout << "branch_a_router_save_load_test: " << configs.size() << " configurations, " << failed << " failed\n";
  return failed == 0 ? 0 : 1;
}
