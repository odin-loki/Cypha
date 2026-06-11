#include "cypha/multilabel_dif.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "cypha/create_model.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/nig_field.hpp"
#include "cypha/sync_infer.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;

void score_matrix_no_field(const CyphaInferModel& m, const double* h_row_major, int n,
                           std::vector<double>& llr_out) {
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  llr_out.assign(static_cast<std::size_t>(n * K), 0.0);
  if (K == 0) {
    return;
  }

  std::vector<double> ctx;
  context_prior_for_labels(m, m.labels, ctx);

  for (int i = 0; i < n; ++i) {
    const double* h = h_row_major + static_cast<std::size_t>(i * d);
    for (int k = 0; k < K; ++k) {
      double cross = 0.0;
      double d_sq = 0.0;
      for (int j = 0; j < d; ++j) {
        const double Dkj = m.D[static_cast<std::size_t>(k * d + j)];
        const double rp = (h[j] - m.mu_world[static_cast<std::size_t>(j)]) * m.inv_v[static_cast<std::size_t>(j)];
        cross += Dkj * rp;
        d_sq += Dkj * Dkj * m.inv_v[static_cast<std::size_t>(j)];
      }
      const double u_k = m.v_mean / (m.n_obs[static_cast<std::size_t>(k)] + 1.0);
      llr_out[static_cast<std::size_t>(i * K + k)] =
          cross - 0.5 * d_sq - u_k + ctx[static_cast<std::size_t>(k)];
    }
  }
}

void world_gate_no_field(const CyphaInferModel& m, const double* h_row_major, int n, std::vector<double>& gates_out) {
  const int d = m.d_latent;
  gates_out.assign(static_cast<std::size_t>(n), 1.0);
  if (d <= 0 || n <= 0) {
    return;
  }

  double r_base = m.v_mean;
  if (m.has_mahal_ema && std::isfinite(m.mahal_ema) && m.mahal_ema > kEps) {
    r_base = m.mahal_ema;
  }

  for (int i = 0; i < n; ++i) {
    const double* h = h_row_major + static_cast<std::size_t>(i * d);
    double mahal_per_dim = 0.0;
    for (int j = 0; j < d; ++j) {
      double diff = h[j] - m.mu_world[static_cast<std::size_t>(j)];
      mahal_per_dim += diff * diff * m.inv_v[static_cast<std::size_t>(j)];
    }
    mahal_per_dim /= static_cast<double>(std::max(d, 1));
    const double r_eff = nig_R_eff_gh(mahal_per_dim, r_base, 1.0, 1.0);
    gates_out[static_cast<std::size_t>(i)] = r_base / std::max(r_eff, r_base);
  }
}

int find_label_index(const std::vector<std::string>& labels, const std::string& name) {
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

std::uint32_t MultiLabelDif::label_seed(const MultiLabelDifParams& p, const std::string& label) {
  auto it = p.label_rng_seeds.find(label);
  if (it != p.label_rng_seeds.end()) {
    return it->second;
  }
  return static_cast<std::uint32_t>(std::hash<std::string>{}(label));
}

MultiLabelDif::MultiLabelDif(MultiLabelDifParams params) : params_(std::move(params)) {
  if (params_.input_dim <= 0 || params_.field_dim <= 0) {
    throw std::invalid_argument("MultiLabelDif: invalid dims");
  }
  if (params_.train.replay_cap <= 0) {
    params_.train.replay_cap = 10000;
  }
}

void MultiLabelDif::init_binary_clf(const std::string& label, BinaryClf& out) {
  FreshModelParams fp;
  fp.input_dim = params_.input_dim;
  fp.field_dim = params_.field_dim;
  fp.world_lr = params_.world_lr;
  fp.delta_lr = params_.delta_lr;
  CNode root = create_fresh_model_root(fp);

  auto enc_it = params_.initial_enc_w.find(label);
  if (enc_it != params_.initial_enc_w.end()) {
    const int d = params_.input_dim;
    if (static_cast<int>(enc_it->second.size()) != d * d) {
      throw std::runtime_error("initial_enc_w size mismatch for " + label);
    }
    for (auto& kv : root.map) {
      if (kv.first == "enc_W") {
        kv.second.tensor = enc_it->second;
        break;
      }
    }
  }
  auto patch_tensor = [&](const char* key, const std::vector<double>& data, std::uint32_t r,
                          std::uint32_t c) {
    CNode t;
    t.kind = CNode::Tensor;
    t.shape = {r, c};
    t.tensor = data;
    for (auto& kv : root.map) {
      if (kv.first == key) {
        kv.second = std::move(t);
        return;
      }
    }
    root.map.emplace_back(key, std::move(t));
  };

  auto inj_it = params_.initial_w_inject.find(label);
  if (inj_it != params_.initial_w_inject.end()) {
    const int d = params_.input_dim;
    const int fd = params_.field_dim;
    if (static_cast<int>(inj_it->second.size()) != fd * d) {
      throw std::runtime_error("initial_w_inject size mismatch for " + label);
    }
    patch_tensor("w_inject", inj_it->second, static_cast<std::uint32_t>(fd), static_cast<std::uint32_t>(d));
  }
  auto wt_it = params_.initial_field_w_t.find(label);
  if (wt_it != params_.initial_field_w_t.end()) {
    const int fd = params_.field_dim;
    if (static_cast<int>(wt_it->second.size()) != fd * fd) {
      throw std::runtime_error("initial_field_w_t size mismatch for " + label);
    }
    patch_tensor("field_W_T", wt_it->second, static_cast<std::uint32_t>(fd), static_cast<std::uint32_t>(fd));
    std::vector<float> aeff;
    recompute_field_a_eff(fd, wt_it->second, aeff);
    std::vector<double> aeff64(aeff.begin(), aeff.end());
    patch_tensor("field_a_eff", aeff64, static_cast<std::uint32_t>(fd), static_cast<std::uint32_t>(fd));
  }
  auto sr_it = params_.initial_field_sr_vec.find(label);
  if (sr_it != params_.initial_field_sr_vec.end()) {
    const int fd = params_.field_dim;
    if (static_cast<int>(sr_it->second.size()) != fd) {
      throw std::runtime_error("initial_field_sr_vec size mismatch for " + label);
    }
    CNode sr;
    sr.kind = CNode::Tensor;
    sr.shape = {static_cast<std::uint32_t>(fd)};
    sr.tensor = sr_it->second;
    for (auto& kv : root.map) {
      if (kv.first == "field_sr_vec") {
        kv.second = std::move(sr);
        break;
      }
    }
  }

  out.infer = CyphaInferModel::from_root(root, nullptr, params_.field_dim);
  // Match Python ``CyphaDIF.__init__`` warm-start (not ``create_fresh_model_root`` disk default).
  out.infer.mahal_ema = 1.0;
  out.infer.has_mahal_ema = true;
  out.mem = CyphaDifMemoryState::from_cypha_root(root, nullptr, params_.field_dim);
  out.replay = ReplayBuffer(params_.train.replay_cap);
  out.rng = std::mt19937(label_seed(params_, label));
  out.enc_update_count = 0;
  out.total_steps = 0;
}

MultiLabelDif::BinaryClf& MultiLabelDif::get_or_create(const std::string& label) {
  auto it = classifiers_.find(label);
  if (it != classifiers_.end()) {
    return it->second;
  }
  auto ins = classifiers_.emplace(label, BinaryClf(params_.train.replay_cap));
  init_binary_clf(label, ins.first->second);
  return ins.first->second;
}

std::unordered_map<std::string, double> MultiLabelDif::train_step(const double* x, int d,
                                                                  const std::unordered_map<std::string, bool>& labels) {
  if (d != params_.input_dim) {
    throw std::invalid_argument("train_step: d mismatch");
  }
  std::unordered_map<std::string, double> losses;
  for (const auto& pr : labels) {
    BinaryClf& clf = get_or_create(pr.first);
    const std::string y = pr.second ? "pos" : "neg";
    TrainStepExtras extras{};
    extras.total_steps = &clf.total_steps;
    double loss = dif_train_step_vector(clf.infer, clf.mem, clf.replay, x, d, y, params_.world_lr, params_.delta_lr,
                                        params_.world_lr, params_.delta_lr, params_.ood_sigma, params_.train, clf.rng,
                                        clf.enc_update_count, nullptr, &extras);
    losses[pr.first] = loss;
  }
  return losses;
}

std::unordered_map<std::string, double> MultiLabelDif::predict(const double* x, int d) const {
  if (d != params_.input_dim) {
    throw std::invalid_argument("predict: d mismatch");
  }
  std::unordered_map<std::string, double> result;
  for (const auto& pr : classifiers_) {
    const BinaryClf& clf = pr.second;
    std::vector<double> h;
    batch_encode(clf.infer, x, 1, h);
    CyphaInferOptions opt;
    opt.use_field = true;
    InferAtHResult ir = infer_at_h(clf.infer, h.data(), opt);
    if (ir.label == "pos") {
      result[pr.first] = ir.confidence;
    } else if (ir.label == "neg") {
      result[pr.first] = 1.0 - ir.confidence;
    } else {
      result[pr.first] = 0.5;
    }
  }
  return result;
}

std::unordered_map<std::string, std::vector<double>> MultiLabelDif::predict_batch(const double* x_row_major, int n,
                                                                                int d) const {
  if (d != params_.input_dim) {
    throw std::invalid_argument("predict_batch: d mismatch");
  }
  std::unordered_map<std::string, std::vector<double>> result;
  for (const auto& pr : classifiers_) {
    const BinaryClf& clf = pr.second;
    std::vector<double> h;
    batch_encode(clf.infer, x_row_major, n, h);
    std::vector<double> llr;
    score_matrix_no_field(clf.infer, h.data(), n, llr);
    const int K = static_cast<int>(clf.infer.labels.size());
    std::vector<double> probs;
    if (K > 0) {
      std::vector<double> scaled(static_cast<std::size_t>(n * K));
      for (int i = 0; i < n * K; ++i) {
        scaled[static_cast<std::size_t>(i)] = llr[static_cast<std::size_t>(i)] / (clf.infer.temperature + kEps);
      }
      softmax_batch_like_python(scaled.data(), n, K, kEps, probs);
    }
    const int pi = find_label_index(clf.infer.labels, "pos");
    std::vector<double> gates;
    world_gate_no_field(clf.infer, h.data(), n, gates);
    std::vector<double> p_pos(static_cast<std::size_t>(n), 0.5);
    if (pi >= 0) {
      for (int i = 0; i < n; ++i) {
        p_pos[static_cast<std::size_t>(i)] =
            probs[static_cast<std::size_t>(i * K + pi)] * gates[static_cast<std::size_t>(i)];
      }
    }
    result[pr.first] = std::move(p_pos);
  }
  return result;
}

std::vector<std::string> MultiLabelDif::labels() const {
  std::vector<std::string> out;
  out.reserve(classifiers_.size());
  for (const auto& pr : classifiers_) {
    out.push_back(pr.first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace cypha
