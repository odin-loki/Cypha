#include "cypha/train_step_vector.hpp"

#include <limits>
#include <stdexcept>

#include "cypha/encoder_contrastive.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/nig_field.hpp"
#include "cypha/sync_infer.hpp"

#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha {

namespace {

constexpr double kDeliberateLo = 0.25;
constexpr double kDeliberateHi = 0.40;
constexpr double kOodEma = 0.01;
constexpr double kLlrScaleAlpha = 0.002;

void collect_nonempty_deltas(const CyphaDifMemoryState& mem, std::vector<std::vector<double>>& out) {
  const int d = mem.d_latent;
  const int K = static_cast<int>(mem.labels.size());
  constexpr double thr = 1e-16;
  for (int k = 0; k < K; ++k) {
    double n2 = 0.0;
    for (int j = 0; j < d; ++j) {
      double v = mem.D[static_cast<std::size_t>(k * d + j)];
      n2 += v * v;
    }
    if (n2 <= thr) {
      continue;
    }
    std::vector<double> row(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      row[static_cast<std::size_t>(j)] = mem.D[static_cast<std::size_t>(k * d + j)];
    }
    out.push_back(std::move(row));
  }
}

}  // namespace

double dif_train_step_vector(CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay,
                             const double* x_preprocessed, int d, const std::string& y_label,
                             double world_lr_step, double delta_lr_step, double world_lr_replay,
                             double delta_lr_replay, double ood_sigma, const TrainStepParams& tsp,
                             std::mt19937& rng, int& enc_update_count, MemoryTrainMeta* meta_out,
                             TrainStepExtras* extras) {
  if (d != infer.d_latent || d != mem.d_latent) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::vector<double> H;
  batch_encode(infer, x_preprocessed, 1, H);

  // Match Python `train_step`: `classes` includes `label` before `memory.train` when the class is new.
  std::vector<std::string> classes_for_ctx = mem.labels;
  {
    bool have = false;
    for (const auto& lb : classes_for_ctx) {
      if (lb == y_label) {
        have = true;
        break;
      }
    }
    if (!have) {
      classes_for_ctx.push_back(y_label);
    }
  }
  std::vector<double> ctx_vec;
  context_prior_for_labels(infer, classes_for_ctx, ctx_vec);
  std::unordered_map<std::string, double> ctx_map;
  for (std::size_t i = 0; i < classes_for_ctx.size(); ++i) {
    ctx_map[classes_for_ctx[i]] = ctx_vec[i];
  }

  MemoryTrainMeta meta_local;
  MemoryTrainMeta* meta = meta_out != nullptr ? meta_out : &meta_local;
  double loss = mem.memory_train(H.data(), y_label, infer.field_h.data(), ctx_map, infer.temperature,
                                  ood_sigma, world_lr_step, delta_lr_step, meta);
  sync_infer_model_from_memory(infer, mem);

  if (extras != nullptr && extras->use_kernel_llr && extras->kernel_mem != nullptr) {
    extras->kernel_mem->update(H.data(), y_label, classes_for_ctx, delta_lr_step);
  }

  replay.push(H.data(), x_preprocessed, d, y_label, loss);

  const std::string& pred = meta->pred_label;
  if (!meta->correct && pred != "__unknown__") {
    std::vector<double> mu_k;
    std::vector<double> v_k;
    std::vector<double> mu_j;
    std::vector<double> v_j;
    if (mem.class_mean_and_variance(y_label, mu_k, v_k) &&
        mem.class_mean_and_variance(pred, mu_j, v_j)) {
      contrastive_update_encoder_w(infer.enc_w, d, x_preprocessed, H.data(), mu_k.data(), v_k.data(),
                                   mu_j.data(), v_j.data(), 1.0, tsp.enc_lr, enc_update_count);
    }
  }

  if (meta->post_conf > kDeliberateLo && meta->post_conf < kDeliberateHi && !meta->llr_rank1.empty()) {
    std::vector<double> mu_a;
    std::vector<double> v_a;
    std::vector<double> mu_b;
    std::vector<double> v_b;
    if (mem.class_mean_and_variance(meta->llr_rank0, mu_a, v_a) &&
        mem.class_mean_and_variance(meta->llr_rank1, mu_b, v_b)) {
      contrastive_update_encoder_w(infer.enc_w, d, x_preprocessed, H.data(), mu_a.data(), v_a.data(),
                                   mu_b.data(), v_b.data(), 0.3, tsp.enc_lr, enc_update_count);
    }
  }

  // Match Python `CyphaDIF.train_step`: dedup after memory.train + encoder updates, before replay sampling.
  if (extras != nullptr && extras->total_steps != nullptr && (*extras->total_steps % 5 == 0)) {
    mem.dedup_check(y_label);
  }

  if (replay.size() >= 10 && tsp.replay_ratio > 0.0) {
    double gate_u = 0.0;
    if (extras != nullptr && extras->replay_u01 != nullptr && extras->replay_u01_pos != nullptr) {
      if (*(extras->replay_u01_pos) >= extras->replay_u01_len) {
        throw std::runtime_error("dif_train_step_vector: replay_u01 exhausted (gate)");
      }
      gate_u = extras->replay_u01[(*extras->replay_u01_pos)++];
    } else {
      std::uniform_real_distribution<double> u01(0.0, 1.0);
      gate_u = u01(rng);
    }
    if (gate_u < tsp.replay_ratio) {
      int ns = std::min(4, replay.size());
      std::vector<std::vector<double>> rh;
      std::vector<std::string> rl;
      const double* fu = (extras != nullptr) ? extras->replay_u01 : nullptr;
      std::size_t* fp = (extras != nullptr) ? extras->replay_u01_pos : nullptr;
      std::size_t fl = (extras != nullptr) ? extras->replay_u01_len : 0;
      replay.sample(ns, rng, rh, rl, fu, fp, fl);
      for (std::size_t t = 0; t < rh.size(); ++t) {
        context_prior_for_labels(infer, mem.labels, ctx_vec);
        ctx_map.clear();
        for (std::size_t i = 0; i < mem.labels.size(); ++i) {
          ctx_map[mem.labels[i]] = ctx_vec[i];
        }
        mem.memory_train(rh[t].data(), rl[t], infer.field_h.data(), ctx_map, infer.temperature, ood_sigma,
                         world_lr_replay, delta_lr_replay, nullptr);
        sync_infer_model_from_memory(infer, mem);
      }
    }
  }

  if (extras != nullptr && extras->total_steps != nullptr) {
    *extras->total_steps += 1;

    if (tsp.align_every > 0 && (*extras->total_steps % tsp.align_every == 0)) {
      std::vector<std::vector<double>> deltas;
      collect_nonempty_deltas(mem, deltas);
      encoder_align_to_offsets(infer.enc_w, d, deltas);
    }

    if (!infer.field_w_t.empty() && static_cast<int>(infer.field_a_eff.size()) == infer.field_dim * infer.field_dim) {
      std::vector<double> signal;
      if (latent_to_field_signal(H.data(), d, infer.w_inject, infer.field_dim, signal)) {
        nig_field_inject(infer.field_h, signal.data(), infer.field_dim, 0.05);
        std::vector<double> h_old = infer.field_h;
        nig_field_evolve(infer.field_a_eff, infer.field_dim, h_old.data(), infer.field_h);
        infer.field_step += 1;
        if (*extras->total_steps % 50 == 0) {
          nig_field_update_causal(infer.field_w_t, infer.field_dim, h_old.data(), infer.field_h.data(), 0.0002,
                                  infer.field_sr_vec, infer.field_a_eff);
        }
      }
    }

    context_record_step(infer, y_label, meta->correct);

    if (!mem.labels.empty()) {
      double win = std::abs(meta->post_llr_max);
      infer.llr_scale_ema = (1.0 - kLlrScaleAlpha) * infer.llr_scale_ema + kLlrScaleAlpha * win;
      infer.llr_scale_n += 1;
    }

    if (tsp.temp_recalib_every > 0 && (*extras->total_steps % tsp.temp_recalib_every == 0)) {
      auto_recalibrate_temperature(infer);
    }

    if (*extras->total_steps % 20 == 0 && extras->ood_sigma != nullptr && extras->llr_ema != nullptr) {
      context_prior_for_labels(infer, mem.labels, ctx_vec);
      ctx_map.clear();
      for (std::size_t i = 0; i < mem.labels.size(); ++i) {
        ctx_map[mem.labels[i]] = ctx_vec[i];
      }
      double mx = memory_max_classify_llr(mem, H.data(), infer.field_h.data(), ctx_map);
      *extras->llr_ema = (1.0 - kOodEma) * (*extras->llr_ema) + kOodEma * mx;
      *extras->ood_sigma = std::max(1.0, std::abs(*extras->llr_ema));

      double mahal_c = memory_mahal_world_scalar(mem, H.data());
      double prev_ema =
          infer.has_mahal_ema && std::isfinite(infer.mahal_ema) ? infer.mahal_ema : 1.0;
      double prev_var = infer.mahal_std_ema * infer.mahal_std_ema;
      infer.has_mahal_ema = true;
      infer.mahal_ema = (1.0 - kOodEma) * prev_ema + kOodEma * mahal_c;
      double mahal_var = (1.0 - kOodEma) * prev_var + kOodEma * (mahal_c - prev_ema) * (mahal_c - prev_ema);
      infer.mahal_std_ema = std::max(std::sqrt(std::max(mahal_var, 0.0)), 0.05);
    }
  }

  return loss;
}

GhTrainStepResult dif_gh_train_step_vector(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay, const double* x_preprocessed, int d,
    const std::string& y_label, const std::vector<double>& gh_inv_v_clean, double gh_r_base, double chi, double psi,
    double nig_alpha, double world_lr_nominal, double delta_lr_nominal, double ood_sigma, const TrainStepParams& tsp,
    std::mt19937& rng, int& enc_update_count, MemoryTrainMeta* meta_out, TrainStepExtras* extras) {
  GhTrainStepResult out{};
  if (static_cast<int>(gh_inv_v_clean.size()) != d || d != infer.d_latent || d != mem.d_latent) {
    return out;
  }
  std::vector<double> H;
  batch_encode(infer, x_preprocessed, 1, H);
  double mahal_sq = 0.0;
  for (int j = 0; j < d; ++j) {
    double dj = H[static_cast<std::size_t>(j)] - mem.world_mu[static_cast<std::size_t>(j)];
    mahal_sq += dj * dj * gh_inv_v_clean[static_cast<std::size_t>(j)];
  }
  mahal_sq /= static_cast<double>(std::max(d, 1));
  out.r_eff = nig_R_eff_gh(mahal_sq, gh_r_base, chi, psi);
  const double gh_scale = gh_train_lr_scale(mahal_sq, gh_r_base, chi, psi);
  double wlr = world_lr_nominal * gh_scale;
  double dlr = delta_lr_nominal * gh_scale;
  TrainStepParams tsp_run = tsp;
  tsp_run.enc_lr *= gh_scale;
  out.loss = dif_train_step_vector(infer, mem, replay, x_preprocessed, d, y_label, wlr, dlr, wlr, dlr, ood_sigma,
                                   tsp_run, rng, enc_update_count, meta_out, extras);
  auto adapted = nig_adapt_session_chi(chi, psi, mahal_sq, gh_r_base, nig_alpha);
  out.chi_new = adapted.first;
  out.psi_new = adapted.second;
  return out;
}

std::vector<double> dif_train_classify_sequence(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay,
    const std::vector<std::vector<double>>& xs, const std::vector<std::string>& labels, double world_lr_step,
    double delta_lr_step, double world_lr_replay, double delta_lr_replay, double ood_sigma, const TrainStepParams& tsp,
    std::mt19937& rng, int& enc_update_count, TrainStepExtras* extras) {
  if (xs.size() != labels.size()) {
    throw std::invalid_argument("dif_train_classify_sequence: xs/labels size mismatch");
  }
  std::vector<double> losses;
  losses.reserve(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const int d = static_cast<int>(xs[i].size());
    if (d <= 0) {
      throw std::invalid_argument("dif_train_classify_sequence: empty x row");
    }
    if (d != infer.d_latent || d != mem.d_latent) {
      throw std::invalid_argument("dif_train_classify_sequence: x dim vs model mismatch");
    }
    double loss = dif_train_step_vector(infer, mem, replay, xs[i].data(), d, labels[i], world_lr_step, delta_lr_step,
                                        world_lr_replay, delta_lr_replay, ood_sigma, tsp, rng, enc_update_count,
                                        nullptr, extras);
    losses.push_back(loss);
  }
  return losses;
}

std::vector<double> dif_gh_train_classify_sequence(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay,
    const std::vector<std::vector<double>>& xs, const std::vector<std::string>& labels,
    const std::vector<double>& gh_inv_v_clean, double gh_r_base, double& chi, double& psi, double nig_alpha,
    double world_lr_nominal, double delta_lr_nominal, double ood_sigma, const TrainStepParams& tsp,
    std::mt19937& rng, int& enc_update_count, TrainStepExtras* extras) {
  if (xs.size() != labels.size()) {
    throw std::invalid_argument("dif_gh_train_classify_sequence: xs/labels size mismatch");
  }
  std::vector<double> losses;
  losses.reserve(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const int d = static_cast<int>(xs[i].size());
    if (d <= 0) {
      throw std::invalid_argument("dif_gh_train_classify_sequence: empty x row");
    }
    if (d != infer.d_latent || d != mem.d_latent) {
      throw std::invalid_argument("dif_gh_train_classify_sequence: x dim vs model mismatch");
    }
    if (static_cast<int>(gh_inv_v_clean.size()) != d) {
      throw std::invalid_argument("dif_gh_train_classify_sequence: gh_inv_v_clean dim mismatch");
    }
    GhTrainStepResult step = dif_gh_train_step_vector(
        infer, mem, replay, xs[i].data(), d, labels[i], gh_inv_v_clean, gh_r_base, chi, psi, nig_alpha,
        world_lr_nominal, delta_lr_nominal, ood_sigma, tsp, rng, enc_update_count, nullptr, extras);
    chi = step.chi_new;
    psi = step.psi_new;
    losses.push_back(step.loss);
  }
  return losses;
}

}  // namespace cypha
