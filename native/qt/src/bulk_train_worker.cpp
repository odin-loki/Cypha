#include "bulk_train_worker.h"

#include <QMetaType>
#include <QVector>

#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/train_step_vector.hpp"

namespace {

constexpr double kGhNigAdaptAlphaShell = 0.98;

}  // namespace

Q_DECLARE_METATYPE(BulkNativeTrainResult)
Q_DECLARE_METATYPE(QVector<BulkNativeTrainLogEntry>)

BulkNativeTrainWorker::BulkNativeTrainWorker(QObject* parent) : QObject(parent) {
  static const bool registered = [] {
    qRegisterMetaType<BulkNativeTrainResult>("BulkNativeTrainResult");
    qRegisterMetaType<QVector<BulkNativeTrainLogEntry>>("QVector<BulkNativeTrainLogEntry>");
    return true;
  }();
  (void)registered;
}

void BulkNativeTrainWorker::requestCancel() { cancel_.store(true, std::memory_order_relaxed); }

void BulkNativeTrainWorker::run(BulkNativeTrainJob job) {
  cancel_.store(false, std::memory_order_relaxed);
  if (job.cancel_flag != nullptr) {
    job.cancel_flag->store(false, std::memory_order_relaxed);
  }

  BulkNativeTrainResult result{};
  result.total_steps = job.total_steps_start;
  result.ema_loss = job.ema_loss_start;
  result.llr_ema = job.llr_ema_start;
  result.win_total = job.win_total_start;
  result.win_correct = job.win_correct_start;
  result.gh_chi = job.gh_chi_start;
  result.gh_psi = job.gh_psi_start;
  result.enc_updates = job.enc_updates_start;

  QVector<BulkNativeTrainLogEntry> log;
  log.reserve(job.train_n);

  if (job.model == nullptr || job.mem == nullptr || job.replay == nullptr || job.train_n <= 0) {
    emit error(QStringLiteral("Native training state not ready."));
    emit finished(false, result, {});
    return;
  }

  double ood_sigma = job.ood_sigma;
  int total_steps = job.total_steps_start;
  double llr_ema = job.llr_ema_start;
  double ema_loss = job.ema_loss_start;
  int win_total = job.win_total_start;
  int win_correct = job.win_correct_start;
  double gh_chi = job.gh_chi_start;
  double gh_psi = job.gh_psi_start;
  int enc_updates = job.enc_updates_start;
  std::mt19937 rng = job.rng;

  try {
    for (int i = 0; i < job.train_n; ++i) {
      if (cancel_.load(std::memory_order_relaxed) ||
          (job.cancel_flag != nullptr && job.cancel_flag->load(std::memory_order_relaxed))) {
        result.cancelled = true;
        break;
      }

      std::vector<double> x_raw(static_cast<std::size_t>(job.data.n_features));
      const std::size_t row_base =
          static_cast<std::size_t>(i) * static_cast<std::size_t>(job.data.n_features);
      for (int j = 0; j < job.data.n_features; ++j) {
        x_raw[static_cast<std::size_t>(j)] =
            job.data.x_rowmajor[row_base + static_cast<std::size_t>(j)];
      }

      std::vector<double> x_lat = x_raw;
      if (job.pre != nullptr) {
        x_lat = job.pre->transform_one(x_raw);
      }

      const std::string yl = job.data.y_class[static_cast<std::size_t>(i)];
      cypha::MemoryTrainMeta meta{};
      double loss = 0.0;

      cypha::TrainStepExtras extras{};
      extras.total_steps = &total_steps;
      extras.ood_sigma = &ood_sigma;
      extras.llr_ema = &llr_ema;
      std::vector<double> ru = job.replay_u01;
      std::size_t ru_pos = 0;
      if (!ru.empty()) {
        extras.replay_u01 = ru.data();
        extras.replay_u01_len = ru.size();
        extras.replay_u01_pos = &ru_pos;
      }
      if (job.use_kernel_llr && job.kernel_mem != nullptr) {
        extras.kernel_mem = job.kernel_mem;
        extras.use_kernel_llr = true;
        extras.kernel_blend = job.kernel_blend;
      }

      if (job.use_gh && static_cast<int>(job.gh_inv_v.size()) == job.model->d_latent) {
        const auto gh = cypha::dif_gh_train_step_vector(
            *job.model, *job.mem, *job.replay, x_lat.data(), job.model->d_latent, yl, job.gh_inv_v,
            job.gh_R_base, gh_chi, gh_psi, kGhNigAdaptAlphaShell, job.world_lr, job.delta_lr,
            ood_sigma, job.tsp, rng, enc_updates, &meta, &extras);
        loss = gh.loss;
        gh_chi = gh.chi_new;
        gh_psi = gh.psi_new;
      } else {
        loss = cypha::dif_train_step_vector(
            *job.model, *job.mem, *job.replay, x_lat.data(), job.model->d_latent, yl, job.world_lr,
            job.delta_lr, job.world_lr, job.delta_lr, ood_sigma, job.tsp, rng, enc_updates, &meta,
            &extras);
      }
      if (meta.correct) {
        job.model->total_correct += 1;
      }

      if (win_total < 200) {
        ++win_total;
      }
      win_correct = static_cast<int>(win_correct * (win_total == 200 ? 199.0 / 200.0 : 1.0) +
                                     (meta.correct ? 1 : 0));
      ema_loss = 0.97 * ema_loss + 0.03 * loss;
      ++total_steps;

      const double roll_acc =
          (win_total > 0) ? 100.0 * static_cast<double>(win_correct) / static_cast<double>(win_total)
                          : 0.0;

      log.append({total_steps, QString::fromStdString(yl), loss, meta.correct});

      emit progress(i + 1, job.train_n);
      emit lossReported(total_steps, loss);
      emit valAccReported(total_steps, roll_acc);
      emit stepLogged(total_steps, loss, meta.correct);
    }
  } catch (const std::exception& ex) {
    emit error(QString::fromUtf8(ex.what()));
    emit finished(false, result, log);
    return;
  }

  result.total_steps = total_steps;
  result.ema_loss = ema_loss;
  result.llr_ema = llr_ema;
  result.win_total = win_total;
  result.win_correct = win_correct;
  result.gh_chi = gh_chi;
  result.gh_psi = gh_psi;
  result.enc_updates = enc_updates;
  result.steps_completed = log.size();

  emit finished(!result.cancelled, result, log);
}
