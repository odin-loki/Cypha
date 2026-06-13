#include "bulk_mke_train_worker.h"

#include <cmath>

BulkMkeTrainWorker::BulkMkeTrainWorker(QObject* parent) : QObject(parent) {}

void BulkMkeTrainWorker::requestCancel() { cancel_.store(true, std::memory_order_relaxed); }

void BulkMkeTrainWorker::run(BulkMkeTrainJob job) {
  cancel_.store(false, std::memory_order_relaxed);
  if (job.cancel_flag != nullptr) {
    job.cancel_flag->store(false, std::memory_order_relaxed);
  }

  if (job.model == nullptr || job.mem == nullptr || job.replay == nullptr || job.w_by_label == nullptr ||
      job.p_by_label == nullptr || job.n_rows <= 0) {
    emit finished(false, 0, 0.0, 0.0, false, QStringLiteral("MKE bulk: invalid job."));
    return;
  }

  const int d = job.model->d_latent;
  double sum_err_sq = 0.0;
  int n_ok = 0;

  const std::string router_str = job.router_train_label.toStdString();
  const std::string* router_override = job.router_train_label.isEmpty() ? nullptr : &router_str;
  int enc_updates = job.enc_updates != nullptr ? *job.enc_updates : job.enc_updates_start;

  for (int i = 0; i < job.n_rows; ++i) {
    if (cancel_.load(std::memory_order_relaxed) ||
        (job.cancel_flag != nullptr && job.cancel_flag->load(std::memory_order_relaxed))) {
      emit finished(false, n_ok, n_ok > 0 ? sum_err_sq / n_ok : 0.0,
                    n_ok > 0 ? std::sqrt(sum_err_sq / n_ok) : 0.0, true, QString());
      return;
    }

    const double* xraw = job.data.x_rowmajor.data() + static_cast<std::size_t>(i) * job.data.n_features;
    std::vector<double> phi(xraw, xraw + job.data.n_features);
    if (job.pre != nullptr) {
      phi = job.pre->transform_one(phi);
    }
    if (static_cast<int>(phi.size()) != d) {
      emit finished(false, n_ok, 0.0, 0.0, false,
                    QStringLiteral("Dim mismatch: phi.size=%1 model.d_latent=%2").arg(phi.size()).arg(d));
      return;
    }

    const double y_target = job.data.y_regression[static_cast<std::size_t>(i)];
    cypha::TrainStepExtras extras{};
    if (job.total_steps != nullptr) {
      extras.total_steps = job.total_steps;
    }

    cypha::regression::MkeScalarTrainStepOutputs mke_out{};
    cypha::regression::mke_scalar_train_step_from_phi(
        *job.model, *job.mem, *job.replay, phi.data(), d, y_target, *job.w_by_label, *job.p_by_label, nullptr,
        job.model->temperature, job.forgetting_factor, job.pi_floor, job.tsp, job.world_lr, job.delta_lr,
        job.ood_sigma, job.rng, enc_updates, &extras, router_override, 1e-9, &mke_out);

    sum_err_sq += mke_out.err_sq;
    ++n_ok;
    emit progress(i + 1, job.n_rows);
  }

  const double mse = n_ok > 0 ? sum_err_sq / n_ok : 0.0;
  const double rmse = n_ok > 0 ? std::sqrt(mse) : 0.0;
  if (job.enc_updates != nullptr) {
    *job.enc_updates = enc_updates;
  }
  emit finished(true, n_ok, mse, rmse, false, QString());
}
