#include "bulk_train_worker.h"

#include <QMetaType>
#include <QVector>

#include "cypha/curriculum.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/train_step_vector.hpp"

namespace {

constexpr double kGhNigAdaptAlphaShell = 0.98;

bool cancelled(const BulkNativeTrainJob& job, const std::atomic<bool>& worker_cancel) {
  return worker_cancel.load(std::memory_order_relaxed) ||
         (job.cancel_flag != nullptr && job.cancel_flag->load(std::memory_order_relaxed));
}

std::vector<int> compute_train_order(const BulkNativeTrainJob& job, const std::atomic<bool>& worker_cancel) {
  std::vector<int> order(static_cast<std::size_t>(job.train_n));
  for (int i = 0; i < job.train_n; ++i) {
    order[static_cast<std::size_t>(i)] = i;
  }
  if ((!job.sort_by_uncertainty && !job.curriculum) || job.model == nullptr || job.train_n <= 0) {
    return order;
  }

  struct RankRow {
    int index;
    double entropy;
    double confidence;
  };
  std::vector<RankRow> ranked;
  ranked.reserve(static_cast<std::size_t>(job.train_n));

  cypha::CsvDenseChunkReader reader(job.csv_path, job.csv_spec, job.train_n, job.chunk_rows);
  cypha::CsvDenseResult chunk;
  int global_start = 0;
  const int k = static_cast<int>(job.model->labels.size());
  const double eps = 1e-8;
  const double T = job.model->temperature;

  while (reader.read_chunk(chunk, global_start)) {
    if (cancelled(job, worker_cancel)) {
      break;
    }

    std::vector<double> x_latent;
    const int chunk_train_rows = std::min(chunk.n_rows, job.train_n - global_start);
    if (chunk_train_rows <= 0) {
      continue;
    }
    x_latent.reserve(static_cast<std::size_t>(chunk_train_rows) * static_cast<std::size_t>(job.model->d_latent));

    for (int i = 0; i < chunk_train_rows; ++i) {
      std::vector<double> x(chunk.n_features);
      const std::size_t row_base =
          static_cast<std::size_t>(i) * static_cast<std::size_t>(chunk.n_features);
      for (int j = 0; j < chunk.n_features; ++j) {
        x[static_cast<std::size_t>(j)] = chunk.x_rowmajor[row_base + static_cast<std::size_t>(j)];
      }
      if (job.pre != nullptr) {
        x = job.pre->transform_one(x);
      }
      if (static_cast<int>(x.size()) != job.model->d_latent) {
        throw std::runtime_error("uncertainty rank: input dim mismatch after preprocessor");
      }
      x_latent.insert(x_latent.end(), x.begin(), x.end());
    }

    std::vector<double> H;
    cypha::batch_encode(*job.model, x_latent.data(), chunk_train_rows, H);
    std::vector<double> llr;
    cypha::score_matrix_use_field(*job.model, H.data(), chunk_train_rows, llr, job.kernel_mem, job.use_kernel_llr,
                                  job.kernel_blend);

    std::vector<double> z(static_cast<std::size_t>(chunk_train_rows) * static_cast<std::size_t>(k));
    for (int i = 0; i < chunk_train_rows * k; ++i) {
      z[static_cast<std::size_t>(i)] = llr[static_cast<std::size_t>(i)] / (T + eps);
    }
    std::vector<double> probs;
    cypha::softmax_batch_reference(z.data(), chunk_train_rows, k, eps, probs);

    for (int i = 0; i < chunk_train_rows; ++i) {
      const int global_row = global_start + i;
      const double* prow = probs.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(k);
      ranked.push_back({global_row, cypha::row_entropy_from_probs(prow, k, eps),
                        cypha::row_max_softmax_confidence(prow, k)});
    }
  }

  if (static_cast<int>(ranked.size()) != job.train_n) {
    throw std::runtime_error("uncertainty rank: row count mismatch");
  }

  if (job.curriculum) {
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

  for (std::size_t i = 0; i < ranked.size(); ++i) {
    order[i] = ranked[i].index;
  }
  return order;
}

void train_one_row(BulkNativeTrainJob& job, const cypha::CsvDenseResult& chunk, int chunk_row, int& total_steps,
                   double& ood_sigma, double& llr_ema, double& ema_loss, int& win_total, int& win_correct,
                   double& gh_chi, double& gh_psi, int& enc_updates, std::mt19937& rng,
                   BulkNativeTrainLogEntry& log_entry) {
  std::vector<double> x_raw(static_cast<std::size_t>(chunk.n_features));
  const std::size_t row_base =
      static_cast<std::size_t>(chunk_row) * static_cast<std::size_t>(chunk.n_features);
  for (int j = 0; j < chunk.n_features; ++j) {
    x_raw[static_cast<std::size_t>(j)] = chunk.x_rowmajor[row_base + static_cast<std::size_t>(j)];
  }

  std::vector<double> x_lat = x_raw;
  if (job.pre != nullptr) {
    x_lat = job.pre->transform_one(x_raw);
  }

  const std::string yl = chunk.y_class[static_cast<std::size_t>(chunk_row)];
  cypha::MemoryTrainMeta meta{};

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

  double loss = 0.0;
  if (job.use_gh && static_cast<int>(job.gh_inv_v.size()) == job.model->d_latent) {
    const auto gh = cypha::dif_gh_train_step_vector(
        *job.model, *job.mem, *job.replay, x_lat.data(), job.model->d_latent, yl, job.gh_inv_v, job.gh_R_base,
        gh_chi, gh_psi, kGhNigAdaptAlphaShell, job.world_lr, job.delta_lr, ood_sigma, job.tsp, rng, enc_updates,
        &meta, &extras);
    loss = gh.loss;
    gh_chi = gh.chi_new;
    gh_psi = gh.psi_new;
  } else {
    loss = cypha::dif_train_step_vector(*job.model, *job.mem, *job.replay, x_lat.data(), job.model->d_latent, yl,
                                        job.world_lr, job.delta_lr, job.world_lr, job.delta_lr, ood_sigma, job.tsp,
                                        rng, enc_updates, &meta, &extras);
  }
  if (meta.correct) {
    job.model->total_correct += 1;
  }

  if (win_total < 200) {
    ++win_total;
  }
  win_correct = static_cast<int>(win_correct * (win_total == 200 ? 199.0 / 200.0 : 1.0) + (meta.correct ? 1 : 0));
  ema_loss = 0.97 * ema_loss + 0.03 * loss;
  ++total_steps;

  log_entry = {total_steps, QString::fromStdString(yl), loss, meta.correct};
}

void append_val_row(const cypha::CsvDenseResult& chunk, int chunk_row, cypha::CsvDenseResult& val_holdout) {
  if (val_holdout.n_features == 0) {
    val_holdout.n_features = chunk.n_features;
  }
  const std::size_t row_base =
      static_cast<std::size_t>(chunk_row) * static_cast<std::size_t>(chunk.n_features);
  for (int j = 0; j < chunk.n_features; ++j) {
    val_holdout.x_rowmajor.push_back(chunk.x_rowmajor[row_base + static_cast<std::size_t>(j)]);
  }
  val_holdout.y_class.push_back(chunk.y_class[static_cast<std::size_t>(chunk_row)]);
  ++val_holdout.n_rows;
}

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

  if (job.model == nullptr || job.mem == nullptr || job.replay == nullptr || job.train_n <= 0 ||
      job.csv_path.empty()) {
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
    const std::vector<int> train_order = compute_train_order(job, cancel_);
    int trained = 0;
    int next_want = train_order[static_cast<std::size_t>(trained)];

    cypha::CsvDenseChunkReader reader(job.csv_path, job.csv_spec, job.total_rows, job.chunk_rows);
    cypha::CsvDenseResult chunk;
    int global_start = 0;
    while (reader.read_chunk(chunk, global_start)) {
      for (int i = 0; i < chunk.n_rows; ++i) {
        const int global_row = global_start + i;
        if (global_row >= job.total_rows) {
          break;
        }
        if (cancelled(job, cancel_)) {
          result.cancelled = true;
          break;
        }

        if (global_row >= job.train_n) {
          append_val_row(chunk, i, result.val_holdout);
          continue;
        }

        if (trained < job.train_n && global_row == next_want) {
          BulkNativeTrainLogEntry entry;
          train_one_row(job, chunk, i, total_steps, ood_sigma, llr_ema, ema_loss, win_total, win_correct, gh_chi,
                        gh_psi, enc_updates, rng, entry);
          log.append(entry);
          ++trained;

          const double roll_acc =
              (win_total > 0) ? 100.0 * static_cast<double>(win_correct) / static_cast<double>(win_total) : 0.0;

          emit progress(trained, job.train_n);
          emit lossReported(total_steps, entry.loss);
          emit valAccReported(total_steps, roll_acc);
          emit stepLogged(total_steps, entry.loss, entry.correct);

          if (trained < job.train_n) {
            next_want = train_order[static_cast<std::size_t>(trained)];
          }
        }
      }
      if (result.cancelled) {
        break;
      }
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
