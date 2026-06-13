#pragma once

#include <QObject>
#include <QString>
#include <QVector>

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cypha/csv_ingest.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace cypha {
class CyphaDifMemoryState;
class CyphaInferModel;
class KernelMemory;
}  // namespace cypha

/// Parameters for one bulk-native-train run (copied/snapshotted on the main thread).
struct BulkNativeTrainJob {
  cypha::CsvDenseResult data;
  int train_n{0};

  cypha::CyphaInferModel* model{nullptr};
  cypha::CyphaDifMemoryState* mem{nullptr};
  cypha::ReplayBuffer* replay{nullptr};
  cypha::PreprocessorState* pre{nullptr};
  cypha::KernelMemory* kernel_mem{nullptr};

  bool use_gh{false};
  bool use_kernel_llr{false};
  double kernel_blend{0.5};
  double world_lr{0.008};
  double delta_lr{0.05};
  double ood_sigma{15.0};
  std::vector<double> gh_inv_v{};
  double gh_R_base{1.0};
  cypha::TrainStepParams tsp{};
  std::vector<double> replay_u01{};

  int total_steps_start{0};
  double llr_ema_start{0.0};
  double ema_loss_start{0.0};
  int win_total_start{0};
  int win_correct_start{0};
  double gh_chi_start{1.0};
  double gh_psi_start{1.0};
  int enc_updates_start{0};
  std::mt19937 rng{424242u};

  std::atomic<bool>* cancel_flag{nullptr};
};

/// Final scalar state written by the worker before `finished`.
struct BulkNativeTrainResult {
  int total_steps{0};
  double ema_loss{0.0};
  double llr_ema{0.0};
  int win_total{0};
  int win_correct{0};
  double gh_chi{1.0};
  double gh_psi{1.0};
  int enc_updates{0};
  int steps_completed{0};
  bool cancelled{false};
};

/// Per-step log row for the native train log table (batched on main thread at end).
struct BulkNativeTrainLogEntry {
  int step_n{};
  QString label;
  double loss{};
  bool correct{};
};

/// Runs `dif_train_step_vector` / GH train off the GUI thread; emits Qt signals for live UI.
class BulkNativeTrainWorker final : public QObject {
  Q_OBJECT

 public:
  explicit BulkNativeTrainWorker(QObject* parent = nullptr);

 public slots:
  void run(BulkNativeTrainJob job);
  void requestCancel();

 signals:
  void progress(int step, int total);
  void lossReported(int step, double loss);
  void valAccReported(int step, double acc_percent);
  void stepLogged(int step, double loss, bool correct);
  void finished(bool success, BulkNativeTrainResult result, QVector<BulkNativeTrainLogEntry> log);
  void error(QString message);

 private:
  std::atomic<bool> cancel_{false};
};
