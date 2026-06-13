#pragma once

#include <QObject>
#include <QString>

#include <atomic>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "cypha/csv_ingest.hpp"
#include "cypha/mke_scalar_train_step.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace cypha {
class CyphaDifMemoryState;
class CyphaInferModel;
}  // namespace cypha

/// Parameters for one bulk MKE regression train run (snapshotted on the main thread).
struct BulkMkeTrainJob {
  cypha::CsvDenseResult data;
  int n_rows{0};

  cypha::CyphaInferModel* model{nullptr};
  cypha::CyphaDifMemoryState* mem{nullptr};
  cypha::ReplayBuffer* replay{nullptr};
  cypha::PreprocessorState* pre{nullptr};
  std::unordered_map<std::string, std::vector<double>>* w_by_label{nullptr};
  std::unordered_map<std::string, std::vector<double>>* p_by_label{nullptr};

  double forgetting_factor{0.99};
  double pi_floor{1e-4};
  cypha::TrainStepParams tsp{};
  double world_lr{0.008};
  double delta_lr{0.05};
  double ood_sigma{15.0};
  std::mt19937 rng{424242u};
  int enc_updates_start{0};
  int* enc_updates{nullptr};
  int* total_steps{nullptr};
  QString router_train_label;

  std::atomic<bool>* cancel_flag{nullptr};
};

/// Runs ``mke_scalar_train_step_from_phi`` per CSV row off the GUI thread.
class BulkMkeTrainWorker final : public QObject {
  Q_OBJECT

 public:
  explicit BulkMkeTrainWorker(QObject* parent = nullptr);

 public slots:
  void run(BulkMkeTrainJob job);
  void requestCancel();

 signals:
  void progress(int step, int total);
  void finished(bool success, int n_ok, double mse, double rmse, bool cancelled, QString error);

 private:
  std::atomic<bool> cancel_{false};
};
