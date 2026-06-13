#pragma once

#include <QObject>
#include <QString>
#include <QVector>

#include <atomic>
#include <vector>

#include "cypha/csv_ingest.hpp"

/// Parameters for one bulk REST /update run (snapshotted on the main thread).
struct BulkRestUpdateJob {
  QString base_url;
  cypha::CsvDenseResult data;
  int n_rows{0};
  bool regression{false};
  bool use_gh{false};
  QString mke_correct_label;
  QString router_train_label;
  QVector<double> replay_u01;
  std::atomic<bool>* cancel_flag{nullptr};
};

/// Runs POST /update per CSV row off the GUI thread.
class BulkRestUpdateWorker final : public QObject {
  Q_OBJECT

 public:
  explicit BulkRestUpdateWorker(QObject* parent = nullptr);

 public slots:
  void run(BulkRestUpdateJob job);
  void requestCancel();

 signals:
  void progress(int step, int total);
  void stepLoss(int step, double loss);
  void finished(bool success, QVector<double> losses, bool cancelled, QString error);

 private:
  std::atomic<bool> cancel_{false};
};
