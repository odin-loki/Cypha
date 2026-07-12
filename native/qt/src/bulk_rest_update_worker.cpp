#include "bulk_rest_update_worker.h"

#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaType>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

BulkRestUpdateWorker::BulkRestUpdateWorker(QObject* parent) : QObject(parent) {
  static const bool registered = [] {
    qRegisterMetaType<QVector<double>>("QVector<double>");
    return true;
  }();
  (void)registered;
}

void BulkRestUpdateWorker::requestCancel() { cancel_.store(true, std::memory_order_relaxed); }

namespace {
// Same timeout as the main-thread REST helpers in shell_main.cpp (http_post_json/http_get_json).
constexpr int kHttpTransferTimeoutMs = 15000;
}  // namespace

void BulkRestUpdateWorker::run(BulkRestUpdateJob job) {
  cancel_.store(false, std::memory_order_relaxed);
  if (job.cancel_flag != nullptr) {
    job.cancel_flag->store(false, std::memory_order_relaxed);
  }

  QVector<double> losses;
  losses.reserve(job.n_rows);

  if (job.base_url.isEmpty() || job.n_rows <= 0) {
    emit finished(false, {}, false, QStringLiteral("Bulk REST /update: invalid job."));
    return;
  }

  QString base = job.base_url.trimmed();
  while (base.endsWith(QLatin1Char('/'))) {
    base.chop(1);
  }
  const QUrl update_url(base + QStringLiteral("/update"));

  QNetworkAccessManager nam;

  for (int i = 0; i < job.n_rows; ++i) {
    if (cancel_.load(std::memory_order_relaxed) ||
        (job.cancel_flag != nullptr && job.cancel_flag->load(std::memory_order_relaxed))) {
      emit finished(false, losses, true, QString());
      return;
    }

    QJsonArray arr;
    const std::size_t row_base =
        static_cast<std::size_t>(i) * static_cast<std::size_t>(job.data.n_features);
    for (int j = 0; j < job.data.n_features; ++j) {
      arr.append(job.data.x_rowmajor[row_base + static_cast<std::size_t>(j)]);
    }
    QJsonObject body;
    body[QStringLiteral("input")] = arr;
    body[QStringLiteral("use_gh")] = job.use_gh;
    if (job.regression) {
      body[QStringLiteral("correct_label")] = job.mke_correct_label;
      body[QStringLiteral("regression_y")] = job.data.y_regression[static_cast<std::size_t>(i)];
      if (!job.router_train_label.isEmpty()) {
        body[QStringLiteral("router_train_label")] = job.router_train_label;
      }
    } else {
      body[QStringLiteral("correct_label")] =
          QString::fromStdString(job.data.y_class[static_cast<std::size_t>(i)]);
    }
    if (!job.replay_u01.isEmpty()) {
      QJsonArray ru;
      for (double u : job.replay_u01) {
        ru.append(u);
      }
      body[QStringLiteral("replay_u01")] = ru;
    }

    QNetworkRequest req(update_url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    req.setTransferTimeout(kHttpTransferTimeoutMs);
    QNetworkReply* reply = nam.post(req, QJsonDocument(body).toJson(QJsonDocument::Compact));
    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    const QNetworkReply::NetworkError nerr = reply->error();
    const QString err_s = reply->errorString();
    const QByteArray raw = reply->readAll();
    reply->deleteLater();

    if (nerr != QNetworkReply::NoError) {
      const QString reason = nerr == QNetworkReply::TimeoutError
                                  ? QStringLiteral("Request timed out")
                                  : err_s;
      emit finished(false, losses, false,
                    QStringLiteral("%1\n%2").arg(reason, QString::fromUtf8(raw)));
      return;
    }

    QJsonParseError pe{};
    const QJsonDocument doc = QJsonDocument::fromJson(raw, &pe);
    if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
      emit finished(false, losses, false, QStringLiteral("Bad JSON response from /update"));
      return;
    }
    const QJsonObject obj = doc.object();
    if (obj.contains(QStringLiteral("detail"))) {
      emit finished(false, losses, false, obj.value(QStringLiteral("detail")).toString());
      return;
    }

    const double loss = obj.value(QStringLiteral("loss")).toDouble();
    losses.append(loss);
    emit progress(i + 1, job.n_rows);
    emit stepLoss(i, loss);
  }

  emit finished(true, losses, false, QString());
}
