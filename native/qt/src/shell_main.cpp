// Cypha Qt 6 Widgets shell — native infer (optional preprocessor), REST /predict + /update, spawn cypha_rest.
// --smoke <model.cypha> [f_field.json]
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QFont>
#include <QFrame>
#include <QFormLayout>
#include <QGroupBox>
#include <QHash>
#include <QHBoxLayout>
#include <QSet>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMessageBox>
#include <QPainter>
#include <QPaintEvent>
#include <QCursor>
#include <QMouseEvent>
#include <QToolTip>
#include <QWheelEvent>
#include <QPolygonF>
#include <QPointF>
#include <QHeaderView>
#include <QMutex>
#include <QProgressBar>
#include <QProgressDialog>
#include <QScrollArea>
#include <QTextBrowser>
#include <QSettings>
#include <QSizePolicy>
#include <QTabWidget>
#include <QThread>
#include <QTimer>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QAction>
#include <QBrush>
#include <QColor>
#include <QDialog>
#include <QDialogButtonBox>
#include <QMenuBar>
#include <QSplitter>
#include <QStandardPaths>
#include <QTemporaryFile>
#include <QTextStream>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QStringList>
#include <QTableWidget>
#include <QTextCursor>
#include <QUrl>
#include <QVector>
#include <QVBoxLayout>
#include <QWidget>
#include <QStyle>
#include <QPalette>

#if defined(CYPHA_SHELL_QT_CHARTS)
#include <QtCharts/QAbstractAxis>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <optional>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/csv_ingest.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/mke_scalar_train_step.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/registry.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"
#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

#include "bulk_train_worker.h"
#include "bulk_rest_update_worker.h"
#include "bulk_mke_train_worker.h"

#ifdef CYPHA_SHELL_EXPERIMENT_DB
#include "cypha/experiment_db.hpp"
#include "cypha/experiment_db_crud.hpp"

namespace {
/// Canonical experiments DB schema (matches Python ExperimentDB._SCHEMA).
/// Embedded here to avoid runtime DDL file dependency.
constexpr const char* kExperimentDdl = R"SQL(
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT DEFAULT '',
    dataset_name  TEXT DEFAULT '',
    task          TEXT DEFAULT 'classification',
    created_at    REAL,
    tags          TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL,
    name              TEXT NOT NULL,
    config            TEXT,
    status            TEXT DEFAULT 'pending',
    created_at        REAL,
    updated_at        REAL,
    finished_at       REAL,
    duration_s        REAL DEFAULT 0,
    accuracy          REAL DEFAULT 0,
    macro_f1          REAL DEFAULT 0,
    r2_score          REAL DEFAULT 0,
    rmse              REAL DEFAULT 0,
    n_steps           INTEGER DEFAULT 0,
    n_classes         INTEGER DEFAULT 0,
    checkpoint_path   TEXT,
    preprocessor_path TEXT,
    metrics_history   TEXT DEFAULT '[]',
    tags              TEXT DEFAULT '[]',
    notes             TEXT DEFAULT '',
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
)SQL";
}  // namespace

#endif  // CYPHA_SHELL_EXPERIMENT_DB

namespace {

constexpr double kGhNigAdaptAlphaShell = 0.98;

std::filesystem::path qstring_to_fs_path(const QString& q) {
#if defined(_WIN32)
  return std::filesystem::path(q.toStdWString());
#else
  return std::filesystem::path(q.toUtf8().constData());
#endif
}

bool embedded_world_f_field_ok(const cypha::CNode& root, int d, int fd) {
  const cypha::CNode& world = cypha::map_get_required(root, "world");
  const cypha::CNode* wff = cypha::map_get(world, "F_field");
  const int expected = d * fd;
  return wff != nullptr && wff->kind == cypha::CNode::Tensor && wff->shape.size() == 2 &&
         static_cast<int>(wff->shape[0]) == d && static_cast<int>(wff->shape[1]) == fd &&
         static_cast<int>(wff->tensor.size()) == expected;
}

bool load_f_field_json_qt(const QString& path, int d, int fd, std::vector<double>& out, QString* err_out) {
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly)) {
    if (err_out != nullptr) {
      *err_out = QStringLiteral("Cannot open F_field JSON");
    }
    return false;
  }
  QJsonParseError pe{};
  const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &pe);
  if (pe.error != QJsonParseError::NoError) {
    if (err_out != nullptr) {
      *err_out = QStringLiteral("JSON parse: %1").arg(pe.errorString());
    }
    return false;
  }
  if (!doc.isArray()) {
    if (err_out != nullptr) {
      *err_out = QStringLiteral("F_field JSON root must be a 2D array");
    }
    return false;
  }
  const QJsonArray outer = doc.array();
  if (outer.size() != d) {
    if (err_out != nullptr) {
      *err_out = QStringLiteral("F_field JSON: expected %1 rows, got %2").arg(d).arg(outer.size());
    }
    return false;
  }
  out.clear();
  out.reserve(static_cast<std::size_t>(d * fd));
  for (int i = 0; i < d; ++i) {
    const QJsonValue rowv = outer.at(i);
    if (!rowv.isArray()) {
      if (err_out != nullptr) {
        *err_out = QStringLiteral("F_field JSON: row %1 is not an array").arg(i);
      }
      return false;
    }
    const QJsonArray row = rowv.toArray();
    if (row.size() != fd) {
      if (err_out != nullptr) {
        *err_out = QStringLiteral("F_field JSON: row %1 length %2, expected %3")
                       .arg(i)
                       .arg(row.size())
                       .arg(fd);
      }
      return false;
    }
    for (int j = 0; j < fd; ++j) {
      out.push_back(row.at(j).toDouble());
    }
  }
  return static_cast<int>(out.size()) == d * fd;
}

bool try_load_cypha_paths(const QString& cypha_path, const QString& ff_json_path,
                          std::vector<double>& f_field_storage, std::unique_ptr<cypha::CyphaInferModel>& out,
                          QString* err_out) {
  try {
    cypha::CNode root = cypha::load_cypha_file(cypha_path.toUtf8().constData());
    const cypha::CNode& enc = cypha::map_get_required(root, "enc_W");
    const int d = static_cast<int>(enc.shape[0]);
    const cypha::CNode& fh = cypha::map_get_required(root, "field_h");
    const int fd = static_cast<int>(fh.shape[0]);

    const double* ff_ptr = nullptr;
    if (embedded_world_f_field_ok(root, d, fd)) {
      f_field_storage.clear();
    } else {
      if (ff_json_path.isEmpty()) {
        if (err_out != nullptr) {
          *err_out = QStringLiteral(
              "world.F_field is not embedded in this .cypha — set an F_field JSON path (same format as "
              "cypha_rest --f-field-json).");
        }
        return false;
      }
      QString jerr;
      if (!load_f_field_json_qt(ff_json_path, d, fd, f_field_storage, &jerr)) {
        if (err_out != nullptr) {
          *err_out = jerr;
        }
        return false;
      }
      ff_ptr = f_field_storage.data();
    }
    out.reset(new cypha::CyphaInferModel(cypha::CyphaInferModel::from_root(root, ff_ptr, fd)));
    return true;
  } catch (const std::exception& e) {
    if (err_out != nullptr) {
      *err_out = QString::fromUtf8(e.what());
    }
    return false;
  }
}

int best_label_and_conf(const cypha::CyphaInferModel& m, const cypha::PreprocessorState* pre,
                        const std::vector<double>& x_in, std::string* label_out, double* conf_out,
                        bool use_gh = true, double gh_chi = 1.0, double gh_psi = 1.0,
                        const cypha::KernelMemory* kernel_mem = nullptr, bool use_kernel_llr = false,
                        double kernel_blend = 0.5) {
  std::vector<double> x_latent;
  const double* x_ptr = nullptr;
  if (pre != nullptr) {
    if (static_cast<int>(x_in.size()) != pre->input_dim) {
      return -1;
    }
    x_latent = pre->transform_one(x_in);
    if (static_cast<int>(x_latent.size()) != m.d_latent) {
      return -3;
    }
    x_ptr = x_latent.data();
  } else {
    if (static_cast<int>(x_in.size()) != m.d_latent) {
      return -1;
    }
    x_ptr = x_in.data();
  }

  if (m.labels.empty()) {
    return -2;
  }

  std::vector<double> H;
  cypha::batch_encode(m, x_ptr, 1, H);

  if (use_gh) {
    cypha::CyphaInferOptions kopt{};
    kopt.kernel_mem = kernel_mem;
    kopt.use_kernel_llr = use_kernel_llr && kernel_mem != nullptr;
    kopt.kernel_blend = kernel_blend;
    const cypha::GhInferAtHResult gh =
        cypha::gh_infer_at_h(m, H.data(), gh_chi, gh_psi, kGhNigAdaptAlphaShell, &kopt);
    *label_out = gh.label;
    *conf_out = gh.confidence;
  } else {
    cypha::CyphaInferOptions iopt{};
    iopt.deliberation_lo = m.deliberation_lo;
    iopt.deliberation_hi = m.deliberation_hi;
    iopt.use_field = true;
    iopt.kernel_mem = kernel_mem;
    iopt.use_kernel_llr = use_kernel_llr && kernel_mem != nullptr;
    iopt.kernel_blend = kernel_blend;
    const cypha::InferAtHResult inf = cypha::infer_at_h(m, H.data(), iopt);
    *label_out = inf.label;
    *conf_out = inf.confidence;
  }
  return 0;
}

int run_smoke(const QString& cypha_path, const QString& ff_json_path) {
  std::vector<double> ff_store;
  std::unique_ptr<cypha::CyphaInferModel> model;
  QString err;
  if (!try_load_cypha_paths(cypha_path, ff_json_path, ff_store, model, &err)) {
    std::fprintf(stderr, "cypha_qt_shell --smoke: load failed: %s\n", err.toUtf8().constData());
    return 2;
  }
  std::vector<double> x(static_cast<std::size_t>(model->d_latent), 0.0);
  std::string label;
  double conf = 0.0;
  const int rc = best_label_and_conf(*model, nullptr, x, &label, &conf);
  if (rc != 0) {
    std::fprintf(stderr, "cypha_qt_shell --smoke: infer failed (%d)\n", rc);
    return 3;
  }
  std::printf("cypha_qt_shell smoke OK label=%s conf=%.6f d=%d\n", label.c_str(), conf, model->d_latent);
  return 0;
}

bool parse_feature_vector(const QString& t, int expected_d, std::vector<double>& x, QString* err) {
  const QStringList parts = t.split(QLatin1Char(','), Qt::SkipEmptyParts);
  x.clear();
  x.reserve(static_cast<std::size_t>(parts.size()));
  for (const QString& p : parts) {
    bool ok = false;
    const double v = p.trimmed().toDouble(&ok);
    if (!ok) {
      if (err != nullptr) {
        *err = QStringLiteral("Not a number: \"%1\"").arg(p.trimmed());
      }
      return false;
    }
    x.push_back(v);
  }
  if (static_cast<int>(x.size()) != expected_d) {
    if (err != nullptr) {
      *err = QStringLiteral("Expected %1 values, got %2.").arg(expected_d).arg(static_cast<int>(x.size()));
    }
    return false;
  }
  return true;
}

bool parse_token_ids(const QString& t, std::vector<int>& ids, QString* err) {
  ids.clear();
  const QString trimmed = t.trimmed();
  if (trimmed.isEmpty()) {
    if (err != nullptr) {
      *err = QStringLiteral("Enter at least one token id.");
    }
    return false;
  }
  const QStringList parts = trimmed.split(QLatin1Char(','), Qt::SkipEmptyParts);
  ids.reserve(static_cast<std::size_t>(parts.size()));
  for (const QString& p : parts) {
    bool ok = false;
    const int v = p.trimmed().toInt(&ok);
    if (!ok || v < 0) {
      if (err != nullptr) {
        *err = QStringLiteral("Not a non-negative integer: \"%1\"").arg(p.trimmed());
      }
      return false;
    }
    ids.push_back(v);
  }
  return true;
}

/// Read just the first (header) row of a CSV file without a full parse.
QStringList read_csv_header(const QString& path) {
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    return {};
  }
  const QByteArray line = f.readLine();
  if (line.isEmpty()) {
    return {};
  }
  const auto rows = cypha::parse_csv_utf8(QString::fromUtf8(line).trimmed().toStdString(), ',');
  if (rows.empty()) {
    return {};
  }
  QStringList out;
  for (const auto& s : rows[0]) {
    out << QString::fromStdString(s);
  }
  return out;
}

/// Read up to `max_rows` raw rows (+ header row 0) for preview.
std::vector<QStringList> read_csv_preview(const QString& path, int max_rows = 8) {
  std::vector<QStringList> result;
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    return result;
  }
  int total = 0;
  while (!f.atEnd() && total <= max_rows) {
    const QByteArray raw = f.readLine();
    const QString trimmed = QString::fromUtf8(raw).trimmed();
    if (trimmed.isEmpty()) {
      continue;
    }
    const auto rows = cypha::parse_csv_utf8(trimmed.toStdString(), ',');
    if (rows.empty() || rows[0].empty()) {
      continue;
    }
    QStringList row;
    for (const auto& s : rows[0]) {
      row << QString::fromStdString(s);
    }
    result.push_back(std::move(row));
    ++total;
  }
  return result;
}

struct HttpJsonResult {
  bool ok{false};
  QString err;
  QJsonObject obj;
  QByteArray raw;
};

// Transfer timeout for one-shot REST calls made from the GUI thread (health/ready/models/predict/
// update/load). Keeps the UI from freezing indefinitely if cypha_rest hangs or is unreachable.
constexpr int kHttpTransferTimeoutMs = 15000;

HttpJsonResult http_post_json(const QUrl& url, const QJsonObject& body) {
  HttpJsonResult r;
  QNetworkRequest req(url);
  req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
  req.setTransferTimeout(kHttpTransferTimeoutMs);
  QNetworkAccessManager nam;
  QNetworkReply* reply = nam.post(req, QJsonDocument(body).toJson(QJsonDocument::Compact));
  QEventLoop loop;
  QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
  loop.exec();
  const QNetworkReply::NetworkError nerr = reply->error();
  const QString err_s = reply->errorString();
  r.raw = reply->readAll();
  reply->deleteLater();
  if (nerr != QNetworkReply::NoError) {
    const QString reason = nerr == QNetworkReply::TimeoutError
                                ? QStringLiteral("Request timed out")
                                : err_s;
    r.err = QStringLiteral("%1\n%2").arg(reason, QString::fromUtf8(r.raw));
    return r;
  }
  QJsonParseError pe{};
  const QJsonDocument doc = QJsonDocument::fromJson(r.raw, &pe);
  if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
    r.err = QStringLiteral("Bad JSON response");
    return r;
  }
  r.obj = doc.object();
  r.ok = true;
  return r;
}

HttpJsonResult http_get_json(const QUrl& url) {
  HttpJsonResult r;
  QNetworkRequest req(url);
  req.setTransferTimeout(kHttpTransferTimeoutMs);
  QNetworkAccessManager nam;
  QNetworkReply* reply = nam.get(req);
  QEventLoop loop;
  QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
  loop.exec();
  const QNetworkReply::NetworkError nerr = reply->error();
  const QString err_s = reply->errorString();
  r.raw = reply->readAll();
  reply->deleteLater();
  if (nerr != QNetworkReply::NoError) {
    const QString reason = nerr == QNetworkReply::TimeoutError
                                ? QStringLiteral("Request timed out")
                                : err_s;
    r.err = QStringLiteral("%1\n%2").arg(reason, QString::fromUtf8(r.raw));
    return r;
  }
  QJsonParseError pe{};
  const QJsonDocument doc = QJsonDocument::fromJson(r.raw, &pe);
  if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
    r.err = QStringLiteral("Bad JSON response");
    return r;
  }
  r.obj = doc.object();
  r.ok = true;
  return r;
}

QString listen_to_http_base(const QString& listen) {
  const QString t = listen.trimmed();
  if (t.isEmpty()) {
    return QString();
  }
  return QStringLiteral("http://") + t;
}

QVector<double> loss_ema_series(const QVector<double>& v, double alpha) {
  QVector<double> o;
  if (v.isEmpty()) {
    return o;
  }
  o.reserve(v.size());
  double e = v[0];
  o.push_back(e);
  for (int i = 1; i < v.size(); ++i) {
    e = alpha * v[i] + (1.0 - alpha) * e;
    o.push_back(e);
  }
  return o;
}

/// Minimal SVG (no Qt Svg module) — same geometry as `SimpleLossChart`.
bool write_loss_chart_svg(const QString& path, const QVector<double>& rest, const QVector<double>& natv,
                          const QVector<double>& rest_ema, const QVector<double>& natv_ema) {
  const bool have_r = !rest.isEmpty();
  const bool have_n = !natv.isEmpty();
  if (!have_r && !have_n) {
    return false;
  }
  auto expand_range = [](double& lo, double& hi, const QVector<double>& v) {
    for (double x : v) {
      lo = std::min(lo, x);
      hi = std::max(hi, x);
    }
  };
  double lo = have_r ? rest[0] : natv[0];
  double hi = lo;
  if (have_r) {
    expand_range(lo, hi, rest);
  }
  if (have_n) {
    expand_range(lo, hi, natv);
  }
  if (!rest_ema.isEmpty()) {
    expand_range(lo, hi, rest_ema);
  }
  if (!natv_ema.isEmpty()) {
    expand_range(lo, hi, natv_ema);
  }
  if (!(hi > lo)) {
    hi = lo + 1e-9;
  }
  const int max_n = std::max(rest.size(), natv.size());
  const double denom = max_n > 1 ? static_cast<double>(max_n - 1) : 1.0;

  constexpr int W = 880;
  constexpr int H = 200;
  constexpr int ml = 48;
  constexpr int mr = 12;
  constexpr int mt = 10;
  constexpr int mb = 40;
  const int pw = W - ml - mr;
  const int ph = H - mt - mb;

  auto xy = [&](int i, double yval) -> std::pair<double, double> {
    const double tx = static_cast<double>(i) / denom;
    const double ynorm = (yval - lo) / (hi - lo);
    const double px = static_cast<double>(ml) + tx * static_cast<double>(pw);
    const double py = static_cast<double>(H - mb) - ynorm * static_cast<double>(ph);
    return {px, py};
  };

  auto poly_points = [&](const QVector<double>& v) -> QString {
    if (v.isEmpty()) {
      return QString();
    }
    QString s;
    for (int i = 0; i < v.size(); ++i) {
      const auto pr = xy(i, v[static_cast<std::size_t>(i)]);
      if (i > 0) {
        s += QLatin1Char(' ');
      }
      s += QString::number(pr.first, 'f', 2);
      s += QLatin1Char(',');
      s += QString::number(pr.second, 'f', 2);
    }
    return s;
  };

  QFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    return false;
  }
  QTextStream ts(&f);
  ts << QStringLiteral("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
  ts << QStringLiteral("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%1\" height=\"%2\" "
                      "viewBox=\"0 0 %1 %2\">\n")
          .arg(W)
          .arg(H);
  ts << QStringLiteral("<rect width=\"100%\" height=\"100%\" fill=\"#202226\"/>\n");
  ts << QStringLiteral("<rect x=\"%1\" y=\"%2\" width=\"%3\" height=\"%4\" fill=\"none\" "
                      "stroke=\"#5a5a64\" stroke-width=\"1\"/>\n")
          .arg(ml)
          .arg(mt)
          .arg(pw)
          .arg(ph);

  auto emit_poly = [&](const QString& pts, const char* stroke, bool dashed) {
    if (pts.isEmpty()) {
      return;
    }
    const QString dash_attr =
        dashed ? QStringLiteral("stroke-dasharray=\"6 4\"") : QString();
    ts << QStringLiteral("<polyline fill=\"none\" stroke=\"%1\" stroke-width=\"%2\" %3 points=\"%4\"/>\n")
              .arg(QLatin1String(stroke),
                   QString::number(dashed ? 1.5 : 2.0, 'f', 1),
                   dash_attr,
                   pts);
  };

  emit_poly(poly_points(natv_ema), "#ffaa5a", true);
  emit_poly(poly_points(rest_ema), "#6eb4ff", true);
  emit_poly(poly_points(natv), "#ffaa5a", false);
  emit_poly(poly_points(rest), "#6eb4ff", false);

  ts << QStringLiteral(
      "<text x=\"%1\" y=\"%2\" fill=\"#b4b4be\" font-size=\"11\" font-family=\"sans-serif\">"
      "REST solid / dashed EMA · Native solid / dashed EMA</text>\n")
          .arg(ml)
          .arg(H - 14);
  ts << QStringLiteral("</svg>\n");
  return true;
}

void root_map_assign(std::vector<std::pair<std::string, cypha::CNode>>& map, std::string key, cypha::CNode val) {
  for (auto& kv : map) {
    if (kv.first == key) {
      kv.second = std::move(val);
      return;
    }
  }
  map.emplace_back(std::move(key), std::move(val));
}

/// Session-only fields aligned with Python `save_state` / `.cypha` (not stored on `CyphaInferModel`).
struct NativeSessionSnapshotPatch {
  double ood_sigma{15.0};
  double gh_chi{1.0};
  double gh_psi{1.0};
  double gh_r_base{1.0};
  const std::vector<double>* gh_inv_v_clean{nullptr};
  /// Python `feat_dim`. If `< 0`, save path uses `m.d_latent` only (latent-native bundle).
  int feat_dim{-1};
};

/// Patch top-level training snapshot fields from `CyphaInferModel` after native `train_step` / GH.
/// `merge_state_into_root_for_save` already updated `world` / `classes`; this updates encoder, field head,
/// temperature, context Tier-1+2 (`ctx_hist_packed`, `ctx_cooccur`, `mid_trans`, …), `field_W_T` / `w_inject`
/// when present, `field_step`, and scalars. If `session_patch` is non-null, writes `ood_sigma` and GH session
/// keys (`gh_chi_session`, `gh_psi_session`, `gh_R_base`, `gh_inv_v_clean`). Writes `feat_dim` from
/// `session_patch->feat_dim` when `>= 0`, else `m.d_latent`. Writes optional **`field_a_eff`** (float64 tensor,
/// same layout as Python **`save_state`**) when the live model has a full **`field_a_eff`** buffer; native
/// **`load_cypha_*`** prefers that tensor over recomputing from **`field_W_T`** when shapes match.
void patch_infer_training_snapshot(cypha::CNode& root, const cypha::CyphaInferModel& m, std::int64_t total_steps_save,
                                   double llr_ema_session, const NativeSessionSnapshotPatch* session_patch) {
  for (auto& kv : root.map) {
    if (kv.first == "enc_W") {
      kv.second.kind = cypha::CNode::Tensor;
      kv.second.shape = {static_cast<std::uint32_t>(m.d_latent), static_cast<std::uint32_t>(m.d_latent)};
      kv.second.tensor = m.enc_w;
    } else if (kv.first == "field_h") {
      kv.second.kind = cypha::CNode::Tensor;
      kv.second.shape = {static_cast<std::uint32_t>(m.field_dim)};
      kv.second.tensor = m.field_h;
    } else if (kv.first == "temperature") {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = m.temperature;
    } else if (kv.first == "base_temp") {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = m.base_temp;
    } else if (kv.first == "mahal_ema" && m.has_mahal_ema) {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = m.mahal_ema;
    } else if (kv.first == "mahal_std_ema") {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = m.mahal_std_ema;
    } else if (kv.first == "llr_scale_ema") {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = m.llr_scale_ema;
    } else if (kv.first == "llr_scale_n") {
      kv.second.kind = cypha::CNode::Int;
      kv.second.i = m.llr_scale_n;
    } else if (kv.first == "llr_scale_baseline") {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = m.llr_scale_baseline;
    } else if (kv.first == "llr_ema") {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = llr_ema_session;
    } else if (kv.first == "mid_n") {
      kv.second.kind = cypha::CNode::Float;
      kv.second.f = m.mid_n;
    } else if (kv.first == "mid_freq") {
      kv.second.kind = cypha::CNode::Map;
      kv.second.map.clear();
      for (const auto& pr : m.mid_freq) {
        cypha::CNode vn;
        vn.kind = cypha::CNode::Float;
        vn.f = pr.second;
        kv.second.map.emplace_back(pr.first, std::move(vn));
      }
    } else if (kv.first == "total_steps") {
      kv.second.kind = cypha::CNode::Int;
      kv.second.i = total_steps_save;
    }
  }

  cypha::CNode ctx_hist;
  ctx_hist.kind = cypha::CNode::Map;
  ctx_hist.map.clear();
  for (std::size_t i = 0; i < m.ctx_history.size(); ++i) {
    cypha::CNode entry;
    entry.kind = cypha::CNode::Map;
    cypha::CNode ls;
    ls.kind = cypha::CNode::Str;
    ls.s = m.ctx_history[i].first;
    cypha::CNode cb;
    cb.kind = cypha::CNode::Bool;
    cb.b = m.ctx_history[i].second;
    entry.map.emplace_back("l", std::move(ls));
    entry.map.emplace_back("c", std::move(cb));
    ctx_hist.map.emplace_back(std::to_string(static_cast<int>(i)), std::move(entry));
  }
  root_map_assign(root.map, "ctx_hist_packed", std::move(ctx_hist));

  cypha::CNode ctx_co;
  ctx_co.kind = cypha::CNode::Map;
  ctx_co.map.clear();
  for (const auto& outer : m.cooccur) {
    cypha::CNode row;
    row.kind = cypha::CNode::Map;
    for (const auto& inner : outer.second) {
      cypha::CNode vi;
      vi.kind = cypha::CNode::Int;
      vi.i = static_cast<std::int64_t>(std::llround(inner.second));
      row.map.emplace_back(inner.first, std::move(vi));
    }
    ctx_co.map.emplace_back(outer.first, std::move(row));
  }
  root_map_assign(root.map, "ctx_cooccur", std::move(ctx_co));

  cypha::CNode ctx_tot;
  ctx_tot.kind = cypha::CNode::Map;
  ctx_tot.map.clear();
  for (const auto& pr : m.cooccur_tot) {
    cypha::CNode vf;
    vf.kind = cypha::CNode::Float;
    vf.f = pr.second;
    ctx_tot.map.emplace_back(pr.first, std::move(vf));
  }
  root_map_assign(root.map, "ctx_cooccur_tot", std::move(ctx_tot));

  cypha::CNode clbl;
  clbl.kind = cypha::CNode::Str;
  clbl.s = m.ctx_last_label;
  root_map_assign(root.map, "ctx_last_label", std::move(clbl));

  cypha::CNode mid_tr;
  mid_tr.kind = cypha::CNode::Map;
  mid_tr.map.clear();
  for (const auto& outer : m.mid_trans) {
    cypha::CNode row;
    row.kind = cypha::CNode::Map;
    for (const auto& inner : outer.second) {
      cypha::CNode vf;
      vf.kind = cypha::CNode::Float;
      vf.f = inner.second;
      row.map.emplace_back(inner.first, std::move(vf));
    }
    mid_tr.map.emplace_back(outer.first, std::move(row));
  }
  root_map_assign(root.map, "mid_trans", std::move(mid_tr));

  if (!m.field_w_t.empty() && m.field_dim > 0 &&
      static_cast<int>(m.field_w_t.size()) == m.field_dim * m.field_dim) {
    cypha::CNode wt;
    wt.kind = cypha::CNode::Tensor;
    wt.shape = {static_cast<std::uint32_t>(m.field_dim), static_cast<std::uint32_t>(m.field_dim)};
    wt.tensor = m.field_w_t;
    root_map_assign(root.map, "field_W_T", std::move(wt));
  }
  if (!m.w_inject.empty() && m.field_dim > 0 && m.d_latent > 0 &&
      static_cast<int>(m.w_inject.size()) == m.field_dim * m.d_latent) {
    cypha::CNode wj;
    wj.kind = cypha::CNode::Tensor;
    wj.shape = {static_cast<std::uint32_t>(m.field_dim), static_cast<std::uint32_t>(m.d_latent)};
    wj.tensor = m.w_inject;
    root_map_assign(root.map, "w_inject", std::move(wj));
  }

  cypha::CNode fstep_n;
  fstep_n.kind = cypha::CNode::Int;
  fstep_n.i = m.field_step;
  root_map_assign(root.map, "field_step", std::move(fstep_n));

  if (!m.field_a_eff.empty() && m.field_dim > 0 &&
      static_cast<int>(m.field_a_eff.size()) == m.field_dim * m.field_dim) {
    cypha::CNode ae;
    ae.kind = cypha::CNode::Tensor;
    ae.shape = {static_cast<std::uint32_t>(m.field_dim), static_cast<std::uint32_t>(m.field_dim)};
    ae.tensor.resize(static_cast<std::size_t>(m.field_dim * m.field_dim));
    for (int i = 0; i < m.field_dim * m.field_dim; ++i) {
      ae.tensor[static_cast<std::size_t>(i)] = static_cast<double>(m.field_a_eff[static_cast<std::size_t>(i)]);
    }
    root_map_assign(root.map, "field_a_eff", std::move(ae));
  }

  cypha::CNode llw;
  llw.kind = cypha::CNode::Float;
  llw.f = -1.5;
  root_map_assign(root.map, "ll_world_ema", std::move(llw));

  cypha::CNode tcor_n;
  tcor_n.kind = cypha::CNode::Int;
  tcor_n.i = m.total_correct;
  root_map_assign(root.map, "total_correct", std::move(tcor_n));

  {
    int fdim = m.d_latent;
    if (session_patch != nullptr && session_patch->feat_dim >= 0) {
      fdim = session_patch->feat_dim;
    }
    cypha::CNode fdn;
    fdn.kind = cypha::CNode::Int;
    fdn.i = static_cast<std::int64_t>(fdim);
    root_map_assign(root.map, "feat_dim", std::move(fdn));
  }

  if (session_patch != nullptr) {
    cypha::CNode ood;
    ood.kind = cypha::CNode::Float;
    ood.f = session_patch->ood_sigma;
    root_map_assign(root.map, "ood_sigma", std::move(ood));

    cypha::CNode gchi;
    gchi.kind = cypha::CNode::Float;
    gchi.f = session_patch->gh_chi;
    root_map_assign(root.map, "gh_chi_session", std::move(gchi));
    cypha::CNode gpsi;
    gpsi.kind = cypha::CNode::Float;
    gpsi.f = session_patch->gh_psi;
    root_map_assign(root.map, "gh_psi_session", std::move(gpsi));
    cypha::CNode grb;
    grb.kind = cypha::CNode::Float;
    grb.f = session_patch->gh_r_base;
    root_map_assign(root.map, "gh_R_base", std::move(grb));

    if (session_patch->gh_inv_v_clean != nullptr &&
        static_cast<int>(session_patch->gh_inv_v_clean->size()) == m.d_latent) {
      cypha::CNode ginv;
      ginv.kind = cypha::CNode::Tensor;
      ginv.shape = {static_cast<std::uint32_t>(m.d_latent)};
      ginv.tensor = *session_patch->gh_inv_v_clean;
      root_map_assign(root.map, "gh_inv_v_clean", std::move(ginv));
    }
  }
}

/// Lightweight loss curve (no Qt Charts dependency — keeps CI on qt6-base-dev only).
class SimpleLossChart final : public QWidget {
 public:
  explicit SimpleLossChart(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(128);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMouseTracking(true);
  }
  void set_loss_runs(const QVector<double>& rest, const QVector<double>& native_v,
                     const QVector<double>& rest_ema, const QVector<double>& native_ema) {
    losses_rest_ = rest;
    losses_native_ = native_v;
    losses_rest_ema_ = rest_ema;
    losses_native_ema_ = native_ema;
    recompute_data_extents();
    reset_view_to_data();
    update();
  }
  void clear_losses() {
    compare_mode_ = false;
    compare_series_.clear();
    losses_rest_.clear();
    losses_native_.clear();
    losses_rest_ema_.clear();
    losses_native_ema_.clear();
    view_initialized_ = false;
    update();
  }

  /// Overlay multiple named loss curves (experiment run comparison).
  void set_compare_loss_runs(const QVector<QPair<QString, QVector<double>>>& runs) {
    compare_mode_ = true;
    compare_series_.clear();
    static const QList<QColor> kPalette = {
        QColor(110, 180, 255), QColor(255, 170, 90),  QColor(140, 220, 140),
        QColor(220, 140, 220), QColor(255, 220, 100), QColor(100, 200, 200),
        QColor(255, 130, 130), QColor(180, 180, 255),
    };
    for (int i = 0; i < runs.size(); ++i) {
      CompareSeries s;
      s.label = runs[i].first;
      s.losses = runs[i].second;
      s.color = kPalette[i % kPalette.size()];
      compare_series_.append(s);
    }
    losses_rest_.clear();
    losses_native_.clear();
    losses_rest_ema_.clear();
    losses_native_ema_.clear();
    recompute_data_extents();
    reset_view_to_data();
    update();
  }
  void set_y_range_lock(bool locked, double lo, double hi) {
    y_range_locked_ = locked;
    y_lock_lo_ = lo;
    y_lock_hi_ = hi;
    recompute_data_extents();
    clamp_view_to_data();
    update();
  }

 protected:
  void paintEvent(QPaintEvent* e) override {
    Q_UNUSED(e);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QRect full = rect();
    painter.fillRect(full, QColor(32, 34, 38));
    const bool have_r = !losses_rest_.isEmpty();
    const bool have_n = !losses_native_.isEmpty();
    const bool have_cmp = compare_mode_ && !compare_series_.isEmpty();
    if (!have_r && !have_n && !have_cmp) {
      painter.setPen(QColor(140, 140, 150));
      painter.drawText(full.adjusted(8, 0, -8, 0), Qt::AlignCenter,
                       compare_mode_
                           ? QStringLiteral("Compare runs — select rows in Experiments and click Compare")
                           : QStringLiteral("Per-step loss — bulk REST (blue) vs native (orange); dashed = EMA"));
      return;
    }

    if (!view_initialized_) {
      reset_view_to_data();
    }

    const double lo = view_y0_;
    const double hi = view_y1_;
    int max_n = std::max(losses_rest_.size(), losses_native_.size());
    if (have_cmp) {
      for (const CompareSeries& s : compare_series_) {
        max_n = std::max(max_n, static_cast<int>(s.losses.size()));
      }
    }
    const QRect plot = plot_rect(full);

    // ── Grid lines + Y-axis ticks ─────────────────────────────────────────────
    const int kYTicks = 5;
    QFont tick_font = painter.font();
    tick_font.setPointSizeF(std::max(6.5, tick_font.pointSizeF() - 1.0));
    painter.setFont(tick_font);
    for (int ti = 0; ti <= kYTicks; ++ti) {
      const double frac = static_cast<double>(ti) / kYTicks;
      const double val  = lo + frac * (hi - lo);
      const int    py   = static_cast<int>(plot.bottom() - frac * plot.height());
      painter.setPen(QPen(QColor(55, 58, 65), 1, Qt::SolidLine));
      painter.drawLine(plot.left(), py, plot.right(), py);
      painter.setPen(QColor(150, 150, 160));
      const QString lbl = QString::number(val, 'g', 3);
      const QRect lr(full.left(), py - 9, plot.left() - 4, 18);
      painter.drawText(lr, Qt::AlignRight | Qt::AlignVCenter, lbl);
    }

    // ── X-axis ticks ──────────────────────────────────────────────────────────
    const int kXTicks = 4;
    for (int ti = 0; ti <= kXTicks; ++ti) {
      const double frac = static_cast<double>(ti) / kXTicks;
      const double step_val = view_x0_ + frac * (view_x1_ - view_x0_);
      const int    px       = static_cast<int>(plot.left() + frac * plot.width());
      painter.setPen(QPen(QColor(55, 58, 65), 1, Qt::SolidLine));
      painter.drawLine(px, plot.top(), px, plot.bottom());
      painter.setPen(QColor(150, 150, 160));
      const QString lbl = QString::number(std::round(step_val));
      const QRect xr(px - 20, plot.bottom() + 4, 40, 16);
      painter.drawText(xr, Qt::AlignHCenter | Qt::AlignTop, lbl);
    }

  // ── Plot border ───────────────────────────────────────────────────────────
    painter.setPen(QPen(QColor(90, 90, 100), 1));
    painter.drawRect(plot);

    // ── Series ────────────────────────────────────────────────────────────────
    auto draw_series = [&](const QVector<double>& v, const QColor& col, bool dashed, int width) {
      if (v.isEmpty()) return;
      const QColor use = dashed ? QColor(col.red(), col.green(), col.blue(), 200) : col;
      QPen pen(use, width);
      if (dashed) pen.setStyle(Qt::DashLine);
      painter.setPen(pen);
      if (v.size() == 1) {
        const QPointF pt = data_to_pixel(plot, 0.0, v[0], view_x0_, view_x1_, lo, hi);
        painter.drawEllipse(pt, 3.0, 3.0);
        return;
      }
      QPolygonF poly;
      const int n = v.size();
      for (int i = 0; i < n; ++i) {
        poly << data_to_pixel(plot, static_cast<double>(i), v[i], view_x0_, view_x1_, lo, hi);
      }
      painter.drawPolyline(poly);
    };
    if (have_cmp) {
      for (const CompareSeries& s : compare_series_) {
        draw_series(s.losses, s.color, false, 2);
      }
    } else {
      draw_series(losses_native_ema_, QColor(255, 170, 90), true, 1);
      draw_series(losses_rest_ema_,   QColor(110, 180, 255), true, 1);
      draw_series(losses_native_,     QColor(255, 170, 90), false, 2);
      draw_series(losses_rest_,       QColor(110, 180, 255), false, 2);
    }

    // ── Legend ────────────────────────────────────────────────────────────────
    const int legend_h = 20;
    const int ley  = full.bottom() - legend_h + 2;
    const int leh  = legend_h - 4;
    struct LegItem { QColor col; bool dashed; QString text; };
    QList<LegItem> items;
    if (have_cmp) {
      for (const CompareSeries& s : compare_series_) {
        items.append({s.color, false, s.label});
      }
    } else {
      items = {
          {QColor(110, 180, 255), false, QStringLiteral("REST")},
          {QColor(110, 180, 255), true,  QStringLiteral("REST EMA")},
          {QColor(255, 170, 90),  false, QStringLiteral("Native")},
          {QColor(255, 170, 90),  true,  QStringLiteral("Native EMA")},
      };
    }
    int lx = 56;
    for (const auto& item : items) {
      QPen pen(item.col, item.dashed ? 1 : 2);
      if (item.dashed) pen.setStyle(Qt::DashLine);
      painter.setPen(pen);
      painter.drawLine(lx, ley + leh / 2, lx + 18, ley + leh / 2);
      painter.setPen(QColor(200, 200, 210));
      const QRect tr(lx + 22, ley, 120, leh);
      painter.drawText(tr, Qt::AlignLeft | Qt::AlignVCenter, item.text);
      lx += std::min(144, static_cast<int>(56 + item.text.size() * 7));
    }
  }

  void mouseMoveEvent(QMouseEvent* e) override {
    if (panning_) {
      const QRect plot = plot_rect(rect());
      if (plot.width() > 0 && plot.height() > 0) {
        const double dx =
            -static_cast<double>(e->pos().x() - pan_anchor_.x()) / plot.width() * (view_x1_ - view_x0_);
        const double dy =
            static_cast<double>(e->pos().y() - pan_anchor_.y()) / plot.height() * (view_y1_ - view_y0_);
        view_x0_ += dx;
        view_x1_ += dx;
        view_y0_ += dy;
        view_y1_ += dy;
        clamp_view_to_data();
        pan_anchor_ = e->pos();
        update();
      }
    }
    update_tooltip(e->pos());
    QWidget::mouseMoveEvent(e);
  }

  void mousePressEvent(QMouseEvent* e) override {
    if (e->button() == Qt::LeftButton && has_data()) {
      panning_ = true;
      pan_anchor_ = e->pos();
      setCursor(Qt::ClosedHandCursor);
    }
    QWidget::mousePressEvent(e);
  }

  void mouseReleaseEvent(QMouseEvent* e) override {
    if (e->button() == Qt::LeftButton && panning_) {
      panning_ = false;
      unsetCursor();
    }
    QWidget::mouseReleaseEvent(e);
  }

  void leaveEvent(QEvent* e) override {
    Q_UNUSED(e);
    QToolTip::hideText();
    if (!panning_) {
      unsetCursor();
    }
    QWidget::leaveEvent(e);
  }

  void wheelEvent(QWheelEvent* e) override {
    if (!has_data()) {
      e->ignore();
      return;
    }
    const QRect plot = plot_rect(rect());
    if (!plot.contains(e->position().toPoint()) || plot.width() <= 0 || plot.height() <= 0) {
      e->ignore();
      return;
    }
    const double factor = e->angleDelta().y() > 0 ? 0.85 : 1.15;
    const double frac_x = (e->position().x() - plot.left()) / plot.width();
    const double frac_y = (plot.bottom() - e->position().y()) / plot.height();
    const double cx = view_x0_ + frac_x * (view_x1_ - view_x0_);
    const double cy = view_y0_ + frac_y * (view_y1_ - view_y0_);
    const double half_w = (view_x1_ - view_x0_) * factor * 0.5;
    const double half_h = (view_y1_ - view_y0_) * factor * 0.5;
    view_x0_ = cx - half_w;
    view_x1_ = cx + half_w;
    view_y0_ = cy - half_h;
    view_y1_ = cy + half_h;
    clamp_view_to_data();
    update();
    e->accept();
  }

 private:
  static QRect plot_rect(const QRect& full) {
    constexpr int legend_h = 20;
    constexpr int x_label_h = 22;
    return full.adjusted(54, 10, -12, -(x_label_h + legend_h));
  }

  static QPointF data_to_pixel(const QRect& plot, double x, double y, double x0, double x1, double y0, double y1) {
    const double xspan = x1 - x0;
    const double yspan = y1 - y0;
    const double tx = xspan > 0.0 ? (x - x0) / xspan : 0.0;
    const double ty = yspan > 0.0 ? (y - y0) / yspan : 0.0;
    return QPointF(plot.left() + tx * plot.width(), plot.bottom() - ty * plot.height());
  }

  bool has_data() const {
    return !losses_rest_.isEmpty() || !losses_native_.isEmpty() ||
           (compare_mode_ && !compare_series_.isEmpty());
  }

  void recompute_data_extents() {
    const bool have_r = !losses_rest_.isEmpty();
    const bool have_n = !losses_native_.isEmpty();
    const bool have_cmp = compare_mode_ && !compare_series_.isEmpty();
    if (!have_r && !have_n && !have_cmp) {
      return;
    }
    auto expand_range = [](double& lo, double& hi, const QVector<double>& v) {
      for (double x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    };
    int max_n = std::max(losses_rest_.size(), losses_native_.size());
    if (have_cmp) {
      for (const CompareSeries& s : compare_series_) {
        max_n = std::max(max_n, static_cast<int>(s.losses.size()));
      }
    }
    data_x0_ = 0.0;
    data_x1_ = max_n > 1 ? static_cast<double>(max_n - 1) : 1.0;
    if (y_range_locked_) {
      data_y0_ = y_lock_lo_;
      data_y1_ = y_lock_hi_;
    } else {
      double lo = 0.0;
      double hi = 1.0;
      bool seeded = false;
      auto seed_range = [&](const QVector<double>& v) {
        if (v.isEmpty()) return;
        if (!seeded) {
          lo = v[0];
          hi = lo;
          seeded = true;
        }
        expand_range(lo, hi, v);
      };
      seed_range(losses_rest_);
      seed_range(losses_native_);
      if (have_cmp) {
        for (const CompareSeries& s : compare_series_) {
          seed_range(s.losses);
        }
      }
      if (!losses_rest_ema_.isEmpty()) expand_range(lo, hi, losses_rest_ema_);
      if (!losses_native_ema_.isEmpty()) expand_range(lo, hi, losses_native_ema_);
      if (!seeded) {
        lo = 0.0;
        hi = 1.0;
      }
      if (!(hi > lo)) hi = lo + 1e-9;
      data_y0_ = lo;
      data_y1_ = hi;
    }
  }

  void reset_view_to_data() {
    if (!has_data()) {
      view_initialized_ = false;
      return;
    }
    recompute_data_extents();
    view_x0_ = data_x0_;
    view_x1_ = data_x1_;
    view_y0_ = data_y0_;
    view_y1_ = data_y1_;
    view_initialized_ = true;
  }

  void clamp_view_to_data() {
    if (!has_data()) {
      return;
    }
    recompute_data_extents();
    const double min_x_span = std::max((data_x1_ - data_x0_) / 200.0, 1.0);
    const double min_y_span = std::max((data_y1_ - data_y0_) / 200.0, 1e-9);
    double vw = view_x1_ - view_x0_;
    double vh = view_y1_ - view_y0_;
    if (vw < min_x_span) {
      const double cx = (view_x0_ + view_x1_) * 0.5;
      vw = min_x_span;
      view_x0_ = cx - vw * 0.5;
      view_x1_ = cx + vw * 0.5;
    }
    if (vh < min_y_span) {
      const double cy = (view_y0_ + view_y1_) * 0.5;
      vh = min_y_span;
      view_y0_ = cy - vh * 0.5;
      view_y1_ = cy + vh * 0.5;
    }
    vw = view_x1_ - view_x0_;
    vh = view_y1_ - view_y0_;
    const double dw = data_x1_ - data_x0_;
    const double dh = data_y1_ - data_y0_;
    if (vw >= dw) {
      view_x0_ = data_x0_;
      view_x1_ = data_x1_;
    } else {
      if (view_x0_ < data_x0_) {
        view_x0_ = data_x0_;
        view_x1_ = view_x0_ + vw;
      }
      if (view_x1_ > data_x1_) {
        view_x1_ = data_x1_;
        view_x0_ = view_x1_ - vw;
      }
    }
    if (vh >= dh) {
      view_y0_ = data_y0_;
      view_y1_ = data_y1_;
    } else {
      if (view_y0_ < data_y0_) {
        view_y0_ = data_y0_;
        view_y1_ = view_y0_ + vh;
      }
      if (view_y1_ > data_y1_) {
        view_y1_ = data_y1_;
        view_y0_ = view_y1_ - vh;
      }
    }
    view_initialized_ = true;
  }

  void update_tooltip(const QPoint& pos) {
    if (!has_data()) {
      QToolTip::hideText();
      return;
    }
    const QRect plot = plot_rect(rect());
    if (!plot.contains(pos)) {
      QToolTip::hideText();
      return;
    }
    const double lo = view_y0_;
    const double hi = view_y1_;
    double best_dist = 1e30;
    int best_step = 0;
    double best_loss = 0.0;
    QString best_label;

    auto check_series = [&](const QVector<double>& v, const QString& label) {
      if (v.isEmpty()) return;
      for (int i = 0; i < v.size(); ++i) {
        const QPointF pt = data_to_pixel(plot, static_cast<double>(i), v[i], view_x0_, view_x1_, lo, hi);
        const double dx = pos.x() - pt.x();
        const double dy = pos.y() - pt.y();
        const double dist = dx * dx + dy * dy;
        if (dist < best_dist) {
          best_dist = dist;
          best_step = i;
          best_loss = v[i];
          best_label = label;
        }
      }
    };
    if (compare_mode_) {
      for (const CompareSeries& s : compare_series_) {
        check_series(s.losses, s.label);
      }
    } else {
      check_series(losses_rest_, QStringLiteral("REST"));
      check_series(losses_native_, QStringLiteral("Native"));
      check_series(losses_rest_ema_, QStringLiteral("REST EMA"));
      check_series(losses_native_ema_, QStringLiteral("Native EMA"));
    }

    const QString tip =
        QStringLiteral("(%1, %2) %3").arg(best_step).arg(best_loss, 0, 'g', 6).arg(best_label);
    QToolTip::showText(mapToGlobal(pos), tip, this);
  }

  struct CompareSeries {
    QString label;
    QVector<double> losses;
    QColor color;
  };

  QVector<double> losses_rest_{};
  QVector<double> losses_native_{};
  QVector<double> losses_rest_ema_{};
  QVector<double> losses_native_ema_{};
  bool compare_mode_{false};
  QVector<CompareSeries> compare_series_{};
  bool y_range_locked_{false};
  double y_lock_lo_{-10.0};
  double y_lock_hi_{0.0};
  double data_x0_{0.0};
  double data_x1_{1.0};
  double data_y0_{0.0};
  double data_y1_{1.0};
  double view_x0_{0.0};
  double view_x1_{1.0};
  double view_y0_{0.0};
  double view_y1_{1.0};
  bool view_initialized_{false};
  bool panning_{false};
  QPoint pan_anchor_{};
};

// ── Per-class accuracy bar chart (painted, no Qt Charts dependency) ──────────
class PerClassAccuracyBar final : public QWidget {
 public:
  explicit PerClassAccuracyBar(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(80);
    setMaximumHeight(120);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  }

  /// Update the bar data.  ``labels`` and ``acc`` must be same length.
  void set_data(const QStringList& labels, const QVector<double>& acc) {
    labels_ = labels;
    acc_    = acc;
    update();
  }
  void clear() { labels_.clear(); acc_.clear(); update(); }

 protected:
  void paintEvent(QPaintEvent* e) override {
    Q_UNUSED(e);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QRect full = rect();
    painter.fillRect(full, QColor(28, 30, 34));

    const int K = labels_.size();
    if (K == 0) {
      painter.setPen(QColor(120, 120, 130));
      painter.drawText(full, Qt::AlignCenter, QStringLiteral("Per-class accuracy — no classes yet"));
      return;
    }

    // Margins
    const int margin_l = 4, margin_r = 4, margin_t = 12, margin_b = 20;
    const QRect plot = full.adjusted(margin_l, margin_t, -margin_r, -margin_b);

    const int bar_gap = 2;
    const int bar_w   = std::max(2, (plot.width() - bar_gap * (K - 1)) / K);
    const double max_acc = 100.0;

    // Palette: cycle through 8 distinct colors
    static const QColor kPalette[] = {
        QColor(110, 180, 255), QColor(255, 170,  90), QColor(120, 220, 130),
        QColor(255, 110, 130), QColor(200, 150, 255), QColor(255, 220,  80),
        QColor(90,  210, 210), QColor(200, 200, 200),
    };

    QFont lbl_font = painter.font();
    lbl_font.setPointSizeF(std::max(5.5, lbl_font.pointSizeF() - 2.0));
    painter.setFont(lbl_font);

    for (int k = 0; k < K; ++k) {
      const double acc_k = (k < acc_.size()) ? acc_[k] : 0.0;
      const double frac  = std::max(0.0, std::min(acc_k / max_acc, 1.0));
      const QColor col   = kPalette[k % 8];

      const int bx = plot.left() + k * (bar_w + bar_gap);
      const int bh = static_cast<int>(frac * plot.height());
      const int by = plot.bottom() - bh;

      // Bar background
      painter.fillRect(QRect(bx, plot.top(), bar_w, plot.height()), QColor(45, 48, 54));
      // Bar fill
      painter.fillRect(QRect(bx, by, bar_w, bh), col);

      // Accuracy label above bar
      painter.setPen(QColor(220, 220, 230));
      const QString pct = QString::number(acc_k, 'f', 0) + QLatin1Char('%');
      painter.drawText(QRect(bx - 6, by - 14, bar_w + 12, 14),
                       Qt::AlignHCenter | Qt::AlignVCenter, pct);

      // Class label below
      painter.setPen(col);
      painter.drawText(QRect(bx - 4, plot.bottom() + 3, bar_w + 8, margin_b - 3),
                       Qt::AlignHCenter | Qt::AlignTop,
                       labels_[k].left(6));  // truncate long names
    }

    // Y-axis reference line at 50%
    const int y50 = static_cast<int>(plot.bottom() - 0.5 * plot.height());
    painter.setPen(QPen(QColor(80, 85, 95), 1, Qt::DashLine));
    painter.drawLine(plot.left(), y50, plot.right(), y50);
    painter.setPen(QColor(110, 110, 120));
    painter.drawText(QRect(0, y50 - 8, margin_l + 16, 16), Qt::AlignLeft | Qt::AlignVCenter,
                     QStringLiteral("50%"));
  }

 private:
  QStringList    labels_;
  QVector<double> acc_;
};

#if defined(CYPHA_SHELL_QT_CHARTS)

class LossChartPanel final : public QChartView {
 public:
  explicit LossChartPanel(QWidget* parent = nullptr) : QChartView(parent) {
    setMinimumHeight(128);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setRenderHint(QPainter::Antialiasing, true);
    setMouseTracking(true);
    auto* ch = new QChart();
    ch->setBackgroundBrush(QBrush(QColor(32, 34, 38)));
    ch->setBackgroundRoundness(0);
    ch->setTitleBrush(QBrush(QColor(200, 200, 210)));
    series_rest_ema_ = new QLineSeries();
    series_rest_ema_->setName(QStringLiteral("REST EMA"));
    {
      QPen pr(QColor(110, 180, 255, 200));
      pr.setStyle(Qt::DashLine);
      pr.setWidthF(1.2);
      series_rest_ema_->setPen(pr);
    }
    series_native_ema_ = new QLineSeries();
    series_native_ema_->setName(QStringLiteral("Native EMA"));
    {
      QPen pn(QColor(255, 170, 90, 200));
      pn.setStyle(Qt::DashLine);
      pn.setWidthF(1.2);
      series_native_ema_->setPen(pn);
    }
    series_rest_ = new QLineSeries();
    series_rest_->setName(QStringLiteral("REST"));
    series_rest_->setPen(QPen(QColor(110, 180, 255), 2));
    series_native_ = new QLineSeries();
    series_native_->setName(QStringLiteral("Native"));
    series_native_->setPen(QPen(QColor(255, 170, 90), 2));
    ch->addSeries(series_rest_ema_);
    ch->addSeries(series_native_ema_);
    ch->addSeries(series_rest_);
    ch->addSeries(series_native_);
    ch->legend()->setAlignment(Qt::AlignBottom);
    ch->legend()->setLabelColor(QColor(200, 200, 210));
    ch->legend()->hide();
    setChart(ch);
    const auto connect_hover = [this](QLineSeries* s) {
      connect(s, &QLineSeries::hovered, this, [s, this](const QPointF& pt, bool state) {
        if (state) {
          QToolTip::showText(QCursor::pos(),
                             QStringLiteral("(%1, %2) %3").arg(static_cast<int>(pt.x())).arg(pt.y(), 0, 'g', 6)
                                 .arg(s->name()));
        } else {
          QToolTip::hideText();
        }
      });
    };
    connect_hover(series_rest_);
    connect_hover(series_native_);
    connect_hover(series_rest_ema_);
    connect_hover(series_native_ema_);
  }

  void set_loss_runs(const QVector<double>& rest, const QVector<double>& natv, const QVector<double>& rest_ema,
                     const QVector<double>& natv_ema) {
    series_rest_->clear();
    series_native_->clear();
    series_rest_ema_->clear();
    series_native_ema_->clear();
    QChart* ch = chart();
    const bool have_r = !rest.isEmpty();
    const bool have_n = !natv.isEmpty();
    has_data_ = have_r || have_n;
    if (!have_r && !have_n) {
      ch->setTitle(QStringLiteral("Per-step loss — bulk REST vs bulk native (dashed = EMA)"));
      const QList<QAbstractAxis*> ax = ch->axes();
      for (QAbstractAxis* a : ax) {
        ch->removeAxis(a);
        a->deleteLater();
      }
      ch->legend()->hide();
      view_initialized_ = false;
      return;
    }
    ch->setTitle(QString());
    for (int i = 0; i < rest_ema.size(); ++i) {
      series_rest_ema_->append(i, rest_ema[i]);
    }
    for (int i = 0; i < natv_ema.size(); ++i) {
      series_native_ema_->append(i, natv_ema[i]);
    }
    for (int i = 0; i < rest.size(); ++i) {
      series_rest_->append(i, rest[i]);
    }
    for (int i = 0; i < natv.size(); ++i) {
      series_native_->append(i, natv[i]);
    }
    if (ch->axes().isEmpty()) {
      ch->createDefaultAxes();
      for (QAbstractAxis* ax : ch->axes()) {
        ax->setLabelsColor(QColor(190, 190, 200));
        ax->setLinePenColor(QColor(80, 85, 95));
        ax->setGridLinePen(QPen(QColor(55, 58, 65), 1, Qt::SolidLine));
        if (auto* va = qobject_cast<QValueAxis*>(ax)) {
          va->setTickCount(6);
          va->setLabelFormat(QStringLiteral("%.3g"));
        }
      }
    }
    recompute_data_extents(rest, natv, rest_ema, natv_ema);
    reset_view_to_data();
    apply_view_to_axes();
    ch->legend()->setVisible(true);
  }

  void clear_losses() { set_loss_runs({}, {}, {}, {}); }

  void set_y_range_lock(bool locked, double lo, double hi) {
    y_range_locked_ = locked;
    y_lock_lo_ = lo;
    y_lock_hi_ = hi;
    if (!has_data_) {
      return;
    }
    recompute_data_extents_from_series();
    clamp_view_to_data();
    apply_view_to_axes();
  }

 protected:
  void mouseMoveEvent(QMouseEvent* e) override {
    if (panning_ && chart() != nullptr) {
      const QRectF plot = chart()->plotArea();
      if (plot.width() > 0 && plot.height() > 0) {
        const double dx =
            -static_cast<double>(e->pos().x() - pan_anchor_.x()) / plot.width() * (view_x1_ - view_x0_);
        const double dy =
            static_cast<double>(e->pos().y() - pan_anchor_.y()) / plot.height() * (view_y1_ - view_y0_);
        view_x0_ += dx;
        view_x1_ += dx;
        view_y0_ += dy;
        view_y1_ += dy;
        clamp_view_to_data();
        apply_view_to_axes();
        pan_anchor_ = e->pos();
      }
    }
    QChartView::mouseMoveEvent(e);
  }

  void mousePressEvent(QMouseEvent* e) override {
    if (e->button() == Qt::LeftButton && has_data_) {
      panning_ = true;
      pan_anchor_ = e->pos();
      setCursor(Qt::ClosedHandCursor);
    }
    QChartView::mousePressEvent(e);
  }

  void mouseReleaseEvent(QMouseEvent* e) override {
    if (e->button() == Qt::LeftButton && panning_) {
      panning_ = false;
      unsetCursor();
    }
    QChartView::mouseReleaseEvent(e);
  }

  void leaveEvent(QEvent* e) override {
    QToolTip::hideText();
    if (!panning_) {
      unsetCursor();
    }
    QChartView::leaveEvent(e);
  }

  void wheelEvent(QWheelEvent* e) override {
    if (!has_data_ || chart() == nullptr) {
      e->ignore();
      return;
    }
    const QRectF plot = chart()->plotArea();
    if (!plot.contains(e->position().toPoint()) || plot.width() <= 0 || plot.height() <= 0) {
      e->ignore();
      return;
    }
    const double factor = e->angleDelta().y() > 0 ? 0.85 : 1.15;
    const double frac_x = (e->position().x() - plot.left()) / plot.width();
    const double frac_y = (plot.bottom() - e->position().y()) / plot.height();
    const double cx = view_x0_ + frac_x * (view_x1_ - view_x0_);
    const double cy = view_y0_ + frac_y * (view_y1_ - view_y0_);
    const double half_w = (view_x1_ - view_x0_) * factor * 0.5;
    const double half_h = (view_y1_ - view_y0_) * factor * 0.5;
    view_x0_ = cx - half_w;
    view_x1_ = cx + half_w;
    view_y0_ = cy - half_h;
    view_y1_ = cy + half_h;
    clamp_view_to_data();
    apply_view_to_axes();
    e->accept();
  }

 private:
  void recompute_data_extents(const QVector<double>& rest, const QVector<double>& natv,
                              const QVector<double>& rest_ema, const QVector<double>& natv_ema) {
    const bool have_r = !rest.isEmpty();
    const bool have_n = !natv.isEmpty();
    const int max_n = std::max(rest.size(), natv.size());
    data_x0_ = 0.0;
    data_x1_ = max_n > 1 ? static_cast<double>(max_n - 1) : 1.0;
    if (y_range_locked_) {
      data_y0_ = y_lock_lo_;
      data_y1_ = y_lock_hi_;
      return;
    }
    double lo = have_r ? rest[0] : natv[0];
    double hi = lo;
    auto expand = [&](const QVector<double>& v) {
      for (double x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    };
    if (have_r) expand(rest);
    if (have_n) expand(natv);
    if (!rest_ema.isEmpty()) expand(rest_ema);
    if (!natv_ema.isEmpty()) expand(natv_ema);
    if (!(hi > lo)) hi = lo + 1e-9;
    data_y0_ = lo;
    data_y1_ = hi;
  }

  void recompute_data_extents_from_series() {
    QVector<double> rest;
    QVector<double> natv;
    QVector<double> rest_ema;
    QVector<double> natv_ema;
    for (const QPointF& p : series_rest_->points()) rest.append(p.y());
    for (const QPointF& p : series_native_->points()) natv.append(p.y());
    for (const QPointF& p : series_rest_ema_->points()) rest_ema.append(p.y());
    for (const QPointF& p : series_native_ema_->points()) natv_ema.append(p.y());
    recompute_data_extents(rest, natv, rest_ema, natv_ema);
  }

  void reset_view_to_data() {
    view_x0_ = data_x0_;
    view_x1_ = data_x1_;
    view_y0_ = data_y0_;
    view_y1_ = data_y1_;
    view_initialized_ = true;
  }

  void clamp_view_to_data() {
    if (!has_data_) {
      return;
    }
    recompute_data_extents_from_series();
    const double min_x_span = std::max((data_x1_ - data_x0_) / 200.0, 1.0);
    const double min_y_span = std::max((data_y1_ - data_y0_) / 200.0, 1e-9);
    double vw = view_x1_ - view_x0_;
    double vh = view_y1_ - view_y0_;
    if (vw < min_x_span) {
      const double cx = (view_x0_ + view_x1_) * 0.5;
      vw = min_x_span;
      view_x0_ = cx - vw * 0.5;
      view_x1_ = cx + vw * 0.5;
    }
    if (vh < min_y_span) {
      const double cy = (view_y0_ + view_y1_) * 0.5;
      vh = min_y_span;
      view_y0_ = cy - vh * 0.5;
      view_y1_ = cy + vh * 0.5;
    }
    vw = view_x1_ - view_x0_;
    vh = view_y1_ - view_y0_;
    const double dw = data_x1_ - data_x0_;
    const double dh = data_y1_ - data_y0_;
    if (vw >= dw) {
      view_x0_ = data_x0_;
      view_x1_ = data_x1_;
    } else {
      if (view_x0_ < data_x0_) {
        view_x0_ = data_x0_;
        view_x1_ = view_x0_ + vw;
      }
      if (view_x1_ > data_x1_) {
        view_x1_ = data_x1_;
        view_x0_ = view_x1_ - vw;
      }
    }
    if (vh >= dh) {
      view_y0_ = data_y0_;
      view_y1_ = data_y1_;
    } else {
      if (view_y0_ < data_y0_) {
        view_y0_ = data_y0_;
        view_y1_ = view_y0_ + vh;
      }
      if (view_y1_ > data_y1_) {
        view_y1_ = data_y1_;
        view_y0_ = view_y1_ - vh;
      }
    }
    view_initialized_ = true;
  }

  void apply_view_to_axes() {
    QChart* ch = chart();
    if (!ch) {
      return;
    }
    const QList<QAbstractAxis*> vy = ch->axes(Qt::Vertical);
    if (!vy.isEmpty()) {
      if (auto* va = qobject_cast<QValueAxis*>(vy.front())) {
        va->setRange(view_y0_, view_y1_);
      }
    }
    const QList<QAbstractAxis*> hx = ch->axes(Qt::Horizontal);
    if (!hx.isEmpty()) {
      if (auto* ha = qobject_cast<QValueAxis*>(hx.front())) {
        ha->setRange(view_x0_, view_x1_);
      }
    }
  }

  QLineSeries* series_rest_ema_{};
  QLineSeries* series_native_ema_{};
  QLineSeries* series_rest_{};
  QLineSeries* series_native_{};
  bool has_data_{false};
  bool y_range_locked_{false};
  double y_lock_lo_{-10.0};
  double y_lock_hi_{0.0};
  double data_x0_{0.0};
  double data_x1_{1.0};
  double data_y0_{0.0};
  double data_y1_{1.0};
  double view_x0_{0.0};
  double view_x1_{1.0};
  double view_y0_{0.0};
  double view_y1_{1.0};
  bool view_initialized_{false};
  bool panning_{false};
  QPoint pan_anchor_{};
};

/// Rolling accuracy (window=200, matches training progress panel) — Qt Charts only.
class RollingAccChartPanel final : public QChartView {
 public:
  explicit RollingAccChartPanel(QWidget* parent = nullptr) : QChartView(parent) {
    setMinimumHeight(128);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setRenderHint(QPainter::Antialiasing, true);
    auto* ch = new QChart();
    ch->setBackgroundBrush(QBrush(QColor(32, 34, 38)));
    ch->setBackgroundRoundness(0);
    ch->setTitleBrush(QBrush(QColor(200, 200, 210)));
    ch->setTitle(QStringLiteral("Rolling accuracy (window=200)"));
    series_ = new QLineSeries();
    series_->setName(QStringLiteral("Native"));
    series_->setPen(QPen(QColor(120, 220, 130), 2));
    ch->addSeries(series_);
    ch->legend()->hide();
    setChart(ch);
  }

  void set_series(const QVector<double>& acc_frac) {
    series_->clear();
QChart* ch = chart();
    if (acc_frac.isEmpty()) {
      ch->setTitle(QStringLiteral("Rolling accuracy (window=200)"));
      const QList<QAbstractAxis*> ax = ch->axes();
      for (QAbstractAxis* a : ax) {
        ch->removeAxis(a);
        a->deleteLater();
      }
      return;
    }
    ch->setTitle(QString());
    for (int i = 0; i < acc_frac.size(); ++i) {
      series_->append(i, acc_frac[i]);
    }
    if (ch->axes().isEmpty()) {
      ch->createDefaultAxes();
      for (QAbstractAxis* ax : ch->axes()) {
        ax->setLabelsColor(QColor(190, 190, 200));
        ax->setLinePenColor(QColor(80, 85, 95));
        ax->setGridLinePen(QPen(QColor(55, 58, 65), 1, Qt::SolidLine));
        if (auto* va = qobject_cast<QValueAxis*>(ax)) {
          va->setTickCount(6);
          va->setLabelFormat(QStringLiteral("%.2f"));
        }
      }
    }
    const QList<QAbstractAxis*> vy = ch->axes(Qt::Vertical);
    if (!vy.isEmpty()) {
      if (auto* va = qobject_cast<QValueAxis*>(vy.front())) {
        va->setRange(0.0, 1.0);
      }
    }
    const QList<QAbstractAxis*> hx = ch->axes(Qt::Horizontal);
    if (!hx.isEmpty()) {
      if (auto* ha = qobject_cast<QValueAxis*>(hx.front())) {
        ha->setRange(0.0, acc_frac.size() > 1 ? static_cast<double>(acc_frac.size() - 1) : 1.0);
      }
    }
  }

 private:
QLineSeries* series_{};
};
#endif

/// Text metrics log when Qt Charts is not linked (CI qt6-base-only builds).
class MetricsHistoryPanel final : public QPlainTextEdit {
 public:
  explicit MetricsHistoryPanel(QWidget* parent = nullptr) : QPlainTextEdit(parent) {
    setReadOnly(true);
    setMaximumBlockCount(500);
    setMinimumHeight(96);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setPlaceholderText(QStringLiteral(
        "Per-step metrics — run bulk REST /update or bulk native train.\n"
        "Columns: step | loss | roll_acc (native) | source"));
    QFont mono = font();
    mono.setStyleHint(QFont::Monospace);
    mono.setFamily(QStringLiteral("Consolas"));
    setFont(mono);
  }

  void append_entry(int step, double loss, double roll_acc_frac, bool have_roll_acc,
                    const QString& source, bool correct) {
    QString line = QStringLiteral("step %1 | loss %2").arg(step).arg(loss, 0, 'g', 6);
    if (have_roll_acc) {
      line += QStringLiteral(" | roll_acc %1%").arg(roll_acc_frac * 100.0, 0, 'f', 1);
    }
    line += QStringLiteral(" | %1").arg(source);
    if (source == QStringLiteral("native")) {
      line += correct ? QStringLiteral(" \u2713") : QStringLiteral(" \u2717");
    }
    appendPlainText(line);
    moveCursor(QTextCursor::End);
  }

  void clear_history() { clear(); }
};

/// Loss + metrics: Qt Charts tabs when available; painted loss + text history otherwise.
class LossMetricsPanel final : public QWidget {
 public:
  explicit LossMetricsPanel(QWidget* parent = nullptr) : QWidget(parent) {
    auto* lay = new QVBoxLayout(this);
    lay->setContentsMargins(0, 0, 0, 0);
#if defined(CYPHA_SHELL_QT_CHARTS)
    tabs_ = new QTabWidget(this);
    loss_chart_ = new LossChartPanel(tabs_);
    acc_chart_  = new RollingAccChartPanel(tabs_);
    tabs_->addTab(loss_chart_, QStringLiteral("Loss"));
    tabs_->addTab(acc_chart_, QStringLiteral("Rolling accuracy"));
    lay->addWidget(tabs_);
#else
    loss_chart_ = new SimpleLossChart(this);
    metrics_text_ = new MetricsHistoryPanel(this);
    lay->addWidget(loss_chart_);
    lay->addWidget(metrics_text_);
#endif
  }

  QWidget* capture_widget() const {
#if defined(CYPHA_SHELL_QT_CHARTS)
    return tabs_->currentWidget();
#else
    return loss_chart_;
#endif
  }

  void set_loss_runs(const QVector<double>& rest, const QVector<double>& natv, const QVector<double>& rest_ema,
                     const QVector<double>& natv_ema) {
    loss_chart_->set_loss_runs(rest, natv, rest_ema, natv_ema);
  }

  void clear_losses() {
    loss_chart_->clear_losses();
    roll_acc_.clear();
    roll_correct_window_.clear();
#if defined(CYPHA_SHELL_QT_CHARTS)
    acc_chart_->set_series({});
#else
    metrics_text_->clear_history();
#endif
  }

  void set_y_range_lock(bool locked, double lo, double hi) { loss_chart_->set_y_range_lock(locked, lo, hi); }

  void append_rest_step(int step_idx, double loss) {
#if !defined(CYPHA_SHELL_QT_CHARTS)
    metrics_text_->append_entry(step_idx, loss, 0.0, false, QStringLiteral("REST"), false);
#else
    Q_UNUSED(step_idx);
    Q_UNUSED(loss);
#endif
  }

  void append_native_step(int step, double loss, bool correct) {
    roll_correct_window_.append(correct ? 1 : 0);
    while (roll_correct_window_.size() > kRollWin) {
      roll_correct_window_.removeFirst();
    }
    int sum = 0;
    for (int v : roll_correct_window_) {
      sum += v;
    }
    const double roll = static_cast<double>(sum) / static_cast<double>(roll_correct_window_.size());
    roll_acc_.append(roll);
#if defined(CYPHA_SHELL_QT_CHARTS)
    acc_chart_->set_series(roll_acc_);
#else
    metrics_text_->append_entry(step, loss, roll, true, QStringLiteral("native"), correct);
#endif
  }

  void reset_native_roll_window() { roll_correct_window_.clear(); roll_acc_.clear(); }

  void refresh_acc_chart() {
#if defined(CYPHA_SHELL_QT_CHARTS)
    acc_chart_->set_series(roll_acc_);
#endif
  }

 private:
  static constexpr int kRollWin = 200;
  QVector<double> roll_acc_{};
  QVector<int>    roll_correct_window_{};
#if defined(CYPHA_SHELL_QT_CHARTS)
  QTabWidget*           tabs_{};
  LossChartPanel*       loss_chart_{};
  RollingAccChartPanel* acc_chart_{};
#else
  SimpleLossChart*      loss_chart_{};
  MetricsHistoryPanel*  metrics_text_{};
#endif
};

enum class LossPlotSource { RestBulk, NativeBulk };

constexpr int kBulkChartRefreshEvery = 50;
constexpr int kBulkProgressRefreshEvery = 10;

using BulkLogEntry = BulkNativeTrainLogEntry;

QString native_quickstart_html() {
  return QStringLiteral(
      "<h1>Native quick start (v2.4)</h1>"
      "<p>Install &rarr; validate &rarr; bench &rarr; tune &rarr; REST. Python is optional after install.</p>"
      "<h2>1. Build</h2>"
      "<pre>cmake -S native -B &lt;build-dir&gt; -DCMAKE_BUILD_TYPE=Release "
      "-DCYPHA_BUILD_QT=ON -DCYPHA_BUILD_EXPERIMENT_DB=ON -G Ninja\ncmake --build &lt;build-dir&gt; --parallel</pre>"
      "<p>Build outside OneDrive on Windows (cloud sync locks object files), e.g. "
      "<code>C:\\Temp\\cypha_native_build</code>. Full instructions: "
      "<code>docs/native/CYPHALM_NATIVE_BUILD.md</code>.</p>"
      "<h2>2. Validate</h2>"
      "<pre>powershell -File scripts\\cypha_native_validate_all.ps1</pre>"
      "<p>Or: <code>ctest -R native_</code> and REST smoke.</p>"
      "<h2>3. Bench</h2>"
      "<pre>cypha_bench_run --list-domains\ncypha_bench_run --domain 1\ncypha_bench_run --report-only</pre>"
      "<h2>4. Tune</h2>"
      "<pre>cypha_tune_run --config bench/config/cyphalm_hybrid_lstm_tune_smoke.json --dry-run</pre>"
      "<h2>5. REST</h2>"
      "<pre>cypha_rest --listen 127.0.0.1:8099 --cypha model.cypha --registry ~/.cypha/models</pre>"
      "<p>CyphaDIF: <code>GET /health</code>, <code>GET /ready</code>, <code>POST /predict</code>, "
      "<code>POST /update</code>, <code>GET /models</code>, <code>POST /load</code>.</p>"
      "<p>CyphaLM: <code>POST /lm/load</code>, <code>POST /generate</code>.</p>"
      "<h2>6. Qt shell tabs</h2>"
      "<ol><li>Data &mdash; CSV inspect (dataset_widget)</li>"
      "<li>Model &mdash; F_field / preprocessor sidecars</li>"
      "<li>Train &mdash; bulk native/REST, loss charts</li>"
      "<li>Predict &mdash; confidence + explanation panel</li>"
      "<li>Registry &mdash; ModelRegistry scan / load / REST /load</li>"
      "<li>Server &mdash; spawn cypha_rest</li>"
      "<li>Experiments &mdash; SQLite experiment DB (when enabled)</li>"
      "<li>CyphaLM &mdash; checkpoint decode</li>"
      "<li>Help &mdash; this page</li></ol>"
      "<p>Full doc: <code>docs/native/NATIVE_QUICKSTART.md</code></p>");
}

QString default_registry_root() {
  const QString home = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
  return home + QStringLiteral("/.cypha/models");
}

QString cypha_config_dir() {
  return QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + QStringLiteral("/.cypha");
}

QString shell_settings_path() { return cypha_config_dir() + QStringLiteral("/shell_settings.json"); }

void apply_cypha_theme(bool dark) {
  QApplication::setStyle(QStringLiteral("Fusion"));
  if (!dark) {
    QApplication::setPalette(QApplication::style()->standardPalette());
    return;
  }
  QPalette palette;
  palette.setColor(QPalette::Window, QColor(53, 53, 53));
  palette.setColor(QPalette::WindowText, Qt::white);
  palette.setColor(QPalette::Base, QColor(25, 25, 25));
  palette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
  palette.setColor(QPalette::ToolTipBase, Qt::white);
  palette.setColor(QPalette::ToolTipText, Qt::white);
  palette.setColor(QPalette::Text, Qt::white);
  palette.setColor(QPalette::Button, QColor(53, 53, 53));
  palette.setColor(QPalette::ButtonText, Qt::white);
  palette.setColor(QPalette::BrightText, Qt::red);
  palette.setColor(QPalette::Link, QColor(42, 130, 218));
  palette.setColor(QPalette::Highlight, QColor(42, 130, 218));
  palette.setColor(QPalette::HighlightedText, Qt::black);
  QApplication::setPalette(palette);
}

struct StudioPreferences {
  bool inference_use_gh = true;
  double inference_ood_threshold = 3.0;
  double inference_chi = 1.0;
  double inference_psi = 1.0;
  QString registry_root_override;
  int csv_chunk_rows_override = 0;
  QString dataset_dialog_start_dir;
  int ui_font_pt = 0;
  bool dark_theme = false;

  QString effective_registry_root() const {
    const QString r = registry_root_override.trimmed();
    return r.isEmpty() ? default_registry_root() : r;
  }
};

QSettings studio_qsettings() { return QSettings(QStringLiteral("Cypha"), QStringLiteral("CyphaStudio")); }

void merge_shell_settings_json(StudioPreferences& p) {
  QFile f(shell_settings_path());
  if (!f.open(QIODevice::ReadOnly)) {
    return;
  }
  QJsonParseError pe{};
  const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &pe);
  if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
    return;
  }
  const QJsonObject o = doc.object();
  if (o.contains(QStringLiteral("dark_theme"))) {
    p.dark_theme = o.value(QStringLiteral("dark_theme")).toBool();
  }
}

void write_shell_settings_json(const StudioPreferences& p) {
  QDir().mkpath(cypha_config_dir());
  QJsonObject o;
  o[QStringLiteral("dark_theme")] = p.dark_theme;
  QFile f(shell_settings_path());
  if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
    f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
  }
}

StudioPreferences load_studio_preferences() {
  StudioPreferences p;
  QSettings s = studio_qsettings();
  const auto read_bool = [&](const char* key, bool def) {
    const QVariant v = s.value(QString::fromLatin1(key), def);
    if (v.typeId() == QMetaType::Bool) {
      return v.toBool();
    }
    return QString(v.toString()).toLower() == QLatin1String("true") ||
           QString(v.toString()) == QLatin1String("1");
  };
  const auto read_int = [&](const char* key, int def) {
    bool ok = false;
    const int v = s.value(QString::fromLatin1(key), def).toInt(&ok);
    return ok ? v : def;
  };
  const auto read_dbl = [&](const char* key, double def) {
    bool ok = false;
    const double v = s.value(QString::fromLatin1(key), def).toDouble(&ok);
    return ok ? v : def;
  };
  p.inference_use_gh = read_bool("studio/inference_use_gh", p.inference_use_gh);
  p.inference_ood_threshold = read_dbl("studio/inference_ood_threshold", p.inference_ood_threshold);
  p.inference_chi = read_dbl("studio/inference_chi", p.inference_chi);
  p.inference_psi = read_dbl("studio/inference_psi", p.inference_psi);
  p.registry_root_override = s.value(QStringLiteral("studio/registry_root_override"), QString()).toString();
  p.csv_chunk_rows_override = read_int("studio/csv_chunk_rows_override", 0);
  p.dataset_dialog_start_dir = s.value(QStringLiteral("studio/dataset_dialog_start_dir"), QString()).toString();
  p.ui_font_pt = read_int("studio/ui_font_pt", 0);
  p.dark_theme = read_bool("studio/dark_theme", p.dark_theme);
  merge_shell_settings_json(p);
  return p;
}

void save_studio_preferences(const StudioPreferences& p) {
  QSettings s = studio_qsettings();
  s.setValue(QStringLiteral("studio/inference_use_gh"), p.inference_use_gh);
  s.setValue(QStringLiteral("studio/inference_ood_threshold"), p.inference_ood_threshold);
  s.setValue(QStringLiteral("studio/inference_chi"), p.inference_chi);
  s.setValue(QStringLiteral("studio/inference_psi"), p.inference_psi);
  s.setValue(QStringLiteral("studio/registry_root_override"), p.registry_root_override);
  s.setValue(QStringLiteral("studio/csv_chunk_rows_override"), p.csv_chunk_rows_override);
  s.setValue(QStringLiteral("studio/dataset_dialog_start_dir"), p.dataset_dialog_start_dir);
  s.setValue(QStringLiteral("studio/ui_font_pt"), p.ui_font_pt);
  s.setValue(QStringLiteral("studio/dark_theme"), p.dark_theme);
  write_shell_settings_json(p);
}

struct ConfusionMatrix {
  QStringList labels;
  QVector<QVector<int>> counts;
};

ConfusionMatrix confusion_counts(const QVector<QString>& y_true, const QVector<QString>& y_pred) {
  QSet<QString> label_set;
  for (const QString& s : y_true) {
    label_set.insert(s);
  }
  for (const QString& s : y_pred) {
    label_set.insert(s);
  }
  QStringList labels = label_set.values();
  std::sort(labels.begin(), labels.end());
  const int n = labels.size();
  QHash<QString, int> idx;
  for (int i = 0; i < n; ++i) {
    idx[labels[i]] = i;
  }
  QVector<QVector<int>> cm(n, QVector<int>(n, 0));
  const int m = std::min(y_true.size(), y_pred.size());
  for (int k = 0; k < m; ++k) {
    const int ti = idx.value(y_true[k], -1);
    const int pi = idx.value(y_pred[k], -1);
    if (ti >= 0 && pi >= 0) {
      cm[ti][pi] += 1;
    }
  }
  return {labels, cm};
}

void show_confusion_dialog(QWidget* parent, const QVector<QString>& y_true, const QVector<QString>& y_pred,
                           const QString& title = QStringLiteral("Confusion matrix (test set)")) {
  if (y_true.isEmpty() || y_pred.isEmpty()) {
    QMessageBox::information(parent, QStringLiteral("Confusion matrix"),
                             QStringLiteral("No labeled batch-predict results. Use a CSV with a target column."));
    return;
  }
  const ConfusionMatrix cm = confusion_counts(y_true, y_pred);
  const int n = cm.labels.size();
  if (n == 0) {
    return;
  }
  int max_count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      max_count = std::max(max_count, cm.counts[i][j]);
    }
  }
  QDialog dlg(parent);
  dlg.setWindowTitle(title);
  dlg.resize(std::min(80 + n * 72, 900), std::min(120 + n * 28, 700));
  auto* lay = new QVBoxLayout(&dlg);
  auto* tbl = new QTableWidget(n, n, &dlg);
  QStringList hhdr;
  QStringList vhdr;
  for (const QString& lb : cm.labels) {
    hhdr << QStringLiteral("pred %1").arg(lb);
    vhdr << QStringLiteral("true %1").arg(lb);
  }
  tbl->setHorizontalHeaderLabels(hhdr);
  tbl->setVerticalHeaderLabels(vhdr);
  tbl->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  tbl->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const int c = cm.counts[i][j];
      auto* item = new QTableWidgetItem(QString::number(c));
      item->setTextAlignment(Qt::AlignCenter);
      if (c > 0 && max_count > 0) {
        const double t = static_cast<double>(c) / static_cast<double>(max_count);
        const int alpha = static_cast<int>(40 + t * 180);
        if (i == j) {
          item->setBackground(QBrush(QColor(40, 160, 90, alpha)));
        } else {
          item->setBackground(QBrush(QColor(200, 70, 70, alpha)));
        }
      }
      tbl->setItem(i, j, item);
    }
  }
  lay->addWidget(tbl);
  auto* box = new QDialogButtonBox(QDialogButtonBox::Close, &dlg);
  QObject::connect(box, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
  QObject::connect(box, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
  lay->addWidget(box);
  dlg.exec();
}

struct ModelCardEditResult {
  QString name;
  QString version;
};

class ModelCardEditorDialog final : public QDialog {
 public:
  explicit ModelCardEditorDialog(const QString& card_path, const QString& default_name,
                                 QWidget* parent = nullptr)
      : QDialog(parent), card_path_(card_path) {
    setWindowTitle(QStringLiteral("Model card editor"));
    resize(480, 360);

    QFile f(card_path_);
    if (f.open(QIODevice::ReadOnly)) {
      QJsonParseError pe{};
      const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &pe);
      if (pe.error == QJsonParseError::NoError && doc.isObject()) {
        card_obj_ = doc.object();
      }
    }

    auto* root = new QVBoxLayout(this);
    auto* form = new QFormLayout();
    name_edit_ = new QLineEdit(this);
    name_edit_->setText(card_obj_.value(QStringLiteral("name")).toString(default_name));
    form->addRow(QStringLiteral("Name:"), name_edit_);

    version_edit_ = new QLineEdit(this);
    version_edit_->setText(card_obj_.value(QStringLiteral("version")).toString(QStringLiteral("1.0.0")));
    form->addRow(QStringLiteral("Version:"), version_edit_);

    desc_edit_ = new QPlainTextEdit(this);
    desc_edit_->setPlainText(card_obj_.value(QStringLiteral("description")).toString());
    desc_edit_->setMaximumHeight(100);
    form->addRow(QStringLiteral("Description:"), desc_edit_);

    tags_edit_ = new QLineEdit(this);
    QStringList tag_parts;
    const QJsonValue tags_v = card_obj_.value(QStringLiteral("tags"));
    if (tags_v.isArray()) {
      for (const QJsonValue& tv : tags_v.toArray()) {
        const QString t = tv.toString().trimmed();
        if (!t.isEmpty()) {
          tag_parts << t;
        }
      }
    } else if (tags_v.isString()) {
      tag_parts = tags_v.toString().split(QLatin1Char(','), Qt::SkipEmptyParts);
      for (QString& t : tag_parts) {
        t = t.trimmed();
      }
    }
    tags_edit_->setText(tag_parts.join(QStringLiteral(", ")));
    tags_edit_->setPlaceholderText(QStringLiteral("comma-separated tags"));
    form->addRow(QStringLiteral("Tags:"), tags_edit_);

    task_combo_ = new QComboBox(this);
    task_combo_->addItems({QStringLiteral("classification"), QStringLiteral("regression"),
                           QStringLiteral("generation"), QStringLiteral("other")});
    const QString task = card_obj_.value(QStringLiteral("task")).toString(QStringLiteral("classification"));
    const int ti = task_combo_->findText(task);
    task_combo_->setCurrentIndex(ti >= 0 ? ti : 0);
    form->addRow(QStringLiteral("Task type:"), task_combo_);

    root->addLayout(form);
    auto* note = new QLabel(
        QStringLiteral("Edits are written to card.json before registry register."), this);
    note->setWordWrap(true);
    note->setStyleSheet(QStringLiteral("color: #888; font-size: 11px;"));
    root->addWidget(note);

    auto* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(btns, &QDialogButtonBox::accepted, this, &ModelCardEditorDialog::accept);
    connect(btns, &QDialogButtonBox::rejected, this, &ModelCardEditorDialog::reject);
    root->addWidget(btns);
  }

  ModelCardEditResult edit_result() const { return result_; }

  bool write_card(QString* err = nullptr) const {
    QFileInfo fi(card_path_);
    if (!fi.absoluteDir().exists() && !QDir().mkpath(fi.absolutePath())) {
      if (err != nullptr) {
        *err = QStringLiteral("Could not create directory for card.json");
      }
      return false;
    }
    QFile f(card_path_);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
      if (err != nullptr) {
        *err = QStringLiteral("Could not write %1").arg(card_path_);
      }
      return false;
    }
    f.write(QJsonDocument(card_obj_).toJson(QJsonDocument::Indented));
    return true;
  }

 private:
  void accept() override {
    const QString name = name_edit_->text().trimmed();
    const QString version = version_edit_->text().trimmed();
    if (name.isEmpty() || version.isEmpty()) {
      QMessageBox::warning(this, QStringLiteral("Model card"),
                           QStringLiteral("Name and version are required."));
      return;
    }
    card_obj_[QStringLiteral("name")] = name;
    card_obj_[QStringLiteral("version")] = version;
    card_obj_[QStringLiteral("description")] = desc_edit_->toPlainText();
    card_obj_[QStringLiteral("task")] = task_combo_->currentText();

    QJsonArray tags_arr;
    for (const QString& part : tags_edit_->text().split(QLatin1Char(','), Qt::SkipEmptyParts)) {
      const QString t = part.trimmed();
      if (!t.isEmpty()) {
        tags_arr.append(t);
      }
    }
    card_obj_[QStringLiteral("tags")] = tags_arr;

    result_.name = name;
    result_.version = version;
    QString write_err;
    if (!write_card(&write_err)) {
      QMessageBox::warning(this, QStringLiteral("Model card"), write_err);
      return;
    }
    QDialog::accept();
  }

  QString card_path_;
  ModelCardEditResult result_;
  QJsonObject card_obj_;
  QLineEdit* name_edit_{};
  QLineEdit* version_edit_{};
  QPlainTextEdit* desc_edit_{};
  QLineEdit* tags_edit_{};
  QComboBox* task_combo_{};
};

class SettingsDialog final : public QDialog {
 public:
  explicit SettingsDialog(StudioPreferences prefs, QWidget* parent = nullptr) : QDialog(parent), prefs_(std::move(prefs)) {
    setWindowTitle(QStringLiteral("Studio Settings"));
    resize(520, 420);
    auto* root = new QVBoxLayout(this);
    auto* tabs = new QTabWidget(this);
    root->addWidget(tabs);
    tabs->addTab(build_inference_tab(), QStringLiteral("Inference"));
    tabs->addTab(build_paths_tab(), QStringLiteral("Paths & data"));
    tabs->addTab(build_appearance_tab(), QStringLiteral("Appearance"));
    tabs->addTab(build_environment_tab(), QStringLiteral("Environment"));
    auto* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel |
                                          QDialogButtonBox::RestoreDefaults,
                                      this);
    connect(btns, &QDialogButtonBox::accepted, this, &SettingsDialog::accept);
    connect(btns, &QDialogButtonBox::rejected, this, &SettingsDialog::reject);
    connect(btns->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked, this,
            &SettingsDialog::restore_defaults);
    root->addWidget(btns);
  }

  StudioPreferences result_preferences() const { return prefs_; }

 private:
  QWidget* build_inference_tab() {
    auto* w = new QWidget(this);
    auto* lay = new QVBoxLayout(w);
    auto* grp = new QGroupBox(QStringLiteral("Classification inference pipeline"), w);
    auto* form = new QFormLayout(grp);
    use_gh_ = new QCheckBox(QStringLiteral("Use GH inference path (when model supports it)"), grp);
    use_gh_->setChecked(prefs_.inference_use_gh);
    form->addRow(use_gh_);
    ood_ = new QDoubleSpinBox(grp);
    ood_->setRange(0.1, 100.0);
    ood_->setDecimals(2);
    ood_->setValue(prefs_.inference_ood_threshold);
    form->addRow(QStringLiteral("OOD threshold:"), ood_);
    chi_ = new QDoubleSpinBox(grp);
    chi_->setRange(0.01, 100.0);
    chi_->setDecimals(4);
    chi_->setValue(prefs_.inference_chi);
    form->addRow(QStringLiteral("GH χ:"), chi_);
    psi_ = new QDoubleSpinBox(grp);
    psi_->setRange(0.01, 100.0);
    psi_->setDecimals(4);
    psi_->setValue(prefs_.inference_psi);
    form->addRow(QStringLiteral("GH ψ:"), psi_);
    lay->addWidget(grp);
    auto* note = new QLabel(QStringLiteral("Changes apply to the next prediction (OOD threshold and χ/ψ)."), w);
    note->setWordWrap(true);
    note->setStyleSheet(QStringLiteral("color: #888; font-size: 11px;"));
    lay->addWidget(note);
    lay->addStretch();
    return w;
  }

  QWidget* build_paths_tab() {
    auto* w = new QWidget(this);
    auto* lay = new QVBoxLayout(w);
    lay->addWidget(new QLabel(QStringLiteral("<b>Model registry root</b>"), w));
    auto* reg_row = new QHBoxLayout();
    reg_edit_ = new QLineEdit(prefs_.registry_root_override, w);
    reg_edit_->setPlaceholderText(QStringLiteral("Default: %1").arg(default_registry_root()));
    auto* reg_br = new QPushButton(QStringLiteral("Browse…"), w);
    connect(reg_br, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Registry root"), reg_edit_->text());
      if (!path.isEmpty()) {
        reg_edit_->setText(path);
      }
    });
    reg_row->addWidget(reg_edit_);
    reg_row->addWidget(reg_br);
    lay->addLayout(reg_row);
    csv_chunk_ = new QSpinBox(w);
    csv_chunk_->setRange(0, 10'000'000);
    csv_chunk_->setSpecialValueText(QStringLiteral("From env / full file"));
    csv_chunk_->setValue(prefs_.csv_chunk_rows_override > 0 ? prefs_.csv_chunk_rows_override : 0);
    auto* form = new QFormLayout();
    form->addRow(QStringLiteral("CSV chunk rows (0 = auto):"), csv_chunk_);
    lay->addLayout(form);
    lay->addWidget(new QLabel(QStringLiteral("<b>Dataset file dialog start folder</b>"), w));
    auto* ds_row = new QHBoxLayout();
    ds_dir_ = new QLineEdit(prefs_.dataset_dialog_start_dir, w);
    ds_dir_->setPlaceholderText(QStringLiteral("Use recent-path history"));
    auto* ds_br = new QPushButton(QStringLiteral("Browse…"), w);
    connect(ds_br, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Dataset dialog folder"),
                                                             ds_dir_->text());
      if (!path.isEmpty()) {
        ds_dir_->setText(path);
      }
    });
    ds_row->addWidget(ds_dir_);
    ds_row->addWidget(ds_br);
    lay->addLayout(ds_row);
    lay->addStretch();
    return w;
  }

  QWidget* build_appearance_tab() {
    auto* w = new QWidget(this);
    auto* lay = new QVBoxLayout(w);
    dark_theme_ = new QCheckBox(QStringLiteral("Dark theme (Fusion palette)"), w);
    dark_theme_->setChecked(prefs_.dark_theme);
    lay->addWidget(dark_theme_);
    font_pt_ = new QSpinBox(w);
    font_pt_->setRange(0, 24);
    font_pt_->setValue(prefs_.ui_font_pt);
    font_pt_->setToolTip(QStringLiteral("0 = do not change application font. 9–14 recommended."));
    auto* form = new QFormLayout();
    form->addRow(QStringLiteral("UI font size (pt, 0 = default):"), font_pt_);
    lay->addLayout(form);
    lay->addStretch();
    return w;
  }

  QWidget* build_environment_tab() {
    auto* w = new QWidget(this);
    auto* lay = new QVBoxLayout(w);
    const QString txt = QStringLiteral(
        "<p><b>Variables</b> (see <code>docs/studio/CYPHA_ENV.md</code>)</p>"
        "<table cellspacing='8' style='font-size:12px'>"
        "<tr><td><code>CYPHA_REGISTRY_ROOT</code></td><td>Model tree root</td></tr>"
        "<tr><td><code>CYPHA_API_HOST</code></td><td>REST bind host</td></tr>"
        "<tr><td><code>CYPHA_API_PORT</code></td><td>REST port</td></tr>"
        "<tr><td><code>CYPHA_CORS_ORIGINS</code></td><td>CORS allow list</td></tr>"
        "<tr><td><code>CYPHA_CSV_CHUNK_ROWS</code></td><td>CSV streaming chunk size</td></tr>"
        "</table>"
        "<p><b>Effective registry</b> (read-only):<br><code>%1</code></p>")
                            .arg(default_registry_root());
    auto* lbl = new QLabel(txt, w);
    lbl->setWordWrap(true);
    lbl->setTextFormat(Qt::RichText);
    lay->addWidget(lbl);
    lay->addStretch();
    return w;
  }

  void restore_defaults() {
    const StudioPreferences d;
    use_gh_->setChecked(d.inference_use_gh);
    ood_->setValue(d.inference_ood_threshold);
    chi_->setValue(d.inference_chi);
    psi_->setValue(d.inference_psi);
    reg_edit_->clear();
    csv_chunk_->setValue(0);
    ds_dir_->clear();
    font_pt_->setValue(0);
    dark_theme_->setChecked(d.dark_theme);
  }

  void accept() override {
    prefs_.inference_use_gh = use_gh_->isChecked();
    prefs_.inference_ood_threshold = ood_->value();
    prefs_.inference_chi = chi_->value();
    prefs_.inference_psi = psi_->value();
    prefs_.registry_root_override = reg_edit_->text().trimmed();
    const int cr = csv_chunk_->value();
    prefs_.csv_chunk_rows_override = cr > 0 ? cr : 0;
    prefs_.dataset_dialog_start_dir = ds_dir_->text().trimmed();
    prefs_.ui_font_pt = font_pt_->value();
    prefs_.dark_theme = dark_theme_->isChecked();
    save_studio_preferences(prefs_);
    QDialog::accept();
  }

  StudioPreferences prefs_;
  QCheckBox* use_gh_{};
  QDoubleSpinBox* ood_{};
  QDoubleSpinBox* chi_{};
  QDoubleSpinBox* psi_{};
  QLineEdit* reg_edit_{};
  QSpinBox* csv_chunk_{};
  QLineEdit* ds_dir_{};
  QSpinBox* font_pt_{};
  QCheckBox* dark_theme_{};
};

class MainWindow final : public QMainWindow {
 public:
  MainWindow() {
    studio_prefs_ = load_studio_preferences();
    apply_cypha_theme(studio_prefs_.dark_theme);
    native_gh_chi_ = studio_prefs_.inference_chi;
    native_gh_psi_ = studio_prefs_.inference_psi;
    if (studio_prefs_.ui_font_pt > 0) {
      QFont f = QApplication::font();
      f.setPointSize(studio_prefs_.ui_font_pt);
      QApplication::setFont(f);
    }
    setWindowTitle(QStringLiteral("Cypha — Qt shell"));
    auto* central = new QWidget(this);
    auto* main_layout = new QVBoxLayout(central);
    main_layout->setSpacing(8);
    main_layout->setContentsMargins(10, 10, 10, 10);
    setMinimumSize(920, 920);

    workflow_banner_ = new QLabel(
        QStringLiteral(
            "<p style='margin:0'><b>Workflow</b> &mdash; <b>1 Data</b> &rarr; <b>2 Model</b> &rarr; "
            "<b>3 Train</b> &rarr; <b>4 Predict</b> &rarr; <b>5 Registry</b> &rarr; <b>6 Server</b>"
            " &rarr; <b>7 Experiments</b> (when enabled) &rarr; <b>8 CyphaLM</b> &rarr; <b>9 Help</b></p>"
            "<p style='margin:8px 0 0 0; color:#c8d6e8; font-size:12px'>Work left to right. The window "
            "geometry and active tab are saved when you quit.</p>"),
        central);
    workflow_banner_->setWordWrap(true);
    workflow_banner_->setTextFormat(Qt::RichText);
    workflow_banner_->setStyleSheet(
        QStringLiteral("QLabel { background-color: #152535; color: #f0f5fa; padding: 14px 18px; "
                       "border-radius: 10px; border: 1px solid #2a4158; }"));
    main_layout->addWidget(workflow_banner_);

    auto* header_bar = new QWidget(central);
    auto* row1 = new QHBoxLayout(header_bar);
    load_btn_ = new QPushButton(QStringLiteral("Load model…"), header_bar);
    new_model_btn_ = new QPushButton(QStringLiteral("New model…"), header_bar);
    new_model_btn_->setToolTip(QStringLiteral(
        "Create a fresh empty model in-memory (no existing .cypha needed).\n"
        "Specify input_dim and field_dim, then train directly from CSV."));
    path_label_ = new QLabel(QStringLiteral("(no model)"), header_bar);
    path_label_->setWordWrap(true);
    row1->addWidget(load_btn_);
    row1->addWidget(new_model_btn_);
    row1->addWidget(path_label_, 1);
    main_layout->addWidget(header_bar);

    main_tabs_ = new QTabWidget(central);
    main_tabs_->setMovable(false);
    main_tabs_->setDocumentMode(true);
    main_tabs_->setUsesScrollButtons(true);
    main_layout->addWidget(main_tabs_, 1);

    auto make_page = [&](const QString& title) -> std::pair<QWidget*, QVBoxLayout*> {
      auto* scr = new QScrollArea(main_tabs_);
      scr->setWidgetResizable(true);
      scr->setFrameShape(QFrame::NoFrame);
      scr->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
      auto* inner = new QWidget(scr);
      auto* lay = new QVBoxLayout(inner);
      lay->setSpacing(10);
      lay->setContentsMargins(6, 6, 6, 14);
      inner->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
      scr->setWidget(inner);
      main_tabs_->addTab(scr, title);
      return {inner, lay};
    };

    const auto pg_data = make_page(QStringLiteral("1 · Data"));
    QWidget* const inner_data = pg_data.first;
    QVBoxLayout* const lay_data = pg_data.second;

    const auto pg_model = make_page(QStringLiteral("2 · Model"));
    QWidget* const inner_model = pg_model.first;
    QVBoxLayout* const lay_model = pg_model.second;

    const auto pg_train = make_page(QStringLiteral("3 · Train"));
    QWidget* const inner_train = pg_train.first;
    QVBoxLayout* const lay_train = pg_train.second;

    const auto pg_predict = make_page(QStringLiteral("4 · Predict"));
    QWidget* const inner_predict = pg_predict.first;
    QVBoxLayout* const lay_predict = pg_predict.second;

    const auto pg_registry = make_page(QStringLiteral("5 · Registry"));
    QWidget* const inner_registry = pg_registry.first;
    QVBoxLayout* const lay_registry = pg_registry.second;

    const auto pg_server = make_page(QStringLiteral("6 · Server"));
    QWidget* const inner_server = pg_server.first;
    QVBoxLayout* const lay_server = pg_server.second;

    {
      auto* model_intro = new QLabel(
          QStringLiteral("<b>Sidecars</b> &mdash; add <tt>F_field</tt> JSON if the blob has no embedded "
                         "<tt>world.F_field</tt>; <tt>preprocessor.json</tt> matches <tt>cypha_rest --pre</tt>."),
          inner_model);
      model_intro->setWordWrap(true);
      model_intro->setTextFormat(Qt::RichText);
      lay_model->addWidget(model_intro);
    }

    auto* row_ff = new QHBoxLayout();
    ff_btn_ = new QPushButton(QStringLiteral("F field JSON…"), inner_model);
    ff_label_ = new QLabel(QStringLiteral("(optional, if world.F_field not in .cypha)"), inner_model);
    ff_label_->setWordWrap(true);
    row_ff->addWidget(ff_btn_);
    row_ff->addWidget(ff_label_, 1);
    lay_model->addLayout(row_ff);

    auto* row_pre = new QHBoxLayout();
    pre_btn_ = new QPushButton(QStringLiteral("Preprocessor JSON…"), inner_model);
    pre_clear_btn_ = new QPushButton(QStringLiteral("Clear pre"), inner_model);
    fit_pre_btn_ = new QPushButton(QStringLiteral("Fit preprocessor…"), inner_model);
    fit_pre_btn_->setToolTip(QStringLiteral(
        "Fit a new preprocessor (scale + optional PCA) from the loaded CSV.\n"
        "RFF requires Python — use Python toolchain to add RFF weights."));
    pre_label_ = new QLabel(QStringLiteral("(optional, same as cypha_rest --pre)"), inner_model);
    pre_label_->setWordWrap(true);
    row_pre->addWidget(pre_btn_);
    row_pre->addWidget(pre_clear_btn_);
    row_pre->addWidget(fit_pre_btn_);
    row_pre->addWidget(pre_label_, 1);
    lay_model->addLayout(row_pre);
    lay_model->addStretch(1);

    // ── Dataset panel ────────────────────────────────────────────────────────
    auto* ds_grp = new QGroupBox(QStringLiteral("Dataset"), inner_data);
    auto* ds_vbox = new QVBoxLayout(ds_grp);

    // CSV file picker row
    auto* row_csv = new QHBoxLayout();
    csv_btn_ = new QPushButton(QStringLiteral("Training CSV…"), ds_grp);
    csv_label_ = new QLabel(QStringLiteral("(none)"), ds_grp);
    csv_label_->setWordWrap(true);
    row_csv->addWidget(csv_btn_);
    row_csv->addWidget(csv_label_, 1);
    ds_vbox->addLayout(row_csv);

    // Column picker: target combo + feature list side-by-side
    auto* col_splitter = new QSplitter(Qt::Horizontal, ds_grp);
    {
      auto* left_w = new QWidget(ds_grp);
      auto* left_vbox = new QVBoxLayout(left_w);
      left_vbox->setContentsMargins(0, 0, 0, 0);
      left_vbox->addWidget(new QLabel(QStringLiteral("Target column:"), left_w));
      col_target_combo_ = new QComboBox(left_w);
      col_target_combo_->setToolTip(QStringLiteral("Select which column is the prediction target.\nSyncs to the 'target name' field below."));
      left_vbox->addWidget(col_target_combo_);
      left_vbox->addWidget(new QLabel(QStringLiteral("Feature columns (check = include):"), left_w));
      col_feature_list_ = new QListWidget(left_w);
      col_feature_list_->setToolTip(QStringLiteral("Checked columns become features.\nSyncs to the 'feature names' field below."));
      col_feature_list_->setSelectionMode(QAbstractItemView::NoSelection);
      left_vbox->addWidget(col_feature_list_, 1);
      col_splitter->addWidget(left_w);
    }
    {
      auto* right_w = new QWidget(ds_grp);
      auto* right_vbox = new QVBoxLayout(right_w);
      right_vbox->setContentsMargins(0, 0, 0, 0);
      right_vbox->addWidget(new QLabel(QStringLiteral("Data preview (first 8 rows):"), right_w));
      csv_preview_table_ = new QTableWidget(right_w);
      csv_preview_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      csv_preview_table_->setAlternatingRowColors(true);
      csv_preview_table_->horizontalHeader()->setStretchLastSection(true);
      csv_preview_table_->setMinimumHeight(140);
      right_vbox->addWidget(csv_preview_table_, 1);
      col_splitter->addWidget(right_w);
    }
    col_splitter->setStretchFactor(0, 1);
    col_splitter->setStretchFactor(1, 3);
    ds_vbox->addWidget(col_splitter);

    // Stats + val split row
    auto* row_ds_stats = new QHBoxLayout();
    csv_stats_label_ = new QLabel(QStringLiteral("(no CSV loaded)"), ds_grp);
    csv_stats_label_->setWordWrap(true);
    row_ds_stats->addWidget(csv_stats_label_, 1);
    row_ds_stats->addWidget(new QLabel(QStringLiteral("Val split %:"), ds_grp));
    val_split_spin_ = new QSpinBox(ds_grp);
    val_split_spin_->setRange(0, 40);
    val_split_spin_->setValue(0);
    val_split_spin_->setToolTip(QStringLiteral(
        "Hold out last N% of rows as a validation set.\n"
        "After bulk native train, val accuracy is computed and shown."));
    val_split_spin_->setMaximumWidth(64);
    row_ds_stats->addWidget(val_split_spin_);
    ds_vbox->addLayout(row_ds_stats);

    // Manual target/feature overrides (collapsed-ish, visible for power users)
    auto* row_csv_tgt = new QHBoxLayout();
    row_csv_tgt->addWidget(new QLabel(QStringLiteral("target name"), ds_grp));
    csv_target_name_edit_ = new QLineEdit(ds_grp);
    csv_target_name_edit_->setPlaceholderText(QStringLiteral("header name (empty → use index)"));
    row_csv_tgt->addWidget(csv_target_name_edit_, 1);
    row_csv_tgt->addWidget(new QLabel(QStringLiteral("or idx"), ds_grp));
    csv_target_index_edit_ = new QLineEdit(ds_grp);
    csv_target_index_edit_->setText(QStringLiteral("-1"));
    csv_target_index_edit_->setMaximumWidth(56);
    row_csv_tgt->addWidget(csv_target_index_edit_);
    ds_vbox->addLayout(row_csv_tgt);

    auto* row_csv_feat = new QHBoxLayout();
    row_csv_feat->addWidget(new QLabel(QStringLiteral("feature names"), ds_grp));
    csv_feature_names_edit_ = new QLineEdit(ds_grp);
    csv_feature_names_edit_->setPlaceholderText(QStringLiteral("comma-separated; empty = all columns except target"));
    row_csv_feat->addWidget(csv_feature_names_edit_, 1);
    ds_vbox->addLayout(row_csv_feat);

    auto* row_csv_act = new QHBoxLayout();
    csv_inspect_btn_ = new QPushButton(QStringLiteral("Inspect CSV"), ds_grp);
    csv_fill_row0_btn_ = new QPushButton(QStringLiteral("Fill features from row 0"), ds_grp);
    row_csv_act->addWidget(csv_inspect_btn_);
    row_csv_act->addWidget(csv_fill_row0_btn_);
    row_csv_act->addStretch(1);
    ds_vbox->addLayout(row_csv_act);

    lay_data->addWidget(ds_grp);

    dataset_info_ = new QPlainTextEdit(inner_data);
    dataset_info_->setReadOnly(true);
    dataset_info_->setPlaceholderText(QStringLiteral("CSV inspect summary and registry scan notes."));
    dataset_info_->setMaximumBlockCount(80);
    dataset_info_->setMinimumHeight(120);
    lay_data->addWidget(dataset_info_);
    lay_data->addStretch(1);

    csv_regression_chk_ = new QCheckBox(
        QStringLiteral("CSV target is numeric regression (MKE: send regression_y on /update)"), inner_train);
    lay_train->addWidget(csv_regression_chk_);

    sort_by_uncertainty_chk_ = new QCheckBox(
        QStringLiteral("Sort by uncertainty (active learning — entropy from batch LLR before bulk native train)"),
        inner_train);
    sort_by_uncertainty_chk_->setToolTip(QStringLiteral(
        "Reorders training rows by descending softmax entropy (native batch_encode + score_matrix_use_field). "
        "Val split rows are unchanged."));
    lay_train->addWidget(sort_by_uncertainty_chk_);

    curriculum_chk_ = new QCheckBox(
        QStringLiteral("Curriculum (hardest-first — ascending softmax confidence before bulk native train)"),
        inner_train);
    curriculum_chk_->setToolTip(QStringLiteral(
        "Reorders training rows by ascending max softmax confidence (low confidence first). "
        "Mutually exclusive with uncertainty sort."));
    lay_train->addWidget(curriculum_chk_);

    auto* row_bulk = new QHBoxLayout();
    csv_bulk_train_btn_ = new QPushButton(QStringLiteral("Bulk REST /update"), inner_train);
    csv_bulk_native_btn_ = new QPushButton(QStringLiteral("Bulk native train"), inner_train);
    csv_bulk_native_btn_->setEnabled(false);
    csv_bulk_native_btn_->setToolTip(
        QStringLiteral("Classification CSV only — in-process dif_train_step (needs F_field in .cypha or JSON)"));
    bulk_native_cancel_btn_ = new QPushButton(QStringLiteral("Cancel train"), inner_train);
    bulk_native_cancel_btn_->setEnabled(false);
    bulk_native_cancel_btn_->setToolTip(QStringLiteral("Stop bulk native training (finishes current step first)"));
    row_bulk->addWidget(csv_bulk_train_btn_);
    row_bulk->addWidget(csv_bulk_native_btn_);
    row_bulk->addWidget(bulk_native_cancel_btn_);
    row_bulk->addWidget(new QLabel(QStringLiteral("max rows"), inner_train));
    csv_bulk_max_rows_spin_ = new QSpinBox(inner_train);
    csv_bulk_max_rows_spin_->setRange(0, 99000000);
    csv_bulk_max_rows_spin_->setValue(0);
    csv_bulk_max_rows_spin_->setToolTip(QStringLiteral("0 = all rows in the CSV"));
    csv_bulk_max_rows_spin_->setMaximumWidth(120);
    row_bulk->addWidget(csv_bulk_max_rows_spin_);
    row_bulk->addStretch(1);
    lay_train->addLayout(row_bulk);

    auto* row_ru01 = new QHBoxLayout();
    replay_u01_btn_ = new QPushButton(QStringLiteral("replay_u01 JSON…"), inner_train);
    replay_u01_label_ = new QLabel(QStringLiteral("(optional array for REST /update + native replay)"), inner_train);
    replay_u01_label_->setWordWrap(true);
    row_ru01->addWidget(replay_u01_btn_);
    row_ru01->addWidget(replay_u01_label_, 1);
    lay_train->addLayout(row_ru01);

    auto* row_mke = new QHBoxLayout();
    row_mke->addWidget(new QLabel(QStringLiteral("MKE correct_label"), inner_train));
    mke_correct_label_edit_ = new QLineEdit(inner_train);
    mke_correct_label_edit_->setPlaceholderText(QStringLiteral("empty → first class in model"));
    row_mke->addWidget(mke_correct_label_edit_, 1);
    row_mke->addWidget(new QLabel(QStringLiteral("router_train_label"), inner_train));
    mke_router_label_edit_ = new QLineEdit(inner_train);
    mke_router_label_edit_->setPlaceholderText(QStringLiteral("optional"));
    mke_router_label_edit_->setMaximumWidth(140);
    row_mke->addWidget(mke_router_label_edit_);
    lay_train->addLayout(row_mke);

    // ── Experiments panel (M6) ────────────────────────────────────────────────
#ifdef CYPHA_SHELL_EXPERIMENT_DB
    QWidget* inner_experiment = nullptr;
    QVBoxLayout* lay_experiment = nullptr;
    {
      auto* scr_exp = new QScrollArea(main_tabs_);
      scr_exp->setWidgetResizable(true);
      scr_exp->setFrameShape(QFrame::NoFrame);
      inner_experiment = new QWidget(scr_exp);
      lay_experiment = new QVBoxLayout(inner_experiment);
      lay_experiment->setSpacing(10);
      lay_experiment->setContentsMargins(6, 6, 6, 14);
      inner_experiment->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
      scr_exp->setWidget(inner_experiment);
      main_tabs_->addTab(scr_exp, QStringLiteral("7 · Experiments"));
    }
    {
      auto* exp_grp = new QGroupBox(QStringLiteral("Experiment DB (SQLite — mirrors cypha_studio ExperimentWidget)"),
                                    inner_experiment);
      auto* exp_form = new QFormLayout(exp_grp);

      auto* exp_db_row = new QHBoxLayout();
      exp_db_btn_ = new QPushButton(QStringLiteral("Open experiments.db…"), exp_grp);
      exp_db_label_ = new QLabel(QStringLiteral("(no DB — auto-detect ~/.cypha/experiments.db on click)"), exp_grp);
      exp_db_label_->setWordWrap(true);
      exp_db_row->addWidget(exp_db_btn_);
      exp_db_row->addWidget(exp_db_label_, 1);
      exp_form->addRow(exp_db_row);

      exp_name_edit_ = new QLineEdit(QStringLiteral("default"), exp_grp);
      exp_name_edit_->setToolTip(QStringLiteral("Experiment name — created automatically if it does not exist"));
      exp_run_name_edit_ = new QLineEdit(QStringLiteral("run"), exp_grp);
      exp_run_name_edit_->setToolTip(QStringLiteral("Run name"));
      exp_form->addRow(QStringLiteral("Experiment name:"), exp_name_edit_);
      exp_form->addRow(QStringLiteral("Run name:"),        exp_run_name_edit_);

      auto* exp_btn_row = new QHBoxLayout();
      exp_start_run_btn_ = new QPushButton(QStringLiteral("Start run"), exp_grp);
      exp_start_run_btn_->setEnabled(false);
      exp_start_run_btn_->setToolTip(QStringLiteral("Insert a pending run record in the DB"));
      exp_finish_run_btn_ = new QPushButton(QStringLiteral("Finish run (log metrics)"), exp_grp);
      exp_finish_run_btn_->setEnabled(false);
      exp_finish_run_btn_->setToolTip(
          QStringLiteral("Finish the current run: write accuracy / n_steps / checkpoint_path to DB"));
      exp_btn_row->addWidget(exp_start_run_btn_);
      exp_btn_row->addWidget(exp_finish_run_btn_);
      exp_btn_row->addStretch(1);
      exp_form->addRow(exp_btn_row);

      exp_status_label_ = new QLabel(QStringLiteral("(no active run)"), exp_grp);
      exp_status_label_->setWordWrap(true);
      exp_form->addRow(exp_status_label_);

      auto* exp_hdr_row = new QHBoxLayout();
      exp_hdr_row->addWidget(new QLabel(QStringLiteral("Experiments:"), exp_grp));
      exp_hdr_row->addStretch(1);
      exp_refresh_btn_ = new QPushButton(QStringLiteral("Refresh"), exp_grp);
      exp_hdr_row->addWidget(exp_refresh_btn_);
      exp_form->addRow(exp_hdr_row);

      exp_experiments_table_ = new QTableWidget(0, 4, exp_grp);
      exp_experiments_table_->setHorizontalHeaderLabels(
          {QStringLiteral("Experiment ID"), QStringLiteral("Name"), QStringLiteral("Task"),
           QStringLiteral("Created")});
      exp_experiments_table_->horizontalHeader()->setStretchLastSection(false);
      exp_experiments_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
      exp_experiments_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
      exp_experiments_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
      exp_experiments_table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
      exp_experiments_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      exp_experiments_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
      exp_experiments_table_->setSelectionMode(QAbstractItemView::SingleSelection);
      exp_experiments_table_->setMaximumHeight(120);
      exp_experiments_table_->setAlternatingRowColors(true);
      exp_form->addRow(exp_experiments_table_);

      // Recent runs table
      exp_runs_table_ = new QTableWidget(0, 6, exp_grp);
      exp_runs_table_->setHorizontalHeaderLabels(
          {QStringLiteral("Run ID"), QStringLiteral("Name"), QStringLiteral("Status"),
           QStringLiteral("Acc %"), QStringLiteral("Steps"), QStringLiteral("Checkpoint")});
      exp_runs_table_->horizontalHeader()->setStretchLastSection(false);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
      exp_runs_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      exp_runs_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
      exp_runs_table_->setSelectionMode(QAbstractItemView::ExtendedSelection);
      exp_runs_table_->setMaximumHeight(160);
      exp_runs_table_->setAlternatingRowColors(true);
      exp_form->addRow(exp_runs_table_);

      auto* exp_cmp_row = new QHBoxLayout();
      exp_compare_btn_ = new QPushButton(QStringLiteral("Compare selected runs"), exp_grp);
      exp_compare_btn_->setEnabled(false);
      exp_compare_btn_->setToolTip(
          QStringLiteral("Plot metrics_history loss curves for selected runs (one colour per run)"));
      exp_compare_clear_btn_ = new QPushButton(QStringLiteral("Clear compare chart"), exp_grp);
      exp_compare_clear_btn_->setEnabled(false);
      exp_cmp_row->addWidget(exp_compare_btn_);
      exp_cmp_row->addWidget(exp_compare_clear_btn_);
      exp_cmp_row->addStretch(1);
      exp_form->addRow(exp_cmp_row);

      exp_compare_summary_label_ = new QLabel(QStringLiteral("(select 2+ runs with loss history)"), exp_grp);
      exp_compare_summary_label_->setWordWrap(true);
      exp_form->addRow(exp_compare_summary_label_);

      exp_compare_stats_table_ = new QTableWidget(0, 9, exp_grp);
      exp_compare_stats_table_->setHorizontalHeaderLabels(
          {QStringLiteral("Run"), QStringLiteral("Acc %"), QStringLiteral("Macro F1"),
           QStringLiteral("R²"), QStringLiteral("RMSE"), QStringLiteral("Steps"),
           QStringLiteral("Dur (s)"), QStringLiteral("Final loss"), QStringLiteral("Min loss")});
      exp_compare_stats_table_->horizontalHeader()->setStretchLastSection(false);
      exp_compare_stats_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
      for (int col = 1; col < 9; ++col) {
        exp_compare_stats_table_->horizontalHeader()->setSectionResizeMode(col, QHeaderView::ResizeToContents);
      }
      exp_compare_stats_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      exp_compare_stats_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
      exp_compare_stats_table_->setSelectionMode(QAbstractItemView::NoSelection);
      exp_compare_stats_table_->setAlternatingRowColors(true);
      exp_compare_stats_table_->setMaximumHeight(120);
      exp_compare_stats_table_->setVisible(false);
      exp_form->addRow(exp_compare_stats_table_);

      exp_compare_chart_ = new SimpleLossChart(exp_grp);
      exp_compare_chart_->setMinimumHeight(160);
      exp_form->addRow(exp_compare_chart_);

      lay_experiment->addWidget(exp_grp);
    }
#endif  // CYPHA_SHELL_EXPERIMENT_DB

    // ── CyphaLM panel (native decode + optional REST /lm/*) ───────────────────
    const auto pg_lm = make_page(QStringLiteral("8 · CyphaLM"));
    QWidget* const inner_lm = pg_lm.first;
    QVBoxLayout* const lay_lm = pg_lm.second;
    {
      auto* lm_intro = new QLabel(
          QStringLiteral("<b>CyphaLM</b> &mdash; load a <tt>.json</tt> checkpoint (+ sibling <tt>.npz</tt>) "
                         "for in-process decode, or tick REST to use <tt>cypha_rest</tt> "
                         "<tt>/lm/load</tt> and <tt>/generate</tt>."),
          inner_lm);
      lm_intro->setWordWrap(true);
      lm_intro->setTextFormat(Qt::RichText);
      lay_lm->addWidget(lm_intro);
    }
    {
      auto* lm_ckpt_grp = new QGroupBox(QStringLiteral("Checkpoint"), inner_lm);
      auto* lm_ckpt_form = new QFormLayout(lm_ckpt_grp);
      auto* lm_path_row = new QHBoxLayout();
      lm_path_edit_ = new QLineEdit(lm_ckpt_grp);
      lm_path_edit_->setPlaceholderText(QStringLiteral("path/to/checkpoint.json"));
      lm_browse_btn_ = new QPushButton(QStringLiteral("Browse…"), lm_ckpt_grp);
      lm_path_row->addWidget(lm_path_edit_, 1);
      lm_path_row->addWidget(lm_browse_btn_);
      lm_ckpt_form->addRow(QStringLiteral("Path:"), lm_path_row);

      lm_use_rest_chk_ = new QCheckBox(
          QStringLiteral("Use REST (base URL from Server tab — start cypha_rest first)"), lm_ckpt_grp);
      lm_ckpt_form->addRow(lm_use_rest_chk_);

      auto* lm_load_row = new QHBoxLayout();
      lm_load_btn_ = new QPushButton(QStringLiteral("Load checkpoint"), lm_ckpt_grp);
      lm_status_label_ = new QLabel(QStringLiteral("(not loaded)"), lm_ckpt_grp);
      lm_status_label_->setWordWrap(true);
      lm_load_row->addWidget(lm_load_btn_);
      lm_load_row->addWidget(lm_status_label_, 1);
      lm_ckpt_form->addRow(lm_load_row);
      lay_lm->addWidget(lm_ckpt_grp);
    }
    {
      auto* lm_gen_grp = new QGroupBox(QStringLiteral("Generate"), inner_lm);
      auto* lm_gen_form = new QFormLayout(lm_gen_grp);
      lm_prompt_edit_ = new QLineEdit(lm_gen_grp);
      lm_prompt_edit_->setPlaceholderText(QStringLiteral("comma-separated token ids, e.g. 1,2,3"));
      lm_gen_form->addRow(QStringLiteral("prompt_ids:"), lm_prompt_edit_);

      lm_n_tokens_spin_ = new QSpinBox(lm_gen_grp);
      lm_n_tokens_spin_->setRange(1, 4096);
      lm_n_tokens_spin_->setValue(64);
      lm_gen_form->addRow(QStringLiteral("n_tokens (max_tokens):"), lm_n_tokens_spin_);

      lm_strategy_combo_ = new QComboBox(lm_gen_grp);
      lm_strategy_combo_->addItems({QStringLiteral("top_p"), QStringLiteral("temperature"),
                                    QStringLiteral("greedy"), QStringLiteral("top_k")});
      lm_strategy_combo_->setCurrentText(QStringLiteral("top_p"));
      lm_gen_form->addRow(QStringLiteral("strategy:"), lm_strategy_combo_);

      lm_top_p_spin_ = new QDoubleSpinBox(lm_gen_grp);
      lm_top_p_spin_->setRange(0.01, 1.0);
      lm_top_p_spin_->setSingleStep(0.05);
      lm_top_p_spin_->setValue(0.92);
      lm_gen_form->addRow(QStringLiteral("top_p:"), lm_top_p_spin_);

      lm_temperature_spin_ = new QDoubleSpinBox(lm_gen_grp);
      lm_temperature_spin_->setRange(0.01, 5.0);
      lm_temperature_spin_->setSingleStep(0.05);
      lm_temperature_spin_->setValue(0.9);
      lm_gen_form->addRow(QStringLiteral("temperature:"), lm_temperature_spin_);

      lm_epistemic_halt_chk_ = new QCheckBox(
          QStringLiteral("Epistemic halt (Paper IV — halt when r_eu exceeds threshold)"), lm_gen_grp);
      lm_epistemic_halt_chk_->setChecked(false);
      lm_epistemic_halt_chk_->setToolTip(QStringLiteral(
          "Matches REST /generate \"epistemic_halt\": true — stops decode when epistemic ratio "
          "r_eu is high (AdaptiveEpistemicThreshold when native)."));
      lm_gen_form->addRow(lm_epistemic_halt_chk_);

      lm_generate_btn_ = new QPushButton(QStringLiteral("Generate"), lm_gen_grp);
      lm_generate_btn_->setEnabled(false);
      lm_gen_form->addRow(lm_generate_btn_);

      lm_output_edit_ = new QPlainTextEdit(lm_gen_grp);
      lm_output_edit_->setReadOnly(true);
      lm_output_edit_->setPlaceholderText(QStringLiteral("generated_ids and decoded text appear here"));
      lm_output_edit_->setMinimumHeight(100);
      lm_gen_form->addRow(lm_output_edit_);
      lay_lm->addWidget(lm_gen_grp);
    }
    {
      auto* bench_grp = new QGroupBox(QStringLiteral("Bench (optional)"), inner_lm);
      auto* bench_form = new QFormLayout(bench_grp);
      auto* bench_bin_row = new QHBoxLayout();
      bench_bin_edit_ = new QLineEdit(bench_grp);
      bench_bin_edit_->setPlaceholderText(QStringLiteral("path/to/cypha_bench_run"));
#if defined(_WIN32)
      {
        const QString guess =
            QApplication::applicationDirPath() + QStringLiteral("/cypha_bench_run.exe");
        if (QFileInfo::exists(guess)) {
          bench_bin_edit_->setText(guess);
        }
      }
#endif
      bench_browse_btn_ = new QPushButton(QStringLiteral("Browse…"), bench_grp);
      bench_bin_row->addWidget(bench_bin_edit_, 1);
      bench_bin_row->addWidget(bench_browse_btn_);
      bench_form->addRow(QStringLiteral("Binary:"), bench_bin_row);

      bench_run_d17_btn_ = new QPushButton(QStringLiteral("Run cypha_bench_run --domain 17"), bench_grp);
      bench_form->addRow(bench_run_d17_btn_);

      bench_log_edit_ = new QPlainTextEdit(bench_grp);
      bench_log_edit_->setReadOnly(true);
      bench_log_edit_->setMaximumBlockCount(800);
      bench_log_edit_->setPlaceholderText(QStringLiteral("bench stdout/stderr"));
      bench_log_edit_->setMinimumHeight(120);
      bench_form->addRow(bench_log_edit_);
      lay_lm->addWidget(bench_grp);
    }
    lay_lm->addStretch(1);

    // ── Registry tab (ModelRegistry parity) ───────────────────────────────────
    {
      auto* reg_intro = new QLabel(
          QStringLiteral("<b>Model registry</b> &mdash; on-disk layout matches Python <tt>ModelRegistry</tt> "
                         "(<tt>&lt;root&gt;/&lt;name&gt;/&lt;version&gt;/model.cypha</tt>). "
                         "Scan locally or <tt>GET /models</tt> from a running <tt>cypha_rest</tt>."),
          inner_registry);
      reg_intro->setWordWrap(true);
      reg_intro->setTextFormat(Qt::RichText);
      lay_registry->addWidget(reg_intro);
    }
    auto* row_reg = new QHBoxLayout();
    reg_root_btn_ = new QPushButton(QStringLiteral("Registry root…"), inner_registry);
    reg_root_label_ = new QLabel(QStringLiteral("(none — passed to cypha_rest --registry when set)"), inner_registry);
    reg_root_label_->setWordWrap(true);
    row_reg->addWidget(reg_root_btn_);
    row_reg->addWidget(reg_root_label_, 1);
    lay_registry->addLayout(row_reg);

    auto* row_reg_act = new QHBoxLayout();
    reg_scan_btn_ = new QPushButton(QStringLiteral("Scan (native)"), inner_registry);
    reg_rest_models_btn_ = new QPushButton(QStringLiteral("GET /models (REST)"), inner_registry);
    reg_load_btn_ = new QPushButton(QStringLiteral("Load bundle (local .cypha)"), inner_registry);
    reg_rest_load_btn_ = new QPushButton(QStringLiteral("POST /load (REST)"), inner_registry);
    reg_register_btn_ = new QPushButton(QStringLiteral("Register current…"), inner_registry);
    row_reg_act->addWidget(reg_scan_btn_);
    row_reg_act->addWidget(reg_rest_models_btn_);
    row_reg_act->addWidget(reg_load_btn_);
    row_reg_act->addWidget(reg_rest_load_btn_);
    row_reg_act->addWidget(reg_register_btn_);
    row_reg_act->addStretch(1);
    lay_registry->addLayout(row_reg_act);

    reg_models_table_ = new QTableWidget(0, 4, inner_registry);
    reg_models_table_->setHorizontalHeaderLabels(
        {QStringLiteral("Name"), QStringLiteral("Version"), QStringLiteral("Model path"),
         QStringLiteral("Preprocessor")});
    reg_models_table_->horizontalHeader()->setStretchLastSection(true);
    reg_models_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    reg_models_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    reg_models_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    reg_models_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    reg_models_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    reg_models_table_->setSelectionMode(QAbstractItemView::SingleSelection);
    reg_models_table_->setAlternatingRowColors(true);
    reg_models_table_->setMinimumHeight(160);
    lay_registry->addWidget(reg_models_table_);

    reg_combo_ = new QComboBox(inner_registry);
    reg_combo_->setMinimumContentsLength(28);
    reg_combo_->setVisible(false);
    lay_registry->addWidget(reg_combo_);

    auto* row_card = new QHBoxLayout();
    card_btn_ = new QPushButton(QStringLiteral("card.json…"), inner_registry);
    card_label_ = new QLabel(QStringLiteral("(optional path — Register opens card editor; default beside .cypha)"),
                             inner_registry);
    card_label_->setWordWrap(true);
    row_card->addWidget(card_btn_);
    row_card->addWidget(card_label_, 1);
    lay_registry->addLayout(row_card);
    lay_registry->addStretch(1);

    // ── Help / About tab ────────────────────────────────────────────────────────
    {
      auto* scr_help = new QScrollArea(main_tabs_);
      scr_help->setWidgetResizable(true);
      scr_help->setFrameShape(QFrame::NoFrame);
      auto* inner_help = new QWidget(scr_help);
      auto* lay_help = new QVBoxLayout(inner_help);
      lay_help->setContentsMargins(6, 6, 6, 14);
      help_browser_ = new QTextBrowser(inner_help);
      help_browser_->setOpenExternalLinks(true);
      help_browser_->setStyleSheet(QStringLiteral("QTextBrowser { background-color: #1e1e1e; color: #e6e6e6; }"));
      help_browser_->document()->setDefaultStyleSheet(
          QStringLiteral("body { color: #e0e0e0; } h1 { color: #7eb8ff; } h2 { color: #9cdcfe; } "
                         "code, pre { color: #e0c896; background: #2d2d2d; } a { color: #6cb6ff; }"));
      help_browser_->setHtml(native_quickstart_html());
      lay_help->addWidget(help_browser_);
      scr_help->setWidget(inner_help);
      main_tabs_->addTab(scr_help, QStringLiteral("9 · Help"));
    }

    {
      auto* rest_intro = new QLabel(
          QStringLiteral("<b>REST server</b> &mdash; spawn <tt>cypha_rest</tt> or set base URL below."),
          inner_server);
      rest_intro->setWordWrap(true);
      rest_intro->setTextFormat(Qt::RichText);
      lay_server->addWidget(rest_intro);
    }

    features_hint_ = new QLabel(QStringLiteral("Features: comma-separated values"), inner_predict);
    features_hint_->setWordWrap(true);
    lay_predict->addWidget(features_hint_);

    features_edit_ = new QLineEdit(inner_predict);
    features_edit_->setPlaceholderText(QStringLiteral("0,0,0,…"));
    lay_predict->addWidget(features_edit_);

    predict_btn_ = new QPushButton(QStringLiteral("Predict (native)"), inner_predict);
    predict_btn_->setEnabled(false);
    lay_predict->addWidget(predict_btn_);

    predict_return_explanation_chk_ =
        new QCheckBox(QStringLiteral("return_explanation (class_details, world_mu_distance, all_scores)"),
                      inner_predict);
    predict_return_explanation_chk_->setChecked(true);
    lay_predict->addWidget(predict_return_explanation_chk_);

    predict_self_correct_chk_ =
        new QCheckBox(QStringLiteral("Self-correct (REST /predict)"), inner_predict);
    predict_self_correct_chk_->setChecked(false);
    predict_self_correct_chk_->setToolTip(
        QStringLiteral("POST /predict with \"self_correct\": true (Paper IV epistemic loop)."));
    lay_predict->addWidget(predict_self_correct_chk_);

    predict_rest_btn_ = new QPushButton(QStringLiteral("Predict (REST POST /predict)"), inner_predict);
    predict_rest_btn_->setEnabled(false);
    lay_predict->addWidget(predict_rest_btn_);

    {
      auto* conf_grp = new QGroupBox(QStringLiteral("Confidence & explanation (confidence_widget parity)"),
                                     inner_predict);
      auto* conf_vbox = new QVBoxLayout(conf_grp);

      conf_pred_label_ = new QLabel(QStringLiteral("Prediction: —"), conf_grp);
      conf_pred_label_->setStyleSheet(QStringLiteral("font-weight: bold;"));
      conf_vbox->addWidget(conf_pred_label_);

      auto* ood_row = new QHBoxLayout();
      conf_ood_label_ = new QLabel(QStringLiteral("OOD score: —"), conf_grp);
      conf_ood_bar_ = new QProgressBar(conf_grp);
      conf_ood_bar_->setRange(0, 100);
      conf_ood_bar_->setTextVisible(false);
      conf_ood_bar_->setFixedHeight(14);
      ood_row->addWidget(conf_ood_label_);
      ood_row->addWidget(conf_ood_bar_, 1);
      conf_vbox->addLayout(ood_row);

      conf_scores_table_ = new QTableWidget(0, 2, conf_grp);
      conf_scores_table_->setHorizontalHeaderLabels(
          {QStringLiteral("Class"), QStringLiteral("Score / confidence")});
      conf_scores_table_->horizontalHeader()->setStretchLastSection(true);
      conf_scores_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      conf_scores_table_->setMaximumHeight(140);
      conf_scores_table_->setAlternatingRowColors(true);
      conf_vbox->addWidget(conf_scores_table_);

      conf_explanation_edit_ = new QPlainTextEdit(conf_grp);
      conf_explanation_edit_->setReadOnly(true);
      conf_explanation_edit_->setPlaceholderText(
          QStringLiteral("explanation JSON (return_explanation) — class_details, world_mu_distance, r_eff"));
      conf_explanation_edit_->setMinimumHeight(100);
      conf_vbox->addWidget(conf_explanation_edit_);

      lay_predict->addWidget(conf_grp);
    }

    auto* hp_group = new QGroupBox(QStringLiteral("Native train hyperparameters"), inner_train);
    auto* hp_form = new QFormLayout(hp_group);
    hp_world_lr_edit_ = new QLineEdit(QStringLiteral("0.008"), hp_group);
    hp_delta_lr_edit_ = new QLineEdit(QStringLiteral("0.05"), hp_group);
    hp_ood_sigma_edit_ = new QLineEdit(QStringLiteral("15"), hp_group);
    hp_enc_lr_edit_ = new QLineEdit(QStringLiteral("0.002"), hp_group);
    hp_replay_ratio_edit_ = new QLineEdit(QStringLiteral("0.30"), hp_group);
    hp_replay_cap_spin_ = new QSpinBox(hp_group);
    hp_replay_cap_spin_->setRange(8, 10000000);
    hp_replay_cap_spin_->setValue(10000);
    hp_align_every_spin_ = new QSpinBox(hp_group);
    hp_align_every_spin_->setRange(0, 9999999);
    hp_align_every_spin_->setValue(500);
    hp_temp_recalib_spin_ = new QSpinBox(hp_group);
    hp_temp_recalib_spin_->setRange(0, 9999999);
    hp_temp_recalib_spin_->setValue(0);
    hp_form->addRow(QStringLiteral("world_lr"), hp_world_lr_edit_);
    hp_form->addRow(QStringLiteral("delta_lr"), hp_delta_lr_edit_);
    hp_form->addRow(QStringLiteral("ood_sigma"), hp_ood_sigma_edit_);
    hp_form->addRow(QStringLiteral("enc_lr"), hp_enc_lr_edit_);
    hp_form->addRow(QStringLiteral("replay_ratio"), hp_replay_ratio_edit_);
    hp_form->addRow(QStringLiteral("replay_cap"), hp_replay_cap_spin_);
    hp_form->addRow(QStringLiteral("align_every"), hp_align_every_spin_);
    hp_form->addRow(QStringLiteral("temp_recalib_every"), hp_temp_recalib_spin_);
    use_kernel_llr_chk_ = new QCheckBox(QStringLiteral("Nyström kernel LLR (nonlinear boundaries)"), hp_group);
    use_kernel_llr_chk_->setToolTip(
        QStringLiteral("Train/infer with KernelMemory; persisted via patch_kernel_into_root on save"));
    kernel_blend_spin_ = new QDoubleSpinBox(hp_group);
    kernel_blend_spin_->setRange(0.0, 1.0);
    kernel_blend_spin_->setSingleStep(0.05);
    kernel_blend_spin_->setDecimals(2);
    kernel_blend_spin_->setValue(0.5);
    hp_form->addRow(use_kernel_llr_chk_);
    hp_form->addRow(QStringLiteral("kernel_blend"), kernel_blend_spin_);
    auto* hp_row = new QHBoxLayout();
    hp_apply_btn_ = new QPushButton(QStringLiteral("Apply to native train"), hp_group);
    hp_defaults_btn_ = new QPushButton(QStringLiteral("Reset defaults"), hp_group);
    hp_row->addWidget(hp_apply_btn_);
    hp_row->addWidget(hp_defaults_btn_);
    hp_row->addStretch(1);
    hp_form->addRow(hp_row);
    lay_train->addWidget(hp_group);

    native_train_one_btn_ = new QPushButton(QStringLiteral("Train 1 row (native, from feature box + correct_label)"),
                                            inner_train);
    native_train_one_btn_->setEnabled(false);
    native_train_one_btn_->setToolTip(
        QStringLiteral("dif_train_step_vector / GH — mutates loaded model in memory (not saved to disk)"));
    lay_train->addWidget(native_train_one_btn_);

    save_native_btn_ = new QPushButton(QStringLiteral("Save trained model (.cypha)…"), inner_train);
    save_native_btn_->setEnabled(false);
    save_native_btn_->setToolTip(
        QStringLiteral("merge_state_into_root_for_save + infer snapshot + optional kernel_mem "
                       "(patch_kernel_into_root) — see native/qt/README.md"));
    lay_train->addWidget(save_native_btn_);

    // ── MKE Regressor panel ───────────────────────────────────────────────────
    {
      auto* mke_grp = new QGroupBox(QStringLiteral("MKE Regressor (native, online — regression mode)"), inner_train);
      auto* mke_form = new QFormLayout(mke_grp);
      mke_ff_spin_ = new QDoubleSpinBox(mke_grp);
      mke_ff_spin_->setRange(0.9, 1.0);
      mke_ff_spin_->setValue(0.99);
      mke_ff_spin_->setSingleStep(0.005);
      mke_ff_spin_->setDecimals(4);
      mke_ff_spin_->setToolTip(QStringLiteral("RLS forgetting factor (0.99 = slow forget, 1.0 = no forgetting)"));
      mke_pi_spin_ = new QDoubleSpinBox(mke_grp);
      mke_pi_spin_->setRange(0.0, 0.1);
      mke_pi_spin_->setValue(1e-4);
      mke_pi_spin_->setSingleStep(1e-5);
      mke_pi_spin_->setDecimals(6);
      mke_pi_spin_->setToolTip(QStringLiteral("Expert weight floor π₀ (prevents zero routing collapse)"));
      mke_form->addRow(QStringLiteral("Forgetting factor:"), mke_ff_spin_);
      mke_form->addRow(QStringLiteral("π floor:"),           mke_pi_spin_);

      auto* mke_btn_row = new QHBoxLayout();
      mke_init_btn_ = new QPushButton(QStringLiteral("Init / Reset MKE state"), mke_grp);
      mke_init_btn_->setToolTip(QStringLiteral(
          "Clear per-label expert weights (w) and RLS covariance (P); "
          "first training step will re-initialize lazily."));
      mke_bulk_btn_ = new QPushButton(QStringLiteral("Bulk MKE train CSV"), mke_grp);
      mke_bulk_btn_->setEnabled(false);
      mke_bulk_btn_->setToolTip(QStringLiteral(
          "Load the selected training CSV (regression mode required), run "
          "mke_scalar_train_step_from_phi on each row, show final MSE/RMSE."));
      mke_predict_btn_ = new QPushButton(QStringLiteral("MKE predict (from feature box)"), mke_grp);
      mke_predict_btn_->setEnabled(false);
      mke_btn_row->addWidget(mke_init_btn_);
      mke_btn_row->addWidget(mke_bulk_btn_);
      mke_btn_row->addWidget(mke_predict_btn_);
      mke_form->addRow(mke_btn_row);

      mke_result_label_ = new QLabel(QStringLiteral("(no MKE results yet)"), mke_grp);
      mke_result_label_->setWordWrap(true);
      mke_form->addRow(mke_result_label_);

      lay_train->addWidget(mke_grp);
    }

    {
      auto* row_loss = new QHBoxLayout();
      loss_chart_ = new LossMetricsPanel(inner_train);
      row_loss->addWidget(loss_chart_, 1);
      auto* loss_btn_col = new QVBoxLayout();
      loss_chart_save_btn_ = new QPushButton(QStringLiteral("Save chart PNG…"), inner_train);
      loss_chart_save_btn_->setToolTip(
          QStringLiteral("Raster export — REST (blue) and native (orange) when both bulks were run."));
      loss_btn_col->addWidget(loss_chart_save_btn_);
      loss_csv_save_btn_ = new QPushButton(QStringLiteral("Save loss CSV…"), inner_train);
      loss_csv_save_btn_->setToolTip(QStringLiteral(
          "step, loss_rest, loss_rest_ema, loss_native, loss_native_ema (α=0.08 EMA; blank if no bulk)."));
      loss_btn_col->addWidget(loss_csv_save_btn_);
      loss_svg_save_btn_ = new QPushButton(QStringLiteral("Save chart SVG…"), inner_train);
      loss_svg_save_btn_->setToolTip(
          QStringLiteral("Vector export (embedded polyline — no Qt Svg module required)."));
      loss_btn_col->addWidget(loss_svg_save_btn_);
      loss_chart_clear_btn_ = new QPushButton(QStringLiteral("Clear chart"), inner_train);
      loss_btn_col->addWidget(loss_chart_clear_btn_);
      loss_ema_chk_ = new QCheckBox(QStringLiteral("EMA overlay (α=0.08)"), inner_train);
      loss_ema_chk_->setChecked(true);
      loss_btn_col->addWidget(loss_ema_chk_);
      loss_y_lock_chk_ = new QCheckBox(QStringLiteral("Y lock"), inner_train);
      loss_y_lock_chk_->setToolTip(QStringLiteral("Pin Y axis to manual min/max instead of auto-ranging."));
      loss_btn_col->addWidget(loss_y_lock_chk_);
      auto* yrw = new QHBoxLayout();
      loss_y_min_spin_ = new QDoubleSpinBox(inner_train);
      loss_y_min_spin_->setRange(-1e9, 1e9);
      loss_y_min_spin_->setValue(-10.0);
      loss_y_min_spin_->setDecimals(3);
      loss_y_min_spin_->setToolTip(QStringLiteral("Y axis minimum (active when Y lock is on)"));
      loss_y_min_spin_->setEnabled(false);
      loss_y_max_spin_ = new QDoubleSpinBox(inner_train);
      loss_y_max_spin_->setRange(-1e9, 1e9);
      loss_y_max_spin_->setValue(0.0);
      loss_y_max_spin_->setDecimals(3);
      loss_y_max_spin_->setToolTip(QStringLiteral("Y axis maximum (active when Y lock is on)"));
      loss_y_max_spin_->setEnabled(false);
      yrw->addWidget(new QLabel(QStringLiteral("min"), inner_train));
      yrw->addWidget(loss_y_min_spin_);
      yrw->addWidget(new QLabel(QStringLiteral("max"), inner_train));
      yrw->addWidget(loss_y_max_spin_);
      loss_btn_col->addLayout(yrw);
      loss_btn_col->addStretch(1);
      row_loss->addLayout(loss_btn_col);
      lay_train->addLayout(row_loss);

      connect(loss_chart_save_btn_, &QPushButton::clicked, this, [this]() {
        const QPixmap pm = loss_chart_->capture_widget()->grab();
        if (pm.isNull()) {
          QMessageBox::warning(this, QStringLiteral("Export"), QStringLiteral("Could not capture the chart."));
          return;
        }
        const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save loss chart"),
                                                          QStringLiteral("cypha_loss.png"),
                                                          QStringLiteral("PNG image (*.png);;All (*)"));
        if (path.isEmpty()) {
          return;
        }
        if (!pm.save(path, "PNG")) {
          QMessageBox::warning(this, QStringLiteral("Export"),
                               QStringLiteral("Failed to write PNG:\n%1").arg(path));
        }
      });

      connect(loss_csv_save_btn_, &QPushButton::clicked, this, [this]() {
        if (last_loss_plot_rest_.isEmpty() && last_loss_plot_native_.isEmpty()) {
          QMessageBox::information(this, QStringLiteral("Export"),
                                   QStringLiteral("Run bulk REST /update or bulk native train first."));
          return;
        }
        const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save loss series"),
                                                          QStringLiteral("cypha_loss.csv"),
                                                          QStringLiteral("CSV (*.csv);;All (*)"));
        if (path.isEmpty()) {
          return;
        }
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
          QMessageBox::warning(this, QStringLiteral("Export"), QStringLiteral("Cannot open file for write."));
          return;
        }
        QTextStream out(&f);
        out << QStringLiteral("step,loss_rest,loss_rest_ema,loss_native,loss_native_ema\n");
        constexpr double kEmaAlpha = 0.08;
        const QVector<double> er = loss_ema_series(last_loss_plot_rest_, kEmaAlpha);
        const QVector<double> en = loss_ema_series(last_loss_plot_native_, kEmaAlpha);
        const int nr = last_loss_plot_rest_.size();
        const int nn = last_loss_plot_native_.size();
        const int nm = std::max(nr, nn);
        for (int i = 0; i < nm; ++i) {
          out << i << QLatin1Char(',');
          if (i < nr) {
            out << QString::number(last_loss_plot_rest_[i], 'g', 17);
          }
          out << QLatin1Char(',');
          if (i < er.size()) {
            out << QString::number(er[i], 'g', 17);
          }
          out << QLatin1Char(',');
          if (i < nn) {
            out << QString::number(last_loss_plot_native_[i], 'g', 17);
          }
          out << QLatin1Char(',');
          if (i < en.size()) {
            out << QString::number(en[i], 'g', 17);
          }
          out << QLatin1Char('\n');
        }
      });

      connect(loss_svg_save_btn_, &QPushButton::clicked, this, [this]() {
        if (last_loss_plot_rest_.isEmpty() && last_loss_plot_native_.isEmpty()) {
          QMessageBox::information(this, QStringLiteral("Export"),
                                   QStringLiteral("Run bulk REST /update or bulk native train first."));
          return;
        }
        const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save loss chart SVG"),
                                                          QStringLiteral("cypha_loss.svg"),
                                                          QStringLiteral("SVG (*.svg);;All (*)"));
        if (path.isEmpty()) {
          return;
        }
        constexpr double alpha = 0.08;
        QVector<double> er;
        QVector<double> en;
        if (loss_ema_chk_->isChecked()) {
          er = loss_ema_series(last_loss_plot_rest_, alpha);
          en = loss_ema_series(last_loss_plot_native_, alpha);
        }
        if (!write_loss_chart_svg(path, last_loss_plot_rest_, last_loss_plot_native_, er, en)) {
          QMessageBox::warning(this, QStringLiteral("Export"),
                               QStringLiteral("Could not write SVG:\n%1").arg(path));
        }
      });

      connect(loss_chart_clear_btn_, &QPushButton::clicked, this, [this]() {
        last_loss_plot_rest_.clear();
        last_loss_plot_native_.clear();
        if (loss_chart_ != nullptr) {
          loss_chart_->clear_losses();
        }
        refresh_loss_chart();
      });

      connect(loss_ema_chk_, &QCheckBox::stateChanged, this, [this](int) { refresh_loss_chart(); });
      connect(loss_y_lock_chk_, &QCheckBox::stateChanged, this, [this](int state) {
        const bool locked = (state == Qt::Checked);
        loss_y_min_spin_->setEnabled(locked);
        loss_y_max_spin_->setEnabled(locked);
        refresh_loss_chart();
      });
      connect(loss_y_min_spin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
              [this](double) { refresh_loss_chart(); });
      connect(loss_y_max_spin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
              [this](double) { refresh_loss_chart(); });
    }

    // ── Batch predict panel ───────────────────────────────────────────────────
    {
      auto* bp_grp = new QGroupBox(QStringLiteral("Batch predict (native)"), inner_predict);
      auto* bp_vbox = new QVBoxLayout(bp_grp);

      auto* bp_row1 = new QHBoxLayout();
      batch_predict_csv_btn_ = new QPushButton(QStringLiteral("Predict CSV…"), bp_grp);
      batch_predict_csv_btn_->setEnabled(false);
      batch_predict_csv_btn_->setToolTip(QStringLiteral(
          "Load a CSV, run best_label_and_conf on each row (native, no server), "
          "display label + confidence per row."));
      bp_row1->addWidget(batch_predict_csv_btn_);
      batch_predict_export_btn_ = new QPushButton(QStringLiteral("Export results CSV…"), bp_grp);
      batch_predict_export_btn_->setEnabled(false);
      bp_row1->addWidget(batch_predict_export_btn_);
      bp_row1->addStretch(1);
      bp_vbox->addLayout(bp_row1);

      batch_predict_table_ = new QTableWidget(0, 3, bp_grp);
      batch_predict_table_->setHorizontalHeaderLabels(
          {QStringLiteral("Row"), QStringLiteral("Predicted label"), QStringLiteral("Confidence")});
      batch_predict_table_->horizontalHeader()->setStretchLastSection(true);
      batch_predict_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      batch_predict_table_->setMaximumHeight(200);
      bp_vbox->addWidget(batch_predict_table_);

      lay_predict->addWidget(bp_grp);
    }

    {
      auto* tlog_hdr = new QHBoxLayout();
      tlog_hdr->addWidget(new QLabel(QStringLiteral("Native train log:"), inner_train));
      tlog_hdr->addStretch(1);
      train_log_clear_btn_ = new QPushButton(QStringLiteral("Clear log"), inner_train);
      train_log_export_btn_ = new QPushButton(QStringLiteral("Export CSV…"), inner_train);
      tlog_hdr->addWidget(train_log_clear_btn_);
      tlog_hdr->addWidget(train_log_export_btn_);
      lay_train->addLayout(tlog_hdr);
      train_log_table_ = new QTableWidget(0, 4, inner_train);
      train_log_table_->setHorizontalHeaderLabels({QStringLiteral("#"),
                                                   QStringLiteral("Label"),
                                                   QStringLiteral("Loss"),
                                                   QStringLiteral("Correct")});
      train_log_table_->horizontalHeader()->setStretchLastSection(true);
      train_log_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
      train_log_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
      train_log_table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
      train_log_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      train_log_table_->setSelectionMode(QAbstractItemView::SingleSelection);
      train_log_table_->setMaximumHeight(150);
      train_log_table_->setAlternatingRowColors(true);
      train_log_table_->setToolTip(
          QStringLiteral("Per-step native train history: #=cumulative step, Correct=✓/✗. "
                         "Capped at 2000 rows; older rows removed."));
      lay_train->addWidget(train_log_table_);
    }

    // ── Training progress panel ───────────────────────────────────────────────
    {
      auto* prog_grp = new QGroupBox(QStringLiteral("Training progress (native)"), inner_train);
      auto* prog_vbox = new QVBoxLayout(prog_grp);

      auto* prog_row1 = new QHBoxLayout();
      train_prog_label_ = new QLabel(QStringLiteral("Steps: 0  |  Acc(win): —  |  EMA loss: —  |  Classes: 0"),
                                     prog_grp);
      train_prog_reset_btn_ = new QPushButton(QStringLiteral("Reset stats"), prog_grp);
      prog_row1->addWidget(train_prog_label_, 1);
      prog_row1->addWidget(train_prog_reset_btn_);
      prog_vbox->addLayout(prog_row1);

      train_prog_class_table_ = new QTableWidget(0, 4, prog_grp);
      train_prog_class_table_->setHorizontalHeaderLabels(
          {QStringLiteral("Class"), QStringLiteral("N (obs)"),
           QStringLiteral("N correct"), QStringLiteral("Acc %")});
      train_prog_class_table_->horizontalHeader()->setStretchLastSection(false);
      train_prog_class_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
      train_prog_class_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
      train_prog_class_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
      train_prog_class_table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
      train_prog_class_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      train_prog_class_table_->setMaximumHeight(160);
      train_prog_class_table_->setAlternatingRowColors(true);
      prog_vbox->addWidget(train_prog_class_table_);

      train_prog_acc_bar_ = new PerClassAccuracyBar(prog_grp);
      prog_vbox->addWidget(train_prog_acc_bar_);

      lay_train->addWidget(prog_grp);
    }

    lay_train->addStretch(1);
    lay_predict->addStretch(1);
    lay_server->addStretch(1);

    lay_server->addWidget(new QLabel(QStringLiteral("Spawn cypha_rest (optional):"), inner_server));
    auto* row_bin = new QHBoxLayout();
    rest_browse_btn_ = new QPushButton(QStringLiteral("cypha_rest binary…"), inner_server);
    rest_bin_edit_ = new QLineEdit(inner_server);
    rest_bin_edit_->setPlaceholderText(QStringLiteral("path/to/cypha_rest"));
    {
      // Auto-discover: cypha_rest is typically installed/built alongside cypha_qt_shell
      // (see native/qt/CMakeLists.txt install() + windeployqt packaging). Manual browse
      // below remains available as a fallback when that's not the case.
#if defined(_WIN32)
      const QString exe_name = QStringLiteral("cypha_rest.exe");
#else
      const QString exe_name = QStringLiteral("cypha_rest");
#endif
      const QString guess = QApplication::applicationDirPath() + QLatin1Char('/') + exe_name;
      if (QFileInfo::exists(guess)) {
        rest_bin_edit_->setText(guess);
      }
    }
    row_bin->addWidget(rest_browse_btn_);
    row_bin->addWidget(rest_bin_edit_, 1);
    lay_server->addLayout(row_bin);

    auto* row_listen = new QHBoxLayout();
    row_listen->addWidget(new QLabel(QStringLiteral("--listen"), inner_server));
    rest_listen_edit_ = new QLineEdit(inner_server);
    rest_listen_edit_->setText(QStringLiteral("127.0.0.1:8099"));
    row_listen->addWidget(rest_listen_edit_, 1);
    lay_server->addLayout(row_listen);

    auto* row_srv = new QHBoxLayout();
    rest_start_btn_ = new QPushButton(QStringLiteral("Start server"), inner_server);
    rest_stop_btn_ = new QPushButton(QStringLiteral("Stop server"), inner_server);
    rest_stop_btn_->setEnabled(false);
    rest_status_label_ = new QLabel(QStringLiteral("Server: stopped"), inner_server);
    row_srv->addWidget(rest_start_btn_);
    row_srv->addWidget(rest_stop_btn_);
    row_srv->addWidget(rest_status_label_, 1);
    lay_server->addLayout(row_srv);

    auto* row_log_hdr = new QHBoxLayout();
    row_log_hdr->addWidget(new QLabel(QStringLiteral("cypha_rest log:"), inner_server));
    rest_health_btn_ = new QPushButton(QStringLiteral("GET /health"), inner_server);
    rest_ready_btn_ = new QPushButton(QStringLiteral("GET /ready"), inner_server);
    rest_models_btn_ = new QPushButton(QStringLiteral("GET /models"), inner_server);
    rest_clear_log_btn_ = new QPushButton(QStringLiteral("Clear log"), inner_server);
    row_log_hdr->addStretch(1);
    row_log_hdr->addWidget(rest_health_btn_);
    row_log_hdr->addWidget(rest_ready_btn_);
    row_log_hdr->addWidget(rest_models_btn_);
    row_log_hdr->addWidget(rest_clear_log_btn_);
    lay_server->addLayout(row_log_hdr);

    rest_log_ = new QPlainTextEdit(inner_server);
    rest_log_->setReadOnly(true);
    rest_log_->setMaximumBlockCount(500);
    rest_log_->setPlaceholderText(QStringLiteral("stdout/stderr from spawned cypha_rest appear here."));
    rest_log_->setMinimumHeight(120);
    lay_server->addWidget(rest_log_);

    lay_server->addWidget(new QLabel(QStringLiteral("REST base URL (must match --listen):"), inner_server));
    rest_base_edit_ = new QLineEdit(inner_server);
    rest_base_edit_->setPlaceholderText(QStringLiteral("http://127.0.0.1:8099"));
    lay_server->addWidget(rest_base_edit_);

    use_gh_chk_ = new QCheckBox(
        QStringLiteral("use_gh for native/REST predict, /update, and bulk train"), inner_server);
    use_gh_chk_->setChecked(studio_prefs_.inference_use_gh);
    lay_server->addWidget(use_gh_chk_);

    auto* row_rest_load = new QHBoxLayout();
    row_rest_load->addWidget(new QLabel(QStringLiteral("POST /load"), inner_server));
    rest_load_name_edit_ = new QLineEdit(inner_server);
    rest_load_name_edit_->setPlaceholderText(QStringLiteral("registry name"));
    rest_load_version_edit_ = new QLineEdit(inner_server);
    rest_load_version_edit_->setPlaceholderText(QStringLiteral("version"));
    rest_load_version_edit_->setText(QStringLiteral("latest"));
    rest_load_version_edit_->setMaximumWidth(100);
    rest_post_load_btn_ = new QPushButton(QStringLiteral("Load"), inner_server);
    row_rest_load->addWidget(rest_load_name_edit_, 1);
    row_rest_load->addWidget(rest_load_version_edit_);
    row_rest_load->addWidget(rest_post_load_btn_);
    lay_server->addLayout(row_rest_load);

    auto* row_upd = new QHBoxLayout();
    row_upd->addWidget(new QLabel(QStringLiteral("correct_label"), inner_server));
    update_label_edit_ = new QLineEdit(inner_server);
    update_label_edit_->setPlaceholderText(QStringLiteral("class name for POST /update"));
    row_upd->addWidget(update_label_edit_, 1);
    update_rest_btn_ = new QPushButton(QStringLiteral("POST /update (train step)"), inner_server);
    update_rest_btn_->setEnabled(false);
    row_upd->addWidget(update_rest_btn_);
    lay_server->addLayout(row_upd);

    result_label_ = new QLabel(QStringLiteral("—"), central);
    result_label_->setWordWrap(true);
    result_label_->setFrameShape(QFrame::StyledPanel);
    result_label_->setMinimumHeight(48);
    result_label_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    main_layout->addWidget(result_label_);

    setCentralWidget(central);

    registry_root_ = studio_prefs_.effective_registry_root();
    if (reg_root_label_ != nullptr) {
      reg_root_label_->setText(registry_root_.isEmpty() ? QStringLiteral("(none)") : registry_root_);
    }

    auto* file_menu = menuBar()->addMenu(QStringLiteral("&File"));
    auto* settings_act = file_menu->addAction(QStringLiteral("Settings…"));
    connect(settings_act, &QAction::triggered, this, [this]() {
      SettingsDialog dlg(studio_prefs_, this);
      if (dlg.exec() != QDialog::Accepted) {
        return;
      }
      studio_prefs_ = dlg.result_preferences();
      apply_studio_preferences();
    });

    auto* view_menu = menuBar()->addMenu(QStringLiteral("&View"));
    view_confusion_action_ = view_menu->addAction(QStringLiteral("Confusion Matrix…"));
    view_confusion_action_->setEnabled(false);
    view_confusion_action_->setToolTip(
        QStringLiteral("Show confusion matrix after batch predict on a labeled CSV."));
    connect(view_confusion_action_, &QAction::triggered, this, [this]() {
      show_confusion_dialog(this, batch_predict_y_true_, batch_predict_y_pred_);
    });

    QSettings ui_settings(QStringLiteral("Cypha"), QStringLiteral("CyphaQtShell"));
    if (ui_settings.contains(QStringLiteral("geometry"))) {
      restoreGeometry(ui_settings.value(QStringLiteral("geometry")).toByteArray());
    } else {
      resize(980, 980);
    }
    if (main_tabs_ != nullptr) {
      const int tab = ui_settings.value(QStringLiteral("mainTab"), 0).toInt();
      if (tab >= 0 && tab < main_tabs_->count()) {
        main_tabs_->setCurrentIndex(tab);
      }
    }
    if (ui_settings.contains(QStringLiteral("restListen")) && rest_listen_edit_ != nullptr) {
      rest_listen_edit_->setText(ui_settings.value(QStringLiteral("restListen")).toString());
    }
    if (ui_settings.contains(QStringLiteral("restBaseUrl")) && rest_base_edit_ != nullptr) {
      rest_base_edit_->setText(ui_settings.value(QStringLiteral("restBaseUrl")).toString());
    }

    rest_proc_.setProcessChannelMode(QProcess::SeparateChannels);
    connect(&rest_proc_, &QProcess::readyReadStandardOutput, this, [this]() {
      append_server_log(QStringLiteral("[stdout] "), QString::fromUtf8(rest_proc_.readAllStandardOutput()));
    });
    connect(&rest_proc_, &QProcess::readyReadStandardError, this, [this]() {
      append_server_log(QStringLiteral("[stderr] "), QString::fromUtf8(rest_proc_.readAllStandardError()));
    });

    connect(&rest_proc_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
      rest_status_label_->setText(QStringLiteral("Server: error (see log)"));
      rest_start_btn_->setEnabled(true);
      rest_stop_btn_->setEnabled(false);
      append_server_log(QStringLiteral("[process] error: %1\n").arg(rest_proc_.errorString()));
    });
    connect(&rest_proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus st) {
              rest_status_label_->setText(QStringLiteral("Server: stopped"));
              rest_start_btn_->setEnabled(true);
              rest_stop_btn_->setEnabled(false);
              append_server_log(QStringLiteral("[process] exited code=%1 status=%2\n")
                                    .arg(code)
                                    .arg(st == QProcess::NormalExit ? QStringLiteral("normal")
                                                                     : QStringLiteral("crash")));
            });

    connect(load_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Open .cypha"), QString(),
                                                        QStringLiteral("Cypha model (*.cypha);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      cypha_path_ = path;
      apply_model_load();
    });

    connect(new_model_btn_, &QPushButton::clicked, this, [this]() {
      // Dialog: input_dim + field_dim
      QDialog dlg(this);
      dlg.setWindowTitle(QStringLiteral("New model — parameters"));
      auto* form = new QFormLayout(&dlg);
      auto* dim_spin = new QSpinBox(&dlg);
      dim_spin->setRange(1, 4096);
      dim_spin->setValue(8);
      dim_spin->setToolTip(QStringLiteral("Raw feature / latent dimension (enc_W is d×d identity at start)"));
      auto* fd_spin = new QSpinBox(&dlg);
      fd_spin->setRange(1, 4096);
      fd_spin->setValue(24);
      fd_spin->setToolTip(QStringLiteral("Temporal field dimension"));
      auto* temp_spin = new QDoubleSpinBox(&dlg);
      temp_spin->setRange(0.01, 100.0);
      temp_spin->setValue(1.0);
      temp_spin->setSingleStep(0.1);
      form->addRow(QStringLiteral("Input / latent dim (d):"), dim_spin);
      form->addRow(QStringLiteral("Field dim (fd):"),         fd_spin);
      form->addRow(QStringLiteral("Temperature:"),           temp_spin);
      auto* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
      form->addRow(btns);
      connect(btns, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
      connect(btns, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
      if (dlg.exec() != QDialog::Accepted) return;

      cypha::FreshModelParams p;
      p.input_dim   = dim_spin->value();
      p.field_dim   = fd_spin->value();
      p.temperature = temp_spin->value();

      // Save to a temp file so reinit_native_train_state can load it via path
      const QString tmp_dir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
      const QString tmp_path = tmp_dir + QStringLiteral("/cypha_new_%1_%2.cypha")
                                             .arg(p.input_dim).arg(p.field_dim);
      try {
        cypha::create_and_save_fresh_model(tmp_path.toUtf8().constData(), p);
      } catch (const std::exception& e) {
        QMessageBox::critical(this, QStringLiteral("New model failed"),
                              QString::fromUtf8(e.what()));
        return;
      }
      ff_path_.clear();
      ff_label_->setText(QStringLiteral("(embedded in fresh model)"));
      f_field_flat_.clear();
      cypha_path_ = tmp_path;
      apply_model_load();
      path_label_->setText(QStringLiteral("New model (unsaved) — d=%1, fd=%2  [%3]")
                               .arg(p.input_dim).arg(p.field_dim).arg(tmp_path));
    });

    connect(ff_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("F_field JSON"), QString(),
                                                        QStringLiteral("JSON (*.json);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      ff_path_ = path;
      ff_label_->setText(path);
      if (!cypha_path_.isEmpty()) {
        apply_model_load();
      }
    });

    connect(pre_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("preprocessor.json"), QString(),
                                                        QStringLiteral("JSON (*.json);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      pre_path_ = path;
      pre_label_->setText(path);
      reload_preprocessor_only();
    });

    connect(pre_clear_btn_, &QPushButton::clicked, this, [this]() {
      pre_path_.clear();
      pre_.reset();
      pre_label_->setText(QStringLiteral("(optional)"));
      update_features_hint();
    });

    connect(fit_pre_btn_, &QPushButton::clicked, this, [this]() {
      if (!last_csv_ok_ || last_csv_.n_rows == 0) {
        QMessageBox::information(this, QStringLiteral("Fit preprocessor"),
                                 QStringLiteral("Run \"Inspect CSV\" on a classification CSV first."));
        return;
      }
      if (last_csv_.x_rowmajor.empty()) {
        QMessageBox::information(this, QStringLiteral("Fit preprocessor"),
                                 QStringLiteral("No numeric feature data in the last inspected CSV."));
        return;
      }
      open_fit_preprocessor_dialog();
    });

    // Column picker → text field sync
    connect(col_target_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
      // Uncheck the newly selected target and re-check previously unchecked target
      col_picker_updating_ = true;
      for (int i = 0; i < col_feature_list_->count(); ++i) {
        const bool is_target = (i == col_target_combo_->currentIndex());
        if (is_target) {
          col_feature_list_->item(i)->setCheckState(Qt::Unchecked);
        }
      }
      col_picker_updating_ = false;
      sync_col_picker_to_text_fields();
    });

    connect(col_feature_list_, &QListWidget::itemChanged, this, [this](QListWidgetItem*) {
      sync_col_picker_to_text_fields();
    });

    connect(csv_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Training CSV"),
                                                        file_dialog_start_dir(),
                                                        QStringLiteral("CSV (*.csv);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      csv_path_ = path;
      csv_label_->setText(QFileInfo(path).fileName());
      last_csv_ok_ = false;
      populate_column_picker(path);
    });

    connect(csv_inspect_btn_, &QPushButton::clicked, this, [this]() {
      if (csv_path_.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("CSV"), QStringLiteral("Choose a CSV file first."));
        return;
      }
      try {
        const cypha::CsvDenseResult loaded =
            cypha::load_csv_dense(qstring_to_fs_path(csv_path_), build_csv_spec());
        last_csv_ = loaded;
        last_csv_ok_ = true;

        // Stats label
        QString stats = QStringLiteral("rows=%1  features=%2").arg(last_csv_.n_rows).arg(last_csv_.n_features);
        if (!last_csv_.y_class.empty()) {
          // Count unique classes
          std::unordered_map<std::string, int> cls_cnt;
          for (const auto& y : last_csv_.y_class) { cls_cnt[y]++; }
          stats += QStringLiteral("  classes=%1").arg(static_cast<int>(cls_cnt.size()));
        }
        if (val_split_spin_->value() > 0) {
          const int val_n = static_cast<int>(last_csv_.n_rows * val_split_spin_->value() / 100.0);
          stats += QStringLiteral("  val=%1").arg(val_n);
        }
        csv_stats_label_->setText(stats);

        // Legacy text panel
        QString lines = QStringLiteral("CSV: %1\nrows=%2 features=%3\n")
                            .arg(csv_path_)
                            .arg(last_csv_.n_rows)
                            .arg(last_csv_.n_features);
        if (!last_csv_.y_class.empty()) {
          lines += QStringLiteral("first target (class): %1\n")
                       .arg(QString::fromStdString(last_csv_.y_class[0]));
        }
        if (!last_csv_.y_regression.empty()) {
          lines += QStringLiteral("first target (regression): %1\n").arg(last_csv_.y_regression[0]);
        }
        QStringList xs;
        int cap = last_csv_.n_features;
        if (cap > 16) {
          cap = 16;
        }
        for (int j = 0; j < cap; ++j) {
          xs << QString::number(last_csv_.x_rowmajor[static_cast<std::size_t>(j)], 'g', 8);
        }
        if (last_csv_.n_features > cap) {
          xs << QStringLiteral("…");
        }
        lines += QStringLiteral("first row X (truncated): %1\n").arg(xs.join(QLatin1Char(',')));
        dataset_info_->setPlainText(lines);

        // Preview table
        refresh_csv_preview();
      } catch (const std::exception& ex) {
        last_csv_ok_ = false;
        QMessageBox::warning(this, QStringLiteral("CSV"), QString::fromUtf8(ex.what()));
      }
    });

    connect(csv_fill_row0_btn_, &QPushButton::clicked, this, [this]() {
      if (!last_csv_ok_ || last_csv_.n_rows < 1) {
        QMessageBox::information(this, QStringLiteral("CSV"),
                                 QStringLiteral("Run Inspect CSV successfully first."));
        return;
      }
      const int need = feature_input_dim();
      if (need <= 0 || !model_) {
        QMessageBox::information(this, QStringLiteral("Features"),
                                 QStringLiteral("Load a .cypha (and optional preprocessor) first."));
        return;
      }
      if (last_csv_.n_features != need) {
        QMessageBox::warning(
            this, QStringLiteral("Features"),
            QStringLiteral("CSV has %1 feature columns but the current model/preprocessor expects %2.")
                .arg(last_csv_.n_features)
                .arg(need));
        return;
      }
      QStringList parts;
      for (int j = 0; j < last_csv_.n_features; ++j) {
        parts << QString::number(last_csv_.x_rowmajor[static_cast<std::size_t>(j)], 'g', 17);
      }
      features_edit_->setText(parts.join(QLatin1Char(',')));
    });

    connect(csv_bulk_train_btn_, &QPushButton::clicked, this, [this]() {
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Bulk /update"),
                                 QStringLiteral("Set REST base URL (start cypha_rest or enter URL)."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      if (csv_path_.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Bulk /update"),
                                 QStringLiteral("Choose a training CSV first."));
        return;
      }
      if (bulk_rest_thread_ != nullptr || bulk_thread_ != nullptr || bulk_mke_thread_ != nullptr) {
        QMessageBox::information(this, QStringLiteral("Bulk /update"),
                                 QStringLiteral("Bulk training already in progress."));
        return;
      }
      cypha::CsvDenseResult data;
      try {
        data = cypha::load_csv_dense(qstring_to_fs_path(csv_path_), build_csv_spec());
      } catch (const std::exception& ex) {
        QMessageBox::warning(this, QStringLiteral("Bulk /update"), QString::fromUtf8(ex.what()));
        return;
      }
      const bool reg_mode = csv_regression_chk_->isChecked();
      if (reg_mode && data.y_regression.empty()) {
        QMessageBox::information(this, QStringLiteral("Bulk /update"),
                                 QStringLiteral("Enable “regression target” and use a numeric target column."));
        return;
      }
      if (!reg_mode && data.y_class.empty()) {
        QMessageBox::information(this, QStringLiteral("Bulk /update"),
                                 QStringLiteral("Classification CSV needs string targets (or enable regression mode)."));
        return;
      }
      int n = data.n_rows;
      const int cap = csv_bulk_max_rows_spin_->value();
      if (cap > 0 && cap < n) {
        n = cap;
      }

      QString mke_clab = mke_correct_label_edit_->text().trimmed();
      if (mke_clab.isEmpty()) {
        mke_clab = default_mke_correct_label();
      }

      BulkRestUpdateJob job{};
      job.base_url = base;
      job.data = std::move(data);
      job.n_rows = n;
      job.regression = reg_mode;
      job.use_gh = use_gh_chk_->isChecked();
      job.mke_correct_label = mke_clab;
      job.router_train_label = mke_router_label_edit_->text().trimmed();
      for (double u : replay_u01_cache_) {
        job.replay_u01.append(u);
      }
      job.cancel_flag = &bulk_rest_cancel_flag_;

      bulk_rest_cancel_flag_.store(false);
      bulk_rest_accum_losses_.clear();
      set_bulk_training_ui(true);
      result_label_->setText(QStringLiteral("Bulk REST /update 0 / %1…").arg(n));

      auto* prog = new QProgressDialog(QStringLiteral("POST /update per CSV row…"),
                                       QStringLiteral("Cancel"), 0, n, this);
      prog->setWindowModality(Qt::WindowModal);
      prog->setMinimumDuration(0);
      prog->setAttribute(Qt::WA_DeleteOnClose);

      bulk_rest_thread_ = new QThread(this);
      bulk_rest_worker_ = new BulkRestUpdateWorker();
      bulk_rest_worker_->moveToThread(bulk_rest_thread_);

      connect(bulk_rest_thread_, &QThread::started, bulk_rest_worker_,
              [job = std::move(job), w = bulk_rest_worker_]() mutable { w->run(std::move(job)); });
      connect(bulk_rest_worker_, &BulkRestUpdateWorker::progress, prog,
              [prog](int step, int total) {
                prog->setMaximum(total);
                prog->setValue(step);
              });
      connect(bulk_rest_worker_, &BulkRestUpdateWorker::progress, this,
              [this, n](int step, int /*total*/) {
                result_label_->setText(QStringLiteral("Bulk REST /update %1 / %2…").arg(step).arg(n));
              });
      connect(bulk_rest_worker_, &BulkRestUpdateWorker::stepLoss, this,
              [this](int step, double loss) {
                bulk_rest_accum_losses_.append(loss);
                if (loss_chart_ != nullptr) {
                  loss_chart_->append_rest_step(step, loss);
                }
              });
      connect(prog, &QProgressDialog::canceled, bulk_rest_worker_,
              &BulkRestUpdateWorker::requestCancel);
      connect(bulk_rest_worker_, &BulkRestUpdateWorker::finished, this,
              [this, prog](bool ok, QVector<double> losses, bool cancelled, QString error) {
                prog->close();
                on_bulk_rest_finished(ok, std::move(losses), cancelled, std::move(error));
              });
      connect(bulk_rest_worker_, &BulkRestUpdateWorker::finished, bulk_rest_thread_, &QThread::quit);
      connect(bulk_rest_thread_, &QThread::finished, bulk_rest_thread_, &QObject::deleteLater);
      connect(bulk_rest_thread_, &QThread::finished, this, [this]() {
        bulk_rest_thread_ = nullptr;
        if (bulk_rest_worker_ != nullptr) {
          bulk_rest_worker_->deleteLater();
          bulk_rest_worker_ = nullptr;
        }
      });

      bulk_rest_thread_->start();
    });

    connect(csv_bulk_native_btn_, &QPushButton::clicked, this, [this]() {
      if (csv_regression_chk_->isChecked()) {
        QMessageBox::information(this, QStringLiteral("Bulk native"),
                                 QStringLiteral("Native bulk supports classification CSV only; use Bulk REST for MKE."));
        return;
      }
      if (csv_path_.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Bulk native"), QStringLiteral("Choose a training CSV first."));
        return;
      }
      if (bulk_thread_ != nullptr) {
        QMessageBox::information(this, QStringLiteral("Bulk native"), QStringLiteral("Training already in progress."));
        return;
      }
      const cypha::CsvDenseSpec spec = build_csv_spec();
      if (spec.regression) {
        QMessageBox::information(this, QStringLiteral("Bulk native"),
                                 QStringLiteral("Need a classification CSV with string targets."));
        return;
      }
      int total_n = 0;
      try {
        total_n = cypha::count_csv_dense_rows(qstring_to_fs_path(csv_path_), spec);
      } catch (const std::exception& ex) {
        QMessageBox::warning(this, QStringLiteral("Bulk native"), QString::fromUtf8(ex.what()));
        return;
      }
      if (total_n <= 0) {
        QMessageBox::information(this, QStringLiteral("Bulk native"), QStringLiteral("CSV has no data rows."));
        return;
      }
      const int cap = csv_bulk_max_rows_spin_->value();
      if (cap > 0 && cap < total_n) {
        total_n = cap;
      }
      if (!native_train_ok_) {
        QMessageBox::warning(this, QStringLiteral("Bulk native"),
                             QStringLiteral("Native training state not ready (need F_field + successful model load)."));
        return;
      }
      const int val_pct = (val_split_spin_ != nullptr) ? val_split_spin_->value() : 0;
      const int val_n   = (val_pct > 0) ? std::max(1, static_cast<int>(total_n * val_pct / 100.0)) : 0;
      const int train_n = total_n - val_n;
      if (train_n <= 0) {
        QMessageBox::information(this, QStringLiteral("Bulk native"), QStringLiteral("No training rows."));
        return;
      }

      bulk_val_n_    = val_n;
      bulk_total_n_  = total_n;
      bulk_train_data_ = {};
      bulk_accum_losses_.clear();
      bulk_accum_log_.clear();
      bulk_train_n_ = train_n;
      if (loss_chart_ != nullptr) {
        loss_chart_->reset_native_roll_window();
      }

      BulkNativeTrainJob job{};
      job.csv_path = qstring_to_fs_path(csv_path_);
      job.csv_spec = spec;
      job.total_rows = total_n;
      job.train_n = train_n;
      job.chunk_rows = effective_csv_chunk_rows();
      job.sort_by_uncertainty =
          sort_by_uncertainty_chk_ != nullptr && sort_by_uncertainty_chk_->isChecked();
      job.curriculum = curriculum_chk_ != nullptr && curriculum_chk_->isChecked();
      job.model = model_.get();
      job.mem = native_mem_.get();
      job.replay = native_replay_.get();
      job.pre = pre_.get();
      job.kernel_mem = native_kernel_mem_.get();
      job.use_gh = use_gh_chk_->isChecked();
      job.use_kernel_llr = native_use_kernel_llr_;
      job.kernel_blend = native_kernel_blend_;
      job.world_lr = native_world_lr_;
      job.delta_lr = native_delta_lr_;
      job.ood_sigma = native_ood_sigma_;
      job.gh_inv_v = native_gh_inv_v_;
      job.gh_R_base = native_gh_R_base_;
      job.tsp = native_tsp_;
      job.replay_u01 = replay_u01_cache_;
      job.total_steps_start = native_total_steps_;
      job.llr_ema_start = native_llr_ema_;
      job.ema_loss_start = train_prog_ema_loss_;
      job.win_total_start = train_prog_win_total_;
      job.win_correct_start = train_prog_win_correct_;
      job.gh_chi_start = native_gh_chi_;
      job.gh_psi_start = native_gh_psi_;
      job.enc_updates_start = native_enc_updates_;
      job.rng = native_rng_;
      job.cancel_flag = &bulk_cancel_flag_;

      bulk_cancel_flag_.store(false);
      set_bulk_training_ui(true);
      result_label_->setText(QStringLiteral("Training 0 / %1…").arg(train_n));
      sync_native_kernel_from_ui();

      bulk_thread_ = new QThread(this);
      bulk_worker_ = new BulkNativeTrainWorker();
      bulk_worker_->moveToThread(bulk_thread_);

      connect(bulk_thread_, &QThread::started, bulk_worker_, [job = std::move(job), w = bulk_worker_]() mutable {
        w->run(std::move(job));
      });
      connect(bulk_worker_, &BulkNativeTrainWorker::progress, this,
              [this](int step, int total) { on_bulk_progress(step, total); });
      connect(bulk_worker_, &BulkNativeTrainWorker::lossReported, this,
              [this](int step, double loss) { on_bulk_loss(step, loss); });
      connect(bulk_worker_, &BulkNativeTrainWorker::valAccReported, this,
              [this](int step, double acc) { on_bulk_val_acc(step, acc); });
      connect(bulk_worker_, &BulkNativeTrainWorker::stepLogged, this,
              [this](int step, double loss, bool correct) { on_bulk_step_logged(step, loss, correct); });
      connect(bulk_worker_, &BulkNativeTrainWorker::error, this,
              [this](const QString& msg) { on_bulk_error(msg); });
      connect(bulk_worker_, &BulkNativeTrainWorker::finished, this,
              [this](bool ok, BulkNativeTrainResult result, QVector<BulkLogEntry> log) {
                on_bulk_worker_finished(ok, result, std::move(log));
              });
      connect(bulk_worker_, &BulkNativeTrainWorker::finished, bulk_thread_, &QThread::quit);
      connect(bulk_thread_, &QThread::finished, bulk_thread_, &QObject::deleteLater);
      connect(bulk_thread_, &QThread::finished, this, [this]() {
        bulk_thread_ = nullptr;
        if (bulk_worker_ != nullptr) {
          bulk_worker_->deleteLater();
          bulk_worker_ = nullptr;
        }
      });

      bulk_thread_->start();
    });

    connect(bulk_native_cancel_btn_, &QPushButton::clicked, this, [this]() {
      // Cancels whichever bulk operation is currently running: native, REST/update, or MKE.
      bulk_cancel_flag_.store(true, std::memory_order_relaxed);
      bulk_rest_cancel_flag_.store(true, std::memory_order_relaxed);
      bulk_mke_cancel_flag_.store(true, std::memory_order_relaxed);
      if (bulk_worker_ != nullptr) {
        QMetaObject::invokeMethod(bulk_worker_, "requestCancel", Qt::QueuedConnection);
      }
      if (bulk_rest_worker_ != nullptr) {
        QMetaObject::invokeMethod(bulk_rest_worker_, "requestCancel", Qt::QueuedConnection);
      }
      if (bulk_mke_worker_ != nullptr) {
        QMetaObject::invokeMethod(bulk_mke_worker_, "requestCancel", Qt::QueuedConnection);
      }
    });

    // ── MKE Regressor connects ─────────────────────────────────────────────────
    connect(mke_init_btn_, &QPushButton::clicked, this, [this]() {
      mke_w_by_label_.clear();
      mke_p_by_label_.clear();
      if (mke_result_label_ != nullptr)
        mke_result_label_->setText(QStringLiteral("MKE state cleared — will re-initialize on first train step."));
    });

    connect(mke_bulk_btn_, &QPushButton::clicked, this, [this]() {
      if (!native_train_ok_ || model_ == nullptr || native_mem_ == nullptr || native_replay_ == nullptr) {
        QMessageBox::information(this, QStringLiteral("MKE bulk"),
                                 QStringLiteral("Load a model with F_field and native train state first."));
        return;
      }
      if (!last_csv_ok_ || last_csv_.n_rows == 0) {
        QMessageBox::information(this, QStringLiteral("MKE bulk"),
                                 QStringLiteral("Inspect a CSV first (regression mode checkbox must be ON)."));
        return;
      }
      if (last_csv_.y_regression.empty()) {
        QMessageBox::warning(this, QStringLiteral("MKE bulk"),
                             QStringLiteral("No regression targets loaded.  Enable \"CSV target is numeric "
                                            "regression\" and Inspect CSV again."));
        return;
      }
      if (bulk_mke_thread_ != nullptr || bulk_thread_ != nullptr || bulk_rest_thread_ != nullptr) {
        QMessageBox::information(this, QStringLiteral("MKE bulk"), QStringLiteral("Bulk training already in progress."));
        return;
      }

      const int n = last_csv_.n_rows;
      BulkMkeTrainJob job{};
      job.data = last_csv_;
      job.n_rows = n;
      job.model = model_.get();
      job.mem = native_mem_.get();
      job.replay = native_replay_.get();
      job.pre = pre_.get();
      job.w_by_label = &mke_w_by_label_;
      job.p_by_label = &mke_p_by_label_;
      job.forgetting_factor = mke_ff_spin_->value();
      job.pi_floor = mke_pi_spin_->value();
      job.tsp = native_tsp_;
      job.world_lr = native_world_lr_;
      job.delta_lr = native_delta_lr_;
      job.ood_sigma = native_ood_sigma_;
      job.rng = native_rng_;
      job.enc_updates_start = native_enc_updates_;
      job.enc_updates = &native_enc_updates_;
      job.total_steps = &native_total_steps_;
      job.router_train_label = mke_router_label_edit_ != nullptr ? mke_router_label_edit_->text().trimmed() : QString{};
      job.cancel_flag = &bulk_mke_cancel_flag_;

      bulk_mke_cancel_flag_.store(false);
      set_bulk_training_ui(true);
      if (mke_result_label_ != nullptr) {
        mke_result_label_->setText(QStringLiteral("MKE bulk train 0 / %1…").arg(n));
      }

      auto* prog = new QProgressDialog(QStringLiteral("MKE bulk train…"), QStringLiteral("Cancel"), 0, n, this);
      prog->setWindowModality(Qt::WindowModal);
      prog->setMinimumDuration(0);
      prog->setAttribute(Qt::WA_DeleteOnClose);

      bulk_mke_thread_ = new QThread(this);
      bulk_mke_worker_ = new BulkMkeTrainWorker();
      bulk_mke_worker_->moveToThread(bulk_mke_thread_);

      connect(bulk_mke_thread_, &QThread::started, bulk_mke_worker_,
              [job = std::move(job), w = bulk_mke_worker_]() mutable { w->run(std::move(job)); });
      connect(bulk_mke_worker_, &BulkMkeTrainWorker::progress, prog,
              [prog](int step, int total) {
                prog->setMaximum(total);
                prog->setValue(step);
              });
      connect(bulk_mke_worker_, &BulkMkeTrainWorker::progress, this,
              [this, n](int step, int /*total*/) {
                if (mke_result_label_ != nullptr) {
                  mke_result_label_->setText(QStringLiteral("MKE bulk train %1 / %2…").arg(step).arg(n));
                }
              });
      connect(prog, &QProgressDialog::canceled, bulk_mke_worker_, &BulkMkeTrainWorker::requestCancel);
      connect(bulk_mke_worker_, &BulkMkeTrainWorker::finished, this,
              [this, prog](bool ok, int n_ok, double mse, double rmse, bool cancelled, QString error) {
                prog->close();
                on_bulk_mke_finished(ok, n_ok, mse, rmse, cancelled, std::move(error));
              });
      connect(bulk_mke_worker_, &BulkMkeTrainWorker::finished, bulk_mke_thread_, &QThread::quit);
      connect(bulk_mke_thread_, &QThread::finished, bulk_mke_thread_, &QObject::deleteLater);
      connect(bulk_mke_thread_, &QThread::finished, this, [this]() {
        bulk_mke_thread_ = nullptr;
        if (bulk_mke_worker_ != nullptr) {
          bulk_mke_worker_->deleteLater();
          bulk_mke_worker_ = nullptr;
        }
      });

      bulk_mke_thread_->start();
    });

    connect(mke_predict_btn_, &QPushButton::clicked, this, [this]() {
      if (!model_ || native_mem_ == nullptr) return;
      const QString t = features_edit_->text().trimmed();
      if (t.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("MKE predict"),
                                 QStringLiteral("Enter comma-separated features."));
        return;
      }
      std::vector<double> x_raw;
      QString perr;
      if (!parse_feature_vector(t, feature_input_dim(), x_raw, &perr)) {
        QMessageBox::warning(this, QStringLiteral("Input"), perr);
        return;
      }
      std::vector<double> phi = x_raw;
      if (pre_ != nullptr) phi = pre_->transform_one(x_raw);
      const int d = model_->d_latent;
      const int K = static_cast<int>(model_->labels.size());
      if (static_cast<int>(phi.size()) != d || K == 0) {
        QMessageBox::warning(this, QStringLiteral("MKE predict"),
                             QStringLiteral("Dim mismatch or no classes in model."));
        return;
      }
      // Compute LLR via native infer pipeline
      std::vector<double> h_out;
      cypha::batch_encode(*model_, phi.data(), 1, h_out);
      std::vector<double> llr;
      cypha::score_matrix_use_field(*model_, h_out.data(), 1, llr);

      // Build expert mu array (phi @ w_k for each k, where phi is the latent encoding)
      // Note: mke_scalar_train_step_from_phi uses phi (= RFF features = d_latent-dim),
      // so w_k has size d.
      std::vector<double> expert_mu(static_cast<std::size_t>(K), 0.0);
      for (int k = 0; k < K; ++k) {
        const std::string& lbl = model_->labels[static_cast<std::size_t>(k)];
        auto it = mke_w_by_label_.find(lbl);
        if (it == mke_w_by_label_.end()) continue;
        const std::vector<double>& wk = it->second;
        if (static_cast<int>(wk.size()) != d) continue;
        double dot = 0.0;
        for (int j = 0; j < d; ++j) dot += phi[static_cast<std::size_t>(j)] * wk[static_cast<std::size_t>(j)];
        expert_mu[static_cast<std::size_t>(k)] = dot;
      }
      double entropy = 0.0;
      const double y_hat = cypha::regression::mke_scalar_predict_from_llr(
          llr.data(), K, model_->temperature, 1e-9, expert_mu.data(), &entropy);
      if (mke_result_label_ != nullptr)
        mke_result_label_->setText(
            QStringLiteral("MKE predict: y_hat=%1  routing_entropy=%2")
                .arg(y_hat, 0, 'g', 8).arg(entropy, 0, 'g', 6));
    });

    connect(replay_u01_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("replay_u01 JSON array"), QString(),
                                                        QStringLiteral("JSON (*.json);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      QFile f(path);
      if (!f.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, QStringLiteral("replay_u01"), QStringLiteral("Cannot open file."));
        return;
      }
      QJsonParseError pe{};
      const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &pe);
      if (pe.error != QJsonParseError::NoError || !doc.isArray()) {
        QMessageBox::warning(this, QStringLiteral("replay_u01"),
                             QStringLiteral("File must be a JSON array of numbers."));
        return;
      }
      replay_u01_cache_.clear();
      for (const QJsonValue& v : doc.array()) {
        if (!v.isDouble()) {
          QMessageBox::warning(this, QStringLiteral("replay_u01"), QStringLiteral("Array must contain numbers."));
          replay_u01_cache_.clear();
          return;
        }
        replay_u01_cache_.push_back(v.toDouble());
      }
      replay_u01_label_->setText(QStringLiteral("%1 (%2 values)").arg(path).arg(replay_u01_cache_.size()));
    });

    connect(native_train_one_btn_, &QPushButton::clicked, this, [this]() {
      if (!native_train_ok_ || model_ == nullptr) {
        QMessageBox::information(this, QStringLiteral("Native train"),
                                 QStringLiteral("Load a model with F_field and ensure native train is available."));
        return;
      }
      const QString t = features_edit_->text().trimmed();
      if (t.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Native train"), QStringLiteral("Enter feature values."));
        return;
      }
      const QString clab = update_label_edit_->text().trimmed();
      if (clab.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Native train"), QStringLiteral("Enter correct_label."));
        return;
      }
      std::vector<double> x_raw;
      QString perr;
      if (!parse_feature_vector(t, feature_input_dim(), x_raw, &perr)) {
        QMessageBox::warning(this, QStringLiteral("Native train"), perr);
        return;
      }
      std::vector<double> x_latent = x_raw;
      if (pre_ != nullptr) {
        x_latent = pre_->transform_one(x_raw);
      }
      double loss = 0.0;
      cypha::MemoryTrainMeta meta{};
      if (!run_native_train_on_latent(x_latent, clab.toStdString(), &loss, &meta)) {
        QMessageBox::warning(this, QStringLiteral("Native train"), QStringLiteral("train_step failed."));
        return;
      }
      append_train_log_entry(native_total_steps_, clab, loss, meta.correct);
      if (loss_chart_ != nullptr) {
        loss_chart_->append_native_step(native_total_steps_, loss, meta.correct);
      }
      refresh_train_progress(true);  // update progress panel after single step
      result_label_->setText(QStringLiteral("[native train] step=%1 loss=%2 %3")
                                 .arg(native_total_steps_)
                                 .arg(loss, 0, 'g', 8)
                                 .arg(meta.correct ? QStringLiteral("\u2713") : QStringLiteral("\u2717")));
    });

    connect(hp_apply_btn_, &QPushButton::clicked, this, [this]() {
      apply_native_hparams_from_ui();
    });
    connect(hp_defaults_btn_, &QPushButton::clicked, this, [this]() {
      set_native_hparams_defaults();
    });
    connect(save_native_btn_, &QPushButton::clicked, this, [this]() {
      if (!native_train_ok_ || model_ == nullptr || native_mem_ == nullptr) {
        QMessageBox::information(this, QStringLiteral("Save"),
                                 QStringLiteral("Load a model with F_field first (native train state required)."));
        return;
      }
      const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save .cypha"), QString(),
                                                        QStringLiteral("Cypha model (*.cypha);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      QString err;
      if (!save_native_model_to_path(path, &err)) {
        QMessageBox::warning(this, QStringLiteral("Save failed"), err);
        return;
      }
      QMessageBox::information(this, QStringLiteral("Save"), QStringLiteral("Wrote %1").arg(path));
    });

    connect(train_log_clear_btn_, &QPushButton::clicked, this, [this]() {
      if (train_log_table_ != nullptr) {
        train_log_table_->setRowCount(0);
      }
    });

    connect(train_prog_reset_btn_, &QPushButton::clicked, this, [this]() {
      train_prog_win_correct_ = 0;
      train_prog_win_total_   = 0;
      train_prog_ema_loss_    = 0.0;
      if (loss_chart_ != nullptr) {
        loss_chart_->reset_native_roll_window();
      }
      if (train_prog_class_table_ != nullptr) {
        train_prog_class_table_->setRowCount(0);
      }
      if (train_prog_acc_bar_ != nullptr) {
        train_prog_acc_bar_->clear();
      }
      if (train_prog_label_ != nullptr) {
        train_prog_label_->setText(QStringLiteral(
            "Steps: 0  |  Acc(win): —  |  EMA loss: —  |  Classes: 0"));
      }
    });

    connect(train_log_export_btn_, &QPushButton::clicked, this, [this]() {
      if (train_log_table_ == nullptr || train_log_table_->rowCount() == 0) {
        QMessageBox::information(this, QStringLiteral("Export"), QStringLiteral("No log rows to export."));
        return;
      }
      const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Export train log"),
                                                        QStringLiteral("native_train_log.csv"),
                                                        QStringLiteral("CSV (*.csv);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      QFile f(path);
      if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("Export"), QStringLiteral("Cannot open file for write."));
        return;
      }
      QTextStream out(&f);
      out << QStringLiteral("step,label,loss,correct\n");
      for (int r = 0; r < train_log_table_->rowCount(); ++r) {
        auto* i0 = train_log_table_->item(r, 0);
        auto* i1 = train_log_table_->item(r, 1);
        auto* i2 = train_log_table_->item(r, 2);
        auto* i3 = train_log_table_->item(r, 3);
        out << (i0 ? i0->text() : QString()) << QLatin1Char(',')
            << (i1 ? i1->text() : QString()) << QLatin1Char(',')
            << (i2 ? i2->text() : QString()) << QLatin1Char(',')
            << (i3 ? i3->text() : QString()) << QLatin1Char('\n');
      }
    });

    // ── Experiment DB (M6) connects ───────────────────────────────────────────
#ifdef CYPHA_SHELL_EXPERIMENT_DB
    connect(exp_db_btn_, &QPushButton::clicked, this, [this]() {
      // Default path: ~/.cypha/experiments.db
      const QString home = QDir::homePath();
      const QString default_path = home + QStringLiteral("/.cypha/experiments.db");
      const QString path = QFileDialog::getSaveFileName(
          this, QStringLiteral("Open / create experiments.db"), default_path,
          QStringLiteral("SQLite DB (*.db);;All (*)"));
      if (path.isEmpty()) return;

      // Create parent dir if needed
      QFileInfo fi(path);
      if (!fi.dir().exists()) {
        fi.dir().mkpath(QStringLiteral("."));
      }
      std::string err;
      auto db = std::make_unique<cypha::ExperimentDb>();
      if (!db->open_file_rw(path.toUtf8().constData(), false, &err)) {
        QMessageBox::critical(this, QStringLiteral("Experiments DB"),
                              QStringLiteral("Cannot open: %1").arg(QString::fromStdString(err)));
        return;
      }
      // Apply canonical schema (idempotent DDL)
      if (!experiment_db_apply_canonical_schema(*db, kExperimentDdl, &err)) {
        QMessageBox::critical(this, QStringLiteral("Experiments DB"),
                              QStringLiteral("Schema error: %1").arg(QString::fromStdString(err)));
        return;
      }
      exp_db_ = std::move(db);
      exp_db_label_->setText(path);
      exp_start_run_btn_->setEnabled(true);
      exp_status_label_->setText(QStringLiteral("DB opened: %1").arg(path));
      experiment_refresh_all();
    });

    connect(exp_start_run_btn_, &QPushButton::clicked, this, [this]() {
      if (!exp_db_) return;
      const QString exp_name = exp_name_edit_->text().trimmed();
      const QString run_name = exp_run_name_edit_->text().trimmed();
      if (exp_name.isEmpty() || run_name.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Start run"),
                                 QStringLiteral("Enter experiment and run names."));
        return;
      }
      // Insert/ensure experiment
      const double now = static_cast<double>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;
      std::string exp_id = "exp_" + exp_name.toStdString();
      std::replace(exp_id.begin(), exp_id.end(), ' ', '_');
      std::string err;
      // Insert experiment — ignore "already exists" errors
      experiment_db_insert_experiment(*exp_db_, exp_id.c_str(),
                                      exp_name.toUtf8().constData(),
                                      "", "", "classification", now, "[]", &err);
      err.clear();

      // Generate run_id from timestamp
      const qint64 ts_ms = static_cast<qint64>(now * 1000.0);
      const std::string run_id = "run_" + std::to_string(ts_ms);
      const std::string config_json = "{\"shell\":\"qt\",\"model\":\""
          + (model_ ? std::to_string(model_->d_latent) + "d" : "none") + "\"}";
      if (!experiment_db_insert_run_pending(*exp_db_, run_id.c_str(), exp_id.c_str(),
                                            run_name.toUtf8().constData(),
                                            config_json.c_str(), now, now,
                                            "[]", "", &err)) {
        QMessageBox::warning(this, QStringLiteral("Start run"),
                             QStringLiteral("Insert failed: %1").arg(QString::fromStdString(err)));
        return;
      }
      exp_active_run_id_ = run_id;
      exp_active_experiment_id_ = exp_id;
      exp_finish_run_btn_->setEnabled(true);
      exp_status_label_->setText(QStringLiteral("Active run: %1 (%2)")
                                     .arg(QString::fromStdString(run_id))
                                     .arg(run_name));
      experiment_refresh_all();
    });

    connect(exp_refresh_btn_, &QPushButton::clicked, this, [this]() { experiment_refresh_all(); });

    connect(exp_runs_table_, &QTableWidget::itemSelectionChanged, this, [this]() {
      if (exp_compare_btn_ == nullptr || exp_runs_table_ == nullptr) {
        return;
      }
      exp_compare_btn_->setEnabled(exp_db_ != nullptr && exp_runs_table_->selectionModel() != nullptr &&
                                   !exp_runs_table_->selectionModel()->selectedRows().isEmpty());
    });

    connect(exp_compare_btn_, &QPushButton::clicked, this, [this]() { experiment_compare_selected_runs(); });

    connect(exp_compare_clear_btn_, &QPushButton::clicked, this, [this]() {
      if (exp_compare_chart_ != nullptr) {
        exp_compare_chart_->clear_losses();
      }
      if (exp_compare_summary_label_ != nullptr) {
        exp_compare_summary_label_->setText(QStringLiteral("(select 2+ runs with loss history)"));
      }
      if (exp_compare_stats_table_ != nullptr) {
        exp_compare_stats_table_->setRowCount(0);
        exp_compare_stats_table_->setVisible(false);
      }
      if (exp_compare_clear_btn_ != nullptr) {
        exp_compare_clear_btn_->setEnabled(false);
      }
    });

    connect(exp_experiments_table_, &QTableWidget::itemSelectionChanged, this, [this]() {
      if (exp_experiments_table_ == nullptr || exp_experiments_table_->currentRow() < 0) {
        return;
      }
      auto* id_item = exp_experiments_table_->item(exp_experiments_table_->currentRow(), 0);
      if (id_item != nullptr) {
        const QString full_id = id_item->data(Qt::UserRole).toString();
        if (!full_id.isEmpty()) {
          exp_active_experiment_id_ = full_id.toStdString();
          if (exp_name_edit_ != nullptr) {
            auto* name_item = exp_experiments_table_->item(exp_experiments_table_->currentRow(), 1);
            if (name_item != nullptr) {
              exp_name_edit_->setText(name_item->text());
            }
          }
        }
      }
      experiment_refresh_runs_table();
    });

    connect(exp_finish_run_btn_, &QPushButton::clicked, this, [this]() {
      if (!exp_db_ || exp_active_run_id_.empty()) return;
      const double now = static_cast<double>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;

      // Compute metrics from current native train state
      double accuracy = 0.0;
      int n_total = 0;
      int n_correct_total = 0;
      if (native_mem_ != nullptr) {
        for (std::size_t k = 0; k < native_mem_->n_obs_buf.size(); ++k) {
          n_total   += static_cast<int>(native_mem_->n_obs_buf[k]);
          n_correct_total += (k < native_mem_->n_correct.size())
                                 ? static_cast<int>(native_mem_->n_correct[k]) : 0;
        }
        if (n_total > 0)
          accuracy = static_cast<double>(n_correct_total) / n_total;
      }
      const int K = (model_ != nullptr) ? static_cast<int>(model_->labels.size()) : 0;

      const QString checkpoint_path = cypha_path_.isEmpty() ? QString{} : cypha_path_;

      std::string err;
      const bool ok = experiment_db_finish_run(
          *exp_db_, exp_active_run_id_.c_str(),
          "finished", now, now,
          /*duration_s=*/0.0,
          /*accuracy=*/accuracy,
          /*macro_f1=*/-1.0,
          /*r2_score=*/-1.0,
          /*rmse=*/-1.0,
          /*n_steps=*/native_total_steps_,
          /*n_classes=*/K,
          checkpoint_path.toUtf8().constData(),
          pre_path_.isEmpty() ? "" : pre_path_.toUtf8().constData(),
          "[]",
          &err);
      if (!ok) {
        QMessageBox::warning(this, QStringLiteral("Finish run"),
                             QStringLiteral("Failed: %1").arg(QString::fromStdString(err)));
        return;
      }
      exp_status_label_->setText(
          QStringLiteral("Run finished: %1  acc=%2  steps=%3  classes=%4")
              .arg(QString::fromStdString(exp_active_run_id_))
              .arg(accuracy, 0, 'f', 4)
              .arg(native_total_steps_)
              .arg(K));
      exp_active_run_id_.clear();
      exp_finish_run_btn_->setEnabled(false);
      experiment_refresh_all();
    });
#endif  // CYPHA_SHELL_EXPERIMENT_DB

    connect(reg_root_btn_, &QPushButton::clicked, this, [this]() {
      const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Registry root"), QString());
      if (dir.isEmpty()) {
        return;
      }
      registry_root_ = dir;
      reg_root_label_->setText(dir);
    });

    connect(reg_scan_btn_, &QPushButton::clicked, this, [this]() {
      if (registry_root_.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Registry"), QStringLiteral("Set a registry root directory."));
        return;
      }
      reg_refs_ = cypha::registry_scan(registry_root_.toUtf8().constData());
      reg_combo_->clear();
      for (const cypha::RegistryModelRef& r : reg_refs_) {
        reg_combo_->addItem(QStringLiteral("%1 / %2").arg(QString::fromStdString(r.name),
                                                            QString::fromStdString(r.version)));
      }
      refresh_registry_table_scan();
      dataset_info_->setPlainText(QStringLiteral("registry_scan: %1 bundle(s) under\n%2")
                                      .arg(reg_refs_.size())
                                      .arg(registry_root_));
    });

    connect(reg_models_table_, &QTableWidget::itemSelectionChanged, this, [this]() {
      sync_registry_selection_to_rest_load();
    });

    connect(reg_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
      if (idx >= 0 && idx < static_cast<int>(reg_refs_.size())) {
        sync_registry_selection_to_rest_load();
      }
    });

    connect(reg_rest_models_btn_, &QPushButton::clicked, this, [this]() {
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/models"),
                                 QStringLiteral("Enter REST base URL on the Server tab (or start cypha_rest)."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      const QUrl url(base + QStringLiteral("/models?summary=1"));
      const HttpJsonResult r = http_get_json(url);
      if (!r.ok) {
        QMessageBox::warning(this, QStringLiteral("/models"), r.err);
        return;
      }
      const QJsonArray models = r.obj.value(QStringLiteral("models")).toArray();
      refresh_registry_table_rest(models);
      dataset_info_->setPlainText(QStringLiteral("GET /models: %1 entries from %2").arg(models.size()).arg(base));
    });

    connect(reg_rest_load_btn_, &QPushButton::clicked, this, [this]() {
      sync_registry_selection_to_rest_load();
      if (rest_post_load_btn_ != nullptr) {
        QMetaObject::invokeMethod(rest_post_load_btn_, "click", Qt::QueuedConnection);
      }
    });

    connect(reg_load_btn_, &QPushButton::clicked, this, [this]() {
      const int i = selected_registry_index();
      if (i < 0 || i >= static_cast<int>(reg_refs_.size())) {
        QMessageBox::information(this, QStringLiteral("Registry"),
                                 QStringLiteral("Scan the registry and pick an entry."));
        return;
      }
      const cypha::RegistryModelRef& r = reg_refs_[static_cast<std::size_t>(i)];
      cypha_path_ = QString::fromStdString(r.model_path);
      if (!r.preprocessor_path.empty()) {
        pre_path_ = QString::fromStdString(r.preprocessor_path);
        pre_label_->setText(pre_path_);
      } else {
        pre_path_.clear();
        pre_label_->setText(QStringLiteral("(optional)"));
      }
      apply_model_load();
    });

    connect(card_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("card.json"), QString(),
                                                        QStringLiteral("JSON (*.json);;All (*)"));
      if (path.isEmpty()) {
        return;
      }
      card_path_ = path;
      card_label_->setText(path);
    });

    connect(reg_register_btn_, &QPushButton::clicked, this, [this]() {
      if (registry_root_.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Register"), QStringLiteral("Set registry root first."));
        return;
      }
      if (cypha_path_.isEmpty() || !QFile::exists(cypha_path_)) {
        QMessageBox::warning(this, QStringLiteral("Register"), QStringLiteral("Load a .cypha first."));
        return;
      }
      QString card = card_path_;
      if (card.isEmpty()) {
        const QFileInfo fi(cypha_path_);
        card = fi.absoluteDir().filePath(QStringLiteral("card.json"));
      }
      const QString default_name = QFileInfo(cypha_path_).completeBaseName();
      ModelCardEditorDialog dlg(card, default_name, this);
      if (dlg.exec() != QDialog::Accepted) {
        return;
      }
      card_path_ = card;
      card_label_->setText(card);
      const ModelCardEditResult er = dlg.edit_result();
      const auto ow =
          QMessageBox::question(this, QStringLiteral("Register"),
                                QStringLiteral("Overwrite if %1/%2 already exists?")
                                    .arg(er.name, er.version),
                                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
      const bool overwrite = (ow == QMessageBox::Yes);
      std::string root_u8 = registry_root_.toStdString();
      std::string name_u8 = er.name.toStdString();
      std::string ver_u8 = er.version.toStdString();
      std::string cy_u8 = cypha_path_.toStdString();
      std::string card_u8 = card.toStdString();
      std::string pre_u8;
      const char* pre_ptr = nullptr;
      if (!pre_path_.isEmpty() && QFile::exists(pre_path_)) {
        pre_u8 = pre_path_.toStdString();
        pre_ptr = pre_u8.c_str();
      }
      std::string err;
      if (!cypha::registry_register_bundle(root_u8.c_str(), name_u8.c_str(), ver_u8.c_str(), cy_u8.c_str(),
                                           card_u8.c_str(), pre_ptr, overwrite, &err)) {
        QMessageBox::warning(this, QStringLiteral("Register"), QString::fromStdString(err));
        return;
      }
      QMessageBox::information(this, QStringLiteral("Register"),
                               QStringLiteral("Registered %1 / %2").arg(er.name, er.version));
      reg_refs_ = cypha::registry_scan(registry_root_.toUtf8().constData());
      reg_combo_->clear();
      for (const cypha::RegistryModelRef& r : reg_refs_) {
        reg_combo_->addItem(QStringLiteral("%1 / %2").arg(QString::fromStdString(r.name),
                                                            QString::fromStdString(r.version)));
      }
      refresh_registry_table_scan();
    });

    connect(rest_browse_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("cypha_rest"), QString(),
                                                        QStringLiteral("All (*)"));
      if (!path.isEmpty()) {
        rest_bin_edit_->setText(path);
      }
    });

    connect(rest_start_btn_, &QPushButton::clicked, this, [this]() {
      if (cypha_path_.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Server"), QStringLiteral("Load a .cypha first."));
        return;
      }
      const QString bin = rest_bin_edit_->text().trimmed();
      if (bin.isEmpty() || !QFile::exists(bin)) {
        QMessageBox::warning(this, QStringLiteral("Server"), QStringLiteral("Set a valid cypha_rest binary path."));
        return;
      }
      if (rest_proc_.state() != QProcess::NotRunning) {
        return;
      }
      QStringList args;
      args << QStringLiteral("--listen") << rest_listen_edit_->text().trimmed();
      args << QStringLiteral("--cypha") << cypha_path_;
      if (!ff_path_.isEmpty()) {
        args << QStringLiteral("--f-field-json") << ff_path_;
      }
      if (!pre_path_.isEmpty()) {
        args << QStringLiteral("--pre") << pre_path_;
      }
      if (!registry_root_.isEmpty() && QFileInfo(registry_root_).isDir()) {
        args << QStringLiteral("--registry") << registry_root_;
      }
      append_server_log(QStringLiteral("[process] starting: %1 %2\n")
                            .arg(bin, args.join(QLatin1Char(' '))));
      rest_proc_.start(bin, args);
      if (!rest_proc_.waitForStarted(5000)) {
        QMessageBox::warning(this, QStringLiteral("Server"), QStringLiteral("Failed to start cypha_rest."));
        append_server_log(QStringLiteral("[process] waitForStarted failed\n"));
        return;
      }
      const QString base_guess = listen_to_http_base(rest_listen_edit_->text());
      if (!base_guess.isEmpty()) {
        rest_base_edit_->setText(base_guess);
      }
      rest_status_label_->setText(QStringLiteral("Server: running (PID %1)").arg(rest_proc_.processId()));
      rest_start_btn_->setEnabled(false);
      rest_stop_btn_->setEnabled(true);
    });

    connect(rest_stop_btn_, &QPushButton::clicked, this, [this]() {
      if (rest_proc_.state() != QProcess::NotRunning) {
        append_server_log(QStringLiteral("[process] stop requested\n"));
        rest_proc_.terminate();
        if (!rest_proc_.waitForFinished(4000)) {
          rest_proc_.kill();
        }
      }
    });

    connect(rest_clear_log_btn_, &QPushButton::clicked, this, [this]() {
      if (rest_log_ != nullptr) {
        rest_log_->clear();
      }
    });

    connect(rest_health_btn_, &QPushButton::clicked, this, [this]() {
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/health"), QStringLiteral("Enter REST base URL first."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      const QUrl url(base + QStringLiteral("/health"));
      const HttpJsonResult r = http_get_json(url);
      if (!r.ok) {
        append_server_log(QStringLiteral("[GET /health] %1\n").arg(r.err));
        QMessageBox::warning(this, QStringLiteral("/health"), r.err);
        return;
      }
      const QString pretty = QString::fromUtf8(QJsonDocument(r.obj).toJson(QJsonDocument::Indented));
      append_server_log(QStringLiteral("[GET /health]\n%1\n").arg(pretty));
      const QString st = r.obj.value(QStringLiteral("status")).toString();
      const QString mdl = r.obj.value(QStringLiteral("model")).toString();
      result_label_->setText(QStringLiteral("[REST /health] status=%1 model=%2").arg(st, mdl));
    });

    connect(rest_ready_btn_, &QPushButton::clicked, this, [this]() {
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/ready"), QStringLiteral("Enter REST base URL first."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      const QUrl url(base + QStringLiteral("/ready"));
      const HttpJsonResult r = http_get_json(url);
      if (!r.ok) {
        append_server_log(QStringLiteral("[GET /ready] %1\n").arg(r.err));
        QMessageBox::warning(this, QStringLiteral("/ready"), r.err);
        return;
      }
      const QString pretty = QString::fromUtf8(QJsonDocument(r.obj).toJson(QJsonDocument::Indented));
      append_server_log(QStringLiteral("[GET /ready]\n%1\n").arg(pretty));
      const bool rd = r.obj.value(QStringLiteral("ready")).toBool();
      result_label_->setText(QStringLiteral("[REST /ready] ready=%1").arg(rd ? QStringLiteral("true") : QStringLiteral("false")));
    });

    connect(rest_models_btn_, &QPushButton::clicked, this, [this]() {
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/models"), QStringLiteral("Enter REST base URL first."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      const QUrl url(base + QStringLiteral("/models?summary=1"));
      const HttpJsonResult r = http_get_json(url);
      if (!r.ok) {
        append_server_log(QStringLiteral("[GET /models] %1\n").arg(r.err));
        QMessageBox::warning(this, QStringLiteral("/models"), r.err);
        return;
      }
      const QString pretty = QString::fromUtf8(QJsonDocument(r.obj).toJson(QJsonDocument::Indented));
      append_server_log(QStringLiteral("[GET /models]\n%1\n").arg(pretty));
      const QJsonArray models = r.obj.value(QStringLiteral("models")).toArray();
      refresh_registry_table_rest(models);
      result_label_->setText(QStringLiteral("[REST /models] entries: %1").arg(models.size()));
    });

    bench_proc_.setProcessChannelMode(QProcess::SeparateChannels);
    connect(&bench_proc_, &QProcess::readyReadStandardOutput, this, [this]() {
      append_bench_log(QString::fromUtf8(bench_proc_.readAllStandardOutput()));
    });
    connect(&bench_proc_, &QProcess::readyReadStandardError, this, [this]() {
      append_bench_log(QString::fromUtf8(bench_proc_.readAllStandardError()));
    });
    connect(&bench_proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus st) {
              bench_run_d17_btn_->setEnabled(true);
              append_bench_log(QStringLiteral("\n[bench] exited code=%1 status=%2\n")
                                   .arg(code)
                                   .arg(st == QProcess::NormalExit ? QStringLiteral("normal")
                                                                   : QStringLiteral("crash")));
            });

    connect(lm_browse_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(
          this, QStringLiteral("Open CyphaLM checkpoint"), QString(),
          QStringLiteral("CyphaLM checkpoint (*.json);;All (*)"));
      if (!path.isEmpty()) {
        lm_path_edit_->setText(path);
      }
    });
    connect(lm_load_btn_, &QPushButton::clicked, this, [this]() { lm_do_load(); });
    connect(lm_generate_btn_, &QPushButton::clicked, this, [this]() { lm_do_generate(); });
    connect(lm_use_rest_chk_, &QCheckBox::stateChanged, this, [this](int) { lm_refresh_generate_enabled(); });
    connect(bench_browse_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getOpenFileName(
          this, QStringLiteral("cypha_bench_run binary"), QString(),
#if defined(_WIN32)
          QStringLiteral("Executable (*.exe);;All (*)"));
#else
          QString());
#endif
      if (!path.isEmpty()) {
        bench_bin_edit_->setText(path);
      }
    });
    connect(bench_run_d17_btn_, &QPushButton::clicked, this, [this]() {
      if (bench_proc_.state() != QProcess::NotRunning) {
        QMessageBox::information(this, QStringLiteral("Bench"),
                                 QStringLiteral("A bench run is already in progress."));
        return;
      }
      const QString bin = bench_bin_edit_->text().trimmed();
      if (bin.isEmpty() || !QFileInfo::exists(bin)) {
        QMessageBox::warning(this, QStringLiteral("Bench"),
                             QStringLiteral("Set a valid cypha_bench_run binary path."));
        return;
      }
      bench_log_edit_->clear();
      append_bench_log(QStringLiteral("[bench] starting: %1 --domain 17\n").arg(bin));
      bench_run_d17_btn_->setEnabled(false);
      bench_proc_.setProgram(bin);
      bench_proc_.setArguments({QStringLiteral("--domain"), QStringLiteral("17")});
      bench_proc_.setWorkingDirectory(QFileInfo(bin).absolutePath());
      bench_proc_.start();
      if (!bench_proc_.waitForStarted(5000)) {
        append_bench_log(QStringLiteral("[bench] waitForStarted failed: %1\n").arg(bench_proc_.errorString()));
        bench_run_d17_btn_->setEnabled(true);
      }
    });

    connect(rest_post_load_btn_, &QPushButton::clicked, this, [this]() {
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/load"),
                                 QStringLiteral("Set REST base URL. Server must have been started with --registry."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      const QString name = rest_load_name_edit_->text().trimmed();
      if (name.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/load"), QStringLiteral("Enter registry model name."));
        return;
      }
      QString ver = rest_load_version_edit_->text().trimmed();
      if (ver.isEmpty()) {
        ver = QStringLiteral("latest");
      }
      QJsonObject body;
      body[QStringLiteral("name")] = name;
      body[QStringLiteral("version")] = ver;
      const QUrl url(base + QStringLiteral("/load"));
      const HttpJsonResult r = http_post_json(url, body);
      if (!r.ok) {
        append_server_log(QStringLiteral("[POST /load] %1\n").arg(r.err));
        QMessageBox::warning(this, QStringLiteral("/load"), r.err);
        return;
      }
      if (r.obj.contains(QStringLiteral("detail"))) {
        const QString d = r.obj.value(QStringLiteral("detail")).toString();
        append_server_log(QStringLiteral("[POST /load] detail: %1\n").arg(d));
        QMessageBox::warning(this, QStringLiteral("/load"), d);
        return;
      }
      const QString pretty = QString::fromUtf8(QJsonDocument(r.obj).toJson(QJsonDocument::Indented));
      append_server_log(QStringLiteral("[POST /load] OK\n%1\n").arg(pretty));
      result_label_->setText(QStringLiteral("[REST /load] server reloaded bundle — use GET /health to confirm"));
    });

    connect(predict_btn_, &QPushButton::clicked, this, [this]() {
      if (!model_) {
        return;
      }
      const QString t = features_edit_->text().trimmed();
      if (t.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Predict"), QStringLiteral("Enter comma-separated numbers."));
        return;
      }
      std::vector<double> x;
      QString perr;
      if (!parse_feature_vector(t, feature_input_dim(), x, &perr)) {
        QMessageBox::warning(this, QStringLiteral("Input"), perr);
        return;
      }
      std::string label;
      double conf = 0.0;
      QJsonObject scores;
      double anomaly = 0.0;
      bool is_ood = false;
      QJsonObject expl;
      if (!native_predict_detail(x, &label, &conf, &scores, &anomaly, &is_ood, &expl)) {
        QMessageBox::warning(this, QStringLiteral("Infer failed"), QStringLiteral("Native predict failed."));
        return;
      }
      const QString qlab = QString::fromStdString(label);
      update_confidence_panel(qlab, conf, anomaly, is_ood, scores, expl);
      result_label_->setText(QStringLiteral("[native] label: %1\nconfidence: %2\nanomaly: %3")
                                 .arg(qlab)
                                 .arg(conf, 0, 'g', 8)
                                 .arg(anomaly, 0, 'g', 8));
    });

    // ── Batch predict: load CSV → run native infer on each row ────────────────
    connect(batch_predict_csv_btn_, &QPushButton::clicked, this, [this]() {
      if (!model_) return;
      const QString csv_path = QFileDialog::getOpenFileName(this, QStringLiteral("Batch predict CSV"),
                                                            file_dialog_start_dir(),
                                                            QStringLiteral("CSV (*.csv);;All (*)"));
      if (csv_path.isEmpty()) return;

      cypha::CsvDenseResult mat;
      try {
        mat = cypha::load_csv_dense(qstring_to_fs_path(csv_path), build_csv_spec());
      } catch (const std::exception& e) {
        QMessageBox::warning(this, QStringLiteral("Batch predict"), QString::fromUtf8(e.what()));
        return;
      }
      const int rows = mat.n_rows;
      if (rows == 0) {
        QMessageBox::information(this, QStringLiteral("Batch predict"), QStringLiteral("CSV has no data rows."));
        return;
      }

      QProgressDialog prog(QStringLiteral("Running batch predict…"), QStringLiteral("Cancel"), 0, rows, this);
      prog.setWindowModality(Qt::WindowModal);
      prog.setMinimumDuration(500);

      batch_predict_results_.clear();
      batch_predict_y_true_.clear();
      batch_predict_y_pred_.clear();
      batch_predict_table_->setRowCount(0);
      if (view_confusion_action_ != nullptr) {
        view_confusion_action_->setEnabled(false);
      }

      const int feat_per_row = mat.n_features;
      const bool has_labels = !mat.y_class.empty();
      for (int i = 0; i < rows; i++) {
        if (prog.wasCanceled()) break;
        prog.setValue(i);

        const double* xraw = mat.x_rowmajor.data() + static_cast<std::size_t>(i) * feat_per_row;
        std::vector<double> xvec(xraw, xraw + feat_per_row);

        std::string lbl;
        double conf = 0.0;
        const int rc = best_label_and_conf(*model_, pre_.get(), xvec, &lbl, &conf, use_gh_chk_->isChecked(),
                                           native_gh_chi_, native_gh_psi_);
        if (rc != 0) {
          lbl  = "(error)";
          conf = 0.0;
        }
        const QString pred_lbl = QString::fromStdString(lbl);
        batch_predict_results_.push_back({QString::number(i), pred_lbl, QString::number(conf, 'g', 6)});
        batch_predict_y_pred_.push_back(pred_lbl);
        if (has_labels) {
          batch_predict_y_true_.push_back(QString::fromStdString(mat.y_class[static_cast<std::size_t>(i)]));
        }

        const int trow = batch_predict_table_->rowCount();
        batch_predict_table_->insertRow(trow);
        batch_predict_table_->setItem(trow, 0, new QTableWidgetItem(QString::number(i)));
        batch_predict_table_->setItem(trow, 1, new QTableWidgetItem(pred_lbl));
        batch_predict_table_->setItem(trow, 2, new QTableWidgetItem(QString::number(conf, 'g', 6)));
      }
      prog.setValue(rows);
      batch_predict_export_btn_->setEnabled(!batch_predict_results_.empty());
      if (view_confusion_action_ != nullptr) {
        view_confusion_action_->setEnabled(has_labels && !batch_predict_y_true_.isEmpty() &&
                                           batch_predict_y_true_.size() == batch_predict_y_pred_.size());
      }
      result_label_->setText(QStringLiteral("Batch predict: %1 rows classified%2")
                                 .arg(static_cast<int>(batch_predict_results_.size()))
                                 .arg(has_labels ? QStringLiteral(" (labels available for confusion matrix)")
                                                 : QString()));
    });

    connect(batch_predict_export_btn_, &QPushButton::clicked, this, [this]() {
      const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Export batch predict results"),
                                                        QString(), QStringLiteral("CSV (*.csv);;All (*)"));
      if (path.isEmpty()) return;
      QFile f(path);
      if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("Export"), QStringLiteral("Cannot open file for writing."));
        return;
      }
      QTextStream ts(&f);
      ts << "row,predicted_label,confidence\n";
      for (const auto& r : batch_predict_results_) {
        ts << r[0] << "," << r[1] << "," << r[2] << "\n";
      }
    });

    connect(predict_rest_btn_, &QPushButton::clicked, this, [this]() {
      if (!model_) {
        return;
      }
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("REST"), QStringLiteral("Enter REST base URL."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      const QUrl url(base + QStringLiteral("/predict"));
      const QString t = features_edit_->text().trimmed();
      if (t.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("REST"), QStringLiteral("Enter comma-separated numbers."));
        return;
      }
      std::vector<double> x;
      QString perr;
      if (!parse_feature_vector(t, feature_input_dim(), x, &perr)) {
        QMessageBox::warning(this, QStringLiteral("Input"), perr);
        return;
      }
      QJsonArray arr;
      for (double v : x) {
        arr.append(v);
      }
      QJsonObject body;
      body[QStringLiteral("input")] = arr;
      body[QStringLiteral("use_gh")] = use_gh_chk_->isChecked();
      body[QStringLiteral("return_explanation")] = predict_return_explanation_chk_->isChecked();
      if (predict_self_correct_chk_ != nullptr) {
        body[QStringLiteral("self_correct")] = predict_self_correct_chk_->isChecked();
      }
      const HttpJsonResult r = http_post_json(url, body);
      if (!r.ok) {
        QMessageBox::warning(this, QStringLiteral("REST"), r.err);
        return;
      }
      if (r.obj.contains(QStringLiteral("detail"))) {
        QMessageBox::warning(this, QStringLiteral("REST"),
                             QStringLiteral("Server: %1").arg(r.obj.value(QStringLiteral("detail")).toString()));
        return;
      }
      const QString pretty = QString::fromUtf8(QJsonDocument(r.obj).toJson(QJsonDocument::Indented));
      append_server_log(QStringLiteral("[POST /predict]\n%1\n").arg(pretty));

      update_confidence_from_predict_json(r.obj);
      const QString lab = r.obj.value(QStringLiteral("label")).toString();
      const double conf = r.obj.value(QStringLiteral("confidence")).toDouble();
      const double anomaly = r.obj.value(QStringLiteral("anomaly_score")).toDouble();
      const bool ood = r.obj.value(QStringLiteral("is_ood")).toBool();
      const double lat = r.obj.value(QStringLiteral("latency_ms")).toDouble();
      QString lines = QStringLiteral("[REST /predict]\nlabel: %1\nconfidence: %2\nanomaly_score: %3\nis_ood: %4\n"
                                     "latency_ms: %5\n")
                            .arg(lab)
                            .arg(conf, 0, 'g', 8)
                            .arg(anomaly, 0, 'g', 8)
                            .arg(ood ? QStringLiteral("true") : QStringLiteral("false"))
                            .arg(lat, 0, 'g', 8);
      const QJsonValue rv = r.obj.value(QStringLiteral("regression_val"));
      if (!rv.isNull()) {
        lines += QStringLiteral("regression_val: %1\nuncertainty: %2\n")
                     .arg(rv.toDouble(), 0, 'g', 8)
                     .arg(r.obj.value(QStringLiteral("uncertainty")).toDouble(), 0, 'g', 8);
      }
      result_label_->setText(lines);
    });

    connect(update_rest_btn_, &QPushButton::clicked, this, [this]() {
      QString base = rest_base_edit_->text().trimmed();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("REST"), QStringLiteral("Enter REST base URL."));
        return;
      }
      while (base.endsWith(QLatin1Char('/'))) {
        base.chop(1);
      }
      const QUrl url(base + QStringLiteral("/update"));
      const QString t = features_edit_->text().trimmed();
      if (t.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/update"), QStringLiteral("Enter input features first."));
        return;
      }
      const QString clab = update_label_edit_->text().trimmed();
      if (clab.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("/update"), QStringLiteral("Enter correct_label."));
        return;
      }
      std::vector<double> x;
      QString perr;
      if (!parse_feature_vector(t, feature_input_dim(), x, &perr)) {
        QMessageBox::warning(this, QStringLiteral("Input"), perr);
        return;
      }
      QJsonArray arr;
      for (double v : x) {
        arr.append(v);
      }
      QJsonObject body;
      body[QStringLiteral("input")] = arr;
      body[QStringLiteral("correct_label")] = clab;
      body[QStringLiteral("use_gh")] = use_gh_chk_->isChecked();
      add_replay_u01_to_json(body);
      const HttpJsonResult r = http_post_json(url, body);
      if (!r.ok) {
        QMessageBox::warning(this, QStringLiteral("/update"), r.err);
        return;
      }
      if (r.obj.contains(QStringLiteral("detail"))) {
        QMessageBox::warning(this, QStringLiteral("/update"),
                             r.obj.value(QStringLiteral("detail")).toString());
        return;
      }
      const double loss = r.obj.value(QStringLiteral("loss")).toDouble();
      const int nc = static_cast<int>(r.obj.value(QStringLiteral("n_corrections")).toDouble());
      result_label_->setText(QStringLiteral("[REST /update] loss: %1\nn_corrections: %2")
                                 .arg(loss, 0, 'g', 8)
                                 .arg(nc));
    });
  }

 protected:
  void closeEvent(QCloseEvent* e) override {
    bulk_cancel_flag_.store(true, std::memory_order_relaxed);
    bulk_rest_cancel_flag_.store(true, std::memory_order_relaxed);
    bulk_mke_cancel_flag_.store(true, std::memory_order_relaxed);
    if (bulk_worker_ != nullptr) {
      QMetaObject::invokeMethod(bulk_worker_, "requestCancel", Qt::QueuedConnection);
    }
    if (bulk_rest_worker_ != nullptr) {
      QMetaObject::invokeMethod(bulk_rest_worker_, "requestCancel", Qt::QueuedConnection);
    }
    if (bulk_mke_worker_ != nullptr) {
      QMetaObject::invokeMethod(bulk_mke_worker_, "requestCancel", Qt::QueuedConnection);
    }
    if (bulk_thread_ != nullptr) {
      bulk_thread_->quit();
      bulk_thread_->wait(30000);
    }
    if (bulk_rest_thread_ != nullptr) {
      bulk_rest_thread_->quit();
      bulk_rest_thread_->wait(30000);
    }
    if (bulk_mke_thread_ != nullptr) {
      bulk_mke_thread_->quit();
      bulk_mke_thread_->wait(30000);
    }
    if (rest_proc_.state() != QProcess::NotRunning) {
      rest_proc_.terminate();
      rest_proc_.waitForFinished(3000);
    }
    if (bench_proc_.state() != QProcess::NotRunning) {
      bench_proc_.terminate();
      bench_proc_.waitForFinished(3000);
    }
    QSettings ui_settings(QStringLiteral("Cypha"), QStringLiteral("CyphaQtShell"));
    ui_settings.setValue(QStringLiteral("geometry"), saveGeometry());
    if (main_tabs_ != nullptr) {
      ui_settings.setValue(QStringLiteral("mainTab"), main_tabs_->currentIndex());
    }
    if (rest_listen_edit_ != nullptr) {
      ui_settings.setValue(QStringLiteral("restListen"), rest_listen_edit_->text());
    }
    if (rest_base_edit_ != nullptr) {
      ui_settings.setValue(QStringLiteral("restBaseUrl"), rest_base_edit_->text());
    }
    QMainWindow::closeEvent(e);
  }

 private:
  // ── Dataset panel helpers ─────────────────────────────────────────────────

  /// Read header → populate col_target_combo_ and col_feature_list_.
  /// Also sets up the column picker→text sync connects (once per load).
  void populate_column_picker(const QString& path) {
    const QStringList hdrs = read_csv_header(path);
    if (hdrs.isEmpty()) {
      return;
    }
    csv_col_headers_ = hdrs;

    // Block signals while we repopulate
    col_picker_updating_ = true;
    col_target_combo_->clear();
    col_feature_list_->clear();

    for (const QString& h : hdrs) {
      col_target_combo_->addItem(h);
      auto* item = new QListWidgetItem(h, col_feature_list_);
      item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
      item->setCheckState(Qt::Checked);
    }

    // Default target = last column (index -1 convention)
    if (!hdrs.isEmpty()) {
      col_target_combo_->setCurrentIndex(hdrs.size() - 1);
      // Uncheck last column as feature (it's the target)
      if (col_feature_list_->count() > 0) {
        col_feature_list_->item(hdrs.size() - 1)->setCheckState(Qt::Unchecked);
      }
    }
    col_picker_updating_ = false;

    sync_col_picker_to_text_fields();

    // Preview
    refresh_csv_preview();
    csv_stats_label_->setText(QStringLiteral("header loaded — run Inspect CSV for full stats"));
  }

  /// Sync the combo/list selections into csv_target_name_edit_ + csv_feature_names_edit_.
  void sync_col_picker_to_text_fields() {
    if (col_picker_updating_) {
      return;
    }
    const int tgt_idx = col_target_combo_->currentIndex();
    if (tgt_idx >= 0 && tgt_idx < csv_col_headers_.size()) {
      csv_target_name_edit_->setText(csv_col_headers_[tgt_idx]);
      csv_target_index_edit_->setText(QStringLiteral("-1"));
    }
    QStringList feat_names;
    for (int i = 0; i < col_feature_list_->count(); ++i) {
      const QListWidgetItem* item = col_feature_list_->item(i);
      if (item->checkState() == Qt::Checked) {
        feat_names << item->text();
      }
    }
    csv_feature_names_edit_->setText(feat_names.join(QLatin1Char(',')));
  }

  /// Refresh the raw CSV preview table (first 8 data rows, all raw columns).
  void refresh_csv_preview() {
    if (csv_path_.isEmpty() || csv_preview_table_ == nullptr) {
      return;
    }
    const auto rows = read_csv_preview(csv_path_, 8);
    if (rows.empty()) {
      return;
    }
    csv_preview_table_->setUpdatesEnabled(false);
    csv_preview_table_->clearContents();

    // First row is the header
    const QStringList& header = rows[0];
    const int ncols = header.size();
    csv_preview_table_->setColumnCount(ncols);
    csv_preview_table_->setHorizontalHeaderLabels(header);

    const int ndata = static_cast<int>(rows.size()) - 1;
    csv_preview_table_->setRowCount(ndata);
    for (int r = 0; r < ndata; ++r) {
      const QStringList& row = rows[static_cast<std::size_t>(r + 1)];
      for (int c = 0; c < ncols && c < row.size(); ++c) {
        csv_preview_table_->setItem(r, c, new QTableWidgetItem(row[c]));
      }
    }
    csv_preview_table_->resizeColumnsToContents();
    csv_preview_table_->setUpdatesEnabled(true);
  }

  /// Open the "Fit preprocessor" configuration dialog.
  void open_fit_preprocessor_dialog() {
    const int n_rows = last_csv_.n_rows;
    const int n_cols = last_csv_.n_features;
    if (n_rows < 2 || n_cols < 1) {
      QMessageBox::information(this, QStringLiteral("Fit preprocessor"),
                               QStringLiteral("Not enough data (%1 rows × %2 cols).").arg(n_rows).arg(n_cols));
      return;
    }

    auto* dlg = new QDialog(this);
    dlg->setWindowTitle(QStringLiteral("Fit preprocessor"));
    auto* vbox = new QVBoxLayout(dlg);
    auto* form = new QFormLayout();
    vbox->addLayout(form);

    auto* scale_chk = new QCheckBox(QStringLiteral("Scale (z-score normalise each feature)"), dlg);
    scale_chk->setChecked(true);
    form->addRow(scale_chk);

    auto* pca_spin = new QSpinBox(dlg);
    pca_spin->setRange(0, n_cols);
    pca_spin->setValue(0);
    pca_spin->setToolTip(QStringLiteral("0 = no PCA. Max = %1 (n_features).").arg(n_cols));
    form->addRow(QStringLiteral("PCA dim (0 = none):"), pca_spin);

    auto* rff_spin = new QSpinBox(dlg);
    rff_spin->setRange(0, 4096);
    rff_spin->setValue(0);
    rff_spin->setToolTip(QStringLiteral("0 = no RFF. Typical: 64–256."));
    form->addRow(QStringLiteral("RFF dim (0 = none):"), rff_spin);

    auto* gamma_spin = new QDoubleSpinBox(dlg);
    gamma_spin->setRange(0.001, 1000.0);
    gamma_spin->setDecimals(4);
    gamma_spin->setValue(1.0);
    gamma_spin->setSingleStep(0.1);
    form->addRow(QStringLiteral("RFF gamma:"), gamma_spin);

    auto* auto_gamma_combo = new QComboBox(dlg);
    auto_gamma_combo->addItem(QStringLiteral("Manual gamma"), 0);
    auto_gamma_combo->addItem(QStringLiteral("Auto: median pairwise"), 1);
    auto_gamma_combo->addItem(QStringLiteral("Auto: cross-validation"), 2);
    auto_gamma_combo->setToolTip(
        QStringLiteral("CV uses regression targets when loaded; otherwise 5-fold reconstruction MSE."));
    form->addRow(QStringLiteral("RFF gamma mode:"), auto_gamma_combo);

    auto update_gamma_enabled = [&]() {
      gamma_spin->setEnabled(auto_gamma_combo->currentData().toInt() == 0);
    };
    update_gamma_enabled();
    connect(auto_gamma_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), dlg,
            [update_gamma_enabled](int) { update_gamma_enabled(); });

    auto* info_lbl = new QLabel(dlg);
    auto update_info = [&]() {
      const int pca_d = pca_spin->value();
      const int rff_d = rff_spin->value();
      int out_d = (pca_d > 0 && pca_d < n_cols) ? pca_d : n_cols;
      if (rff_d > 0) {
        out_d = rff_d;
      }
      info_lbl->setText(QStringLiteral("input_dim = %1   →   output_dim = %2")
                            .arg(n_cols)
                            .arg(out_d));
    };
    update_info();
    connect(pca_spin, QOverload<int>::of(&QSpinBox::valueChanged), dlg, [update_info](int) { update_info(); });
    connect(rff_spin, QOverload<int>::of(&QSpinBox::valueChanged), dlg, [update_info](int) { update_info(); });
    form->addRow(info_lbl);

    auto* note_lbl = new QLabel(
        QStringLiteral("<i>Fits scale + PCA (when dim > 0) + optional RFF from the %1 rows × %2 cols feature matrix.</i>")
            .arg(n_rows)
            .arg(n_cols),
        dlg);
    note_lbl->setWordWrap(true);
    vbox->addWidget(note_lbl);

    auto* btns = new QDialogButtonBox(dlg);
    auto* fit_use_btn  = btns->addButton(QStringLiteral("Fit && use"),  QDialogButtonBox::AcceptRole);
    auto* fit_save_btn = btns->addButton(QStringLiteral("Fit && save…"), QDialogButtonBox::ActionRole);
    btns->addButton(QDialogButtonBox::Cancel);
    vbox->addWidget(btns);

    connect(btns, &QDialogButtonBox::rejected, dlg, &QDialog::reject);

    auto do_fit = [&](bool save) {
      cypha::PreprocessorState ps;
      ps.scale   = scale_chk->isChecked();
      const int pca_d = pca_spin->value();
      ps.pca_dim = (pca_d > 0 && pca_d < n_cols) ? pca_d : -1;
      const int rff_d = rff_spin->value();
      ps.rff_dim = rff_d > 0 ? rff_d : -1;
      ps.rff_gamma = gamma_spin->value();
      const int gamma_mode = auto_gamma_combo->currentData().toInt();
      ps.auto_rff_gamma = (gamma_mode == 1);
      ps.auto_rff_gamma_cv = (gamma_mode == 2);
      ps.seed    = 42;
      const std::vector<double>* y_ptr = nullptr;
      if (ps.auto_rff_gamma_cv && !last_csv_.y_regression.empty() &&
          static_cast<int>(last_csv_.y_regression.size()) == n_rows) {
        y_ptr = &last_csv_.y_regression;
      }
      try {
        ps.fit_from_design_matrix(last_csv_.x_rowmajor, n_rows, n_cols, y_ptr, 1);
      } catch (const std::exception& ex) {
        QMessageBox::critical(dlg, QStringLiteral("Fit failed"), QString::fromUtf8(ex.what()));
        return false;
      }

      if (save) {
        const QString save_path = QFileDialog::getSaveFileName(
            dlg, QStringLiteral("Save preprocessor.json"), QString(),
            QStringLiteral("JSON (*.json);;All (*)"));
        if (save_path.isEmpty()) {
          return false;
        }
        // Serialise to JSON (mirror Python PREPROCESSOR_CONTRACT schema).
        QJsonObject j;
        j[QStringLiteral("scale")]   = ps.scale;
        j[QStringLiteral("pca_dim")] = ps.pca_dim;
        j[QStringLiteral("rff_dim")] = ps.rff_dim;
        j[QStringLiteral("rff_gamma")] = ps.rff_gamma;
        j[QStringLiteral("auto_rff_gamma")] = ps.auto_rff_gamma;
        j[QStringLiteral("auto_rff_gamma_cv")] = ps.auto_rff_gamma_cv;
        j[QStringLiteral("seed")]    = ps.seed;
        j[QStringLiteral("fitted")]  = ps.fitted;
        j[QStringLiteral("input_dim")]  = ps.input_dim;
        j[QStringLiteral("output_dim")] = ps.output_dim;
        auto vec_to_arr = [](const std::vector<double>& v) {
          QJsonArray a;
          for (double x : v) { a.append(x); }
          return a;
        };
        j[QStringLiteral("mean")]   = vec_to_arr(ps.mean);
        j[QStringLiteral("stddev")] = vec_to_arr(ps.stddev);
        QJsonArray pca_comps;
        for (const auto& row : ps.pca_components) {
          pca_comps.append(vec_to_arr(row));
        }
        j[QStringLiteral("pca_components")] = pca_comps;
        j[QStringLiteral("pca_mean")] = vec_to_arr(ps.pca_mean);
        // RFF weights empty (not fitted here)
        j[QStringLiteral("rff_w")] = QJsonArray{};
        j[QStringLiteral("rff_b")] = QJsonArray{};
        QFile f(save_path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
          QMessageBox::critical(dlg, QStringLiteral("Save failed"),
                                QStringLiteral("Cannot write to %1").arg(save_path));
          return false;
        }
        f.write(QJsonDocument(j).toJson(QJsonDocument::Indented));
        f.close();
        pre_path_ = save_path;
        pre_label_->setText(QFileInfo(save_path).fileName() + QStringLiteral(" (fitted)"));
      } else {
        pre_path_.clear();
        pre_label_->setText(QStringLiteral("(fitted in-memory)"));
      }

      pre_.reset(new cypha::PreprocessorState(std::move(ps)));
      update_features_hint();
      return true;
    };

    connect(fit_use_btn,  &QPushButton::clicked, dlg, [&]() { if (do_fit(false)) dlg->accept(); });
    connect(fit_save_btn, &QPushButton::clicked, dlg, [&]() { do_fit(true); });

    dlg->exec();
  }

  QString file_dialog_start_dir() const {
    const QString d = studio_prefs_.dataset_dialog_start_dir.trimmed();
    if (!d.isEmpty() && QFileInfo(d).isDir()) {
      return d;
    }
    return QString();
  }

  void apply_studio_preferences() {
    native_gh_chi_ = studio_prefs_.inference_chi;
    native_gh_psi_ = studio_prefs_.inference_psi;
    apply_cypha_theme(studio_prefs_.dark_theme);
    if (use_gh_chk_ != nullptr) {
      use_gh_chk_->setChecked(studio_prefs_.inference_use_gh);
    }
    registry_root_ = studio_prefs_.effective_registry_root();
    if (reg_root_label_ != nullptr) {
      reg_root_label_->setText(registry_root_);
    }
    if (studio_prefs_.ui_font_pt > 0) {
      QFont f = QApplication::font();
      f.setPointSize(studio_prefs_.ui_font_pt);
      QApplication::setFont(f);
    }
  }

  cypha::CsvDenseSpec build_csv_spec() const {
    cypha::CsvDenseSpec s;
    s.has_header = true;
    s.delimiter = ',';
    const QString tn = csv_target_name_edit_->text().trimmed();
    if (!tn.isEmpty()) {
      s.target_col_name = tn.toStdString();
    } else {
      bool ok = false;
      const int idx = csv_target_index_edit_->text().trimmed().toInt(&ok);
      s.target_col_index = ok ? idx : -1;
    }
    const QString fn = csv_feature_names_edit_->text().trimmed();
    if (!fn.isEmpty()) {
      for (const QString& part : fn.split(QLatin1Char(','), Qt::SkipEmptyParts)) {
        const QString t = part.trimmed();
        if (!t.isEmpty()) {
          s.feature_col_names.push_back(t.toStdString());
        }
      }
    }
    s.regression = csv_regression_chk_->isChecked();
    return s;
  }

  int effective_csv_chunk_rows() const {
    if (studio_prefs_.csv_chunk_rows_override > 0) {
      return studio_prefs_.csv_chunk_rows_override;
    }
    if (const char* env = std::getenv("CYPHA_CSV_CHUNK_ROWS")) {
      bool ok = false;
      const int v = QString::fromLocal8Bit(env).trimmed().toInt(&ok);
      if (ok && v > 0) {
        return v;
      }
    }
    return 4096;
  }

  void add_replay_u01_to_json(QJsonObject& body) const {
    if (replay_u01_cache_.empty()) {
      return;
    }
    QJsonArray a;
    for (double u : replay_u01_cache_) {
      a.append(u);
    }
    body[QStringLiteral("replay_u01")] = a;
  }

  QString default_mke_correct_label() const {
    if (model_ != nullptr && !model_->labels.empty()) {
      return QString::fromStdString(model_->labels[0]);
    }
    return QStringLiteral("class");
  }

  void snapshot_gh_native() {
    native_gh_inv_v_.clear();
    native_gh_R_base_ = 1.0;
    native_gh_chi_ = 1.0;
    native_gh_psi_ = 1.0;
    if (model_ == nullptr) {
      return;
    }
    const int d = model_->d_latent;
    native_gh_inv_v_ = model_->inv_v;
    double mean_inv = 0.0;
    for (int j = 0; j < d; ++j) {
      mean_inv += native_gh_inv_v_[static_cast<std::size_t>(j)];
    }
    mean_inv /= static_cast<double>(std::max(d, 1));
    native_gh_R_base_ = 1.0 / (mean_inv + 1e-8);
  }

  void reinit_native_train_state() {
    native_mem_.reset();
    native_replay_.reset();
    native_kernel_mem_.reset();
    native_use_kernel_llr_ = false;
    native_kernel_blend_ = 0.5;
    native_enc_updates_ = 0;
    native_total_steps_ = 0;
    native_llr_ema_ = 0.0;
    native_train_ok_ = false;
    // Reset training progress panel
    train_prog_win_correct_ = 0;
    train_prog_win_total_   = 0;
    train_prog_ema_loss_    = 0.0;
    if (train_prog_class_table_ != nullptr) train_prog_class_table_->setRowCount(0);
    if (train_prog_acc_bar_ != nullptr) train_prog_acc_bar_->clear();
    if (train_prog_label_ != nullptr)
      train_prog_label_->setText(QStringLiteral("Steps: 0  |  Acc(win): —  |  EMA loss: —  |  Classes: 0"));
    if (native_train_one_btn_ != nullptr) {
      native_train_one_btn_->setEnabled(false);
    }
    if (csv_bulk_native_btn_ != nullptr) {
      csv_bulk_native_btn_->setEnabled(false);
    }
    if (save_native_btn_ != nullptr) {
      save_native_btn_->setEnabled(false);
    }
    if (model_ == nullptr || cypha_path_.isEmpty()) {
      return;
    }
    try {
      cypha::CNode root = cypha::load_cypha_file(cypha_path_.toUtf8().constData());
      const cypha::CNode& enc = cypha::map_get_required(root, "enc_W");
      const int d = static_cast<int>(enc.shape[0]);
      const cypha::CNode& fh = cypha::map_get_required(root, "field_h");
      const int fd = static_cast<int>(fh.shape[0]);
      const double* ff_ptr = nullptr;
      if (embedded_world_f_field_ok(root, d, fd)) {
        const cypha::CNode& world = cypha::map_get_required(root, "world");
        const cypha::CNode* wff = cypha::map_get(world, "F_field");
        ff_ptr = wff->tensor.data();
      } else if (!f_field_flat_.empty()) {
        ff_ptr = f_field_flat_.data();
      } else {
        return;
      }
      native_mem_.reset(new cypha::CyphaDifMemoryState(cypha::CyphaDifMemoryState::from_cypha_root(root, ff_ptr, fd)));
      load_native_kernel_from_cypha_root(root, d);
      load_native_hparams_from_widgets_silent();
      native_replay_.reset(new cypha::ReplayBuffer(native_tsp_.replay_cap));
      native_replay_cap_applied_ = native_tsp_.replay_cap;
      snapshot_gh_native();
      native_train_ok_ = true;
      if (native_train_one_btn_ != nullptr) native_train_one_btn_->setEnabled(true);
      if (csv_bulk_native_btn_ != nullptr) csv_bulk_native_btn_->setEnabled(true);
      if (save_native_btn_ != nullptr) save_native_btn_->setEnabled(true);
      if (mke_bulk_btn_ != nullptr) mke_bulk_btn_->setEnabled(true);
    } catch (const std::exception&) {
      native_mem_.reset();
      native_replay_.reset();
      if (save_native_btn_ != nullptr) save_native_btn_->setEnabled(false);
      if (mke_bulk_btn_ != nullptr) mke_bulk_btn_->setEnabled(false);
    }
  }

  bool run_native_train_on_latent(const std::vector<double>& x_latent, const std::string& y_label, double* loss_out,
                                   cypha::MemoryTrainMeta* meta_out = nullptr) {
    if (!native_train_ok_ || model_ == nullptr || native_mem_ == nullptr || native_replay_ == nullptr) {
      return false;
    }
    if (static_cast<int>(x_latent.size()) != model_->d_latent) {
      return false;
    }
    cypha::TrainStepExtras extras{};
    extras.total_steps = &native_total_steps_;
    extras.ood_sigma = &native_ood_sigma_;
    extras.llr_ema = &native_llr_ema_;
    std::vector<double> ru = replay_u01_cache_;
    std::size_t ru_pos = 0;
    if (!ru.empty()) {
      extras.replay_u01 = ru.data();
      extras.replay_u01_len = ru.size();
      extras.replay_u01_pos = &ru_pos;
    }
    fill_kernel_train_extras(extras);
    cypha::MemoryTrainMeta meta_local{};
    cypha::MemoryTrainMeta* meta = (meta_out != nullptr) ? meta_out : &meta_local;
    double loss = 0.0;
    if (use_gh_chk_->isChecked() && static_cast<int>(native_gh_inv_v_.size()) == model_->d_latent) {
      const cypha::GhTrainStepResult gh = cypha::dif_gh_train_step_vector(
          *model_, *native_mem_, *native_replay_, x_latent.data(), model_->d_latent, y_label, native_gh_inv_v_,
          native_gh_R_base_, native_gh_chi_, native_gh_psi_, kGhNigAdaptAlphaShell, native_world_lr_, native_delta_lr_,
          native_ood_sigma_, native_tsp_, native_rng_, native_enc_updates_, meta, &extras);
      loss = gh.loss;
      native_gh_chi_ = gh.chi_new;
      native_gh_psi_ = gh.psi_new;
    } else {
      loss = cypha::dif_train_step_vector(*model_, *native_mem_, *native_replay_, x_latent.data(), model_->d_latent,
                                          y_label, native_world_lr_, native_delta_lr_, native_world_lr_,
                                          native_delta_lr_, native_ood_sigma_, native_tsp_, native_rng_,
                                          native_enc_updates_, meta, &extras);
    }
    if (meta->correct) {
      model_->total_correct += 1;
    }
    // Rolling accuracy window (last 200 steps)
    if (train_prog_win_total_ < 200) {
      ++train_prog_win_total_;
    }
    // Slide: replace oldest — approximated by full-window decay when at cap
    train_prog_win_correct_ = static_cast<int>(
        train_prog_win_correct_ * (train_prog_win_total_ == 200 ? 199.0/200.0 : 1.0)
        + (meta->correct ? 1 : 0));
    // EMA loss
    train_prog_ema_loss_ = 0.97 * train_prog_ema_loss_ + 0.03 * loss;

    if (loss_out != nullptr) {
      *loss_out = loss;
    }
    return true;
  }

  /// Refresh the training progress label + optionally the class distribution table.
  /// ``update_class_table`` is expensive (O(K)); only call it after bulk or single step completes.
  // ── Bulk training thread helpers ─────────────────────────────────────────
  void set_bulk_training_ui(bool training) {
    if (csv_bulk_native_btn_ != nullptr) csv_bulk_native_btn_->setEnabled(!training);
    if (bulk_native_cancel_btn_ != nullptr) bulk_native_cancel_btn_->setEnabled(training);
    if (csv_bulk_train_btn_  != nullptr) csv_bulk_train_btn_->setEnabled(!training);
    if (native_train_one_btn_ != nullptr) native_train_one_btn_->setEnabled(!training);
    if (save_native_btn_     != nullptr) save_native_btn_->setEnabled(!training);
    if (load_btn_            != nullptr) load_btn_->setEnabled(!training);
    if (mke_bulk_btn_        != nullptr) mke_bulk_btn_->setEnabled(!training && native_train_ok_ && last_csv_ok_);
    if (sort_by_uncertainty_chk_ != nullptr) sort_by_uncertainty_chk_->setEnabled(!training);
    if (curriculum_chk_ != nullptr) curriculum_chk_->setEnabled(!training);
  }

  void on_bulk_progress(int step, int total) {
    result_label_->setText(QStringLiteral("Training %1 / %2…").arg(step).arg(total));
  }

  void on_bulk_loss(int step, double loss) {
    Q_UNUSED(step);
    bulk_accum_losses_.append(loss);
    train_prog_ema_loss_ = 0.97 * train_prog_ema_loss_ + 0.03 * loss;

    const int n = bulk_accum_losses_.size();
    if (n % kBulkChartRefreshEvery == 0 || n == bulk_train_n_) {
      apply_losses_to_chart(LossPlotSource::NativeBulk, QVector<double>(bulk_accum_losses_));
    }
  }

  void on_bulk_step_logged(int step, double loss, bool correct) {
    Q_UNUSED(step);
    Q_UNUSED(loss);
    if (loss_chart_ == nullptr) {
      return;
    }
    const int n = bulk_accum_losses_.size();
    if (n % kBulkChartRefreshEvery == 0 || n == bulk_train_n_) {
      loss_chart_->append_native_step(step, loss, correct);
    }
  }

  void on_bulk_val_acc(int step, double acc_percent) {
    native_total_steps_ = step;
    if (step % kBulkProgressRefreshEvery != 0 && step != bulk_train_n_) {
      return;
    }
    if (train_prog_label_ == nullptr) {
      return;
    }
    const int K = (model_ != nullptr) ? static_cast<int>(model_->labels.size()) : 0;
    const QString loss_str =
        (step > 0) ? QString::number(train_prog_ema_loss_, 'f', 4) : QStringLiteral("—");
    train_prog_label_->setText(
        QStringLiteral("Steps: %1  |  Acc(win200): %2%  |  EMA loss: %3  |  Classes: %4")
            .arg(step)
            .arg(QString::number(acc_percent, 'f', 1))
            .arg(loss_str)
            .arg(K));
  }

  void on_bulk_error(const QString& msg) {
    QMessageBox::warning(this, QStringLiteral("Bulk native"), msg);
  }

  void on_bulk_rest_finished(bool ok, QVector<double> losses, bool cancelled, QString error) {
    set_bulk_training_ui(false);
    if (!error.isEmpty()) {
      QMessageBox::warning(this, QStringLiteral("Bulk /update"), error);
    }
    if (!losses.isEmpty()) {
      apply_losses_to_chart(LossPlotSource::RestBulk, std::move(losses));
    } else if (!bulk_rest_accum_losses_.isEmpty()) {
      apply_losses_to_chart(LossPlotSource::RestBulk, QVector<double>(bulk_rest_accum_losses_));
    }
    bulk_rest_accum_losses_.clear();

    if (!last_loss_plot_rest_.isEmpty()) {
      double sum = 0.0;
      for (double v : last_loss_plot_rest_) {
        sum += v;
      }
      result_label_->setText(QStringLiteral("[bulk /update] steps=%1 mean_loss=%2 last_loss=%3")
                                 .arg(last_loss_plot_rest_.size())
                                 .arg(sum / static_cast<double>(last_loss_plot_rest_.size()), 0, 'g', 8)
                                 .arg(last_loss_plot_rest_.last(), 0, 'g', 8));
      QString extra = dataset_info_->toPlainText();
      if (!extra.isEmpty() && !extra.endsWith(QLatin1Char('\n'))) {
        extra += QLatin1Char('\n');
      }
      extra += QStringLiteral("bulk /update: %1 steps, mean_loss=%2\n")
                   .arg(last_loss_plot_rest_.size())
                   .arg(sum / static_cast<double>(last_loss_plot_rest_.size()), 0, 'g', 8);
      dataset_info_->setPlainText(extra);
    } else if (cancelled) {
      result_label_->setText(QStringLiteral("[bulk /update] canceled"));
    } else if (!ok) {
      result_label_->setText(QStringLiteral("[bulk /update] failed"));
    }
  }

  void on_bulk_worker_finished(bool /*success*/, BulkNativeTrainResult result, QVector<BulkLogEntry> log) {
    native_total_steps_     = result.total_steps;
    native_llr_ema_         = result.llr_ema;
    train_prog_ema_loss_    = result.ema_loss;
    train_prog_win_total_   = result.win_total;
    train_prog_win_correct_ = result.win_correct;
    native_gh_chi_          = result.gh_chi;
    native_gh_psi_          = result.gh_psi;
    native_enc_updates_     = result.enc_updates;
    bulk_train_data_        = std::move(result.val_holdout);

    bulk_accum_log_ = std::move(log);
    on_bulk_finish(result.cancelled);
  }

  void on_bulk_mke_finished(bool ok, int n_ok, double mse, double rmse, bool cancelled, QString error) {
    set_bulk_training_ui(false);
    if (!error.isEmpty()) {
      if (mke_result_label_ != nullptr) {
        mke_result_label_->setText(error);
      }
      QMessageBox::warning(this, QStringLiteral("MKE bulk"), error);
      return;
    }
    if (cancelled) {
      if (mke_result_label_ != nullptr) {
        mke_result_label_->setText(QStringLiteral("MKE bulk cancelled (%1 steps done).").arg(n_ok));
      }
      return;
    }
    if (ok && n_ok > 0 && mke_result_label_ != nullptr) {
      mke_result_label_->setText(QStringLiteral("MKE bulk done: %1 steps  MSE=%2  RMSE=%3")
                                     .arg(n_ok)
                                     .arg(mse, 0, 'g', 6)
                                     .arg(rmse, 0, 'g', 6));
      refresh_train_progress(true);
    }
  }

  void on_bulk_finish(bool cancelled) {

    // Final chart update + metrics panel catch-up
    if (!bulk_accum_losses_.isEmpty()) {
      apply_losses_to_chart(LossPlotSource::NativeBulk, std::move(bulk_accum_losses_));
    }
    bulk_accum_losses_.clear();

    if (loss_chart_ != nullptr && !bulk_accum_log_.isEmpty()) {
      loss_chart_->reset_native_roll_window();
      for (const auto& e : bulk_accum_log_) {
        loss_chart_->append_native_step(e.step_n, e.loss, e.correct);
      }
      loss_chart_->refresh_acc_chart();
    }

    // Write training log table (all at once, updates disabled for speed)
    if (train_log_table_ != nullptr && !bulk_accum_log_.isEmpty()) {
      train_log_table_->setUpdatesEnabled(false);
      for (const auto& e : bulk_accum_log_)
        append_train_log_entry(e.step_n, e.label, e.loss, e.correct);
      train_log_table_->setUpdatesEnabled(true);
      train_log_table_->scrollToBottom();
    }
    bulk_accum_log_.clear();

    refresh_train_progress(true);

    // Val accuracy on held-out rows
    QString val_suffix;
    if (bulk_val_n_ > 0 && model_ != nullptr && bulk_train_data_.n_rows > 0) {
      int val_correct = 0;
      for (int i = 0; i < bulk_train_data_.n_rows; ++i) {
        std::vector<double> xr(static_cast<std::size_t>(bulk_train_data_.n_features));
        const std::size_t rb =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(bulk_train_data_.n_features);
        for (int j = 0; j < bulk_train_data_.n_features; ++j)
          xr[static_cast<std::size_t>(j)] = bulk_train_data_.x_rowmajor[rb + static_cast<std::size_t>(j)];
        std::string pred; double conf = 0.0;
        if (best_label_and_conf(*model_, pre_.get(), xr, &pred, &conf, use_gh_chk_->isChecked(),
                                native_gh_chi_, native_gh_psi_) == 0 &&
            pred == bulk_train_data_.y_class[static_cast<std::size_t>(i)])
          ++val_correct;
      }
      const int val_n = bulk_train_data_.n_rows;
      const double acc = static_cast<double>(val_correct) / static_cast<double>(val_n);
      val_suffix = QStringLiteral("  val_acc=%1/%2 (%3%)")
          .arg(val_correct).arg(val_n)
          .arg(static_cast<int>(std::round(acc * 100.0)));
      if (csv_stats_label_ != nullptr) {
        const QString base = csv_stats_label_->text().split(QStringLiteral("  val_acc")).first();
        csv_stats_label_->setText(base + val_suffix);
      }
    }

    const QString status = cancelled
        ? QStringLiteral("Bulk native cancelled.  steps=%1").arg(native_total_steps_)
        : QStringLiteral("Bulk native done.  steps=%1  ema_loss=%2")
              .arg(native_total_steps_)
              .arg(train_prog_ema_loss_, 0, 'f', 4);
    result_label_->setText(status + val_suffix);

    set_bulk_training_ui(false);
  }

  void refresh_train_progress(bool update_class_table = false) {
    if (train_prog_label_ == nullptr) return;

    const int K = (model_ != nullptr) ? static_cast<int>(model_->labels.size()) : 0;
    const QString acc_str = (train_prog_win_total_ > 0)
        ? QString::number(100.0 * train_prog_win_correct_ / train_prog_win_total_, 'f', 1) + QLatin1Char('%')
        : QStringLiteral("—");
    const QString loss_str = (native_total_steps_ > 0)
        ? QString::number(train_prog_ema_loss_, 'f', 4)
        : QStringLiteral("—");
    train_prog_label_->setText(
        QStringLiteral("Steps: %1  |  Acc(win200): %2  |  EMA loss: %3  |  Classes: %4")
            .arg(native_total_steps_).arg(acc_str).arg(loss_str).arg(K));

    if (!update_class_table || train_prog_class_table_ == nullptr) return;
    if (native_mem_ == nullptr) return;

    const int nk = static_cast<int>(native_mem_->labels.size());
    train_prog_class_table_->setRowCount(nk);
    QStringList bar_labels;
    QVector<double> bar_acc;
    for (int k = 0; k < nk; ++k) {
      const QString lbl   = QString::fromStdString(native_mem_->labels[static_cast<std::size_t>(k)]);
      const double  n_obs = (static_cast<int>(native_mem_->n_obs_buf.size()) > k)
                                ? native_mem_->n_obs_buf[static_cast<std::size_t>(k)] : 0.0;
      const std::int64_t n_cor = (static_cast<int>(native_mem_->n_correct.size()) > k)
                                     ? native_mem_->n_correct[static_cast<std::size_t>(k)] : 0;
      const double acc_k = (n_obs > 0) ? 100.0 * static_cast<double>(n_cor) / n_obs : 0.0;
      train_prog_class_table_->setItem(k, 0, new QTableWidgetItem(lbl));
      train_prog_class_table_->setItem(k, 1, new QTableWidgetItem(QString::number(static_cast<int>(n_obs))));
      train_prog_class_table_->setItem(k, 2, new QTableWidgetItem(QString::number(static_cast<int>(n_cor))));
      train_prog_class_table_->setItem(k, 3, new QTableWidgetItem(QString::number(acc_k, 'f', 1)));
      bar_labels << lbl;
      bar_acc    << acc_k;
    }
    if (train_prog_acc_bar_ != nullptr) {
      train_prog_acc_bar_->set_data(bar_labels, bar_acc);
    }
  }

  /// If `train_hparams.json` exists next to the loaded `.cypha`, fill the hparams form (same keys as
  /// `cypha_rest` / `fixtures/train_hparams.json`). Returns whether the file was read successfully.
  bool try_load_train_hparams_adjacent() {
    if (cypha_path_.isEmpty() || hp_world_lr_edit_ == nullptr) {
      return false;
    }
    const QFileInfo fi(cypha_path_);
    const QString hp_path = fi.absoluteDir().filePath(QStringLiteral("train_hparams.json"));
    QFile f(hp_path);
    if (!f.open(QIODevice::ReadOnly)) {
      return false;
    }
    QJsonParseError pe{};
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &pe);
    if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
      return false;
    }
    const QJsonObject j = doc.object();
    if (j.contains(QStringLiteral("world_lr"))) {
      hp_world_lr_edit_->setText(QString::number(j[QStringLiteral("world_lr")].toDouble(), 'g', 17));
    }
    if (j.contains(QStringLiteral("delta_lr"))) {
      hp_delta_lr_edit_->setText(QString::number(j[QStringLiteral("delta_lr")].toDouble(), 'g', 17));
    }
    if (j.contains(QStringLiteral("ood_sigma"))) {
      hp_ood_sigma_edit_->setText(QString::number(j[QStringLiteral("ood_sigma")].toDouble(), 'g', 17));
    }
    if (j.contains(QStringLiteral("enc_lr"))) {
      hp_enc_lr_edit_->setText(QString::number(j[QStringLiteral("enc_lr")].toDouble(), 'g', 17));
    }
    if (j.contains(QStringLiteral("replay_ratio"))) {
      hp_replay_ratio_edit_->setText(QString::number(j[QStringLiteral("replay_ratio")].toDouble(), 'g', 17));
    }
    if (j.contains(QStringLiteral("replay_cap"))) {
      int cap = static_cast<int>(j[QStringLiteral("replay_cap")].toInt());
      if (cap < 8) {
        cap = 8;
      }
      hp_replay_cap_spin_->setValue(cap);
    }
    if (j.contains(QStringLiteral("temp_recalib_every"))) {
      int tr = static_cast<int>(j[QStringLiteral("temp_recalib_every")].toInt());
      if (tr < 0) {
        tr = 0;
      }
      hp_temp_recalib_spin_->setValue(tr);
    }
    if (j.contains(QStringLiteral("align_every"))) {
      int ae = static_cast<int>(j[QStringLiteral("align_every")].toInt());
      if (ae < 0) {
        ae = 0;
      }
      hp_align_every_spin_->setValue(ae);
    }
    return true;
  }

  void load_native_hparams_from_widgets_silent() {
    bool ok = false;
    double v = hp_world_lr_edit_->text().trimmed().toDouble(&ok);
    if (ok) {
      native_world_lr_ = v;
    }
    v = hp_delta_lr_edit_->text().trimmed().toDouble(&ok);
    if (ok) {
      native_delta_lr_ = v;
    }
    v = hp_ood_sigma_edit_->text().trimmed().toDouble(&ok);
    if (ok) {
      native_ood_sigma_ = v;
    }
    v = hp_enc_lr_edit_->text().trimmed().toDouble(&ok);
    if (ok) {
      native_tsp_.enc_lr = v;
    }
    v = hp_replay_ratio_edit_->text().trimmed().toDouble(&ok);
    if (ok) {
      native_tsp_.replay_ratio = v;
    }
    native_tsp_.replay_cap = hp_replay_cap_spin_->value();
    native_tsp_.align_every = hp_align_every_spin_->value();
    native_tsp_.temp_recalib_every = hp_temp_recalib_spin_->value();
  }

  void apply_native_hparams_from_ui() {
    bool ok = false;
    const double wlr = hp_world_lr_edit_->text().trimmed().toDouble(&ok);
    if (!ok) {
      QMessageBox::warning(this, QStringLiteral("Hyperparameters"), QStringLiteral("Invalid world_lr."));
      return;
    }
    const double dlr = hp_delta_lr_edit_->text().trimmed().toDouble(&ok);
    if (!ok) {
      QMessageBox::warning(this, QStringLiteral("Hyperparameters"), QStringLiteral("Invalid delta_lr."));
      return;
    }
    const double ood = hp_ood_sigma_edit_->text().trimmed().toDouble(&ok);
    if (!ok) {
      QMessageBox::warning(this, QStringLiteral("Hyperparameters"), QStringLiteral("Invalid ood_sigma."));
      return;
    }
    const double elr = hp_enc_lr_edit_->text().trimmed().toDouble(&ok);
    if (!ok) {
      QMessageBox::warning(this, QStringLiteral("Hyperparameters"), QStringLiteral("Invalid enc_lr."));
      return;
    }
    const double rr = hp_replay_ratio_edit_->text().trimmed().toDouble(&ok);
    if (!ok) {
      QMessageBox::warning(this, QStringLiteral("Hyperparameters"), QStringLiteral("Invalid replay_ratio."));
      return;
    }
    native_world_lr_ = wlr;
    native_delta_lr_ = dlr;
    native_ood_sigma_ = ood;
    native_tsp_.enc_lr = elr;
    native_tsp_.replay_ratio = rr;
    native_tsp_.replay_cap = hp_replay_cap_spin_->value();
    native_tsp_.align_every = hp_align_every_spin_->value();
    native_tsp_.temp_recalib_every = hp_temp_recalib_spin_->value();
    if (native_train_ok_ && native_replay_ != nullptr &&
        native_tsp_.replay_cap != native_replay_cap_applied_) {
      native_replay_.reset(new cypha::ReplayBuffer(native_tsp_.replay_cap));
      native_replay_cap_applied_ = native_tsp_.replay_cap;
    }
  }

  void set_native_hparams_defaults() {
    hp_world_lr_edit_->setText(QStringLiteral("0.008"));
    hp_delta_lr_edit_->setText(QStringLiteral("0.05"));
    hp_ood_sigma_edit_->setText(QStringLiteral("15"));
    hp_enc_lr_edit_->setText(QStringLiteral("0.002"));
    hp_replay_ratio_edit_->setText(QStringLiteral("0.30"));
    hp_replay_cap_spin_->setValue(10000);
    hp_align_every_spin_->setValue(500);
    hp_temp_recalib_spin_->setValue(0);
    if (use_kernel_llr_chk_ != nullptr) {
      use_kernel_llr_chk_->setChecked(false);
    }
    if (kernel_blend_spin_ != nullptr) {
      kernel_blend_spin_->setValue(0.5);
    }
  }

  void load_native_kernel_from_cypha_root(const cypha::CNode& root, int d) {
    native_kernel_mem_.reset();
    native_use_kernel_llr_ = false;
    native_kernel_blend_ = 0.5;
    if (d <= 0) {
      if (use_kernel_llr_chk_ != nullptr) {
        use_kernel_llr_chk_->setChecked(false);
      }
      if (kernel_blend_spin_ != nullptr) {
        kernel_blend_spin_->setValue(0.5);
      }
      return;
    }
    cypha::KernelMemory km(d, 256, 0);
    bool use = false;
    double blend = 0.5;
    if (cypha::try_load_kernel_from_root(root, km, use, blend)) {
      native_use_kernel_llr_ = use;
      native_kernel_blend_ = blend;
      native_kernel_mem_.reset(new cypha::KernelMemory(std::move(km)));
    }
    if (use_kernel_llr_chk_ != nullptr) {
      use_kernel_llr_chk_->setChecked(native_use_kernel_llr_);
    }
    if (kernel_blend_spin_ != nullptr) {
      kernel_blend_spin_->setValue(native_kernel_blend_);
    }
  }

  void sync_native_kernel_from_ui() {
    if (use_kernel_llr_chk_ == nullptr || model_ == nullptr) {
      native_use_kernel_llr_ = false;
      return;
    }
    native_use_kernel_llr_ = use_kernel_llr_chk_->isChecked();
    native_kernel_blend_ =
        kernel_blend_spin_ != nullptr ? kernel_blend_spin_->value() : native_kernel_blend_;
    if (!native_use_kernel_llr_) {
      return;
    }
    const int d = model_->d_latent;
    if (native_kernel_mem_ == nullptr || native_kernel_mem_->feat_dim() != d) {
      native_kernel_mem_.reset(new cypha::KernelMemory(d, 256, static_cast<std::uint64_t>(native_rng_())));
    }
  }

  void fill_kernel_train_extras(cypha::TrainStepExtras& extras) {
    sync_native_kernel_from_ui();
    if (native_use_kernel_llr_ && native_kernel_mem_ != nullptr) {
      extras.kernel_mem = native_kernel_mem_.get();
      extras.use_kernel_llr = true;
      extras.kernel_blend = native_kernel_blend_;
    }
  }

  void fill_kernel_infer_options(cypha::CyphaInferOptions& opt) const {
    opt.kernel_mem = nullptr;
    opt.use_kernel_llr = false;
    opt.kernel_blend = native_kernel_blend_;
    if (native_use_kernel_llr_ && native_kernel_mem_ != nullptr) {
      opt.kernel_mem = native_kernel_mem_.get();
      opt.use_kernel_llr = true;
      opt.kernel_blend = native_kernel_blend_;
    }
  }

  void refresh_loss_chart() {
    if (loss_chart_ == nullptr) {
      return;
    }
    constexpr double kEmaAlpha = 0.08;
    QVector<double> er;
    QVector<double> en;
    if (loss_ema_chk_ != nullptr && loss_ema_chk_->isChecked()) {
      er = loss_ema_series(last_loss_plot_rest_, kEmaAlpha);
      en = loss_ema_series(last_loss_plot_native_, kEmaAlpha);
    }
    loss_chart_->set_loss_runs(last_loss_plot_rest_, last_loss_plot_native_, er, en);
    loss_chart_->refresh_acc_chart();
    if (loss_y_lock_chk_ != nullptr && loss_y_lock_chk_->isChecked() &&
        loss_y_min_spin_ != nullptr && loss_y_max_spin_ != nullptr) {
      loss_chart_->set_y_range_lock(true, loss_y_min_spin_->value(), loss_y_max_spin_->value());
    } else {
      loss_chart_->set_y_range_lock(false, 0.0, 1.0);
    }
  }

  void append_train_log_entry(int step_n, const QString& label, double loss, bool correct) {
    if (train_log_table_ == nullptr) {
      return;
    }
    const int row = train_log_table_->rowCount();
    train_log_table_->insertRow(row);
    train_log_table_->setItem(row, 0, new QTableWidgetItem(QString::number(step_n)));
    train_log_table_->setItem(row, 1, new QTableWidgetItem(label));
    train_log_table_->setItem(row, 2, new QTableWidgetItem(QString::number(loss, 'g', 6)));
    train_log_table_->setItem(row, 3, new QTableWidgetItem(correct ? QStringLiteral("\u2713") : QStringLiteral("\u2717")));
    train_log_table_->scrollToBottom();
    while (train_log_table_->rowCount() > 2000) {
      train_log_table_->removeRow(0);
    }
  }

  void apply_losses_to_chart(LossPlotSource src, QVector<double> losses) {
    if (src == LossPlotSource::RestBulk) {
      last_loss_plot_rest_ = std::move(losses);
    } else {
      last_loss_plot_native_ = std::move(losses);
    }
    refresh_loss_chart();
  }

  bool save_native_model_to_path(const QString& path, QString* err_out) {
    try {
      cypha::CNode root = cypha::load_cypha_file(cypha_path_.toUtf8().constData());
      cypha::CNode merged = cypha::CyphaDifMemoryState::merge_state_into_root_for_save(root, *native_mem_);
      const std::int64_t ts =
          static_cast<std::int64_t>(model_->saved_total_steps) + static_cast<std::int64_t>(native_total_steps_);
      NativeSessionSnapshotPatch sess{};
      sess.ood_sigma = native_ood_sigma_;
      sess.gh_chi = native_gh_chi_;
      sess.gh_psi = native_gh_psi_;
      sess.gh_r_base = native_gh_R_base_;
      sess.gh_inv_v_clean = native_gh_inv_v_.empty() ? nullptr : &native_gh_inv_v_;
      sess.feat_dim = pre_ != nullptr ? pre_->input_dim : -1;
      patch_infer_training_snapshot(merged, *model_, ts, native_llr_ema_, &sess);
      sync_native_kernel_from_ui();
      if (native_kernel_mem_ != nullptr) {
        cypha::patch_kernel_into_root(merged, *native_kernel_mem_, native_use_kernel_llr_, native_kernel_blend_);
      } else if (model_ != nullptr) {
        cypha::KernelMemory dummy(model_->d_latent, 256, 0);
        cypha::patch_kernel_into_root(merged, dummy, false, native_kernel_blend_);
      }
      cypha::save_cypha_file(path.toUtf8().constData(), merged);
      return true;
    } catch (const std::exception& ex) {
      if (err_out != nullptr) {
        *err_out = QString::fromUtf8(ex.what());
      }
      return false;
    }
  }

  void append_server_log(const QString& prefix, const QString& chunk) {
    if (rest_log_ == nullptr || chunk.isEmpty()) {
      return;
    }
    rest_log_->moveCursor(QTextCursor::End);
    rest_log_->insertPlainText(prefix + chunk);
    if (!chunk.endsWith(QLatin1Char('\n'))) {
      rest_log_->insertPlainText(QStringLiteral("\n"));
    }
    rest_log_->moveCursor(QTextCursor::End);
  }

  void append_server_log(const QString& line) {
    if (rest_log_ == nullptr || line.isEmpty()) {
      return;
    }
    rest_log_->moveCursor(QTextCursor::End);
    rest_log_->insertPlainText(line);
    if (!line.endsWith(QLatin1Char('\n'))) {
      rest_log_->insertPlainText(QStringLiteral("\n"));
    }
    rest_log_->moveCursor(QTextCursor::End);
  }

  void append_bench_log(const QString& chunk) {
    if (bench_log_edit_ == nullptr || chunk.isEmpty()) {
      return;
    }
    bench_log_edit_->moveCursor(QTextCursor::End);
    bench_log_edit_->insertPlainText(chunk);
    bench_log_edit_->moveCursor(QTextCursor::End);
  }

  void lm_refresh_generate_enabled() {
    if (lm_generate_btn_ == nullptr) {
      return;
    }
    const bool rest_mode = lm_use_rest_chk_ != nullptr && lm_use_rest_chk_->isChecked();
    lm_generate_btn_->setEnabled(lm_loaded_ || rest_mode);
  }

  QString rest_base_normalized() const {
    QString base = rest_base_edit_ != nullptr ? rest_base_edit_->text().trimmed() : QString();
    while (base.endsWith(QLatin1Char('/'))) {
      base.chop(1);
    }
    return base;
  }

  void lm_do_load() {
    const QString path = lm_path_edit_ != nullptr ? lm_path_edit_->text().trimmed() : QString();
    if (path.isEmpty()) {
      QMessageBox::information(this, QStringLiteral("CyphaLM"), QStringLiteral("Enter a checkpoint path."));
      return;
    }
    if (!QFileInfo::exists(path)) {
      QMessageBox::warning(this, QStringLiteral("CyphaLM"),
                           QStringLiteral("File not found:\n%1").arg(path));
      return;
    }

    const bool rest_mode = lm_use_rest_chk_ != nullptr && lm_use_rest_chk_->isChecked();
    if (rest_mode) {
      const QString base = rest_base_normalized();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("CyphaLM"),
                                 QStringLiteral("Set REST base URL on the Server tab (or start cypha_rest)."));
        return;
      }
      QJsonObject body;
      body[QStringLiteral("checkpoint_path")] = path;
      const HttpJsonResult r = http_post_json(QUrl(base + QStringLiteral("/lm/load")), body);
      if (!r.ok) {
        QMessageBox::warning(this, QStringLiteral("CyphaLM /lm/load"), r.err);
        return;
      }
      if (r.obj.contains(QStringLiteral("detail"))) {
        QMessageBox::warning(this, QStringLiteral("CyphaLM /lm/load"),
                             r.obj.value(QStringLiteral("detail")).toString());
        return;
      }
      lm_model_.reset();
      lm_loaded_ = true;
      lm_source_path_ = path;
      lm_n_generations_ = 0;
      const QJsonObject summary = r.obj.value(QStringLiteral("summary")).toObject();
      const QString status =
          QStringLiteral("REST loaded — vocab %1, field_dim %2")
              .arg(summary.value(QStringLiteral("vocab_size")).toInt())
              .arg(summary.value(QStringLiteral("field_dim")).toInt());
      if (lm_status_label_ != nullptr) {
        lm_status_label_->setText(status);
      }
      if (lm_output_edit_ != nullptr) {
        lm_output_edit_->setPlainText(QString::fromUtf8(
            QJsonDocument(r.obj).toJson(QJsonDocument::Indented)));
      }
    } else {
      try {
        lm_model_ = std::make_unique<cypha::cyphalm::CyphaLMModel>(
            cypha::cyphalm::load_cyphalm_model(qstring_to_fs_path(path).string()));
        lm_loaded_ = true;
        lm_source_path_ = path;
        lm_n_generations_ = 0;
        const auto& cfg = lm_model_->config();
        if (lm_status_label_ != nullptr) {
          lm_status_label_->setText(QStringLiteral("Native loaded — vocab %1, field_dim %2, experts %3")
                                        .arg(cfg.vocab_size)
                                        .arg(cfg.field_dim)
                                        .arg(cfg.n_experts));
        }
        if (lm_output_edit_ != nullptr) {
          lm_output_edit_->clear();
        }
      } catch (const std::exception& ex) {
        lm_model_.reset();
        lm_loaded_ = false;
        lm_source_path_.clear();
        QMessageBox::warning(this, QStringLiteral("CyphaLM load"),
                             QString::fromUtf8(ex.what()));
        if (lm_status_label_ != nullptr) {
          lm_status_label_->setText(QStringLiteral("(load failed)"));
        }
        lm_refresh_generate_enabled();
        return;
      }
    }
    lm_refresh_generate_enabled();
    result_label_->setText(QStringLiteral("[CyphaLM] loaded %1").arg(path));
  }

  void lm_do_generate() {
    std::vector<int> prompt_ids;
    QString perr;
    if (!parse_token_ids(lm_prompt_edit_ != nullptr ? lm_prompt_edit_->text() : QString(), prompt_ids,
                         &perr)) {
      QMessageBox::warning(this, QStringLiteral("CyphaLM generate"), perr);
      return;
    }
    const int max_tokens = lm_n_tokens_spin_ != nullptr ? lm_n_tokens_spin_->value() : 64;
    const QString strategy =
        lm_strategy_combo_ != nullptr ? lm_strategy_combo_->currentText() : QStringLiteral("top_p");
    const double top_p = lm_top_p_spin_ != nullptr ? lm_top_p_spin_->value() : 0.92;
    const double temperature =
        lm_temperature_spin_ != nullptr ? lm_temperature_spin_->value() : 0.9;

    const bool rest_mode = lm_use_rest_chk_ != nullptr && lm_use_rest_chk_->isChecked();
    if (rest_mode) {
      const QString base = rest_base_normalized();
      if (base.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("CyphaLM"),
                                 QStringLiteral("Set REST base URL on the Server tab."));
        return;
      }
      QJsonArray prompt_arr;
      for (int id : prompt_ids) {
        prompt_arr.append(id);
      }
      QJsonObject body;
      body[QStringLiteral("prompt_ids")] = prompt_arr;
      body[QStringLiteral("max_tokens")] = max_tokens;
      body[QStringLiteral("strategy")] = strategy;
      body[QStringLiteral("top_p")] = top_p;
      body[QStringLiteral("temperature")] = temperature;
      if (lm_epistemic_halt_chk_ != nullptr) {
        body[QStringLiteral("epistemic_halt")] = lm_epistemic_halt_chk_->isChecked();
      }
      const auto t0 = std::chrono::steady_clock::now();
      const HttpJsonResult r = http_post_json(QUrl(base + QStringLiteral("/generate")), body);
      const double latency_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (!r.ok) {
        QMessageBox::warning(this, QStringLiteral("CyphaLM /generate"), r.err);
        return;
      }
      if (r.obj.contains(QStringLiteral("detail"))) {
        QMessageBox::warning(this, QStringLiteral("CyphaLM /generate"),
                             r.obj.value(QStringLiteral("detail")).toString());
        return;
      }
      ++lm_n_generations_;
      if (lm_output_edit_ != nullptr) {
        QString out = QString::fromUtf8(QJsonDocument(r.obj).toJson(QJsonDocument::Indented));
        out += QStringLiteral("\n\nlatency_ms: %1").arg(latency_ms, 0, 'f', 2);
        lm_output_edit_->setPlainText(out);
      }
      result_label_->setText(
          QStringLiteral("[CyphaLM REST] generated %1 tokens in %2 ms")
              .arg(r.obj.value(QStringLiteral("n_tokens")).toInt())
              .arg(latency_ms, 0, 'f', 1));
      return;
    }

    if (!lm_model_) {
      QMessageBox::information(this, QStringLiteral("CyphaLM"),
                               QStringLiteral("Load a checkpoint first."));
      return;
    }

    cypha::cyphalm::DecodeParams params;
    params.strategy = cypha::cyphalm::decode_strategy_from_string(strategy.toStdString());
    params.top_p = top_p;
    params.temperature = temperature;
    params.top_k = 40;
    if (lm_epistemic_halt_chk_ != nullptr) {
      params.epistemic_halt = lm_epistemic_halt_chk_->isChecked();
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
      const cypha::cyphalm::GenerateOutput gen = cypha::cyphalm::generate_decode(
          *lm_model_, prompt_ids, max_tokens, params, &lm_epistemic_threshold_);
      const double latency_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ++lm_n_generations_;

      QStringList id_parts;
      id_parts.reserve(static_cast<int>(gen.generated_ids.size()));
      for (int id : gen.generated_ids) {
        id_parts << QString::number(id);
      }
      QString decoded;
      try {
        std::vector<std::uint32_t> uids;
        uids.reserve(gen.generated_ids.size());
        for (int id : gen.generated_ids) {
          uids.push_back(static_cast<std::uint32_t>(id));
        }
        decoded = QString::fromStdString(lm_model_->decode_tokens(uids));
      } catch (...) {
        decoded = QStringLiteral("(decode unavailable)");
      }

      QString out = QStringLiteral("generated_ids: [%1]\n").arg(id_parts.join(QStringLiteral(", ")));
      out += QStringLiteral("n_tokens: %1\n").arg(static_cast<int>(gen.generated_ids.size()));
      out += QStringLiteral("strategy: %1\n").arg(strategy);
      out += QStringLiteral("halted_on_uncertainty: %1\n")
                 .arg(gen.halted_on_uncertainty ? QStringLiteral("true") : QStringLiteral("false"));
      out += QStringLiteral("halted_on_epistemic: %1\n")
                 .arg(gen.halted_on_epistemic ? QStringLiteral("true") : QStringLiteral("false"));
      if (gen.r_eu_proxy > 0.0) {
        out += QStringLiteral("r_eu_proxy: %1\n").arg(gen.r_eu_proxy, 0, 'f', 4);
      }
      out += QStringLiteral("latency_ms: %1\n\n").arg(latency_ms, 0, 'f', 2);
      out += QStringLiteral("decoded: %1").arg(decoded);
      if (lm_output_edit_ != nullptr) {
        lm_output_edit_->setPlainText(out);
      }
      result_label_->setText(QStringLiteral("[CyphaLM native] %1 tokens, %2 ms")
                                 .arg(static_cast<int>(gen.generated_ids.size()))
                                 .arg(latency_ms, 0, 'f', 1));
    } catch (const std::exception& ex) {
      QMessageBox::warning(this, QStringLiteral("CyphaLM generate"), QString::fromUtf8(ex.what()));
    }
  }

  int feature_input_dim() const {
    if (pre_) {
      return pre_->input_dim;
    }
    return model_ ? model_->d_latent : 0;
  }

  void update_features_hint() {
    if (!model_) {
      features_hint_->setText(pre_ ? QStringLiteral("Preprocessor loaded — load a .cypha next")
                                   : QStringLiteral("Features: load a model first"));
      return;
    }
    if (pre_) {
      features_hint_->setText(QStringLiteral("Features: %1 raw values (preprocessor → latent dim %2)")
                                  .arg(pre_->input_dim)
                                  .arg(model_->d_latent));
    } else {
      features_hint_->setText(
          QStringLiteral("Features: %1 values (latent / encoder input, no preprocessor)").arg(model_->d_latent));
    }
  }

  void reload_preprocessor_only() {
    pre_.reset();
    if (!pre_path_.isEmpty()) {
      try {
        pre_.reset(new cypha::PreprocessorState(
            cypha::PreprocessorState::from_json_file(pre_path_.toUtf8().constData())));
      } catch (const std::exception& ex) {
        QMessageBox::warning(this, QStringLiteral("Preprocessor"),
                             QStringLiteral("Failed to load preprocessor.json:\n%1").arg(QString::fromUtf8(ex.what())));
        pre_path_.clear();
        pre_label_->setText(QStringLiteral("(optional)"));
      }
    }
    update_features_hint();
  }

  void refresh_registry_table_scan() {
    if (reg_models_table_ == nullptr) {
      return;
    }
    reg_models_table_->setRowCount(0);
    for (const cypha::RegistryModelRef& r : reg_refs_) {
      const int row = reg_models_table_->rowCount();
      reg_models_table_->insertRow(row);
      reg_models_table_->setItem(row, 0, new QTableWidgetItem(QString::fromStdString(r.name)));
      reg_models_table_->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(r.version)));
      reg_models_table_->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(r.model_path)));
      reg_models_table_->setItem(row, 3,
                                 new QTableWidgetItem(QString::fromStdString(r.preprocessor_path)));
    }
  }

  void refresh_registry_table_rest(const QJsonArray& models) {
    if (reg_models_table_ == nullptr) {
      return;
    }
    reg_models_table_->setRowCount(0);
    reg_refs_.clear();
    for (const QJsonValue& v : models) {
      if (!v.isObject()) {
        continue;
      }
      const QJsonObject o = v.toObject();
      cypha::RegistryModelRef ref;
      ref.name = o.value(QStringLiteral("name")).toString().toStdString();
      ref.version = o.value(QStringLiteral("version")).toString().toStdString();
      ref.model_path = o.value(QStringLiteral("model_path")).toString().toStdString();
      ref.preprocessor_path = o.value(QStringLiteral("preprocessor_path")).toString().toStdString();
      ref.card_path = o.value(QStringLiteral("card_path")).toString().toStdString();
      reg_refs_.push_back(ref);
      const int row = reg_models_table_->rowCount();
      reg_models_table_->insertRow(row);
      reg_models_table_->setItem(row, 0, new QTableWidgetItem(QString::fromStdString(ref.name)));
      reg_models_table_->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(ref.version)));
      reg_models_table_->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(ref.model_path)));
      reg_models_table_->setItem(row, 3,
                                 new QTableWidgetItem(QString::fromStdString(ref.preprocessor_path)));
    }
    if (reg_combo_ != nullptr) {
      reg_combo_->clear();
      for (const cypha::RegistryModelRef& r : reg_refs_) {
        reg_combo_->addItem(QStringLiteral("%1 / %2").arg(QString::fromStdString(r.name),
                                                            QString::fromStdString(r.version)));
      }
    }
  }

  int selected_registry_index() const {
    if (reg_models_table_ != nullptr && reg_models_table_->currentRow() >= 0) {
      return reg_models_table_->currentRow();
    }
    if (reg_combo_ != nullptr) {
      return reg_combo_->currentIndex();
    }
    return -1;
  }

  void sync_registry_selection_to_rest_load() {
    const int idx = selected_registry_index();
    if (idx < 0 || idx >= static_cast<int>(reg_refs_.size())) {
      return;
    }
    const cypha::RegistryModelRef& r = reg_refs_[static_cast<std::size_t>(idx)];
    if (rest_load_name_edit_ != nullptr) {
      rest_load_name_edit_->setText(QString::fromStdString(r.name));
    }
    if (rest_load_version_edit_ != nullptr) {
      rest_load_version_edit_->setText(QString::fromStdString(r.version));
    }
  }

  void update_confidence_panel(const QString& label, double conf, double anomaly, bool is_ood,
                               const QJsonObject& all_scores, const QJsonObject& explanation) {
    if (conf_pred_label_ != nullptr) {
      conf_pred_label_->setText(QStringLiteral("Prediction: %1  (confidence %2)")
                                    .arg(label)
                                    .arg(conf, 0, 'g', 6));
    }
    if (conf_ood_label_ != nullptr && conf_ood_bar_ != nullptr) {
      const int pct = std::clamp(
          static_cast<int>(anomaly / std::max(studio_prefs_.inference_ood_threshold, 1e-6) * 100.0), 0, 100);
      conf_ood_bar_->setValue(pct);
      conf_ood_label_->setText(QStringLiteral("OOD score: %1  %2")
                                   .arg(anomaly, 0, 'g', 4)
                                   .arg(is_ood ? QStringLiteral("[OOD]") : QStringLiteral("[in-distribution]")));
      const QString bar_style = is_ood
                                    ? QStringLiteral("QProgressBar::chunk { background-color: #e05050; }")
                                    : QStringLiteral("QProgressBar::chunk { background-color: #40a060; }");
      conf_ood_bar_->setStyleSheet(bar_style);
    }
    if (conf_scores_table_ != nullptr) {
      conf_scores_table_->setRowCount(0);
      const QStringList keys = all_scores.keys();
      QStringList sorted = keys;
      std::sort(sorted.begin(), sorted.end(), [&](const QString& a, const QString& b) {
        return all_scores.value(a).toDouble() > all_scores.value(b).toDouble();
      });
      for (const QString& k : sorted) {
        const int row = conf_scores_table_->rowCount();
        conf_scores_table_->insertRow(row);
        conf_scores_table_->setItem(row, 0, new QTableWidgetItem(k));
        conf_scores_table_->setItem(row, 1,
                                    new QTableWidgetItem(QString::number(all_scores.value(k).toDouble(), 'g', 6)));
      }
    }
    if (conf_explanation_edit_ != nullptr) {
      if (explanation.isEmpty()) {
        conf_explanation_edit_->clear();
      } else {
        conf_explanation_edit_->setPlainText(
            QString::fromUtf8(QJsonDocument(explanation).toJson(QJsonDocument::Indented)));
      }
    }
  }

  void update_confidence_from_predict_json(const QJsonObject& obj) {
    const QString lab = obj.value(QStringLiteral("label")).toString();
    const double conf = obj.value(QStringLiteral("confidence")).toDouble();
    const double anomaly = obj.value(QStringLiteral("anomaly_score")).toDouble();
    const bool ood = obj.value(QStringLiteral("is_ood")).toBool();
    QJsonObject scores;
    const QJsonValue sc = obj.value(QStringLiteral("all_scores"));
    if (sc.isObject()) {
      scores = sc.toObject();
    }
    QJsonObject expl;
    const QJsonValue ex = obj.value(QStringLiteral("explanation"));
    if (ex.isObject()) {
      expl = ex.toObject();
    } else if (predict_return_explanation_chk_ != nullptr && predict_return_explanation_chk_->isChecked()) {
      expl = obj;
      expl.remove(QStringLiteral("latency_ms"));
    }
    update_confidence_panel(lab, conf, anomaly, ood, scores, expl);
  }

  bool native_predict_detail(const std::vector<double>& x_in, std::string* label_out, double* conf_out,
                             QJsonObject* scores_out, double* anomaly_out, bool* is_ood_out,
                             QJsonObject* explanation_out) {
    if (!model_ || label_out == nullptr || conf_out == nullptr) {
      return false;
    }
    std::vector<double> x_latent;
    const double* x_ptr = nullptr;
    if (pre_ != nullptr) {
      if (static_cast<int>(x_in.size()) != pre_->input_dim) {
        return false;
      }
      x_latent = pre_->transform_one(x_in);
      if (static_cast<int>(x_latent.size()) != model_->d_latent) {
        return false;
      }
      x_ptr = x_latent.data();
    } else {
      if (static_cast<int>(x_in.size()) != model_->d_latent) {
        return false;
      }
      x_ptr = x_in.data();
    }
    if (model_->labels.empty()) {
      return false;
    }
    std::vector<double> H;
    cypha::batch_encode(*model_, x_ptr, 1, H);
    const int k = static_cast<int>(model_->labels.size());
    std::vector<double> llrs;
    double anomaly = 0.0;
    bool is_ood = false;
    double r_eff = 0.0;
    if (use_gh_chk_ != nullptr && use_gh_chk_->isChecked()) {
      cypha::CyphaInferOptions kopt{};
      fill_kernel_infer_options(kopt);
      const cypha::GhInferAtHResult gh = cypha::gh_infer_at_h(*model_, H.data(), native_gh_chi_, native_gh_psi_,
                                                              kGhNigAdaptAlphaShell, &kopt);
      *label_out = gh.label;
      *conf_out = gh.confidence;
      llrs = gh.llrs;
      r_eff = gh.r_eff;
      const double r_base =
          (model_->has_mahal_ema && model_->mahal_ema > 0.0) ? model_->mahal_ema : 1.0;
      anomaly = cypha::gh_infer_anomaly_score(r_eff, r_base);
      is_ood = anomaly > studio_prefs_.inference_ood_threshold;
    } else {
      cypha::CyphaInferOptions iopt{};
      iopt.deliberation_lo = model_->deliberation_lo;
      iopt.deliberation_hi = model_->deliberation_hi;
      iopt.use_field = true;
      fill_kernel_infer_options(iopt);
      const cypha::InferAtHResult inf = cypha::infer_at_h(*model_, H.data(), iopt);
      *label_out = inf.label;
      *conf_out = inf.confidence;
      llrs = inf.llrs;
    }
    if (scores_out != nullptr) {
      QJsonObject scores;
      for (int j = 0; j < k && j < static_cast<int>(llrs.size()); ++j) {
        scores[QString::fromStdString(model_->labels[static_cast<std::size_t>(j)])] = llrs[static_cast<std::size_t>(j)];
      }
      *scores_out = scores;
    }
    if (anomaly_out != nullptr) {
      *anomaly_out = anomaly;
    }
    if (is_ood_out != nullptr) {
      *is_ood_out = is_ood;
    }
    const bool want_expl =
        predict_return_explanation_chk_ != nullptr && predict_return_explanation_chk_->isChecked();
    if (want_expl && explanation_out != nullptr) {
      QJsonObject expl;
      expl[QStringLiteral("label")] = QString::fromStdString(*label_out);
      expl[QStringLiteral("confidence")] = *conf_out;
      expl[QStringLiteral("anomaly_score")] = anomaly;
      expl[QStringLiteral("is_ood")] = is_ood;
      expl[QStringLiteral("r_eff")] = r_eff;
      QJsonObject cdet;
      const int d = model_->d_latent;
      for (int ci = 0; ci < k; ++ci) {
        double sumsq = 0.0;
        for (int j = 0; j < d; ++j) {
          const double v = model_->D[static_cast<std::size_t>(ci * d + j)];
          sumsq += v * v;
        }
        QJsonObject row;
        row[QStringLiteral("n_obs")] = model_->n_obs[static_cast<std::size_t>(ci)];
        row[QStringLiteral("delta_mu_norm")] = std::sqrt(sumsq);
        cdet[QString::fromStdString(model_->labels[static_cast<std::size_t>(ci)])] = row;
      }
      expl[QStringLiteral("class_details")] = cdet;
      double wh = 0.0;
      for (int j = 0; j < d; ++j) {
        const double t = H[static_cast<std::size_t>(j)] - model_->mu_world[static_cast<std::size_t>(j)];
        wh += t * t;
      }
      expl[QStringLiteral("world_mu_distance")] = std::sqrt(wh);
      if (scores_out != nullptr) {
        expl[QStringLiteral("all_scores")] = *scores_out;
      }
      *explanation_out = expl;
    }
    return true;
  }

#ifdef CYPHA_SHELL_EXPERIMENT_DB
  void experiment_refresh_experiments_table() {
    if (exp_experiments_table_ == nullptr || !exp_db_) {
      return;
    }
    exp_experiments_table_->setRowCount(0);
    std::vector<cypha::ExperimentDbExperimentRow> rows;
    std::string err;
    if (!experiment_db_list_experiments(*exp_db_, &rows, &err)) {
      return;
    }
    for (const auto& e : rows) {
      const int row = exp_experiments_table_->rowCount();
      exp_experiments_table_->insertRow(row);
      const QString id_short = QString::fromStdString(e.experiment_id).right(16);
      exp_experiments_table_->setItem(row, 0, new QTableWidgetItem(id_short));
      auto* id_item = exp_experiments_table_->item(row, 0);
      id_item->setData(Qt::UserRole, QString::fromStdString(e.experiment_id));
      exp_experiments_table_->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(e.name)));
      exp_experiments_table_->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(e.task)));
      exp_experiments_table_->setItem(row, 3,
                                      new QTableWidgetItem(QString::number(e.created_at, 'f', 1)));
    }
  }

  void experiment_refresh_runs_table() {
    if (exp_runs_table_ == nullptr || !exp_db_) return;
    exp_runs_table_->setRowCount(0);
    std::string filter_exp = exp_active_experiment_id_;
    if (exp_experiments_table_ != nullptr && exp_experiments_table_->currentRow() >= 0) {
      auto* id_item = exp_experiments_table_->item(exp_experiments_table_->currentRow(), 0);
      if (id_item != nullptr) {
        const QString full_id = id_item->data(Qt::UserRole).toString();
        if (!full_id.isEmpty()) {
          filter_exp = full_id.toStdString();
        }
      }
    }
    std::vector<cypha::ExperimentDbRunRow> rows;
    std::string err;
    if (!experiment_db_list_runs(*exp_db_,
                                 filter_exp.empty() ? nullptr : filter_exp.c_str(),
                                 nullptr,
                                 50, 0,
                                 &rows, &err)) {
      return;
    }
    for (const auto& r : rows) {
      const int row = exp_runs_table_->rowCount();
      exp_runs_table_->insertRow(row);
      const QString run_id_short = QString::fromStdString(r.run_id).right(12);
      auto* id_item = new QTableWidgetItem(run_id_short);
      id_item->setData(Qt::UserRole, QString::fromStdString(r.run_id));
      exp_runs_table_->setItem(row, 0, id_item);
      exp_runs_table_->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(r.name)));
      exp_runs_table_->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(r.status)));
      const QString acc_str = (r.accuracy >= 0.0)
                                  ? QString::number(r.accuracy * 100.0, 'f', 1)
                                  : QStringLiteral("—");
      exp_runs_table_->setItem(row, 3, new QTableWidgetItem(acc_str));
      exp_runs_table_->setItem(row, 4, new QTableWidgetItem(QString::number(r.n_steps)));
      exp_runs_table_->setItem(row, 5, new QTableWidgetItem(QString::fromStdString(r.checkpoint_path)));
    }
  }

  void experiment_refresh_all() {
    experiment_refresh_experiments_table();
    experiment_refresh_runs_table();
  }

  static QVector<double> parse_loss_curve_from_metrics_json(const std::string& metrics_json) {
    QVector<double> losses;
    QJsonParseError err;
    const QJsonDocument doc =
        QJsonDocument::fromJson(QByteArray::fromStdString(metrics_json), &err);
    if (!doc.isArray()) {
      return losses;
    }
    for (const QJsonValue& v : doc.array()) {
      if (!v.isObject()) {
        continue;
      }
      const QJsonObject o = v.toObject();
      if (o.contains(QStringLiteral("loss"))) {
        losses.append(o.value(QStringLiteral("loss")).toDouble());
      }
    }
    return losses;
  }

  static QString format_compare_metric(double v, bool as_percent) {
    if (std::isnan(v)) {
      return QStringLiteral("—");
    }
    if (as_percent) {
      return QString::number(v * 100.0, 'f', 1);
    }
    return QString::number(v, 'g', 4);
  }

  static QString format_compare_duration(double duration_s) {
    if (duration_s <= 0.0 || std::isnan(duration_s)) {
      return QStringLiteral("—");
    }
    return QString::number(duration_s, 'f', 1);
  }

  static QString format_compare_loss(double v) {
    if (std::isnan(v)) {
      return QStringLiteral("—");
    }
    return QString::number(v, 'g', 4);
  }

  static double loss_curve_min(const QVector<double>& losses) {
    if (losses.isEmpty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return *std::min_element(losses.constBegin(), losses.constEnd());
  }

  static double loss_curve_final(const QVector<double>& losses) {
    if (losses.isEmpty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return losses.back();
  }

  void experiment_populate_compare_stats_table(
      const QVector<QPair<QString, cypha::ExperimentDbRunCompareRow>>& rows,
      const QVector<QVector<double>>& loss_curves) {
    if (exp_compare_stats_table_ == nullptr) {
      return;
    }
    exp_compare_stats_table_->setRowCount(rows.size());
    for (int i = 0; i < rows.size(); ++i) {
      const QString& label = rows[i].first;
      const cypha::ExperimentDbRunCompareRow& c = rows[i].second;
      const QVector<double>& losses = loss_curves[i];
      exp_compare_stats_table_->setItem(i, 0, new QTableWidgetItem(label));
      exp_compare_stats_table_->setItem(i, 1, new QTableWidgetItem(format_compare_metric(c.accuracy, true)));
      exp_compare_stats_table_->setItem(i, 2, new QTableWidgetItem(format_compare_metric(c.macro_f1, true)));
      exp_compare_stats_table_->setItem(i, 3, new QTableWidgetItem(format_compare_metric(c.r2_score, false)));
      exp_compare_stats_table_->setItem(i, 4, new QTableWidgetItem(format_compare_metric(c.rmse, false)));
      exp_compare_stats_table_->setItem(i, 5, new QTableWidgetItem(QString::number(c.n_steps)));
      exp_compare_stats_table_->setItem(i, 6, new QTableWidgetItem(format_compare_duration(c.duration_s)));
      exp_compare_stats_table_->setItem(i, 7, new QTableWidgetItem(format_compare_loss(loss_curve_final(losses))));
      exp_compare_stats_table_->setItem(i, 8, new QTableWidgetItem(format_compare_loss(loss_curve_min(losses))));
    }
    exp_compare_stats_table_->setVisible(!rows.isEmpty());
  }

  static QString experiment_compare_summary_text(
      const QVector<QPair<QString, cypha::ExperimentDbRunCompareRow>>& rows,
      const QVector<QVector<double>>& loss_curves, int plotted_runs) {
    QStringList summary_bits;
    for (int i = 0; i < rows.size(); ++i) {
      const QString& label = rows[i].first;
      const cypha::ExperimentDbRunCompareRow& c = rows[i].second;
      const QVector<double>& losses = loss_curves[i];
      summary_bits.append(
          QStringLiteral("%1 acc=%2 f1=%3 steps=%4 dur=%5s final_loss=%6")
              .arg(label)
              .arg(format_compare_metric(c.accuracy, true) + QLatin1Char('%'))
              .arg(format_compare_metric(c.macro_f1, true))
              .arg(c.n_steps)
              .arg(format_compare_duration(c.duration_s))
              .arg(format_compare_loss(loss_curve_final(losses))));
    }

    QString headline;
    if (plotted_runs > 0) {
      headline = QStringLiteral("%1 run(s) plotted").arg(plotted_runs);
    }
    if (rows.size() >= 2) {
      int best_idx = 0;
      int worst_idx = 0;
      for (int i = 1; i < rows.size(); ++i) {
        if (rows[i].second.accuracy > rows[best_idx].second.accuracy) {
          best_idx = i;
        }
        if (rows[i].second.accuracy < rows[worst_idx].second.accuracy) {
          worst_idx = i;
        }
      }
      const double spread = (rows[best_idx].second.accuracy - rows[worst_idx].second.accuracy) * 100.0;
      if (!std::isnan(spread) && spread > 0.0) {
        headline += QStringLiteral(" · best acc %1 (%2) · Δacc %3 pp vs worst")
                        .arg(format_compare_metric(rows[best_idx].second.accuracy, true) + QLatin1Char('%'))
                        .arg(rows[best_idx].first)
                        .arg(QString::number(spread, 'f', 1));
      }
    }
    if (!summary_bits.isEmpty()) {
      if (!headline.isEmpty()) {
        headline += QStringLiteral(" — ");
      }
      headline += summary_bits.join(QStringLiteral("  |  "));
    }
    return headline.isEmpty() ? QStringLiteral("(select 2+ runs with loss history)") : headline;
  }

  void experiment_compare_selected_runs() {
    if (!exp_db_ || exp_runs_table_ == nullptr || exp_compare_chart_ == nullptr) {
      return;
    }
    const auto selected = exp_runs_table_->selectionModel()->selectedRows();
    if (selected.isEmpty()) {
      QMessageBox::information(this, QStringLiteral("Compare runs"),
                               QStringLiteral("Select one or more runs in the table."));
      return;
    }

    std::vector<std::string> run_ids;
    run_ids.reserve(static_cast<std::size_t>(selected.size()));
    for (const QModelIndex& idx : selected) {
      auto* id_item = exp_runs_table_->item(idx.row(), 0);
      if (id_item == nullptr) {
        continue;
      }
      const QString rid = id_item->data(Qt::UserRole).toString();
      if (!rid.isEmpty()) {
        run_ids.push_back(rid.toStdString());
      }
    }
    if (run_ids.empty()) {
      return;
    }

    std::vector<cypha::ExperimentDbRunCompareRow> cmp;
    std::string err;
    if (!experiment_db_compare_runs(*exp_db_, run_ids, &cmp, &err)) {
      QMessageBox::warning(this, QStringLiteral("Compare runs"),
                           QStringLiteral("compare_runs failed: %1").arg(QString::fromStdString(err)));
      return;
    }

    QVector<QPair<QString, QVector<double>>> series;
    QVector<QPair<QString, cypha::ExperimentDbRunCompareRow>> stats_rows;
    QVector<QVector<double>> stats_losses;
    for (const std::string& rid : run_ids) {
      cypha::ExperimentDbRunRow row{};
      bool found = false;
      if (!experiment_db_get_run(*exp_db_, rid.c_str(), &row, &found, &err) || !found) {
        continue;
      }
      const QVector<double> losses = parse_loss_curve_from_metrics_json(row.metrics_history_json);
      QString label = QString::fromStdString(row.name);
      if (label.isEmpty()) {
        label = QString::fromStdString(rid).right(12);
      }

      const cypha::ExperimentDbRunCompareRow* cmp_row = nullptr;
      for (const auto& c : cmp) {
        if (c.run_id == rid) {
          cmp_row = &c;
          break;
        }
      }
      if (cmp_row != nullptr) {
        stats_rows.append(qMakePair(label, *cmp_row));
        stats_losses.append(losses);
      }

      if (losses.isEmpty()) {
        continue;
      }
      series.append(qMakePair(label, losses));
    }

    if (series.isEmpty()) {
      QMessageBox::information(this, QStringLiteral("Compare runs"),
                               QStringLiteral("Selected runs have no loss entries in metrics_history."));
      return;
    }

    exp_compare_chart_->set_compare_loss_runs(series);
    experiment_populate_compare_stats_table(stats_rows, stats_losses);
    if (exp_compare_clear_btn_ != nullptr) {
      exp_compare_clear_btn_->setEnabled(true);
    }
    if (exp_compare_summary_label_ != nullptr) {
      exp_compare_summary_label_->setText(
          experiment_compare_summary_text(stats_rows, stats_losses, series.size()));
    }
  }
#endif  // CYPHA_SHELL_EXPERIMENT_DB

  void apply_model_load() {
    QString err;
    std::unique_ptr<cypha::CyphaInferModel> m;
    if (!try_load_cypha_paths(cypha_path_, ff_path_, f_field_flat_, m, &err)) {
      QMessageBox::warning(this, QStringLiteral("Load failed"), err);
      model_.reset();
      predict_btn_->setEnabled(false);
      predict_rest_btn_->setEnabled(false);
      update_rest_btn_->setEnabled(false);
      if (batch_predict_csv_btn_ != nullptr) batch_predict_csv_btn_->setEnabled(false);
      if (mke_predict_btn_ != nullptr) mke_predict_btn_->setEnabled(false);
      if (mke_bulk_btn_ != nullptr) mke_bulk_btn_->setEnabled(false);
      if (native_train_one_btn_ != nullptr) {
        native_train_one_btn_->setEnabled(false);
      }
      if (csv_bulk_native_btn_ != nullptr) {
        csv_bulk_native_btn_->setEnabled(false);
      }
      if (save_native_btn_ != nullptr) {
        save_native_btn_->setEnabled(false);
      }
      path_label_->setText(cypha_path_.isEmpty() ? QStringLiteral("(no model)") : cypha_path_);
      update_features_hint();
      return;
    }
    model_ = std::move(m);
    path_label_->setText(cypha_path_);
    predict_btn_->setEnabled(true);
    predict_rest_btn_->setEnabled(true);
    update_rest_btn_->setEnabled(true);
    if (batch_predict_csv_btn_ != nullptr) batch_predict_csv_btn_->setEnabled(true);
    if (mke_predict_btn_ != nullptr) mke_predict_btn_->setEnabled(true);
    features_edit_->clear();
    (void)try_load_train_hparams_adjacent();
    reload_preprocessor_only();
    reinit_native_train_state();
    result_label_->setText(
        QStringLiteral("Loaded: latent dim %1, %2 classes")
            .arg(model_->d_latent)
            .arg(static_cast<int>(model_->labels.size())));
  }

  QProcess rest_proc_;
  QProcess bench_proc_;
  QTabWidget* main_tabs_{};
  QLabel* workflow_banner_{};
  QPushButton* load_btn_{};
  QPushButton* new_model_btn_{};
  QPushButton* ff_btn_{};
  QPushButton* pre_btn_{};
  QPushButton* pre_clear_btn_{};
  QPushButton* predict_btn_{};
  QPushButton* predict_rest_btn_{};
  QCheckBox* predict_return_explanation_chk_{};
  QCheckBox* predict_self_correct_chk_{};
  QPushButton* update_rest_btn_{};
  QPushButton* rest_browse_btn_{};
  QPushButton* rest_start_btn_{};
  QPushButton* rest_stop_btn_{};
  QPushButton* rest_health_btn_{};
  QPushButton* rest_ready_btn_{};
  QPushButton* rest_models_btn_{};
  QPushButton* rest_clear_log_btn_{};
  QPlainTextEdit* rest_log_{};
  QLabel* path_label_{};
  QLabel* ff_label_{};
  QLabel* pre_label_{};
  QLabel* features_hint_{};
  QLabel* result_label_{};
  QLabel* rest_status_label_{};
  QLineEdit* features_edit_{};
  QLineEdit* rest_base_edit_{};
  QLineEdit* rest_bin_edit_{};
  QLineEdit* rest_listen_edit_{};
  QLineEdit* update_label_edit_{};
  QPushButton* csv_btn_{};
  QPushButton* csv_inspect_btn_{};
  QPushButton* csv_fill_row0_btn_{};
  QPushButton* csv_bulk_train_btn_{};
  QPushButton* csv_bulk_native_btn_{};
  QPushButton* bulk_native_cancel_btn_{};
  // ── Experiment DB (M6) ──────────────────────────────────────────────────────
#ifdef CYPHA_SHELL_EXPERIMENT_DB
  std::unique_ptr<cypha::ExperimentDb> exp_db_;
  std::string exp_active_run_id_;
  std::string exp_active_experiment_id_;
  QPushButton* exp_db_btn_{};
  QLabel*      exp_db_label_{};
  QLineEdit*   exp_name_edit_{};
  QLineEdit*   exp_run_name_edit_{};
  QPushButton* exp_start_run_btn_{};
  QPushButton* exp_finish_run_btn_{};
  QPushButton* exp_refresh_btn_{};
  QLabel*      exp_status_label_{};
  QTableWidget* exp_experiments_table_{};
  QTableWidget* exp_runs_table_{};
  QPushButton*  exp_compare_btn_{};
  QPushButton*  exp_compare_clear_btn_{};
  QLabel*       exp_compare_summary_label_{};
  QTableWidget* exp_compare_stats_table_{};
  SimpleLossChart* exp_compare_chart_{};
#else
  // Stub pointers so guards are not needed in non-DB code paths
  void* exp_db_btn_{};
  void* exp_start_run_btn_{};
  void* exp_finish_run_btn_{};
#endif
  // ── MKE Regressor state ─────────────────────────────────────────────────────
  QDoubleSpinBox* mke_ff_spin_{};
  QDoubleSpinBox* mke_pi_spin_{};
  QPushButton*    mke_init_btn_{};
  QPushButton*    mke_bulk_btn_{};
  QPushButton*    mke_predict_btn_{};
  QLabel*         mke_result_label_{};
  std::unordered_map<std::string, std::vector<double>> mke_w_by_label_;
  std::unordered_map<std::string, std::vector<double>> mke_p_by_label_;
  // ── Batch predict ────────────────────────────────────────────────────────────
  QPushButton* batch_predict_csv_btn_{};
  QPushButton* batch_predict_export_btn_{};
  QTableWidget* batch_predict_table_{};
  std::vector<std::array<QString,3>> batch_predict_results_;
  QVector<QString> batch_predict_y_true_;
  QVector<QString> batch_predict_y_pred_;
  QAction* view_confusion_action_{};
  StudioPreferences studio_prefs_;
  // ── Training progress panel ─────────────────────────────────────────────────
  QLabel*              train_prog_label_{};
  QPushButton*         train_prog_reset_btn_{};
  QTableWidget*        train_prog_class_table_{};
  PerClassAccuracyBar* train_prog_acc_bar_{};
  int    train_prog_win_correct_{0};
  int    train_prog_win_total_{0};
  double train_prog_ema_loss_{0.0};
  QSpinBox* csv_bulk_max_rows_spin_{};
  LossMetricsPanel* loss_chart_{};
  QPushButton* loss_chart_save_btn_{};
  QPushButton* loss_csv_save_btn_{};
  QPushButton* loss_svg_save_btn_{};
  QPushButton* loss_chart_clear_btn_{};
  QCheckBox* loss_ema_chk_{};
  QCheckBox* loss_y_lock_chk_{};
  QDoubleSpinBox* loss_y_min_spin_{};
  QDoubleSpinBox* loss_y_max_spin_{};
  QTableWidget* train_log_table_{};
  QPushButton* train_log_clear_btn_{};
  QPushButton* train_log_export_btn_{};
  QVector<double> last_loss_plot_rest_{};
  QVector<double> last_loss_plot_native_{};
  QCheckBox* csv_regression_chk_{};
  QCheckBox* sort_by_uncertainty_chk_{};
  QCheckBox* curriculum_chk_{};
  QCheckBox* use_gh_chk_{};
  QPushButton* replay_u01_btn_{};
  QLabel* replay_u01_label_{};
  QLineEdit* mke_correct_label_edit_{};
  QLineEdit* mke_router_label_edit_{};
  QPushButton* native_train_one_btn_{};
  QPushButton* save_native_btn_{};
  QLineEdit* hp_world_lr_edit_{};
  QLineEdit* hp_delta_lr_edit_{};
  QLineEdit* hp_ood_sigma_edit_{};
  QLineEdit* hp_enc_lr_edit_{};
  QLineEdit* hp_replay_ratio_edit_{};
  QCheckBox* use_kernel_llr_chk_{};
  QDoubleSpinBox* kernel_blend_spin_{};
  QSpinBox* hp_replay_cap_spin_{};
  QSpinBox* hp_align_every_spin_{};
  QSpinBox* hp_temp_recalib_spin_{};
  QPushButton* hp_apply_btn_{};
  QPushButton* hp_defaults_btn_{};
  QLineEdit* rest_load_name_edit_{};
  QLineEdit* rest_load_version_edit_{};
  QPushButton* rest_post_load_btn_{};
  QPushButton* reg_root_btn_{};
  QPushButton* reg_scan_btn_{};
  QPushButton* reg_rest_models_btn_{};
  QPushButton* reg_load_btn_{};
  QPushButton* reg_rest_load_btn_{};
  QPushButton* reg_register_btn_{};
  QTableWidget* reg_models_table_{};
  QPushButton* card_btn_{};
  QLineEdit* csv_target_name_edit_{};
  QLineEdit* csv_target_index_edit_{};
  QLineEdit* csv_feature_names_edit_{};
  QLabel* csv_label_{};
  QLabel* reg_root_label_{};
  QLabel* card_label_{};
  QComboBox* reg_combo_{};
  QLabel* conf_pred_label_{};
  QLabel* conf_ood_label_{};
  QProgressBar* conf_ood_bar_{};
  QTableWidget* conf_scores_table_{};
  QPlainTextEdit* conf_explanation_edit_{};
  QTextBrowser* help_browser_{};
  QPlainTextEdit* dataset_info_{};
  // ── Dataset panel ────────────────────────────────────────────────────────
  QComboBox*    col_target_combo_{};
  QListWidget*  col_feature_list_{};
  QTableWidget* csv_preview_table_{};
  QLabel*       csv_stats_label_{};
  QSpinBox*     val_split_spin_{};
  QPushButton*  fit_pre_btn_{};
  QStringList   csv_col_headers_;
  bool          col_picker_updating_{false};
  // ── Bulk training thread state ────────────────────────────────────────────
  QThread*               bulk_thread_{};
  BulkNativeTrainWorker* bulk_worker_{};
  std::atomic<bool>      bulk_cancel_flag_{false};
  QThread*                  bulk_rest_thread_{};
  BulkRestUpdateWorker*     bulk_rest_worker_{};
  std::atomic<bool>         bulk_rest_cancel_flag_{false};
  QVector<double>           bulk_rest_accum_losses_;
  QThread*                  bulk_mke_thread_{};
  BulkMkeTrainWorker*       bulk_mke_worker_{};
  std::atomic<bool>         bulk_mke_cancel_flag_{false};
  int                    bulk_train_n_{0};
  QVector<double>                bulk_accum_losses_;
  QVector<BulkLogEntry>          bulk_accum_log_;
  int                            bulk_val_n_{};
  int                            bulk_total_n_{};
  cypha::CsvDenseResult          bulk_train_data_{};
  QString cypha_path_;
  QString ff_path_;
  QString pre_path_;
  QString csv_path_;
  QString registry_root_;
  QString card_path_;
  std::vector<cypha::RegistryModelRef> reg_refs_;
  cypha::CsvDenseResult last_csv_{};
  bool last_csv_ok_{false};
  std::vector<double> f_field_flat_;
  std::unique_ptr<cypha::CyphaInferModel> model_;
  std::unique_ptr<cypha::PreprocessorState> pre_;
  std::vector<double> replay_u01_cache_;
  std::unique_ptr<cypha::CyphaDifMemoryState> native_mem_;
  std::unique_ptr<cypha::KernelMemory> native_kernel_mem_;
  bool native_use_kernel_llr_{false};
  double native_kernel_blend_{0.5};
  std::unique_ptr<cypha::ReplayBuffer> native_replay_;
  std::mt19937 native_rng_{424242u};
  cypha::TrainStepParams native_tsp_{};
  double native_world_lr_{0.008};
  double native_delta_lr_{0.05};
  double native_ood_sigma_{15.0};
  int native_enc_updates_{0};
  int native_total_steps_{0};
  double native_llr_ema_{0.0};
  std::vector<double> native_gh_inv_v_{};
  double native_gh_R_base_{1.0};
  double native_gh_chi_{1.0};
  double native_gh_psi_{1.0};
  bool native_train_ok_{false};
  int native_replay_cap_applied_{10000};
  // ── CyphaLM tab ─────────────────────────────────────────────────────────────
  QLineEdit* lm_path_edit_{};
  QPushButton* lm_browse_btn_{};
  QPushButton* lm_load_btn_{};
  QCheckBox* lm_use_rest_chk_{};
  QLabel* lm_status_label_{};
  QLineEdit* lm_prompt_edit_{};
  QSpinBox* lm_n_tokens_spin_{};
  QComboBox* lm_strategy_combo_{};
  QDoubleSpinBox* lm_top_p_spin_{};
  QDoubleSpinBox* lm_temperature_spin_{};
  QCheckBox* lm_epistemic_halt_chk_{};
  QPushButton* lm_generate_btn_{};
  QPlainTextEdit* lm_output_edit_{};
  QLineEdit* bench_bin_edit_{};
  QPushButton* bench_browse_btn_{};
  QPushButton* bench_run_d17_btn_{};
  QPlainTextEdit* bench_log_edit_{};
  std::unique_ptr<cypha::cyphalm::CyphaLMModel> lm_model_;
  bool lm_loaded_{false};
  QString lm_source_path_;
  int lm_n_generations_{0};
  cypha::intelligence::EpistemicThreshold lm_epistemic_threshold_{0.5, 5.0};
};

}  // namespace

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  QApplication::setApplicationName(QStringLiteral("Cypha Qt shell"));
  apply_cypha_theme(load_studio_preferences().dark_theme);

  const QStringList args = QApplication::arguments();
  for (int i = 1; i + 1 < args.size(); ++i) {
    if (args.at(i) == QStringLiteral("--smoke")) {
      const QString cypha = args.at(i + 1);
      QString ff;
      if (i + 2 < args.size() && !args.at(i + 2).startsWith(QLatin1Char('-'))) {
        ff = args.at(i + 2);
      }
      return run_smoke(cypha, ff);
    }
  }
  if (args.contains(QStringLiteral("--help")) || args.contains(QStringLiteral("-h"))) {
    std::printf(
        "cypha_qt_shell [options]\n"
        "  GUI: tabs — Data, Model, Train, Predict (confidence+explanation), Registry, Server,\n"
        "       Experiments (SQLite when built), CyphaLM, Help (NATIVE_QUICKSTART).\n"
        "       load .cypha; CSV inspect + bulk REST /update + loss chart;\n"
        "       optional F_field JSON; preprocessor.json; native predict; spawn cypha_rest\n"
        "       (--registry when set); REST /health, /ready, /models, /load, /predict, /update;\n"
        "       native train_step + bulk CSV; save .cypha (merge + infer patch); train hparams UI;\n"
        "       loss chart export PNG/SVG/CSV; EMA smoothing; Y lock; training log table;\n"
        "       POST /predict return_explanation → confidence panel (class_details, OOD meter);\n"
        "       optional replay_u01 JSON; regression_y bulk for MKE.\n"
        "       CyphaLM: native checkpoint + generate; REST /lm/load + /generate; bench --domain 17.\n"
        "  --smoke <path.cypha> [f_field.json]  headless load + zero-vector native predict (CI).\n"
        "  -h, --help            this message\n");
    return 0;
  }

  MainWindow w;
  w.show();
  return app.exec();
}
