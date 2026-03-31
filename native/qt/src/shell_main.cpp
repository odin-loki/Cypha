// Cypha Qt 6 Widgets shell — native infer (optional preprocessor), REST /predict + /update, spawn cypha_rest.
// --smoke <model.cypha> [f_field.json]
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
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
#include <QPolygonF>
#include <QPointF>
#include <QMutex>
#include <QProgressDialog>
#include <QThread>
#include <QTimer>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QDialog>
#include <QDialogButtonBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QTemporaryFile>
#include <QTextStream>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QSpinBox>
#include <QStringList>
#include <QTableWidget>
#include <QTextCursor>
#include <QUrl>
#include <QVector>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/csv_ingest.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/mke_scalar_train_step.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/registry.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

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
                        const std::vector<double>& x_in, std::string* label_out, double* conf_out) {
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

  std::vector<double> H;
  cypha::batch_encode(m, x_ptr, 1, H);
  std::vector<double> llr;
  cypha::score_matrix_use_field(m, H.data(), 1, llr);
  const int k = static_cast<int>(m.labels.size());
  if (k <= 0) {
    return -2;
  }
  constexpr double kEps = 1e-8;
  std::vector<double> z(static_cast<std::size_t>(k));
  for (int j = 0; j < k; ++j) {
    z[static_cast<std::size_t>(j)] = llr[static_cast<std::size_t>(j)] / (m.temperature + kEps);
  }
  std::vector<double> probs;
  cypha::softmax_batch_like_python(z.data(), 1, k, kEps, probs);
  std::vector<double> gates;
  cypha::world_gate_vector_use_field(m, H.data(), 1, 1.0, 1.0, gates);
  int bi = 0;
  for (int j = 1; j < k; ++j) {
    if (probs[static_cast<std::size_t>(j)] > probs[static_cast<std::size_t>(bi)]) {
      bi = j;
    }
  }
  *label_out = m.labels[static_cast<std::size_t>(bi)];
  *conf_out = probs[static_cast<std::size_t>(bi)] * gates[0];
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

HttpJsonResult http_post_json(const QUrl& url, const QJsonObject& body) {
  HttpJsonResult r;
  QNetworkRequest req(url);
  req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
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
    r.err = QStringLiteral("%1\n%2").arg(err_s, QString::fromUtf8(r.raw));
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
    r.err = QStringLiteral("%1\n%2").arg(err_s, QString::fromUtf8(r.raw));
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
  }
  void set_loss_runs(const QVector<double>& rest, const QVector<double>& native_v,
                     const QVector<double>& rest_ema, const QVector<double>& native_ema) {
    losses_rest_ = rest;
    losses_native_ = native_v;
    losses_rest_ema_ = rest_ema;
    losses_native_ema_ = native_ema;
    update();
  }
  void clear_losses() {
    losses_rest_.clear();
    losses_native_.clear();
    losses_rest_ema_.clear();
    losses_native_ema_.clear();
    update();
  }
  void set_y_range_lock(bool locked, double lo, double hi) {
    y_range_locked_ = locked;
    y_lock_lo_ = lo;
    y_lock_hi_ = hi;
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
    if (!have_r && !have_n) {
      painter.setPen(QColor(140, 140, 150));
      painter.drawText(full.adjusted(8, 0, -8, 0), Qt::AlignCenter,
                       QStringLiteral("Per-step loss — bulk REST (blue) vs native (orange); dashed = EMA"));
      return;
    }

    // ── Compute range ────────────────────────────────────────────────────────
    auto expand_range = [](double& lo, double& hi, const QVector<double>& v) {
      for (double x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    };
    double lo = y_range_locked_ ? y_lock_lo_ : (have_r ? losses_rest_[0] : losses_native_[0]);
    double hi = y_range_locked_ ? y_lock_hi_ : lo;
    if (!y_range_locked_) {
      if (have_r)  expand_range(lo, hi, losses_rest_);
      if (have_n)  expand_range(lo, hi, losses_native_);
      if (!losses_rest_ema_.isEmpty())   expand_range(lo, hi, losses_rest_ema_);
      if (!losses_native_ema_.isEmpty()) expand_range(lo, hi, losses_native_ema_);
    }
    if (!(hi > lo)) hi = lo + 1e-9;

    const int max_n = std::max(losses_rest_.size(), losses_native_.size());

    // Margins: left=54 (Y labels), right=12, top=10, bottom=22 (X labels) + 22 (legend)
    const int legend_h = 20;
    const int x_label_h = 22;
    const QRect plot = full.adjusted(54, 10, -12, -(x_label_h + legend_h));

    // ── Grid lines + Y-axis ticks ─────────────────────────────────────────────
    const int kYTicks = 5;
    QFont tick_font = painter.font();
    tick_font.setPointSizeF(std::max(6.5, tick_font.pointSizeF() - 1.0));
    painter.setFont(tick_font);
    for (int ti = 0; ti <= kYTicks; ++ti) {
      const double frac = static_cast<double>(ti) / kYTicks;
      const double val  = lo + frac * (hi - lo);
      const int    py   = static_cast<int>(plot.bottom() - frac * plot.height());
      // Grid line
      painter.setPen(QPen(QColor(55, 58, 65), 1, Qt::SolidLine));
      painter.drawLine(plot.left(), py, plot.right(), py);
      // Tick label (right-aligned in left margin)
      painter.setPen(QColor(150, 150, 160));
      const QString lbl = QString::number(val, 'g', 3);
      const QRect lr(full.left(), py - 9, plot.left() - 4, 18);
      painter.drawText(lr, Qt::AlignRight | Qt::AlignVCenter, lbl);
    }

    // ── X-axis ticks ──────────────────────────────────────────────────────────
    const int kXTicks = 4;
    for (int ti = 0; ti <= kXTicks; ++ti) {
      const double frac = static_cast<double>(ti) / kXTicks;
      const int    step = static_cast<int>(std::round(frac * (max_n - 1)));
      const int    px   = static_cast<int>(plot.left() + frac * plot.width());
      painter.setPen(QPen(QColor(55, 58, 65), 1, Qt::SolidLine));
      painter.drawLine(px, plot.top(), px, plot.bottom());
      painter.setPen(QColor(150, 150, 160));
      const QString lbl = QString::number(step);
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
        const double ynorm = (v[0] - lo) / (hi - lo);
        const double py    = plot.bottom() - ynorm * plot.height();
        painter.drawEllipse(QPointF(plot.center().x(), py), 3.0, 3.0);
        return;
      }
      QPolygonF poly;
      const int    n     = v.size();
      const double denom = max_n > 1 ? static_cast<double>(max_n - 1) : 1.0;
      for (int i = 0; i < n; ++i) {
        const double tx    = static_cast<double>(i) / denom;
        const double ynorm = (v[i] - lo) / (hi - lo);
        poly << QPointF(plot.left() + tx * plot.width(),
                        plot.bottom() - ynorm * plot.height());
      }
      painter.drawPolyline(poly);
    };
    draw_series(losses_native_ema_, QColor(255, 170, 90), true, 1);
    draw_series(losses_rest_ema_,   QColor(110, 180, 255), true, 1);
    draw_series(losses_native_,     QColor(255, 170, 90), false, 2);
    draw_series(losses_rest_,       QColor(110, 180, 255), false, 2);

    // ── Legend ────────────────────────────────────────────────────────────────
    const int ley  = full.bottom() - legend_h + 2;
    const int leh  = legend_h - 4;
    struct LegItem { QColor col; bool dashed; QString text; };
    const QList<LegItem> items = {
        {QColor(110, 180, 255), false, QStringLiteral("REST")},
        {QColor(110, 180, 255), true,  QStringLiteral("REST EMA")},
        {QColor(255, 170, 90),  false, QStringLiteral("Native")},
        {QColor(255, 170, 90),  true,  QStringLiteral("Native EMA")},
    };
    int lx = 56;
    for (const auto& item : items) {
      QPen pen(item.col, item.dashed ? 1 : 2);
      if (item.dashed) pen.setStyle(Qt::DashLine);
      painter.setPen(pen);
      painter.drawLine(lx, ley + leh/2, lx + 18, ley + leh/2);
      painter.setPen(QColor(200, 200, 210));
      const QRect tr(lx + 22, ley, 80, leh);
      painter.drawText(tr, Qt::AlignLeft | Qt::AlignVCenter, item.text);
      lx += 104;
    }
  }

 private:
  QVector<double> losses_rest_{};
  QVector<double> losses_native_{};
  QVector<double> losses_rest_ema_{};
  QVector<double> losses_native_ema_{};
  bool y_range_locked_{false};
  double y_lock_lo_{-10.0};
  double y_lock_hi_{0.0};
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
#include <QBrush>
#include <QtCharts/QAbstractAxis>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

class LossChartPanel final : public QtCharts::QChartView {
 public:
  explicit LossChartPanel(QWidget* parent = nullptr) : QChartView(parent) {
    setMinimumHeight(128);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setRenderHint(QPainter::Antialiasing, true);
    auto* ch = new QtCharts::QChart();
    ch->setBackgroundBrush(QBrush(QColor(32, 34, 38)));
    ch->setBackgroundRoundness(0);
    ch->setTitleBrush(QBrush(QColor(200, 200, 210)));
    series_rest_ema_ = new QtCharts::QLineSeries();
    series_rest_ema_->setName(QStringLiteral("REST EMA"));
    {
      QPen pr(QColor(110, 180, 255, 200));
      pr.setStyle(Qt::DashLine);
      pr.setWidthF(1.2);
      series_rest_ema_->setPen(pr);
    }
    series_native_ema_ = new QtCharts::QLineSeries();
    series_native_ema_->setName(QStringLiteral("Native EMA"));
    {
      QPen pn(QColor(255, 170, 90, 200));
      pn.setStyle(Qt::DashLine);
      pn.setWidthF(1.2);
      series_native_ema_->setPen(pn);
    }
    series_rest_ = new QtCharts::QLineSeries();
    series_rest_->setName(QStringLiteral("REST /update"));
    series_rest_->setPen(QPen(QColor(110, 180, 255), 2));
    series_native_ = new QtCharts::QLineSeries();
    series_native_->setName(QStringLiteral("Native train"));
    series_native_->setPen(QPen(QColor(255, 170, 90), 2));
    ch->addSeries(series_rest_ema_);
    ch->addSeries(series_native_ema_);
    ch->addSeries(series_rest_);
    ch->addSeries(series_native_);
    ch->legend()->setAlignment(Qt::AlignBottom);
    ch->legend()->setLabelColor(QColor(200, 200, 210));
    ch->legend()->hide();
    setChart(ch);
  }

  void set_loss_runs(const QVector<double>& rest, const QVector<double>& natv, const QVector<double>& rest_ema,
                     const QVector<double>& natv_ema) {
    series_rest_->clear();
    series_native_->clear();
    series_rest_ema_->clear();
    series_native_ema_->clear();
    QtCharts::QChart* ch = chart();
    const bool have_r = !rest.isEmpty();
    const bool have_n = !natv.isEmpty();
    if (!have_r && !have_n) {
      ch->setTitle(QStringLiteral("Per-step loss — bulk REST vs bulk native (dashed = EMA)"));
      const QList<QtCharts::QAbstractAxis*> ax = ch->axes();
      for (QtCharts::QAbstractAxis* a : ax) {
        ch->removeAxis(a);
        a->deleteLater();
      }
      ch->legend()->hide();
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
      // Style axes for dark background
      for (QtCharts::QAbstractAxis* ax : ch->axes()) {
        ax->setLabelsColor(QColor(190, 190, 200));
        ax->setLinePenColor(QColor(80, 85, 95));
        ax->setGridLinePen(QPen(QColor(55, 58, 65), 1, Qt::SolidLine));
        if (auto* va = qobject_cast<QtCharts::QValueAxis*>(ax)) {
          va->setTickCount(6);
          va->setLabelFormat(QStringLiteral("%.3g"));
        }
      }
    }
    double lo = have_r ? rest[0] : natv[0];
    double hi = lo;
    auto expand = [&](const QVector<double>& v) {
      for (double x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    };
    if (have_r) {
      expand(rest);
    }
    if (have_n) {
      expand(natv);
    }
    if (!rest_ema.isEmpty()) {
      expand(rest_ema);
    }
    if (!natv_ema.isEmpty()) {
      expand(natv_ema);
    }
    if (!(hi > lo)) {
      hi = lo + 1e-9;
    }
    const int max_n = std::max(rest.size(), natv.size());
    const QList<QtCharts::QAbstractAxis*> vy = ch->axes(Qt::Vertical);
    if (!vy.isEmpty()) {
      if (auto* va = qobject_cast<QtCharts::QValueAxis*>(vy.front())) {
        va->setRange(lo, hi);
      }
    }
    const QList<QtCharts::QAbstractAxis*> hx = ch->axes(Qt::Horizontal);
    if (!hx.isEmpty()) {
      if (auto* ha = qobject_cast<QtCharts::QValueAxis*>(hx.front())) {
        ha->setRange(0.0, max_n > 1 ? static_cast<double>(max_n - 1) : 1.0);
      }
    }
    ch->legend()->setVisible(true);
  }

  void clear_losses() { set_loss_runs({}, {}, {}, {}); }
  void set_y_range_lock(bool locked, double lo, double hi) {
    QtCharts::QChart* ch = chart();
    if (!ch) return;
    const QList<QtCharts::QAbstractAxis*> vy = ch->axes(Qt::Vertical);
    if (vy.isEmpty()) return;
    if (auto* va = qobject_cast<QtCharts::QValueAxis*>(vy.front())) {
      if (locked) {
        va->setRange(lo, hi);
      }
    }
  }

 private:
  QtCharts::QLineSeries* series_rest_ema_{};
  QtCharts::QLineSeries* series_native_ema_{};
  QtCharts::QLineSeries* series_rest_{};
  QtCharts::QLineSeries* series_native_{};
};
#else
using LossChartPanel = SimpleLossChart;
#endif

enum class LossPlotSource { RestBulk, NativeBulk };

// ─────────────────────────────────────────────────────────────────────────────
// BulkTrainState — shared between the training thread and the main thread.
// The background thread writes results; the main thread polls via QTimer.
// ─────────────────────────────────────────────────────────────────────────────
struct BulkLogEntry { int step_n{}; QString label; double loss{}; bool correct{}; };

struct BulkTrainState {
  std::atomic<int>  step_count{0};
  std::atomic<bool> cancel{false};
  std::atomic<bool> done{false};
  QMutex            steps_mutex;
  QVector<BulkLogEntry> new_steps;  // guarded by steps_mutex
  // Final scalars — written once (before done=true), read by main thread after
  int    final_total_steps{};
  double final_ema_loss{};
  double final_llr_ema{};
  int    final_win_total{};
  int    final_win_correct{};
  double final_gh_chi{1.0};
  double final_gh_psi{1.0};
  int    final_enc_updates{};
  QString error_msg;
};

class MainWindow final : public QMainWindow {
 public:
  MainWindow() {
    setWindowTitle(QStringLiteral("Cypha — Qt shell (native + REST)"));
    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    auto* row1 = new QHBoxLayout();
    load_btn_ = new QPushButton(QStringLiteral("Load model…"), central);
    new_model_btn_ = new QPushButton(QStringLiteral("New model…"), central);
    new_model_btn_->setToolTip(QStringLiteral(
        "Create a fresh empty model in-memory (no existing .cypha needed).\n"
        "Specify input_dim and field_dim, then train directly from CSV."));
    path_label_ = new QLabel(QStringLiteral("(no model)"), central);
    path_label_->setWordWrap(true);
    row1->addWidget(load_btn_);
    row1->addWidget(new_model_btn_);
    row1->addWidget(path_label_, 1);
    layout->addLayout(row1);

    auto* row_ff = new QHBoxLayout();
    ff_btn_ = new QPushButton(QStringLiteral("F field JSON…"), central);
    ff_label_ = new QLabel(QStringLiteral("(optional, if world.F_field not in .cypha)"), central);
    ff_label_->setWordWrap(true);
    row_ff->addWidget(ff_btn_);
    row_ff->addWidget(ff_label_, 1);
    layout->addLayout(row_ff);

    auto* row_pre = new QHBoxLayout();
    pre_btn_ = new QPushButton(QStringLiteral("Preprocessor JSON…"), central);
    pre_clear_btn_ = new QPushButton(QStringLiteral("Clear pre"), central);
    fit_pre_btn_ = new QPushButton(QStringLiteral("Fit preprocessor…"), central);
    fit_pre_btn_->setToolTip(QStringLiteral(
        "Fit a new preprocessor (scale + optional PCA) from the loaded CSV.\n"
        "RFF requires Python — use Python toolchain to add RFF weights."));
    pre_label_ = new QLabel(QStringLiteral("(optional, same as cypha_rest --pre)"), central);
    pre_label_->setWordWrap(true);
    row_pre->addWidget(pre_btn_);
    row_pre->addWidget(pre_clear_btn_);
    row_pre->addWidget(fit_pre_btn_);
    row_pre->addWidget(pre_label_, 1);
    layout->addLayout(row_pre);

    // ── Dataset panel ────────────────────────────────────────────────────────
    auto* ds_grp = new QGroupBox(QStringLiteral("Dataset"), central);
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

    layout->addWidget(ds_grp);

    csv_regression_chk_ = new QCheckBox(
        QStringLiteral("CSV target is numeric regression (MKE: send regression_y on /update)"), central);
    layout->addWidget(csv_regression_chk_);

    auto* row_bulk = new QHBoxLayout();
    csv_bulk_train_btn_ = new QPushButton(QStringLiteral("Bulk REST /update"), central);
    csv_bulk_native_btn_ = new QPushButton(QStringLiteral("Bulk native train"), central);
    csv_bulk_native_btn_->setEnabled(false);
    csv_bulk_native_btn_->setToolTip(
        QStringLiteral("Classification CSV only — in-process dif_train_step (needs F_field in .cypha or JSON)"));
    row_bulk->addWidget(csv_bulk_train_btn_);
    row_bulk->addWidget(csv_bulk_native_btn_);
    row_bulk->addWidget(new QLabel(QStringLiteral("max rows"), central));
    csv_bulk_max_rows_spin_ = new QSpinBox(central);
    csv_bulk_max_rows_spin_->setRange(0, 99000000);
    csv_bulk_max_rows_spin_->setValue(0);
    csv_bulk_max_rows_spin_->setToolTip(QStringLiteral("0 = all rows in the CSV"));
    csv_bulk_max_rows_spin_->setMaximumWidth(120);
    row_bulk->addWidget(csv_bulk_max_rows_spin_);
    row_bulk->addStretch(1);
    layout->addLayout(row_bulk);

    auto* row_ru01 = new QHBoxLayout();
    replay_u01_btn_ = new QPushButton(QStringLiteral("replay_u01 JSON…"), central);
    replay_u01_label_ = new QLabel(QStringLiteral("(optional array for REST /update + native replay)"), central);
    replay_u01_label_->setWordWrap(true);
    row_ru01->addWidget(replay_u01_btn_);
    row_ru01->addWidget(replay_u01_label_, 1);
    layout->addLayout(row_ru01);

    auto* row_mke = new QHBoxLayout();
    row_mke->addWidget(new QLabel(QStringLiteral("MKE correct_label"), central));
    mke_correct_label_edit_ = new QLineEdit(central);
    mke_correct_label_edit_->setPlaceholderText(QStringLiteral("empty → first class in model"));
    row_mke->addWidget(mke_correct_label_edit_, 1);
    row_mke->addWidget(new QLabel(QStringLiteral("router_train_label"), central));
    mke_router_label_edit_ = new QLineEdit(central);
    mke_router_label_edit_->setPlaceholderText(QStringLiteral("optional"));
    mke_router_label_edit_->setMaximumWidth(140);
    row_mke->addWidget(mke_router_label_edit_);
    layout->addLayout(row_mke);

    // ── Experiments panel (M6) ────────────────────────────────────────────────
#ifdef CYPHA_SHELL_EXPERIMENT_DB
    {
      auto* exp_grp = new QGroupBox(QStringLiteral("Experiments (M6 — SQLite tracking)"), central);
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

      // Recent runs table
      exp_runs_table_ = new QTableWidget(0, 5, exp_grp);
      exp_runs_table_->setHorizontalHeaderLabels(
          {QStringLiteral("Run ID"), QStringLiteral("Name"), QStringLiteral("Status"),
           QStringLiteral("Acc %"), QStringLiteral("Steps")});
      exp_runs_table_->horizontalHeader()->setStretchLastSection(false);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
      exp_runs_table_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
      exp_runs_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
      exp_runs_table_->setMaximumHeight(140);
      exp_runs_table_->setAlternatingRowColors(true);
      exp_form->addRow(exp_runs_table_);

      layout->addWidget(exp_grp);
    }
#endif  // CYPHA_SHELL_EXPERIMENT_DB

    layout->addWidget(new QLabel(QStringLiteral("Model registry (on-disk layout = Python ModelRegistry):"), central));
    auto* row_reg = new QHBoxLayout();
    reg_root_btn_ = new QPushButton(QStringLiteral("Registry root…"), central);
    reg_root_label_ = new QLabel(QStringLiteral("(none — optional for cypha_rest --registry)"), central);
    reg_root_label_->setWordWrap(true);
    row_reg->addWidget(reg_root_btn_);
    row_reg->addWidget(reg_root_label_, 1);
    layout->addLayout(row_reg);

    auto* row_reg_act = new QHBoxLayout();
    reg_scan_btn_ = new QPushButton(QStringLiteral("Scan"), central);
    reg_load_btn_ = new QPushButton(QStringLiteral("Load selected bundle"), central);
    reg_register_btn_ = new QPushButton(QStringLiteral("Register current…"), central);
    row_reg_act->addWidget(reg_scan_btn_);
    row_reg_act->addWidget(reg_load_btn_);
    row_reg_act->addWidget(reg_register_btn_);
    row_reg_act->addStretch(1);
    layout->addLayout(row_reg_act);

    reg_combo_ = new QComboBox(central);
    reg_combo_->setMinimumContentsLength(28);
    layout->addWidget(reg_combo_);

    auto* row_card = new QHBoxLayout();
    card_btn_ = new QPushButton(QStringLiteral("card.json…"), central);
    card_label_ = new QLabel(QStringLiteral("(optional — for Register; else card.json next to .cypha)"), central);
    card_label_->setWordWrap(true);
    row_card->addWidget(card_btn_);
    row_card->addWidget(card_label_, 1);
    layout->addLayout(row_card);

    dataset_info_ = new QPlainTextEdit(central);
    dataset_info_->setReadOnly(true);
    dataset_info_->setPlaceholderText(QStringLiteral("CSV inspect summary and registry scan notes."));
    dataset_info_->setMaximumBlockCount(80);
    dataset_info_->setMinimumHeight(88);
    layout->addWidget(dataset_info_);

    auto* row_loss = new QHBoxLayout();
    loss_chart_ = new LossChartPanel(central);
    row_loss->addWidget(loss_chart_, 1);
    auto* loss_btn_col = new QVBoxLayout();
    loss_chart_save_btn_ = new QPushButton(QStringLiteral("Save chart PNG…"), central);
    loss_chart_save_btn_->setToolTip(
        QStringLiteral("Raster export — REST (blue) and native (orange) when both bulks were run."));
    loss_btn_col->addWidget(loss_chart_save_btn_);
    loss_csv_save_btn_ = new QPushButton(QStringLiteral("Save loss CSV…"), central);
    loss_csv_save_btn_->setToolTip(QStringLiteral(
        "step, loss_rest, loss_rest_ema, loss_native, loss_native_ema (α=0.08 EMA; blank if no bulk)."));
    loss_btn_col->addWidget(loss_csv_save_btn_);
    loss_svg_save_btn_ = new QPushButton(QStringLiteral("Save chart SVG…"), central);
    loss_svg_save_btn_->setToolTip(
        QStringLiteral("Vector export (embedded polyline — no Qt Svg module required)."));
    loss_btn_col->addWidget(loss_svg_save_btn_);
    loss_chart_clear_btn_ = new QPushButton(QStringLiteral("Clear chart"), central);
    loss_btn_col->addWidget(loss_chart_clear_btn_);
    loss_ema_chk_ = new QCheckBox(QStringLiteral("EMA overlay (α=0.08)"), central);
    loss_ema_chk_->setChecked(true);
    loss_btn_col->addWidget(loss_ema_chk_);
    loss_y_lock_chk_ = new QCheckBox(QStringLiteral("Y lock"), central);
    loss_y_lock_chk_->setToolTip(QStringLiteral("Pin Y axis to manual min/max instead of auto-ranging."));
    loss_btn_col->addWidget(loss_y_lock_chk_);
    auto* yrw = new QHBoxLayout();
    loss_y_min_spin_ = new QDoubleSpinBox(central);
    loss_y_min_spin_->setRange(-1e9, 1e9);
    loss_y_min_spin_->setValue(-10.0);
    loss_y_min_spin_->setDecimals(3);
    loss_y_min_spin_->setToolTip(QStringLiteral("Y axis minimum (active when Y lock is on)"));
    loss_y_min_spin_->setEnabled(false);
    loss_y_max_spin_ = new QDoubleSpinBox(central);
    loss_y_max_spin_->setRange(-1e9, 1e9);
    loss_y_max_spin_->setValue(0.0);
    loss_y_max_spin_->setDecimals(3);
    loss_y_max_spin_->setToolTip(QStringLiteral("Y axis maximum (active when Y lock is on)"));
    loss_y_max_spin_->setEnabled(false);
    yrw->addWidget(new QLabel(QStringLiteral("min"), central));
    yrw->addWidget(loss_y_min_spin_);
    yrw->addWidget(new QLabel(QStringLiteral("max"), central));
    yrw->addWidget(loss_y_max_spin_);
    loss_btn_col->addLayout(yrw);
    loss_btn_col->addStretch(1);
    row_loss->addLayout(loss_btn_col);
    layout->addLayout(row_loss);

    connect(loss_chart_save_btn_, &QPushButton::clicked, this, [this]() {
      const QPixmap pm = loss_chart_->grab();
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

    features_hint_ = new QLabel(QStringLiteral("Features: comma-separated values"), central);
    features_hint_->setWordWrap(true);
    layout->addWidget(features_hint_);

    features_edit_ = new QLineEdit(central);
    features_edit_->setPlaceholderText(QStringLiteral("0,0,0,…"));
    layout->addWidget(features_edit_);

    predict_btn_ = new QPushButton(QStringLiteral("Predict (native)"), central);
    predict_btn_->setEnabled(false);
    layout->addWidget(predict_btn_);

    native_train_one_btn_ = new QPushButton(QStringLiteral("Train 1 row (native, from feature box + correct_label)"),
                                            central);
    native_train_one_btn_->setEnabled(false);
    native_train_one_btn_->setToolTip(
        QStringLiteral("dif_train_step_vector / GH — mutates loaded model in memory (not saved to disk)"));
    layout->addWidget(native_train_one_btn_);

    auto* hp_group = new QGroupBox(QStringLiteral("Native train hyperparameters"), central);
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
    auto* hp_row = new QHBoxLayout();
    hp_apply_btn_ = new QPushButton(QStringLiteral("Apply to native train"), hp_group);
    hp_defaults_btn_ = new QPushButton(QStringLiteral("Reset defaults"), hp_group);
    hp_row->addWidget(hp_apply_btn_);
    hp_row->addWidget(hp_defaults_btn_);
    hp_row->addStretch(1);
    hp_form->addRow(hp_row);
    layout->addWidget(hp_group);

    save_native_btn_ = new QPushButton(QStringLiteral("Save trained model (.cypha)…"), central);
    save_native_btn_->setEnabled(false);
    save_native_btn_->setToolTip(
        QStringLiteral("merge_state_into_root_for_save + infer snapshot: enc/field/temp/context/mid_trans/"
                       "field_W_T/w_inject/scalars — see native/qt/README.md for remaining Python gaps"));
    layout->addWidget(save_native_btn_);

    // ── MKE Regressor panel ───────────────────────────────────────────────────
    {
      auto* mke_grp = new QGroupBox(QStringLiteral("MKE Regressor (native, online — regression mode)"), central);
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

      layout->addWidget(mke_grp);
    }

    // ── Batch predict panel ───────────────────────────────────────────────────
    {
      auto* bp_grp = new QGroupBox(QStringLiteral("Batch predict (native)"), central);
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

      layout->addWidget(bp_grp);
    }

    {
      auto* tlog_hdr = new QHBoxLayout();
      tlog_hdr->addWidget(new QLabel(QStringLiteral("Native train log:"), central));
      tlog_hdr->addStretch(1);
      train_log_clear_btn_ = new QPushButton(QStringLiteral("Clear log"), central);
      train_log_export_btn_ = new QPushButton(QStringLiteral("Export CSV…"), central);
      tlog_hdr->addWidget(train_log_clear_btn_);
      tlog_hdr->addWidget(train_log_export_btn_);
      layout->addLayout(tlog_hdr);
      train_log_table_ = new QTableWidget(0, 4, central);
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
      layout->addWidget(train_log_table_);
    }

    // ── Training progress panel ───────────────────────────────────────────────
    {
      auto* prog_grp = new QGroupBox(QStringLiteral("Training progress (native)"), central);
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

      layout->addWidget(prog_grp);
    }

    layout->addWidget(new QLabel(QStringLiteral("Spawn cypha_rest (optional):"), central));
    auto* row_bin = new QHBoxLayout();
    rest_browse_btn_ = new QPushButton(QStringLiteral("cypha_rest binary…"), central);
    rest_bin_edit_ = new QLineEdit(central);
    rest_bin_edit_->setPlaceholderText(QStringLiteral("path/to/cypha_rest"));
    row_bin->addWidget(rest_browse_btn_);
    row_bin->addWidget(rest_bin_edit_, 1);
    layout->addLayout(row_bin);

    auto* row_listen = new QHBoxLayout();
    row_listen->addWidget(new QLabel(QStringLiteral("--listen"), central));
    rest_listen_edit_ = new QLineEdit(central);
    rest_listen_edit_->setText(QStringLiteral("127.0.0.1:8765"));
    row_listen->addWidget(rest_listen_edit_, 1);
    layout->addLayout(row_listen);

    auto* row_srv = new QHBoxLayout();
    rest_start_btn_ = new QPushButton(QStringLiteral("Start server"), central);
    rest_stop_btn_ = new QPushButton(QStringLiteral("Stop server"), central);
    rest_stop_btn_->setEnabled(false);
    rest_status_label_ = new QLabel(QStringLiteral("Server: stopped"), central);
    row_srv->addWidget(rest_start_btn_);
    row_srv->addWidget(rest_stop_btn_);
    row_srv->addWidget(rest_status_label_, 1);
    layout->addLayout(row_srv);

    auto* row_log_hdr = new QHBoxLayout();
    row_log_hdr->addWidget(new QLabel(QStringLiteral("cypha_rest log:"), central));
    rest_health_btn_ = new QPushButton(QStringLiteral("GET /health"), central);
    rest_ready_btn_ = new QPushButton(QStringLiteral("GET /ready"), central);
    rest_models_btn_ = new QPushButton(QStringLiteral("GET /models"), central);
    rest_clear_log_btn_ = new QPushButton(QStringLiteral("Clear log"), central);
    row_log_hdr->addStretch(1);
    row_log_hdr->addWidget(rest_health_btn_);
    row_log_hdr->addWidget(rest_ready_btn_);
    row_log_hdr->addWidget(rest_models_btn_);
    row_log_hdr->addWidget(rest_clear_log_btn_);
    layout->addLayout(row_log_hdr);

    rest_log_ = new QPlainTextEdit(central);
    rest_log_->setReadOnly(true);
    rest_log_->setMaximumBlockCount(500);
    rest_log_->setPlaceholderText(QStringLiteral("stdout/stderr from spawned cypha_rest appear here."));
    rest_log_->setMinimumHeight(120);
    layout->addWidget(rest_log_);

    layout->addWidget(new QLabel(QStringLiteral("REST base URL (must match --listen):"), central));
    rest_base_edit_ = new QLineEdit(central);
    rest_base_edit_->setPlaceholderText(QStringLiteral("http://127.0.0.1:8765"));
    layout->addWidget(rest_base_edit_);

    use_gh_chk_ = new QCheckBox(QStringLiteral("use_gh for REST predict, /update, and bulk train"), central);
    use_gh_chk_->setChecked(true);
    layout->addWidget(use_gh_chk_);

    predict_return_explanation_chk_ =
        new QCheckBox(QStringLiteral("POST /predict return_explanation (class_details, world distance)"), central);
    predict_return_explanation_chk_->setChecked(false);
    layout->addWidget(predict_return_explanation_chk_);

    auto* row_rest_load = new QHBoxLayout();
    row_rest_load->addWidget(new QLabel(QStringLiteral("POST /load"), central));
    rest_load_name_edit_ = new QLineEdit(central);
    rest_load_name_edit_->setPlaceholderText(QStringLiteral("registry name"));
    rest_load_version_edit_ = new QLineEdit(central);
    rest_load_version_edit_->setPlaceholderText(QStringLiteral("version"));
    rest_load_version_edit_->setText(QStringLiteral("latest"));
    rest_load_version_edit_->setMaximumWidth(100);
    rest_post_load_btn_ = new QPushButton(QStringLiteral("Load"), central);
    row_rest_load->addWidget(rest_load_name_edit_, 1);
    row_rest_load->addWidget(rest_load_version_edit_);
    row_rest_load->addWidget(rest_post_load_btn_);
    layout->addLayout(row_rest_load);

    predict_rest_btn_ = new QPushButton(QStringLiteral("Predict (REST POST /predict)"), central);
    predict_rest_btn_->setEnabled(false);
    layout->addWidget(predict_rest_btn_);

    auto* row_upd = new QHBoxLayout();
    row_upd->addWidget(new QLabel(QStringLiteral("correct_label"), central));
    update_label_edit_ = new QLineEdit(central);
    update_label_edit_->setPlaceholderText(QStringLiteral("class name for POST /update"));
    row_upd->addWidget(update_label_edit_, 1);
    update_rest_btn_ = new QPushButton(QStringLiteral("POST /update (train step)"), central);
    update_rest_btn_->setEnabled(false);
    row_upd->addWidget(update_rest_btn_);
    layout->addLayout(row_upd);

    result_label_ = new QLabel(QStringLiteral("—"), central);
    result_label_->setWordWrap(true);
    layout->addWidget(result_label_);

    setCentralWidget(central);
    resize(700, 1020);

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
      const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Training CSV"), QString(),
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
      QProgressDialog prog(QStringLiteral("POST /update per CSV row…"), QStringLiteral("Cancel"), 0, n, this);
      prog.setWindowModality(Qt::WindowModal);
      prog.setMinimumDuration(0);
      QVector<double> losses;
      losses.reserve(n);
      const QUrl update_url(base + QStringLiteral("/update"));
      const bool use_gh = use_gh_chk_->isChecked();
      QString mke_clab = mke_correct_label_edit_->text().trimmed();
      if (mke_clab.isEmpty()) {
        mke_clab = default_mke_correct_label();
      }
      const QString router_ov = mke_router_label_edit_->text().trimmed();
      for (int i = 0; i < n; ++i) {
        prog.setValue(i);
        QCoreApplication::processEvents();
        if (prog.wasCanceled()) {
          break;
        }
        QJsonArray arr;
        const std::size_t row_base =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(data.n_features);
        for (int j = 0; j < data.n_features; ++j) {
          arr.append(data.x_rowmajor[row_base + static_cast<std::size_t>(j)]);
        }
        QJsonObject body;
        body[QStringLiteral("input")] = arr;
        body[QStringLiteral("use_gh")] = use_gh;
        if (reg_mode) {
          body[QStringLiteral("correct_label")] = mke_clab;
          body[QStringLiteral("regression_y")] = data.y_regression[static_cast<std::size_t>(i)];
          if (!router_ov.isEmpty()) {
            body[QStringLiteral("router_train_label")] = router_ov;
          }
        } else {
          body[QStringLiteral("correct_label")] =
              QString::fromStdString(data.y_class[static_cast<std::size_t>(i)]);
        }
        add_replay_u01_to_json(body);
        const HttpJsonResult r = http_post_json(update_url, body);
        if (!r.ok) {
          QMessageBox::warning(this, QStringLiteral("Bulk /update"), r.err);
          break;
        }
        if (r.obj.contains(QStringLiteral("detail"))) {
          QMessageBox::warning(this, QStringLiteral("Bulk /update"),
                               r.obj.value(QStringLiteral("detail")).toString());
          break;
        }
        losses.append(r.obj.value(QStringLiteral("loss")).toDouble());
      }
      prog.setValue(n);
      apply_losses_to_chart(LossPlotSource::RestBulk, std::move(losses));
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
      } else if (prog.wasCanceled()) {
        result_label_->setText(QStringLiteral("[bulk /update] canceled"));
      }
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
      cypha::CsvDenseResult data;
      try {
        data = cypha::load_csv_dense(qstring_to_fs_path(csv_path_), build_csv_spec());
      } catch (const std::exception& ex) {
        QMessageBox::warning(this, QStringLiteral("Bulk native"), QString::fromUtf8(ex.what()));
        return;
      }
      if (data.y_class.empty()) {
        QMessageBox::information(this, QStringLiteral("Bulk native"),
                                 QStringLiteral("Need a classification CSV with string targets."));
        return;
      }
      if (!native_train_ok_) {
        QMessageBox::warning(this, QStringLiteral("Bulk native"),
                             QStringLiteral("Native training state not ready (need F_field + successful model load)."));
        return;
      }
      int total_n = data.n_rows;
      const int cap = csv_bulk_max_rows_spin_->value();
      if (cap > 0 && cap < total_n) total_n = cap;
      const int val_pct = (val_split_spin_ != nullptr) ? val_split_spin_->value() : 0;
      const int val_n   = (val_pct > 0) ? std::max(1, static_cast<int>(total_n * val_pct / 100.0)) : 0;
      const int train_n = total_n - val_n;
      if (train_n <= 0) {
        QMessageBox::information(this, QStringLiteral("Bulk native"), QStringLiteral("No training rows."));
        return;
      }

      // ── Snapshot all hyperparams before handing off to thread ──────────────
      const bool     use_gh_snap     = use_gh_chk_->isChecked();
      const double   world_lr_snap   = native_world_lr_;
      const double   delta_lr_snap   = native_delta_lr_;
      const double   ood_sigma_snap  = native_ood_sigma_;
      const auto     gh_inv_v_snap   = native_gh_inv_v_;
      const double   gh_R_base_snap  = native_gh_R_base_;
      const auto     tsp_snap        = native_tsp_;
      const auto     replay_u01_snap = replay_u01_cache_;
      int            total_steps_w   = native_total_steps_;
      double         llr_ema_w       = native_llr_ema_;
      double         ema_loss_w      = train_prog_ema_loss_;
      int            win_total_w     = train_prog_win_total_;
      int            win_correct_w   = train_prog_win_correct_;
      double         gh_chi_w        = native_gh_chi_;
      double         gh_psi_w        = native_gh_psi_;
      int            enc_updates_w   = native_enc_updates_;
      std::mt19937   rng_w           = native_rng_;

      // ── Prepare shared state + keep data for val eval ──────────────────────
      bulk_val_n_    = val_n;
      bulk_total_n_  = total_n;
      bulk_train_data_ = data;   // copy — main thread needs it for val eval
      bulk_accum_losses_.clear();
      bulk_accum_log_.clear();

      auto bulk_state = std::make_shared<BulkTrainState>();
      bulk_state_ = bulk_state;

      // ── Disable UI ─────────────────────────────────────────────────────────
      set_bulk_training_ui(true);
      result_label_->setText(QStringLiteral("Training 0 / %1…").arg(train_n));

      // ── Launch background thread ───────────────────────────────────────────
      // Safety: model_, native_mem_, native_replay_, pre_ are unique_ptrs owned
      // by MainWindow. During training all buttons that touch them are disabled.
      // The thread is joined (wait()) before MainWindow destructs.
      bulk_thread_ = QThread::create([
          this, data = std::move(data), bulk_state, train_n,
          use_gh_snap, world_lr_snap, delta_lr_snap, ood_sigma_snap,
          gh_inv_v_snap, gh_R_base_snap, tsp_snap, replay_u01_snap,
          total_steps_w, llr_ema_w, ema_loss_w, win_total_w, win_correct_w,
          gh_chi_w, gh_psi_w, enc_updates_w, rng_w
      ]() mutable {
        double ema_loss    = ema_loss_w;
        int    win_total   = win_total_w;
        int    win_correct = win_correct_w;
        double gh_chi      = gh_chi_w;
        double gh_psi      = gh_psi_w;
        int    enc_updates = enc_updates_w;
        double llr_ema     = llr_ema_w;
        int    total_steps = total_steps_w;
        double ood_sigma   = ood_sigma_snap;

        for (int i = 0; i < train_n; ++i) {
          if (bulk_state->cancel.load(std::memory_order_relaxed)) break;

          std::vector<double> x_raw(static_cast<std::size_t>(data.n_features));
          const std::size_t row_base =
              static_cast<std::size_t>(i) * static_cast<std::size_t>(data.n_features);
          for (int j = 0; j < data.n_features; ++j)
            x_raw[static_cast<std::size_t>(j)] = data.x_rowmajor[row_base + static_cast<std::size_t>(j)];

          std::vector<double> x_lat = x_raw;
          if (pre_ != nullptr) x_lat = pre_->transform_one(x_raw);

          const std::string yl = data.y_class[static_cast<std::size_t>(i)];
          cypha::MemoryTrainMeta meta{};
          double loss = 0.0;

          cypha::TrainStepExtras extras{};
          extras.total_steps = &total_steps;
          extras.ood_sigma   = &ood_sigma;
          extras.llr_ema     = &llr_ema;
          std::vector<double> ru = replay_u01_snap;
          std::size_t ru_pos = 0;
          if (!ru.empty()) {
            extras.replay_u01     = ru.data();
            extras.replay_u01_len = ru.size();
            extras.replay_u01_pos = &ru_pos;
          }

          if (use_gh_snap && static_cast<int>(gh_inv_v_snap.size()) == model_->d_latent) {
            const auto gh = cypha::dif_gh_train_step_vector(
                *model_, *native_mem_, *native_replay_,
                x_lat.data(), model_->d_latent, yl,
                gh_inv_v_snap, gh_R_base_snap, gh_chi, gh_psi,
                kGhNigAdaptAlphaShell, world_lr_snap, delta_lr_snap,
                ood_sigma, tsp_snap, rng_w, enc_updates, &meta, &extras);
            loss   = gh.loss;
            gh_chi = gh.chi_new;
            gh_psi = gh.psi_new;
          } else {
            loss = cypha::dif_train_step_vector(
                *model_, *native_mem_, *native_replay_,
                x_lat.data(), model_->d_latent, yl,
                world_lr_snap, delta_lr_snap, world_lr_snap, delta_lr_snap,
                ood_sigma, tsp_snap, rng_w, enc_updates, &meta, &extras);
          }
          if (meta.correct) model_->total_correct += 1;

          if (win_total < 200) ++win_total;
          win_correct = static_cast<int>(
              win_correct * (win_total == 200 ? 199.0 / 200.0 : 1.0) + (meta.correct ? 1 : 0));
          ema_loss = 0.97 * ema_loss + 0.03 * loss;
          ++total_steps;

          {
            QMutexLocker lock(&bulk_state->steps_mutex);
            bulk_state->new_steps.append({total_steps, QString::fromStdString(yl), loss, meta.correct});
          }
          bulk_state->step_count.fetch_add(1, std::memory_order_relaxed);
        }

        bulk_state->final_total_steps = total_steps;
        bulk_state->final_ema_loss    = ema_loss;
        bulk_state->final_llr_ema     = llr_ema;
        bulk_state->final_win_total   = win_total;
        bulk_state->final_win_correct = win_correct;
        bulk_state->final_gh_chi      = gh_chi;
        bulk_state->final_gh_psi      = gh_psi;
        bulk_state->final_enc_updates = enc_updates;
        bulk_state->done.store(true, std::memory_order_release);
      });

      bulk_poll_timer_ = new QTimer(this);
      connect(bulk_poll_timer_, &QTimer::timeout, this, [this, train_n]() {
        on_bulk_poll(train_n);
      });
      bulk_poll_timer_->start(80);
      bulk_thread_->start();
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
      const int n = last_csv_.n_rows;
      const int d = model_->d_latent;

      QProgressDialog prog(QStringLiteral("MKE bulk train…"), QStringLiteral("Cancel"), 0, n, this);
      prog.setWindowModality(Qt::WindowModal);
      prog.setMinimumDuration(300);

      const double ff     = mke_ff_spin_->value();
      const double pi_flr = mke_pi_spin_->value();
      const QString router_lbl = mke_router_label_edit_ != nullptr
                                     ? mke_router_label_edit_->text().trimmed()
                                     : QString{};

      double sum_err_sq = 0.0;
      int n_ok = 0;

      for (int i = 0; i < n; i++) {
        if (prog.wasCanceled()) break;
        prog.setValue(i);

        const double* xraw = last_csv_.x_rowmajor.data() + static_cast<std::size_t>(i) * last_csv_.n_features;
        std::vector<double> phi(xraw, xraw + last_csv_.n_features);
        if (pre_ != nullptr) {
          phi = pre_->transform_one(phi);
        }
        if (static_cast<int>(phi.size()) != d) {
          if (mke_result_label_ != nullptr)
            mke_result_label_->setText(
                QStringLiteral("Dim mismatch: phi.size=%1 model.d_latent=%2").arg(phi.size()).arg(d));
          break;
        }
        const double y_target = last_csv_.y_regression[static_cast<std::size_t>(i)];
        cypha::TrainStepExtras extras{};
        extras.total_steps = &native_total_steps_;

        const std::string* router_override = nullptr;
        std::string router_str;
        if (!router_lbl.isEmpty()) {
          router_str    = router_lbl.toStdString();
          router_override = &router_str;
        }

        cypha::regression::MkeScalarTrainStepOutputs mke_out{};
        cypha::regression::mke_scalar_train_step_from_phi(
            *model_, *native_mem_, *native_replay_,
            phi.data(), d, y_target,
            mke_w_by_label_, mke_p_by_label_,
            nullptr,  // gh_scales: null → uniform 1
            model_->temperature, ff, pi_flr,
            native_tsp_, native_world_lr_, native_delta_lr_, native_ood_sigma_,
            native_rng_, native_enc_updates_, &extras,
            router_override, 1e-9, &mke_out);

        sum_err_sq += mke_out.err_sq;
        ++n_ok;
      }
      prog.setValue(n);

      if (n_ok > 0) {
        const double mse  = sum_err_sq / n_ok;
        const double rmse = std::sqrt(mse);
        if (mke_result_label_ != nullptr)
          mke_result_label_->setText(
              QStringLiteral("MKE bulk done: %1 steps  MSE=%2  RMSE=%3")
                  .arg(n_ok).arg(mse, 0, 'g', 6).arg(rmse, 0, 'g', 6));
        refresh_train_progress(true);
      }
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
      experiment_refresh_runs_table();
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
      experiment_refresh_runs_table();
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
      dataset_info_->setPlainText(QStringLiteral("registry_scan: %1 bundle(s) under\n%2")
                                      .arg(reg_refs_.size())
                                      .arg(registry_root_));
    });

    connect(reg_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
      if (idx >= 0 && idx < static_cast<int>(reg_refs_.size())) {
        const cypha::RegistryModelRef& r = reg_refs_[static_cast<std::size_t>(idx)];
        rest_load_name_edit_->setText(QString::fromStdString(r.name));
        rest_load_version_edit_->setText(QString::fromStdString(r.version));
      }
    });

    connect(reg_load_btn_, &QPushButton::clicked, this, [this]() {
      const int i = reg_combo_->currentIndex();
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
        const QString guess = fi.absoluteDir().filePath(QStringLiteral("card.json"));
        if (QFile::exists(guess)) {
          card = guess;
        }
      }
      if (card.isEmpty() || !QFile::exists(card)) {
        QMessageBox::warning(this, QStringLiteral("Register"),
                             QStringLiteral("Need card.json — use “card.json…” or place it beside the .cypha."));
        return;
      }
      bool ok_n = false;
      const QString name =
          QInputDialog::getText(this, QStringLiteral("Register"), QStringLiteral("Model name:"),
                                QLineEdit::Normal, QString(), &ok_n);
      if (!ok_n || name.trimmed().isEmpty()) {
        return;
      }
      bool ok_v = false;
      const QString version =
          QInputDialog::getText(this, QStringLiteral("Register"), QStringLiteral("Version:"),
                                QLineEdit::Normal, QStringLiteral("1.0.0"), &ok_v);
      if (!ok_v || version.trimmed().isEmpty()) {
        return;
      }
      const auto ow =
          QMessageBox::question(this, QStringLiteral("Register"),
                                QStringLiteral("Overwrite if %1/%2 already exists?")
                                    .arg(name.trimmed(), version.trimmed()),
                                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
      const bool overwrite = (ow == QMessageBox::Yes);
      std::string root_u8 = registry_root_.toStdString();
      std::string name_u8 = name.trimmed().toStdString();
      std::string ver_u8 = version.trimmed().toStdString();
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
                               QStringLiteral("Registered %1 / %2").arg(name.trimmed(), version.trimmed()));
      reg_refs_ = cypha::registry_scan(registry_root_.toUtf8().constData());
      reg_combo_->clear();
      for (const cypha::RegistryModelRef& r : reg_refs_) {
        reg_combo_->addItem(QStringLiteral("%1 / %2").arg(QString::fromStdString(r.name),
                                                            QString::fromStdString(r.version)));
      }
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
      result_label_->setText(QStringLiteral("[REST /models] entries: %1").arg(models.size()));
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
      const int rc = best_label_and_conf(*model_, pre_.get(), x, &label, &conf);
      if (rc != 0) {
        QMessageBox::warning(this, QStringLiteral("Infer failed"), QStringLiteral("Code %1").arg(rc));
        return;
      }
      result_label_->setText(QStringLiteral("[native] label: %1\nconfidence: %2")
                                 .arg(QString::fromStdString(label))
                                 .arg(conf, 0, 'g', 8));
    });

    // ── Batch predict: load CSV → run native infer on each row ────────────────
    connect(batch_predict_csv_btn_, &QPushButton::clicked, this, [this]() {
      if (!model_) return;
      const QString csv_path = QFileDialog::getOpenFileName(this, QStringLiteral("Batch predict CSV"),
                                                            QString(), QStringLiteral("CSV (*.csv);;All (*)"));
      if (csv_path.isEmpty()) return;

      // Load CSV — no target column required; all numeric columns become features.
      cypha::CsvDenseSpec spec;
      // target_col_index = -1 means last column is target by default, but we still
      // have all data in x_rowmajor minus that one column.  Use x_rowmajor directly.
      cypha::CsvDenseResult mat;
      try {
        mat = cypha::load_csv_dense(qstring_to_fs_path(csv_path), spec);
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
      batch_predict_table_->setRowCount(0);

      const int feat_per_row = mat.n_features;
      for (int i = 0; i < rows; i++) {
        if (prog.wasCanceled()) break;
        prog.setValue(i);

        const double* xraw = mat.x_rowmajor.data() + static_cast<std::size_t>(i) * feat_per_row;
        std::vector<double> xvec(xraw, xraw + feat_per_row);

        std::string lbl;
        double conf = 0.0;
        const int rc = best_label_and_conf(*model_, pre_.get(), xvec, &lbl, &conf);
        if (rc != 0) {
          lbl  = "(error)";
          conf = 0.0;
        }
        batch_predict_results_.push_back({QString::number(i), QString::fromStdString(lbl),
                                          QString::number(conf, 'g', 6)});

        const int trow = batch_predict_table_->rowCount();
        batch_predict_table_->insertRow(trow);
        batch_predict_table_->setItem(trow, 0, new QTableWidgetItem(QString::number(i)));
        batch_predict_table_->setItem(trow, 1, new QTableWidgetItem(QString::fromStdString(lbl)));
        batch_predict_table_->setItem(trow, 2, new QTableWidgetItem(QString::number(conf, 'g', 6)));
      }
      prog.setValue(rows);
      batch_predict_export_btn_->setEnabled(!batch_predict_results_.empty());
      result_label_->setText(QStringLiteral("Batch predict: %1 rows classified").arg(
          static_cast<int>(batch_predict_results_.size())));
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
      const QJsonValue expl = r.obj.value(QStringLiteral("explanation"));
      if (expl.isObject()) {
        lines += QStringLiteral("explanation: (see cypha_rest log above — class_details / world_mu_distance)\n");
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
    // Stop any in-progress bulk training before closing
    if (bulk_state_ != nullptr) bulk_state_->cancel.store(true);
    if (bulk_poll_timer_ != nullptr) { bulk_poll_timer_->stop(); }
    if (bulk_thread_ != nullptr) { bulk_thread_->wait(); }
    if (rest_proc_.state() != QProcess::NotRunning) {
      rest_proc_.terminate();
      rest_proc_.waitForFinished(3000);
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

    auto* info_lbl = new QLabel(dlg);
    auto update_info = [&]() {
      const int pca_d = pca_spin->value();
      const int out_d = (pca_d > 0 && pca_d < n_cols) ? pca_d : n_cols;
      info_lbl->setText(QStringLiteral("input_dim = %1   →   output_dim = %2\n"
                                       "Note: RFF requires Python — add RFF weights after saving.")
                            .arg(n_cols)
                            .arg(out_d));
    };
    update_info();
    connect(pca_spin, QOverload<int>::of(&QSpinBox::valueChanged), dlg, [update_info](int) { update_info(); });
    form->addRow(info_lbl);

    auto* note_lbl = new QLabel(
        QStringLiteral("<i>Fits scale + PCA (when dim > 0) from the %1 rows × %2 cols feature matrix.</i>")
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
      ps.rff_dim = -1;
      ps.seed    = 42;
      try {
        ps.fit_from_design_matrix(last_csv_.x_rowmajor, n_rows, n_cols);
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
    if (csv_bulk_train_btn_  != nullptr) csv_bulk_train_btn_->setEnabled(!training);
    if (native_train_one_btn_ != nullptr) native_train_one_btn_->setEnabled(!training);
    if (save_native_btn_     != nullptr) save_native_btn_->setEnabled(!training);
    if (load_btn_            != nullptr) load_btn_->setEnabled(!training);
  }

  void on_bulk_poll(int train_n) {
    if (bulk_state_ == nullptr) return;

    // Drain new steps from the worker
    QVector<BulkLogEntry> drained;
    {
      QMutexLocker lock(&bulk_state_->steps_mutex);
      drained.swap(bulk_state_->new_steps);
    }
    for (const auto& e : drained) {
      bulk_accum_losses_.append(e.loss);
      bulk_accum_log_.append(e);
    }

    const int done_so_far = bulk_state_->step_count.load(std::memory_order_relaxed);
    result_label_->setText(
        QStringLiteral("Training %1 / %2…").arg(done_so_far).arg(train_n));

    // Live chart refresh every 200 steps
    if (!bulk_accum_losses_.isEmpty() && done_so_far % 200 == 0) {
      apply_losses_to_chart(LossPlotSource::NativeBulk, QVector<double>(bulk_accum_losses_));
    }

    if (bulk_state_->done.load(std::memory_order_acquire)) {
      bulk_poll_timer_->stop();
      bulk_poll_timer_->deleteLater();
      bulk_poll_timer_ = nullptr;
      bulk_thread_->wait();
      bulk_thread_->deleteLater();
      bulk_thread_ = nullptr;
      on_bulk_finish(train_n);
    }
  }

  void on_bulk_finish(int train_n) {
    if (bulk_state_ == nullptr) return;
    const bool cancelled = bulk_state_->cancel.load();

    // Sync final scalar state back to MainWindow
    native_total_steps_     = bulk_state_->final_total_steps;
    native_llr_ema_         = bulk_state_->final_llr_ema;
    train_prog_ema_loss_    = bulk_state_->final_ema_loss;
    train_prog_win_total_   = bulk_state_->final_win_total;
    train_prog_win_correct_ = bulk_state_->final_win_correct;
    native_gh_chi_          = bulk_state_->final_gh_chi;
    native_gh_psi_          = bulk_state_->final_gh_psi;
    native_enc_updates_     = bulk_state_->final_enc_updates;
    bulk_state_.reset();

    // Final chart update
    if (!bulk_accum_losses_.isEmpty()) {
      apply_losses_to_chart(LossPlotSource::NativeBulk, std::move(bulk_accum_losses_));
    }
    bulk_accum_losses_.clear();

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
    if (bulk_val_n_ > 0 && model_ != nullptr) {
      const int val_start = bulk_total_n_ - bulk_val_n_;
      int val_correct = 0;
      for (int i = val_start; i < bulk_total_n_; ++i) {
        std::vector<double> xr(static_cast<std::size_t>(bulk_train_data_.n_features));
        const std::size_t rb =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(bulk_train_data_.n_features);
        for (int j = 0; j < bulk_train_data_.n_features; ++j)
          xr[static_cast<std::size_t>(j)] = bulk_train_data_.x_rowmajor[rb + static_cast<std::size_t>(j)];
        std::string pred; double conf = 0.0;
        if (best_label_and_conf(*model_, pre_.get(), xr, &pred, &conf) == 0 &&
            pred == bulk_train_data_.y_class[static_cast<std::size_t>(i)])
          ++val_correct;
      }
      const double acc = static_cast<double>(val_correct) / static_cast<double>(bulk_val_n_);
      val_suffix = QStringLiteral("  val_acc=%1/%2 (%3%)")
          .arg(val_correct).arg(bulk_val_n_)
          .arg(static_cast<int>(std::round(acc * 100.0)));
      if (csv_stats_label_ != nullptr) {
        const QString base = csv_stats_label_->text().split(QStringLiteral("  val_acc")).first();
        csv_stats_label_->setText(base + val_suffix);
      }
    }

    const int steps_done = native_total_steps_ - (native_total_steps_ - train_n);
    (void)steps_done;  // informational
    const QString status = cancelled
        ? QStringLiteral("Bulk native cancelled.  steps=%1").arg(native_total_steps_)
        : QStringLiteral("Bulk native done.  steps=%1  ema_loss=%.4f")
              .arg(native_total_steps_).arg(train_prog_ema_loss_);
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
  /// `cypha_rest` / `parity_fixtures/train_hparams.json`). Returns whether the file was read successfully.
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

#ifdef CYPHA_SHELL_EXPERIMENT_DB
  void experiment_refresh_runs_table() {
    if (exp_runs_table_ == nullptr || !exp_db_) return;
    exp_runs_table_->setRowCount(0);
    // Query last 30 runs ordered by created_at DESC
    std::vector<cypha::ExperimentDbRunRow> rows;
    std::string err;
    if (!experiment_db_list_runs(*exp_db_,
                                 exp_active_experiment_id_.empty()
                                     ? nullptr
                                     : exp_active_experiment_id_.c_str(),
                                 nullptr,   // status filter
                                 30, 0,     // limit, offset
                                 &rows, &err)) {
      return;
    }
    for (const auto& r : rows) {
      const int row = exp_runs_table_->rowCount();
      exp_runs_table_->insertRow(row);
      const QString run_id_short = QString::fromStdString(r.run_id).right(12);
      exp_runs_table_->setItem(row, 0, new QTableWidgetItem(run_id_short));
      exp_runs_table_->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(r.name)));
      exp_runs_table_->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(r.status)));
      const QString acc_str = (r.accuracy >= 0.0)
                                  ? QString::number(r.accuracy * 100.0, 'f', 1)
                                  : QStringLiteral("—");
      exp_runs_table_->setItem(row, 3, new QTableWidgetItem(acc_str));
      exp_runs_table_->setItem(row, 4, new QTableWidgetItem(QString::number(r.n_steps)));
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
  QPushButton* load_btn_{};
  QPushButton* new_model_btn_{};
  QPushButton* ff_btn_{};
  QPushButton* pre_btn_{};
  QPushButton* pre_clear_btn_{};
  QPushButton* predict_btn_{};
  QPushButton* predict_rest_btn_{};
  QCheckBox* predict_return_explanation_chk_{};
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
  QLabel*      exp_status_label_{};
  QTableWidget* exp_runs_table_{};
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
  // ── Training progress panel ─────────────────────────────────────────────────
  QLabel*              train_prog_label_{};
  QPushButton*         train_prog_reset_btn_{};
  QTableWidget*        train_prog_class_table_{};
  PerClassAccuracyBar* train_prog_acc_bar_{};
  int    train_prog_win_correct_{0};
  int    train_prog_win_total_{0};
  double train_prog_ema_loss_{0.0};
  QSpinBox* csv_bulk_max_rows_spin_{};
  LossChartPanel* loss_chart_{};
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
  QPushButton* reg_load_btn_{};
  QPushButton* reg_register_btn_{};
  QPushButton* card_btn_{};
  QLineEdit* csv_target_name_edit_{};
  QLineEdit* csv_target_index_edit_{};
  QLineEdit* csv_feature_names_edit_{};
  QLabel* csv_label_{};
  QLabel* reg_root_label_{};
  QLabel* card_label_{};
  QComboBox* reg_combo_{};
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
  QThread*                       bulk_thread_{};
  QTimer*                        bulk_poll_timer_{};
  std::shared_ptr<BulkTrainState> bulk_state_;
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
};

}  // namespace

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  QApplication::setApplicationName(QStringLiteral("Cypha Qt shell"));

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
        "  GUI: load .cypha; CSV inspect + bulk REST /update + loss chart; registry;\n"
        "       optional F_field JSON; preprocessor.json; native predict; spawn cypha_rest\n"
        "       (--registry when set); REST /health, /ready, /models, /load, /predict, /update;\n"
        "       native train_step + bulk CSV; save .cypha (merge + infer patch); train hparams UI;\n"
        "       auto-load train_hparams.json beside the .cypha (when present);\n"
        "       loss chart (REST vs native + optional EMA) → PNG / SVG / CSV; Clear chart;\n"
        "       Y lock (manual Y axis min/max for loss chart);\n"
        "       training log table (step, label, loss, correct / EMA — Export CSV);\n"
        "       POST /predict return_explanation + full JSON in cypha_rest log;\n"
        "       optional replay_u01 JSON; regression_y bulk for MKE.\n"
        "  --smoke <path.cypha> [f_field.json]  headless load + zero-vector native predict (CI).\n"
        "  -h, --help            this message\n");
    return 0;
  }

  MainWindow w;
  w.show();
  return app.exec();
}
