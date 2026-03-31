#include "cypha/experiment_db_crud.hpp"

#include "cypha/experiment_db.hpp"

#include <sqlite3.h>

#include <cstdio>
#include <string>
#include <vector>

namespace cypha {

namespace {

bool finalize_stmt_rc(sqlite3* raw, sqlite3_stmt* st, int step_rc, std::string* err_out) {
  if (step_rc != SQLITE_DONE && step_rc != SQLITE_ROW) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    sqlite3_finalize(st);
    return false;
  }
  sqlite3_finalize(st);
  return true;
}

bool trim_trailing_space(std::string* s) {
  if (s == nullptr) {
    return false;
  }
  while (!s->empty()) {
    const char c = s->back();
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
      s->pop_back();
    } else {
      break;
    }
  }
  return true;
}

bool append_json_array_element(const std::string& array_json, const char* elem, std::string* out, std::string* err_out) {
  if (out == nullptr || elem == nullptr) {
    if (err_out != nullptr) {
      *err_out = "append_json_array_element: bad args";
    }
    return false;
  }
  std::string s = array_json;
  trim_trailing_space(&s);
  if (s.empty()) {
    s = "[]";
  }
  if (s.size() < 2U || s.front() != '[' || s.back() != ']') {
    if (err_out != nullptr) {
      *err_out = "metrics_history is not a JSON array";
    }
    return false;
  }
  if (s == "[]") {
    *out = std::string("[") + elem + "]";
    return true;
  }
  *out = s.substr(0, s.size() - 1U) + "," + elem + "]";
  return true;
}

std::string col_text_or_empty(sqlite3_stmt* st, int i) {
  if (sqlite3_column_type(st, i) == SQLITE_NULL) {
    return {};
  }
  const char* t = reinterpret_cast<const char*>(sqlite3_column_text(st, i));
  return t != nullptr ? std::string(t) : std::string{};
}

void fill_experiment_row(sqlite3_stmt* st, ExperimentDbExperimentRow* o) {
  o->experiment_id = col_text_or_empty(st, 0);
  o->name = col_text_or_empty(st, 1);
  o->description = col_text_or_empty(st, 2);
  o->dataset_name = col_text_or_empty(st, 3);
  o->task = col_text_or_empty(st, 4);
  o->created_at = sqlite3_column_double(st, 5);
  o->tags_json = col_text_or_empty(st, 6);
}

void fill_run_row(sqlite3_stmt* st, ExperimentDbRunRow* o) {
  o->run_id = col_text_or_empty(st, 0);
  o->experiment_id = col_text_or_empty(st, 1);
  o->name = col_text_or_empty(st, 2);
  o->config_json = col_text_or_empty(st, 3);
  o->status = col_text_or_empty(st, 4);
  o->created_at = sqlite3_column_double(st, 5);
  o->updated_at = sqlite3_column_double(st, 6);
  if (sqlite3_column_type(st, 7) == SQLITE_NULL) {
    o->finished_at = std::nullopt;
  } else {
    o->finished_at = sqlite3_column_double(st, 7);
  }
  o->duration_s = sqlite3_column_double(st, 8);
  o->accuracy = sqlite3_column_double(st, 9);
  o->macro_f1 = sqlite3_column_double(st, 10);
  o->r2_score = sqlite3_column_double(st, 11);
  o->rmse = sqlite3_column_double(st, 12);
  o->n_steps = sqlite3_column_int(st, 13);
  o->n_classes = sqlite3_column_int(st, 14);
  o->checkpoint_path = col_text_or_empty(st, 15);
  o->preprocessor_path = col_text_or_empty(st, 16);
  o->metrics_history_json = col_text_or_empty(st, 17);
  o->tags_json = col_text_or_empty(st, 18);
  o->notes = col_text_or_empty(st, 19);
}

const char* leaderboard_order_sql(ExperimentDbLeaderboardMetric m) {
  switch (m) {
    case ExperimentDbLeaderboardMetric::kAccuracy:
      return "accuracy DESC, run_id ASC";
    case ExperimentDbLeaderboardMetric::kMacroF1:
      return "macro_f1 DESC, run_id ASC";
    case ExperimentDbLeaderboardMetric::kR2Score:
      return "r2_score DESC, run_id ASC";
    case ExperimentDbLeaderboardMetric::kRmse:
      return "rmse ASC, run_id ASC";
    default:
      return "accuracy DESC, run_id ASC";
  }
}

}  // namespace

bool experiment_db_apply_canonical_schema(ExperimentDb& db, const char* ddl_sql, std::string* err_out) {
  if (!db.exec(ddl_sql, err_out)) {
    return false;
  }
  return db.foreign_keys_on(err_out);
}

bool experiment_db_insert_experiment(ExperimentDb& db, const char* experiment_id, const char* name,
                                     const char* description, const char* dataset_name, const char* task,
                                     double created_at, const char* tags_json, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr) {
    if (err_out != nullptr) {
      *err_out = "db not open";
    }
    return false;
  }
  static const char* kSql =
      "INSERT INTO experiments (experiment_id,name,description,dataset_name,task,created_at,tags) "
      "VALUES (?,?,?,?,?,?,?)";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, experiment_id, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 2, name, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 3, description, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 4, dataset_name, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 5, task, -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(st, 6, created_at);
  sqlite3_bind_text(st, 7, tags_json, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  return finalize_stmt_rc(raw, st, rc, err_out);
}

bool experiment_db_insert_run_pending(ExperimentDb& db, const char* run_id, const char* experiment_id,
                                      const char* run_name, const char* config_json, double created_at,
                                      double updated_at, const char* tags_json, const char* notes,
                                      std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr) {
    if (err_out != nullptr) {
      *err_out = "db not open";
    }
    return false;
  }
  static const char* kSql =
      "INSERT INTO runs (run_id,experiment_id,name,config,status,created_at,updated_at,tags,notes) "
      "VALUES (?,?,?,?,?,?,?,?,?)";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, run_id, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 2, experiment_id, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 3, run_name, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 4, config_json, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 5, "pending", -1, SQLITE_STATIC);
  sqlite3_bind_double(st, 6, created_at);
  sqlite3_bind_double(st, 7, updated_at);
  sqlite3_bind_text(st, 8, tags_json, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 9, notes, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  return finalize_stmt_rc(raw, st, rc, err_out);
}

bool experiment_db_finish_run(ExperimentDb& db, const char* run_id, const char* status, double updated_at,
                              double finished_at, double duration_s, double accuracy, double macro_f1,
                              double r2_score, double rmse, int n_steps, int n_classes,
                              const char* checkpoint_path, const char* preprocessor_path,
                              const char* metrics_history_json, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr) {
    if (err_out != nullptr) {
      *err_out = "db not open";
    }
    return false;
  }
  static const char* kSql =
      "UPDATE runs SET status=?, updated_at=?, finished_at=?, duration_s=?, "
      "accuracy=?, macro_f1=?, r2_score=?, rmse=?, n_steps=?, n_classes=?, "
      "checkpoint_path=?, preprocessor_path=?, metrics_history=? "
      "WHERE run_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, status, -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(st, 2, updated_at);
  sqlite3_bind_double(st, 3, finished_at);
  sqlite3_bind_double(st, 4, duration_s);
  sqlite3_bind_double(st, 5, accuracy);
  sqlite3_bind_double(st, 6, macro_f1);
  sqlite3_bind_double(st, 7, r2_score);
  sqlite3_bind_double(st, 8, rmse);
  sqlite3_bind_int(st, 9, n_steps);
  sqlite3_bind_int(st, 10, n_classes);
  sqlite3_bind_text(st, 11, checkpoint_path, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 12, preprocessor_path, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 13, metrics_history_json, -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(st, 14, run_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (!finalize_stmt_rc(raw, st, rc, err_out)) {
    return false;
  }
  if (sqlite3_changes(raw) != 1) {
    if (err_out != nullptr) {
      *err_out = "finish_run: no row updated";
    }
    return false;
  }
  return true;
}

int experiment_db_scalar_int_query(ExperimentDb& db, const char* count_sql, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || count_sql == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return -1;
  }
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, count_sql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return -1;
  }
  const int rc = sqlite3_step(st);
  int n = -1;
  if (rc == SQLITE_ROW) {
    n = sqlite3_column_int(st, 0);
  } else if (rc != SQLITE_DONE) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    sqlite3_finalize(st);
    return -1;
  }
  sqlite3_finalize(st);
  return n;
}

bool experiment_db_append_metrics_json(ExperimentDb& db, const char* run_id, const char* metrics_object_json,
                                       double updated_at, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || run_id == nullptr || metrics_object_json == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  static const char* kSelect = "SELECT metrics_history FROM runs WHERE run_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSelect, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, run_id, -1, SQLITE_TRANSIENT);
  const int rc_sel = sqlite3_step(st);
  if (rc_sel != SQLITE_ROW) {
    if (err_out != nullptr && rc_sel != SQLITE_DONE) {
      *err_out = sqlite3_errmsg(raw);
    } else if (err_out != nullptr) {
      *err_out = "append_metrics: run not found";
    }
    sqlite3_finalize(st);
    return false;
  }
  const char* prev = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
  const std::string hist_in = (prev != nullptr) ? std::string(prev) : std::string("[]");
  sqlite3_finalize(st);

  std::string hist_out;
  if (!append_json_array_element(hist_in, metrics_object_json, &hist_out, err_out)) {
    return false;
  }

  static const char* kUpdate = "UPDATE runs SET metrics_history=?, updated_at=? WHERE run_id=?";
  if (sqlite3_prepare_v2(raw, kUpdate, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, hist_out.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(st, 2, updated_at);
  sqlite3_bind_text(st, 3, run_id, -1, SQLITE_TRANSIENT);
  const int rc_up = sqlite3_step(st);
  if (!finalize_stmt_rc(raw, st, rc_up, err_out)) {
    return false;
  }
  if (sqlite3_changes(raw) != 1) {
    if (err_out != nullptr) {
      *err_out = "append_metrics: no row updated";
    }
    return false;
  }
  return true;
}

bool experiment_db_fail_run(ExperimentDb& db, const char* run_id, const char* notes_error, double finished_at,
                            double updated_at, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || run_id == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  static const char* kSql =
      "UPDATE runs SET status='failed', notes=?, finished_at=?, updated_at=? WHERE run_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, notes_error != nullptr ? notes_error : "", -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(st, 2, finished_at);
  sqlite3_bind_double(st, 3, updated_at);
  sqlite3_bind_text(st, 4, run_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (!finalize_stmt_rc(raw, st, rc, err_out)) {
    return false;
  }
  if (sqlite3_changes(raw) != 1) {
    if (err_out != nullptr) {
      *err_out = "fail_run: no row updated";
    }
    return false;
  }
  return true;
}

bool experiment_db_delete_run(ExperimentDb& db, const char* run_id, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || run_id == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  static const char* kSql = "DELETE FROM runs WHERE run_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, run_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (!finalize_stmt_rc(raw, st, rc, err_out)) {
    return false;
  }
  if (sqlite3_changes(raw) != 1) {
    if (err_out != nullptr) {
      *err_out = "delete_run: no row deleted";
    }
    return false;
  }
  return true;
}

bool experiment_db_delete_experiment(ExperimentDb& db, const char* experiment_id, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || experiment_id == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  {
    static const char* kDelRuns = "DELETE FROM runs WHERE experiment_id=?";
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(raw, kDelRuns, -1, &st, nullptr) != SQLITE_OK) {
      if (err_out != nullptr) {
        *err_out = sqlite3_errmsg(raw);
      }
      return false;
    }
    sqlite3_bind_text(st, 1, experiment_id, -1, SQLITE_TRANSIENT);
    const int rc = sqlite3_step(st);
    if (!finalize_stmt_rc(raw, st, rc, err_out)) {
      return false;
    }
  }
  {
    static const char* kDelExp = "DELETE FROM experiments WHERE experiment_id=?";
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(raw, kDelExp, -1, &st, nullptr) != SQLITE_OK) {
      if (err_out != nullptr) {
        *err_out = sqlite3_errmsg(raw);
      }
      return false;
    }
    sqlite3_bind_text(st, 1, experiment_id, -1, SQLITE_TRANSIENT);
    const int rc = sqlite3_step(st);
    if (!finalize_stmt_rc(raw, st, rc, err_out)) {
      return false;
    }
  }
  if (sqlite3_changes(raw) != 1) {
    if (err_out != nullptr) {
      *err_out = "delete_experiment: experiment row missing";
    }
    return false;
  }
  return true;
}

bool experiment_db_get_experiment(ExperimentDb& db, const char* experiment_id, ExperimentDbExperimentRow* out,
                                  bool* found, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || experiment_id == nullptr || out == nullptr || found == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  static const char* kSql = "SELECT * FROM experiments WHERE experiment_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, experiment_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (rc == SQLITE_DONE) {
    *found = false;
    sqlite3_finalize(st);
    return true;
  }
  if (rc != SQLITE_ROW) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    sqlite3_finalize(st);
    return false;
  }
  fill_experiment_row(st, out);
  *found = true;
  sqlite3_finalize(st);
  return true;
}

bool experiment_db_get_run(ExperimentDb& db, const char* run_id, ExperimentDbRunRow* out, bool* found,
                           std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || run_id == nullptr || out == nullptr || found == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  static const char* kSql = "SELECT * FROM runs WHERE run_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, run_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (rc == SQLITE_DONE) {
    *found = false;
    sqlite3_finalize(st);
    return true;
  }
  if (rc != SQLITE_ROW) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    sqlite3_finalize(st);
    return false;
  }
  fill_run_row(st, out);
  *found = true;
  sqlite3_finalize(st);
  return true;
}

bool experiment_db_list_experiments(ExperimentDb& db, std::vector<ExperimentDbExperimentRow>* out,
                                    std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || out == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  out->clear();
  static const char* kSql = "SELECT * FROM experiments ORDER BY created_at DESC";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  for (;;) {
    const int rc = sqlite3_step(st);
    if (rc == SQLITE_DONE) {
      break;
    }
    if (rc != SQLITE_ROW) {
      if (err_out != nullptr) {
        *err_out = sqlite3_errmsg(raw);
      }
      sqlite3_finalize(st);
      return false;
    }
    ExperimentDbExperimentRow row{};
    fill_experiment_row(st, &row);
    out->push_back(std::move(row));
  }
  sqlite3_finalize(st);
  return true;
}

bool experiment_db_list_runs(ExperimentDb& db, const char* experiment_id_or_null, const char* status_or_null, int limit,
                             int offset, std::vector<ExperimentDbRunRow>* out, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || out == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  out->clear();
  const int lim = limit < 0 ? 0 : limit;
  const int off = offset < 0 ? 0 : offset;
  std::string sql = "SELECT * FROM runs";
  const bool fil_exp = experiment_id_or_null != nullptr && experiment_id_or_null[0] != '\0';
  const bool fil_st = status_or_null != nullptr && status_or_null[0] != '\0';
  if (fil_exp && fil_st) {
    sql += " WHERE experiment_id=? AND status=?";
  } else if (fil_exp) {
    sql += " WHERE experiment_id=?";
  } else if (fil_st) {
    sql += " WHERE status=?";
  }
  sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  int bi = 1;
  if (fil_exp) {
    sqlite3_bind_text(st, bi++, experiment_id_or_null, -1, SQLITE_TRANSIENT);
  }
  if (fil_st) {
    sqlite3_bind_text(st, bi++, status_or_null, -1, SQLITE_TRANSIENT);
  }
  sqlite3_bind_int(st, bi++, lim);
  sqlite3_bind_int(st, bi++, off);
  for (;;) {
    const int rc = sqlite3_step(st);
    if (rc == SQLITE_DONE) {
      break;
    }
    if (rc != SQLITE_ROW) {
      if (err_out != nullptr) {
        *err_out = sqlite3_errmsg(raw);
      }
      sqlite3_finalize(st);
      return false;
    }
    ExperimentDbRunRow row{};
    fill_run_row(st, &row);
    out->push_back(std::move(row));
  }
  sqlite3_finalize(st);
  return true;
}

bool experiment_db_update_run_status(ExperimentDb& db, const char* run_id, const char* status, double updated_at,
                                     std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || run_id == nullptr || status == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  static const char* kSql = "UPDATE runs SET status=?, updated_at=? WHERE run_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, status, -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(st, 2, updated_at);
  sqlite3_bind_text(st, 3, run_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (!finalize_stmt_rc(raw, st, rc, err_out)) {
    return false;
  }
  if (sqlite3_changes(raw) != 1) {
    if (err_out != nullptr) {
      *err_out = "update_run_status: no row updated";
    }
    return false;
  }
  return true;
}

bool experiment_db_best_done_run(ExperimentDb& db, const char* experiment_id, ExperimentDbLeaderboardMetric metric,
                                 ExperimentDbRunRow* out, bool* found, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || experiment_id == nullptr || out == nullptr || found == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  char buf[512];
  const int n = std::snprintf(buf, sizeof(buf),
                              "SELECT * FROM runs WHERE experiment_id=? AND status='done' ORDER BY %s LIMIT 1",
                              leaderboard_order_sql(metric));
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf)) {
    if (err_out != nullptr) {
      *err_out = "best_done_run: SQL buffer";
    }
    return false;
  }
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, buf, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, experiment_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (rc == SQLITE_DONE) {
    *found = false;
    sqlite3_finalize(st);
    return true;
  }
  if (rc != SQLITE_ROW) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    sqlite3_finalize(st);
    return false;
  }
  fill_run_row(st, out);
  *found = true;
  sqlite3_finalize(st);
  return true;
}

bool experiment_db_leaderboard(ExperimentDb& db, const char* experiment_id, ExperimentDbLeaderboardMetric metric,
                               int top_n, std::vector<ExperimentDbRunRow>* out, std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || experiment_id == nullptr || out == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  out->clear();
  if (top_n <= 0) {
    return true;
  }
  char buf[512];
  const int n = std::snprintf(buf, sizeof(buf),
                              "SELECT * FROM runs WHERE experiment_id=? AND status='done' ORDER BY %s LIMIT ?",
                              leaderboard_order_sql(metric));
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf)) {
    if (err_out != nullptr) {
      *err_out = "leaderboard: SQL buffer";
    }
    return false;
  }
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, buf, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, experiment_id, -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(st, 2, top_n);
  for (;;) {
    const int rc = sqlite3_step(st);
    if (rc == SQLITE_DONE) {
      break;
    }
    if (rc != SQLITE_ROW) {
      if (err_out != nullptr) {
        *err_out = sqlite3_errmsg(raw);
      }
      sqlite3_finalize(st);
      return false;
    }
    ExperimentDbRunRow row{};
    fill_run_row(st, &row);
    out->push_back(std::move(row));
  }
  sqlite3_finalize(st);
  return true;
}

bool experiment_db_compare_runs(ExperimentDb& db, const std::vector<std::string>& run_ids_in_order,
                                std::vector<ExperimentDbRunCompareRow>* out, std::string* err_out) {
  if (out == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  out->clear();
  for (const std::string& rid : run_ids_in_order) {
    ExperimentDbRunRow row{};
    bool found = false;
    if (!experiment_db_get_run(db, rid.c_str(), &row, &found, err_out)) {
      return false;
    }
    if (!found) {
      continue;
    }
    ExperimentDbRunCompareRow c{};
    c.run_id = std::move(row.run_id);
    c.name = std::move(row.name);
    c.status = std::move(row.status);
    c.accuracy = row.accuracy;
    c.macro_f1 = row.macro_f1;
    c.r2_score = row.r2_score;
    c.rmse = row.rmse;
    c.n_steps = row.n_steps;
    c.duration_s = row.duration_s;
    c.config_json = std::move(row.config_json);
    out->push_back(std::move(c));
  }
  return true;
}

bool experiment_db_update_run_notes(ExperimentDb& db, const char* run_id, const char* notes, double updated_at,
                                    std::string* err_out) {
  sqlite3* raw = db.get();
  if (raw == nullptr || run_id == nullptr) {
    if (err_out != nullptr) {
      *err_out = "bad args";
    }
    return false;
  }
  static const char* kSql = "UPDATE runs SET notes=?, updated_at=? WHERE run_id=?";
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(raw, kSql, -1, &st, nullptr) != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = sqlite3_errmsg(raw);
    }
    return false;
  }
  sqlite3_bind_text(st, 1, notes != nullptr ? notes : "", -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(st, 2, updated_at);
  sqlite3_bind_text(st, 3, run_id, -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(st);
  if (!finalize_stmt_rc(raw, st, rc, err_out)) {
    return false;
  }
  if (sqlite3_changes(raw) != 1) {
    if (err_out != nullptr) {
      *err_out = "update_run_notes: no row updated";
    }
    return false;
  }
  return true;
}

}  // namespace cypha
