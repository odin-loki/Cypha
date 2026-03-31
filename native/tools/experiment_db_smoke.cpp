// M6: apply Python-canonical experiment DDL; insert/update/select; optional on-disk reopen check.
// Usage: experiment_db_smoke <experiment_ddl.sql> [db_path]
//        — omit db_path for :memory:; with db_path, file is removed first, then reopened read-only.
#include "cypha/experiment_db.hpp"

#include <sqlite3.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int usage(const char* argv0) {
  std::fprintf(stderr, "usage: %s <experiment_ddl.sql> [db_path]\n", argv0 ? argv0 : "experiment_db_smoke");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    return usage(argv[0]);
  }

  std::ifstream in(argv[1]);
  if (!in) {
    std::perror(argv[1]);
    return 1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string ddl = ss.str();
  if (ddl.find("CREATE TABLE IF NOT EXISTS experiments") == std::string::npos ||
      ddl.find("CREATE TABLE IF NOT EXISTS runs") == std::string::npos) {
    std::fprintf(stderr, "DDL file missing expected CREATE TABLE statements\n");
    return 1;
  }

  const bool use_file = (argc == 3);
  cypha::ExperimentDb db;
  std::string open_err;
  if (use_file) {
    if (!db.open_file_rw(argv[2], true, &open_err)) {
      std::fprintf(stderr, "open_file_rw: %s\n", open_err.c_str());
      return 1;
    }
  } else {
    if (!db.open_memory(&open_err)) {
      std::fprintf(stderr, "open_memory: %s\n", open_err.c_str());
      return 1;
    }
  }

  std::string ddl_err;
  if (!db.exec(ddl.c_str(), &ddl_err)) {
    std::fprintf(stderr, "sqlite3_exec DDL: %s\n", ddl_err.c_str());
    return 1;
  }

  sqlite3* raw = db.get();
  int rc = SQLITE_OK;

  auto count_query = [&](const char* sql) -> int {
    sqlite3_stmt* st = nullptr;
    rc = sqlite3_prepare_v2(raw, sql, -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      std::fprintf(stderr, "prepare: %s\n", sqlite3_errmsg(raw));
      return -1;
    }
    int n = -1;
    if (sqlite3_step(st) == SQLITE_ROW) {
      n = sqlite3_column_int(st, 0);
    }
    sqlite3_finalize(st);
    return n;
  };

  const int n_tables = count_query(
      "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
      "AND name IN ('experiments','runs')");
  if (n_tables != 2) {
    std::fprintf(stderr, "expected 2 tables (experiments, runs), got %d\n", n_tables);
    return 1;
  }

  const int n_indexes = count_query(
      "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
      "AND name IN ('idx_runs_experiment','idx_runs_status')");
  if (n_indexes != 2) {
    std::fprintf(stderr, "expected 2 indexes (idx_runs_*), got %d\n", n_indexes);
    return 1;
  }

  auto exec_dml = [&](const char* sql) -> bool {
    std::string e;
    if (!db.exec(sql, &e)) {
      std::fprintf(stderr, "sqlite3_exec: %s\n", e.c_str());
      return false;
    }
    return true;
  };

  std::string prag_err;
  if (!db.foreign_keys_on(&prag_err)) {
    std::fprintf(stderr, "PRAGMA foreign_keys: %s\n", prag_err.c_str());
    return 1;
  }
  if (!exec_dml(
          "INSERT INTO experiments (experiment_id,name,description,dataset_name,task,created_at,tags) "
          "VALUES ('exp_smoke','smoke-suite','','ds','classification',1700000000.0,'[]');")) {
    return 1;
  }
  if (!exec_dml(
          "INSERT INTO runs (run_id,experiment_id,name,config,status,created_at,updated_at,tags,notes) "
          "VALUES ('run_smoke01','exp_smoke','run-one','{}','pending',1700000001.0,1700000001.0,'[]','');")) {
    return 1;
  }

  if (count_query("SELECT COUNT(*) FROM experiments WHERE experiment_id='exp_smoke'") != 1) {
    std::fprintf(stderr, "expected 1 experiment row after insert\n");
    return 1;
  }
  if (count_query("SELECT COUNT(*) FROM runs WHERE run_id='run_smoke01'") != 1) {
    std::fprintf(stderr, "expected 1 run row after insert\n");
    return 1;
  }
  if (count_query(
          "SELECT COUNT(*) FROM runs r JOIN experiments e ON r.experiment_id=e.experiment_id "
          "WHERE r.run_id='run_smoke01'") != 1) {
    std::fprintf(stderr, "join experiments/runs failed\n");
    return 1;
  }

  {
    char* err = nullptr;
    const int r = sqlite3_exec(
        raw,
        "INSERT INTO runs (run_id,experiment_id,name,config,status,created_at,updated_at,tags,notes) "
        "VALUES ('run_badfk','missing_exp','x','{}','pending',1.0,1.0,'[]','');",
        nullptr, nullptr, &err);
    if (r == SQLITE_OK) {
      std::fprintf(stderr, "expected FOREIGN KEY constraint failure\n");
      sqlite3_free(err);
      return 1;
    }
    sqlite3_free(err);
  }

  const char* k_metrics_json = R"([{"epoch":0,"loss":0.5}])";
  sqlite3_stmt* up = nullptr;
  rc = sqlite3_prepare_v2(
      raw,
      "UPDATE runs SET status=?, updated_at=?, finished_at=?, duration_s=?, "
      "accuracy=?, macro_f1=?, r2_score=?, rmse=?, n_steps=?, n_classes=?, "
      "checkpoint_path=?, preprocessor_path=?, metrics_history=? "
      "WHERE run_id=?",
      -1, &up, nullptr);
  if (rc != SQLITE_OK) {
    std::fprintf(stderr, "prepare UPDATE: %s\n", sqlite3_errmsg(raw));
    return 1;
  }
  sqlite3_bind_text(up, 1, "done", -1, SQLITE_STATIC);
  sqlite3_bind_double(up, 2, 1700000900.0);
  sqlite3_bind_double(up, 3, 1700000800.0);
  sqlite3_bind_double(up, 4, 42.5);
  sqlite3_bind_double(up, 5, 0.91);
  sqlite3_bind_double(up, 6, 0.88);
  sqlite3_bind_double(up, 7, 0.77);
  sqlite3_bind_double(up, 8, 0.12);
  sqlite3_bind_int(up, 9, 100);
  sqlite3_bind_int(up, 10, 3);
  sqlite3_bind_text(up, 11, "ckpt.cypha", -1, SQLITE_STATIC);
  sqlite3_bind_text(up, 12, "prep.json", -1, SQLITE_STATIC);
  sqlite3_bind_text(up, 13, k_metrics_json, -1, SQLITE_STATIC);
  sqlite3_bind_text(up, 14, "run_smoke01", -1, SQLITE_STATIC);
  rc = sqlite3_step(up);
  if (rc != SQLITE_DONE) {
    std::fprintf(stderr, "UPDATE step: %s\n", sqlite3_errmsg(raw));
    sqlite3_finalize(up);
    return 1;
  }
  if (sqlite3_changes(raw) != 1) {
    std::fprintf(stderr, "expected 1 row updated\n");
    sqlite3_finalize(up);
    return 1;
  }
  sqlite3_finalize(up);

  sqlite3_stmt* sel = nullptr;
  rc = sqlite3_prepare_v2(
      raw,
      "SELECT status, accuracy, macro_f1, r2_score, rmse, n_steps, n_classes, duration_s, "
      "checkpoint_path, preprocessor_path, metrics_history FROM runs WHERE run_id=?",
      -1, &sel, nullptr);
  if (rc != SQLITE_OK) {
    std::fprintf(stderr, "prepare SELECT: %s\n", sqlite3_errmsg(raw));
    return 1;
  }
  sqlite3_bind_text(sel, 1, "run_smoke01", -1, SQLITE_STATIC);
  rc = sqlite3_step(sel);
  if (rc != SQLITE_ROW) {
    std::fprintf(stderr, "SELECT expected one row\n");
    sqlite3_finalize(sel);
    return 1;
  }
  const char* st = reinterpret_cast<const char*>(sqlite3_column_text(sel, 0));
  if (st == nullptr || std::strcmp(st, "done") != 0) {
    std::fprintf(stderr, "status mismatch\n");
    sqlite3_finalize(sel);
    return 1;
  }
  if (std::abs(sqlite3_column_double(sel, 1) - 0.91) > 1e-9 ||
      std::abs(sqlite3_column_double(sel, 2) - 0.88) > 1e-9 ||
      std::abs(sqlite3_column_double(sel, 3) - 0.77) > 1e-9 ||
      std::abs(sqlite3_column_double(sel, 4) - 0.12) > 1e-9) {
    std::fprintf(stderr, "scalar metric mismatch\n");
    sqlite3_finalize(sel);
    return 1;
  }
  if (sqlite3_column_int(sel, 5) != 100 || sqlite3_column_int(sel, 6) != 3) {
    std::fprintf(stderr, "integer metric mismatch\n");
    sqlite3_finalize(sel);
    return 1;
  }
  if (std::abs(sqlite3_column_double(sel, 7) - 42.5) > 1e-9) {
    std::fprintf(stderr, "duration_s mismatch\n");
    sqlite3_finalize(sel);
    return 1;
  }
  const char* ck = reinterpret_cast<const char*>(sqlite3_column_text(sel, 8));
  const char* pp = reinterpret_cast<const char*>(sqlite3_column_text(sel, 9));
  const char* mh = reinterpret_cast<const char*>(sqlite3_column_text(sel, 10));
  if (ck == nullptr || std::strcmp(ck, "ckpt.cypha") != 0 || pp == nullptr ||
      std::strcmp(pp, "prep.json") != 0 || mh == nullptr || std::strcmp(mh, k_metrics_json) != 0) {
    std::fprintf(stderr, "text column mismatch\n");
    sqlite3_finalize(sel);
    return 1;
  }
  sqlite3_finalize(sel);

  db.close();

  if (use_file) {
    cypha::ExperimentDb ro;
    std::string re;
    if (!ro.open_file_readonly(argv[2], &re)) {
      std::fprintf(stderr, "reopen RO: %s\n", re.c_str());
      return 1;
    }
    sqlite3* rdb = ro.get();
    auto count_ro = [&](const char* sql) -> int {
      sqlite3_stmt* st = nullptr;
      int r2 = sqlite3_prepare_v2(rdb, sql, -1, &st, nullptr);
      if (r2 != SQLITE_OK) {
        std::fprintf(stderr, "prepare: %s\n", sqlite3_errmsg(rdb));
        return -1;
      }
      int n = -1;
      if (sqlite3_step(st) == SQLITE_ROW) {
        n = sqlite3_column_int(st, 0);
      }
      sqlite3_finalize(st);
      return n;
    };
    const int n = count_ro("SELECT COUNT(*) FROM runs WHERE run_id='run_smoke01' AND status='done'");
    if (n != 1) {
      std::fprintf(stderr, "reopen: expected persisted run row\n");
      return 1;
    }
  }

  return 0;
}
