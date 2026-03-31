// CTest: native experiment_db_crud vs canonical DDL (Python ExperimentDB-compatible inserts + finish_run).
// Usage: experiment_db_crud_parity <experiment_ddl.sql>
#include "cypha/experiment_db.hpp"
#include "cypha/experiment_db_crud.hpp"

#include <sqlite3.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int usage(const char* argv0) {
  std::fprintf(stderr, "usage: %s <experiment_ddl.sql>\n", argv0 ? argv0 : "experiment_db_crud_parity");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
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

  cypha::ExperimentDb db;
  std::string err;
  if (!db.open_memory(&err)) {
    std::fprintf(stderr, "open_memory: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_apply_canonical_schema(db, ddl.c_str(), &err)) {
    std::fprintf(stderr, "schema: %s\n", err.c_str());
    return 1;
  }

  {
    bool found_miss = false;
    cypha::ExperimentDbExperimentRow miss{};
    if (!cypha::experiment_db_get_experiment(db, "no_such_experiment", &miss, &found_miss, &err)) {
      std::fprintf(stderr, "get_experiment missing: %s\n", err.c_str());
      return 1;
    }
    if (found_miss) {
      std::fprintf(stderr, "expected no_such_experiment absent\n");
      return 1;
    }
  }

  const double t0 = 1700001000.0;
  if (!cypha::experiment_db_insert_experiment(db, "crud_ex", "parity-crud", "desc", "ds", "classification", t0,
                                                "[]", &err)) {
    std::fprintf(stderr, "insert_experiment: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_insert_run_pending(db, "run_crud_01", "crud_ex", "run-a", "{}", t0 + 1.0, t0 + 1.0,
                                               "[]", "", &err)) {
    std::fprintf(stderr, "insert_run: %s\n", err.c_str());
    return 1;
  }

  int n_ex = cypha::experiment_db_scalar_int_query(db, "SELECT COUNT(*) FROM experiments WHERE experiment_id='crud_ex'", &err);
  if (n_ex != 1) {
    std::fprintf(stderr, "expected 1 experiment, got %d (%s)\n", n_ex, err.c_str());
    return 1;
  }

  {
    bool gf = false;
    cypha::ExperimentDbExperimentRow er{};
    if (!cypha::experiment_db_get_experiment(db, "crud_ex", &er, &gf, &err) || !gf) {
      std::fprintf(stderr, "get_experiment crud_ex: %s\n", err.c_str());
      return 1;
    }
    if (er.name != "parity-crud" || er.dataset_name != "ds") {
      std::fprintf(stderr, "get_experiment field mismatch\n");
      return 1;
    }
    bool rf = false;
    cypha::ExperimentDbRunRow rr{};
    if (!cypha::experiment_db_get_run(db, "run_crud_01", &rr, &rf, &err) || !rf) {
      std::fprintf(stderr, "get_run run_crud_01: %s\n", err.c_str());
      return 1;
    }
    if (rr.status != "pending" || rr.finished_at.has_value()) {
      std::fprintf(stderr, "get_run pending shape mismatch\n");
      return 1;
    }
  }
  if (!cypha::experiment_db_update_run_status(db, "run_crud_01", "running", t0 + 2.0, &err)) {
    std::fprintf(stderr, "update_run_status: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_insert_run_pending(db, "run_b_newer", "crud_ex", "run-b", "{}", t0 + 50.0, t0 + 50.0,
                                                 "[]", "", &err)) {
    std::fprintf(stderr, "insert run_b_newer: %s\n", err.c_str());
    return 1;
  }
  {
    std::vector<cypha::ExperimentDbRunRow> lr;
    if (!cypha::experiment_db_list_runs(db, "crud_ex", nullptr, 100, 0, &lr, &err)) {
      std::fprintf(stderr, "list_runs: %s\n", err.c_str());
      return 1;
    }
    if (lr.size() != 2U || lr[0].run_id != "run_b_newer" || lr[1].run_id != "run_crud_01") {
      std::fprintf(stderr, "list_runs order or count wrong (size=%zu)\n", lr.size());
      return 1;
    }
  }
  int n_ru = cypha::experiment_db_scalar_int_query(
      db, "SELECT COUNT(*) FROM runs WHERE run_id='run_crud_01' AND status='running'", &err);
  if (n_ru != 1) {
    std::fprintf(stderr, "expected run_crud_01 running, got %d (%s)\n", n_ru, err.c_str());
    return 1;
  }
  int n_pending_b = cypha::experiment_db_scalar_int_query(
      db, "SELECT COUNT(*) FROM runs WHERE run_id='run_b_newer' AND status='pending'", &err);
  if (n_pending_b != 1) {
    std::fprintf(stderr, "expected run_b_newer pending, got %d (%s)\n", n_pending_b, err.c_str());
    return 1;
  }

  if (!cypha::experiment_db_update_run_notes(db, "run_b_newer", "parity-note", t0 + 52.0, &err)) {
    std::fprintf(stderr, "update_run_notes: %s\n", err.c_str());
    return 1;
  }
  {
    bool nf = false;
    cypha::ExperimentDbRunRow nr{};
    if (!cypha::experiment_db_get_run(db, "run_b_newer", &nr, &nf, &err) || !nf || nr.notes != "parity-note") {
      std::fprintf(stderr, "get_run after update_run_notes mismatch\n");
      return 1;
    }
  }

  const char* k_metrics = R"([{"epoch":0,"loss":0.25}])";
  if (!cypha::experiment_db_finish_run(db, "run_crud_01", "done", 1700002000.0, 1700001900.0, 900.0, 0.93, 0.90,
                                       0.80, 0.11, 50, 4, "m.cypha", "p.json", k_metrics, &err)) {
    std::fprintf(stderr, "finish_run: %s\n", err.c_str());
    return 1;
  }

  int n_done = cypha::experiment_db_scalar_int_query(
      db, "SELECT COUNT(*) FROM runs WHERE run_id='run_crud_01' AND status='done'", &err);
  if (n_done != 1) {
    std::fprintf(stderr, "expected 1 done run, got %d (%s)\n", n_done, err.c_str());
    return 1;
  }

  sqlite3_stmt* st = nullptr;
  sqlite3* raw = db.get();
  if (sqlite3_prepare_v2(raw, "SELECT accuracy, macro_f1, metrics_history FROM runs WHERE run_id='run_crud_01'", -1,
                           &st, nullptr) != SQLITE_OK) {
    std::fprintf(stderr, "prepare verify: %s\n", sqlite3_errmsg(raw));
    return 1;
  }
  if (sqlite3_step(st) != SQLITE_ROW) {
    std::fprintf(stderr, "verify SELECT no row\n");
    sqlite3_finalize(st);
    return 1;
  }
  if (std::abs(sqlite3_column_double(st, 0) - 0.93) > 1e-9 || std::abs(sqlite3_column_double(st, 1) - 0.90) > 1e-9) {
    std::fprintf(stderr, "metric mismatch\n");
    sqlite3_finalize(st);
    return 1;
  }
  const char* mh = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
  if (mh == nullptr || std::strcmp(mh, k_metrics) != 0) {
    std::fprintf(stderr, "metrics_history mismatch\n");
    sqlite3_finalize(st);
    return 1;
  }
  sqlite3_finalize(st);

  const char* k_metrics2 = R"({"step":51,"accuracy":0.94})";
  const char* k_metrics_both =
      R"([{"epoch":0,"loss":0.25},{"step":51,"accuracy":0.94}])";
  if (!cypha::experiment_db_append_metrics_json(db, "run_crud_01", k_metrics2, 1700002100.0, &err)) {
    std::fprintf(stderr, "append_metrics: %s\n", err.c_str());
    return 1;
  }
  if (sqlite3_prepare_v2(raw, "SELECT metrics_history FROM runs WHERE run_id='run_crud_01'", -1, &st, nullptr) !=
      SQLITE_OK) {
    std::fprintf(stderr, "prepare after append: %s\n", sqlite3_errmsg(raw));
    return 1;
  }
  if (sqlite3_step(st) != SQLITE_ROW) {
    std::fprintf(stderr, "append verify: no row\n");
    sqlite3_finalize(st);
    return 1;
  }
  {
    const char* m2 = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
    if (m2 == nullptr || std::strcmp(m2, k_metrics_both) != 0) {
      std::fprintf(stderr, "append metrics_history mismatch\n");
      sqlite3_finalize(st);
      return 1;
    }
  }
  sqlite3_finalize(st);

  if (!cypha::experiment_db_insert_run_pending(db, "run_low_acc", "crud_ex", "low", "{}", t0 + 60.0, t0 + 60.0, "[]", "",
                                                 &err)) {
    std::fprintf(stderr, "insert run_low_acc: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_finish_run(db, "run_low_acc", "done", 1700002050.0, 1700002040.0, 40.0, 0.10, 0.05, 0.02,
                                       0.99, 2, 2, "", "", "[]", &err)) {
    std::fprintf(stderr, "finish_run run_low_acc: %s\n", err.c_str());
    return 1;
  }
  {
    bool bf = false;
    cypha::ExperimentDbRunRow best{};
    if (!cypha::experiment_db_best_done_run(db, "crud_ex", cypha::ExperimentDbLeaderboardMetric::kAccuracy, &best, &bf,
                                            &err) ||
        !bf) {
      std::fprintf(stderr, "best_done_run: %s\n", err.c_str());
      return 1;
    }
    if (best.run_id != "run_crud_01" || std::abs(best.accuracy - 0.93) > 1e-9) {
      std::fprintf(stderr, "best_done_run wrong row\n");
      return 1;
    }
    std::vector<cypha::ExperimentDbRunRow> board;
    if (!cypha::experiment_db_leaderboard(db, "crud_ex", cypha::ExperimentDbLeaderboardMetric::kAccuracy, 10, &board,
                                          &err)) {
      std::fprintf(stderr, "leaderboard: %s\n", err.c_str());
      return 1;
    }
    if (board.size() < 2U || board[0].run_id != "run_crud_01" || board[1].run_id != "run_low_acc") {
      std::fprintf(stderr, "leaderboard order wrong (size=%zu)\n", board.size());
      return 1;
    }
  }

  {
    const std::vector<std::string> ids = {"run_low_acc", "does_not_exist", "run_crud_01"};
    std::vector<cypha::ExperimentDbRunCompareRow> cmp;
    if (!cypha::experiment_db_compare_runs(db, ids, &cmp, &err)) {
      std::fprintf(stderr, "compare_runs: %s\n", err.c_str());
      return 1;
    }
    if (cmp.size() != 2U || cmp[0].run_id != "run_low_acc" || cmp[1].run_id != "run_crud_01") {
      std::fprintf(stderr, "compare_runs order or count wrong (size=%zu)\n", cmp.size());
      return 1;
    }
    if (std::abs(cmp[0].accuracy - 0.10) > 1e-9 || std::abs(cmp[1].accuracy - 0.93) > 1e-9) {
      std::fprintf(stderr, "compare_runs accuracy mismatch\n");
      return 1;
    }
  }

  if (!cypha::experiment_db_insert_experiment(db, "tmp_fail", "f", "", "", "classification", t0 + 10.0, "[]", &err)) {
    std::fprintf(stderr, "insert tmp_fail exp: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_insert_run_pending(db, "run_fail_01", "tmp_fail", "rf", "{}", t0 + 11.0, t0 + 11.0, "[]",
                                                 "", &err)) {
    std::fprintf(stderr, "insert run_fail: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_fail_run(db, "run_fail_01", "boom", 1700002200.0, 1700002200.0, &err)) {
    std::fprintf(stderr, "fail_run: %s\n", err.c_str());
    return 1;
  }
  int n_fail = cypha::experiment_db_scalar_int_query(
      db, "SELECT COUNT(*) FROM runs WHERE run_id='run_fail_01' AND status='failed' AND notes='boom'", &err);
  if (n_fail != 1) {
    std::fprintf(stderr, "expected 1 failed run, got %d (%s)\n", n_fail, err.c_str());
    return 1;
  }

  if (!cypha::experiment_db_insert_experiment(db, "gone_ex", "g", "", "", "classification", t0 + 20.0, "[]", &err)) {
    std::fprintf(stderr, "insert gone_ex: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_insert_run_pending(db, "run_gone", "gone_ex", "x", "{}", t0 + 21.0, t0 + 21.0, "[]", "",
                                                 &err)) {
    std::fprintf(stderr, "insert run_gone: %s\n", err.c_str());
    return 1;
  }
  {
    std::vector<cypha::ExperimentDbExperimentRow> le;
    if (!cypha::experiment_db_list_experiments(db, &le, &err)) {
      std::fprintf(stderr, "list_experiments: %s\n", err.c_str());
      return 1;
    }
    if (le.size() != 3U || le[0].experiment_id != "gone_ex" || le[1].experiment_id != "tmp_fail" ||
        le[2].experiment_id != "crud_ex") {
      std::fprintf(stderr, "list_experiments order or count wrong (size=%zu)\n", le.size());
      return 1;
    }
  }
  if (!cypha::experiment_db_delete_run(db, "run_gone", &err)) {
    std::fprintf(stderr, "delete_run: %s\n", err.c_str());
    return 1;
  }
  int n_gone_runs =
      cypha::experiment_db_scalar_int_query(db, "SELECT COUNT(*) FROM runs WHERE experiment_id='gone_ex'", &err);
  if (n_gone_runs != 0) {
    std::fprintf(stderr, "expected 0 runs on gone_ex, got %d\n", n_gone_runs);
    return 1;
  }
  if (!cypha::experiment_db_delete_experiment(db, "gone_ex", &err)) {
    std::fprintf(stderr, "delete_experiment gone_ex: %s\n", err.c_str());
    return 1;
  }
  int n_gone_ex = cypha::experiment_db_scalar_int_query(db, "SELECT COUNT(*) FROM experiments WHERE experiment_id='gone_ex'", &err);
  if (n_gone_ex != 0) {
    std::fprintf(stderr, "gone_ex still present\n");
    return 1;
  }

  if (!cypha::experiment_db_delete_experiment(db, "tmp_fail", &err)) {
    std::fprintf(stderr, "delete_experiment tmp_fail: %s\n", err.c_str());
    return 1;
  }
  if (!cypha::experiment_db_delete_experiment(db, "crud_ex", &err)) {
    std::fprintf(stderr, "delete_experiment crud_ex: %s\n", err.c_str());
    return 1;
  }
  int n_all_e = cypha::experiment_db_scalar_int_query(db, "SELECT COUNT(*) FROM experiments", &err);
  int n_all_r = cypha::experiment_db_scalar_int_query(db, "SELECT COUNT(*) FROM runs", &err);
  if (n_all_e != 0 || n_all_r != 0) {
    std::fprintf(stderr, "expected empty DB, experiments=%d runs=%d\n", n_all_e, n_all_r);
    return 1;
  }

  std::printf("experiment_db_crud_parity OK\n");
  return 0;
}
