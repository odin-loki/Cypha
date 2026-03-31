#pragma once

/// CRUD-style operations on the canonical experiments SQLite schema (``EXPERIMENTS_SCHEMA.md``).
/// Requires an open ``ExperimentDb`` and (typically) ``experiment_db_apply_canonical_schema`` first.
/// Linked from ``experiment_db_crud_parity`` only — not ``cypha_core``.

#include <optional>
#include <string>
#include <vector>

namespace cypha {

class ExperimentDb;

/// One ``experiments`` row (Python ``Experiment`` / ``SELECT * FROM experiments`` order).
struct ExperimentDbExperimentRow {
  std::string experiment_id;
  std::string name;
  std::string description;
  std::string dataset_name;
  std::string task;
  double created_at{};
  std::string tags_json;
};

/// One ``runs`` row (Python ``Run`` / ``SELECT * FROM runs`` order).
struct ExperimentDbRunRow {
  std::string run_id;
  std::string experiment_id;
  std::string name;
  std::string config_json;
  std::string status;
  double created_at{};
  double updated_at{};
  std::optional<double> finished_at;
  double duration_s{};
  double accuracy{};
  double macro_f1{};
  double r2_score{};
  double rmse{};
  int n_steps{};
  int n_classes{};
  std::string checkpoint_path;
  std::string preprocessor_path;
  std::string metrics_history_json;
  std::string tags_json;
  std::string notes;
};

enum class ExperimentDbLeaderboardMetric { kAccuracy, kMacroF1, kR2Score, kRmse };

/// Python ``compare_runs`` summary row (``r2`` → ``r2_score``).
struct ExperimentDbRunCompareRow {
  std::string run_id;
  std::string name;
  std::string status;
  double accuracy{};
  double macro_f1{};
  double r2_score{};
  double rmse{};
  int n_steps{};
  double duration_s{};
  std::string config_json;
};

/// Apply exported DDL (full ``_SCHEMA`` script) + ``PRAGMA foreign_keys=ON``.
bool experiment_db_apply_canonical_schema(ExperimentDb& db, const char* ddl_sql, std::string* err_out);

/// Insert one ``experiments`` row (same columns as Python ``create_experiment`` / implicit order).
bool experiment_db_insert_experiment(ExperimentDb& db, const char* experiment_id, const char* name,
                                     const char* description, const char* dataset_name, const char* task,
                                     double created_at, const char* tags_json, std::string* err_out);

/// Insert a ``pending`` run (``finished_at`` NULL; other numerics at SQLite defaults).
bool experiment_db_insert_run_pending(ExperimentDb& db, const char* run_id, const char* experiment_id,
                                      const char* run_name, const char* config_json, double created_at,
                                      double updated_at, const char* tags_json, const char* notes,
                                      std::string* err_out);

/// Update run to terminal state + scalar metrics (matches ``experiment_db_smoke`` / Python ``finish_run`` tail).
bool experiment_db_finish_run(ExperimentDb& db, const char* run_id, const char* status, double updated_at,
                              double finished_at, double duration_s, double accuracy, double macro_f1,
                              double r2_score, double rmse, int n_steps, int n_classes,
                              const char* checkpoint_path, const char* preprocessor_path,
                              const char* metrics_history_json, std::string* err_out);

/// ``SELECT COUNT(*)`` for static audit SQL (callers must pass fixed SQL).
int experiment_db_scalar_int_query(ExperimentDb& db, const char* count_sql, std::string* err_out);

/// Read ``metrics_history``, append one JSON object (no surrounding ``[``/``]``), write back — same pattern as Python ``log_metrics``.
bool experiment_db_append_metrics_json(ExperimentDb& db, const char* run_id, const char* metrics_object_json,
                                       double updated_at, std::string* err_out);

/// Match Python ``fail_run`` (status ``failed``, notes, timestamps).
bool experiment_db_fail_run(ExperimentDb& db, const char* run_id, const char* notes_error, double finished_at,
                            double updated_at, std::string* err_out);

bool experiment_db_delete_run(ExperimentDb& db, const char* run_id, std::string* err_out);

/// Delete all runs for the experiment, then the experiment row (Python ``delete_experiment`` order).
bool experiment_db_delete_experiment(ExperimentDb& db, const char* experiment_id, std::string* err_out);

/// ``get_experiment`` / ``get_run`` — on success, ``*found`` tells whether a row exists (``out`` valid only if ``*found``).
bool experiment_db_get_experiment(ExperimentDb& db, const char* experiment_id, ExperimentDbExperimentRow* out,
                                  bool* found, std::string* err_out);
bool experiment_db_get_run(ExperimentDb& db, const char* run_id, ExperimentDbRunRow* out, bool* found,
                           std::string* err_out);

/// Python ``list_experiments`` — ``ORDER BY created_at DESC``.
bool experiment_db_list_experiments(ExperimentDb& db, std::vector<ExperimentDbExperimentRow>* out,
                                      std::string* err_out);

/// Python ``list_runs`` — optional ``experiment_id`` / ``status`` filter, ``ORDER BY created_at DESC``, paginated.
bool experiment_db_list_runs(ExperimentDb& db, const char* experiment_id_or_null, const char* status_or_null,
                             int limit, int offset, std::vector<ExperimentDbRunRow>* out, std::string* err_out);

/// Common ``update_run`` slice: status + ``updated_at`` only.
bool experiment_db_update_run_status(ExperimentDb& db, const char* run_id, const char* status, double updated_at,
                                       std::string* err_out);

/// Best ``done`` run by metric (``rmse`` uses ascending order; accuracy / macro_f1 / r2_score descending). Tie-break ``run_id``.
bool experiment_db_best_done_run(ExperimentDb& db, const char* experiment_id, ExperimentDbLeaderboardMetric metric,
                                 ExperimentDbRunRow* out, bool* found, std::string* err_out);

/// Python ``leaderboard`` — ``done`` runs only, same ordering as ``best_done_run``, capped at ``top_n`` (``<= 0`` → empty ``out``).
bool experiment_db_leaderboard(ExperimentDb& db, const char* experiment_id, ExperimentDbLeaderboardMetric metric,
                                 int top_n, std::vector<ExperimentDbRunRow>* out, std::string* err_out);

/// Python ``compare_runs``: preserve ``run_ids`` order; omit missing ids (no error).
bool experiment_db_compare_runs(ExperimentDb& db, const std::vector<std::string>& run_ids_in_order,
                                std::vector<ExperimentDbRunCompareRow>* out, std::string* err_out);

/// ``update_run`` slice: ``notes`` + ``updated_at`` only.
bool experiment_db_update_run_notes(ExperimentDb& db, const char* run_id, const char* notes, double updated_at,
                                    std::string* err_out);

}  // namespace cypha
