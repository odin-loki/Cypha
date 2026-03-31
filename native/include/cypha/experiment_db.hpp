#pragma once

/// SQLite session for the canonical experiments schema (``EXPERIMENTS_SCHEMA.md`` / Python ``ExperimentDB``).
/// Linked only from ``experiment_db_smoke`` — not part of ``cypha_core`` (no SQLite dependency there).
///
/// Low-level helpers ``experiment_sqlite_exec`` / ``experiment_sqlite_pragma_foreign_keys_on`` remain available
/// for ad-hoc SQL; ``ExperimentDb`` handles connection lifecycle.

#include <string>

struct sqlite3;

namespace cypha {

bool experiment_sqlite_exec(sqlite3* db, const char* sql, std::string* err_out);
bool experiment_sqlite_pragma_foreign_keys_on(sqlite3* db, std::string* err_out);

class ExperimentDb {
 public:
  ExperimentDb() = default;
  ~ExperimentDb() { close(); }
  ExperimentDb(const ExperimentDb&) = delete;
  ExperimentDb& operator=(const ExperimentDb&) = delete;
  ExperimentDb(ExperimentDb&& o) noexcept : db_(o.db_) { o.db_ = nullptr; }
  ExperimentDb& operator=(ExperimentDb&& o) noexcept {
    if (this != &o) {
      close();
      db_ = o.db_;
      o.db_ = nullptr;
    }
    return *this;
  }

  /// In-memory database (``:memory:``).
  bool open_memory(std::string* err_out);
  /// On-disk RW+create. If ``remove_existing_first``, deletes the file before open (parity with smoke tool).
  bool open_file_rw(const char* path, bool remove_existing_first, std::string* err_out);
  /// Read-only open (e.g. verify persistence after close).
  bool open_file_readonly(const char* path, std::string* err_out);
  void close();

  bool exec(const char* sql, std::string* err_out) { return experiment_sqlite_exec(db_, sql, err_out); }
  bool foreign_keys_on(std::string* err_out) { return experiment_sqlite_pragma_foreign_keys_on(db_, err_out); }

  sqlite3* get() const { return db_; }
  explicit operator bool() const { return db_ != nullptr; }

 private:
  sqlite3* db_{nullptr};
};

}  // namespace cypha
