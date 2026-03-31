#include "cypha/experiment_db.hpp"

#include <cstdio>
#include <sqlite3.h>

namespace cypha {

bool experiment_sqlite_exec(sqlite3* db, const char* sql, std::string* err_out) {
  if (db == nullptr || sql == nullptr) {
    if (err_out != nullptr) {
      *err_out = "null db or sql";
    }
    return false;
  }
  char* errmsg = nullptr;
  const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errmsg);
  if (rc != SQLITE_OK) {
    if (err_out != nullptr) {
      *err_out = errmsg != nullptr ? errmsg : sqlite3_errmsg(db);
    }
    sqlite3_free(errmsg);
    return false;
  }
  return true;
}

bool experiment_sqlite_pragma_foreign_keys_on(sqlite3* db, std::string* err_out) {
  return experiment_sqlite_exec(db, "PRAGMA foreign_keys=ON;", err_out);
}

void ExperimentDb::close() {
  if (db_ != nullptr) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
}

bool ExperimentDb::open_memory(std::string* err_out) {
  close();
  int rc = sqlite3_open_v2(":memory:", &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
  if (rc != SQLITE_OK || db_ == nullptr) {
    if (err_out != nullptr) {
      *err_out = db_ != nullptr ? sqlite3_errmsg(db_) : "sqlite3_open_v2 failed";
    }
    if (db_ != nullptr) {
      sqlite3_close(db_);
      db_ = nullptr;
    }
    return false;
  }
  return true;
}

bool ExperimentDb::open_file_rw(const char* path, bool remove_existing_first, std::string* err_out) {
  close();
  if (path == nullptr) {
    if (err_out != nullptr) {
      *err_out = "null path";
    }
    return false;
  }
  if (remove_existing_first) {
    (void)std::remove(path);
  }
  const int rc =
      sqlite3_open_v2(path, &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
  if (rc != SQLITE_OK || db_ == nullptr) {
    if (err_out != nullptr) {
      *err_out = db_ != nullptr ? sqlite3_errmsg(db_) : "sqlite3_open_v2 failed";
    }
    if (db_ != nullptr) {
      sqlite3_close(db_);
      db_ = nullptr;
    }
    return false;
  }
  return true;
}

bool ExperimentDb::open_file_readonly(const char* path, std::string* err_out) {
  close();
  if (path == nullptr) {
    if (err_out != nullptr) {
      *err_out = "null path";
    }
    return false;
  }
  const int rc = sqlite3_open_v2(path, &db_, SQLITE_OPEN_READONLY, nullptr);
  if (rc != SQLITE_OK || db_ == nullptr) {
    if (err_out != nullptr) {
      *err_out = db_ != nullptr ? sqlite3_errmsg(db_) : "sqlite3_open_v2 failed";
    }
    if (db_ != nullptr) {
      sqlite3_close(db_);
      db_ = nullptr;
    }
    return false;
  }
  return true;
}

}  // namespace cypha
