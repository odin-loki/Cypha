#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cypha/env.hpp"

namespace cypha {

/// Minimum batch rows before considering row-parallel score / LLR.
constexpr int kScoreParallelRowThreshold = 16;

/// Minimum n*d*K FMAs before OpenMP pays off (tuned on MinGW OpenMP; tiny batches stay serial).
constexpr long long kScoreParallelMinWork = 1000000;

inline bool score_parallel_rows_enabled() {
  const std::optional<std::string> v = env_get("CYPHA_SCORE_PARALLEL_ROWS");
  if (!v.has_value() || v->empty()) {
    return true;
  }
  return (*v)[0] != '0';
}

/// True when batch is large enough that row parallelism beats serial (or forced off).
inline bool should_parallel_score_rows(int n, int d, int K) {
  if (!score_parallel_rows_enabled() || n < kScoreParallelRowThreshold) {
    return false;
  }
  if (d > 0 && K > 0) {
    const long long work = static_cast<long long>(n) * static_cast<long long>(d) * static_cast<long long>(K);
    if (work < kScoreParallelMinWork) {
      return false;
    }
  }
  return true;
}

template <class F>
inline void parallel_for_rows(int begin, int end, F&& f) {
  const int nrows = end - begin;
  if (nrows <= 0) {
    return;
  }
  if (!score_parallel_rows_enabled() || nrows < kScoreParallelRowThreshold) {
    for (int i = begin; i < end; ++i) {
      f(i);
    }
    return;
  }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
  for (int i = begin; i < end; ++i) {
    f(i);
  }
#else
  unsigned hw = std::thread::hardware_concurrency();
  int nt = hw ? static_cast<int>(hw) : 4;
  nt = std::min(nt, nrows);
  if (nt <= 1) {
    for (int i = begin; i < end; ++i) {
      f(i);
    }
    return;
  }
  const int chunk = (nrows + nt - 1) / nt;
  std::vector<std::thread> th;
  th.reserve(static_cast<std::size_t>(nt));
  for (int t = 0; t < nt; ++t) {
    const int lo = begin + t * chunk;
    const int hi = std::min(end, lo + chunk);
    if (lo >= hi) {
      break;
    }
    th.emplace_back([lo, hi, &f]() {
      for (int i = lo; i < hi; ++i) {
        f(i);
      }
    });
  }
  for (auto& x : th) {
    x.join();
  }
#endif
}

template <class F>
inline void parallel_for_score_rows(int n, int d, int K, F&& f) {
  if (!should_parallel_score_rows(n, d, K)) {
    for (int i = 0; i < n; ++i) {
      f(i);
    }
    return;
  }
  parallel_for_rows(0, n, std::forward<F>(f));
}

}  // namespace cypha
