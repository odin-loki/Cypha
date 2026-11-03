#pragma once

#include <cstddef>

namespace cypha::cyphalm {

/// Set OpenMP thread count (0 = runtime default).
void set_thread_count(int threads);

int effective_thread_count();

/// Parallel for over [0, n) when OpenMP is enabled; otherwise serial.
template <typename Fn>
void parallel_for(std::size_t n, Fn fn) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        fn(static_cast<std::size_t>(i));
    }
#else
    for (std::size_t i = 0; i < n; ++i) fn(i);
#endif
}

}  // namespace cypha::cyphalm
