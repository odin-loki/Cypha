#include "cypha/cyphalm/cyphalm_parallel.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace cypha::cyphalm {

void set_thread_count(int threads) {
#if defined(_OPENMP)
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif
}

int effective_thread_count() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace cypha::cyphalm
