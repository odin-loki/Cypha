/// Smoke test: d17 and d21 bench corpora load without throwing.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "cypha/cyphalm/cyphalm_corpus.hpp"

namespace {

int smoke_profile(const char* profile) {
    try {
        const auto corpus =
            cypha::cyphalm::load_bench_corpus(profile, 500'000, 256);
        if (corpus.train_ids.size() < 256 || corpus.eval_ids.size() < 64) {
            std::fprintf(stderr, "corpus_smoke: %s split too short\n", profile);
            return 1;
        }
        std::printf("corpus_smoke: %s source=%s train=%zu eval=%zu\n", profile,
                    corpus.source.c_str(), corpus.train_ids.size(), corpus.eval_ids.size());
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "corpus_smoke: %s failed: %s\n", profile, ex.what());
        return 1;
    }
}

}  // namespace

int main() {
    if (smoke_profile("d17") != 0) return 1;
    if (smoke_profile("d21") != 0) return 1;
    std::puts("corpus_smoke: PASS");
    return 0;
}
