#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::cyphalm {

struct LMCorpus {
    std::string source;
    std::string profile;
    std::vector<int> train_ids;
    std::vector<int> eval_ids;
    int vocab_size = 128;
};

/// Load train/eval token ids for bench profile ``d17`` (WikiText-2) or ``d04`` (Gutenberg).
LMCorpus load_bench_corpus(const std::string& profile, int max_chars, int vocab_size);

std::vector<int> synthetic_corpus(int n_tokens, int vocab_size, std::uint64_t seed);

}  // namespace cypha::cyphalm
