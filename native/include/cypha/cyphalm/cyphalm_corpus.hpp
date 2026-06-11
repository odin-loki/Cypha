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
/// When ``bpe_merges`` and ``bpe_vocab`` are set and readable, corpus text is BPE-encoded
/// and ``vocab_size`` is taken from the tokenizer.
LMCorpus load_bench_corpus(const std::string& profile, int max_chars, int vocab_size,
                           const std::string& bpe_merges = "",
                           const std::string& bpe_vocab = "");

std::vector<int> synthetic_corpus(int n_tokens, int vocab_size, std::uint64_t seed);

}  // namespace cypha::cyphalm
