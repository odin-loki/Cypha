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

/// True when ``CYPHA_BENCH_FULL_CORPUS=1`` (full WikiText train + official valid eval).
bool bench_full_corpus_enabled();

/// Load train/eval token ids for bench profile ``d17`` (WikiText-2) or ``d04`` (Gutenberg).
/// When ``CYPHA_BENCH_FULL_CORPUS=1`` and WikiText files exist, uses ``wiki.train.tokens`` +
/// ``wiki.valid.tokens`` (``max_chars`` ignored for train). Otherwise 80/20 split with ``max_chars`` cap.
/// When ``bpe_merges`` and ``bpe_vocab`` are set and readable, corpus text is BPE-encoded
/// and ``vocab_size`` is taken from the tokenizer.
LMCorpus load_bench_corpus(const std::string& profile, int max_chars, int vocab_size,
                           const std::string& bpe_merges = "",
                           const std::string& bpe_vocab = "");

/// Load train/eval token ids from a text file (80/20 split). Relative paths resolve under repo root.
LMCorpus load_corpus_file(const std::string& corpus_path, const std::string& profile, int max_chars,
                          int vocab_size, const std::string& bpe_merges = "",
                          const std::string& bpe_vocab = "");

std::vector<int> synthetic_corpus(int n_tokens, int vocab_size, std::uint64_t seed);

}  // namespace cypha::cyphalm
