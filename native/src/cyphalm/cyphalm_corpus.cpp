#include "cypha/cyphalm/cyphalm_corpus.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <unordered_map>

#include "cypha/cyphalm/bpe_tokenizer.hpp"
#include "cypha/bench/bench_paths.hpp"

namespace fs = std::filesystem;

namespace cypha::cyphalm {

namespace {

std::string repo_root_from_native() {
    const fs::path native = fs::path(__FILE__).parent_path().parent_path().parent_path();
    return (native.parent_path()).string();
}

std::string read_text_file(const fs::path& p, int max_chars) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open corpus: " + p.string());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (max_chars > 0 && static_cast<int>(text.size()) > max_chars) {
        text.resize(static_cast<std::size_t>(max_chars));
    }
    return text;
}

void build_vocab(const std::string& text, int vocab_size,
                 std::unordered_map<char, int>& c2i) {
    std::vector<char> uniq;
    uniq.reserve(256);
    for (unsigned char ch : text) {
        char c = static_cast<char>(ch);
        if (std::find(uniq.begin(), uniq.end(), c) == uniq.end()) {
            uniq.push_back(c);
        }
    }
    std::sort(uniq.begin(), uniq.end());
    const int limit = std::max(1, vocab_size - 1);
    if (static_cast<int>(uniq.size()) > limit) {
        uniq.resize(static_cast<std::size_t>(limit));
    }
    c2i.clear();
    c2i['?'] = 0;
    for (std::size_t i = 0; i < uniq.size(); ++i) {
        c2i[uniq[i]] = static_cast<int>(i + 1);
    }
}

std::vector<int> encode(const std::string& text, const std::unordered_map<char, int>& c2i) {
    std::vector<int> out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        auto it = c2i.find(static_cast<char>(ch));
        out.push_back(it == c2i.end() ? 0 : it->second);
    }
    return out;
}

std::vector<int> bpe_encode_ids(const std::string& text, const BpeTokenizer& tok) {
    const auto uids = tok.encode(text);
    std::vector<int> out;
    out.reserve(uids.size());
    for (std::uint32_t id : uids) out.push_back(static_cast<int>(id));
    return out;
}

LMCorpus from_text(const std::string& text, const std::string& source,
                   const std::string& profile, int vocab_size,
                   const BpeTokenizer* bpe = nullptr) {
    std::vector<int> ids;
    int effective_vocab = vocab_size;
    if (bpe != nullptr) {
        ids = bpe_encode_ids(text, *bpe);
        effective_vocab = static_cast<int>(bpe->vocab_size());
    } else {
        std::unordered_map<char, int> c2i;
        build_vocab(text, vocab_size, c2i);
        ids = encode(text, c2i);
    }
    if (ids.size() < 512) {
        throw std::runtime_error("corpus too short after encoding");
    }
    const std::size_t split = static_cast<std::size_t>(ids.size() * 8 / 10);
    LMCorpus c;
    c.source = source;
    c.profile = profile;
    c.vocab_size = effective_vocab;
    c.train_ids.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(split));
    c.eval_ids.assign(ids.begin() + static_cast<std::ptrdiff_t>(split), ids.end());
    return c;
}

LMCorpus from_train_eval_text(const std::string& train_text, const std::string& eval_text,
                              const std::string& source, const std::string& profile, int vocab_size,
                              const BpeTokenizer* bpe = nullptr) {
    std::vector<int> train_ids;
    std::vector<int> eval_ids;
    int effective_vocab = vocab_size;
    if (bpe != nullptr) {
        train_ids = bpe_encode_ids(train_text, *bpe);
        eval_ids = bpe_encode_ids(eval_text, *bpe);
        effective_vocab = static_cast<int>(bpe->vocab_size());
    } else {
        std::unordered_map<char, int> c2i;
        build_vocab(train_text + eval_text, vocab_size, c2i);
        train_ids = encode(train_text, c2i);
        eval_ids = encode(eval_text, c2i);
    }
    if (train_ids.size() < 256 || eval_ids.size() < 64) {
        throw std::runtime_error("official split too short after encoding");
    }
    LMCorpus c;
    c.source = source + "_official_split";
    c.profile = profile;
    c.vocab_size = effective_vocab;
    c.train_ids = std::move(train_ids);
    c.eval_ids = std::move(eval_ids);
    return c;
}

}  // namespace

bool bench_full_corpus_enabled() { return cypha::bench::bench_env_truthy("CYPHA_BENCH_FULL_CORPUS"); }

LMCorpus load_bench_corpus(const std::string& profile, int max_chars, int vocab_size,
                           const std::string& bpe_merges, const std::string& bpe_vocab) {
    std::unique_ptr<BpeTokenizer> bpe;
    if (!bpe_merges.empty() && !bpe_vocab.empty() &&
        fs::is_regular_file(bpe_merges) && fs::is_regular_file(bpe_vocab)) {
        bpe = std::make_unique<BpeTokenizer>(BpeTokenizer::load(bpe_merges, bpe_vocab));
    }
    const BpeTokenizer* bpe_ptr = bpe.get();
    const fs::path root = fs::path(repo_root_from_native()) / "bench" / "data";
    if (profile == "d17" || profile == "d21") {
        const fs::path wt_dir = root / "wikitext2" / "wikitext-2";
        const fs::path wt_train = wt_dir / "wiki.train.tokens";
        const fs::path wt_valid = wt_dir / "wiki.valid.tokens";
        if (bench_full_corpus_enabled() && fs::is_regular_file(wt_train) && fs::is_regular_file(wt_valid)) {
            return from_train_eval_text(read_text_file(wt_train, 0), read_text_file(wt_valid, max_chars),
                                        "wikitext2", profile, vocab_size, bpe_ptr);
        }
        if (fs::is_regular_file(wt_train)) {
            const int cap = bench_full_corpus_enabled() ? 0 : max_chars;
            return from_text(read_text_file(wt_train, cap), "wikitext2", profile, vocab_size, bpe_ptr);
        }
        for (const char* name : {"moby_dick.txt", "alice.txt", "sherlock_holmes.txt"}) {
            const fs::path p = root / "gutenberg" / name;
            if (fs::is_regular_file(p)) {
                return from_text(read_text_file(p, max_chars), "gutenberg_fallback", profile, vocab_size,
                                 bpe_ptr);
            }
        }
    }
    if (profile == "d04") {
        for (const char* name : {"moby_dick.txt", "alice.txt", "sherlock_holmes.txt"}) {
            const fs::path p = root / "gutenberg" / name;
            if (fs::is_regular_file(p)) {
                return from_text(read_text_file(p, max_chars), name, profile, vocab_size,
                                 bpe_ptr);
            }
        }
    }
    throw std::runtime_error("no corpus for profile " + profile + " under bench/data");
}

LMCorpus load_corpus_file(const std::string& corpus_path, const std::string& profile, int max_chars,
                          int vocab_size, const std::string& bpe_merges,
                          const std::string& bpe_vocab) {
    std::unique_ptr<BpeTokenizer> bpe;
    if (!bpe_merges.empty() && !bpe_vocab.empty() &&
        fs::is_regular_file(bpe_merges) && fs::is_regular_file(bpe_vocab)) {
        bpe = std::make_unique<BpeTokenizer>(BpeTokenizer::load(bpe_merges, bpe_vocab));
    }
    const BpeTokenizer* bpe_ptr = bpe.get();

    fs::path p(corpus_path);
    if (!p.is_absolute()) {
        const fs::path root = fs::path(repo_root_from_native());
        const fs::path under_root = root / p;
        if (fs::is_regular_file(under_root)) {
            p = under_root;
        } else if (!fs::is_regular_file(p)) {
            p = fs::current_path() / corpus_path;
        }
    }
    if (!fs::is_regular_file(p)) {
        throw std::runtime_error("cannot open corpus: " + corpus_path);
    }
    return from_text(read_text_file(p, max_chars), p.filename().string(), profile, vocab_size, bpe_ptr);
}

std::vector<int> synthetic_corpus(int n_tokens, int vocab_size, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(1, std::max(1, vocab_size - 1));
    std::vector<int> ids(static_cast<std::size_t>(n_tokens));
    for (auto& t : ids) t = dist(rng);
    return ids;
}

}  // namespace cypha::cyphalm
