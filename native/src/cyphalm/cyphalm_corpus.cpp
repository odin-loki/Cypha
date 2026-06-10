#include "cypha/cyphalm/cyphalm_corpus.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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

LMCorpus from_text(const std::string& text, const std::string& source,
                   const std::string& profile, int vocab_size) {
    std::unordered_map<char, int> c2i;
    build_vocab(text, vocab_size, c2i);
    auto ids = encode(text, c2i);
    if (ids.size() < 512) {
        throw std::runtime_error("corpus too short after encoding");
    }
    const std::size_t split = static_cast<std::size_t>(ids.size() * 8 / 10);
    LMCorpus c;
    c.source = source;
    c.profile = profile;
    c.vocab_size = vocab_size;
    c.train_ids.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(split));
    c.eval_ids.assign(ids.begin() + static_cast<std::ptrdiff_t>(split), ids.end());
    return c;
}

}  // namespace

LMCorpus load_bench_corpus(const std::string& profile, int max_chars, int vocab_size) {
    const fs::path root = fs::path(repo_root_from_native()) / "cypha_bench" / "data";
    if (profile == "d17") {
        const fs::path wt = root / "wikitext2" / "wikitext-2" / "wiki.train.tokens";
        if (fs::is_regular_file(wt)) {
            return from_text(read_text_file(wt, max_chars), "wikitext2", profile, vocab_size);
        }
    }
    if (profile == "d04") {
        for (const char* name : {"moby_dick.txt", "alice.txt", "sherlock_holmes.txt"}) {
            const fs::path p = root / "gutenberg" / name;
            if (fs::is_regular_file(p)) {
                return from_text(read_text_file(p, max_chars), name, profile, vocab_size);
            }
        }
    }
    throw std::runtime_error("no corpus for profile " + profile + " under cypha_bench/data");
}

std::vector<int> synthetic_corpus(int n_tokens, int vocab_size, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(1, std::max(1, vocab_size - 1));
    std::vector<int> ids(static_cast<std::size_t>(n_tokens));
    for (auto& t : ids) t = dist(rng);
    return ids;
}

}  // namespace cypha::cyphalm
