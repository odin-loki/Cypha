/// Trace a text back to the pretraining corpus with the ∞-gram index (after
/// OLMoTrace): every span of at least --min-len bytes that occurs verbatim in
/// the corpus, where it occurs, how often, and the share of the text covered.
/// Uses: attributing generations, checking held-out text for contamination,
/// finding duplicates.
///
///   cyphalm_trace --corpus enwik8 --corpus-bytes 95000000 --text out.txt [--min-len 32]
///   cyphalm_trace --corpus INDEX.igr --text enwik8 --offset 96000000 --bytes 16384
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/cyphalm/infinigram.hpp"
#include "nlohmann/json.hpp"

int main(int argc, char** argv) {
    std::string corpus, text_path;
    std::size_t corpus_bytes = 0, offset = 0, bytes = 0, min_len = 32, cap = 1 << 16, top = 20;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--corpus") corpus = next();
            else if (a == "--corpus-bytes") corpus_bytes = std::stoull(next());
            else if (a == "--text") text_path = next();
            else if (a == "--offset") offset = std::stoull(next());
            else if (a == "--bytes") bytes = std::stoull(next());
            else if (a == "--min-len") min_len = std::stoull(next());
            else if (a == "--top") top = std::stoull(next());
            else {
                std::cerr << "unknown arg " << a << "\n";
                return 2;
            }
        }
    } catch (const std::exception& e) {  // missing value, not a number
        std::cerr << "cyphalm_trace: bad arguments: " << e.what() << "\n";
        return 2;
    }
    if (corpus.empty() || text_path.empty()) {
        std::cerr << "need --corpus (IGR index or plain text) and --text\n";
        return 2;
    }
    // Read the text before building the index (minutes on a large corpus).
    std::ifstream f(text_path, std::ios::binary | std::ios::ate);
    const std::streamoff end = f ? static_cast<std::streamoff>(f.tellg()) : -1;
    if (end < 0) {
        std::cerr << "cyphalm_trace: cannot open --text " << text_path << "\n";
        return 1;
    }
    const std::size_t fsize = static_cast<std::size_t>(end);
    if (offset > fsize) offset = fsize;
    if (bytes == 0 || offset + bytes > fsize) bytes = fsize - offset;
    std::vector<std::uint8_t> t(bytes);
    f.seekg(static_cast<std::streamoff>(offset));
    f.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(bytes));
    if (!f) {
        std::cerr << "cyphalm_trace: cannot read " << bytes << " bytes of " << text_path << "\n";
        return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<const cypha::cyphalm::InfiniGram> ig;
    try {
        ig = cypha::cyphalm::InfiniGram::open(corpus, corpus_bytes);
    } catch (const std::exception& e) {
        std::cerr << "cyphalm_trace: cannot index --corpus " << corpus << ": " << e.what() << "\n";
        return 1;
    }
    const double index_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Matching statistics: ms[i] = longest prefix of t[i..] in the corpus;
    // ms[i] >= ms[i-1] - 1.
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<std::size_t> ms(bytes), where(bytes), cnt(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        const std::size_t lb = i > 0 && ms[i - 1] > 0 ? ms[i - 1] - 1 : 0;
        ms[i] = ig->match_prefix(t.data() + i, bytes - i, cap, lb, where[i], cnt[i]);
    }
    const double trace_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    // Maximal spans >= min_len (not contained in the previous one) and coverage.
    std::vector<char> covered(bytes, 0);
    struct Span {
        std::size_t at, len, corpus_pos, count;
    };
    std::vector<Span> spans;
    std::size_t reach = 0, longest = 0;
    for (std::size_t i = 0; i < bytes; ++i) {
        longest = std::max(longest, ms[i]);
        if (ms[i] < min_len) continue;
        std::fill(covered.begin() + static_cast<std::ptrdiff_t>(i),
                  covered.begin() + static_cast<std::ptrdiff_t>(i + ms[i]), 1);
        if (i + ms[i] > reach) {
            if (i >= reach || spans.empty()) spans.push_back({i, ms[i], where[i], cnt[i]});
            else if (ms[i] > spans.back().len) spans.push_back({i, ms[i], where[i], cnt[i]});
            reach = i + ms[i];
        }
    }
    std::size_t cov = 0;
    for (char c : covered) cov += c != 0;
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.len > b.len; });
    nlohmann::json out;
    out["corpus"] = corpus;
    out["corpus_bytes"] = ig->size();
    out["text"] = text_path;
    out["offset"] = offset;
    out["bytes"] = bytes;
    out["min_len"] = min_len;
    out["coverage"] = bytes ? static_cast<double>(cov) / static_cast<double>(bytes) : 0.0;
    out["longest_match"] = longest;
    double mean = 0.0;
    for (std::size_t v : ms) mean += static_cast<double>(v);
    out["mean_match"] = bytes ? mean / static_cast<double>(bytes) : 0.0;
    out["spans"] = spans.size();
    out["index_seconds"] = index_s;
    out["trace_seconds"] = trace_s;
    nlohmann::json list = nlohmann::json::array();
    for (std::size_t k = 0; k < std::min(top, spans.size()); ++k) {
        const auto& s = spans[k];
        std::string snip(t.begin() + static_cast<std::ptrdiff_t>(s.at),
                         t.begin() + static_cast<std::ptrdiff_t>(s.at + std::min<std::size_t>(s.len, 80)));
        for (char& c : snip)
            if (static_cast<unsigned char>(c) < 32 || static_cast<unsigned char>(c) > 126) c = '.';
        list.push_back({{"text_offset", offset + s.at}, {"len", s.len}, {"corpus_offset", s.corpus_pos},
                        {"occurrences", s.count}, {"snippet", snip}});
    }
    out["longest_spans"] = list;
    std::cout << out.dump(1) << std::endl;
    return 0;
}
