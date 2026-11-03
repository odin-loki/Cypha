/// Smoke test: BpeTokenizer encode -> decode round trip.
///
/// BpeTokenizer only exposes a file-based loader (BpeTokenizer::load(merges_path,
/// vocab_path)), so this tool writes a minimal fixture (an identity byte vocab covering
/// every character in the test string, plus one merge rule to exercise apply_bpe) to a
/// temp directory, loads it, and checks that decode(encode(text)) == text exactly.
/// decode() is a straight concatenation of the token strings that encode() produced, in
/// order, so as long as every token apply_bpe() emits is present in the vocab (i.e. never
/// falls back to <unk>/id-0), the round trip is byte-exact -- there is no documented lossy
/// transform (no lowercasing, no whitespace normalization) in the tokenizer's contract.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#include "cypha/cyphalm/bpe_tokenizer.hpp"

namespace {

namespace fs = std::filesystem;

std::string json_escape(char c) {
    switch (c) {
        case '"': return "\\\"";
        case '\\': return "\\\\";
        default: return std::string(1, c);
    }
}

}  // namespace

int main() {
    const std::string text = "the cat sat, the cat ran; the mat was flat.";

    std::set<char> unique_chars;
    for (char c : text) unique_chars.insert(c);

    const fs::path dir = fs::temp_directory_path() / "cypha_bpe_tokenizer_smoke";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path vocab_path = dir / "vocab.json";
    const fs::path merges_path = dir / "merges.txt";

    {
        std::ofstream vf(vocab_path, std::ios::trunc);
        if (!vf) {
            std::fprintf(stderr, "bpe_tokenizer_smoke: cannot write fixture vocab %s\n",
                         vocab_path.string().c_str());
            return 1;
        }
        vf << "{\n";
        std::uint32_t id = 0;
        bool first = true;
        for (char c : unique_chars) {
            if (!first) vf << ",\n";
            first = false;
            vf << "  \"" << json_escape(c) << "\": " << id;
            ++id;
        }
        // Merged-token entry for the "t"+"h" merge rule below, so apply_bpe's output is
        // always found in vocab (never falls back to <unk>/id 0).
        vf << ",\n  \"th\": " << id << "\n";
        vf << "}\n";
    }
    {
        std::ofstream mf(merges_path, std::ios::trunc);
        if (!mf) {
            std::fprintf(stderr, "bpe_tokenizer_smoke: cannot write fixture merges %s\n",
                         merges_path.string().c_str());
            return 1;
        }
        mf << "t h\n";
    }

    int rc = 0;
    try {
        const cypha::cyphalm::BpeTokenizer tok =
            cypha::cyphalm::BpeTokenizer::load(merges_path.string(), vocab_path.string());
        if (tok.vocab_size() == 0) {
            std::fprintf(stderr, "bpe_tokenizer_smoke: loaded tokenizer has empty vocab\n");
            rc = 1;
        } else {
            const std::vector<std::uint32_t> ids = tok.encode(text);
            if (ids.empty()) {
                std::fprintf(stderr, "bpe_tokenizer_smoke: encode produced no tokens\n");
                rc = 1;
            } else {
                const std::string decoded = tok.decode(ids);
                if (decoded != text) {
                    std::fprintf(stderr,
                                 "bpe_tokenizer_smoke: round-trip mismatch\n  original: \"%s\"\n"
                                 "  decoded:  \"%s\"\n",
                                 text.c_str(), decoded.c_str());
                    rc = 1;
                } else {
                    std::printf(
                        "bpe_tokenizer_smoke: vocab_size=%u tokens=%zu round-trip exact PASS\n",
                        tok.vocab_size(), ids.size());
                }
            }
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "bpe_tokenizer_smoke: exception: %s\n", ex.what());
        rc = 1;
    }

    fs::remove(vocab_path, ec);
    fs::remove(merges_path, ec);
    fs::remove(dir, ec);

    if (rc == 0) std::puts("bpe_tokenizer_smoke: PASS");
    return rc;
}
