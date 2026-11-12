// k5_skelfree: the bank of whole-sentence skeletons for skeleton-first free sentences (docs/reports/K5_SKELFREE.md).
//
//   k5_skelfree mine --k5 TEXT --text CLEAN --lo A --hi B --out BANK [--min N]
//       Every sentence of CLEAN's documents (lines) that start in [A, B), cut as K5's gate cuts a draft
//       (structure::split), becomes a skeleton: its sequence of K5 exchange classes. A sentence is kept if it starts
//       a sentence (it is its document's first, or the one before it ended, and its first token begins with a capital
//       letter A-Z or a digit), ends with . ! or ?, every token is a K5
//       word (no <unk>, no newline), and it has 5 to 32 tokens. Skeletons are
//       counted; those seen at least N times (default 1) are written, most frequent first. Prints the counts.
//   k5_skelfree show --k5 TEXT --bank BANK [--n N] [--skip S]
//       A bank's skeletons, each class code shown as its most probable word under P(w | class).
//   k5_skelfree selftest
//
// Author: Odin Loch

#include "cypha/lm/planner/free_skel.hpp"
#include "cypha/lm/planner/skeleton.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace pl = cypha::cyphalm::planner;
namespace st = cypha::cyphalm::structure;
namespace wkn = cypha::cyphalm::wkn;

namespace {

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d = "") const {
        const auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
};

bool load_k5(const std::string& path, wkn::K5& m) {
    std::string err;
    const wkn::Options bo;
    if (!m.load(path, 12, err, &bo)) {
        std::cerr << err << "\n";
        return false;
    }
    std::fprintf(stderr, "k5: vocab %zu, %zu tokens\n", m.vocab().size(), m.tokens());
    return true;
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    out.resize(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    in.read(out.data(), static_cast<std::streamsize>(out.size()));
    return static_cast<bool>(in);
}

// The skeleton of one sentence (false if it is not kept by the rule above).
bool free_shape(const wkn::K5& m, const st::Sent& s, std::vector<std::uint16_t>& codes) {
    codes.clear();
    if (!s.ended || s.tok.size() < pl::kMinCodes || s.tok.size() > pl::kMaxCodes) return false;
    const auto& cls = m.classes().classes();
    for (const auto& t : s.tok) {
        if (t == "\n") return false;
        const std::uint32_t id = m.id(t);
        if (id == wkn::kUnk || id >= cls.size()) return false;
        codes.push_back(static_cast<std::uint16_t>(cls[id]));
    }
    return true;
}

int cmd_mine(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    std::string text;
    if (!read_file(a.get("--text"), text)) {
        std::cerr << "cannot read " << a.get("--text") << "\n";
        return 1;
    }
    const std::size_t lo = std::strtoull(a.get("--lo", "0").c_str(), nullptr, 10);
    const std::size_t hi = std::strtoull(a.get("--hi", std::to_string(text.size())).c_str(), nullptr, 10);
    const std::uint32_t mn = static_cast<std::uint32_t>(std::atoi(a.get("--min", "1").c_str()));
    std::map<std::vector<std::uint16_t>, std::uint32_t> cnt;
    std::uint64_t sents = 0, kept = 0, unended = 0, unk = 0, len_out = 0, docs = 0;
    std::vector<std::uint64_t> by_len(pl::kMaxCodes + 1, 0);
    std::vector<std::uint16_t> codes;
    std::uint64_t mid = 0;
    pl::for_docs(text, lo, hi, [&](const std::vector<st::Sent>& ss, std::size_t) {
        ++docs;
        bool start = true;  // the document's first sentence, or one after a sentence that ended
        for (const auto& s : ss) {
            ++sents;
            const bool at_start = start;
            start = s.ended;
            // it begins inside a sentence (the one before was cut at 240 bytes, or this one opens with a closing mark)
            const char c0 = s.tok.empty() || s.tok[0].empty() ? 0 : s.tok[0][0];
            if (!at_start || !((c0 >= 'A' && c0 <= 'Z') || (c0 >= '0' && c0 <= '9'))) {
                ++mid;
                continue;
            }
            if (!s.ended) {
                ++unended;
                continue;
            }
            if (s.tok.size() < pl::kMinCodes || s.tok.size() > pl::kMaxCodes) {
                ++len_out;
                continue;
            }
            if (!free_shape(m, s, codes)) {
                ++unk;
                continue;
            }
            ++kept, ++by_len[codes.size()];
            ++cnt[codes];
        }
    });
    std::vector<std::pair<std::uint32_t, const std::vector<std::uint16_t>*>> v;
    std::uint64_t seen2 = 0, n2 = 0, wrote_sents = 0;
    for (const auto& [c, n] : cnt) {
        if (n >= 2) ++n2, seen2 += n;
        if (n >= mn) v.push_back({n, &c});
    }
    std::stable_sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    std::ofstream o(a.get("--out"));
    o << "K5FREE-SKEL 1\n";
    for (const auto& [n, c] : v) {
        o << n;
        for (const auto x : *c) o << " " << x;
        o << "\n";
        wrote_sents += n;
    }
    std::printf("documents %llu, sentences %llu: kept %llu; starting inside a sentence %llu, not ended %llu, outside 5-32 tokens %llu, "
                "with an unknown word %llu\n",
                static_cast<unsigned long long>(docs), static_cast<unsigned long long>(sents), static_cast<unsigned long long>(kept),
                static_cast<unsigned long long>(mid),
                static_cast<unsigned long long>(unended), static_cast<unsigned long long>(len_out), static_cast<unsigned long long>(unk));
    std::printf("distinct skeletons %zu; seen at least twice %llu (holding %llu sentences); written (min %u) %zu, holding %llu sentences\n",
                cnt.size(), static_cast<unsigned long long>(n2), static_cast<unsigned long long>(seen2), mn, v.size(),
                static_cast<unsigned long long>(wrote_sents));
    std::printf("kept sentences by length (tokens):");
    for (std::size_t L = pl::kMinCodes; L <= pl::kMaxCodes; ++L) std::printf(" %zu:%llu", L, static_cast<unsigned long long>(by_len[L]));
    std::printf("\n");
    return o ? 0 : 1;
}

int cmd_show(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    pl::FreeBank b;
    std::string err;
    if (!b.load(a.get("--bank"), err)) {
        std::cerr << err << "\n";
        return 1;
    }
    const auto& cls = m.classes().classes();
    const auto& pwc = m.classes().pwc();
    std::vector<std::uint32_t> top;
    for (std::uint32_t w = 2; w < cls.size(); ++w) {
        if (m.vocab()[w] == "\n") continue;
        if (top.size() <= cls[w]) top.resize(cls[w] + 1, 0);
        if (top[cls[w]] == 0 || pwc[w] > pwc[top[cls[w]]]) top[cls[w]] = w;
    }
    const std::size_t n = static_cast<std::size_t>(std::atoi(a.get("--n", "20").c_str()));
    const std::size_t skip = static_cast<std::size_t>(std::atoi(a.get("--skip", "0").c_str()));
    for (std::size_t i = skip; i < b.shapes.size() && i < skip + n; ++i) {
        std::printf("%u\t", b.count[i]);
        for (const auto c : b.shapes[i]) std::printf("%s%s", c < top.size() && top[c] ? m.vocab()[top[c]].c_str() : "?", " ");
        std::printf("\n");
    }
    return 0;
}

int cmd_selftest() {
    std::string err;
    if (!pl::free_skel_self_test(err)) {
        std::cerr << "selftest: " << err << "\n";
        return 1;
    }
    std::puts("k5_skelfree selftest: ok");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: k5_skelfree mine|show|selftest ...\n";
        return 2;
    }
    const std::string cmd = argv[1];
    Args a;
    for (int i = 2; i + 1 < argc; i += 2) a.kv[argv[i]] = argv[i + 1];
    if (cmd == "mine") return cmd_mine(a);
    if (cmd == "show") return cmd_show(a);
    if (cmd == "selftest") return cmd_selftest();
    std::cerr << "unknown command " << cmd << "\n";
    return 2;
}
