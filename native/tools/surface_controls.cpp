// Controls for the surface score S = H * D * L (surface_score.cpp).
//
// A text-quality score has to rank real text above text with no meaning. This
// scores both against the same reference as surface_score.cpp:
//   wiki_window          200 bytes of the training slice (real text)
//   random_corpus_words  words drawn at random from the training slice
//   shuffled_window      the words of a wiki window in random order
//   loop                 one short phrase repeated
// and, from stdin, every "GEN<TAB>kind<TAB>text" line (trie_context_8k output),
// averaged per kind.
//
// Measured 2026-10-26: wiki window 0.933, its words shuffled 0.928, random
// corpus words 0.918, loop 0.027. The trie's nucleus sample scores 0.775.
// From the 8K context run: correct copied answers 0.768; wrong answers 0.823
// (Thanks), 0.838 (held-out Dolly), 0.860 (wiki facts), 0.887 (paraphrase).
// S does not separate sense from nonsense, or right answers from wrong ones.
// Report: docs/reports/CYPHALM_QA_AND_CONTEXT_TEST.md
//
// Usage: trie_context_8k ... | surface_controls [enwik9_or_enwik8]

#define main surface_score_main
#include "surface_score.cpp"
#undef main

#include <map>
#include <random>

int main(int argc, char** argv) {
    const std::string wiki_path = argc > 1 ? argv[1] : "/mnt/d/cypha_llm/enwik9";
    std::ifstream in(wiki_path, std::ios::binary);
    std::vector<char> raw(kTrain);
    in.read(raw.data(), static_cast<std::streamsize>(kTrain));
    if (!in) {
        std::cerr << "short corpus " << wiki_path << "\n";
        return 1;
    }
    std::unordered_set<std::string> vocab;
    std::vector<std::string> words;
    for (std::size_t i = 0; i < raw.size();) {
        if (!is_letter(static_cast<unsigned char>(raw[i]))) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < raw.size() && is_letter(static_cast<unsigned char>(raw[j]))) ++j;
        std::string w(raw.begin() + static_cast<std::ptrdiff_t>(i), raw.begin() + static_cast<std::ptrdiff_t>(j));
        words.push_back(w);
        if (j - i >= 3 && j - i <= 32) {
            for (char& c : w)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            vocab.insert(std::move(w));
        }
        i = j;
    }
    double sum_d4 = 0, sum_loop = 0;
    int windows = 0;
    for (std::size_t off = 100000; off + 200 < kTrain && windows < 40; off += 150000) {
        std::string w(raw.begin() + static_cast<std::ptrdiff_t>(off), raw.begin() + static_cast<std::ptrdiff_t>(off + 200));
        sum_d4 += distinct4(w);
        sum_loop += loop8(w);
        ++windows;
    }
    const double ref_d4 = sum_d4 / windows, ref_loop = sum_loop / windows;

    std::mt19937 rng(1);
    auto report = [&](const char* name, const std::string& text) {
        const Parts p = score(text, vocab, ref_d4, ref_loop);
        std::cout << "control " << name << " S=" << p.S << " H=" << p.H << " D=" << p.D << " L=" << p.L << "\n";
    };
    const std::string window(raw.begin() + 100000, raw.begin() + 100200);
    report("wiki_window", window);
    {
        std::uniform_int_distribution<std::size_t> pick(0, words.size() - 1);
        std::string s;
        while (s.size() < 200) s += words[pick(rng)] + " ";
        report("random_corpus_words", s.substr(0, 200));
    }
    {
        std::vector<std::string> toks;
        std::size_t a = 0;
        while (a < window.size()) {
            std::size_t b = window.find_first_of(" \n", a);
            if (b == std::string::npos) b = window.size();
            if (b > a) toks.push_back(window.substr(a, b - a));
            a = b + 1;
        }
        std::shuffle(toks.begin(), toks.end(), rng);
        std::string s;
        for (auto& t : toks) s += t + " ";
        report("shuffled_window", s.substr(0, 200));
    }
    {
        std::string s;
        while (s.size() < 200) s += "the states and ";
        report("loop", s.substr(0, 200));
    }

    std::map<std::string, std::pair<double, int>> agg;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.rfind("GEN\t", 0) != 0) continue;
        const auto t2 = line.find('\t', 4);
        if (t2 == std::string::npos) continue;
        const Parts p = score(unescape(line.substr(t2 + 1)), vocab, ref_d4, ref_loop);
        auto& a = agg[line.substr(4, t2 - 4)];
        a.first += p.S;
        a.second += 1;
    }
    for (auto& [k, v] : agg) std::cout << "generated " << k << " mean_S=" << v.first / v.second << " n=" << v.second << "\n";
    return 0;
}
