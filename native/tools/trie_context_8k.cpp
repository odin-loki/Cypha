// Does a longer count context fix question answering? An order-8192 count
// model on top of the packed trie.
//
// Context model: InfiniGram (native/src/cyphalm/infinigram.cpp) over the
// training bytes. For each position it takes the last <= cap bytes, finds the
// longest suffix that occurs in training, and counts the byte that followed
// every occurrence. That is an order-cap count model that backs off only
// when the longer context never occurred.
//
// Trie: trained as trie_chat.cpp (wiki [0, 7340032), two Dolly passes after a
// <|doc|> break), except 30 Dolly turns (index % 500 == 250) are held out.
// Mixture: P = lam * P_ctx + (1 - lam) * P_trie, one lam per (match length,
// count) bucket, fit on the first half of the wiki holdout [7340032, 8388608)
// and scored on the second half. Caps 4, 16, 64, 512, 8192.
//
// Questions (greedy, 150 bytes, cap 8192): seen; first letter lowercased;
// " Thanks." before "\nAssistant: "; hand paraphrases; the 30 held-out Dolly
// turns; 8 facts in the Wikipedia slice. Output lines "GEN<TAB>kind<TAB>text"
// feed surface_controls for the surface score.
//
// Measured 2026-10-26: trie 2.3199 BPC; +16 bytes 2.0637; +64 2.0711;
// +512 2.0715; +8192 2.0715. Median matched context 10 bytes; 1 position in
// 10^6 matched more than 1024. Seen 3/3, first letter 3/3, Thanks 0/3,
// paraphrase 0/3, held-out Dolly 1/30 ("yes"), wiki facts 0/8.
// Report: docs/reports/CYPHALM_QA_AND_CONTEXT_TEST.md
//
// Build: g++ -O2 -std=c++20 -I native/third_party -I native/third_party/libsais
//        -I native/include native/tools/trie_context_8k.cpp
//        native/src/cyphalm/infinigram.cpp libsais.o
// Usage: trie_context_8k [enwik9_or_enwik8] [dolly.jsonl]

#define main trie_chat_main
#include "trie_chat.cpp"
#undef main

#include "cypha/cyphalm/infinigram.hpp"

#include <cctype>
#include <map>

namespace {

using cypha::cyphalm::InfiniGram;

int len_bucket(int n) { return n < 4 ? 0 : n < 8 ? 1 : n < 16 ? 2 : n < 32 ? 3 : n < 64 ? 4 : n < 128 ? 5 : 6; }
int count_bucket(std::uint64_t t) { return t <= 1 ? 0 : t <= 3 ? 1 : t <= 15 ? 2 : 3; }
int bucket(int n, std::uint64_t t) { return len_bucket(n) * 4 + count_bucket(t); }
constexpr int kBuckets = 28;

std::string lowered(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

void train_span(Model& model, std::vector<std::uint8_t>& data, double w[6], std::size_t a, std::size_t b) {
    for (std::size_t i = a; i < b; ++i) {
        double e[5][256];
        bool matched = false;
        unsigned pred = 0;
        model.fill_experts(data.data(), i, i, Mode::Synth, e, matched, pred, true);
        double py[6];
        const unsigned y = data[i];
        for (int k = 0; k < 5; ++k) py[k] = e[k][y];
        py[5] = (matched && pred == y) ? 1.0 : 0.0;
        eg_update(w, py, matched);
        model.observe(i);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    const std::string wiki_path = argc > 1 ? argv[1] : "/mnt/d/cypha_llm/enwik9";
    const std::string dolly_path = argc > 2 ? argv[2] : "/mnt/d/cypha_llm/chat/dolly.jsonl";
    if (!self_test()) return 1;

    std::ifstream in(wiki_path, std::ios::binary);
    std::vector<std::uint8_t> data(kSample);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(kSample));
    if (static_cast<std::size_t>(in.gcount()) != kSample) {
        std::cerr << "short corpus " << wiki_path << "\n";
        return 1;
    }
    Model model;
    model.data = data.data();
    model.init();
    double w[6];
    w[5] = 0.15;
    for (int i = 0; i < 5; ++i) w[i] = 0.85 / 5.0;
    train_span(model, data, w, 0, kTrain);

    std::vector<Turn> turns;
    load_dolly(dolly_path, turns);
    if (turns.empty()) {
        std::cerr << "dolly missing " << dolly_path << "\n";
        return 1;
    }
    std::vector<char> held(turns.size(), 0);
    std::vector<std::size_t> held_idx;
    for (std::size_t i = 0; i < turns.size(); ++i)
        if (i % 500 == 250) {
            held[i] = 1;
            held_idx.push_back(i);
        }
    std::string chat;
    for (std::size_t i = 0; i < turns.size(); ++i)
        if (!held[i]) chat += turns[i].prompt + turns[i].answer + "\n\n";
    const std::size_t chat_at = data.size();
    const char* boundary = "\n\n<|doc|>\n\n";
    data.insert(data.end(), boundary, boundary + std::strlen(boundary));
    data.insert(data.end(), chat.begin(), chat.end());
    model.data = data.data();
    train_span(model, data, w, chat_at, data.size());
    train_span(model, data, w, chat_at, data.size());
    model.compact_match();
    std::cerr << "trie trained, held out " << held_idx.size() << " dolly turns\n";

    // Index the training bytes only: wiki train + boundary + chat. No wiki holdout.
    std::vector<std::uint8_t> T(data.begin(), data.begin() + kTrain);
    T.insert(T.end(), data.begin() + kSample, data.end());
    InfiniGram ig(T.data(), T.size());
    std::cerr << "index built over " << T.size() << " bytes\n";

    const std::size_t H = kSample - kTrain, half = H / 2;
    std::vector<double> pt(H);
    for (std::size_t k = 0; k < H; ++k) {
        const std::size_t i = kTrain + k;
        double e[5][256];
        bool em = false;
        unsigned pred = 0;
        model.fill_experts(data.data(), i, kTrain, Mode::Synth, e, em, pred, false);
        double M[256];
        mix_dist(w, e, em, pred, M);
        pt[k] = M[data[i]];
    }
    double bits_trie = 0;
    for (std::size_t k = half; k < H; ++k) bits_trie += -std::log2(pt[k]);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "trie_only_bpc " << bits_trie / static_cast<double>(H - half) << "\n";

    const int caps[] = {4, 16, 64, 512, 8192};
    double lam8k[kBuckets] = {0};
    std::vector<int> n8k(H);
    for (int cap : caps) {
        std::vector<double> pinf(H);
        std::vector<int> bk(H);
        int prev = -1;
        for (std::size_t k = 0; k < H; ++k) {
            const std::size_t i = kTrain + k;
            const std::size_t len = std::min<std::size_t>(i, static_cast<std::size_t>(cap));
            const int hint = prev < 0 ? -1 : std::min(prev + 1, cap);
            const auto r = ig.query(data.data() + (i - len), len, cap, hint);
            prev = r.n;
            pinf[k] = r.total ? static_cast<double>(r.count[data[i]]) / static_cast<double>(r.total) : 0.0;
            bk[k] = bucket(r.n, r.total);
            if (cap == 8192) n8k[k] = r.n;
        }
        double lam[kBuckets];
        for (int b = 0; b < kBuckets; ++b) {
            double best = -1e300;
            lam[b] = 0;
            for (int g = 0; g < 200; ++g) {
                const double l = g * 0.005;
                double ll = 0;
                bool any = false;
                for (std::size_t k = 0; k < half; ++k)
                    if (bk[k] == b) {
                        ll += std::log(l * pinf[k] + (1 - l) * pt[k]);
                        any = true;
                    }
                if (!any) break;
                if (ll > best) {
                    best = ll;
                    lam[b] = l;
                }
            }
        }
        double bits = 0;
        for (std::size_t k = half; k < H; ++k) bits += -std::log2(lam[bk[k]] * pinf[k] + (1 - lam[bk[k]]) * pt[k]);
        std::cout << "trie_plus_context_" << cap << "_bpc " << bits / static_cast<double>(H - half) << "\n";
        if (cap == 8192) std::copy(lam, lam + kBuckets, lam8k);
    }

    const int edges[] = {4, 8, 16, 32, 64, 256, 1024, 8193};
    const char* names[] = {"<4", "4-7", "8-15", "16-31", "32-63", "64-255", "256-1023", "1024-8192"};
    std::size_t cnt[8] = {0};
    for (int n : n8k)
        for (int e = 0; e < 8; ++e)
            if (n < edges[e]) {
                ++cnt[e];
                break;
            }
    std::vector<int> sorted = n8k;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(H / 2), sorted.end());
    std::cout << "median_matched_context_bytes " << sorted[H / 2] << "\n";
    for (int e = 0; e < 8; ++e) std::cout << "matched " << names[e] << " " << 100.0 * cnt[e] / H << "%\n";

    auto gen = [&](const std::string& prompt, int& n0) {
        std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
        int prev = -1;
        n0 = -1;
        for (int t = 0; t < 150; ++t) {
            double e[5][256];
            bool m = false;
            unsigned pred = 0;
            model.fill_experts(seq.data(), seq.size(), data.size(), Mode::Synth, e, m, pred, false);
            double M[256];
            mix_dist(w, e, m, pred, M);
            const std::size_t len = std::min<std::size_t>(seq.size(), 8192);
            const auto r =
                ig.query(seq.data() + (seq.size() - len), len, 8192, prev < 0 ? -1 : std::min(prev + 1, 8192));
            prev = r.n;
            if (t == 0) n0 = r.n;
            const double l = lam8k[bucket(r.n, r.total)];
            int best = 0;
            double bp = -1;
            for (int s = 0; s < 256; ++s) {
                const double pc = r.total ? static_cast<double>(r.count[s]) / static_cast<double>(r.total) : 0.0;
                const double p = l * pc + (1 - l) * M[s];
                if (p > bp) {
                    bp = p;
                    best = s;
                }
            }
            seq.push_back(static_cast<std::uint8_t>(best));
        }
        return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
    };

    struct Q {
        std::string kind, prompt, expect;
    };
    std::vector<Q> qs;
    const std::size_t picks[3] = {0, turns.size() / 2, turns.size() - 1};
    for (auto p : picks) qs.push_back({"seen", turns[p].prompt, turns[p].answer.substr(0, 30)});
    for (auto p : picks) {
        std::string s = turns[p].prompt;
        s[6] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[6])));
        qs.push_back({"change_first_letter", s, turns[p].answer.substr(0, 30)});
    }
    for (auto p : picks) {
        const std::string& s = turns[p].prompt;
        qs.push_back({"add_Thanks_at_end", s.substr(0, s.size() - 12) + " Thanks.\nAssistant: ", turns[p].answer.substr(0, 30)});
    }
    qs.push_back({"paraphrase", "User: When did Virgin Blue begin flying?\nAssistant: ", "2000"});
    qs.push_back({"paraphrase", "User: Who are the cast members of Southern Charm?\nAssistant: ", "Conover"});
    qs.push_back({"paraphrase", "User: Tell me about the Masters golf tournament.\nAssistant: ", "Augusta"});
    for (auto j : held_idx) qs.push_back({"heldout_dolly", turns[j].prompt, turns[j].answer.substr(0, 30)});
    const std::pair<const char*, const char*> wiki[] = {
        {"Who wrote Atlas Shrugged?", "Rand"},          {"What is the capital of Algeria?", "Algiers"},
        {"Who was Aristotle's teacher?", "Plato"},      {"What is the capital of Andorra?", "Andorra la Vella"},
        {"Who composed An American in Paris?", "Gershwin"}, {"What is the capital of Albania?", "Tirana"},
        {"Who taught Alexander the Great?", "Aristotle"}, {"What does an aardvark eat?", "ant"}};
    for (auto& [q, a] : wiki) qs.push_back({"wiki_fact", std::string("User: ") + q + "\nAssistant: ", a});

    std::map<std::string, std::pair<int, int>> tally;
    for (auto& q : qs) {
        int n0 = 0;
        const std::string g = gen(q.prompt, n0);
        const bool ok = lowered(g).find(lowered(q.expect)) != std::string::npos;
        tally[q.kind].first += ok;
        tally[q.kind].second += 1;
        std::cout << "== " << q.kind << " ok=" << ok << " matched_ctx=" << n0 << " | Q: " << printable(q.prompt.substr(0, 90))
                  << "\n";
        std::cout << "   want: " << printable(q.expect) << "\n   got:  " << printable(g.substr(0, 130)) << "\n";
        std::cout << "GEN\t" << q.kind << "\t" << printable(g) << "\n";
    }
    for (auto& [k, v] : tally) std::cout << "TALLY " << k << " " << v.first << "/" << v.second << "\n";
    return 0;
}
