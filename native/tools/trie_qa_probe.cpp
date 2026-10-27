// What CyphaLM (the packed trie in trie_chat.cpp) writes after a question.
//
// Trains exactly as trie_chat.cpp (wiki [0, 7340032), then two passes over
// Dolly after a <|doc|> break) by reusing that file's model code. Then asks:
//   seen                 a Dolly prompt byte for byte (turns 0, n/2, n-1)
//   change_first_letter  the same prompt with its first letter lowercased
//   wiki_fact            a fact in the 8 MiB Wikipedia slice, not in Dolly
// and prints what the order-4 mixture writes (greedy, and nucleus 0.9 at T = 0.8).
//
// Measured 2026-10-26 (enwik8 first 8 MiB, which equals enwik9's; Dolly 15k):
//   the mixture writes the same text after every prompt, because every
//   prompt ends in "\nAssistant: " and orders <= 4 see only "nt: ". The
//   earlier "answers" came from copied_answer(), a std::string::find of the
//   whole prompt (975/975 verbatim, 0/975 with one letter lowercased). That
//   search is not CyphaLM and was removed on 2026-10-27, with its measurement
//   here. Report: docs/reports/CYPHALM_QA_AND_CONTEXT_TEST.md
//
// Usage: trie_qa_probe [enwik9_or_enwik8] [dolly.jsonl] | surface_controls [enwik]
//        (the GEN lines give the surface score of the nucleus samples: 0.775)

#define main trie_chat_main
#include "trie_chat.cpp"
#undef main

#include <cctype>

namespace {

std::string sample_mix(const Model& model, const double w[6], const std::string& prompt, std::size_t limit,
                       std::uint64_t seed, bool greedy, int n) {
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int t = 0; t < n; ++t) {
        double e[5][256];
        bool matched = false;
        unsigned pred = 0;
        model.fill_experts(seq.data(), seq.size(), limit, Mode::Synth, e, matched, pred, false);
        double M[256];
        mix_dist(w, e, matched, pred, M);
        int pick = 0;
        if (greedy) {
            for (int s = 1; s < 256; ++s)
                if (M[s] > M[pick]) pick = s;
        } else {
            struct Item {
                int y;
                double p;
            } items[256];
            double mass = 0;
            for (int s = 0; s < 256; ++s) {
                items[s] = {s, std::pow(std::max(M[s], 0.0), 1.0 / kTemperature)};
                mass += items[s].p;
            }
            std::sort(items, items + 256, [](const Item& a, const Item& b) { return a.p > b.p; });
            double acc = 0;
            int keep = 1;
            for (int k = 0; k < 256; ++k) {
                acc += items[k].p / mass;
                keep = k + 1;
                if (acc >= kNucleus) break;
            }
            double km = 0;
            for (int k = 0; k < keep; ++k) km += items[k].p;
            double r = U(rng) * km;
            pick = items[keep - 1].y;
            for (int k = 0; k < keep; ++k) {
                r -= items[k].p;
                if (r <= 0) {
                    pick = items[k].y;
                    break;
                }
            }
        }
        seq.push_back(static_cast<std::uint8_t>(pick));
    }
    return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
}

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
    const std::string chat = load_dolly(dolly_path, turns);
    if (turns.empty()) {
        std::cerr << "dolly missing " << dolly_path << "\n";
        return 1;
    }
    const std::size_t chat_at = data.size();
    const char* boundary = "\n\n<|doc|>\n\n";
    data.insert(data.end(), boundary, boundary + std::strlen(boundary));
    data.insert(data.end(), chat.begin(), chat.end());
    model.data = data.data();
    train_span(model, data, w, chat_at, data.size());
    train_span(model, data, w, chat_at, data.size());
    model.compact_match();
    std::cerr << "trained turns " << turns.size() << " w4 " << w[4] << " wmatch " << w[5] << "\n";

    struct Q {
        std::string kind, prompt, expect;
    };
    std::vector<Q> qs;
    const std::size_t picks[3] = {0, turns.size() / 2, turns.size() - 1};
    for (auto p : picks) qs.push_back({"seen", turns[p].prompt, turns[p].answer.substr(0, 40)});
    for (auto p : picks) {
        std::string s = turns[p].prompt;
        s[6] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[6])));
        qs.push_back({"change_first_letter", s, turns[p].answer.substr(0, 40)});
    }
    const std::pair<const char*, const char*> wiki[] = {
        {"Who wrote Atlas Shrugged?", "Rand"},
        {"What is the capital of Algeria?", "Algiers"},
        {"Who was Aristotle's teacher?", "Plato"},
        {"What is the capital of Andorra?", "Andorra la Vella"},
        {"Who composed An American in Paris?", "Gershwin"},
        {"What is the capital of Albania?", "Tirana"},
        {"Who taught Alexander the Great?", "Aristotle"},
        {"What does an aardvark eat?", "ant"},
    };
    for (auto& [q, a] : wiki) qs.push_back({"wiki_fact", std::string("User: ") + q + "\nAssistant: ", a});

    for (auto& q : qs) {
        const std::string g = sample_mix(model, w, q.prompt, data.size(), 0, true, 120);
        std::string s = sample_mix(model, w, q.prompt, data.size(), kSeed, false, 160);
        const bool hit = lowered(g + s).find(lowered(q.expect)) != std::string::npos;
        std::cout << "== " << q.kind << " | " << printable(q.prompt.substr(0, 110)) << "\n";
        std::cout << "   expect: " << printable(q.expect) << "\n";
        std::cout << "   model_contains_expect: " << (hit ? 1 : 0) << "\n";
        std::cout << "   greedy: " << printable(g.substr(0, 120)) << "\n";
        std::cout << "   sample: " << printable(s.substr(0, 160)) << "\n";
        std::cout << "GEN\tqa_probe_nucleus\t" << printable(s) << "\n";
    }

    return 0;
}
