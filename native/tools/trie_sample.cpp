// Real generated text from the packed trie, for the SCRIBE control ladder (docs/PLAN.md 6.4).
//
// The trie's own training loop (trie_chat.cpp, as reused by trie_qa_probe.cpp) is run on wiki bytes
// only: enwik8 [0, 7340032), no Dolly, no chat. Then, for prompts taken from the HOLDOUT
// [7340032, 8388608), it samples continuations from the learned mixture (EG-weighted experts:
// unigram, order-1, orders 2-4 and the 4-byte match) with temperature T and nucleus 0.9, at
// T = 0.6, 0.8 and 1.0. The match expert may only copy from bytes before 7340032, so a prompt's own
// paragraph can never be copied back. Decoding keeps trie_chat's 11-byte no-repeat rule (a byte that
// would complete a second copy of the last 11 bytes is banned, unless that bans everything).
//
// Prompts: the first 64 bytes (cut back to a UTF-8 character boundary) of paragraph lines in the
// holdout: lines of at least 100 bytes that start with a capital letter or ''' (so not lists,
// tables, templates, headings or links). The candidates are shuffled with a fixed seed and dealt
// to the output files in order, so the files use DISJOINT prompts: the first file is for
// developing a metric, the second for a fresh test. Every prompt is sampled at all three
// temperatures, with a separate random stream per (prompt, temperature).
//
// The trie's training text [0, 7 MiB) lies before every range SCRIBE uses ([16, 72) MiB of enwik8),
// so the model under test and the metric's reference text are disjoint (PLAN V5).
//
// Output, one line each (newlines as \n, bytes outside printable ASCII as \xNN, as trie_qa_probe):
//   PROMPT<TAB>id<TAB>text         the 64-byte prompt (ignored by scribe_ladder)
//   GEN<TAB>trie_T0.8<TAB>text     the continuation (the prompt is NOT part of the text)
//
// Usage: trie_sample <enwik8> <out_dev.txt> [out_fresh.txt] [--prompts 250] [--bytes 600]
//   g++ -O2 -std=c++20 -I native/third_party native/tools/trie_sample.cpp -o trie_sample
// Measured 2026-10-27 on a shared 2-core VM: the whole run (training on 7 MiB, then 2 files x 250 prompts x
// 3 temperatures x 600 bytes = 1,500 continuations) takes about 90 seconds on one core. The trie has
// 404,601 nodes; the learned mixture ends with weight 0.93 on the order-4 expert and 0.06 on the match.
// What it writes is word-salad-like text (see docs/reports/SCRIBE_V1.md): the order-4 byte model rarely
// closes a sentence in 600 bytes.

#define main trie_chat_main
#include "trie_chat.cpp"
#undef main

#include <cstdio>
#include <cstdlib>

namespace {

// Same loop as trie_qa_probe.cpp's train_span: score each byte from counts so far, move the weights,
// then observe the byte.
void train_span(Model& model, const std::vector<std::uint8_t>& data, double w[6], std::size_t a, std::size_t b) {
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
        if (((i + 1) & 0xfffff) == 0) std::cerr << "train " << (i + 1) << " nodes " << model.nodes.size() << "\n";
    }
}

// One continuation: n bytes from the mixture at temperature T, nucleus kNucleus, no-repeat-11 rule.
std::string sample_text(const Model& model, const double w[6], const std::string& prompt, int n, double T,
                        std::uint64_t seed) {
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    std::mt19937 rng(static_cast<std::uint32_t>(seed ^ (seed >> 32)));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int t = 0; t < n; ++t) {
        double e[5][256];
        bool matched = false;
        unsigned pred = 0;
        model.fill_experts(seq.data(), seq.size(), kTrain, Mode::Synth, e, matched, pred, false);
        double M[256];
        mix_dist(w, e, matched, pred, M);
        bool banned[256];
        mark_banned(seq, banned);
        struct Item {
            int y;
            double p;
        } items[256];
        double mass = 0;
        for (int pass = 0; pass < 2 && !(mass > 0); ++pass) {
            mass = 0;
            for (int s = 0; s < 256; ++s) {
                const double base = (pass == 0 && banned[s]) ? 0.0 : std::pow(std::max(M[s], 0.0), 1.0 / T);
                items[s] = {s, base};
                mass += base;
            }
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
        int pick = items[keep - 1].y;
        for (int k = 0; k < keep; ++k) {
            r -= items[k].p;
            if (r <= 0) {
                pick = items[k].y;
                break;
            }
        }
        seq.push_back(static_cast<std::uint8_t>(pick));
    }
    return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
}

bool good_start(const std::string& l) {
    if (l.size() < 100) return false;
    const unsigned char c = static_cast<unsigned char>(l[0]);
    if (l.compare(0, 3, "'''") == 0) return true;
    return c >= 'A' && c <= 'Z';
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::vector<std::string> paths;
    int n_prompts = 250, n_bytes = 600;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prompts" && i + 1 < argc) n_prompts = std::atoi(argv[++i]);
        else if (a == "--bytes" && i + 1 < argc) n_bytes = std::atoi(argv[++i]);
        else paths.push_back(a);
    }
    if (paths.size() < 2 || paths.size() > 3 || n_prompts < 1 || n_bytes < 100) {
        std::fprintf(stderr, "usage: trie_sample <enwik8> <out_dev.txt> [out_fresh.txt] [--prompts 250] [--bytes 600]\n");
        return 2;
    }
    if (!self_test()) return 1;

    std::ifstream in(paths[0], std::ios::binary);
    std::vector<std::uint8_t> data(kSample);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(kSample));
    if (!in || static_cast<std::size_t>(in.gcount()) != kSample) {
        std::cerr << "cannot read " << kSample << " bytes of " << paths[0] << "\n";
        return 1;
    }

    // Prompts from the holdout.
    std::vector<std::string> prompts;
    {
        const std::string hold(reinterpret_cast<const char*>(data.data()) + kTrain, kSample - kTrain);
        std::size_t pos = hold.find('\n');  // the first line is cut by the range
        pos = pos == std::string::npos ? hold.size() : pos + 1;
        while (pos < hold.size()) {
            std::size_t e = hold.find('\n', pos);
            if (e == std::string::npos) break;  // the last line is cut by the range
            const std::string line = hold.substr(pos, e - pos);
            pos = e + 1;
            if (!good_start(line)) continue;
            std::size_t cut = 64;
            while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) --cut;  // character boundary
            prompts.push_back(line.substr(0, cut));
        }
    }
    std::mt19937_64 prng(20261027);
    std::shuffle(prompts.begin(), prompts.end(), prng);
    const std::size_t need = static_cast<std::size_t>(n_prompts) * paths.size() - static_cast<std::size_t>(n_prompts);
    std::cerr << "holdout prompts " << prompts.size() << ", using " << need << "\n";
    if (prompts.size() < need) {
        std::cerr << "too few holdout paragraphs for " << need << " prompts\n";
        return 1;
    }

    // Train on wiki bytes only.
    Model model;
    model.data = data.data();
    model.init();
    double w[6];
    w[5] = 0.15;
    for (int i = 0; i < 5; ++i) w[i] = 0.85 / 5.0;
    train_span(model, data, w, 0, kTrain);
    model.compact_match();
    std::cerr << "trained on [0, " << kTrain << ") nodes " << model.nodes.size() << " w4 " << w[4] << " wmatch " << w[5]
              << "\n";

    const double temps[3] = {0.6, 0.8, 1.0};
    for (std::size_t f = 0; f + 1 < paths.size(); ++f) {
        std::ofstream out(paths[f + 1]);
        if (!out) {
            std::cerr << "cannot write " << paths[f + 1] << "\n";
            return 1;
        }
        for (int k = 0; k < n_prompts; ++k) {
            const std::size_t id = f * static_cast<std::size_t>(n_prompts) + static_cast<std::size_t>(k);
            const std::string& p = prompts[id];
            out << "PROMPT\t" << id << "\t" << printable(p) << "\n";
            for (int ti = 0; ti < 3; ++ti) {
                const std::string text = sample_text(model, w, p, n_bytes, temps[ti], 1000003ull * (id + 1) + static_cast<std::uint64_t>(ti) * 7919ull);
                char kind[32];
                std::snprintf(kind, sizeof kind, "trie_T%.1f", temps[ti]);
                out << "GEN\t" << kind << "\t" << printable(text) << "\n";
            }
            if ((k + 1) % 50 == 0) std::cerr << "file " << f << " prompt " << (k + 1) << "/" << n_prompts << "\n";
        }
    }
    return 0;
}
