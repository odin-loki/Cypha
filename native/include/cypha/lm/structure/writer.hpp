#pragma once

/// K5 with its structure experts and their meta-mixer choosing each sentence (docs/reports/K5_STRUCTURE.md).
///   sample --k5 TEXT --k5-struct MODEL [--k5-struct-temp T]
/// K5 writes its gate's K candidates per sentence (the same streams as its own gate, --k5-gate K); the mixer picks
/// one: the argmax, or a draw from P^(1/T) when T > 0. The copy cap holds inside every candidate, as in K5's gate.
///
/// Scorer is the interface for other writers (wip/k5-mcmc): score(prompt, paragraph so far, candidate) gives every
/// expert's features and the mixed score, a log-linear energy (the second layer's weights times each ranker's score).
///
/// MODEL is a text file: "K5STRUCT 1", "stats PATH", then the mixer (mixer.hpp MetaMixer::write).
///
/// Author: Odin Loch

#include "cypha/lm/structure/experts.hpp"
#include "cypha/lm/structure/mixer.hpp"
#include "cypha/lm/word_kn.hpp"

#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace cypha::cyphalm::structure {

// The weight-set index of each selector for a candidate after context c.
inline void selectors(const Context& c, int* sel) {
    sel[0] = 0;
    sel[1] = static_cast<int>(std::min<std::size_t>(c.index(), 3)) * 3 + c.prompt_bucket();
    sel[2] = c.po1;
}

// The mean log2 probability per word of ids after state s under K5 (<unk> and <s> removed, as the gate's fluency).
inline double k5_mean_lp(const wkn::K5& m, wkn::K5::State s, const std::vector<std::uint32_t>& ids, const wkn::Options& o) {
    std::vector<double> p(m.vocab().size()), tmp;
    double t = 0;
    for (const auto w : ids) {
        m.model_dist(s, o, p.data(), tmp);
        t += std::log2(std::max(p[w] / std::max(1 - p[wkn::kUnk] - p[wkn::kBos], 1e-300), 1e-300));
        wkn::K5::learn(s, w, o);
    }
    return ids.empty() ? 0 : t / static_cast<double>(ids.size());
}
inline double mean_assoc(const std::vector<double>& a, const std::vector<std::uint32_t>& ids) {
    double t = 0;
    for (const auto w : ids) t += a.empty() ? 0.0 : a[w];
    return ids.empty() ? 0 : t / static_cast<double>(ids.size());
}

class Model {
 public:
    Stats st;
    MetaMixer mx;
    std::string stats_path;
    lda::LDA topics;       // STATS.lda, if it exists (wip/k5-discourse)
    eg::EntityGrid grid;   // STATS.grid, if it exists
    // The discourse experts beside a statistics file.
    void load_discourse() {
        if (std::ifstream(stats_path + ".lda")) topics.load(stats_path + ".lda");
        if (std::ifstream(stats_path + ".grid")) grid.load(stats_path + ".grid");
    }
    bool load(const std::string& path, std::string& err) {
        std::ifstream in(path);
        std::string tag;
        int ver = 0;
        if (!in || !(in >> tag >> ver) || tag != "K5STRUCT" || ver != 1) return err = "cannot read the structure model " + path, false;
        if (!(in >> tag >> stats_path) || tag != "stats") return err = path + ": no stats line", false;
        if (!mx.read(in, kNFeat)) return err = path + ": bad mixer", false;
        if (!st.load(stats_path, err)) return false;
        load_discourse();
        return true;
    }
    // K5's training runs, built once from the K5 the writer is given.
    const TrainRuns& runs(const wkn::K5& m) const {
        std::call_once(once_, [&] { tr_.build(m.train_ids()); });
        return tr_;
    }

 private:
    mutable std::once_flag once_;
    mutable TrainRuns tr_;
};

// Raw features into an Item row (z = f / sd).
inline void put_row(const MetaMixer& mx, const double* f, float* z) {
    for (int j = 0; j < kNFeat; ++j) z[j] = static_cast<float>(f[j] / (mx.sd.empty() || mx.sd[static_cast<std::size_t>(j)] <= 0 ? 1.0 : mx.sd[static_cast<std::size_t>(j)]));
}

class Writer final : public wkn::StructureWriter {
 public:
    Writer(std::shared_ptr<const Model> md, double temp) : md_(std::move(md)), temp_(temp) {}
    std::string generate(const wkn::K5& m, std::string_view prompt, std::uint64_t seed, const wkn::Options& o, wkn::GenStats* st) const override {
        wkn::GenStats g;
        if (m.vocab().size() <= 2) return {};
        const Experts ex(md_->st, m, md_->runs(m), &md_->topics, &md_->grid);
        wkn::K5::State s = m.start(prompt, o);
        const std::vector<double> assoc = m.prompt_assoc(s.ctx);
        Context c;
        c.sents = split(wkn::tokens(prompt));
        c.n_prompt = c.sents.size();
        c.prompt_chars = prompt.size();
        const int K = std::max(o.gate_k, 1);
        std::vector<wkn::K5::State> cs(static_cast<std::size_t>(K));
        std::vector<wkn::GenStats> gk(static_cast<std::size_t>(K));
        std::vector<Sent> sents(static_cast<std::size_t>(K));
        std::vector<std::uint32_t> words;
        std::vector<double> lp, P;
        double f[kNFeat];
        Item it;
        it.n = K;
        it.z.resize(static_cast<std::size_t>(K * kNFeat));
        for (std::uint64_t sent = 0; s.step < o.max_words && s.out.size() < o.bytes; ++sent) {
            ex.prepare(c);
            selectors(c, it.sel);
            for (int k = 0; k < K; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                gk[kk] = wkn::GenStats{};
                m.draft(s, seed, sent, k, o, cs[kk], words, lp, gk[kk]);
                Sent& z = sents[kk];
                z.tok.clear();
                for (const auto w : words) z.tok.push_back(m.vocab()[w]);
                z.ended = !z.tok.empty() && (z.tok.back() == "." || z.tok.back() == "!" || z.tok.back() == "?");
                double t = 0;
                for (const double x : lp) t += x;
                ex.features(c, z, words, lp.empty() ? 0 : t / static_cast<double>(lp.size()), mean_assoc(assoc, words), f);
                put_row(md_->mx, f, it.z.data() + kk * kNFeat);
            }
            md_->mx.dist(it, P);
            int best = 0;
            if (temp_ <= 0) {
                for (int k = 1; k < K; ++k)
                    if (P[static_cast<std::size_t>(k)] > P[static_cast<std::size_t>(best)]) best = k;
            } else {
                double z = 0;
                for (double& x : P) z += (x = std::pow(std::max(x, 1e-300), 1.0 / temp_));
                std::mt19937_64 rng(wkn::mix64(seed ^ wkn::mix64(0x57AC000000ull + sent)));
                const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53 * z;
                double cdf = 0;
                for (best = 0; best + 1 < K; ++best)
                    if ((cdf += P[static_cast<std::size_t>(best)]) > u) break;
            }
            const auto bb = static_cast<std::size_t>(best);
            s = std::move(cs[bb]);
            c.sents.push_back(std::move(sents[bb]));
            g.add(gk[bb]);
            g.gate_cands += static_cast<std::size_t>(K) - 1;  // draft() counted the kept one in gk
            ++g.gate_sents;
        }
        std::string out = std::move(s.out);
        wkn::K5::cut_bytes(out, o.bytes);
        if (st) st->add(g);
        return out;
    }

 private:
    std::shared_ptr<const Model> md_;
    double temp_;
};

// For other writers (wip/k5-mcmc): every expert's features and the mixed score of one candidate sentence.
class Scorer {
 public:
    Scorer(std::shared_ptr<const Model> md, const wkn::K5& m, wkn::Options o = {}) : md_(std::move(md)), m_(m), o_(std::move(o)) {}
    struct Scores {
        double f[kNFeat];  // raw features (feat_name(j))
        double mixed = 0;  // sum over rankers of v_m * s_m: the log-linear energy (higher: the mixer prefers it)
    };
    // prompt and paragraph (the sentences written so far) as text; candidate the next sentence's text.
    Scores score(std::string_view prompt, const std::vector<std::string>& paragraph, std::string_view candidate) const {
        Context c;
        c.sents = split(wkn::tokens(prompt));
        c.n_prompt = c.sents.size();
        c.prompt_chars = prompt.size();
        std::string ctx_text(prompt);
        for (const auto& p : paragraph) {
            auto t = wkn::tokens(p);
            Sent s;
            s.tok = t;
            s.ended = !t.empty() && (t.back() == "." || t.back() == "!" || t.back() == "?");
            c.sents.push_back(std::move(s));
            ctx_text += " ";
            ctx_text += p;
        }
        const Experts ex(md_->st, m_, md_->runs(m_), &md_->topics, &md_->grid);
        ex.prepare(c);
        Sent z;
        z.tok = wkn::tokens(candidate);
        z.ended = !z.tok.empty() && (z.tok.back() == "." || z.tok.back() == "!" || z.tok.back() == "?");
        std::vector<std::uint32_t> ids;
        for (const auto& t : z.tok) ids.push_back(m_.id(t));
        const wkn::K5::State s0 = m_.start(prompt, o_);
        const std::vector<double> assoc = m_.prompt_assoc(s0.ctx);
        Scores r;
        ex.features(c, z, ids, k5_mean_lp(m_, m_.start(ctx_text, o_), ids, o_), mean_assoc(assoc, ids), r.f);
        Item it;
        it.n = 1;
        it.z.resize(kNFeat);
        put_row(md_->mx, r.f, it.z.data());
        selectors(c, it.sel);
        const auto& mx = md_->mx;
        for (std::size_t k = 0; k < mx.r.size(); ++k)
            r.mixed += (mx.r.size() == 1 ? 1.0 : mx.v[k]) * mx.r[k].score(it.sel[mx.r[k].selector()], it.z.data());
        return r;
    }

 private:
    std::shared_ptr<const Model> md_;
    const wkn::K5& m_;
    wkn::Options o_;
};

// cyphalm_trie: --k5-struct FILE [--k5-struct-temp T] into o. False and err if the model cannot be read.
inline bool apply_struct_flags(const std::string& file, double temp, wkn::Options& o, std::string& err) {
    auto md = std::make_shared<Model>();
    if (!md->load(file, err)) return false;
    o.structure = std::make_shared<Writer>(md, temp);
    char b[64];
    std::snprintf(b, sizeof b, ",T=%g", temp);
    o.structure_desc = file + b;
    return true;
}

}  // namespace cypha::cyphalm::structure
