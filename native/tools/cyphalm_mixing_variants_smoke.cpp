/// Flag-gated mixing variants (MixingOptions; manifest keys final_temperature,
/// final_temperature_lr, infinigram_mode, neural_mix, ensemble_gate) on a
/// manifest-style composite model (two shards, ∞-gram over a corpus, two
/// small Transformer experts):
///  - InfiniGram::query with min_total: the longest suffix followed by a byte
///    at least k times, against brute force; bounded (hinted) queries along a
///    stream, with the previous ≥16 length + 1 as the bound, equal unbounded.
///  - Defaults written as keys serve bit for bit what a manifest without them
///    serves; save_cyphalm_ensemble_manifest writes only non-default keys and
///    loading gives them back; unknown names throw.
///  - Every variant serves normalised finite distributions, observe returns
///    the served log p, results repeat exactly across loads and with the
///    worker pool off, the learned state moves while mixing weights learn and
///    set_mixing_state restores it.
///  - A fixed final temperature is log softmax(lp / T) of the untempered
///    model's distribution, bit for bit (the harness's temperature scan), and
///    learns nothing else; a learned one moves only while learning.
///  - The ensemble gate serves today's mix until it has learned.
///  - Longest16 has 2048 ∞-gram buckets and refuses a 256-bucket weights file.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/cyphalm/infinigram.hpp"

namespace {

namespace fs = std::filesystem;
using cypha::cyphalm::CyphaLMModel;
using cypha::cyphalm::HpSequenceBackend;
using cypha::cyphalm::InfiniGram;
using cypha::cyphalm::InfinigramMode;
using cypha::cyphalm::MixingOptions;
using cypha::cyphalm::NeuralMix;

int fail(const std::string& what) {
    std::printf("cyphalm_mixing_variants_smoke FAIL: %s\n", what.c_str());
    return 1;
}

std::string corpus(unsigned seed, int variant, std::size_t n) {
    const char* a[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "Then ", "dog ", "ran ", "\n"};
    const char* b[] = {"insects ", "have ", "six ", "legs ", "and ", "wings. ", "The ", "body ", "\n"};
    std::mt19937 rng(seed);
    std::string t;
    while (t.size() < n) t += variant == 0 ? a[rng() % 10] : b[rng() % 9];
    t.resize(n);
    return t;
}

/// A small random Transformer (BGT1), as in cyphalm_neural_smoke.
void write_bgt(const fs::path& path, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    const std::uint32_t gl = 2, gd = 16, gh = 2, gctx = 8;
    std::ofstream os(path, std::ios::binary);
    os.write("BGT1", 4);
    const std::uint32_t hdr[4] = {gl, gd, gh, gctx};
    os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    auto put = [&](std::size_t n, float mean) {
        for (std::size_t i = 0; i < n; ++i) {
            const float f = mean + nd(rng);
            os.write(reinterpret_cast<const char*>(&f), sizeof(f));
        }
    };
    put(256 * gd, 0.0f);
    put(gctx * gd, 0.0f);
    for (std::uint32_t l = 0; l < gl; ++l) {
        put(gd, 1.0f);
        put(gd, 0.0f);
        put(3 * gd * gd, 0.0f);
        put(gd * gd, 0.0f);
        put(gd, 1.0f);
        put(gd, 0.0f);
        put(4 * gd * gd, 0.0f);
        put(4 * gd * gd, 0.0f);
    }
    put(gd, 1.0f);
    put(gd, 0.0f);
}

/// The base manifest's keys: two shards, the ∞-gram over a corpus, two
/// adapting experts.
nlohmann::json base_meta() {
    return {{"cyphalm_ensemble", 1},
            {"members", {{{"checkpoint", "shard_a.json"}}, {{"checkpoint", "shard_b.json"}}}},
            {"learning_rate", 0.05},
            {"infinigram", "corpus.txt"},
            {"neural", {"expert_a.bgt", "expert_b.bgt"}},
            {"neural_learning_rate", 0.1},
            {"neural_adapt", 0.01}};
}

fs::path write_manifest(const fs::path& dir, const std::string& name, const nlohmann::json& extra) {
    nlohmann::json meta = base_meta();
    for (auto it = extra.begin(); it != extra.end(); ++it) meta[it.key()] = it.value();
    const fs::path p = dir / (name + ".json");
    std::ofstream(p) << meta.dump(1);
    return p;
}

void write_parts(const fs::path& dir) {
    fs::create_directories(dir);
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.hp_table_bits = 16;
    const std::string ta = corpus(1, 0, 6000), tb = corpus(2, 1, 6000);
    CyphaLMModel a(cfg), b(cfg);
    for (unsigned char c : ta) a.hp_backend().consume_byte(c);
    for (unsigned char c : tb) b.hp_backend().consume_byte(c);
    cypha::cyphalm::save_cyphalm_model(a, (dir / "shard_a").string());
    cypha::cyphalm::save_cyphalm_model(b, (dir / "shard_b").string());
    {
        std::ofstream os(dir / "corpus.txt", std::ios::binary);
        os << corpus(3, 1, 12000) << corpus(4, 0, 12000);
    }
    write_bgt(dir / "expert_a.bgt", 5);
    write_bgt(dir / "expert_b.bgt", 6);
}

/// Text the model reads: both vocabularies, then a stretch it never saw.
std::string eval_text() {
    return corpus(11, 1, 500) + corpus(12, 0, 500) + "Zebras quietly graze on 42 hills. " + corpus(13, 1, 300);
}

struct Run {
    std::vector<std::vector<double>> dists;  // served distribution per byte
    double nll = 0.0;
    HpSequenceBackend::MixingState state;
};

/// Score and observe every byte of ``text``; checks normalisation and that
/// observe returns the served log p.
Run read_text(CyphaLMModel& m, const std::string& text, std::string& err) {
    Run run;
    auto& h = m.hp_backend();
    for (unsigned char c : text) {
        const auto lp = h.next_byte_log_probs(256);
        double mx = -std::numeric_limits<double>::infinity();
        for (double v : lp) {
            if (!std::isfinite(v)) err = "non-finite log p";
            mx = std::max(mx, v);
        }
        double z = 0.0;
        for (double v : lp) z += std::exp(v - mx);
        // The experts' own log softmax rounds logit differences to float
        // (~1e-7), so today's linear mix is normalised to that too.
        if (std::abs(mx + std::log(z)) > 1e-6) err = "served distribution not normalised";
        const double loss = h.observe_next_byte(c);
        if (loss != -lp[c]) err = "observe does not return the served log p";
        run.nll += loss;
        run.dists.push_back(lp);
    }
    run.state = h.mixing_state();
    return run;
}

int check_query_min_total() {
    std::mt19937 rng(9);
    const std::string text = corpus(21, 0, 6000) + corpus(22, 1, 3000);
    const std::size_t n = text.size();
    InfiniGram ig(reinterpret_cast<const std::uint8_t*>(text.data()), n);
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t p = rng() % (n - 40);
        std::string ctx = text.substr(p, 1 + rng() % 30);
        if (trial % 5 == 0) ctx[rng() % ctx.size()] = 'Z';
        const std::uint64_t k = std::uint64_t{1} << (trial % 6);  // 1 .. 32
        const auto r = ig.query(reinterpret_cast<const std::uint8_t*>(ctx.data()), ctx.size(), 64, -1, k);
        // Brute force: the longest suffix followed by a byte at least k times.
        int best = 0;
        std::vector<std::uint32_t> cnt(256, 0);
        for (int m = static_cast<int>(ctx.size()); m >= 0; --m) {
            std::vector<std::uint32_t> c(256, 0);
            std::uint64_t tot = 0;
            const std::string suf = ctx.substr(ctx.size() - static_cast<std::size_t>(m));
            for (std::size_t i = 0; i + static_cast<std::size_t>(m) < n; ++i) {
                if (m == 0 || text.compare(i, static_cast<std::size_t>(m), suf) == 0) {
                    ++c[static_cast<unsigned char>(text[i + static_cast<std::size_t>(m)])];
                    ++tot;
                }
            }
            if (tot >= k || m == 0) {
                best = m;
                cnt = c;
                break;
            }
        }
        bool same = r.n == best;
        for (int b = 0; b < 256 && same; ++b) same = r.count[b] == cnt[b];
        if (!same) return fail("min_total query differs from brute force (trial " + std::to_string(trial) + ")");
    }
    // Along a stream: the ≥16 length is bounded by the longest match - 1
    // (when that one has fewer) and by the previous ≥16 length + 1.
    std::string stream = text.substr(2000, 2500);
    for (std::size_t i = 97; i < stream.size(); i += 211) stream[i] = 'Z';
    int prev = 0;
    for (std::size_t t = 1; t <= stream.size(); ++t) {
        const std::size_t len = std::min<std::size_t>(t, 64);
        const auto* c = reinterpret_cast<const std::uint8_t*>(stream.data()) + (t - len);
        const auto plain = ig.query(c, len, 64, -1, 16);
        const auto longest = ig.query(c, len, 64);
        if (longest.total < 16) {
            if (plain.n >= longest.n) return fail("≥16 suffix not shorter than a longest match seen < 16 times");
            const auto hinted = ig.query(c, len, longest.n - 1, std::min(longest.n - 1, prev + 1), 16);
            if (hinted.n != plain.n || hinted.total != plain.total || hinted.count != plain.count)
                return fail("bounded ≥16 query differs at " + std::to_string(t));
        } else if (plain.n != longest.n) {
            return fail("longest match seen ≥16 times is not the ≥16 suffix");
        }
        if (plain.total < 16) return fail("≥16 suffix seen fewer than 16 times");
        prev = plain.n;
    }
    return 0;
}

int check_manifest_keys(const fs::path& dir) {
    // Only non-default keys are written; loading gives them back.
    MixingOptions o;
    o.final_temperature = 0.9;
    o.final_temperature_lr = 0.001;
    o.infinigram_mode = InfinigramMode::Longest16;
    o.neural_mix = NeuralMix::Switch;
    o.ensemble_gate = true;
    const fs::path saved = dir / "saved_opts.json";
    cypha::cyphalm::save_cyphalm_ensemble_manifest(saved.string(), {"shard_a.json", "shard_b.json"}, 0.05, o);
    nlohmann::json j;
    std::ifstream(saved) >> j;
    if (j.value("final_temperature", 0.0) != 0.9 || j.value("final_temperature_lr", 0.0) != 0.001 ||
        j.value("infinigram_mode", std::string()) != "longest16" || j.value("neural_mix", std::string()) != "switch" ||
        !j.value("ensemble_gate", false))
        return fail("saved manifest misses a mixing key");
    auto m = cypha::cyphalm::load_cyphalm_model(saved.string());
    if (!(m.hp_backend().mixing_options() == o)) return fail("manifest mixing options do not round-trip");
    const fs::path plain = dir / "saved_plain.json";
    cypha::cyphalm::save_cyphalm_ensemble_manifest(plain.string(), {"shard_a.json", "shard_b.json"}, 0.05,
                                                   MixingOptions{});
    std::ifstream(plain) >> j;
    for (const char* k : {"final_temperature", "final_temperature_lr", "infinigram_mode", "neural_mix", "ensemble_gate"})
        if (j.contains(k)) return fail(std::string("default key written: ") + k);
    if (!(cypha::cyphalm::load_cyphalm_model(plain.string()).hp_backend().mixing_options() == MixingOptions{}))
        return fail("manifest without keys does not load the defaults");
    for (const auto& bad : {nlohmann::json{{"neural_mix", "geometric"}}, nlohmann::json{{"infinigram_mode", "x"}},
                            nlohmann::json{{"final_temperature", 0.0}}}) {
        bool threw = false;
        try {
            (void)cypha::cyphalm::load_cyphalm_model(write_manifest(dir, "bad", bad).string());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) return fail("bad mixing key accepted: " + bad.dump());
    }
    return 0;
}

int check_variants(const fs::path& dir) {
    const std::string text = eval_text();
    std::string err;
    auto base_m = cypha::cyphalm::load_cyphalm_model(write_manifest(dir, "base", nlohmann::json::object()).string());
    const auto start = base_m.hp_backend().mixing_state();
    const Run base = read_text(base_m, text, err);
    if (!err.empty()) return fail("base: " + err);

    // Defaults written as keys: the same model, bit for bit.
    {
        const nlohmann::json keys = {{"final_temperature", 1.0}, {"final_temperature_lr", 0.0},
                                     {"infinigram_mode", "halving"}, {"neural_mix", "linear"},
                                     {"ensemble_gate", false}};
        auto m = cypha::cyphalm::load_cyphalm_model(write_manifest(dir, "defaults", keys).string());
        const Run r = read_text(m, text, err);
        if (r.dists != base.dists || r.nll != base.nll || r.state != base.state)
            return fail("explicit default keys change the served distributions");
    }

    struct Variant {
        const char* name;
        nlohmann::json keys;
    };
    const Variant variants[] = {
        {"final_temperature", {{"final_temperature", 0.9}}},
        {"final_temperature_lr", {{"final_temperature", 0.9}, {"final_temperature_lr", 0.01}}},
        {"longest16", {{"infinigram_mode", "longest16"}}},
        {"log", {{"neural_mix", "log"}}},
        {"switch", {{"neural_mix", "switch"}}},
        {"gate", {{"ensemble_gate", true}}},
        {"all",
         {{"final_temperature", 0.9}, {"final_temperature_lr", 0.01}, {"infinigram_mode", "longest16"},
          {"neural_mix", "switch"}, {"ensemble_gate", true}}},
    };
    for (const Variant& v : variants) {
        const std::string tag = std::string(v.name) + ": ";
        const fs::path mp = write_manifest(dir, v.name, v.keys);
        auto m1 = cypha::cyphalm::load_cyphalm_model(mp.string());
        const auto s0 = m1.hp_backend().mixing_state();
        const Run r1 = read_text(m1, text, err);
        if (!err.empty()) return fail(tag + err);
        if (r1.dists == base.dists) return fail(tag + "serves what the default serves");
        // Determinism: a fresh load, and one on the calling thread only.
        auto m2 = cypha::cyphalm::load_cyphalm_model(mp.string());
        m2.hp_backend().set_parallel(false);
        const Run r2 = read_text(m2, text, err);
        if (r2.dists != r1.dists || r2.nll != r1.nll || r2.state != r1.state)
            return fail(tag + "not repeatable (fresh load, worker pool off)");
        // Learned state moves, and set_mixing_state puts it back exactly.
        if (r1.state == s0) return fail(tag + "no mixing weight moved");
        m1.hp_backend().set_mixing_state(s0);
        if (m1.hp_backend().mixing_state() != s0) return fail(tag + "set_mixing_state does not round-trip");
    }

    // Fixed T: log softmax(lp / T) of the untempered distribution, computed
    // as the harness's scan does, bit for bit; nothing else learns
    // differently.
    {
        auto m = cypha::cyphalm::load_cyphalm_model(dir.string() + "/final_temperature.json");
        const Run r = read_text(m, text, err);
        const double t = 0.9;
        for (std::size_t k = 0; k < base.dists.size(); ++k) {
            const auto& lp = base.dists[k];
            double mx = -1e300;
            for (int b = 0; b < 256; ++b) mx = std::max(mx, lp[static_cast<std::size_t>(b)] / t);
            double z = 0.0;
            for (int b = 0; b < 256; ++b) z += std::exp(lp[static_cast<std::size_t>(b)] / t - mx);
            for (int b = 0; b < 256; ++b) {
                if (r.dists[k][static_cast<std::size_t>(b)] != lp[static_cast<std::size_t>(b)] / t - mx - std::log(z))
                    return fail("fixed final temperature is not the scan's log softmax(lp / T)");
            }
        }
        auto learned = r.state;
        learned.final_temp = base.state.final_temp;
        if (learned != base.state) return fail("a fixed final temperature changed another stage's learning");
        if (m.hp_backend().final_temperatures() != std::vector<double>(8, 0.9))
            return fail("a fixed final temperature moved");
    }
    // Learned T: moves while mixing weights learn, within its bounds; not
    // with learning off or mixing frozen; reset() starts it again.
    {
        auto m = cypha::cyphalm::load_cyphalm_model(dir.string() + "/final_temperature_lr.json");
        auto& h = m.hp_backend();
        const auto t0 = h.final_temperatures();
        h.set_learning(false);
        (void)read_text(m, text.substr(0, 200), err);
        if (h.final_temperatures() != t0) return fail("learned temperature moved with learning off");
        h.set_learning(true);
        h.set_mixing_learning(false);
        (void)read_text(m, text.substr(0, 200), err);
        if (h.final_temperatures() != t0) return fail("learned temperature moved with mixing frozen");
        h.set_mixing_learning(true);
        (void)read_text(m, text, err);
        const auto t1 = h.final_temperatures();
        if (t1 == t0) return fail("learned temperature did not move");
        for (double t : t1)
            if (!(1.0 / t >= 0.8 - 1e-12 && 1.0 / t <= 1.6 + 1e-12)) return fail("learned temperature out of bounds");
        h.reset();
        if (h.final_temperatures() != t0) return fail("reset() kept the learned temperature");
    }
    // The gate starts at today's weights: the first distribution is the
    // ungated one, bit for bit.
    {
        auto m = cypha::cyphalm::load_cyphalm_model(dir.string() + "/gate.json");
        if (m.hp_backend().next_byte_log_probs(256) != base.dists.front())
            return fail("the ensemble gate does not start at today's mix");
        if (m.hp_backend().mixing_state().gate.empty()) return fail("gate state missing from the mixing state");
    }
    // Longest16: 2048 buckets; a 256-bucket weights file is refused.
    {
        auto m = cypha::cyphalm::load_cyphalm_model(dir.string() + "/longest16.json");
        auto& h = m.hp_backend();
        if (h.infinigram_weights().size() != 2048 * 3) return fail("longest16 bucket count");
        bool threw = false;
        try {
            h.set_infinigram_weights(std::vector<double>(256 * 3, 1.0 / 3.0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) return fail("longest16 accepted halving-mode weights");
        h.set_infinigram_mode(InfinigramMode::Halving);
        if (h.infinigram_weights().size() != 256 * 3) return fail("mode change did not reshape the weights");
    }
    // Log / switch weights exist only in their modes.
    if (!start.neural_log.empty() || !start.neural_switch.empty() || !start.gate.empty())
        return fail("default model carries variant state");
    // A plain model with only a final temperature is composite and serves
    // the tempered distribution on every path.
    {
        cypha::cyphalm::CyphaLMConfig cfg;
        cypha::cyphalm::apply_hp_production_recipe(cfg);
        cfg.hp_table_bits = 16;
        CyphaLMModel p(cfg);
        auto& h = p.hp_backend();
        for (unsigned char c : corpus(31, 0, 2000)) h.consume_byte(c);
        if (h.is_composite()) return fail("plain model composite");
        h.set_final_temperature(0.8);
        if (!h.is_composite()) return fail("a final temperature is not a mixing stage");
        const auto lp = h.next_byte_log_probs(256);
        const auto top = static_cast<std::uint8_t>(std::max_element(lp.begin(), lp.end()) - lp.begin());
        if (h.log_prob_byte('t') != lp['t'] || h.serve_greedy_next_byte() != top)
            return fail("serve paths do not read the tempered distribution");
    }
    return 0;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyphalm_mixing_variants_smoke";
    fs::remove_all(dir);
    write_parts(dir);
    int rc = check_query_min_total();
    if (rc == 0) rc = check_manifest_keys(dir);
    if (rc == 0) rc = check_variants(dir);
    fs::remove_all(dir);
    if (rc == 0) std::printf("cyphalm_mixing_variants_smoke OK\n");
    return rc;
}
