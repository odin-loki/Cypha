/// LLM-quality harness for CyphaLM: pretrain, then score held-out text with the
/// full next-byte distribution (not the compression bit path).
///
/// Per held-out byte: full 256-way log P via serve_next_byte_log_probs, then the
/// model learns the true byte (in-context adaptation, as when reading a prompt).
/// Reports NLL (bits/byte), top-1/top-5 accuracy, expected calibration error of
/// the top-1 confidence, mean entropy, and how often bit-greedy decoding picks
/// the distribution's argmax. Ends with greedy / sampled continuations.
///
///   cyphalm_lm_quality --train enwik8 --train-bytes 8388608 --save /tmp/pre
///   cyphalm_lm_quality --load /tmp/pre.json --eval enwik8 --eval-offset 96000000 --eval-bytes 32768
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kLn2 = 0.6931471805599453;

std::vector<std::uint8_t> read_slice(const std::string& path, std::uint64_t offset,
                                     std::uint64_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> out(n);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n));
    out.resize(static_cast<std::size_t>(in.gcount()));
    return out;
}

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string printable(const std::vector<int>& ids) {
    std::string s;
    for (int b : ids) {
        if (b == '\n') s += "\\n";
        else if (b >= 32 && b < 127) s += static_cast<char>(b);
        else s += '?';
    }
    return s;
}

double distinct_ngram_ratio(const std::vector<int>& ids, int n) {
    if (static_cast<int>(ids.size()) < n) return 0.0;
    std::vector<std::uint64_t> grams;
    for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= ids.size(); ++i) {
        std::uint64_t g = 0;
        for (int k = 0; k < n; ++k) g = (g << 8) | static_cast<std::uint64_t>(ids[i + k] & 0xff);
        grams.push_back(g);
    }
    const std::size_t total = grams.size();
    std::sort(grams.begin(), grams.end());
    grams.erase(std::unique(grams.begin(), grams.end()), grams.end());
    return static_cast<double>(grams.size()) / static_cast<double>(total);
}

}  // namespace

int main(int argc, char** argv) {
    std::string train_path, eval_path, save_base, load_json, tier;
    std::uint64_t train_bytes = 0, eval_offset = 0, eval_bytes = 16384;
    int table_bits = 22, gen_bytes = 200, prompt_bytes = 256;
    double temperature = 0.8, top_p = 0.9;
    bool frozen_eval = false;
    bool compare_scoring = false;
    std::string reset_mode = "none";
    int serve_lr = 4, serve_skip = -1, epochs = 1;
    std::string ensemble_json;
    int word_k = 0;
    bool only_default = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--train") train_path = next();
        else if (a == "--train-bytes") train_bytes = std::stoull(next());
        else if (a == "--eval") eval_path = next();
        else if (a == "--eval-offset") eval_offset = std::stoull(next());
        else if (a == "--eval-bytes") eval_bytes = std::stoull(next());
        else if (a == "--save") save_base = next();
        else if (a == "--load") load_json = next();
        else if (a == "--tier") tier = next();
        else if (a == "--table-bits") table_bits = std::stoi(next());
        else if (a == "--gen-bytes") gen_bytes = std::stoi(next());
        else if (a == "--prompt-bytes") prompt_bytes = std::stoi(next());
        else if (a == "--temperature") temperature = std::stod(next());
        else if (a == "--top-p") top_p = std::stod(next());
        else if (a == "--frozen-eval") frozen_eval = true;
        else if (a == "--compare-scoring") compare_scoring = true;
        else if (a == "--reset-stream") reset_mode = next();
        else if (a == "--serve-lr") serve_lr = std::stoi(next());
        else if (a == "--serve-skip") serve_skip = std::stoi(next());
        else if (a == "--epochs") epochs = std::stoi(next());
        else if (a == "--ensemble") ensemble_json = next();
        else if (a == "--word-k") word_k = std::stoi(next());
        else if (a == "--only-default") only_default = true;
        else {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }

    nlohmann::json out;
    out["harness"] = "cyphalm_lm_quality";

    std::unique_ptr<cypha::cyphalm::CyphaLMModel> model;
    const auto t_setup = Clock::now();
    if (!load_json.empty()) {
        model = std::make_unique<cypha::cyphalm::CyphaLMModel>(
            cypha::cyphalm::load_cyphalm_model(load_json));
        out["loaded"] = load_json;
    } else {
        cypha::cyphalm::CyphaLMConfig cfg;
        cfg.hp_table_bits = table_bits;
        cypha::cyphalm::apply_hp_production_recipe(cfg);
        if (!tier.empty()) cypha::cyphalm::apply_hp_lossy_tier(cfg, tier);
        model = std::make_unique<cypha::cyphalm::CyphaLMModel>(cfg);
    }
    auto& hp = model->hp_backend();
    out["tier"] = model->config().hp_lossy_tier.empty() ? "gate24" : model->config().hp_lossy_tier;

    if (!train_path.empty() && train_bytes > 0) {
        const auto train = read_slice(train_path, 0, train_bytes);
        const auto t0 = Clock::now();
        nlohmann::json per_epoch = nlohmann::json::array();
        double bits = 0.0;
        for (int e = 0; e < std::max(1, epochs); ++e) {
            bits = hp.observe_stream_bits(train.data(), train.size());
            per_epoch.push_back(bits / static_cast<double>(train.size()));
        }
        out["train"] = {{"path", train_path},
                        {"bytes", train.size()},
                        {"epochs", std::max(1, epochs)},
                        {"online_bpc", per_epoch.front()},
                        {"bpc_by_epoch", per_epoch},
                        {"seconds", seconds_since(t0)}};
    }
    if (!save_base.empty()) {
        cypha::cyphalm::save_cyphalm_model(*model, save_base);
        out["saved"] = save_base;
    }
    out["setup_seconds"] = seconds_since(t_setup);

    if (!eval_path.empty()) {
        // none: continue the training stream; full: new stream, empty history;
        // keep: new stream that keeps the byte history match models copy from.
        if (reset_mode == "full") model->reset_stream(false);
        else if (reset_mode == "keep") model->reset_stream(true);
        out["reset_stream"] = reset_mode;
        // Serve-time adaptation: mixer rates x serve_lr/4, skip threshold.
        hp.predictor().set_serve_adaptation(serve_lr, 4, serve_skip);
        // Optional second model: its distribution is mixed with the first.
        std::unique_ptr<cypha::cyphalm::CyphaLMModel> model2;
        if (!ensemble_json.empty()) {
            model2 = std::make_unique<cypha::cyphalm::CyphaLMModel>(
                cypha::cyphalm::load_cyphalm_model(ensemble_json));
            if (reset_mode == "full") model2->reset_stream(false);
            else if (reset_mode == "keep") model2->reset_stream(true);
            model2->hp_backend().predictor().set_serve_adaptation(serve_lr, 4, serve_skip);
        }
        double e2_nll = 0.0, lin_nll = 0.0, geo_nll = 0.0, bayes_nll = 0.0;
        double w_bayes = 0.5;
        out["serve_lr_quarters"] = serve_lr;
        out["serve_skip"] = serve_skip;
        const auto ev = read_slice(eval_path, eval_offset, eval_bytes + prompt_bytes);
        const std::size_t n_eval = ev.size() > static_cast<std::size_t>(prompt_bytes)
                                       ? ev.size() - static_cast<std::size_t>(prompt_bytes)
                                       : ev.size();
        double nll_bits = 0.0, entropy_bits = 0.0;
        std::size_t top1 = 0, top5 = 0, greedy_is_argmax = 0;
        constexpr int kBins = 10;
        double bin_conf[kBins] = {}, bin_acc[kBins] = {};
        std::size_t bin_n[kBins] = {};
        const double temps[] = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4};
        constexpr int kTemps = 8;
        double nll_t[kTemps] = {};
        double f_nll = 0.0, f_secs = 0.0, e_secs = 0.0;
        std::size_t f_top1 = 0;
        const auto t0 = Clock::now();
        for (std::size_t k = 0; k < n_eval; ++k) {
            const int truth = ev[k];
            if (compare_scoring) {
                hp.set_frozen_scoring(true);
                const auto tf = Clock::now();
                const auto lpf = hp.serve_next_byte_log_probs(256);
                f_secs += seconds_since(tf);
                hp.set_frozen_scoring(false);
                f_nll += -lpf[static_cast<std::size_t>(truth)] / kLn2;
                int am = 0;
                for (int b = 1; b < 256; ++b)
                    if (lpf[static_cast<std::size_t>(b)] > lpf[static_cast<std::size_t>(am)]) am = b;
                if (am == truth) ++f_top1;
            }
            const auto te = Clock::now();
            const auto lp = hp.serve_next_byte_log_probs(256);
            e_secs += seconds_since(te);
            int argmax = 0;
            double pmax = -1.0, h = 0.0;
            int rank = 0;
            for (int b = 0; b < 256; ++b) {
                const double p = std::exp(lp[static_cast<std::size_t>(b)]);
                if (p > pmax) {
                    pmax = p;
                    argmax = b;
                }
                if (p > 0) h -= p * lp[static_cast<std::size_t>(b)];
                if (lp[static_cast<std::size_t>(b)] > lp[static_cast<std::size_t>(truth)]) ++rank;
            }
            nll_bits += -lp[static_cast<std::size_t>(truth)] / kLn2;
            if (model2) {
                auto& h2 = model2->hp_backend();
                const auto lp2 = h2.serve_next_byte_log_probs(256);
                const double p1 = std::exp(lp[static_cast<std::size_t>(truth)]);
                const double p2 = std::exp(lp2[static_cast<std::size_t>(truth)]);
                e2_nll += -std::log2(p2);
                lin_nll += -std::log2(0.5 * p1 + 0.5 * p2);
                double zg = 0.0;
                for (int b = 0; b < 256; ++b)
                    zg += std::exp(0.5 * (lp[static_cast<std::size_t>(b)] + lp2[static_cast<std::size_t>(b)]));
                geo_nll += -(0.5 * (lp[static_cast<std::size_t>(truth)] + lp2[static_cast<std::size_t>(truth)]) -
                             std::log(zg)) / kLn2;
                // Fixed-share Bayesian weight on model 1 (tracks which model is better lately).
                const double pm = w_bayes * p1 + (1.0 - w_bayes) * p2;
                bayes_nll += -std::log2(pm);
                w_bayes = 0.98 * (w_bayes * p1 / pm) + 0.02 * 0.5;
                h2.set_learning(!frozen_eval);
                h2.observe_next_byte(static_cast<std::uint8_t>(truth));
                h2.set_learning(true);
            }
            for (int t = 0; t < kTemps; ++t) {
                // log softmax(lp / T) at the true byte
                double mx = -1e300;
                for (int b = 0; b < 256; ++b) mx = std::max(mx, lp[static_cast<std::size_t>(b)] / temps[t]);
                double z = 0.0;
                for (int b = 0; b < 256; ++b) z += std::exp(lp[static_cast<std::size_t>(b)] / temps[t] - mx);
                nll_t[t] += -(lp[static_cast<std::size_t>(truth)] / temps[t] - mx - std::log(z)) / kLn2;
            }
            entropy_bits += h / kLn2;
            if (rank == 0) ++top1;
            if (rank < 5) ++top5;
            if (static_cast<int>(hp.serve_greedy_next_byte()) == argmax) ++greedy_is_argmax;
            const int bin = std::min(kBins - 1, static_cast<int>(pmax * kBins));
            bin_conf[bin] += pmax;
            bin_acc[bin] += (argmax == truth) ? 1.0 : 0.0;
            ++bin_n[bin];
            hp.set_learning(!frozen_eval);
            hp.observe_next_byte(static_cast<std::uint8_t>(truth));
            hp.set_learning(true);
        }
        const double secs = seconds_since(t0);
        double ece = 0.0;
        nlohmann::json bins = nlohmann::json::array();
        for (int b = 0; b < kBins; ++b) {
            if (bin_n[b] == 0) continue;
            const double c = bin_conf[b] / static_cast<double>(bin_n[b]);
            const double acc = bin_acc[b] / static_cast<double>(bin_n[b]);
            ece += static_cast<double>(bin_n[b]) / static_cast<double>(n_eval) * std::abs(c - acc);
            bins.push_back({{"bin", b}, {"n", bin_n[b]}, {"confidence", c}, {"accuracy", acc}});
        }
        const double n = static_cast<double>(n_eval);
        nlohmann::json tsweep = nlohmann::json::object();
        for (int t = 0; t < kTemps; ++t) tsweep[std::to_string(temps[t]).substr(0, 3)] = nll_t[t] / n;
        out["eval"] = {{"path", eval_path},
                       {"learning", frozen_eval ? "frozen (pretrained only)" : "online (in-context)"},
                       {"offset", eval_offset},
                       {"bytes", n_eval},
                       {"nll_bits_per_byte", nll_bits / n},
                       {"perplexity_per_byte", std::exp2(nll_bits / n)},
                       {"top1", top1 / n},
                       {"top5", top5 / n},
                       {"mean_entropy_bits", entropy_bits / n},
                       {"ece_top1", ece},
                       {"bit_greedy_equals_argmax", greedy_is_argmax / n},
                       {"calibration_bins", bins},
                       {"nll_bits_by_temperature", tsweep},
                       {"ms_per_byte", 1e3 * secs / n},
                       {"distribution_ms", 1e3 * e_secs / n}};
        if (model2) {
            out["ensemble"] = {{"with", ensemble_json},
                               {"model2_nll", e2_nll / n},
                               {"linear_half_nll", lin_nll / n},
                               {"geometric_half_nll", geo_nll / n},
                               {"fixed_share_nll", bayes_nll / n}};
        }
        if (compare_scoring) {
            out["frozen_scoring"] = {{"nll_bits_per_byte", f_nll / n},
                                     {"top1", f_top1 / n},
                                     {"distribution_ms", 1e3 * f_secs / n}};
        }

        // Continuations from the held-out prompt that follows the eval slice.
        if (gen_bytes > 0 && prompt_bytes > 0 && ev.size() > n_eval) {
            std::vector<int> prompt(ev.begin() + static_cast<std::ptrdiff_t>(n_eval), ev.end());
            nlohmann::json gens = nlohmann::json::array();
            const std::string blob =
                (std::filesystem::temp_directory_path() /
                 ("cyphalm_lm_quality_gen_" + std::to_string(std::random_device{}())))
                    .string();
            cypha::cyphalm::save_cyphalm_model(*model, blob);
            struct Mode {
                const char* name;
                cypha::cyphalm::DecodeStrategy strategy;
                double temperature;
                bool learn_from_output;
                double min_p;
                int no_repeat;
            };
            using DS = cypha::cyphalm::DecodeStrategy;
            const Mode modes[] = {
                {"greedy (default)", DS::Greedy, 0.0, true, 0.0, 0},
                {"greedy norepeat8", DS::Greedy, 0.0, true, 0.0, 8},
                {"greedy norepeat16", DS::Greedy, 0.0, true, 0.0, 16},
                {"greedy frozen norepeat16", DS::Greedy, 0.0, false, 0.0, 16},
                {"top_p T0.8 (default)", DS::TopP, 0.8, true, 0.0, 0},
                {"minp0.1 T0.8 learn", DS::Temperature, 0.8, true, 0.1, 0},
                {"minp0.1 T0.8 frozen", DS::Temperature, 0.8, false, 0.1, 0},
                {"minp0.2 T1.0 frozen", DS::Temperature, 1.0, false, 0.2, 0},
                {"minp0.1 T0.8 learn norepeat16", DS::Temperature, 0.8, true, 0.1, 16},
                {"minp0.1 T0.8 frozen norepeat16", DS::Temperature, 0.8, false, 0.1, 16},
            };
            // Reference: how the judge scores the text that really follows.
            {
                const auto truth = read_slice(eval_path, eval_offset + ev.size(),
                                              static_cast<std::uint64_t>(gen_bytes));
                auto judge = cypha::cyphalm::load_cyphalm_model(blob + ".json");
                auto& jh = judge.hp_backend();
                for (int b : prompt) jh.consume_byte(static_cast<std::uint8_t>(b));
                jh.set_learning(false);
                double jbits = 0.0;
                std::vector<int> tids;
                for (std::uint8_t b : truth) {
                    jbits += jh.observe_next_byte(b) / kLn2;
                    tids.push_back(b);
                }
                gens.push_back({{"mode", "reference (true continuation)"},
                                {"distinct_4gram", distinct_ngram_ratio(tids, 4)},
                                {"judge_bits_per_byte", jbits / std::max<std::size_t>(1, truth.size())},
                                {"text", printable(tids)}});
            }
            for (const Mode& m : modes) {
                if (only_default && std::string(m.name) != "minp0.1 T0.8 frozen") continue;
                auto fresh = cypha::cyphalm::load_cyphalm_model(blob + ".json");
                cypha::cyphalm::DecodeParams p;
                p.strategy = m.strategy;
                p.temperature = m.temperature;
                p.top_p = top_p;
                p.seed = 1234;
                p.learn_from_output = m.learn_from_output;
                p.min_p = m.min_p;
                p.no_repeat_ngram = m.no_repeat;
                const auto t_gen = Clock::now();
                const auto g = cypha::cyphalm::generate_decode(fresh, prompt, gen_bytes, p);
                // Judge: the pretrained model reads the prompt (learning on), then
                // scores the continuation without learning from it. Lower = more
                // natural under what the model knows; copy-paste loops score low too,
                // so read it with distinct_4gram.
                auto judge = cypha::cyphalm::load_cyphalm_model(blob + ".json");
                auto& jh = judge.hp_backend();
                for (int b : prompt) jh.consume_byte(static_cast<std::uint8_t>(b));
                jh.set_learning(false);
                double jbits = 0.0;
                for (int b : g.generated_ids) jbits += jh.observe_next_byte(static_cast<std::uint8_t>(b)) / kLn2;
                gens.push_back({{"mode", m.name},
                                {"ms_per_byte", 1e3 * seconds_since(t_gen) / std::max(1, gen_bytes)},
                                {"distinct_4gram", distinct_ngram_ratio(g.generated_ids, 4)},
                                {"judge_bits_per_byte", jbits / std::max<std::size_t>(1, g.generated_ids.size())},
                                {"text", printable(g.generated_ids)}});
            }
            // Word lookahead (DecodeParams::word_candidates) at K/2 and K candidates,
            // default byte-level settings otherwise.
            if (word_k > 1) {
                for (int k : {std::max(2, word_k / 2), word_k}) {
                    auto fresh = cypha::cyphalm::load_cyphalm_model(blob + ".json");
                    cypha::cyphalm::DecodeParams p;
                    p.seed = 1234;
                    p.word_candidates = k;
                    const auto t_gen = Clock::now();
                    const auto g = cypha::cyphalm::generate_decode(fresh, prompt, gen_bytes, p);
                    auto judge = cypha::cyphalm::load_cyphalm_model(blob + ".json");
                    auto& jh = judge.hp_backend();
                    for (int b : prompt) jh.consume_byte(static_cast<std::uint8_t>(b));
                    jh.set_learning(false);
                    double jbits = 0.0;
                    for (int b : g.generated_ids) jbits += jh.observe_next_byte(static_cast<std::uint8_t>(b)) / kLn2;
                    gens.push_back({{"mode", "word lookahead K" + std::to_string(k)},
                                    {"ms_per_byte", 1e3 * seconds_since(t_gen) / std::max(1, gen_bytes)},
                                    {"distinct_4gram", distinct_ngram_ratio(g.generated_ids, 4)},
                                    {"judge_bits_per_byte", jbits / std::max<std::size_t>(1, g.generated_ids.size())},
                                    {"text", printable(g.generated_ids)}});
                }
            }
            std::error_code ec;
            std::filesystem::remove(blob + ".json", ec);
            std::filesystem::remove(blob + ".hpbin", ec);
            out["prompt_tail"] = printable(std::vector<int>(prompt.end() - std::min<std::ptrdiff_t>(80, static_cast<std::ptrdiff_t>(prompt.size())), prompt.end()));
            out["generations"] = gens;
        }
    }
    std::cout << out.dump(2) << std::endl;
    return 0;
}
