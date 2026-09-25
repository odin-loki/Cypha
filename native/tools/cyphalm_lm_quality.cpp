/// LLM-quality harness for CyphaLM: pretrain, then score held-out text with the
/// full next-byte distribution (not the compression bit path).
///
/// Per held-out byte: full 256-way log P via serve_next_byte_log_probs, then the
/// model learns the true byte (in-context adaptation, as when reading a prompt).
/// Reports NLL (bits/byte), top-1/top-5 accuracy, expected calibration error of
/// the top-1 confidence, mean entropy, and how often bit-greedy decoding picks
/// the distribution's argmax (plain models only). Ends with greedy / sampled
/// continuations (plain models only: generation reloads the saved primary).
///
///   cyphalm_lm_quality --train enwik8 --train-bytes 8388608 --save /tmp/pre
///   cyphalm_lm_quality --load /tmp/pre.json --eval enwik8 --eval-offset 96000000 --eval-bytes 32768
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "hp/shard_merge.hpp"

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

/// Resident set size of this process in MB (Linux /proc; 0 elsewhere).
double rss_mb(const char* field = "VmRSS:") {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(field, 0) == 0) return std::stod(line.substr(std::strlen(field))) / 1024.0;
    }
    return 0.0;
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
    std::uint64_t train_bytes = 0, train_offset = 0, eval_offset = 0, eval_bytes = 16384;
    int table_bits = 22, gen_bytes = 200, prompt_bytes = 256;
    double temperature = 0.8, top_p = 0.9;
    bool frozen_eval = false;
    std::vector<std::string> neural_paths;  // byte LSTM (BLM1) / Transformer (BGT1) experts
    bool session_cache = false;             // ∞-gram index over the text read so far
    double neural_lr = -1.0;                // neural mixing weight step (exponentiated gradient), <0 = manifest/0.1
    double neural_adapt = -1.0;             // output-layer SGD rate (dynamic evaluation), <0 = manifest/default
    std::size_t infinigram_bytes = 0;       // --infinigram given the corpus: index its first N bytes
    double fold_auto = 0.0;  // per-table occupancy fold target (0 = off)
    std::string dump_dist;  // float32 natural-log P, 256 per held-out byte
    bool compare_scoring = false;
    std::string reset_mode = "none";
    int serve_lr = 4, serve_skip = -1, epochs = 1;
    std::vector<std::string> members;  // library ensemble, equal weights
    double ensemble_lr = -1.0;  // <0: the model's config default
    int fold_cm = 0, fold_match = 0, fold_pool = 0, fold_hebb = 0;
    std::uint64_t drop_mask = 0;
    std::uint32_t match_drop = 0;
    std::string infinigram_path, ig_weights_out;
    double tree_prune = 0.0;
    std::vector<std::string> merges;  // shard models merged into --load (equal data)
    int word_k = 0;
    bool only_default = false;
    bool freeze_mixing = false;  // mixing weights stay at their start values (the pre-fix served mixture)
    // Flags that act on a loaded checkpoint or manifest: without --load they
    // would be ignored, so they are an error there.
    const std::set<std::string> load_only = {
        "--member", "--ensemble-lr", "--merge", "--fold", "--drop", "--match-drop", "--fold-auto",
        "--session-cache", "--neural", "--neural-lr", "--neural-adapt", "--infinigram",
        "--infinigram-bytes", "--ig-weights-out", "--freeze-mixing"};
    std::vector<std::string> load_only_given;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (load_only.count(a) != 0) load_only_given.push_back(a);
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--train") train_path = next();
        else if (a == "--train-bytes") train_bytes = std::stoull(next());
        else if (a == "--train-offset") train_offset = std::stoull(next());
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
        else if (a == "--member") members.push_back(next());
        else if (a == "--ensemble-lr") ensemble_lr = std::stod(next());
        else if (a == "--merge") merges.push_back(next());
        else if (a == "--infinigram") infinigram_path = next();
        else if (a == "--infinigram-bytes") infinigram_bytes = std::stoull(next());  // corpus given: index its first N bytes
        else if (a == "--tree-prune") tree_prune = std::stod(next());
        else if (a == "--ig-weights-out") ig_weights_out = next();
        else if (a == "--match-drop") match_drop = static_cast<std::uint32_t>(std::stoul(next(), nullptr, 0));
        else if (a == "--drop") drop_mask = std::stoull(next(), nullptr, 0);  // cm_drop bits
        else if (a == "--fold") {  // CM,MATCH,POOL table bits (0 = keep)
            const std::string v = next();
            std::sscanf(v.c_str(), "%d,%d,%d,%d", &fold_cm, &fold_match, &fold_pool, &fold_hebb);
        }
        else if (a == "--word-k") word_k = std::stoi(next());
        else if (a == "--only-default") only_default = true;
        else if (a == "--dump-dist") dump_dist = next();
        else if (a == "--fold-auto") fold_auto = std::stod(next());
        else if (a == "--neural") neural_paths.push_back(next());  // repeatable
        else if (a == "--session-cache") session_cache = true;
        else if (a == "--neural-lr") neural_lr = std::stod(next());
        else if (a == "--neural-adapt") neural_adapt = std::stod(next());
        else if (a == "--freeze-mixing") freeze_mixing = true;
        else {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }
    if (!(fold_auto >= 0.0 && fold_auto < 1.0)) {  // 0 = off
        std::cerr << "--fold-auto must be in (0, 1), a projected table occupancy (0 = off)\n";
        return 2;
    }
    if (load_json.empty() && !load_only_given.empty()) {
        std::cerr << load_only_given.front() << " needs --load (it applies to a loaded checkpoint or manifest)\n";
        return 2;
    }

    nlohmann::json out;
    out["harness"] = "cyphalm_lm_quality";

    std::unique_ptr<cypha::cyphalm::CyphaLMModel> model;
    const auto t_setup = Clock::now();
    if (!load_json.empty()) {
        model = std::make_unique<cypha::cyphalm::CyphaLMModel>(
            cypha::cyphalm::load_cyphalm_model(load_json));
        out["loaded"] = load_json;
        // Merge equally-sized shard models' tables into the loaded one: one
        // model's RAM for all shards' data (hp::Predictor::merge_shard_tables).
        for (std::size_t k = 0; k < merges.size(); ++k) {
            auto src = cypha::cyphalm::load_cyphalm_model(merges[k]);
            if (!model->hp_backend().predictor().merge_shard_tables(src.hp_backend().predictor(), 1,
                                                                    static_cast<std::uint64_t>(k + 1))) {
                std::cerr << "--merge " << merges[k] << ": table sizes differ from --load (folded or "
                          << "dropped models cannot be merged; merge before folding)\n";
                return 1;
            }
        }
        if (!merges.empty()) {
            model->reset_stream(/*keep_history=*/true);
            out["merged"] = merges;
        }
        if (fold_cm > 0 || fold_match > 0 || fold_pool > 0 || fold_hebb > 0 || drop_mask != 0 || match_drop != 0) {
            model->fold_hp_tables(fold_cm, fold_match, fold_pool, drop_mask, fold_hebb, match_drop);
            out["fold"] = {fold_cm, fold_match, fold_pool, fold_hebb};
            out["drop_mask"] = drop_mask;
            out["match_drop"] = match_drop;
        }
        for (const auto& m : members) {
            auto mm = cypha::cyphalm::load_cyphalm_model(m);
            if (fold_cm > 0 || fold_match > 0 || fold_pool > 0 || fold_hebb > 0 || drop_mask != 0 || match_drop != 0)
                mm.fold_hp_tables(fold_cm, fold_match, fold_pool, drop_mask, fold_hebb, match_drop);
            model->add_ensemble_member(std::move(mm), 1.0 / static_cast<double>(members.size() + 1));
        }
        if (!members.empty()) out["members"] = members;
        if (fold_auto > 0.0) {
            out["fold_auto"] = fold_auto;
            out["fold_auto_freed_mb"] = static_cast<double>(model->hp_backend().fold_auto(fold_auto)) / 1048576.0;
        }
        if (session_cache) {
            model->hp_backend().set_session_cache(true);
            out["session_cache"] = true;
        }
        // One mixing rate for all experts: --neural-lr if given (manifest
        // experts too), else the manifest's, else 0.1.
        auto& hb = model->hp_backend();
        const double nn_eta = neural_lr >= 0.0 ? neural_lr : hb.has_neural() ? hb.neural_learning_rate() : 0.1;
        for (const auto& np : neural_paths) model->attach_neural(np, nn_eta);
        if (hb.has_neural()) {
            hb.set_neural_learning_rate(nn_eta);
            out["neural_learning_rate"] = hb.neural_learning_rate();
        }
        if (neural_adapt >= 0.0 && model->hp_backend().has_neural()) {
            model->hp_backend().set_neural_adaptation(neural_adapt);
            out["neural_adapt"] = neural_adapt;
        }
        if (!neural_paths.empty()) {
            out["neural"] = neural_paths;
        }
        if (!infinigram_path.empty()) {
            const auto t_ig = Clock::now();
            model->attach_infinigram(infinigram_path, infinigram_bytes);
            out["infinigram_attach_seconds"] = seconds_since(t_ig);
            out["infinigram"] = infinigram_path;
        }
        if (ensemble_lr >= 0.0) model->hp_backend().set_ensemble_learning_rate(ensemble_lr);
        if (freeze_mixing) {
            // Every stage's mixing weights stay at their start values while
            // the models still learn: the mixture generation served before
            // prompts were scored.
            model->hp_backend().set_mixing_learning(false);
            out["freeze_mixing"] = true;
        }
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
        const auto train = read_slice(train_path, train_offset, train_bytes);
        const auto t0 = Clock::now();
        nlohmann::json per_epoch = nlohmann::json::array();
        double bits = 0.0;
        for (int e = 0; e < std::max(1, epochs); ++e) {
            bits = hp.observe_stream_bits(train.data(), train.size());
            per_epoch.push_back(bits / static_cast<double>(train.size()));
        }
        out["train"] = {{"path", train_path},
                        {"offset", train_offset},
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
    out["rss_mb_after_setup"] = rss_mb();
    out["rss_anon_mb_after_setup"] = rss_mb("RssAnon:");
    out["rss_file_mb_after_setup"] = rss_mb("RssFile:");

    if (!eval_path.empty()) {
        // none: continue the training stream; full: new stream, empty history;
        // keep: new stream that keeps the byte history match models copy from.
        if (reset_mode == "full") model->reset_stream(false);
        else if (reset_mode == "keep") model->reset_stream(true);
        out["reset_stream"] = reset_mode;
        if (tree_prune > 0.0) {
            hp.set_tree_prune(tree_prune);
            out["tree_prune"] = tree_prune;
        }
        // Serve-time adaptation: mixer rates x serve_lr/4, skip threshold, on
        // every model (the primary and each ensemble member).
        hp.set_serve_adaptation(serve_lr, 4, serve_skip);
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
        // Composite (members, ∞-gram, neural experts, session cache): the bit
        // path is the primary's alone and serve_greedy_next_byte is the mix's
        // argmax by construction, so the bit-greedy metric is not measured.
        const bool composite = hp.is_composite();
        std::ofstream dump;
        if (!dump_dist.empty()) dump.open(dump_dist, std::ios::binary);
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
            if (dump.is_open()) {
                float row[256];
                for (int b = 0; b < 256; ++b) row[b] = static_cast<float>(lp[static_cast<std::size_t>(b)]);
                dump.write(reinterpret_cast<const char*>(row), sizeof(row));
            }
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
            if (!composite && static_cast<int>(hp.serve_greedy_next_byte()) == argmax) ++greedy_is_argmax;
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
                       {"bit_greedy_equals_argmax", composite ? nlohmann::json() : nlohmann::json(greedy_is_argmax / n)},
                       {"calibration_bins", bins},
                       {"nll_bits_by_temperature", tsweep},
                       {"ms_per_byte", 1e3 * secs / n},
                       {"distribution_ms", 1e3 * e_secs / n}};
        if (!ig_weights_out.empty() && hp.has_infinigram()) {
            std::ofstream wf(ig_weights_out);
            wf << nlohmann::json({{"note", "∞-gram mixing weights learned on held-out text"},
                                  {"learned_on", eval_path},
                                  {"offset", eval_offset},
                                  {"weights", hp.infinigram_weights()}}).dump(1) << "\n";
        }
        if (!members.empty()) {
            out["ensemble_lr"] = ensemble_lr;
            out["ensemble_weights_final"] = model->hp_backend().ensemble_weights();
        }
        out["rss_mb_after_eval"] = rss_mb();
        out["rss_anon_mb_after_eval"] = rss_mb("RssAnon:");
        out["rss_file_mb_after_eval"] = rss_mb("RssFile:");
        out["rss_hwm_mb"] = rss_mb("VmHWM:");
        if (compare_scoring) {
            out["frozen_scoring"] = {{"nll_bits_per_byte", f_nll / n},
                                     {"top1", f_top1 / n},
                                     {"distribution_ms", 1e3 * f_secs / n}};
        }

        // Continuations from the held-out prompt that follows the eval slice.
        // Generation and the judge reload the saved model, which is the
        // primary predictor alone (no members, ∞-gram, experts or session
        // cache): skipped for composite models rather than measuring the primary.
        if (gen_bytes > 0 && composite) {
            out["generations"] = "skipped: composite model; use cyphalm_generate --load MANIFEST or cyphalm_gen_bench";
        }
        if (gen_bytes > 0 && !composite && prompt_bytes > 0 && ev.size() > n_eval) {
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
                p.word_candidates = 0;  // byte-level modes
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
