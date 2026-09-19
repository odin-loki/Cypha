/// Large-n CyphaLM eval: bit-serial observe BPC + optional bit-tree top-k.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using Clock = std::chrono::steady_clock;

std::vector<int> load_bytes_file(const std::string& path, int max_n) {
    std::ifstream in(path, std::ios::binary);
    std::vector<int> out;
    int ch;
    while (in && (max_n <= 0 || static_cast<int>(out.size()) < max_n)) {
        ch = in.get();
        if (ch == EOF) break;
        out.push_back(ch & 0xff);
    }
    return out;
}

std::string hp_profile_name() {
#if defined(CYPHA_HP_PROFILE_CHAMP)
    return "champ";
#elif defined(CYPHA_HP_PROFILE_GATE24)
    return "gate24";
#else
    return "light";
#endif
}

cypha::cyphalm::CyphaLMConfig eval_cfg() {
    cypha::cyphalm::CyphaLMConfig cfg;
#if defined(CYPHA_HP_PROFILE_CHAMP)
    cypha::cyphalm::apply_hp_champ_recipe(cfg);
#elif defined(CYPHA_HP_PROFILE_GATE24)
    cypha::cyphalm::apply_hp_gate24_recipe(cfg);
#else
    cypha::cyphalm::apply_hp_production_recipe(cfg);
#endif
    cfg.vocab_size = 256;
    return cfg;
}

nlohmann::json observe_bpc_run(const std::string& label, const std::vector<int>& ids, int n_eval) {
    nlohmann::json j;
    const int n = std::min(n_eval, static_cast<int>(ids.size()));
    j["corpus"] = label;
    j["bytes_requested"] = n_eval;
    j["bytes_evaluated"] = n;
    if (n <= 0) {
        j["status"] = "empty";
        return j;
    }
    auto cfg = eval_cfg();
    cypha::cyphalm::CyphaLMModel model(cfg);
    const auto t0 = Clock::now();
    const double bpc = model.eval_bpc(ids, n);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    j["metric"] = "observe_bit_serial_bpc";
    j["bpc"] = bpc;
    j["eval_ms"] = ms;
    j["bytes_per_sec"] = ms > 0 ? static_cast<double>(n) / (ms / 1000.0) : 0.0;
    j["status"] = "ok";
    return j;
}

struct TopKStats {
    int n = 0;
    int top1 = 0;
    int top5 = 0;
    int top10 = 0;
    double nll_bits = 0.0;
    double entropy_bits = 0.0;
};

void accumulate_topk(const std::vector<double>& log_probs, int truth, TopKStats& st) {
    const int n = static_cast<int>(log_probs.size());
    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ranked.emplace_back(log_probs[static_cast<std::size_t>(i)], i);
    }
    std::partial_sort(ranked.begin(), ranked.begin() + std::min(10, n), ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    if (!ranked.empty() && ranked[0].second == truth) ++st.top1;
    for (int k = 0; k < std::min(5, n); ++k) {
        if (ranked[static_cast<std::size_t>(k)].second == truth) {
            ++st.top5;
            break;
        }
    }
    for (int k = 0; k < std::min(10, n); ++k) {
        if (ranked[static_cast<std::size_t>(k)].second == truth) {
            ++st.top10;
            break;
        }
    }
    double max_lp = -std::numeric_limits<double>::infinity();
    double sum_p = 0.0;
    for (double lp : log_probs) {
        max_lp = std::max(max_lp, lp);
    }
    for (double lp : log_probs) {
        sum_p += std::exp(lp - max_lp);
    }
    const double log_z = max_lp + std::log(sum_p);
    double h = 0.0;
    for (double lp : log_probs) {
        const double p = std::exp(lp - log_z);
        if (p > 0) h -= p * std::log(p);
    }
    const double truth_lp = log_probs[static_cast<std::size_t>(truth)];
    st.nll_bits += -(truth_lp - log_z) / 0.6931471805599453;
    st.entropy_bits += h / 0.6931471805599453;
    ++st.n;
}

nlohmann::json topk_run(const std::string& label, const std::vector<int>& ids, int n_eval) {
    nlohmann::json j;
    const int n = std::min(n_eval, static_cast<int>(ids.size()) - 1);
    j["corpus"] = label;
    j["pairs_requested"] = n_eval;
    j["pairs_evaluated"] = n;
    if (n <= 0) {
        j["status"] = "empty";
        return j;
    }
    auto cfg = eval_cfg();
    cypha::cyphalm::CyphaLMModel model(cfg);
    TopKStats st;
    const auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) {
        const auto pred = model.predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]));
        accumulate_topk(pred.log_probs, ids[static_cast<std::size_t>(i + 1)], st);
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    j["metric"] = "bit_tree_predict_next_topk";
    j["bpc"] = st.n > 0 ? st.nll_bits / st.n : 0.0;
    j["top1_acc"] = st.n > 0 ? static_cast<double>(st.top1) / st.n : 0.0;
    j["top5_acc"] = st.n > 0 ? static_cast<double>(st.top5) / st.n : 0.0;
    j["top10_acc"] = st.n > 0 ? static_cast<double>(st.top10) / st.n : 0.0;
    j["mean_entropy_bits"] = st.n > 0 ? st.entropy_bits / st.n : 0.0;
    j["eval_ms"] = ms;
    j["predict_next_us"] = st.n > 0 ? (ms * 1000.0) / st.n : 0.0;
    j["status"] = "ok";
    return j;
}

nlohmann::json latency_compare(cypha::cyphalm::CyphaLMModel& model, int iters) {
    nlohmann::json j;
    const int k = std::max(1, iters);
    model.reset_context();
    for (int i = 0; i < 64; ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(65 + (i % 26)));
    }
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < k; ++i) {
            (void)model.hp_backend().next_byte_log_probs_bit_tree(256);
        }
        j["bit_tree_next_byte_log_probs_us"] =
            std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / k;
    }
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < k; ++i) {
            (void)model.hp_backend().next_byte_log_probs_legacy(256);
        }
        j["legacy_next_byte_log_probs_us"] =
            std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / k;
    }
    j["iters"] = k;
    j["speedup_bit_tree_vs_legacy"] =
        j["legacy_next_byte_log_probs_us"].get<double>() /
        std::max(1.0, j["bit_tree_next_byte_log_probs_us"].get<double>());
    return j;
}

}  // namespace

int main(int argc, char** argv) {
    int observe_n = 10000;
    int topk_n = 64;
    int latency_iters = 3;
    std::string extra_corpus_path;
    std::string extra_corpus_label = "extra_slice";
    bool skip_topk = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--observe-n" && i + 1 < argc) observe_n = std::stoi(argv[++i]);
        else if (a == "--topk-n" && i + 1 < argc) topk_n = std::stoi(argv[++i]);
        else if (a == "--latency-iters" && i + 1 < argc) latency_iters = std::stoi(argv[++i]);
        else if (a == "--corpus" && i + 2 < argc) {
            extra_corpus_path = argv[++i];
            extra_corpus_label = argv[++i];
        } else if (a == "--skip-topk") skip_topk = true;
    }

    nlohmann::json out;
    auto cfg = eval_cfg();
    out["hp_profile"] = hp_profile_name();
    out["hp_table_bits"] = cfg.hp_table_bits;
    out["hp_slot_max"] = cfg.hp_slot_max;
    out["hp_slot_compile_max"] = cypha::cyphalm::hp_compile_slot_max();

    nlohmann::json observe = nlohmann::json::array();
    std::vector<int> extra_ids;
    if (!extra_corpus_path.empty()) {
        extra_ids = load_bytes_file(extra_corpus_path, observe_n);
        observe.push_back(observe_bpc_run(extra_corpus_label, extra_ids, observe_n));
    }

    const cypha::cyphalm::LMCorpus wiki =
        cypha::cyphalm::load_bench_corpus("d21", 500000, 256);
    if (!wiki.train_ids.empty()) {
        observe.push_back(observe_bpc_run("wikitext2_train_slice", wiki.train_ids, observe_n));
    } else {
        observe.push_back({{"corpus", "wikitext2_train"}, {"status", "missing"}});
    }
    if (!wiki.eval_ids.empty()) {
        observe.push_back(observe_bpc_run("wikitext2_eval_slice", wiki.eval_ids, observe_n));
    }
    out["observe_bpc"] = observe;

    cypha::cyphalm::CyphaLMModel model(cfg);
    out["latency"] = latency_compare(model, latency_iters);

    if (!skip_topk && topk_n > 0) {
        nlohmann::json topk = nlohmann::json::array();
        if (!wiki.train_ids.empty()) {
            topk.push_back(topk_run("wikitext2_train", wiki.train_ids, topk_n));
        }
        if (!extra_ids.empty()) {
            topk.push_back(topk_run(extra_corpus_label, extra_ids, topk_n));
        }
        out["bit_tree_topk"] = topk;
    }

    std::cout << out.dump(2) << std::endl;
    return 0;
}
