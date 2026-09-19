/// CyphaLM (hp) profile + LLM capability harness — measured output only (no invented metrics).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/cyphalm/predictive_codec.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kLog2 = 0.6931471805599453;

struct ProcStatus {
    long vm_rss_kb = -1;
    long vm_hwm_kb = -1;
};

ProcStatus read_proc_status() {
    ProcStatus s;
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &s.vm_rss_kb);
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::sscanf(line.c_str(), "VmHWM: %ld kB", &s.vm_hwm_kb);
        }
    }
#endif
    return s;
}

double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double elapsed_us(Clock::time_point t0, int iters) {
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    return (sec / static_cast<double>(iters)) * 1e6;
}

cypha::cyphalm::CyphaLMConfig production_cfg() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.seed = 42;
    return cfg;
}

std::vector<int> fixture_pattern(int n) {
    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ids.push_back((i * 7 + 3) % 96 + 32);  // printable ASCII-ish
    }
    return ids;
}

std::string sample_text(const std::vector<int>& ids, int max_len = 200) {
    std::string out;
    const int n = std::min(max_len, static_cast<int>(ids.size()));
    for (int i = 0; i < n; ++i) {
        const int c = ids[static_cast<std::size_t>(i)];
        if (c >= 32 && c < 127) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('.');
        }
    }
    return out;
}

nlohmann::json profile_construct_and_rss() {
    nlohmann::json j;
    const auto t0 = Clock::now();
    auto cfg = production_cfg();
    cypha::cyphalm::CyphaLMModel model(cfg);
    j["construct_ms"] = elapsed_ms(t0);
    const auto rss = read_proc_status();
    j["vm_rss_kb"] = rss.vm_rss_kb;
    j["vm_hwm_kb"] = rss.vm_hwm_kb;
    j["hp_table_bits"] = cfg.hp_table_bits;
    j["hp_slot_max"] = cfg.hp_slot_max;
    j["hp_slot_compile_max"] = cypha::cyphalm::hp_compile_slot_max();
    j["vocab_size"] = cfg.vocab_size;
    j["dual_predictor_note"] =
        "HpSequenceBackend holds pred_ + scratch_; next_byte_log_probs clones per vocab byte";
    return j;
}

nlohmann::json profile_latency(cypha::cyphalm::CyphaLMModel& model, int warm, int iters) {
    nlohmann::json j;
    const int kWarm = std::max(1, warm);
    const int kIters = std::max(1, iters);
    const std::uint32_t tok = 65;

    for (int i = 0; i < kWarm; ++i) {
        (void)model.predict_next(tok);
    }
    model.reset_context();
    for (int i = 0; i < kWarm; ++i) {
        (void)model.predict_next(tok);
    }
    const auto t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
        (void)model.predict_next(tok);
    }
    j["predict_next_us"] = elapsed_us(t0, kIters);
    j["predict_next_iters"] = kIters;

    auto& hp = model.hp_backend();
    for (int i = 0; i < kWarm; ++i) {
        (void)hp.next_byte_log_probs(model.config().vocab_size);
    }
    const auto t1 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
        (void)hp.next_byte_log_probs(model.config().vocab_size);
    }
    j["next_byte_log_probs_us"] = elapsed_us(t1, kIters);
    j["next_byte_log_probs_iters"] = kIters;

    const auto rss = read_proc_status();
    j["vm_rss_kb_after_latency"] = rss.vm_rss_kb;
    j["vm_hwm_kb_after_latency"] = rss.vm_hwm_kb;
    return j;
}

nlohmann::json profile_throughput(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids,
                                 int n_train_cap, int n_eval_cap) {
    nlohmann::json j;
    const int n_train = std::min(n_train_cap, static_cast<int>(ids.size()) - 1);
    const int n_eval = std::min(n_eval_cap, static_cast<int>(ids.size()) - 1);

    model.reset_context();
    const auto t0 = Clock::now();
    for (int i = 0; i < n_train; ++i) {
        (void)model.train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]),
                               static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i + 1)]));
    }
    const double train_ms = elapsed_ms(t0);
    j["train_step_n"] = n_train;
    j["train_step_ms"] = train_ms;
    j["train_step_bytes_per_sec"] = (train_ms > 0.0) ? (n_train * 1000.0 / train_ms) : 0.0;

    model.reset_context();
    const auto t1 = Clock::now();
    const double bpc = model.eval_bpc(ids, n_eval);
    const double eval_ms = elapsed_ms(t1);
    j["eval_bpc_n"] = n_eval;
    j["eval_bpc"] = bpc;
    j["eval_bpc_ms"] = eval_ms;
    j["eval_bpc_bytes_per_sec"] = (eval_ms > 0.0) ? (n_eval * 1000.0 / eval_ms) : 0.0;

    auto& hp = model.hp_backend();
    model.reset_context();
    const int n_consume = std::min(512, static_cast<int>(ids.size()));
    const auto t2 = Clock::now();
    for (int i = 0; i < n_consume; ++i) {
        hp.consume_byte(static_cast<std::uint8_t>(ids[static_cast<std::size_t>(i)]));
    }
    const double consume_ms = elapsed_ms(t2);
    j["consume_byte_n"] = n_consume;
    j["consume_byte_ms"] = consume_ms;
    j["consume_byte_per_sec"] = (consume_ms > 0.0) ? (n_consume * 1000.0 / consume_ms) : 0.0;

    const auto rss = read_proc_status();
    j["vm_rss_kb_after_throughput"] = rss.vm_rss_kb;
    j["vm_hwm_kb_after_throughput"] = rss.vm_hwm_kb;
    return j;
}

struct TopKStats {
    int scored = 0;
    int top1 = 0;
    int top5 = 0;
    int top10 = 0;
    double sum_pred_entropy_bits = 0.0;
    double sum_empirical_bits = 0.0;
    double sum_bpc_bits = 0.0;
};

void accumulate_topk(const std::vector<double>& log_probs, int truth, TopKStats& st) {
    if (truth < 0 || truth >= static_cast<int>(log_probs.size())) {
        return;
    }
    std::vector<int> order(log_probs.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(),
                      order.begin() + std::min(10, static_cast<int>(order.size())), order.end(),
                      [&](int a, int b) {
                          return log_probs[static_cast<std::size_t>(a)] >
                                 log_probs[static_cast<std::size_t>(b)];
                      });
    auto rank_of = [&](int t) -> int {
        for (int r = 0; r < static_cast<int>(order.size()); ++r) {
            if (order[static_cast<std::size_t>(r)] == t) {
                return r;
            }
        }
        return static_cast<int>(order.size());
    };
    const int rank = rank_of(truth);
    if (rank == 0) {
        ++st.top1;
    }
    if (rank < 5) {
        ++st.top5;
    }
    if (rank < 10) {
        ++st.top10;
    }
    double mx = log_probs[0];
    for (double v : log_probs) {
        mx = std::max(mx, v);
    }
    double sum_p = 0.0;
    double ent_nats = 0.0;
    for (double lp : log_probs) {
        sum_p += std::exp(lp - mx);
    }
    for (double lp : log_probs) {
        const double p = std::exp(lp - mx) / (sum_p + 1e-300);
        if (p > 1e-300) {
            ent_nats -= p * std::log(p);
        }
    }
    st.sum_pred_entropy_bits += ent_nats / kLog2;
    st.sum_empirical_bits += -log_probs[static_cast<std::size_t>(truth)] / kLog2;
    st.sum_bpc_bits += -log_probs[static_cast<std::size_t>(truth)] / kLog2;
    ++st.scored;
}

TopKStats eval_stream_metrics(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids,
                              int n_eval, bool reset_first) {
    TopKStats st;
    if (reset_first) {
        model.reset_context();
    }
    const int n = std::min(n_eval, static_cast<int>(ids.size()) - 1);
    for (int i = 0; i < n; ++i) {
        const auto pred =
            model.predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]));
        const int truth = ids[static_cast<std::size_t>(i + 1)];
        accumulate_topk(pred.log_probs, truth, st);
    }
    return st;
}

nlohmann::json topk_stats_to_json(const TopKStats& st, const std::string& label) {
    nlohmann::json j;
    j["label"] = label;
    j["n_scored"] = st.scored;
    j["top1_acc"] = st.scored > 0 ? static_cast<double>(st.top1) / st.scored : 0.0;
    j["top5_acc"] = st.scored > 0 ? static_cast<double>(st.top5) / st.scored : 0.0;
    j["top10_acc"] = st.scored > 0 ? static_cast<double>(st.top10) / st.scored : 0.0;
    j["mean_pred_entropy_bits"] =
        st.scored > 0 ? st.sum_pred_entropy_bits / st.scored : 0.0;
    j["mean_empirical_bits"] = st.scored > 0 ? st.sum_empirical_bits / st.scored : 0.0;
    j["eval_bpc"] = st.scored > 0 ? st.sum_bpc_bits / st.scored : 0.0;
    j["entropy_minus_empirical_bits"] =
        j["mean_pred_entropy_bits"].get<double>() - j["mean_empirical_bits"].get<double>();
    return j;
}

nlohmann::json capability_wiki_suite(cypha::cyphalm::CyphaLMModel& model,
                                   const cypha::cyphalm::LMCorpus& corpus, int train_steps,
                                   int n_eval, int prompt_len, int gen_tokens,
                                   bool run_generation) {
    nlohmann::json out;
    out["corpus_source"] = corpus.source;
    out["corpus_profile"] = corpus.profile;
    out["train_byte_tokens_available"] = corpus.train_ids.size();
    out["eval_byte_tokens_available"] = corpus.eval_ids.size();
    out["train_online_steps"] = train_steps;
    out["eval_n"] = n_eval;
    out["note"] = "Byte-level tokens from WikiText-2 raw text (80/20 split of train file cap)";

    std::cerr << "wikitext_capability: eval_only n=" << n_eval << " ...\n";
    const auto t0 = Clock::now();
    const TopKStats cold = eval_stream_metrics(model, corpus.eval_ids, n_eval, true);
    out["eval_only"] = topk_stats_to_json(cold, "wikitext_eval_only_cold_start");
    out["eval_only"]["methodology"] = "one_pass_eval_only";
    out["eval_only"]["eval_ms"] = elapsed_ms(t0);
    std::cerr << "wikitext_capability: eval_only bpc=" << out["eval_only"]["eval_bpc"] << "\n";

    model.reset_context();
    if (train_steps > 0 && corpus.train_ids.size() >= 2) {
        std::cerr << "wikitext_capability: training " << train_steps << " steps ...\n";
        model.train_sequence(corpus.train_ids, train_steps, 1, nullptr);
    }
    const auto t1 = Clock::now();
    const TopKStats trained =
        eval_stream_metrics(model, corpus.eval_ids, n_eval, false);
    out["train_then_eval"] = topk_stats_to_json(trained, "wikitext_train_then_eval");
    out["train_then_eval"]["methodology"] = "online_train_then_eval";
    out["train_then_eval"]["eval_ms"] = elapsed_ms(t1);

    model.reset_context();
    if (train_steps > 0 && corpus.train_ids.size() >= 2) {
        model.train_sequence(corpus.train_ids, train_steps, 1, nullptr);
    }
    const int plen = std::min(prompt_len, static_cast<int>(corpus.eval_ids.size()));
    std::vector<int> prompt(corpus.eval_ids.begin(), corpus.eval_ids.begin() + plen);
    std::string prompt_text;
    for (int b : prompt) {
        if (b >= 32 && b < 127) {
            prompt_text.push_back(static_cast<char>(b));
        } else if (b == 10) {
            prompt_text += "\\n";
        } else {
            prompt_text += ".";
        }
    }
    if (run_generation && gen_tokens > 0) {
        cypha::cyphalm::DecodeParams greedy;
        greedy.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
        const auto gout = cypha::cyphalm::generate_decode(model, prompt, gen_tokens, greedy);
        cypha::cyphalm::DecodeParams sample;
        sample.strategy = cypha::cyphalm::DecodeStrategy::Temperature;
        sample.temperature = 0.9;
        sample.seed = 42;
        model.reset_context();
        if (train_steps > 0 && corpus.train_ids.size() >= 2) {
            model.train_sequence(corpus.train_ids, train_steps, 1, nullptr);
        }
        const auto sout = cypha::cyphalm::generate_decode(model, prompt, gen_tokens, sample);
        nlohmann::json gen;
        gen["prompt_bytes"] = plen;
        gen["prompt_text"] = prompt_text;
        gen["greedy_sample"] = sample_text(gout.generated_ids, gen_tokens);
        gen["greedy_len"] = gout.generated_ids.size();
        gen["temp09_sample"] = sample_text(sout.generated_ids, gen_tokens);
        gen["temp09_len"] = sout.generated_ids.size();
        gen["methodology"] = "online_train_then_generate_from_eval_prefix";
        out["generation"] = gen;
    }
    return out;
}

nlohmann::json capability_context_scaling(cypha::cyphalm::CyphaLMModel& model,
                                          const std::vector<int>& ids,
                                          const std::vector<int>& context_lens) {
    nlohmann::json arr = nlohmann::json::array();
    for (int ctx : context_lens) {
        if (ctx + 1 >= static_cast<int>(ids.size())) {
            continue;
        }
        model.reset_context();
        for (int i = 0; i < ctx; ++i) {
            model.hp_backend().consume_byte(
                static_cast<std::uint8_t>(ids[static_cast<std::size_t>(i)]));
        }
        const int truth = ids[static_cast<std::size_t>(ctx)];
        const auto lp = model.hp_backend().next_byte_log_probs(model.config().vocab_size);
        const double bpc = -lp[static_cast<std::size_t>(truth)] / kLog2;
        nlohmann::json row;
        row["context_bytes"] = ctx;
        row["next_byte_bpc"] = bpc;
        arr.push_back(row);
    }
    return arr;
}

nlohmann::json capability_roundtrip(cypha::cyphalm::CyphaLMModel& model,
                                    const std::vector<std::uint32_t>& tokens) {
    nlohmann::json j;
    cypha::cyphalm::PredictiveCodecOptions opt;
    opt.online_adapt = false;
    opt.use_hidden_knn = false;
    model.reset_context();
    const auto packed = cypha::cyphalm::compress_tokens(model, tokens, opt);
    std::string detail;
    const auto decoded =
        cypha::cyphalm::decompress_tokens(model, packed.bytes, 0, tokens.size(), &detail, opt);
    j["n_tokens"] = tokens.size();
    j["coded_bytes"] = packed.bytes.size();
    j["model_bpc"] = packed.model_bpc;
    j["coded_bpc"] = packed.coded_bpc;
    j["roundtrip_ok"] = (decoded == tokens);
    j["detail"] = detail;
    return j;
}

void print_section(const char* name, const nlohmann::json& j) {
    std::cout << "=== " << name << " ===\n" << j.dump(2) << "\n";
    std::cout.flush();
}

}  // namespace

int main(int argc, char** argv) {
    bool skip_wiki = false;
    bool wiki_only = false;
    bool skip_roundtrip = false;
    bool skip_construct = false;
    bool skip_latency = false;
    bool skip_throughput = false;
    bool skip_negative_control = false;
    bool skip_context = false;
    bool skip_generation_wiki = false;
    int wiki_prompt_len = 32;
    int wiki_max_chars = 100000;
    int latency_warm = 2;
    int latency_iters = 5;
    int train_cap = 64;
    int eval_cap = 64;
    int topk_n = 64;
    int gen_tokens = 32;
    int wiki_train_steps = 128;
    int wiki_eval_n = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--skip-wiki") {
            skip_wiki = true;
        } else if (a == "--wiki-only") {
            wiki_only = true;
        } else if (a == "--skip-roundtrip") {
            skip_roundtrip = true;
        } else if (a == "--skip-construct") {
            skip_construct = true;
        } else if (a == "--skip-latency") {
            skip_latency = true;
        } else if (a == "--skip-throughput") {
            skip_throughput = true;
        } else if (a == "--skip-negative-control") {
            skip_negative_control = true;
        } else if (a == "--skip-context") {
            skip_context = true;
        } else if (a == "--skip-generation-wiki") {
            skip_generation_wiki = true;
        } else if (a == "--wiki-prompt-len" && i + 1 < argc) {
            wiki_prompt_len = std::stoi(argv[++i]);
        } else if (a == "--wiki-max-chars" && i + 1 < argc) {
            wiki_max_chars = std::stoi(argv[++i]);
        } else if (a == "--wiki-train-steps" && i + 1 < argc) {
            wiki_train_steps = std::stoi(argv[++i]);
        } else if (a == "--wiki-eval-n" && i + 1 < argc) {
            wiki_eval_n = std::stoi(argv[++i]);
        } else if (a == "--latency-iters" && i + 1 < argc) {
            latency_iters = std::stoi(argv[++i]);
        } else if (a == "--train-cap" && i + 1 < argc) {
            train_cap = std::stoi(argv[++i]);
        } else if (a == "--eval-cap" && i + 1 < argc) {
            eval_cap = std::stoi(argv[++i]);
        } else if (a == "--topk-n" && i + 1 < argc) {
            topk_n = std::stoi(argv[++i]);
        } else if (a == "--gen-tokens" && i + 1 < argc) {
            gen_tokens = std::stoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            std::puts(
                "cyphalm_llm_profile [--wiki-only] [--skip-wiki] [--skip-negative-control] "
                "[--skip-roundtrip] [--wiki-max-chars N] [--wiki-train-steps N] "
                "[--wiki-eval-n N] [--wiki-prompt-len N] [--latency-iters N] "
                "[--train-cap N] [--eval-cap N] [--topk-n N] [--gen-tokens N]");
            return 0;
        }
    }

    try {
        nlohmann::json report;
        report["harness"] = "cyphalm_llm_profile";
        report["timestamp_utc"] = "measured_at_run";

        if (wiki_only) {
            skip_construct = true;
            skip_latency = true;
            skip_throughput = true;
            skip_negative_control = true;
            skip_context = true;
            skip_roundtrip = true;
        }

        auto cfg = production_cfg();
        if (!skip_construct) {
            print_section("construct_rss", report["construct_rss"] = profile_construct_and_rss());
        }
        cypha::cyphalm::CyphaLMModel model(cfg);
        const auto fixture = fixture_pattern(16384);

        if (!skip_latency) {
            print_section("latency",
                          report["latency"] = profile_latency(model, latency_warm, latency_iters));
        }
        if (!skip_throughput) {
            nlohmann::json tp = profile_throughput(model, fixture, train_cap, eval_cap);
            tp["note"] =
                "Timing on synthetic stream; eval_bpc here is NOT a capability metric (see "
                "wikitext_capability).";
            print_section("throughput_synthetic_timing", report["throughput_synthetic_timing"] = tp);
        }

        if (!skip_negative_control) {
            const TopKStats neg = eval_stream_metrics(model, fixture, std::min(16, topk_n), true);
            nlohmann::json neg_j = topk_stats_to_json(neg, "synthetic_arithmetic_pattern");
            neg_j["note"] =
                "Negative control: near-uniform ~7-8 BPC expected; not natural language.";
            print_section("negative_control_synthetic", report["negative_control_synthetic"] = neg_j);
        }

        const cypha::cyphalm::LMCorpus* wiki_corpus = nullptr;
        std::optional<cypha::cyphalm::LMCorpus> wiki_loaded;
        if (!skip_wiki) {
            try {
                wiki_loaded = cypha::cyphalm::load_bench_corpus("d21", wiki_max_chars, cfg.vocab_size);
                wiki_corpus = &wiki_loaded.value();
                nlohmann::json wiki_cfg;
                wiki_cfg["hp_profile"] = "light";
                wiki_cfg["hp_table_bits"] = cfg.hp_table_bits;
                wiki_cfg["hp_slot_max"] = cfg.hp_slot_max;
                wiki_cfg["wiki_max_chars"] = wiki_max_chars;
                print_section("wikitext_capability",
                              report["wikitext_capability"] = capability_wiki_suite(
                                  model, *wiki_corpus, wiki_train_steps, wiki_eval_n, wiki_prompt_len,
                                  gen_tokens, !skip_generation_wiki));
                report["wikitext_capability"]["config"] = wiki_cfg;
            } catch (const std::exception& ex) {
                nlohmann::json err;
                err["error"] = ex.what();
                print_section("wikitext_capability", report["wikitext_capability"] = err);
            }
        }

        if (!skip_context && wiki_corpus != nullptr) {
            model.reset_context();
            if (wiki_train_steps > 0 && wiki_corpus->train_ids.size() >= 2) {
                model.train_sequence(wiki_corpus->train_ids, wiki_train_steps, 1, nullptr);
            }
            print_section("context_scaling_wikitext",
                          report["context_scaling_wikitext"] = capability_context_scaling(
                              model, wiki_corpus->train_ids, {256, 1024, 8192}));
        }

        if (!skip_roundtrip) {
            std::vector<std::uint32_t> rt;
            for (int i = 0; i < 128; ++i) {
                rt.push_back(static_cast<std::uint32_t>((i * 13 + 5) % 96 + 32));
            }
            print_section("codec_roundtrip",
                          report["codec_roundtrip"] = capability_roundtrip(model, rt));
        }

        const auto rss_final = read_proc_status();
        nlohmann::json peak;
        peak["vm_rss_kb_final"] = rss_final.vm_rss_kb;
        peak["vm_hwm_kb_final"] = rss_final.vm_hwm_kb;
        print_section("peak_rss", report["peak_rss"] = peak);

        std::cout << "=== report_json_one_line ===\n" << report.dump() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cyphalm_llm_profile: %s\n", ex.what());
        return 1;
    }
}
