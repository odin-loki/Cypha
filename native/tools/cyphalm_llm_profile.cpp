/// CyphaLM (hp) profile + LLM capability harness — measured output only (no invented metrics).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
        "HpSequenceBackend holds pred_ + scratch_ + DFS checkpoints; next_byte_log_probs uses bit-tree DFS (legacy: CYPHA_HP_LEGACY_BYTE_LOGPROBS=1)";
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

std::vector<int> load_raw_bytes_file(const std::string& path, int max_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open corpus file: " + path);
    }
    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(std::max(0, max_bytes)));
    int ch = 0;
    while (max_bytes <= 0 || static_cast<int>(ids.size()) < max_bytes) {
        ch = in.get();
        if (ch == std::char_traits<char>::eof()) {
            break;
        }
        ids.push_back(ch & 0xff);
    }
    if (ids.empty()) {
        throw std::runtime_error("corpus file empty: " + path);
    }
    return ids;
}

struct HpCompressResult {
    bool ok = false;
    long long input_bytes = 0;
    long long archive_bytes = 0;
    double archive_bpc = std::numeric_limits<double>::quiet_NaN();
    std::string stderr_tail;
};

HpCompressResult run_hp_compress(const std::string& hp_tool, const std::string& corpus_path,
                                 const std::string& archive_path, int mem_bits, int mixer_lr,
                                 bool gria) {
    HpCompressResult out;
    std::ostringstream cmd;
    cmd << "\"" << hp_tool << "\" c --mem " << mem_bits << " --lr " << mixer_lr;
    if (!gria) {
        cmd << " --no-gria";
    }
    cmd << " \"" << corpus_path << "\" \"" << archive_path << "\" 2>&1";
    std::FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        out.stderr_tail = "popen failed";
        return out;
    }
    std::string captured;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        captured += buf;
    }
    const int rc = pclose(pipe);
    out.stderr_tail = captured.size() > 4000 ? captured.substr(captured.size() - 4000) : captured;

    long long in_bytes = 0;
    long long arc_bytes = 0;
    long long bpc_whole = 0;
    long long bpc_frac = 0;
    if (std::sscanf(captured.c_str(), "\rin %lld B  out %ld B", &in_bytes,
                    reinterpret_cast<long*>(&arc_bytes)) >= 2 ||
        std::sscanf(captured.c_str(), "in %lld B  out %ld B", &in_bytes,
                    reinterpret_cast<long*>(&arc_bytes)) >= 2) {
        out.input_bytes = in_bytes;
        out.archive_bytes = arc_bytes;
    }
    const char* bpc_pos = std::strstr(captured.c_str(), "bpc ");
    if (bpc_pos != nullptr &&
        std::sscanf(bpc_pos, "bpc %lld.%lld", &bpc_whole, &bpc_frac) >= 2) {
        out.archive_bpc =
            static_cast<double>(bpc_whole) + static_cast<double>(bpc_frac) / 1000.0;
    } else if (out.input_bytes > 0 && out.archive_bytes > 0) {
        out.archive_bpc =
            static_cast<double>(out.archive_bytes) * 8.0 / static_cast<double>(out.input_bytes);
    }
    out.ok = (rc == 0 && out.input_bytes > 0 && std::isfinite(out.archive_bpc));
    return out;
}

std::string write_temp_corpus(const std::vector<int>& ids, int n_bytes) {
    const int n = std::min(n_bytes, static_cast<int>(ids.size()));
    const std::string path = "/tmp/cyphalm_bpc_gap_corpus.bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write temp corpus: " + path);
    }
    for (int i = 0; i < n; ++i) {
        const char b = static_cast<char>(ids[static_cast<std::size_t>(i)] & 0xff);
        out.put(b);
    }
    if (!out) {
        throw std::runtime_error("temp corpus write failed: " + path);
    }
    return path;
}

nlohmann::json measure_bpc_gap(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids,
                               int observe_bytes, int clone_eval_n, int clone_train_steps,
                               const std::string& corpus_label, const std::string& corpus_path,
                               const std::string& hp_tool, bool run_hp) {
    nlohmann::json j;
    j["corpus_label"] = corpus_label;
    j["corpus_path"] = corpus_path;
    j["total_bytes_available"] = ids.size();
    j["hp_table_bits"] = model.config().hp_table_bits;
    j["hp_slot_max"] = model.config().hp_slot_max;
    j["hp_slot_compile_max"] = cypha::cyphalm::hp_compile_slot_max();
    j["hp_mixer_lr"] = model.config().hp_mixer_lr;
    j["hp_gria"] = model.config().hp_gria;

    const int n_observe = std::min(observe_bytes, static_cast<int>(ids.size()));
    j["observe_bytes"] = n_observe;

    std::cerr << "bpc_gap: observe_stream n=" << n_observe << " ...\n";
    const auto t0 = Clock::now();
    const double observe_bpc = model.eval_bpc_compress_equivalent(ids, n_observe);
    j["cypha_observe_bpc"] = observe_bpc;
    j["cypha_compress_equivalent_bpc"] = observe_bpc;
    j["cypha_observe_ms"] = elapsed_ms(t0);
    j["cypha_observe_method"] =
        "eval_bpc_compress_equivalent: reset_context; sum -log2 p(bit) via observe_next_byte";

    if (run_hp && !hp_tool.empty()) {
        std::string compress_path = corpus_path;
        std::string temp_corpus;
        if (compress_path.empty() || n_observe < static_cast<int>(ids.size())) {
            temp_corpus = write_temp_corpus(ids, n_observe);
            compress_path = temp_corpus;
        }
        const std::string archive_path = compress_path + ".cyhp.tmp";
        std::cerr << "bpc_gap: hp compress via " << hp_tool << " on " << compress_path
                  << " ...\n";
        const auto t1 = Clock::now();
        const HpCompressResult hp =
            run_hp_compress(hp_tool, compress_path, archive_path, model.config().hp_table_bits,
                            model.config().hp_mixer_lr, model.config().hp_gria);
        j["hp_compress_corpus_path"] = compress_path;
        j["hp_compress_ms"] = elapsed_ms(t1);
        j["hp_compress_ok"] = hp.ok;
        j["hp_input_bytes"] = hp.input_bytes;
        j["hp_archive_bytes"] = hp.archive_bytes;
        j["hp_archive_bpc"] = hp.archive_bpc;
        j["hp_archive_bpc_formula"] = "archive_bytes * 8 / input_bytes (includes header+arith)";
        j["hp_stderr_tail"] = hp.stderr_tail;
        std::remove(archive_path.c_str());
        if (!temp_corpus.empty()) {
            std::remove(temp_corpus.c_str());
        }
        if (hp.ok && std::isfinite(observe_bpc)) {
            j["observe_minus_archive_bpc"] = observe_bpc - hp.archive_bpc;
        }
    } else {
        j["hp_compress_skipped"] = true;
        if (hp_tool.empty()) {
            j["hp_compress_skip_reason"] = "pass --hp-tool PATH to measure archive BPC";
        }
    }

    const int n_clone = std::min(clone_eval_n, static_cast<int>(ids.size()) - 1);
    j["clone_eval_n"] = n_clone;
    if (n_clone > 0) {
        std::cerr << "bpc_gap: eval_bpc clone path n=" << n_clone << " (cold) ...\n";
        const auto t2 = Clock::now();
        j["cypha_eval_bpc_cold"] = model.eval_bpc(ids, n_clone);
        j["cypha_eval_bpc_cold_ms"] = elapsed_ms(t2);
        j["cypha_eval_bpc_cold_method"] =
            "eval_bpc: reset; predict_next(token_i) + clone log_probs; scores n-1 transitions";

        if (clone_train_steps > 0 && ids.size() >= 2) {
            std::cerr << "bpc_gap: train_sequence " << clone_train_steps
                      << " then eval_bpc n=" << n_clone << " ...\n";
            model.reset_context();
            model.train_sequence(ids, clone_train_steps, 1, nullptr);
            const auto t3 = Clock::now();
            j["cypha_eval_bpc_after_train"] = model.eval_bpc(ids, n_clone);
            j["cypha_eval_bpc_after_train_ms"] = elapsed_ms(t3);
            j["clone_train_steps"] = clone_train_steps;
            j["cypha_eval_bpc_after_train_method"] =
                "train_sequence (predict_next+adapt) then eval_bpc without reset";
        }
    }

    j["metric_definitions"] = {
        {"hp_archive_bpc", "Lossless CYHP file size / raw bytes (overhead included)"},
        {"cypha_observe_bpc", "Ideal NLL from hp::Predictor bit path (no arith rounding)"},
        {"cypha_eval_bpc_cold", "256-clone predict_next API; cold start; skips P(b0|empty)"},
        {"cypha_eval_bpc_after_train", "Same clone API after train_sequence (may double-consume)"},
    };
    return j;
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
    bool skip_bpc_gap = false;
    bool bpc_gap_only = false;
    int bpc_gap_bytes = 100000;
    int bpc_gap_clone_n = 16;
    int bpc_gap_clone_train = 32;
    std::string corpus_file;
    std::string hp_tool;
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
        } else if (a == "--skip-bpc-gap") {
            skip_bpc_gap = true;
        } else if (a == "--bpc-gap-only") {
            bpc_gap_only = true;
        } else if (a == "--bpc-gap-bytes" && i + 1 < argc) {
            bpc_gap_bytes = std::stoi(argv[++i]);
        } else if (a == "--bpc-gap-clone-n" && i + 1 < argc) {
            bpc_gap_clone_n = std::stoi(argv[++i]);
        } else if (a == "--bpc-gap-clone-train" && i + 1 < argc) {
            bpc_gap_clone_train = std::stoi(argv[++i]);
        } else if (a == "--corpus-file" && i + 1 < argc) {
            corpus_file = argv[++i];
        } else if (a == "--hp-tool" && i + 1 < argc) {
            hp_tool = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::puts(
                "cyphalm_llm_profile [--wiki-only] [--bpc-gap-only] [--skip-bpc-gap] "
                "[--skip-wiki] [--skip-negative-control] [--skip-roundtrip] "
                "[--wiki-max-chars N] [--wiki-train-steps N] [--wiki-eval-n N] "
                "[--bpc-gap-bytes N] [--bpc-gap-clone-n N] [--bpc-gap-clone-train N] "
                "[--corpus-file PATH] [--hp-tool PATH] [--wiki-prompt-len N] "
                "[--latency-iters N] [--train-cap N] [--eval-cap N] [--topk-n N] "
                "[--gen-tokens N]");
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
        if (bpc_gap_only) {
            skip_construct = true;
            skip_latency = true;
            skip_throughput = true;
            skip_negative_control = true;
            skip_context = true;
            skip_roundtrip = true;
            skip_wiki = true;
            skip_generation_wiki = true;
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
                wiki_cfg["hp_profile"] = "gate24";
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

        if (!skip_bpc_gap) {
            try {
                std::string gap_path = corpus_file;
                std::vector<int> gap_ids;
                std::string gap_label;
                if (!gap_path.empty()) {
                    gap_ids = load_raw_bytes_file(gap_path, bpc_gap_bytes);
                    gap_label = gap_path;
                } else {
                    if (wiki_corpus == nullptr) {
                        wiki_loaded =
                            cypha::cyphalm::load_bench_corpus("d21", wiki_max_chars, cfg.vocab_size);
                        wiki_corpus = &wiki_loaded.value();
                    }
                    gap_ids = wiki_corpus->train_ids;
                    if (static_cast<int>(gap_ids.size()) > bpc_gap_bytes) {
                        gap_ids.resize(static_cast<std::size_t>(bpc_gap_bytes));
                    }
                    gap_label = "wikitext2_train_bytes_cap";
                    gap_path = "bench/data/wikitext2/wikitext-2/wiki.train.tokens";
                }
                const bool run_hp = !hp_tool.empty();
                print_section("bpc_gap_same_corpus",
                              report["bpc_gap_same_corpus"] = measure_bpc_gap(
                                  model, gap_ids, bpc_gap_bytes, bpc_gap_clone_n,
                                  bpc_gap_clone_train, gap_label, gap_path, hp_tool, run_hp));
            } catch (const std::exception& ex) {
                nlohmann::json err;
                err["error"] = ex.what();
                print_section("bpc_gap_same_corpus", report["bpc_gap_same_corpus"] = err);
            }
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
