/// Lossy LLM lever benchmark: RSS, observe BPC, predict_next latency (measured only).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using Clock = std::chrono::steady_clock;

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

std::string sha256_file(const std::string& path) {
    std::ostringstream cmd;
    cmd << "sha256sum \"" << path << "\" 2>/dev/null | awk '{print $1}'";
    std::FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) return "";
    char buf[128];
    std::string hex;
    if (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        hex = buf;
        while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    }
    pclose(pipe);
    return hex;
}

std::string resolve_corpus_path(const std::string& path) {
    namespace fs = std::filesystem;
    if (path.empty()) return path;
    const fs::path p(path);
    if (fs::exists(p)) return fs::absolute(p).string();
    const fs::path native = fs::path(__FILE__).parent_path().parent_path();
    const fs::path repo = native.parent_path();
    const std::vector<fs::path> candidates = {
        repo / path,
        fs::path("..") / path,
        fs::path("../..") / path,
    };
    for (const auto& c : candidates) {
        if (fs::exists(c)) return fs::absolute(c).string();
    }
    return path;
}

std::vector<int> load_bytes_file(const std::string& path, int max_n) {
    const std::string resolved = resolve_corpus_path(path);
    std::ifstream in(resolved, std::ios::binary);
    std::vector<int> out;
    int ch;
    while (in && (max_n <= 0 || static_cast<int>(out.size()) < max_n)) {
        ch = in.get();
        if (ch == EOF) break;
        out.push_back(ch & 0xff);
    }
    return out;
}

nlohmann::json run_variant(const char* label, cypha::cyphalm::CyphaLMConfig cfg,
                           const std::vector<int>& ids, int warmup_n, int eval_n,
                           int latency_iters) {
    nlohmann::json j;
    j["variant"] = label;
    j["hp_table_bits"] = cfg.hp_table_bits;
    j["hp_lossy_mem"] = cfg.hp_lossy_mem;
    j["hp_effective_mem"] = cypha::cyphalm::hp_effective_table_bits(cfg);
    j["hp_serve_compact"] = cfg.hp_serve_compact;
    j["hp_prune_cold_min_n"] = cfg.hp_prune_cold_min_n;

    const auto t_construct = Clock::now();
    cypha::cyphalm::CyphaLMModel model(cfg);
    j["construct_ms"] =
        std::chrono::duration<double, std::milli>(Clock::now() - t_construct).count();

    const auto rss_init = read_proc_status();
    j["vm_rss_kb_init"] = rss_init.vm_rss_kb;
    j["vm_hwm_kb_init"] = rss_init.vm_hwm_kb;

    const int warm = std::min(warmup_n, static_cast<int>(ids.size()));
    for (int i = 0; i < warm; ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(ids[static_cast<std::size_t>(i)]));
    }

    if (cfg.hp_prune_cold_min_n > 0) {
        model.prune_hp_cold_slots(cfg.hp_prune_cold_min_n);
        j["pruned_after_warmup"] = true;
    }

    const auto rss_warm = read_proc_status();
    j["vm_rss_kb_after_warmup"] = rss_warm.vm_rss_kb;

    model.reset_context();
    const int n_eval = std::min(eval_n, static_cast<int>(ids.size()));
    const auto t_bpc = Clock::now();
    const double bpc = model.eval_bpc(ids, n_eval);
    const double bpc_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_bpc).count();
    j["observe_bpc"] = bpc;
    j["observe_bpc_n"] = n_eval;
    j["observe_bpc_ms"] = bpc_ms;
    j["observe_bytes_per_sec"] = bpc_ms > 0 ? static_cast<double>(n_eval) / (bpc_ms / 1000.0) : 0.0;

    model.reset_context();
    for (int i = 0; i < std::min(64, static_cast<int>(ids.size())); ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(ids[static_cast<std::size_t>(i)]));
    }
    if (latency_iters > 0) {
        const auto t_lat = Clock::now();
        for (int i = 0; i < latency_iters; ++i) {
            (void)model.predict_next(
                static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i % ids.size())]));
        }
        const double lat_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t_lat).count();
        j["predict_next_ms_per_call"] = lat_ms / latency_iters;
        j["predict_next_iters"] = latency_iters;
    } else {
        j["predict_next_skipped"] = true;
    }

    const auto rss_peak = read_proc_status();
    j["vm_rss_kb_after_bench"] = rss_peak.vm_rss_kb;
    j["vm_hwm_kb_after_bench"] = rss_peak.vm_hwm_kb;
    return j;
}

}  // namespace

int main(int argc, char** argv) {
    int warmup_n = 65536;
    int eval_n = 100000;
    int latency_iters = 2;
    std::string corpus_path;
    bool enwik_screen = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--warmup-n" && i + 1 < argc) warmup_n = std::stoi(argv[++i]);
        else if (a == "--eval-n" && i + 1 < argc) eval_n = std::stoi(argv[++i]);
        else if (a == "--latency-iters" && i + 1 < argc) latency_iters = std::stoi(argv[++i]);
        else if (a == "--corpus" && i + 1 < argc) corpus_path = argv[++i];
        else if (a == "--enwik-screen") {
            enwik_screen = true;
            corpus_path = "bench/data/enwik8/enwik8.8mb";
            warmup_n = 0;
            eval_n = 8388608;
            latency_iters = 0;
        }
    }

    std::vector<int> ids;
    if (!corpus_path.empty()) {
        corpus_path = resolve_corpus_path(corpus_path);
        ids = load_bytes_file(corpus_path, std::max(warmup_n, eval_n) + 1024);
        if (enwik_screen && static_cast<int>(ids.size()) < eval_n) {
            std::fprintf(stderr,
                         "enwik screen: corpus %s has %zu bytes, need %d\n", corpus_path.c_str(),
                         ids.size(), eval_n);
            return 1;
        }
    }
    if (ids.empty()) {
        const cypha::cyphalm::LMCorpus wiki =
            cypha::cyphalm::load_bench_corpus("d21", 500000, 256);
        ids = wiki.train_ids;
    }
    if (ids.empty()) {
        ids = cypha::cyphalm::synthetic_corpus(std::max(warmup_n, eval_n) + 1024, 256, 42);
    }

    nlohmann::json out;
    out["harness"] = enwik_screen ? "cyphalm_lossy_enwik_screen" : "cyphalm_lossy_bench";
    out["corpus_path"] = corpus_path;
    if (!corpus_path.empty()) {
        out["corpus_sha256"] = sha256_file(corpus_path);
    }
    out["corpus_bytes"] = ids.size();
    out["warmup_n"] = warmup_n;
    out["eval_n"] = eval_n;
    out["latency_iters"] = latency_iters;
    out["gate24_quality_bar_bpc"] = 1.612;
    out["cypha_gate24_ref_observe_bpc"] = 1.611729;
    out["note"] =
        "observe_bpc is compress-faithful bit-serial (eval_bpc); predict_next skipped when "
        "latency_iters=0";

    nlohmann::json variants = nlohmann::json::array();

    auto add_baseline = [&]() {
        auto cfg = cypha::cyphalm::CyphaLMConfig{};
        cypha::cyphalm::apply_hp_production_recipe(cfg);
        cfg.vocab_size = 256;
        variants.push_back(run_variant("gate24_baseline_mem22", cfg, ids, warmup_n, eval_n,
                                       latency_iters));
    };
    auto add_mem20 = [&]() {
        auto cfg = cypha::cyphalm::CyphaLMConfig{};
        cypha::cyphalm::apply_hp_lossy_recipe(cfg, 20);
        cfg.vocab_size = 256;
        variants.push_back(run_variant("lossy_mem20", cfg, ids, warmup_n, eval_n, latency_iters));
    };
    auto add_serve_compact = [&]() {
        auto cfg = cypha::cyphalm::CyphaLMConfig{};
        cypha::cyphalm::apply_hp_production_recipe(cfg);
        cfg.vocab_size = 256;
        cfg.hp_serve_compact = true;
        variants.push_back(run_variant("serve_compact_mem22", cfg, ids, warmup_n, eval_n,
                                       latency_iters));
    };
    auto add_prune = [&]() {
        auto cfg = cypha::cyphalm::CyphaLMConfig{};
        cypha::cyphalm::apply_hp_production_recipe(cfg);
        cfg.vocab_size = 256;
        cfg.hp_prune_cold_min_n = 4;
        variants.push_back(run_variant("prune_cold_min4_mem22", cfg, ids, warmup_n, eval_n,
                                       latency_iters));
    };
    auto add_combo = [&]() {
        auto cfg = cypha::cyphalm::CyphaLMConfig{};
        cypha::cyphalm::apply_hp_lossy_recipe(cfg, 20);
        cfg.vocab_size = 256;
        cfg.hp_serve_compact = true;
        cfg.hp_prune_cold_min_n = 4;
        variants.push_back(run_variant("combo_mem20_compact_prune4", cfg, ids, warmup_n, eval_n,
                                       latency_iters));
    };

    if (enwik_screen) {
        add_baseline();
        add_mem20();
        add_combo();
    } else {
        add_baseline();
        add_mem20();
        add_serve_compact();
        add_prune();
        add_combo();
    }

    out["variants"] = variants;
    std::cout << out.dump(2) << std::endl;
    return 0;
}
