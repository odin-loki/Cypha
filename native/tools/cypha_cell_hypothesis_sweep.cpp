// cypha_cell_hypothesis_sweep — smoke runner for cell hypothesis variants (Tier 1+2).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_tune.hpp"
#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_math_integration.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_completeness.hpp"

namespace {

struct Args {
    std::string profile = "d17";
    int n_train = 4000;
    int n_eval = 500;
    int threads = 0;
    bool smoke = false;
    bool tier1_only = false;
    bool tier2_only = false;
    bool tier2_smoke = false;
    bool tier3_smoke = false;
    bool overnight_sweep = false;
    bool overnight_sweep_smoke = false;
    bool list_variants = false;
    bool n_train_explicit = false;
    bool n_eval_explicit = false;
    bool intelligence_profile = false;
    bool math_integration = false;
    std::int64_t bench_seed = -1;
    std::string cell_variant;
    std::string output_dir;
};

void usage() {
    std::cerr << "usage: cypha_cell_hypothesis_sweep [--smoke] [--tier1-only] [--tier2-only]\n"
              << "       [--tier2-smoke] [--tier3-smoke] [--overnight-sweep] [--overnight-sweep-smoke]\n"
              << "       [--list-variants] [--cell-variant H06] [--output-dir PATH]\n"
              << "       [--profile d17] [--n-train N] [--n-eval M] [--threads T]\n"
              << "       [--intelligence-profile] [--math-integration] [--bench-seed N]\n"
              << "  HISTORICAL research tool (One Cypha cutover 2026-07-18). Living spine: Hybrid L2+Wave2 BPTT / cypha::Cypha.\n"
              << "  overnight sweep: --intelligence-profile + --math-integration auto when "
                 "CYPHA_OVERNIGHT_MATH_INTEGRATION=1\n"
              << "  overnight sweep variants run isolated in child processes (re-invoking this "
                 "binary with --cell-variant); a variant crash is recorded in \"failed\" and does "
                 "not discard other variants' results.\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--profile") a.profile = need("--profile");
        else if (k == "--n-train") {
            a.n_train = std::stoi(need("--n-train"));
            a.n_train_explicit = true;
        } else if (k == "--n-eval") {
            a.n_eval = std::stoi(need("--n-eval"));
            a.n_eval_explicit = true;
        } else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--smoke") a.smoke = true;
        else if (k == "--tier1-only") a.tier1_only = true;
        else if (k == "--tier2-only") a.tier2_only = true;
        else if (k == "--tier2-smoke") a.tier2_smoke = true;
        else if (k == "--tier3-smoke") a.tier3_smoke = true;
        else if (k == "--overnight-sweep-smoke") {
            a.overnight_sweep = true;
            a.overnight_sweep_smoke = true;
        } else if (k == "--overnight-sweep") a.overnight_sweep = true;
        else if (k == "--list-variants") a.list_variants = true;
        else if (k == "--cell-variant") a.cell_variant = need("--cell-variant");
        else if (k == "--output-dir") a.output_dir = need("--output-dir");
        else if (k == "--intelligence-profile") a.intelligence_profile = true;
        else if (k == "--math-integration") a.math_integration = true;
        else if (k == "--bench-seed") {
            a.bench_seed = std::stoll(need("--bench-seed"));
        }
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    if (a.tier2_smoke || a.tier3_smoke) {
        a.n_train = 200;
        a.n_eval = 40;
    }
    if (a.smoke) {
        a.n_train = 400;
        a.n_eval = 80;
        a.tier1_only = true;
    }
    if (a.overnight_sweep) {
        if (a.overnight_sweep_smoke) {
            if (!a.n_train_explicit) a.n_train = 200;
            if (!a.n_eval_explicit) a.n_eval = 40;
        } else {
            if (!a.n_train_explicit) {
                a.n_train = cypha::bench::bench_overnight_enabled() ? cypha::bench::bench_full_n_train()
                                                                    : 500;
            }
            if (!a.n_eval_explicit) {
                a.n_eval = cypha::bench::bench_overnight_enabled() ? 2000 : 80;
            }
        }
    }
    if (cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
        if (!a.n_train_explicit) a.n_train = cypha::bench::bench_scale(a.n_train, 80);
        if (!a.n_eval_explicit) a.n_eval = cypha::bench::bench_scale(a.n_eval, 32);
    }
    if (a.overnight_sweep && cypha::bench::bench_env_truthy("CYPHA_OVERNIGHT_MATH_INTEGRATION")) {
        a.intelligence_profile = true;
        a.math_integration = true;
    }
    if (a.math_integration) {
        a.intelligence_profile = true;
    }
    return a;
}

void ensure_overnight_env() {
    if (!cypha::bench::bench_overnight_enabled()) {
        return;
    }
#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FULL_CORPUS", "1");
    _putenv_s("CYPHA_BENCH_OVERNIGHT", "1");
#else
    setenv("CYPHA_BENCH_FULL_CORPUS", "1", 1);
    setenv("CYPHA_BENCH_OVERNIGHT", "1", 1);
#endif
}

bool should_run(const cypha::cyphalm::CellVariantSpec& v, const Args& args) {
    if (!v.runnable) {
        return false;
    }
    if (!args.cell_variant.empty()) {
        return v.id == args.cell_variant;
    }
    if (args.overnight_sweep_smoke) {
        return v.id == "B2" || v.id == "H06" || v.id == "H14";
    }
    if (args.overnight_sweep) {
        return true;
    }
    if (args.tier2_smoke) {
        return v.id == "H06" || v.id == "H08" || v.id == "H14";
    }
    if (args.tier3_smoke) {
        return v.id == "H09" || v.id == "H12" || v.id == "H18" || v.id == "H16";
    }
    if (args.tier2_only) {
        return v.tier == 2;
    }
    if (args.tier1_only) {
        return v.tier == 1;
    }
    return v.tier <= 2 || v.id[0] == 'B';
}

std::string csv_field(const std::string& raw) {
    if (raw.find_first_of(",\"\n\r") == std::string::npos) {
        return raw;
    }
    std::string out = "\"";
    for (char c : raw) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

/// Test-only crash injection hook (opt-in via env var, unset by default in all real runs):
/// lets us verify in CI/manually that isolated per-variant child processes crashing doesn't
/// discard other variants' already-completed results, without needing to reproduce a real
/// native bug on demand.
void maybe_force_test_crash(const std::string& variant_id) {
    const char* target = std::getenv("CYPHA_SWEEP_FORCE_CRASH_VARIANT");
    if (target != nullptr && variant_id == target) {
        std::cerr << "[cell_sweep] CYPHA_SWEEP_FORCE_CRASH_VARIANT=" << target
                  << " — intentionally aborting for resilience test" << std::endl;
        std::abort();
    }
}

nlohmann::json run_variant(const cypha::cyphalm::CellVariantSpec& spec, const Args& args) {
    maybe_force_test_crash(spec.id);
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_bench_profile(args.profile, cfg);
    cypha::cyphalm::apply_cell_variant(spec.id, cfg);
    if (const char* seed_env = std::getenv("CYPHA_BENCH_SEED")) {
        cfg.seed = static_cast<std::uint64_t>(std::stoull(seed_env));
    }
    if (args.bench_seed >= 0) {
        cfg.seed = static_cast<std::uint64_t>(args.bench_seed);
    }
    if (args.math_integration) {
        cypha::cyphalm::apply_math_integration_preset(cfg);
    }
    if (args.profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;

    cypha::cyphalm::LMCorpus corpus;
    bool synthetic = false;
    try {
        corpus = cypha::cyphalm::load_bench_corpus(args.profile, 1'000'000, cfg.vocab_size,
                                                   cfg.bpe_merges_path, cfg.bpe_vocab_path);
    } catch (const std::exception&) {
        synthetic = true;
        corpus.profile = args.profile;
        corpus.source = "synthetic";
        corpus.vocab_size = cfg.vocab_size;
        corpus.train_ids = cypha::cyphalm::synthetic_corpus(args.n_train + args.n_eval + 64,
                                                            cfg.vocab_size, cfg.seed);
        const std::size_t split = static_cast<std::size_t>(args.n_train);
        corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                               corpus.train_ids.end());
        corpus.train_ids.resize(split);
    }
    cfg.vocab_size = corpus.vocab_size;

    cypha::cyphalm::CyphaLMModel model(cfg);
    cypha::intelligence::IntelligenceProfiler profiler;
    model.train_sequence(corpus.train_ids, args.n_train, cfg.train_epochs);
    const double bpc = model.eval_bpc(corpus.eval_ids, args.n_eval);
    if (args.intelligence_profile) {
        model.accumulate_intelligence_profile(corpus.eval_ids, args.n_eval, profiler);
    }
    const auto alpha_profile = model.compression_profile();

    nlohmann::json row = {
        {"id", spec.id},
        {"name", spec.name},
        {"tier", spec.tier},
        {"bench_mode", spec.bench_mode},
        {"notes", spec.notes},
        {"bpc", std::isnan(bpc) ? nullptr : nlohmann::json(bpc)},
        {"corpus", corpus.source},
        {"synthetic", synthetic},
        {"mean_alpha", alpha_profile.value("mean_alpha", 0.0)},
        {"n_train", args.n_train},
        {"n_eval", args.n_eval},
        {"math_integration", args.math_integration},
    };
    if (spec.id == "B2" && !std::isnan(bpc)) {
        row["delta_vs_b2"] = 0.0;
    }
    if (args.intelligence_profile) {
        const auto completeness = cypha::intelligence::validate_profile_completeness(profiler);
        row["kappa"] = completeness.kappa;
        row["profile_completeness"] =
            cypha::intelligence::profile_completeness_to_json(completeness);
    }
    return row;
}

std::string extract_json_blob(const std::string& mixed) {
    const std::size_t start = mixed.find('{');
    const std::size_t end = mixed.rfind('}');
    if (start != std::string::npos && end != std::string::npos && end >= start) {
        return mixed.substr(start, end - start + 1);
    }
    return mixed;
}

/// Build the child argv for an isolated single-variant re-invocation of this same binary.
/// Always resolves n_train/n_eval to the *already-computed* effective values (not the raw
/// CLI flags) so the child's own parse_args() defaulting logic can't drift from the parent.
std::vector<std::string> isolated_variant_args(const std::string& variant_id, const Args& args) {
    std::vector<std::string> child_args = {
        "--cell-variant", variant_id,
        "--profile",      args.profile,
        "--n-train",      std::to_string(args.n_train),
        "--n-eval",       std::to_string(args.n_eval),
        "--threads",      std::to_string(args.threads),
    };
    if (args.intelligence_profile) {
        child_args.push_back("--intelligence-profile");
    }
    if (args.math_integration) {
        child_args.push_back("--math-integration");
    }
    if (args.bench_seed >= 0) {
        child_args.push_back("--bench-seed");
        child_args.push_back(std::to_string(args.bench_seed));
    }
    return child_args;
}

/// Runs one cell-sweep variant in its own child process (re-invoking this same executable
/// with ``--cell-variant``) so a native crash in one variant (e.g. an access violation)
/// cannot take down the other variants' already-completed results. On success, returns the
/// variant's result row exactly as ``run_variant`` would have produced in-process. On
/// failure (nonzero/abnormal exit, or unparseable stdout), returns std::nullopt and fills
/// ``error_out``/``exit_code_out`` for the caller to record and continue past.
std::optional<nlohmann::json> run_variant_isolated(const std::filesystem::path& self_exe,
                                                    const cypha::cyphalm::CellVariantSpec& spec,
                                                    const Args& args, int& exit_code_out,
                                                    std::string& error_out) {
    const cypha::bench::RunProcessResult proc =
        cypha::bench::run_executable_capture(self_exe, isolated_variant_args(spec.id, args));
    exit_code_out = proc.exit_code;
    if (proc.exit_code != 0) {
        error_out = "child process exit=" + std::to_string(proc.exit_code) +
                   (proc.exit_code == -1073741819 ? " (0xC0000005 ACCESS_VIOLATION)" : "");
        return std::nullopt;
    }
    if (proc.stdout_text.empty()) {
        error_out = "child process produced no stdout";
        return std::nullopt;
    }
    try {
        const nlohmann::json child_out = nlohmann::json::parse(extract_json_blob(proc.stdout_text));
        if (!child_out.contains("results") || !child_out["results"].is_array() ||
            child_out["results"].empty()) {
            error_out = "child stdout JSON missing results[0]";
            return std::nullopt;
        }
        return child_out["results"].front();
    } catch (const std::exception& e) {
        error_out = std::string("child stdout JSON parse failed: ") + e.what();
        return std::nullopt;
    }
}

std::string iso_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

nlohmann::json build_pareto_ranked_variants(const nlohmann::json& rows, double w = 0.1) {
    struct RankedEntry {
        std::string id;
        double kappa;
        double bpc;
        double normalized_bpc;
        double pareto_score;
        bool nondominated;
    };
    std::vector<RankedEntry> ranked;
    ranked.reserve(rows.size());
    double min_bpc = std::numeric_limits<double>::infinity();
    double max_bpc = -std::numeric_limits<double>::infinity();
    for (const auto& row : rows) {
        if (!row.is_object() || !row.contains("kappa") || !row["kappa"].is_number() ||
            !row.contains("bpc") || row["bpc"].is_null() || !row["bpc"].is_number()) {
            continue;
        }
        const double kappa = row["kappa"].get<double>();
        const double bpc = row["bpc"].get<double>();
        if (!std::isfinite(kappa) || !std::isfinite(bpc)) {
            continue;
        }
        min_bpc = std::min(min_bpc, bpc);
        max_bpc = std::max(max_bpc, bpc);
        ranked.push_back({row.value("id", ""), kappa, bpc, 0.0, 0.0, false});
    }
    const double bpc_span = max_bpc - min_bpc;
    for (auto& entry : ranked) {
        entry.normalized_bpc = bpc_span > 0.0 ? (entry.bpc - min_bpc) / bpc_span : 0.0;
        entry.pareto_score = entry.kappa - w * entry.normalized_bpc;
    }
    for (auto& entry : ranked) {
        bool dominated = false;
        for (const auto& other : ranked) {
            if (other.id == entry.id) {
                continue;
            }
            if (other.kappa >= entry.kappa && other.bpc <= entry.bpc &&
                (other.kappa > entry.kappa || other.bpc < entry.bpc)) {
                dominated = true;
                break;
            }
        }
        entry.nondominated = !dominated;
    }
    std::sort(ranked.begin(), ranked.end(), [](const RankedEntry& a, const RankedEntry& b) {
        if (a.nondominated != b.nondominated) {
            return a.nondominated > b.nondominated;
        }
        return a.pareto_score > b.pareto_score;
    });
    nlohmann::json out = nlohmann::json::array();
    for (const auto& entry : ranked) {
        out.push_back({{"id", entry.id},
                       {"kappa", entry.kappa},
                       {"bpc", entry.bpc},
                       {"normalized_bpc", entry.normalized_bpc},
                       {"pareto_score", entry.pareto_score},
                       {"nondominated", entry.nondominated}});
    }
    return out;
}

void append_overnight_progress_log(const std::string& variant_id, int index, int total) {
    const std::filesystem::path log_path =
        cypha::bench::results_dir() / "cell_sweep" / "overnight_progress.log";
    std::filesystem::create_directories(log_path.parent_path());
    std::ofstream log(log_path, std::ios::app);
    if (log) {
        log << iso_timestamp_now() << " variant=" << variant_id << ' ' << index << '/' << total
            << '\n';
    }
}

void write_overnight_artifacts(const std::filesystem::path& out_dir, const nlohmann::json& results,
                               const Args& args, double b2_bpc, const nlohmann::json& failed) {
    std::filesystem::create_directories(out_dir);
    for (const auto& row : results) {
        const std::string id = row.value("id", "unknown");
        std::ofstream variant_out(out_dir / ("variant_" + id + ".json"));
        if (variant_out) {
            variant_out << row.dump(2);
        }
    }

    std::ofstream csv(out_dir / "summary.csv");
    if (csv) {
        csv << "id,name,tier,bench_mode,bpc,delta_vs_b2,mean_alpha,corpus,synthetic,n_train,n_eval";
        if (args.intelligence_profile) {
            csv << ",kappa";
        }
        csv << '\n';
        for (const auto& row : results) {
            const std::string id = row.value("id", "");
            const std::string name = row.value("name", "");
            const int tier = row.value("tier", 0);
            const std::string bench_mode = row.value("bench_mode", "");
            std::string bpc_str;
            if (row["bpc"].is_null()) {
                bpc_str = "";
            } else {
                bpc_str = std::to_string(row["bpc"].get<double>());
            }
            std::string delta_str;
            if (row.contains("delta_vs_b2") && !row["delta_vs_b2"].is_null()) {
                delta_str = std::to_string(row["delta_vs_b2"].get<double>());
            }
            const double mean_alpha = row.value("mean_alpha", 0.0);
            const std::string corpus = row.value("corpus", "");
            const bool synthetic = row.value("synthetic", false);
            csv << csv_field(id) << ',' << csv_field(name) << ',' << tier << ','
                << csv_field(bench_mode) << ',' << bpc_str << ',' << delta_str << ','
                << mean_alpha << ',' << csv_field(corpus) << ',' << (synthetic ? "1" : "0") << ','
                << args.n_train << ',' << args.n_eval;
            if (args.intelligence_profile) {
                std::string kappa_str;
                if (row.contains("kappa") && row["kappa"].is_number()) {
                    kappa_str = std::to_string(row["kappa"].get<double>());
                }
                csv << ',' << kappa_str;
            }
            csv << '\n';
        }
    }

    nlohmann::json manifest = {
        {"profile", args.profile},
        {"n_train", args.n_train},
        {"n_eval", args.n_eval},
        {"overnight", cypha::bench::bench_overnight_enabled()},
        {"overnight_sweep_smoke", args.overnight_sweep_smoke},
        {"math_integration", args.math_integration},
        {"variant_count", results.size()},
        {"b2_bpc", std::isnan(b2_bpc) ? nullptr : nlohmann::json(b2_bpc)},
        {"results", results},
        {"failed", failed},
    };
    if (args.intelligence_profile) {
        manifest["pareto_ranked_variants"] = build_pareto_ranked_variants(results);
        if (!manifest["pareto_ranked_variants"].empty()) {
            manifest["best_pareto_variant"] = manifest["pareto_ranked_variants"].front();
        }
    }
    std::ofstream manifest_out(out_dir / "manifest.json");
    if (manifest_out) {
        manifest_out << manifest.dump(2);
    }
}

std::filesystem::path resolve_self_exe(const char* argv0) {
    if (argv0 != nullptr) {
        const std::filesystem::path self = std::filesystem::absolute(argv0);
        if (std::filesystem::is_regular_file(self)) {
            return self;
        }
        if (self.has_parent_path()) {
            return cypha::bench::resolve_runner_exe("cypha_cell_hypothesis_sweep", self.parent_path());
        }
    }
    return cypha::bench::resolve_runner_exe("cypha_cell_hypothesis_sweep", std::filesystem::current_path());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        cypha::cyphalm::set_thread_count(args.threads);

        if (args.overnight_sweep && cypha::bench::bench_overnight_enabled()) {
            ensure_overnight_env();
        }

        // Isolate each variant in its own child process only for genuine sweep-orchestration
        // invocations (--overnight-sweep[-smoke] without a specific --cell-variant already
        // pinned down) — a single variant crashing (e.g. an access violation) then only loses
        // that one variant's result instead of discarding every variant that already
        // completed. --cell-variant invocations (including the isolated children themselves)
        // always run in-process, so there is no recursive re-isolation.
        const bool isolate_variants = args.overnight_sweep && args.cell_variant.empty();
        const std::filesystem::path self_exe =
            isolate_variants ? resolve_self_exe(argv[0]) : std::filesystem::path();

        if (args.list_variants) {
            nlohmann::json listed = nlohmann::json::array();
            for (const auto& v : cypha::cyphalm::all_cell_variants()) {
                listed.push_back({{"id", v.id},
                                  {"name", v.name},
                                  {"tier", v.tier},
                                  {"runnable", v.runnable},
                                  {"bench_mode", v.bench_mode},
                                  {"notes", v.notes}});
            }
            std::cout << listed.dump(2) << std::endl;
            return 0;
        }

        nlohmann::json results = nlohmann::json::array();
        nlohmann::json skipped = nlohmann::json::array();
        nlohmann::json failed = nlohmann::json::array();
        double b2_bpc = std::numeric_limits<double>::quiet_NaN();

        int variant_total = 0;
        for (const auto& v : cypha::cyphalm::all_cell_variants()) {
            if (should_run(v, args)) {
                ++variant_total;
            }
        }
        int variant_index = 0;
        for (const auto& v : cypha::cyphalm::all_cell_variants()) {
            if (!should_run(v, args)) {
                skipped.push_back({{"id", v.id}, {"name", v.name}, {"tier", v.tier}});
                continue;
            }
            if (args.overnight_sweep && !args.overnight_sweep_smoke) {
                ++variant_index;
                std::cerr << "[cell_sweep] variant " << v.id << " (" << variant_index << "/"
                          << variant_total << ")" << std::endl;
                append_overnight_progress_log(v.id, variant_index, variant_total);
            }
            if (isolate_variants) {
                int child_exit = 0;
                std::string child_error;
                auto row_opt = run_variant_isolated(self_exe, v, args, child_exit, child_error);
                if (!row_opt) {
                    std::cerr << "[cell_sweep] variant " << v.id
                              << " FAILED — recording as failed and continuing (" << child_error
                              << ")" << std::endl;
                    failed.push_back({{"id", v.id},
                                      {"name", v.name},
                                      {"tier", v.tier},
                                      {"exit_code", child_exit},
                                      {"error", child_error}});
                    continue;
                }
                auto row = std::move(*row_opt);
                if (v.id == "B2" && !row["bpc"].is_null()) {
                    b2_bpc = row["bpc"].get<double>();
                }
                results.push_back(std::move(row));
            } else {
                auto row = run_variant(v, args);
                if (v.id == "B2" && !row["bpc"].is_null()) {
                    b2_bpc = row["bpc"].get<double>();
                }
                results.push_back(std::move(row));
            }
        }

        for (auto& row : results) {
            if (!row["bpc"].is_null() && !std::isnan(b2_bpc) && row.value("id", "") != "B2") {
                row["delta_vs_b2"] = row["bpc"].get<double>() - b2_bpc;
            }
        }

        // Write variant_*.json / summary.csv whenever overnight-sweep OR an explicit
        // --output-dir is set (single --cell-variant H15 re-rows previously only printed
        // stdout and left variant_H15.json stale).
        const bool write_artifacts = args.overnight_sweep || !args.output_dir.empty();
        std::filesystem::path artifacts_dir;
        if (write_artifacts) {
            artifacts_dir = args.output_dir.empty()
                                ? cypha::bench::results_dir() / "cell_sweep"
                                : std::filesystem::path(args.output_dir);
            write_overnight_artifacts(artifacts_dir, results, args, b2_bpc, failed);
        }

        nlohmann::json out = {
            {"profile", args.profile},
            {"n_train", args.n_train},
            {"n_eval", args.n_eval},
            {"smoke", args.smoke || args.tier1_only || args.tier2_only || args.tier2_smoke || args.tier3_smoke},
            {"overnight_sweep", args.overnight_sweep},
            {"overnight_sweep_smoke", args.overnight_sweep_smoke},
            {"overnight", cypha::bench::bench_overnight_enabled()},
            {"cell_variant", args.cell_variant.empty() ? nullptr : nlohmann::json(args.cell_variant)},
            {"intelligence_profile", args.intelligence_profile},
            {"math_integration", args.math_integration},
            {"isolated_variants", isolate_variants},
            {"results", results},
            {"skipped", skipped},
            {"failed", failed},
            {"variant_count", cypha::cyphalm::all_cell_variants().size()},
        };
        if (write_artifacts) {
            out["output_dir"] = artifacts_dir.string();
        }
        if (args.intelligence_profile) {
            out["pareto_ranked_variants"] = build_pareto_ranked_variants(results);
            if (!out["pareto_ranked_variants"].empty()) {
                out["best_pareto_variant"] = out["pareto_ranked_variants"].front();
            }
        }
        std::cout << out.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cypha_cell_hypothesis_sweep: " << e.what() << "\n";
        usage();
        return 1;
    }
}
