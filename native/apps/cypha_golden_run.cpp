// cypha_golden_run — unified parity driver: dispatches to parity exes or inline checks.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "cypha/infer_cpu.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string fixture;
    fs::path dir;
    fs::path exe_dir;
    bool list_only = false;
};

struct FixtureEntry {
    const char* name;
    const char* exe;       // nullptr => inline
    const char* kind;      // "exe" | "inline"
    const char* description;
};

void usage() {
    std::cerr << "usage: cypha_golden_run --list\n"
              << "       cypha_golden_run --fixture NAME --dir PATH [--exe-dir DIR]\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--fixture") {
            a.fixture = need("--fixture");
        } else if (k == "--dir") {
            a.dir = need("--dir");
        } else if (k == "--exe-dir") {
            a.exe_dir = need("--exe-dir");
        } else if (k == "--list") {
            a.list_only = true;
        } else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

fs::path resolve_exe_dir(const Args& a, char** argv) {
    if (!a.exe_dir.empty()) {
        return fs::absolute(a.exe_dir);
    }
    if (argv && argv[0]) {
        const fs::path self = fs::absolute(argv[0]);
        if (self.has_parent_path()) {
            return self.parent_path();
        }
    }
    return fs::current_path();
}

std::string exe_name(const std::string& base) {
#ifdef _WIN32
    return base + ".exe";
#else
    return base;
#endif
}

std::string quote_arg(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

int run_process(const fs::path& exe, const std::vector<std::string>& args) {
#ifdef _WIN32
    std::string cmd = quote_arg(exe.string());
    for (const auto& arg : args) {
        cmd += ' ';
        cmd += quote_arg(arg);
    }
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        return 127;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
#else
    std::ostringstream cmd;
    cmd << quote_arg(exe.string());
    for (const auto& arg : args) {
        cmd << ' ' << quote_arg(arg);
    }
    return std::system(cmd.str().c_str());
#endif
}

const FixtureEntry kFixtures[] = {
    {"reference", "cypha_golden", "exe", "M1 infer parity (reference.cypha + native_parity.bin)"},
    {"batch_llr", "batch_llr_golden", "exe", "batch LLR from raw X"},
    {"score_batch", "score_batch_golden", "exe", "score_batch sidecar"},
    {"kernel_llr", "kernel_llr_golden", "exe", "kernel LLR sidecar"},
    {"memory_train", "memory_train_golden", "exe", "one DIFMemory.train step"},
    {"multilabel_dif", "multilabel_dif_golden", "exe", "multilabel DIF"},
    {"merge_from", "merge_from_golden", "exe", "merge_from sidecar"},
    {"similarity_index", "similarity_index_golden", "exe", "similarity index sidecar"},
    {"preprocessor", "preprocessor_golden", "exe", "Preprocessor transform"},
    {"preprocessor_fit", "preprocessor_fit_golden", "exe", "Preprocessor fit (3 variants)"},
    {"csv_ingest", "csv_ingest_golden", "exe", "CSV ingest cases"},
    {"studio_trainer_preprocess_classify_hotpath", "preprocess_train_classify_golden", "exe",
     "Studio trainer preprocess+classify hotpath"},
    {"studio_trainer_preprocess_gh_classify_hotpath", "preprocess_train_classify_golden", "exe",
     "Studio trainer GH preprocess+classify hotpath"},
    {"csv_preprocess_classify_hotpath", "preprocess_train_classify_golden", "exe", "CSV preprocess+classify hotpath"},
    {"nig_adapt", nullptr, "inline", "NIG adapt chi update golden values"},
    {"train_step_vector", "train_step_vector_golden", "exe", "dif_train_step_vector loss"},
    {"dif_regressor_train_step", "dif_regressor_train_step_golden", "exe", "DIF regressor train step"},
    {"regression_mixture", "regression_mixture_golden", "exe", "regression mixture (no fixture dir)"},
    {"regression_m4", "regression_m4_golden", "exe", "M4 regression golden vectors"},
    {"regression_rff", "regression_rff_golden", "exe", "RFF regression sidecar"},
    {"two_stage_pipeline", "regression_two_stage_pipeline_golden", "exe", "two-stage pipeline sidecar"},
    {"two_stage_ridge_fit", "regression_two_stage_ridge_fit_golden", "exe", "two-stage ridge fit sidecar"},
    {"two_stage_e2e_ridge", "regression_two_stage_ridge_fit_golden", "exe", "two-stage e2e ridge sidecar"},
    {"quantile_dif_train", "quantile_dif_train_golden", "exe", "quantile DIF train replay"},
    {"dif_train_replay", "quantile_dif_train_golden", "exe", "DIF train replay fixture"},
    {"studio_trainer_classify_hotpath", "quantile_dif_train_golden", "exe", "Studio trainer classify hotpath"},
    {"studio_trainer_gh_classify_hotpath", "quantile_dif_train_golden", "exe", "Studio trainer GH classify hotpath"},
    {"mke_train_step", "mke_train_step_golden", "exe", "MKE train step"},
    {"mke_train_extended", "mke_train_step_golden", "exe", "MKE train extended"},
    {"generation", "generation_golden", "exe", "generation sidecar"},
    {"gh_infer_deliberation", "gh_infer_deliberation_golden", "exe", "GH infer deliberation"},
    {"embed_table", "embed_table_golden", "exe", "embed table sidecar"},
    {"retrieval", "retrieval_golden", "exe", "retrieval sidecar"},
    {"cyphalm_char_lstm", "cyphalm_char_lstm_golden", "exe", "CyphaLM char-LSTM sidecar"},
    {"cyphalm_model", "cyphalm_model_golden", "exe", "CyphaLM model smoke (no fixture dir)"},
    {"cyphalm_checkpoint_char_lstm", "cyphalm_checkpoint_golden", "exe", "CyphaLM checkpoint char_lstm sidecar"},
    {"cyphalm_hebbian", "cyphalm_hebbian_golden", "exe", "CyphaLM Hebbian (no fixture dir)"},
    {"som", "som_golden", "exe", "SOM/GNG parity (no fixture dir)"},
    {"cyphalm_ssm", "cyphalm_ssm_golden", "exe", "CyphaLM SSM smoke (no fixture dir)"},
    {"cyphalm_ssm_fixture", "cyphalm_golden", "exe", "CyphaLM SSM fixture sidecar"},
    {"cyphalm_golden_suite", "cyphalm_golden", "exe", "CyphaLM parity suite (no fixture dir)"},
};

const FixtureEntry* find_fixture(const std::string& name) {
    for (const FixtureEntry& f : kFixtures) {
        if (name == f.name) {
            return &f;
        }
    }
    return nullptr;
}

fs::path sidecar_path(const fs::path& dir) { return dir / "sidecar.json"; }

std::vector<std::string> build_exe_args(const std::string& fixture, const fs::path& dir) {
    const fs::path abs = fs::absolute(dir);
    if (fixture == "reference") {
        const fs::path root = abs.parent_path();
        return {root.string() + "/reference.cypha", root.string() + "/native_parity.bin"};
    }
    if (fixture == "preprocessor_fit") {
        const fs::path root = abs.parent_path();
        return {abs.string(), (root / "preprocessor_fit_no_scale").string(), (root / "preprocessor_fit_rff").string()};
    }
    if (fixture == "regression_mixture" || fixture == "cyphalm_model" || fixture == "cyphalm_hebbian" ||
        fixture == "som" || fixture == "cyphalm_ssm" || fixture == "cyphalm_golden_suite" || fixture == "nig_adapt") {
        return {};
    }
    if (fixture == "cyphalm_checkpoint_char_lstm") {
        return {sidecar_path(abs).string()};
    }
    if (fs::is_regular_file(sidecar_path(abs))) {
        return {sidecar_path(abs).string()};
    }
    return {abs.string()};
}

int run_inline_nig_adapt() {
    struct Row {
        double chi;
        double psi;
        double innov;
        double R;
        double alpha;
        double chi_new_expected;
    };
    const Row rows[] = {
        {1.0, 1.0, 0.5, 0.1, 0.98, 2.0193943711834006},
        {1.0, 1.0, 2.0, 0.05, 0.98, 5.834957899296454},
        {2.5, 1.0, 0.01, 0.2, 0.98, 1.2218599849092555},
    };
    for (const Row& r : rows) {
        const auto out = cypha::nig_adapt_session_chi(r.chi, r.psi, r.innov, r.R, r.alpha);
        if (std::abs(out.second - r.psi) > 1e-12) {
            std::cerr << "nig_adapt inline: psi drift\n";
            return 2;
        }
        if (std::abs(out.first - r.chi_new_expected) > 1e-9) {
            std::cerr << "nig_adapt inline: chi_new mismatch\n";
            return 1;
        }
    }
    return 0;
}

int dispatch_fixture(const FixtureEntry& entry, const fs::path& dir, const fs::path& exe_dir) {
    if (std::string(entry.kind) == "inline") {
        if (std::string(entry.name) == "nig_adapt") {
            return run_inline_nig_adapt();
        }
        std::cerr << "cypha_golden_run: unsupported inline fixture: " << entry.name << '\n';
        return 2;
    }

    if (entry.exe == nullptr) {
        std::cerr << "cypha_golden_run: missing exe for fixture: " << entry.name << '\n';
        return 2;
    }

    const fs::path exe = exe_dir / exe_name(entry.exe);
    if (!fs::is_regular_file(exe)) {
        std::cerr << "cypha_golden_run: missing executable: " << exe.string() << '\n';
        return 127;
    }

    const std::vector<std::string> args = build_exe_args(entry.name, dir);
    return run_process(exe, args);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.list_only) {
            for (const FixtureEntry& f : kFixtures) {
                std::cout << f.name << '\t' << f.kind << '\t' << (f.exe ? f.exe : "inline") << '\t' << f.description
                          << '\n';
            }
            return 0;
        }

        if (args.fixture.empty() || args.dir.empty()) {
            usage();
            return 2;
        }

        const FixtureEntry* entry = find_fixture(args.fixture);
        if (entry == nullptr) {
            std::cerr << "unknown fixture: " << args.fixture << " (try --list)\n";
            return 2;
        }

        const fs::path exe_dir = resolve_exe_dir(args, argv);
        const int code = dispatch_fixture(*entry, fs::absolute(args.dir), exe_dir);
        if (code != 0) {
            std::cerr << "cypha_golden_run: " << args.fixture << " failed (exit " << code << ")\n";
        }
        return code;
    } catch (const std::exception& ex) {
        std::cerr << "cypha_golden_run error: " << ex.what() << '\n';
        return 2;
    }
}
