/// Validate bench/BASELINE_LOCK.json structure and d17 hybrid baseline pin (reference tolerance).
/// Usage: baseline_lock_validate [--lock-file PATH]
/// Exit 0 when valid, 1 on failure.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

constexpr double kD17PinBpc = 2.664;
constexpr double kD17PinTolerance = 0.05;
constexpr double kD17ProductionPinTolerance = 0.05;
constexpr int kProductionNTrainMin = 300000;

[[noreturn]] void fail(const std::string& msg) {
    std::fprintf(stderr, "baseline_lock_validate: FAIL - %s\n", msg.c_str());
    std::exit(1);
}

void require_key(const Json& obj, const char* key, const char* ctx) {
    if (!obj.contains(key) || obj[key].is_null()) {
        fail(std::string(ctx) + " missing '" + key + "'");
    }
}

void validate_result_section(const Json& section, const char* name, const char* expected_profile,
                             const char* expected_mode) {
    if (!section.is_object()) {
        fail(std::string(name) + " is not an object");
    }
    require_key(section, "status", name);
    require_key(section, "bpc", name);
    require_key(section, "run_at", name);
    require_key(section, "profile", name);
    require_key(section, "mode", name);
    require_key(section, "n_train", name);
    require_key(section, "n_eval", name);
    require_key(section, "runner", name);
    require_key(section, "env", name);

    const std::string status = section["status"].get<std::string>();
    if (status == "pending") {
        fail(std::string(name) + " status is pending");
    }
    if (status != "fast_smoke" && status != "medium_smoke" && status != "production" &&
        status != "completed" && status != "historical") {
        fail(std::string(name) + " status '" + status + "' is not recognized");
    }
    if (!section["bpc"].is_number()) {
        fail(std::string(name) + " bpc must be numeric");
    }
    if (!section["run_at"].is_string() || section["run_at"].get<std::string>().empty()) {
        fail(std::string(name) + " run_at must be a non-empty string");
    }
    if (section["profile"].get<std::string>() != expected_profile) {
        fail(std::string(name) + " profile mismatch");
    }
    if (section["mode"].get<std::string>() != expected_mode) {
        fail(std::string(name) + " mode mismatch");
    }
    if (!section["env"].is_object()) {
        fail(std::string(name) + " env must be an object");
    }
}

Json load_lock(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        fail("cannot read lock file: " + path.string());
    }
    Json j;
    try {
        in >> j;
    } catch (const std::exception& ex) {
        fail(std::string("invalid JSON: ") + ex.what());
    }
    return j;
}

fs::path default_lock_path() {
    const fs::path here = fs::current_path();
    const fs::path candidate = here / ".." / "bench" / "BASELINE_LOCK.json";
    if (fs::is_regular_file(candidate)) {
        return fs::absolute(candidate);
    }
    return fs::absolute(here / "bench" / "BASELINE_LOCK.json");
}

fs::path parse_lock_path(int argc, char** argv, bool* production) {
    *production = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::fprintf(stderr, "usage: baseline_lock_validate [--lock-file PATH] [--production]\n");
            std::exit(0);
        }
        if (arg == "--production") {
            *production = true;
            continue;
        }
        if (arg == "--lock-file") {
            if (i + 1 >= argc) {
                fail("missing value for --lock-file");
            }
            return fs::absolute(argv[++i]);
        }
        fail("unknown arg: " + arg);
    }
    return default_lock_path();
}

void validate_production_tier(const Json& lock) {
    if (!lock.contains("overnight_results") || !lock["overnight_results"].is_object()) {
        fail("overnight_results is not an object (-Production)");
    }
    const Json& overnight = lock["overnight_results"];
    require_key(overnight, "n_train", "overnight_results");
    if (!overnight["n_train"].is_number_integer()) {
        fail("overnight_results n_train must be an integer (-Production)");
    }
    const int n_train = overnight["n_train"].get<int>();
    if (n_train < kProductionNTrainMin) {
        std::printf("  (-Production: n_train=%d < %d, pending_production)\n", n_train,
                    kProductionNTrainMin);
        return;
    }
    require_key(overnight, "status", "overnight_results");
    const std::string status = overnight["status"].get<std::string>();
    if (status == "historical") {
        std::printf("  (-Production: n_train=%d status=historical, archived pin)\n", n_train);
        return;
    }
    if (status != "production" && status != "completed") {
        fail("overnight_results status '" + status +
             "' invalid for production tier (n_train=" + std::to_string(n_train) +
             "; expected production or completed)");
    }
    require_key(overnight, "bpc", "overnight_results");
    if (!overnight["bpc"].is_number()) {
        fail("overnight_results bpc must be numeric (-Production)");
    }
    const double bpc = overnight["bpc"].get<double>();
    const double prod_delta = std::abs(bpc - kD17PinBpc);
    if (prod_delta > kD17ProductionPinTolerance) {
        fail("overnight_results bpc out of production pin tolerance (~" +
             std::to_string(kD17PinBpc) + ", delta " + std::to_string(prod_delta) + ", max " +
             std::to_string(kD17ProductionPinTolerance) + ")");
    }
    std::printf("  (-Production: n_train=%d status=%s bpc pin OK)\n", n_train, status.c_str());
}

void validate_lock(const Json& lock, bool production) {
    if (!lock.contains("schema_version") || !lock["schema_version"].is_number_integer() ||
        lock["schema_version"].get<int>() != 1) {
        fail("schema_version must be 1");
    }

    require_key(lock, "d17_hybrid_baseline", "lock");
    const Json& d17 = lock["d17_hybrid_baseline"];
    if (!d17.is_object()) {
        fail("d17_hybrid_baseline is not an object");
    }
    for (const char* key : {"bpc", "profile", "mode", "n_train", "n_eval"}) {
        require_key(d17, key, "d17_hybrid_baseline");
    }
    if (d17["profile"].get<std::string>() != "d17") {
        fail("d17_hybrid_baseline profile must be d17");
    }
    if (d17["mode"].get<std::string>() != "hybrid") {
        fail("d17_hybrid_baseline mode must be hybrid");
    }
    if (!d17["bpc"].is_number()) {
        fail("d17_hybrid_baseline bpc must be numeric");
    }
    const double pin_delta = std::abs(d17["bpc"].get<double>() - kD17PinBpc);
    if (pin_delta > kD17PinTolerance) {
        fail("d17_hybrid_baseline bpc pin out of reference tolerance (~" +
             std::to_string(kD17PinBpc) + ", delta " + std::to_string(pin_delta) + ")");
    }

    require_key(lock, "overnight_results", "lock");
    require_key(lock, "rpsm_results", "lock");
    validate_result_section(lock["overnight_results"], "overnight_results", "d17", "hybrid");
    validate_result_section(lock["rpsm_results"], "rpsm_results", "d21", "rpsm");

    if (lock.contains("cell_sweep_results") && !lock["cell_sweep_results"].is_null()) {
        validate_result_section(lock["cell_sweep_results"], "cell_sweep_results", "d17", "cell-sweep");
    }

    if (production) {
        validate_production_tier(lock);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool production = false;
        const fs::path lock_path = parse_lock_path(argc, argv, &production);
        const Json lock = load_lock(lock_path);
        validate_lock(lock, production);
        std::printf("baseline_lock_validate: OK %s\n", lock_path.string().c_str());
        return 0;
    } catch (const std::exception& ex) {
        fail(ex.what());
    }
}
