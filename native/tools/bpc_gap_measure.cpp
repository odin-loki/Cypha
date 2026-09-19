/// Quick BPC gap measurement (observe + hp archive + optional clone eval).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

constexpr double kLog2 = 0.6931471805599453;

std::vector<int> load_bytes(const std::string& path, int max_n) {
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

double run_hp_archive_bpc(const std::string& hp_tool, const std::string& raw_path,
                          const std::string& arc_path, int mem, int lr) {
    std::ostringstream cmd;
    cmd << "\"" << hp_tool << "\" c --mem " << mem << " --lr " << lr << " \"" << raw_path
        << "\" \"" << arc_path << "\" 2>&1";
    std::FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) return std::numeric_limits<double>::quiet_NaN();
    std::string cap;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) cap += buf;
    pclose(pipe);
    long long in_b = 0;
    long long arc_b = 0;
    long long whole = 0, frac = 0;
    std::sscanf(cap.c_str(), "\rin %lld B  out %ld B", &in_b, reinterpret_cast<long*>(&arc_b));
    const char* bpc_pos = std::strstr(cap.c_str(), "bpc ");
    if (bpc_pos && std::sscanf(bpc_pos, "bpc %lld.%lld", &whole, &frac) >= 2) {
        return static_cast<double>(whole) + static_cast<double>(frac) / 1000.0;
    }
    if (in_b > 0 && arc_b > 0) {
        return static_cast<double>(arc_b) * 8.0 / static_cast<double>(in_b);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus = "bench/data/enwik8/enwik8.8mb";
    int nbytes = 65536;
    int clone_n = 0;
    std::string hp_tool = "./native/build/hp_gate24";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc) corpus = argv[++i];
        else if (a == "--bytes" && i + 1 < argc) nbytes = std::stoi(argv[++i]);
        else if (a == "--clone-n" && i + 1 < argc) clone_n = std::stoi(argv[++i]);
        else if (a == "--hp-tool" && i + 1 < argc) hp_tool = argv[++i];
    }

    const auto ids = load_bytes(corpus, nbytes);
    if (ids.empty()) {
        std::fprintf(stderr, "empty corpus\n");
        return 1;
    }
    const int n = std::min(nbytes, static_cast<int>(ids.size()));

    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cypha::cyphalm::CyphaLMModel model(cfg);

    const auto t0 = std::chrono::steady_clock::now();
    const double observe_bpc = model.eval_bpc_compress_equivalent(ids, n);
    const double observe_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const std::string tmp = "/tmp/bpc_gap_measure.bin";
    {
        std::ofstream out(tmp, std::ios::binary);
        for (int i = 0; i < n; ++i) out.put(static_cast<char>(ids[static_cast<std::size_t>(i)]));
    }
    const std::string arc = tmp + ".cyhp";
    const auto t1 = std::chrono::steady_clock::now();
    const double archive_bpc =
        run_hp_archive_bpc(hp_tool, tmp, arc, cfg.hp_table_bits, cfg.hp_mixer_lr);
    const double compress_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    std::remove(arc.c_str());

    double clone_bpc = std::numeric_limits<double>::quiet_NaN();
    double clone_ms = 0;
    double bit_tree_us = std::numeric_limits<double>::quiet_NaN();
    double legacy_full_vocab_us = std::numeric_limits<double>::quiet_NaN();
    double predict_next_us = std::numeric_limits<double>::quiet_NaN();
    if (clone_n > 0) {
        model.reset_context();
        for (int i = 0; i < std::min(64, n); ++i) {
            model.hp_backend().consume_byte(
                static_cast<std::uint8_t>(ids[static_cast<std::size_t>(i)]));
        }
        const int iters = std::max(1, std::min(clone_n, 3));
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                (void)model.hp_backend().next_byte_log_probs_bit_tree(256);
            }
            bit_tree_us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                    .count() /
                iters;
        }
        {
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                (void)model.hp_backend().next_byte_log_probs_legacy(256);
            }
            legacy_full_vocab_us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1)
                    .count() /
                iters;
        }
        {
            const auto t2 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                (void)model.predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i % n)]));
                model.reset_context();
                for (int j = 0; j < std::min(64, n); ++j) {
                    model.hp_backend().consume_byte(
                        static_cast<std::uint8_t>(ids[static_cast<std::size_t>(j)]));
                }
            }
            predict_next_us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t2)
                    .count() /
                iters;
        }
        const auto t3 = std::chrono::steady_clock::now();
        clone_bpc = model.eval_bpc(ids, std::min(clone_n, n));
        clone_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t3).count();
    }

    std::printf("{\n");
    std::printf("  \"corpus\": \"%s\",\n", corpus.c_str());
    std::printf("  \"bytes\": %d,\n", n);
    std::printf("  \"hp_profile\": \"gate24\",\n");
    std::printf("  \"hp_table_bits\": %d,\n", cfg.hp_table_bits);
    std::printf("  \"hp_slot_compile_max\": %d,\n", cypha::cyphalm::hp_compile_slot_max());
    std::printf("  \"observe_bpc\": %.6f,\n", observe_bpc);
    std::printf("  \"observe_ms\": %.1f,\n", observe_ms);
    std::printf("  \"archive_bpc\": %.6f,\n", archive_bpc);
    std::printf("  \"compress_ms\": %.1f,\n", compress_ms);
    std::printf("  \"observe_minus_archive_bpc\": %.6f,\n", observe_bpc - archive_bpc);
    if (clone_n > 0) {
        std::printf("  \"latency_iters\": %d,\n", std::max(1, std::min(clone_n, 3)));
        std::printf("  \"bit_tree_next_byte_log_probs_us\": %.1f,\n", bit_tree_us);
        std::printf("  \"legacy_next_byte_log_probs_us\": %.1f,\n", legacy_full_vocab_us);
        std::printf("  \"predict_next_us\": %.1f,\n", predict_next_us);
        std::printf("  \"observe_eval_n\": %d,\n", std::min(clone_n, n));
        std::printf("  \"observe_eval_bpc\": %.6f,\n", clone_bpc);
        std::printf("  \"observe_eval_ms\": %.1f,\n", clone_ms);
    }
    std::printf("  \"status\": \"ok\"\n");
    std::printf("}\n");
    return 0;
}
