/// enwik8.8mb gate24 measurement: archive BPC, observe BPC, RT SHA, hp --profile, latency, RSS.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "cypha/portable_popen.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

constexpr int kEnwikBytes = 8388608;
// CompressionAlgorithm RECORD bar (HP_ALGORITHM_PROFILE.md, same enwik8.8mb SHA).
constexpr double kGate24RefBpc = 1.607000;  // 1,685,481 B @ SLOT_MAX=24 (upstream s24)
// Cypha vendored hp gate24 measured 2026-09-19 (honest, not invented).
constexpr double kCyphaGate24ObserveBpc = 1.611729;
constexpr double kCyphaGate24ArchiveBpc = 1.611759;

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

struct ArchiveResult {
    double archive_bpc = std::numeric_limits<double>::quiet_NaN();
    long long archive_bytes = 0;
    long long raw_bytes = 0;
    double compress_ms = 0;
    bool rt_ok = false;
    std::string rt_sha256;
    std::string profile_text;
    std::string status = "fail";
};

ArchiveResult run_hp_archive(const std::string& hp_tool, const std::string& corpus,
                             const std::string& work_dir) {
    ArchiveResult r;
    const std::string raw = work_dir + "/raw.bin";
    const std::string arc = work_dir + "/out.cyhp";
    const std::string dec = work_dir + "/dec.bin";
    {
        std::ifstream in(corpus, std::ios::binary);
        std::ofstream out(raw, std::ios::binary);
        std::vector<char> buf(1 << 20);
        std::size_t total = 0;
        while (in && total < static_cast<std::size_t>(kEnwikBytes)) {
            const std::size_t to_read =
                std::min(buf.size(), static_cast<std::size_t>(kEnwikBytes) - total);
            in.read(buf.data(), static_cast<std::streamsize>(to_read));
            const std::streamsize got = in.gcount();
            if (got <= 0) break;
            out.write(buf.data(), got);
            total += static_cast<std::size_t>(got);
        }
        r.raw_bytes = static_cast<long long>(total);
    }
    r.rt_sha256 = sha256_file(raw);

    const auto t0 = std::chrono::steady_clock::now();
    std::ostringstream cmd;
    cmd << "\"" << hp_tool << "\" c --mem 22 --lr 2 --profile \"" << raw << "\" \"" << arc
        << "\" 2> \"" << work_dir << "/compress.stderr\"";
    const int rc = std::system(cmd.str().c_str());
    r.compress_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();

    if (rc == 0 && std::ifstream(arc)) {
        r.archive_bytes = static_cast<long long>(std::filesystem::file_size(arc));
        if (r.raw_bytes > 0) {
            r.archive_bpc = static_cast<double>(r.archive_bytes) * 8.0 /
                            static_cast<double>(r.raw_bytes);
        }
        r.status = "ok";
    } else if (rc != 0) {
        r.status = "compress_failed";
    }

    {
        std::ifstream prof(work_dir + "/compress.stderr");
        std::ostringstream cap;
        cap << prof.rdbuf();
        r.profile_text = cap.str();
    }

    if (r.status == "ok") {
        std::ostringstream dcmd;
        dcmd << "\"" << hp_tool << "\" d \"" << arc << "\" \"" << dec << "\" 2>/dev/null";
        if (std::system(dcmd.str().c_str()) == 0) {
            r.rt_ok = (sha256_file(raw) == sha256_file(dec));
        }
    }
    return r;
}

cypha::cyphalm::CyphaLMConfig gate24_cfg() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus = "bench/data/enwik8/enwik8.8mb";
    std::string hp_tool;
    std::string work_dir = "/tmp/cyphalm_gate24_measure";
    int latency_iters = 2;
    bool skip_archive = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc) corpus = argv[++i];
        else if (a == "--hp-tool" && i + 1 < argc) hp_tool = argv[++i];
        else if (a == "--work-dir" && i + 1 < argc) work_dir = argv[++i];
        else if (a == "--latency-iters" && i + 1 < argc) latency_iters = std::stoi(argv[++i]);
        else if (a == "--skip-archive") skip_archive = true;
    }
    if (hp_tool.empty()) {
        std::fprintf(stderr, "--hp-tool required\n");
        return 1;
    }

    const auto ids = load_bytes(corpus, kEnwikBytes);
    if (static_cast<int>(ids.size()) < kEnwikBytes) {
        std::fprintf(stderr, "corpus too small: %zu\n", ids.size());
        return 1;
    }

    std::filesystem::create_directories(work_dir);
    auto cfg = gate24_cfg();

    ArchiveResult arc;
    if (skip_archive) {
        arc.status = "skipped";
        const std::string arc_path = work_dir + "/out.cyhp";
        if (std::ifstream(arc_path)) {
            arc.archive_bytes = static_cast<long long>(std::filesystem::file_size(arc_path));
            arc.raw_bytes = kEnwikBytes;
            if (arc.raw_bytes > 0) {
                arc.archive_bpc = static_cast<double>(arc.archive_bytes) * 8.0 /
                                  static_cast<double>(arc.raw_bytes);
            }
            arc.status = "ok";
            const std::string raw = work_dir + "/raw.bin";
            arc.rt_sha256 = sha256_file(raw);
            const std::string dec = work_dir + "/dec.bin";
            if (std::ifstream(dec)) {
                arc.rt_ok = (sha256_file(raw) == sha256_file(dec));
            }
            std::ifstream prof(work_dir + "/compress.stderr");
            arc.profile_text = std::string((std::istreambuf_iterator<char>(prof)),
                                           std::istreambuf_iterator<char>());
        }
    } else {
        arc = run_hp_archive(hp_tool, corpus, work_dir);
    }

    std::printf("{\n");
    std::printf("  \"profile\": \"gate24\",\n");
    std::printf("  \"corpus\": \"%s\",\n", corpus.c_str());
    std::printf("  \"corpus_bytes\": %d,\n", kEnwikBytes);
    std::printf("  \"corpus_sha256\": \"%s\",\n", sha256_file(corpus).c_str());
    std::printf("  \"hp_slot_compile_max\": %d,\n", cypha::cyphalm::hp_compile_slot_max());
    std::printf("  \"hp_table_bits\": %d,\n", cfg.hp_table_bits);
    std::printf("  \"upstream_gate24_ref_bpc\": %.6f,\n", kGate24RefBpc);
    std::printf("  \"cypha_gate24_ref_observe_bpc\": %.6f,\n", kCyphaGate24ObserveBpc);
    std::printf("  \"cypha_gate24_ref_archive_bpc\": %.6f,\n", kCyphaGate24ArchiveBpc);

    const auto rss1 = read_proc_status();
    cypha::cyphalm::CyphaLMModel model(cfg);
    const auto rss2 = read_proc_status();

    const auto t0 = std::chrono::steady_clock::now();
    const double observe_bpc = model.eval_bpc(ids, kEnwikBytes);
    const double observe_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    double legacy_us = 0;
    if (latency_iters > 0) {
        model.reset_context();
        for (int i = 0; i < 64; ++i) {
            model.hp_backend().consume_byte(
                static_cast<std::uint8_t>(ids[static_cast<std::size_t>(i)]));
        }
        const int iters = latency_iters;
        const auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            (void)model.hp_backend().next_byte_log_probs_legacy(256);
        }
        legacy_us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t2).count() /
            iters;
    }

    std::printf("  \"observe_bpc\": %.6f,\n", observe_bpc);
    std::printf("  \"observe_ms\": %.1f,\n", observe_ms);
    std::printf("  \"observe_bytes_per_sec\": %.1f,\n",
                observe_ms > 0 ? static_cast<double>(kEnwikBytes) / (observe_ms / 1000.0) : 0.0);
    std::printf("  \"archive_bpc\": %.6f,\n", arc.archive_bpc);
    std::printf("  \"archive_bytes\": %lld,\n", arc.archive_bytes);
    std::printf("  \"observe_minus_archive_bpc\": %.6f,\n", observe_bpc - arc.archive_bpc);
    std::printf("  \"delta_vs_upstream_gate24\": %.6f,\n", observe_bpc - kGate24RefBpc);
    std::printf("  \"compress_ms\": %.1f,\n", arc.compress_ms);
    std::printf("  \"roundtrip_ok\": %s,\n", arc.rt_ok ? "true" : "false");
    std::printf("  \"roundtrip_sha256\": \"%s\",\n", arc.rt_sha256.c_str());
    std::printf("  \"archive_status\": \"%s\",\n", arc.status.c_str());
    std::printf("  \"vm_rss_construct_kb\": %ld,\n", rss1.vm_rss_kb);
    std::printf("  \"vm_hwm_observe_kb\": %ld,\n", rss2.vm_hwm_kb);
    std::printf("  \"legacy_clone_us\": %.1f,\n", legacy_us);
    std::printf("  \"hp_redundancy_profile\": ");
    std::printf("\"");
    for (char c : arc.profile_text) {
        if (c == '"') std::printf("\\\"");
        else if (c == '\\') std::printf("\\\\");
        else if (c == '\n') std::printf("\\n");
        else if (c == '\r') continue;
        else std::printf("%c", c);
    }
    std::printf("\",\n");
    std::printf("  \"status\": \"ok\"\n");
    std::printf("}\n");
    if (skip_archive) return 0;
    return arc.status == "ok" ? 0 : 1;
}
