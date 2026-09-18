// hp/tools/bench.cpp — in-process encode benchmark (predictor + coder).
//
// Usage: bench <file> [--mem N]
// Prints: input_bytes coded_bytes peak_rss_kb encode_ms

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hp/coder.hpp"
#include "hp/predictor.hpp"

namespace {

std::vector<std::uint8_t> read_all(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(n));
    if (n > 0 && std::fread(out.data(), 1, static_cast<std::size_t>(n), f) !=
                     static_cast<std::size_t>(n)) {
        std::fclose(f);
        std::fprintf(stderr, "read failed\n");
        std::exit(1);
    }
    std::fclose(f);
    return out;
}

long peak_rss_kb() {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long hwm = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0)
            std::sscanf(line + 6, "%ld", &hwm);
    }
    std::fclose(f);
    return hwm;
}

std::size_t encode_body(const std::vector<std::uint8_t>& buf, hp::Config cfg) {
    hp::Predictor pred(cfg);
    char* mem = nullptr;
    std::size_t mem_len = 0;
    std::FILE* f = open_memstream(&mem, &mem_len);
    if (!f) { std::perror("open_memstream"); std::exit(1); }
    hp::Encoder enc(f);
    for (std::size_t k = 0; k < buf.size(); ++k) {
        const int c = buf[k];
        for (int i = 7; i >= 0; --i) {
            const int p = pred.predict();
            const int bit = (c >> i) & 1;
            enc.encode(bit, hp::clamp_int(p << 4, 1, 65535));
            pred.update(bit);
        }
    }
    enc.flush();
    std::fclose(f);
    std::free(mem);
    return mem_len;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bench <file> [--mem N]\n");
        return 2;
    }
    hp::Config cfg;
    int i = 1;
    for (; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mem") && i + 1 < argc) {
            cfg.table_bits = std::atoi(argv[++i]);
            cfg.normalize();
        } else break;
    }
    if (i >= argc) {
        std::fprintf(stderr, "usage: bench <file> [--mem N]\n");
        return 2;
    }
    cfg.normalize();
    const auto buf = read_all(argv[i]);

    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t nbytes = encode_body(buf, cfg);
    const auto t1 = std::chrono::steady_clock::now();
    const long rss = peak_rss_kb();
    const auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("input_bytes=%zu coded_bytes=%zu peak_rss_kb=%ld encode_ms=%.1f\n",
                buf.size(), nbytes, rss, ms);
    return 0;
}
