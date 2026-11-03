/// Build a CyphaLM ∞-gram index (suffix array) over a corpus slice.
///
///   cyphalm_infinigram_build --text enwik8 --offset 0 --bytes 95000000 --out enwik8_95m.igr
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cypha/cyphalm/infinigram.hpp"

int main(int argc, char** argv) {
    std::string text, out;
    std::uint64_t offset = 0, bytes = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--text") text = next();
        else if (a == "--offset") offset = std::stoull(next());
        else if (a == "--bytes") bytes = std::stoull(next());
        else if (a == "--out") out = next();
        else {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }
    if (text.empty() || out.empty() || bytes == 0) {
        std::cerr << "need --text FILE --bytes N --out INDEX\n";
        return 2;
    }
    std::ifstream in(text, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> data(bytes);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    data.resize(static_cast<std::size_t>(in.gcount()));
    const auto t0 = std::chrono::steady_clock::now();
    cypha::cyphalm::InfiniGram::build(data.data(), data.size(), out);
    std::printf("built %s: %zu bytes indexed in %.1f s\n", out.c_str(), data.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
