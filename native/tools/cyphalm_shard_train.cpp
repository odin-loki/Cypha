/// Train a CyphaLM hp ensemble: split a corpus slice into N equal shards,
/// pretrain one model per shard on worker threads, save each checkpoint and an
/// ensemble manifest that load_cyphalm_model / cyphalm_generate --load serve as
/// one model (CYPHALM_LM_QUALITY_REPORT.md, "Ensembles of shard models").
///
///   cyphalm_shard_train --train enwik8 --bytes 95000000 --shards 11 \
///       --tier lean --table-bits 20 --threads 4 --out /tmp/ens
///   cyphalm_generate --load /tmp/ens/ensemble.json --prompt "..."
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace fs = std::filesystem;

namespace {

std::vector<std::uint8_t> read_slice(const std::string& path, std::uint64_t offset, std::uint64_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> out(n);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n));
    out.resize(static_cast<std::size_t>(in.gcount()));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string train, out_dir, tier = "lean";
    std::uint64_t offset = 0, bytes = 0;
    int shards = 4, threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int table_bits = 20;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--train") train = next();
            else if (a == "--offset") offset = std::stoull(next());
            else if (a == "--bytes") bytes = std::stoull(next());
            else if (a == "--shards") shards = std::stoi(next());
            else if (a == "--threads") threads = std::stoi(next());
            else if (a == "--tier") tier = next();
            else if (a == "--table-bits") table_bits = std::stoi(next());
            else if (a == "--out") out_dir = next();
            else {
                std::cerr << "unknown arg " << a << "\n";
                return 2;
            }
        }
    } catch (const std::exception& e) {  // missing value, not a number
        std::cerr << "cyphalm_shard_train: bad arguments: " << e.what() << "\n";
        return 2;
    }
    if (train.empty() || out_dir.empty() || bytes == 0 || shards < 1) {
        std::cerr << "need --train FILE --bytes N --out DIR [--shards N]\n";
        return 2;
    }
    if (threads < 1) {  // no workers would train nothing and still write a manifest
        std::cerr << "--threads must be at least 1\n";
        return 2;
    }
    if (bytes < static_cast<std::uint64_t>(shards)) {
        std::cerr << "--bytes must be at least --shards (one byte per shard)\n";
        return 2;
    }
    fs::create_directories(out_dir);
    const std::uint64_t per = bytes / static_cast<std::uint64_t>(shards);

    std::atomic<int> next_shard{0};
    std::atomic<bool> failed{false};
    std::mutex io;
    std::vector<std::string> names(static_cast<std::size_t>(shards));
    std::vector<std::string> errors;
    auto worker = [&] {
        // A failed shard stops new ones; its error is reported after the join
        // and no manifest is written (an exception here would end the process).
        for (int s = next_shard++; s < shards && !failed; s = next_shard++) {
            try {
                const auto t0 = std::chrono::steady_clock::now();
                cypha::cyphalm::CyphaLMConfig cfg;
                cfg.hp_table_bits = table_bits;
                cypha::cyphalm::apply_hp_production_recipe(cfg);
                if (!tier.empty()) cypha::cyphalm::apply_hp_lossy_tier(cfg, tier);
                cypha::cyphalm::CyphaLMModel model(cfg);
                const auto data = read_slice(train, offset + static_cast<std::uint64_t>(s) * per, per);
                if (data.empty()) throw std::runtime_error("no training bytes at this offset");
                const double bits = model.hp_backend().observe_stream_bits(data.data(), data.size());
                const std::string name = "shard_" + std::to_string(s);
                cypha::cyphalm::save_cyphalm_model(model, (fs::path(out_dir) / name).string());
                names[static_cast<std::size_t>(s)] = name + ".json";
                const double secs =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::lock_guard<std::mutex> lock(io);
                std::printf("shard %d: %zu bytes, %.4f bpc, %.0f s\n", s, data.size(),
                            bits / static_cast<double>(std::max<std::size_t>(1, data.size())), secs);
                std::fflush(stdout);
            } catch (const std::exception& e) {
                failed = true;
                std::lock_guard<std::mutex> lock(io);
                errors.push_back("shard " + std::to_string(s) + ": " + e.what());
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < std::min(threads, shards); ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    if (failed) {
        for (const auto& e : errors) std::cerr << "cyphalm_shard_train: " << e << "\n";
        std::cerr << "cyphalm_shard_train: no manifest written\n";
        return 1;
    }

    const std::string manifest = (fs::path(out_dir) / "ensemble.json").string();
    cypha::cyphalm::save_cyphalm_ensemble_manifest(manifest, names);
    std::printf("wrote %s (%d members)\n", manifest.c_str(), shards);
    return 0;
}
