// Streaming trainer, version 2 (docs/TRAIN_1TB_RUNBOOK.md): a second run beside the live trainer
// native/tools/trie_stream_train.cpp.
//
// The same model, the same corpus files and the same per-file checkpoint and resume as the live trainer, through
// cypha/cyphalm/stream_trie.hpp, with these changes:
//   - counts never wrap: a count about to pass 2^32 - 1 halves its table (the node's followers, the order-1 row or
//     the unigram), so ratios stay right on a 1 TB corpus (the live trainer's 32-bit counts wrap);
//   - checkpoints are "TRI2" (the live layout plus a trailer: the cap and the number of halvings);
//   - output goes to ROOT/trie/model2 by default, or --out DIR (keep it inside "/trie/", so no trainer reads another's
//     files as text); scripts/train_model.sh numbers the models: model 1 is the live run, model N is <corpus>/trie/model<N>;
//   - --once stops when every finished file is trained and writes done.txt (bytes, nodes, followers, files, time);
//   - --snapshots 8M,64M,1G keeps a copy of the checkpoint at the first file boundary past each size (scaling curve);
//   - --threads N trains each file on N threads (stream_trie.hpp train_file_parallel: one byte range a thread, the
//     shards merged pairwise in parallel, then into the model). On .jsonl files the model equals the one-thread model
//     while no count reaches the cap; a .txt file differs only by the history reset at each cut. Default 1.
//
// Usage: trie_stream_train_v2 ROOT [--out DIR] [--once] [--snapshots LIST] [--cap N] [--threads N]

#include "cypha/cyphalm/stream_trie.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace st = cypha::cyphalm::stream;

namespace {

std::vector<std::string> load_lines(const fs::path& list) {
    std::vector<std::string> out;
    std::ifstream in(list);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

std::uint64_t parse_size(const std::string& s) {
    if (s.empty()) return 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    std::uint64_t mul = 1;
    if (end && *end) {
        const char u = static_cast<char>(*end | 0x20);
        mul = u == 'k' ? (1ull << 10) : u == 'm' ? (1ull << 20) : u == 'g' ? (1ull << 30) : u == 't' ? (1ull << 40) : 1;
    }
    return static_cast<std::uint64_t>(v * static_cast<double>(mul));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: trie_stream_train_v2 ROOT [--out DIR] [--once] [--snapshots 8M,64M,1G] [--cap N] "
                     "[--threads N]\n";
        return 2;
    }
    const fs::path root = argv[1];
    fs::path dir = root / "trie" / "model2";
    bool once = false;
    unsigned threads = 1;
    std::vector<std::pair<std::uint64_t, std::string>> snaps;
    std::uint32_t cap = 0xffffffffu;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) dir = argv[++i];
        else if (a == "--once") once = true;
        else if (a == "--threads" && i + 1 < argc) threads = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--cap" && i + 1 < argc) cap = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--snapshots" && i + 1 < argc) {
            std::string list = argv[++i], item;
            for (std::size_t p = 0; p <= list.size(); ++p) {
                if (p == list.size() || list[p] == ',') {
                    if (!item.empty()) snaps.push_back({parse_size(item), item});
                    item.clear();
                } else {
                    item.push_back(list[p]);
                }
            }
            std::sort(snaps.begin(), snaps.end());
        } else {
            std::cerr << "unknown option " << a << "\n";
            return 2;
        }
    }
    if (cap < 2 || threads < 1 || threads > 256) {
        std::cerr << "--cap must be at least 2 and --threads 1 to 256\n";
        return 2;
    }
    fs::create_directories(dir);
    const fs::path ckpt = dir / "checkpoint.bin";
    const fs::path trained_list = dir / "trained.txt";
    const fs::path snap_list = dir / "snapshots.tsv";

    st::Model model;
    model.init();
    model.cap = cap;
    st::Header h;
    std::string err;
    if (fs::exists(ckpt)) {
        if (!st::load(ckpt, model, h, err) || !h.tri2) {
            std::cerr << "cannot resume " << ckpt << ": " << (err.empty() ? "not a TRI2 checkpoint" : err) << "\n";
            return 1;
        }
        if (model.cap != cap) std::cerr << "note: resuming with the checkpoint's cap " << model.cap << "\n";
        std::cerr << "resumed bytes " << model.bytes_seen << " nodes " << model.nodes.size() << "\n";
    } else {
        std::cerr << "new checkpoint " << ckpt << "\n";
    }
    auto done = load_lines(trained_list);
    std::vector<std::string> taken;
    for (const auto& l : load_lines(snap_list)) taken.push_back(l.substr(0, l.find('\t')));
    const auto t0 = std::chrono::steady_clock::now();

    while (true) {
        std::vector<fs::path> fresh;
        std::error_code ec;
        if (fs::exists(root, ec)) {
            for (auto it = fs::recursive_directory_iterator(root, ec); it != fs::recursive_directory_iterator();
                 it.increment(ec)) {
                if (ec) break;
                if (!it->is_regular_file()) continue;
                const fs::path p = it->path();
                if (!st::trainable(p)) continue;
                const std::string key = p.generic_string();
                if (std::find(done.begin(), done.end(), key) != done.end()) continue;
                fresh.push_back(p);
            }
        }
        std::sort(fresh.begin(), fresh.end());
        if (fresh.empty()) {
            if (once) break;
            std::cerr << "waiting for a finished file\n";
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
        for (const auto& p : fresh) {
            std::cerr << "file " << p.generic_string() << "\n";
            std::uint64_t n = 0;
            const auto tf = std::chrono::steady_clock::now();
            if (!st::train_file_parallel(model, p, threads, n)) {
                std::cerr << "skip unreadable " << p << "\n";
                continue;
            }
            const double fs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count();
            std::cerr << "trained " << p.filename().string() << " bytes " << n << " nodes " << model.nodes.size()
                      << " seen " << model.bytes_seen << " seconds " << fs_ << "\n";
            if (!st::save(ckpt, model, true)) {
                std::cerr << "checkpoint failed\n";
                return 1;
            }
            {
                std::ofstream out(trained_list, std::ios::app);
                out << p.generic_string() << "\n";
            }
            done.push_back(p.generic_string());
            for (const auto& [at, label] : snaps) {
                if (model.bytes_seen < at || std::find(taken.begin(), taken.end(), label) != taken.end()) continue;
                const fs::path snap = dir / ("checkpoint_" + label + ".bin");
                fs::copy_file(ckpt, snap, fs::copy_options::overwrite_existing, ec);
                if (ec) {
                    std::cerr << "snapshot " << label << " failed: " << ec.message() << "\n";
                    continue;
                }
                std::ofstream out(snap_list, std::ios::app);
                out << label << "\t" << model.bytes_seen << "\t" << done.size() << "\t" << snap.filename().string()
                    << "\n";
                taken.push_back(label);
                std::cerr << "snapshot " << label << " at bytes " << model.bytes_seen << "\n";
            }
            std::cerr << "checkpoint bytes " << model.bytes_seen << "\n";
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ofstream out(dir / "done.txt");
    out << "finished " << std::time(nullptr) << "\nbytes " << model.bytes_seen << "\nnodes " << model.nodes.size()
        << "\nfollowers " << model.fols.size() << "\nmatch_keys " << model.mfilled << "\nfiles " << done.size()
        << "\nhalvings uni " << model.halve_uni << " o1 " << model.halve_o1 << " node " << model.halve_node
        << "\nseconds_this_session " << secs << "\n";
    std::cerr << "done: every finished file is trained (" << done.size() << " files, " << model.bytes_seen
              << " bytes)\n";
    return 0;
}
