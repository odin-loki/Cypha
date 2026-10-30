// Eight-core streaming trainer. Reads finished corpus files, checkpoints after
// each file, and resumes after a crash. Labelled chat is not observed.
// A file still named *.part is ignored.
// Counts are 64-bit. A uint32 unigram wraps after 4 GiB, and this corpus is
// hundreds of gigabytes. Checkpoint magic is TR64 (TRI1 is the old 32-bit
// layout, TRI2 is trie_stream_train_v2's). cypha/cyphalm/stream_trie.hpp reads
// it; trie_ckpt verifies, scores and packs it (docs/TRAIN_1TB_RUNBOOK.md).
// --once stops when no finished file is left untrained.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kNil = 0xffffffffu;
constexpr char kMagic[4] = {'T', 'R', '6', '4'};

struct Node {
    std::uint32_t parent = kNil;
    std::uint32_t child_head = kNil;
    std::uint32_t sibling = kNil;
    std::uint32_t fol_head = kNil;
    std::uint64_t total = 0;
    std::uint16_t nchild = 0;
    std::uint16_t nfol = 0;
    std::uint8_t byte = 0;
    std::uint8_t depth = 0;
};

struct Fol {
    std::uint64_t count = 0;
    std::uint32_t next = kNil;
    std::uint8_t sym = 0;
};

static_assert(sizeof(Node) == 32, "Node layout changed");
static_assert(sizeof(Fol) == 16, "Fol layout changed");

std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct Model {
    std::uint64_t uni[256]{};
    std::uint64_t uni_n = 0;
    std::uint64_t o1[256][256]{};
    std::uint64_t o1n[256]{};
    std::vector<Node> nodes;
    std::vector<Fol> fols;
    std::uint32_t root_child[256]{};
    std::vector<std::uint32_t> mk, mv;
    std::vector<std::uint8_t> mb;
    std::vector<std::uint16_t> ms;
    std::uint32_t mmask = 0;
    std::uint32_t mfilled = 0;
    std::uint64_t bytes_seen = 0;
    int hist_n = 0;
    std::uint8_t hist[4]{};

    void init() {
        nodes.clear();
        nodes.push_back(Node{});
        fols.clear();
        for (int i = 0; i < 256; ++i) root_child[i] = kNil;
        mk.assign(1u << 20, kNil);
        mv.assign(mk.size(), 0);
        mb.assign(mk.size(), 0);
        ms.assign(mk.size(), 0);
        mmask = static_cast<std::uint32_t>(mk.size() - 1);
        mfilled = 0;
        bytes_seen = 0;
        hist_n = 0;
    }

    void reset_hist() { hist_n = 0; }

    std::uint32_t ensure_child(std::uint32_t node, std::uint8_t b) {
        if (node == 0) {
            if (root_child[b] != kNil) return root_child[b];
            Node n;
            n.parent = 0;
            n.byte = b;
            n.depth = 1;
            const auto id = static_cast<std::uint32_t>(nodes.size());
            nodes.push_back(n);
            root_child[b] = id;
            nodes[0].nchild++;
            return id;
        }
        std::uint32_t prev = kNil;
        std::uint32_t c = nodes[node].child_head;
        while (c != kNil) {
            if (nodes[c].byte == b) {
                if (prev != kNil) {
                    nodes[prev].sibling = nodes[c].sibling;
                    nodes[c].sibling = nodes[node].child_head;
                    nodes[node].child_head = c;
                }
                return c;
            }
            prev = c;
            c = nodes[c].sibling;
        }
        Node n;
        n.parent = node;
        n.byte = b;
        n.depth = static_cast<std::uint8_t>(nodes[node].depth + 1);
        n.sibling = nodes[node].child_head;
        const auto id = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(n);
        nodes[node].child_head = id;
        nodes[node].nchild++;
        (void)prev;
        return id;
    }

    void add_fol_n(std::uint32_t node, std::uint8_t y, std::uint64_t nadd) {
        if (nadd == 0) return;
        std::uint32_t prev = kNil;
        for (std::uint32_t f = nodes[node].fol_head; f != kNil; f = fols[f].next) {
            if (fols[f].sym == y) {
                fols[f].count += nadd;
                nodes[node].total += nadd;
                if (prev != kNil) {
                    fols[prev].next = fols[f].next;
                    fols[f].next = nodes[node].fol_head;
                    nodes[node].fol_head = f;
                }
                return;
            }
            prev = f;
        }
        Fol f;
        f.sym = y;
        f.count = nadd;
        f.next = nodes[node].fol_head;
        const auto id = static_cast<std::uint32_t>(fols.size());
        fols.push_back(f);
        nodes[node].fol_head = id;
        nodes[node].nfol++;
        nodes[node].total += nadd;
    }

    void add_fol(std::uint32_t node, std::uint8_t y) { add_fol_n(node, y, 1); }

    void match_note(std::uint32_t key, std::uint8_t y) {
        if ((static_cast<std::uint64_t>(mfilled) + 1) * 10 >= mk.size() * 7) {
            auto ok = mk, ov = mv;
            auto ob = mb;
            auto os = ms;
            mk.assign(ok.size() * 2, kNil);
            mv.assign(mk.size(), 0);
            mb.assign(mk.size(), 0);
            ms.assign(mk.size(), 0);
            mmask = static_cast<std::uint32_t>(mk.size() - 1);
            mfilled = 0;
            for (std::size_t i = 0; i < ok.size(); ++i) {
                if (ok[i] == kNil) continue;
                std::uint32_t j = mix32(ok[i]) & mmask;
                while (mk[j] != kNil) j = (j + 1u) & mmask;
                mk[j] = ok[i];
                mv[j] = ov[i];
                mb[j] = ob[i];
                ms[j] = os[i];
                ++mfilled;
            }
        }
        std::uint32_t i = mix32(key) & mmask;
        for (;;) {
            if (mk[i] == kNil) {
                mk[i] = key;
                mv[i] = 1;
                mb[i] = y;
                ms[i] = 1;
                ++mfilled;
                return;
            }
            if (mk[i] == key) {
                if (mv[i] < 0xffffffffu) ++mv[i];
                mb[i] = y;
                if (ms[i] < 65535) ++ms[i];
                return;
            }
            i = (i + 1u) & mmask;
        }
    }

    void match_add(std::uint32_t key, std::uint8_t y, std::uint32_t times) {
        if (times == 0) return;
        if ((static_cast<std::uint64_t>(mfilled) + 1) * 10 >= mk.size() * 7) match_note(key, y), times -= 1;
        if (times == 0) return;
        std::uint32_t i = mix32(key) & mmask;
        for (;;) {
            if (mk[i] == kNil) {
                mk[i] = key;
                mv[i] = times;
                mb[i] = y;
                ms[i] = static_cast<std::uint16_t>(std::min<std::uint32_t>(times, 65535));
                ++mfilled;
                return;
            }
            if (mk[i] == key) {
                const auto room = 0xffffffffu - mv[i];
                mv[i] += std::min(room, times);
                mb[i] = y;
                ms[i] = static_cast<std::uint16_t>(std::min<std::uint32_t>(65535u, ms[i] + times));
                return;
            }
            i = (i + 1u) & mmask;
        }
    }

    void observe_byte(std::uint8_t y) {
        ++uni[y];
        ++uni_n;
        ++bytes_seen;
        if (hist_n >= 1) {
            ++o1[hist[0]][y];
            ++o1n[hist[0]];
        }
        const int maxd = std::min(4, hist_n);
        std::uint32_t n = 0;
        for (int d = 1; d <= maxd; ++d) {
            n = ensure_child(n, hist[d - 1]);
            if (d >= 2) add_fol(n, y);
        }
        if (hist_n >= 4) {
            const std::uint32_t key = (std::uint32_t(hist[3]) << 24) | (std::uint32_t(hist[2]) << 16) |
                                      (std::uint32_t(hist[1]) << 8) | std::uint32_t(hist[0]);
            match_note(key, y);
        }
        if (hist_n < 4) {
            for (int i = hist_n; i > 0; --i) hist[i] = hist[i - 1];
            hist[0] = y;
            ++hist_n;
        } else {
            hist[3] = hist[2];
            hist[2] = hist[1];
            hist[1] = hist[0];
            hist[0] = y;
        }
    }
};

bool save_checkpoint(const fs::path& path, const Model& m) {
    const fs::path tmp = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) return false;
    out.write(kMagic, 4);
    const std::uint32_t nn = static_cast<std::uint32_t>(m.nodes.size());
    const std::uint32_t nf = static_cast<std::uint32_t>(m.fols.size());
    out.write(reinterpret_cast<const char*>(&nn), 4);
    out.write(reinterpret_cast<const char*>(&nf), 4);
    out.write(reinterpret_cast<const char*>(&m.bytes_seen), 8);
    out.write(reinterpret_cast<const char*>(m.uni), sizeof m.uni);
    out.write(reinterpret_cast<const char*>(&m.uni_n), sizeof m.uni_n);
    out.write(reinterpret_cast<const char*>(m.o1), sizeof m.o1);
    out.write(reinterpret_cast<const char*>(m.o1n), sizeof m.o1n);
    out.write(reinterpret_cast<const char*>(m.root_child), sizeof m.root_child);
    out.write(reinterpret_cast<const char*>(m.nodes.data()), nn * sizeof(Node));
    out.write(reinterpret_cast<const char*>(m.fols.data()), nf * sizeof(Fol));
    const std::uint32_t slots = static_cast<std::uint32_t>(m.mk.size());
    out.write(reinterpret_cast<const char*>(&slots), 4);
    out.write(reinterpret_cast<const char*>(&m.mfilled), 4);
    out.write(reinterpret_cast<const char*>(m.mk.data()), slots * 4);
    out.write(reinterpret_cast<const char*>(m.mv.data()), slots * 4);
    out.write(reinterpret_cast<const char*>(m.mb.data()), slots);
    out.write(reinterpret_cast<const char*>(m.ms.data()), slots * 2);
    out.flush();
    if (!out) return false;
    out.close();
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        fs::rename(tmp, path, ec);
    }
    return !ec;
}

bool load_checkpoint(const fs::path& path, Model& m) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0) return false;
    std::uint32_t nn = 0, nf = 0, slots = 0;
    in.read(reinterpret_cast<char*>(&nn), 4);
    in.read(reinterpret_cast<char*>(&nf), 4);
    in.read(reinterpret_cast<char*>(&m.bytes_seen), 8);
    in.read(reinterpret_cast<char*>(m.uni), sizeof m.uni);
    in.read(reinterpret_cast<char*>(&m.uni_n), sizeof m.uni_n);
    in.read(reinterpret_cast<char*>(m.o1), sizeof m.o1);
    in.read(reinterpret_cast<char*>(m.o1n), sizeof m.o1n);
    in.read(reinterpret_cast<char*>(m.root_child), sizeof m.root_child);
    m.nodes.resize(nn);
    m.fols.resize(nf);
    in.read(reinterpret_cast<char*>(m.nodes.data()), nn * sizeof(Node));
    in.read(reinterpret_cast<char*>(m.fols.data()), nf * sizeof(Fol));
    in.read(reinterpret_cast<char*>(&slots), 4);
    in.read(reinterpret_cast<char*>(&m.mfilled), 4);
    m.mk.resize(slots);
    m.mv.resize(slots);
    m.mb.resize(slots);
    m.ms.resize(slots);
    in.read(reinterpret_cast<char*>(m.mk.data()), slots * 4);
    in.read(reinterpret_cast<char*>(m.mv.data()), slots * 4);
    in.read(reinterpret_cast<char*>(m.mb.data()), slots);
    in.read(reinterpret_cast<char*>(m.ms.data()), slots * 2);
    m.mmask = slots ? slots - 1 : 0;
    return static_cast<bool>(in);
}

bool trainable(const fs::path& p) {
    const std::string s = p.generic_string();
    if (s.find("/labelled/") != std::string::npos || s.find("/trie/") != std::string::npos ||
        s.find("/sha256/") != std::string::npos)
        return false;
    if (p.filename().string().ends_with(".part")) return false;
    const auto ext = p.extension().string();
    return ext == ".txt" || ext == ".jsonl";
}

std::string json_string(const std::string& line, std::size_t& i) {
    std::string out;
    if (i >= line.size() || line[i] != '"') return out;
    ++i;
    while (i < line.size()) {
        char c = line[i++];
        if (c == '"') break;
        if (c == '\\' && i < line.size()) {
            char e = line[i++];
            if (e == 'n') out.push_back('\n');
            else if (e == 't') out.push_back('\t');
            else if (e == 'u' && i + 4 <= line.size()) i += 4;
            else out.push_back(e);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

bool extract_text(const std::string& line, std::string& text) {
    const auto key = line.find("\"text\"");
    if (key == std::string::npos) return false;
    auto i = line.find(':', key + 6);
    if (i == std::string::npos) return false;
    ++i;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size() || line[i] != '"') return false;
    text = json_string(line, i);
    return !text.empty();
}

void observe_bytes(Model& m, const std::string& bytes) {
    m.reset_hist();
    for (unsigned char c : bytes) m.observe_byte(c);
}

void merge_model(Model& dst, Model& src) {
    for (int s = 0; s < 256; ++s) {
        dst.uni[s] += src.uni[s];
        dst.uni_n += src.uni[s];
        for (int y = 0; y < 256; ++y) {
            dst.o1[s][y] += src.o1[s][y];
            dst.o1n[s] += src.o1[s][y];
        }
    }
    const auto walk = [&](auto&& self, std::uint32_t dnode, std::uint32_t snode) -> void {
        for (std::uint32_t f = src.nodes[snode].fol_head; f != kNil; f = src.fols[f].next) {
            dst.add_fol_n(dnode, src.fols[f].sym, src.fols[f].count);
        }
        if (snode == 0) {
            for (int b = 0; b < 256; ++b) {
                const std::uint32_t c = src.root_child[b];
                if (c == kNil) continue;
                const std::uint32_t child = dst.ensure_child(0, static_cast<std::uint8_t>(b));
                self(self, child, c);
            }
            return;
        }
        for (std::uint32_t c = src.nodes[snode].child_head; c != kNil; c = src.nodes[c].sibling) {
            const std::uint32_t child = dst.ensure_child(dnode, src.nodes[c].byte);
            self(self, child, c);
        }
    };
    if (!src.nodes.empty()) walk(walk, 0, 0);
    for (std::size_t i = 0; i < src.mk.size(); ++i) {
        if (src.mk[i] == kNil || src.ms[i] == 0) continue;
        dst.match_add(src.mk[i], src.mb[i], src.ms[i]);
    }
    dst.bytes_seen += src.bytes_seen;
}

std::string format_left(double seconds) {
    if (!(seconds >= 0) || seconds > 99 * 24 * 3600) return "--:--:--";
    const int s = static_cast<int>(seconds);
    const int h = s / 3600;
    const int m = (s % 3600) / 60;
    const int r = s % 60;
    char buf[16];
    std::snprintf(buf, sizeof buf, "%d:%02d:%02d", h, m, r);
    return buf;
}

void draw_bar(const std::string& name, std::uint64_t done, std::uint64_t total, double elapsed) {
    const double frac = total ? std::min(1.0, static_cast<double>(done) / static_cast<double>(total)) : 0;
    const double speed = elapsed > 0.05 ? static_cast<double>(done) / elapsed : 0;
    const double left = speed > 1 ? static_cast<double>(total - done) / speed : -1;
    constexpr int width = 28;
    const int filled = static_cast<int>(width * frac);
    std::string bar(filled, '#');
    bar.append(width - filled, '-');
    char line[180];
    std::snprintf(line, sizeof line, "\r[%s] %6.1f/%6.1f MB  %5.1f MB/s  left %s  8 cores  %.24s",
                  bar.c_str(), done / 1e6, total / 1e6, speed / 1e6, format_left(left).c_str(), name.c_str());
    std::fputs(line, stderr);
    std::fflush(stderr);
}

bool train_file(Model& model, const fs::path& path) {
    const auto ext = path.extension().string();
    const auto file_size = fs::file_size(path);
    if (file_size == 0) return true;
    const int cores = 8;
    std::vector<std::uint64_t> cuts(static_cast<std::size_t>(cores) + 1, 0);
    cuts.back() = file_size;
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        for (int i = 1; i < cores; ++i) {
            auto at = static_cast<std::uint64_t>(file_size * i / cores);
            in.seekg(static_cast<std::streamoff>(at));
            if (ext == ".jsonl") {
                std::string skip;
                std::getline(in, skip);
                at = static_cast<std::uint64_t>(in.tellg());
                if (!in) at = file_size;
            }
            cuts[static_cast<std::size_t>(i)] = at;
        }
    }
    std::atomic<std::uint64_t> read_bytes{0};
    std::atomic<bool> failed{false};
    std::vector<Model> shards(static_cast<std::size_t>(cores));
    for (auto& shard : shards) shard.init();
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<bool> running{true};
    std::thread ticker([&] {
        while (running.load()) {
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            draw_bar(path.filename().string(), read_bytes.load(), file_size, elapsed);
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
    });
    std::vector<std::thread> workers;
    for (int i = 0; i < cores; ++i) {
        workers.emplace_back([&, i] {
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                failed = true;
                return;
            }
            const auto begin = cuts[static_cast<std::size_t>(i)];
            const auto end = cuts[static_cast<std::size_t>(i) + 1];
            if (begin >= end) return;
            in.seekg(static_cast<std::streamoff>(begin));
            Model& shard = shards[static_cast<std::size_t>(i)];
            if (ext == ".jsonl") {
                std::string line, text;
                while (static_cast<std::uint64_t>(in.tellg()) < end && std::getline(in, line)) {
                    read_bytes += line.size() + 1;
                    if (!extract_text(line, text)) continue;
                    observe_bytes(shard, text);
                }
            } else {
                shard.reset_hist();
                std::vector<char> buf(1 << 16);
                std::uint64_t pos = begin;
                while (pos < end) {
                    const auto want = std::min<std::uint64_t>(buf.size(), end - pos);
                    in.read(buf.data(), static_cast<std::streamsize>(want));
                    const auto got = static_cast<std::uint64_t>(in.gcount());
                    if (got == 0) break;
                    for (std::uint64_t k = 0; k < got; ++k)
                        shard.observe_byte(static_cast<std::uint8_t>(buf[static_cast<std::size_t>(k)]));
                    pos += got;
                    read_bytes += got;
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    running = false;
    ticker.join();
    std::fputc('\n', stderr);
    if (failed.load()) return false;
    std::cerr << "merging " << cores << " shards\n";
    for (auto& shard : shards) merge_model(model, shard);
    std::cerr << "trained " << path.filename().string() << " nodes " << model.nodes.size()
              << " seen " << model.bytes_seen << "\n";
    return true;
}

std::vector<std::string> load_trained(const fs::path& list) {
    std::vector<std::string> out;
    std::ifstream in(list);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

void append_trained(const fs::path& list, const fs::path& file) {
    std::ofstream out(list, std::ios::app);
    out << file.generic_string() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    bool once = false;
    fs::path root = "/mnt/d/cypha_llm/corpus";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--once") once = true;
        else root = arg;
    }
    const fs::path dir = root / "trie";
    fs::create_directories(dir);
    const fs::path ckpt = dir / "checkpoint.bin";
    const fs::path trained_list = dir / "trained.txt";

    Model model;
    model.init();
    if (fs::exists(ckpt)) {
        std::ifstream probe(ckpt, std::ios::binary);
        char magic[4] = {};
        probe.read(magic, 4);
        if (std::memcmp(magic, kMagic, 4) != 0) {
            std::cerr << "refusing " << ckpt.generic_string() << ": not a TR64 checkpoint (TRI1 is the old 32-bit "
                      << "layout). Move it aside to train a new model.\n";
            return 1;
        }
    }
    if (load_checkpoint(ckpt, model)) {
        std::cerr << "resumed bytes " << model.bytes_seen << " nodes " << model.nodes.size() << "\n";
    } else {
        std::cerr << "new checkpoint\n";
    }
    auto done = load_trained(trained_list);

    while (true) {
        std::vector<fs::path> fresh;
        std::error_code ec;
        if (fs::exists(root, ec)) {
            for (auto it = fs::recursive_directory_iterator(root, ec);
                 it != fs::recursive_directory_iterator(); it.increment(ec)) {
                if (ec) break;
                if (!it->is_regular_file()) continue;
                const fs::path p = it->path();
                if (!trainable(p)) continue;
                const std::string key = p.generic_string();
                if (std::find(done.begin(), done.end(), key) != done.end()) continue;
                fresh.push_back(p);
            }
        }
        std::sort(fresh.begin(), fresh.end());
        if (fresh.empty()) {
            if (once) {
                std::cerr << "done\n";
                return 0;
            }
            std::cerr << "waiting for a finished file\n";
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
        for (const auto& p : fresh) {
            std::cerr << "file " << p.generic_string() << "\n";
            if (!train_file(model, p)) {
                std::cerr << "skip unreadable " << p << "\n";
                continue;
            }
            if (!save_checkpoint(ckpt, model)) {
                std::cerr << "checkpoint failed\n";
                return 1;
            }
            append_trained(trained_list, p);
            done.push_back(p.generic_string());
            std::cerr << "checkpoint bytes " << model.bytes_seen << "\n";
        }
    }
}
