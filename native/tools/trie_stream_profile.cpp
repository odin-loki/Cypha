// Profile a TR64 streaming-trie checkpoint (native/tools/trie_stream_train.cpp).
//
// Structural counts, packed-size estimate, and next-byte scores on a text
// file. The fit slice teaches mixture weights only. Counts stay frozen.
// Default split matches the 8 MiB enwik protocol: fit [0, 7340032),
// score [7340032, 8388608). A second 1 MiB slice at 200 MiB is scored
// with those same weights when the file is long enough.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kNil = 0xffffffffu;
constexpr char kMagic[4] = {'T', 'R', '6', '4'};
constexpr double kAlpha = 0.4;
constexpr double kAgreeFloor = 0.05;
constexpr double kMatchCap = 0.75;
constexpr double kEta = 0.05;
constexpr double kWFloor = 1e-4;
constexpr int kGen = 200;
constexpr double kTemperature = 0.8;
constexpr double kNucleus = 0.9;

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

static_assert(sizeof(Node) == 32, "Node layout");
static_assert(sizeof(Fol) == 16, "Fol layout");

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

    std::uint32_t find_child(std::uint32_t node, std::uint8_t b) const {
        if (node == 0) return root_child[b];
        for (std::uint32_t c = nodes[node].child_head; c != kNil; c = nodes[c].sibling)
            if (nodes[c].byte == b) return c;
        return kNil;
    }

    bool lookup_match(std::uint32_t key, unsigned& pred) const {
        if (mk.empty()) return false;
        std::uint32_t i = mix32(key) & mmask;
        for (;;) {
            if (mk[i] == kNil) return false;
            if (mk[i] == key) {
                if (ms[i] < 2) return false;
                pred = mb[i];
                return true;
            }
            i = (i + 1u) & mmask;
        }
    }

    void fill_kt0(double* p) const {
        const double den = static_cast<double>(uni_n) + 128.0;
        for (int s = 0; s < 256; ++s) p[s] = (static_cast<double>(uni[s]) + 0.5) / den;
    }

    void fill_kt1(unsigned prev, double* p) const {
        const double den = static_cast<double>(o1n[prev]) + 128.0;
        for (int s = 0; s < 256; ++s) p[s] = (static_cast<double>(o1[prev][s]) + 0.5) / den;
    }

    // Stupid backoff through `order`. order 1 is KT order-1, or KT unigram
    // when that context was never seen. Seen followers replace the backoff
    // mass; unseen symbols keep alpha times the parent. Then renormalise.
    // drop1 skips follower counts of 1.
    void backoff(const std::uint8_t* data, std::size_t i, int order, bool drop1, double* out) const {
        double cur[256];
        double nxt[256];
        if (i >= 1 && o1n[data[i - 1]] > 0) fill_kt1(data[i - 1], cur);
        else fill_kt0(cur);
        const int top = std::min(order, std::min(4, static_cast<int>(i)));
        std::uint32_t node = 0;
        for (int o = 2; o <= top; ++o) {
            if (node == kNil) break;
            const std::uint32_t prev = (o == 2) ? find_child(0, data[i - 1]) : node;
            if (prev == kNil) break;
            const std::uint32_t next = find_child(prev, data[i - static_cast<std::size_t>(o)]);
            if (next == kNil) break;
            node = next;
            for (int s = 0; s < 256; ++s) nxt[s] = kAlpha * cur[s];
            if (nodes[node].total > 0 && !(drop1 && nodes[node].total == 1)) {
                std::uint64_t tot = 0;
                for (std::uint32_t f = nodes[node].fol_head; f != kNil; f = fols[f].next) {
                    if (drop1 && fols[f].count < 2) continue;
                    tot += fols[f].count;
                }
                if (tot > 0) {
                    const double inv = 1.0 / static_cast<double>(tot);
                    for (std::uint32_t f = nodes[node].fol_head; f != kNil; f = fols[f].next) {
                        if (drop1 && fols[f].count < 2) continue;
                        nxt[fols[f].sym] = static_cast<double>(fols[f].count) * inv;
                    }
                }
            }
            std::memcpy(cur, nxt, sizeof cur);
        }
        double z = 0;
        for (int s = 0; s < 256; ++s) z += cur[s];
        if (!(z > 0)) z = 1;
        const double inv = 1.0 / z;
        for (int s = 0; s < 256; ++s) out[s] = cur[s] * inv;
    }
};

bool load_model(const std::string& path, Model& m) {
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
    in.read(reinterpret_cast<char*>(&m.uni_n), 8);
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

double entropy_bits(const double* p) {
    double h = 0;
    for (int s = 0; s < 256; ++s)
        if (p[s] > 0) h -= p[s] * std::log2(p[s]);
    return h;
}

double entropy_u(const double* p) {
    double h = 0;
    for (int s = 0; s < 256; ++s)
        if (p[s] > 0) h -= p[s] * std::log(p[s]);
    double u = h / std::log(256.0);
    if (u < 0) u = 0;
    if (u > 1) u = 1;
    return u;
}

int argmax(const double* p) {
    int best = 0;
    for (int s = 1; s < 256; ++s)
        if (p[s] > p[best]) best = s;
    return best;
}

void eg_update(double w[6], const double py[6], bool match_on) {
    const bool active[6] = {true, true, true, true, true, match_on};
    double p = 0;
    for (int i = 0; i < 6; ++i)
        if (active[i]) p += w[i] * py[i];
    if (!(p > 1e-12)) p = 1e-12;
    for (int i = 0; i < 6; ++i) {
        if (!active[i]) continue;
        double ex = kEta * (py[i] / p);
        if (ex > 40) ex = 40;
        w[i] *= std::exp(ex);
    }
    double sum = 0;
    for (int i = 0; i < 6; ++i) {
        if (!(w[i] > kWFloor)) w[i] = kWFloor;
        sum += w[i];
    }
    const double inv = 1.0 / sum;
    for (int i = 0; i < 6; ++i) w[i] *= inv;
}

void mix_dist(const double w[6], const double e[5][256], bool matched, unsigned pred, double* out) {
    double z = 0;
    for (int i = 0; i < 5; ++i) z += w[i];
    if (matched) z += w[5];
    const double inv = 1.0 / z;
    std::memset(out, 0, sizeof(double) * 256);
    for (int i = 0; i < 5; ++i) {
        const double a = w[i] * inv;
        for (int s = 0; s < 256; ++s) out[s] += a * e[i][s];
    }
    if (matched) out[pred] += w[5] * inv;
}

bool match_here(const Model& m, const std::uint8_t* data, std::size_t i, unsigned& pred) {
    if (i < 4) return false;
    const std::uint32_t key = (std::uint32_t(data[i - 4]) << 24) | (std::uint32_t(data[i - 3]) << 16) |
                              (std::uint32_t(data[i - 2]) << 8) | std::uint32_t(data[i - 1]);
    return m.lookup_match(key, pred);
}

struct Acc {
    double bits[6]{};
    double bits_blend = 0;
    double bits_mix = 0;
    double ent = 0;
    std::uint64_t n = 0;
    std::uint64_t match_fire = 0;
    std::uint64_t match_hit = 0;
    std::uint64_t order4 = 0;
    std::uint64_t top1 = 0;
};

const char* kBitName[6] = {"kt0", "kt1", "bo2", "bo3", "bo4", "match_hit"};

void observe_step(const Model& m, const std::uint8_t* data, std::size_t i, const double w[6], Acc& a, bool learn,
                  double* wout) {
    double e[5][256];
    m.fill_kt0(e[0]);
    if (i >= 1) m.fill_kt1(data[i - 1], e[1]);
    else std::memcpy(e[1], e[0], sizeof e[0]);
    m.backoff(data, i, 2, false, e[2]);
    m.backoff(data, i, 3, false, e[3]);
    m.backoff(data, i, 4, false, e[4]);
    unsigned pred = 0;
    const bool matched = match_here(m, data, i, pred);
    const unsigned y = data[i];
    double py[6];
    for (int k = 0; k < 5; ++k) py[k] = e[k][y];
    py[5] = (matched && pred == y) ? 1.0 : 0.0;
    if (learn) {
        eg_update(wout, py, matched);
        return;
    }
    double mix[256];
    mix_dist(w, e, matched, pred, mix);
    double bo[256];
    std::memcpy(bo, e[4], sizeof bo);
    double u = entropy_u(i >= 1 ? e[1] : e[0]);
    double blend[256];
    std::memcpy(blend, bo, sizeof blend);
    if (matched) {
        double agree = e[1][pred] / (e[1][pred] + kAgreeFloor);
        double mw = (1.0 - u) * agree;
        if (mw < 0) mw = 0;
        if (mw > kMatchCap) mw = kMatchCap;
        for (int s = 0; s < 256; ++s) blend[s] *= (1.0 - mw);
        blend[pred] += mw;
    }
    auto take = [&](double p, double& bits) {
        if (!(p > 0) || !std::isfinite(p)) p = 1e-12;
        bits += -std::log2(p);
    };
    for (int k = 0; k < 5; ++k) take(py[k], a.bits[k]);
    take(blend[y], a.bits_blend);
    take(mix[y], a.bits_mix);
    a.ent += entropy_bits(mix);
    if (matched) {
        ++a.match_fire;
        if (pred == y) ++a.match_hit;
    }
    std::uint32_t node = 0;
    int depth = 0;
    for (int d = 1; d <= 4 && static_cast<std::size_t>(d) <= i; ++d) {
        node = m.find_child(node, data[i - static_cast<std::size_t>(d)]);
        if (node == kNil) break;
        depth = d;
    }
    if (depth >= 4) ++a.order4;
    if (argmax(mix) == static_cast<int>(y)) ++a.top1;
    ++a.n;
}

void fit_weights(const Model& m, const std::uint8_t* data, std::size_t n, double w[6]) {
    w[5] = 0.15;
    const double share = (1.0 - w[5]) / 5.0;
    for (int i = 0; i < 5; ++i) w[i] = share;
    Acc ignore;
    for (std::size_t i = 0; i < n; ++i) {
        observe_step(m, data, i, w, ignore, true, w);
        if (((i + 1) & 0xfffff) == 0) std::cerr << "fit " << (i + 1) << "\n";
    }
}

Acc score_range(const Model& m, const std::uint8_t* data, std::size_t begin, std::size_t end, const double w[6]) {
    Acc a;
    for (std::size_t i = begin; i < end; ++i) {
        observe_step(m, data, i, w, a, false, nullptr);
        if (((i - begin + 1) & 0x3ffff) == 0) std::cerr << "score " << (i - begin + 1) << "\n";
    }
    return a;
}

void print_acc(const char* name, const Acc& a) {
    const double n = static_cast<double>(a.n);
    std::cout << name << "_bytes " << a.n << "\n";
    for (int k = 0; k < 5; ++k) std::cout << name << "_bpc_" << kBitName[k] << " " << (a.bits[k] / n) << "\n";
    std::cout << name << "_bpc_blend " << (a.bits_blend / n) << "\n";
    std::cout << name << "_bpc_mix " << (a.bits_mix / n) << "\n";
    std::cout << name << "_entropy " << (a.ent / n) << "\n";
    std::cout << name << "_top1 " << (static_cast<double>(a.top1) / n) << "\n";
    std::cout << name << "_order4 " << (static_cast<double>(a.order4) / n) << "\n";
    std::cout << name << "_match_rate " << (static_cast<double>(a.match_fire) / n) << "\n";
    std::cout << name << "_match_acc " << (a.match_fire ? static_cast<double>(a.match_hit) / static_cast<double>(a.match_fire) : 0)
              << "\n";
}

std::uint64_t rss_bytes() {
    std::ifstream in("/proc/self/status");
    std::string key;
    while (in >> key) {
        if (key == "VmRSS:") {
            std::uint64_t kb = 0;
            in >> kb;
            return kb * 1024ull;
        }
        std::string rest;
        std::getline(in, rest);
    }
    return 0;
}

std::uint64_t packed_bytes(const Model& m, bool drop1) {
    const std::size_t n = m.nodes.size();
    std::vector<char> useful(n, 0);
    auto kept = [&](std::uint32_t id) {
        if (m.nodes[id].total == 0) return false;
        if (drop1 && m.nodes[id].total == 1) return false;
        for (std::uint32_t f = m.nodes[id].fol_head; f != kNil; f = m.fols[f].next) {
            if (!drop1 || m.fols[f].count >= 2) return true;
        }
        return false;
    };
    for (std::uint32_t id = 1; id < n; ++id)
        if (kept(id)) useful[id] = 1;
    for (int pass = 0; pass < 4; ++pass) {
        for (std::uint32_t id = 1; id < n; ++id) {
            if (!useful[id]) continue;
            const std::uint32_t p = m.nodes[id].parent;
            if (p != kNil && p < n) useful[p] = 1;
        }
    }
    std::uint64_t bytes = 0;
    for (std::uint32_t id = 1; id < n; ++id) {
        if (!useful[id]) continue;
        int live = 0;
        for (std::uint32_t c = m.nodes[id].child_head; c != kNil; c = m.nodes[c].sibling)
            if (useful[c]) ++live;
        int nf = 0;
        for (std::uint32_t f = m.nodes[id].fol_head; f != kNil; f = m.fols[f].next)
            if (!drop1 || m.fols[f].count >= 2) ++nf;
        std::uint64_t b = 8;
        if (live <= 1) b += live ? 4 : 0;
        else b += static_cast<std::uint64_t>(live) * 5ull;
        b += static_cast<std::uint64_t>(nf) * 2ull;
        bytes += b;
    }
    return bytes;
}

bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

std::uint32_t gram4(const std::string& s, std::size_t i) {
    return (std::uint32_t(std::uint8_t(s[i])) << 24) | (std::uint32_t(std::uint8_t(s[i + 1])) << 16) |
           (std::uint32_t(std::uint8_t(s[i + 2])) << 8) | std::uint32_t(std::uint8_t(s[i + 3]));
}

double distinct4(const std::string& s) {
    if (s.size() < 4) return 0;
    std::vector<std::uint32_t> g;
    for (std::size_t i = 0; i + 3 < s.size(); ++i) g.push_back(gram4(s, i));
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return static_cast<double>(g.size()) / static_cast<double>(s.size() - 3);
}

double loop8(const std::string& s) {
    if (s.size() < 16) return 0;
    std::unordered_set<std::uint64_t> seen;
    int loops = 0, n = 0;
    for (std::size_t i = 0; i + 8 <= s.size(); ++i) {
        std::uint64_t w = 0;
        for (int k = 0; k < 8; ++k) w = (w << 8) | s[i + static_cast<std::size_t>(k)];
        if (!seen.insert(w).second) ++loops;
        ++n;
    }
    return n ? static_cast<double>(loops) / static_cast<double>(n) : 0;
}

void mark_banned(const std::vector<std::uint8_t>& seq, bool banned[256]) {
    std::memset(banned, 0, 256);
    const std::size_t n = seq.size();
    if (n < 11) return;
    const std::uint8_t* last = seq.data() + (n - 11);
    for (std::size_t s = 0; s + 12 <= n; ++s)
        if (std::memcmp(seq.data() + s, last, 11) == 0) banned[seq[s + 11]] = true;
}

std::string generate(const Model& m, const double w[6], const std::string& prompt, std::mt19937& rng) {
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int t = 0; t < kGen; ++t) {
        const std::size_t i = seq.size();
        double e[5][256];
        m.fill_kt0(e[0]);
        if (i >= 1) m.fill_kt1(seq[i - 1], e[1]);
        else std::memcpy(e[1], e[0], sizeof e[0]);
        m.backoff(seq.data(), i, 2, false, e[2]);
        m.backoff(seq.data(), i, 3, false, e[3]);
        m.backoff(seq.data(), i, 4, false, e[4]);
        unsigned pred = 0;
        const bool matched = match_here(m, seq.data(), i, pred);
        double mix[256];
        mix_dist(w, e, matched, pred, mix);
        bool banned[256];
        mark_banned(seq, banned);
        double raw[256];
        double mass = 0;
        for (int s = 0; s < 256; ++s) {
            if (banned[s]) {
                raw[s] = 0;
                continue;
            }
            raw[s] = std::pow(std::max(mix[s], 0.0), 1.0 / kTemperature);
            mass += raw[s];
        }
        if (!(mass > 0)) {
            mass = 0;
            for (int s = 0; s < 256; ++s) {
                raw[s] = std::pow(std::max(mix[s], 0.0), 1.0 / kTemperature);
                mass += raw[s];
            }
        }
        struct Item {
            int y;
            double p;
        } items[256];
        for (int s = 0; s < 256; ++s) items[s] = {s, raw[s] / mass};
        std::sort(items, items + 256, [](const Item& a, const Item& b) { return a.p > b.p; });
        double acc = 0;
        int keep = 1;
        for (int n = 0; n < 256; ++n) {
            acc += items[n].p;
            keep = n + 1;
            if (acc >= kNucleus) break;
        }
        double keep_mass = 0;
        for (int n = 0; n < keep; ++n) keep_mass += items[n].p;
        double r = U(rng) * keep_mass;
        int pick = items[keep - 1].y;
        for (int n = 0; n < keep; ++n) {
            r -= items[n].p;
            if (r <= 0) {
                pick = items[n].y;
                break;
            }
        }
        seq.push_back(static_cast<std::uint8_t>(pick));
    }
    return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
}

std::string show(const std::string& s) {
    std::string o;
    const char* hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c >= 32 && c < 127) o += static_cast<char>(c);
        else {
            o += "\\x";
            o += hex[c >> 4];
            o += hex[c & 15];
        }
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    const std::string ckpt = argc > 1 ? argv[1] : "/mnt/d/cypha_llm/corpus/trie/checkpoint.bin";
    const std::string text_path = argc > 2 ? argv[2] : "/mnt/d/cypha_llm/enwik9";
    Model model;
    const auto t0 = std::chrono::steady_clock::now();
    if (!load_model(ckpt, model)) {
        std::cerr << "cannot load TR64 checkpoint " << ckpt << "\n";
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "load_s " << load_s << "\n";
    std::cout << "rss_bytes " << rss_bytes() << "\n";
    std::cout << "bytes_seen " << model.bytes_seen << "\n";
    std::cout << "uni_n " << model.uni_n << "\n";
    std::cout << "uni_matches_seen " << (model.uni_n == model.bytes_seen ? 1 : 0) << "\n";
    std::cout << "nodes " << model.nodes.size() << "\n";
    std::cout << "followers " << model.fols.size() << "\n";
    std::cout << "match_slots " << model.mk.size() << "\n";
    std::cout << "match_filled " << model.mfilled << "\n";

    std::uint64_t depth_n[8]{};
    std::uint64_t depth_sum[8]{};
    std::uint64_t depth_max[8]{};
    std::uint64_t once = 0, fol_once = 0;
    for (const Node& n : model.nodes) {
        const int d = n.depth < 8 ? n.depth : 7;
        ++depth_n[d];
        depth_sum[d] += n.total;
        if (n.total > depth_max[d]) depth_max[d] = n.total;
        if (n.depth >= 2 && n.total == 1) ++once;
    }
    for (const Fol& f : model.fols)
        if (f.count == 1) ++fol_once;
    for (int d = 0; d <= 4; ++d) {
        std::cout << "depth" << d << "_nodes " << depth_n[d] << "\n";
        std::cout << "depth" << d << "_total " << depth_sum[d] << "\n";
        std::cout << "depth" << d << "_max " << depth_max[d] << "\n";
    }
    std::cout << "ctx_once " << once << "\n";
    std::cout << "fol_once " << fol_once << "\n";
    std::uint64_t match_once = 0, match_kept = 0;
    for (std::size_t i = 0; i < model.mk.size(); ++i) {
        if (model.mk[i] == kNil) continue;
        if (model.ms[i] >= 2) ++match_kept;
        else ++match_once;
    }
    std::cout << "match_kept " << match_kept << "\n";
    std::cout << "match_once " << match_once << "\n";
    std::cout << "packed_full " << packed_bytes(model, false) << "\n";
    std::cout << "packed_drop1 " << packed_bytes(model, true) << "\n";

    double h0 = 0;
    for (int s = 0; s < 256; ++s) {
        if (model.uni[s] == 0) continue;
        const double p = static_cast<double>(model.uni[s]) / static_cast<double>(model.uni_n);
        h0 -= p * std::log2(p);
    }
    std::cout << "unigram_bits " << h0 << "\n";
    int shown = 0;
    std::vector<int> order(256);
    for (int s = 0; s < 256; ++s) order[s] = s;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return model.uni[a] > model.uni[b]; });
    std::cout << "top_bytes";
    for (int s : order) {
        if (shown++ == 8) break;
        std::cout << " " << s << ":" << model.uni[s];
    }
    std::cout << "\n";

    std::ifstream tin(text_path, std::ios::binary);
    if (!tin) {
        std::cerr << "no text " << text_path << "\n";
        return 1;
    }
    tin.seekg(0, std::ios::end);
    const std::size_t file_n = static_cast<std::size_t>(tin.tellg());
    std::size_t fit_end = argc > 3 ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)) : 7340032u;
    std::size_t hold_end = argc > 4 ? static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10)) : 8388608u;
    if (fit_end > file_n) fit_end = file_n * 3 / 4;
    if (hold_end > file_n) hold_end = file_n;
    if (hold_end <= fit_end) {
        std::cerr << "holdout does not follow the fit slice\n";
        return 1;
    }
    const std::size_t need = std::max(hold_end, std::min(file_n, std::size_t{201048576}));
    std::vector<std::uint8_t> text(need);
    tin.seekg(0);
    tin.read(reinterpret_cast<char*>(text.data()), static_cast<std::streamsize>(need));
    if (static_cast<std::size_t>(tin.gcount()) < hold_end) {
        std::cerr << "short text\n";
        return 1;
    }

    double w[6];
    fit_weights(model, text.data(), fit_end, w);
    std::cout << "w0 " << w[0] << "\n";
    std::cout << "w1 " << w[1] << "\n";
    std::cout << "w2 " << w[2] << "\n";
    std::cout << "w3 " << w[3] << "\n";
    std::cout << "w4 " << w[4] << "\n";
    std::cout << "wmatch " << w[5] << "\n";
    const Acc hold = score_range(model, text.data(), fit_end, hold_end, w);
    print_acc("hold", hold);
    if (file_n >= 201048576 && text.size() >= 201048576) {
        const Acc far = score_range(model, text.data(), 200000000, 201048576, w);
        print_acc("far", far);
    }

    std::unordered_set<std::string> vocab;
    const std::size_t vocab_n = std::min(fit_end, text.size());
    for (std::size_t i = 0; i < vocab_n;) {
        if (!is_letter(text[i])) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < vocab_n && is_letter(text[j])) ++j;
        if (j - i >= 3 && j - i <= 32) {
            std::string word(reinterpret_cast<const char*>(text.data() + i), j - i);
            for (char& c : word)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            vocab.insert(std::move(word));
        }
        i = j;
    }
    double ref_d4 = 0, ref_loop = 0;
    int windows = 0;
    for (std::size_t off = 100000; off + 200 < vocab_n && windows < 40; off += 150000) {
        std::string slice(reinterpret_cast<const char*>(text.data() + off), 200);
        ref_d4 += distinct4(slice);
        ref_loop += loop8(slice);
        ++windows;
    }
    if (windows) {
        ref_d4 /= windows;
        ref_loop /= windows;
    }
    std::cout << "ref_distinct4 " << ref_d4 << "\n";
    std::cout << "ref_loop8 " << ref_loop << "\n";
    std::cout << "vocab " << vocab.size() << "\n";
    std::mt19937 rng(42);
    const char* prompts[] = {"The ", "In 19", "def ", "\n\n"};
    for (const char* prompt : prompts) {
        const std::string sample = generate(model, w, prompt, rng);
        int words = 0, hit = 0;
        for (std::size_t i = 0; i < sample.size();) {
            unsigned char c = static_cast<unsigned char>(sample[i]);
            if (!is_letter(c)) {
                ++i;
                continue;
            }
            std::size_t j = i;
            while (j < sample.size() && is_letter(static_cast<unsigned char>(sample[j]))) ++j;
            if (j - i >= 3) {
                std::string word = sample.substr(i, j - i);
                for (char& ch : word)
                    if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
                ++words;
                if (vocab.count(word)) ++hit;
            }
            i = j;
        }
        const double H = words ? static_cast<double>(hit) / static_cast<double>(words) : 0;
        const double d4 = distinct4(sample);
        const double lp = loop8(sample);
        double D = 1.0 - std::fabs(d4 - ref_d4);
        if (D < 0) D = 0;
        double L = 1.0 - std::max(0.0, lp - ref_loop);
        if (L < 0) L = 0;
        std::cout << "sample prompt " << show(prompt) << "\n";
        std::cout << "sample text " << show(sample) << "\n";
        std::cout << "sample S " << (H * D * L) << " H " << H << " D " << D << " L " << L << "\n";
    }
    std::cout << "rss_end " << rss_bytes() << "\n";
    const double all_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "profile_s " << all_s << "\n";
    return 0;
}
