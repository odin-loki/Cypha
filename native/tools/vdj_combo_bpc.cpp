// Combinatorial context generation (VDJ) against stored n-gram tables.
//
// Every context model is the same Krichevsky–Trofimov counter. The masks are
// r-subsets of an 8-segment library, which is CombinatorialGenerator's C(n,r).
// After the first 1 MiB, models worse than the median analysis bits/byte are
// dropped (optimize_patterns, with bits/byte in place of variance). Survivors
// are mixed with weights 1, 1/2, 1/4, ... by rank. The paper states that scale
// once; the Python reference multiplies it in two places, which would be 1/4^k.
//
// The bytes are the first 8,388,608 of enwik9, SHA-256
// 09f6dd7241a8ae21edfd6762f3c6712a1fd02f7f322c5e77cab8bb88f292ee8e,
// the enwik8.8mb slice in bench/data/enwik8/README.md.
// Reported bits/byte are the last 7 MiB. Counts carry forward, and learning
// stays on, so the figure is an online code length for that region.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kAtoms = 8;
constexpr int kMaxR = 3;
constexpr std::uint32_t kBits = 18;
constexpr std::uint32_t kSize = 1u << kBits;
constexpr std::uint32_t kMask = kSize - 1;
constexpr int kProbe = 12;
constexpr std::size_t kSample = 8'388'608;
constexpr std::size_t kAnalyze = 1'048'576;

const char* kAtomName[kAtoms] = {"b1", "b2", "b3", "b4", "cls", "hi", "lo", "col"};

std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// Open-address table. The full key is stored, so two contexts share a slot
// only when the probe window is already full, and that insert is counted as
// a drop rather than charged to the wrong context.
struct Table {
    struct Slot {
        std::uint64_t key = 0;
        std::uint32_t n = 0;
        std::uint32_t used = 0;
    };
    std::vector<Slot> slots{kSize};
    std::uint64_t drops = 0;

    std::uint32_t get(std::uint64_t key) const {
        std::uint32_t i = static_cast<std::uint32_t>(mix64(key)) & kMask;
        for (int p = 0; p < kProbe; ++p) {
            const Slot& s = slots[i];
            if (!s.used) return 0;
            if (s.key == key) return s.n;
            i = (i + 1) & kMask;
        }
        return 0;
    }

    void add(std::uint64_t key) {
        std::uint32_t i = static_cast<std::uint32_t>(mix64(key)) & kMask;
        for (int p = 0; p < kProbe; ++p) {
            Slot& s = slots[i];
            if (!s.used || s.key == key) {
                s.used = 1;
                s.key = key;
                ++s.n;
                return;
            }
            i = (i + 1) & kMask;
        }
        ++drops;
    }

    std::uint64_t occupied() const {
        std::uint64_t n = 0;
        for (const Slot& s : slots)
            if (s.used) ++n;
        return n;
    }
};

struct Model {
    int r = 0;
    std::array<int, kMaxR> atom{};
    Table totals;
    Table joints;
    double bits = 0;

    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    std::uint64_t context(const std::uint8_t value[kAtoms]) const {
        std::uint64_t ctx = 0;
        for (int k = 0; k < r; ++k) ctx = (ctx << 8) | value[atom[static_cast<std::size_t>(k)]];
        return ctx == 0 ? 1 : ctx;
    }

    static std::uint64_t joint_key(std::uint64_t ctx, unsigned y) { return (ctx << 8) | y; }

    double prob(std::uint64_t ctx, unsigned y) const {
        const std::uint32_t n = totals.get(ctx);
        const std::uint32_t c = joints.get(joint_key(ctx, y));
        return (static_cast<double>(c) + 0.5) / (static_cast<double>(n) + 128.0);
    }

    void update(std::uint64_t ctx, unsigned y) {
        totals.add(ctx);
        joints.add(joint_key(ctx, y));
    }

    std::string name() const {
        if (r == 0) return "unigram";
        std::string s;
        for (int k = 0; k < r; ++k) {
            if (k) s += '+';
            s += kAtomName[atom[static_cast<std::size_t>(k)]];
        }
        return s;
    }
};

std::uint8_t byte_class(unsigned b) {
    if (b == ' ' || b == '\n' || b == '\t' || b == '\r') return 1;
    if (b >= '0' && b <= '9') return 2;
    if (b >= 'a' && b <= 'z') return 3;
    if (b >= 'A' && b <= 'Z') return 4;
    return 5;
}

std::vector<Model> build_models() {
    std::vector<Model> models;
    const int ncomb = 1 + 8 + 28 + 56;  // unigram + C(8,1) + C(8,2) + C(8,3)
    models.reserve(static_cast<std::size_t>(ncomb));
    models.push_back(Model{});

    std::array<int, kMaxR> cur{};
    auto rec = [&](auto&& self, int start, int depth, int r) -> void {
        if (depth == r) {
            Model m;
            m.r = r;
            m.atom = cur;
            models.push_back(std::move(m));
            return;
        }
        for (int i = start; i < kAtoms; ++i) {
            cur[static_cast<std::size_t>(depth)] = i;
            self(self, i + 1, depth + 1, r);
        }
    };
    for (int r = 1; r <= kMaxR; ++r) rec(rec, 0, 0, r);
    return models;
}

std::vector<std::uint8_t> read_prefix(const std::string& path, std::size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::uint8_t> data(n);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n));
    if (static_cast<std::size_t>(in.gcount()) < n) throw std::runtime_error("file shorter than 8 MiB");
    return data;
}

struct Hist {
    std::uint8_t h1 = 0, h2 = 0, h3 = 0, h4 = 0;
    int col = 0;

    void fill(std::uint8_t v[kAtoms]) const {
        v[0] = h1;
        v[1] = h2;
        v[2] = h3;
        v[3] = h4;
        v[4] = byte_class(h1);
        v[5] = static_cast<std::uint8_t>(h1 >> 4);
        v[6] = static_cast<std::uint8_t>(h1 & 15);
        v[7] = static_cast<std::uint8_t>(col > 63 ? 63 : col);
    }

    void advance(unsigned y) {
        h4 = h3;
        h3 = h2;
        h2 = h1;
        h1 = static_cast<std::uint8_t>(y);
        col = (y == '\n') ? 0 : col + 1;
    }
};

std::uint64_t context4(const std::uint8_t v[kAtoms]) {
    std::uint64_t ctx = 0;
    for (int k = 0; k < 4; ++k) ctx = (ctx << 8) | v[k];
    return ctx == 0 ? 1 : ctx;
}

double prob_of(const Table& totals, const Table& joints, std::uint64_t ctx, unsigned y) {
    const std::uint32_t n = totals.get(ctx);
    const std::uint32_t c = joints.get((ctx << 8) | y);
    return (static_cast<double>(c) + 0.5) / (static_cast<double>(n) + 128.0);
}

int find_prefix(const std::vector<Model>& models, std::initializer_list<int> idx) {
    for (int i = 0; i < static_cast<int>(models.size()); ++i) {
        const Model& m = models[static_cast<std::size_t>(i)];
        if (m.r != static_cast<int>(idx.size())) continue;
        bool ok = true;
        int k = 0;
        for (int a : idx) {
            if (m.atom[static_cast<std::size_t>(k)] != a) ok = false;
            ++k;
        }
        if (ok) return i;
    }
    return -1;
}

double mb(std::uint64_t bytes_n) { return static_cast<double>(bytes_n) / (1024.0 * 1024.0); }

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "/mnt/d/cypha_llm/enwik9";
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<std::uint8_t> data = read_prefix(path, kSample);
        const std::span<const std::uint8_t> bytes(data);

        std::vector<Model> models = build_models();
        const int n_models = static_cast<int>(models.size());
        Table g4_tot;
        Table g4_jnt;
        double g4_analyze = 0;
        std::cerr << "models " << n_models << " slots/table " << kSize << "\n";

        Hist hist;
        std::uint8_t atom[kAtoms]{};
        std::vector<double> p(static_cast<std::size_t>(n_models));

        for (std::size_t i = 0; i < kAnalyze; ++i) {
            hist.fill(atom);
            const unsigned y = bytes[i];
            for (int m = 0; m < n_models; ++m) {
                Model& model = models[static_cast<std::size_t>(m)];
                const std::uint64_t ctx = model.context(atom);
                const double pr = model.prob(ctx, y);
                model.bits += -std::log2(pr);
                model.update(ctx, y);
            }
            {
                const std::uint64_t ctx = context4(atom);
                const double pr = prob_of(g4_tot, g4_jnt, ctx, y);
                g4_analyze += -std::log2(pr);
                g4_tot.add(ctx);
                g4_jnt.add((ctx << 8) | y);
            }
            hist.advance(y);
            if (((i + 1) & 0xfffff) == 0) std::cerr << "analyze " << (i + 1) << "\n";
        }

        std::vector<int> order(static_cast<std::size_t>(n_models));
        for (int i = 0; i < n_models; ++i) order[static_cast<std::size_t>(i)] = i;
        std::ranges::sort(order, [&](int a, int b) {
            return models[static_cast<std::size_t>(a)].bits < models[static_cast<std::size_t>(b)].bits;
        });
        const double median = models[static_cast<std::size_t>(order[order.size() / 2])].bits;
        std::vector<int> kept;
        for (int id : order) {
            if (models[static_cast<std::size_t>(id)].bits <= median) kept.push_back(id);
        }
        std::vector<double> geo(kept.size());
        double geo_sum = 0;
        for (std::size_t i = 0; i < kept.size(); ++i) {
            geo[i] = std::ldexp(1.0, -static_cast<int>(i));
            geo_sum += geo[i];
        }

        const int o1 = find_prefix(models, {0});
        const int o2 = find_prefix(models, {0, 1});
        const int o3 = find_prefix(models, {0, 1, 2});
        if (o1 < 0 || o2 < 0 || o3 < 0) throw std::runtime_error("missing plain order model");

        struct Ord {
            int id;
            double bits;
            bool gram4;
        };
        std::vector<Ord> ords{{o1, models[static_cast<std::size_t>(o1)].bits, false},
                              {o2, models[static_cast<std::size_t>(o2)].bits, false},
                              {o3, models[static_cast<std::size_t>(o3)].bits, false},
                              {-1, g4_analyze, true}};
        std::ranges::sort(ords, [](const Ord& a, const Ord& b) { return a.bits < b.bits; });
        std::array<double, 4> ord_w{};
        double ord_sum = 0;
        for (int i = 0; i < 4; ++i) {
            ord_w[static_cast<std::size_t>(i)] = std::ldexp(1.0, -i);
            ord_sum += ord_w[static_cast<std::size_t>(i)];
        }

        std::vector<double> eval_bits(static_cast<std::size_t>(n_models), 0.0);
        double mix_all = 0, mix_eq = 0, mix_geo = 0, mix_ord = 0, g4_eval = 0;
        const double inv_all = 1.0 / static_cast<double>(n_models);
        const double inv_kept = 1.0 / static_cast<double>(kept.size());
        std::uint64_t scored = 0;

        for (std::size_t i = kAnalyze; i < bytes.size(); ++i) {
            hist.fill(atom);
            const unsigned y = bytes[i];
            double sum_all = 0;
            for (int m = 0; m < n_models; ++m) {
                Model& model = models[static_cast<std::size_t>(m)];
                const std::uint64_t ctx = model.context(atom);
                p[static_cast<std::size_t>(m)] = model.prob(ctx, y);
                sum_all += p[static_cast<std::size_t>(m)];
            }
            double sum_eq = 0, sum_geo = 0;
            for (std::size_t k = 0; k < kept.size(); ++k) {
                const double pr = p[static_cast<std::size_t>(kept[k])];
                sum_eq += pr;
                sum_geo += geo[k] * pr;
            }
            const std::uint64_t c4 = context4(atom);
            const double p4 = prob_of(g4_tot, g4_jnt, c4, y);
            double sum_ord = 0;
            for (int k = 0; k < 4; ++k) {
                const double pr = ords[static_cast<std::size_t>(k)].gram4
                                      ? p4
                                      : p[static_cast<std::size_t>(ords[static_cast<std::size_t>(k)].id)];
                sum_ord += ord_w[static_cast<std::size_t>(k)] * pr;
            }
            mix_all += -std::log2(sum_all * inv_all);
            mix_eq += -std::log2(sum_eq * inv_kept);
            mix_geo += -std::log2(sum_geo / geo_sum);
            mix_ord += -std::log2(sum_ord / ord_sum);
            g4_eval += -std::log2(p4);
            for (int m = 0; m < n_models; ++m) {
                eval_bits[static_cast<std::size_t>(m)] += -std::log2(p[static_cast<std::size_t>(m)]);
                Model& model = models[static_cast<std::size_t>(m)];
                model.update(model.context(atom), y);
            }
            g4_tot.add(c4);
            g4_jnt.add((c4 << 8) | y);
            hist.advance(y);
            ++scored;
            if (((i + 1) & 0xfffff) == 0) std::cerr << "eval " << (i + 1) << "\n";
        }

        const double nb = static_cast<double>(scored);
        const double na = static_cast<double>(kAnalyze);
        std::uint64_t occ_all = 0, drops_all = 0, occ_kept = 0;
        for (int i = 0; i < n_models; ++i) {
            occ_all += models[static_cast<std::size_t>(i)].totals.occupied();
            occ_all += models[static_cast<std::size_t>(i)].joints.occupied();
            drops_all += models[static_cast<std::size_t>(i)].totals.drops;
            drops_all += models[static_cast<std::size_t>(i)].joints.drops;
        }
        for (int id : kept) {
            occ_kept += models[static_cast<std::size_t>(id)].totals.occupied();
            occ_kept += models[static_cast<std::size_t>(id)].joints.occupied();
        }
        const std::uint64_t occ4 = g4_tot.occupied() + g4_jnt.occupied();
        constexpr std::uint64_t kSlotBytes = sizeof(Table::Slot);
        const std::uint64_t reserved_one = static_cast<std::uint64_t>(kSize) * kSlotBytes * 2;

        std::cout << "sample_bytes " << bytes.size() << "\n";
        std::cout << "analyze_bytes " << kAnalyze << " eval_bytes " << scored << "\n";
        std::cout << "generated_models " << (n_models - 1) << " kept " << kept.size()
                  << " median_analyze_bpc " << (median / na) << "\n";
        std::cout << "eval_bpc_all_equal " << (mix_all / nb) << "\n";
        std::cout << "eval_bpc_kept_equal " << (mix_eq / nb) << "\n";
        std::cout << "eval_bpc_kept_geometric " << (mix_geo / nb) << "\n";
        std::cout << "eval_bpc_orders_geometric " << (mix_ord / nb) << "\n";
        std::cout << "eval_bpc_unigram " << (eval_bits[0] / nb) << "\n";
        std::cout << "eval_bpc_order1 " << (eval_bits[static_cast<std::size_t>(o1)] / nb) << "\n";
        std::cout << "eval_bpc_order2 " << (eval_bits[static_cast<std::size_t>(o2)] / nb) << "\n";
        std::cout << "eval_bpc_order3 " << (eval_bits[static_cast<std::size_t>(o3)] / nb) << "\n";
        std::cout << "eval_bpc_order4 " << (g4_eval / nb) << "\n";
        std::cout << "analyze_bpc_best " << (models[static_cast<std::size_t>(order[0])].bits / na) << " "
                  << models[static_cast<std::size_t>(order[0])].name() << "\n";
        std::cout << "packed_mb_all " << mb((occ_all + occ4) * kSlotBytes) << "\n";
        std::cout << "packed_mb_kept " << mb(occ_kept * kSlotBytes) << "\n";
        std::cout << "reserved_mb_all " << mb(reserved_one * static_cast<std::uint64_t>(n_models + 1)) << "\n";
        std::cout << "reserved_mb_kept " << mb(reserved_one * static_cast<std::uint64_t>(kept.size())) << "\n";
        std::cout << "hash_drops_all " << drops_all << " hash_drops_order4 " << (g4_tot.drops + g4_jnt.drops)
                  << "\n";
        std::cout << "sample_over_packed_kept "
                  << (static_cast<double>(bytes.size()) / static_cast<double>((occ_kept * kSlotBytes) + 1)) << "\n";
        std::cout << "kept_models";
        for (int id : kept) std::cout << ' ' << models[static_cast<std::size_t>(id)].name();
        std::cout << "\nbest_analyze";
        for (int i = 0; i < 12 && i < n_models; ++i) {
            const Model& m = models[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
            std::cout << ' ' << m.name() << '=' << (m.bits / na);
        }
        std::cout << "\n";
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "seconds " << secs << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vdj_combo_bpc: " << e.what() << "\n";
        return 1;
    }
}
