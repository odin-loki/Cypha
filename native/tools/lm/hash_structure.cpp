// Are context hashes structured, or only the bytes underneath them?
// Uses hp's mix64 / hash2. Reads the first 8 MiB of enwik9.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kN = 2'097'152;  // 2 MiB is enough to see the shape
constexpr int kBuckets = 1 << 16;

std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::uint32_t hash2(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint32_t>(mix64(a * 0x100000001B3ull + b));
}

double entropy_from_counts(const std::vector<std::uint32_t>& c, std::uint64_t n) {
    double h = 0;
    for (std::uint32_t x : c) {
        if (!x) continue;
        const double p = static_cast<double>(x) / static_cast<double>(n);
        h -= p * std::log(p);
    }
    return h / std::log(2.0);
}

struct Shape {
    const char* name;
    std::uint64_t events = 0;
    std::uint64_t unique = 0;
    std::uint64_t singletons = 0;
    double hash_max_over_mean = 0;
    double raw_max_over_mean = 0;
    double empty_hash_frac = 0;
    double empty_raw_frac = 0;
};

template <typename KeyFn>
Shape analyze(const char* name, const std::uint8_t* d, std::size_t n, int order, KeyFn key_at) {
    std::unordered_map<std::uint64_t, std::uint32_t> seen;
    seen.reserve(1 << 20);
    std::vector<std::uint32_t> hash_bins(kBuckets), raw_bins(kBuckets);
    for (std::size_t i = static_cast<std::size_t>(order); i < n; ++i) {
        const std::uint64_t key = key_at(d, i, order);
        auto [it, inserted] = seen.emplace(key, 0);
        ++it->second;
        hash_bins[hash2(0xC0FFEEull + order, key) & (kBuckets - 1)]++;
        raw_bins[static_cast<std::uint32_t>(key) & (kBuckets - 1)]++;
    }
    Shape s;
    s.name = name;
    s.events = n - static_cast<std::size_t>(order);
    s.unique = seen.size();
    for (const auto& kv : seen)
        if (kv.second == 1) ++s.singletons;
    auto ratio = [](const std::vector<std::uint32_t>& b, std::uint64_t events, double& empty) {
        std::uint32_t mx = 0;
        std::uint64_t z = 0;
        for (std::uint32_t x : b) {
            mx = std::max(mx, x);
            if (!x) ++z;
        }
        empty = static_cast<double>(z) / static_cast<double>(b.size());
        const double mean = static_cast<double>(events) / static_cast<double>(b.size());
        return static_cast<double>(mx) / mean;
    };
    s.hash_max_over_mean = ratio(hash_bins, s.events, s.empty_hash_frac);
    s.raw_max_over_mean = ratio(raw_bins, s.events, s.empty_raw_frac);
    return s;
}

}  // namespace

int main() {
    std::ifstream in("/mnt/d/cypha_llm/enwik9", std::ios::binary);
    std::vector<std::uint8_t> d(kN);
    in.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(kN));
    if (static_cast<std::size_t>(in.gcount()) < kN) return 1;

    std::vector<std::uint32_t> freq(256);
    std::uint64_t runs = 0, run_ge4 = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        ++freq[d[i]];
        if (i == 0 || d[i] != d[i - 1]) ++runs;
    }
    std::size_t i = 0;
    while (i < d.size()) {
        std::size_t j = i;
        while (j < d.size() && d[j] == d[i]) ++j;
        if (j - i >= 4) ++run_ge4;
        i = j;
    }
    std::vector<std::pair<int, std::uint32_t>> top;
    for (int b = 0; b < 256; ++b) top.push_back({b, freq[b]});
    std::sort(top.begin(), top.end(), [](auto& a, auto& c) { return a.second > c.second; });

    std::cout << "bytes " << kN << "\n";
    std::cout << "byte_entropy_bits " << entropy_from_counts(freq, kN) << " uniform_would_be 8\n";
    std::cout << "runs " << runs << " runs_of_4_or_more " << run_ge4 << "\n";
    std::cout << "top_bytes";
    for (int k = 0; k < 8; ++k) {
        const int b = top[k].first;
        const char c = (b >= 32 && b < 127) ? static_cast<char>(b) : '?';
        std::cout << ' ' << c << '=' << top[k].second;
    }
    std::cout << "\n";

    auto pack = [](const std::uint8_t* d, std::size_t i, int order) {
        std::uint64_t key = 0;
        for (int k = order; k >= 1; --k) key = (key << 8) | d[i - static_cast<std::size_t>(k)];
        return key;
    };
    // Sparse mask: bytes at lag 1 and lag 3 only. Same family as the mixer's sparse contexts.
    auto sparse = [](const std::uint8_t* d, std::size_t i, int) {
        return (std::uint64_t(d[i - 1]) << 8) | d[i - 3];
    };

    const Shape rows[] = {
        analyze("order1", d.data(), d.size(), 1, pack),
        analyze("order2", d.data(), d.size(), 2, pack),
        analyze("order3", d.data(), d.size(), 3, pack),
        analyze("order4", d.data(), d.size(), 4, pack),
        analyze("sparse_lag1_3", d.data(), d.size(), 3, sparse),
    };
    std::cout << "name unique singletons singleton_frac hash_max/mean raw_max/mean empty_hash empty_raw\n";
    for (const Shape& s : rows) {
        std::cout << s.name << ' ' << s.unique << ' ' << s.singletons << ' '
                  << (static_cast<double>(s.singletons) / static_cast<double>(s.unique)) << ' '
                  << s.hash_max_over_mean << ' ' << s.raw_max_over_mean << ' ' << s.empty_hash_frac << ' '
                  << s.empty_raw_frac << "\n";
    }

    // How many order-3 contexts hang off each order-2 prefix.
    std::unordered_map<std::uint32_t, std::uint32_t> kids;
    for (std::size_t j = 3; j < d.size(); ++j) {
        const std::uint32_t prefix = (std::uint32_t(d[j - 3]) << 8) | d[j - 2];
        kids[prefix]++;  // counts events; recount unique below via a set of full keys per... too big.
    }
    // Unique children: map prefix -> set is heavy. Count unique (prefix, last) pairs already in order3.
    // Branching = unique order3 / unique order2.
    std::unordered_map<std::uint32_t, std::uint32_t> uniq_child;
    std::unordered_map<std::uint64_t, char> once;
    for (std::size_t j = 3; j < d.size(); ++j) {
        const std::uint64_t full = pack(d.data(), j, 3);
        if (!once.emplace(full, 1).second) continue;
        const std::uint32_t prefix = (std::uint32_t(d[j - 3]) << 8) | d[j - 2];
        uniq_child[prefix]++;
    }
    std::uint64_t sum_kids = 0, max_kids = 0, prefixes_with_one = 0;
    for (const auto& kv : uniq_child) {
        sum_kids += kv.second;
        max_kids = std::max<std::uint64_t>(max_kids, kv.second);
        if (kv.second == 1) ++prefixes_with_one;
    }
    std::cout << "order2_prefixes " << uniq_child.size() << " mean_order3_children "
              << (static_cast<double>(sum_kids) / static_cast<double>(uniq_child.size())) << " max_children "
              << max_kids << " prefixes_with_one_child " << prefixes_with_one << "\n";
    return 0;
}
