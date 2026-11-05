// K5 on more Wikipedia (docs/reports/K5_SCALE.md): the overlap check on 32-byte spans.
//
//   k5_overlap clean DOCS OUT REPORT FORBID... [--info INFO...]
//       DOCS: one document per line. FORBID, INFO: one text per line.
//       Every 32-byte span of every FORBID text is hashed (64-bit FNV-1a). For each document:
//         - the hits are its 32-byte windows whose hash is a forbidden span's;
//         - if hits cover more than 10% of its bytes, the whole document is dropped (a near-copy);
//         - otherwise every sentence that overlaps a hit is cut (sentences end at . ! ? followed by a space),
//           and the document is checked again until it holds no hit.
//       OUT gets the kept documents, one per line. REPORT gets counts per FORBID file and per INFO file (INFO files
//       are only counted: documents of OUT that hold one of their spans).
//   k5_overlap check DOCS FORBID...
//       Prints the number of documents (lines) of DOCS that hold a forbidden span; exit 1 if any.
//
//   --gpu (anywhere): hash the windows and look them up on the GPU (native/cuda/k5_overlap_gpu.cu; docs/reports/OPT_GPU.md,
//       GENERATION_PLAN §13.41 O4). The same maths: OUT, REPORT and stdout are byte-identical to the CPU's.
//       --gpu-chunk MB: text sent to the device at a time (default 64). Only in a build with CYPHA_CUDA.
//   --threads N: the CPU's hit passes on N threads (documents are independent; the same outputs).
//   k5_overlap selftest: the CPU's hits against a byte-by-byte check; threads and (if built) the GPU against the CPU.
//   Phase times go to stderr (stdout and the files are unchanged).
//
// Build: g++ -std=c++20 -O2 -o k5_overlap scripts/k5_overlap.cpp                      (CPU only)
//        cmake -S native -B build -DCYPHA_CUDA=ON && cmake --build build --target k5_overlap   (CPU and --gpu)
// Author: Odin Loch
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef CYPHA_CUDA
#include "k5_overlap_gpu.hpp"
#endif

namespace {
constexpr std::size_t K = 32;

std::uint64_t fnv(const char* p) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < K; ++i) h = (h ^ static_cast<unsigned char>(p[i])) * 1099511628211ull;
    return h;
}

double g_hash_s = 0, g_hit_s = 0, g_read_s = 0;  // seconds in span_set, in the hit passes, reading files
unsigned g_threads = 1;                          // --threads: the CPU's hit passes on this many threads
double secs(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

std::vector<std::string> lines(const std::string& path) {
    const auto t0 = std::chrono::steady_clock::now();
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot read " << path << "\n";
        std::exit(2);
    }
    std::vector<std::string> v;
    for (std::string s; std::getline(in, s);) v.push_back(std::move(s));
    g_read_s += secs(t0);
    return v;
}

#ifdef CYPHA_CUDA
static_assert(cypha::cuda_o4::kK == static_cast<int>(K), "the GPU's window must be K bytes");
std::unique_ptr<cypha::cuda_o4::Overlap> g_gpu;  // set by --gpu
#endif

std::vector<std::uint64_t> span_set_of(const std::vector<std::string>& texts) {
    std::vector<std::uint64_t> h;
    const auto t0 = std::chrono::steady_clock::now();
#ifdef CYPHA_CUDA
    if (g_gpu) {
        h = g_gpu->span_set(std::vector<std::string_view>(texts.begin(), texts.end()));
        g_hash_s += secs(t0);
        return h;
    }
#endif
    for (const auto& t : texts)
        for (std::size_t i = 0; i + K <= t.size(); ++i) h.push_back(fnv(t.data() + i));
    std::sort(h.begin(), h.end());
    h.erase(std::unique(h.begin(), h.end()), h.end());
    g_hash_s += secs(t0);
    return h;
}
std::vector<std::uint64_t> span_set(const std::vector<std::string>& files) {
    std::vector<std::string> texts;
    for (const auto& f : files)
        for (auto& t : lines(f)) texts.push_back(std::move(t));
    return span_set_of(texts);
}

// Start positions of the windows of s whose hash is in set.
std::vector<std::size_t> hit_pos(std::string_view s, const std::vector<std::uint64_t>& set) {
    std::vector<std::size_t> p;
    for (std::size_t i = 0; i + K <= s.size(); ++i)
        if (std::binary_search(set.begin(), set.end(), fnv(s.data() + i))) p.push_back(i);
    return p;
}

// The heavy loop: hit_pos of every document of a batch, on the CPU, or on the GPU with --gpu.
using Hits = std::vector<std::vector<std::size_t>>;
template <class Docs>
Hits hit_all(const Docs& docs, const std::vector<std::uint64_t>& set) {
    const auto t0 = std::chrono::steady_clock::now();
    Hits out(docs.size());
#ifdef CYPHA_CUDA
    if (g_gpu) {
        std::vector<std::string_view> v;
        v.reserve(docs.size());
        for (const auto& d : docs) v.emplace_back(d);
        g_gpu->hits(v, set, out);
        g_hit_s += secs(t0);
        return out;
    }
#endif
    if (g_threads > 1) {  // documents are independent: blocks of 64 to whichever thread is free
        std::atomic<std::size_t> next{0};
        const auto work = [&] {
            for (std::size_t a; (a = next.fetch_add(64)) < docs.size();)
                for (std::size_t i = a; i < std::min(a + 64, docs.size()); ++i) out[i] = hit_pos(docs[i], set);
        };
        std::vector<std::thread> th;
        for (unsigned t = 1; t < g_threads; ++t) th.emplace_back(work);
        work();
        for (auto& t : th) t.join();
    } else {
        for (std::size_t i = 0; i < docs.size(); ++i) out[i] = hit_pos(docs[i], set);
    }
    g_hit_s += secs(t0);
    return out;
}

std::size_t covered(const std::vector<std::size_t>& p) {
    std::size_t c = 0, end = 0;
    for (const std::size_t a : p) {
        const std::size_t b = a + K;
        c += b - std::max(a, end);
        end = b;
    }
    return c;
}

// Sentence ranges [a, b): b is one past the space after . ! ? (or the end).
std::vector<std::pair<std::size_t, std::size_t>> sentences(std::string_view s) {
    std::vector<std::pair<std::size_t, std::size_t>> r;
    std::size_t a = 0;
    for (std::size_t i = 0; i + 1 < s.size(); ++i)
        if ((s[i] == '.' || s[i] == '!' || s[i] == '?') && s[i + 1] == ' ') {
            r.emplace_back(a, i + 2);
            a = i + 2;
        }
    if (a < s.size()) r.emplace_back(a, s.size());
    return r;
}

std::string cut(std::string_view s, const std::vector<std::size_t>& p, std::size_t& cut_sent) {
    std::string out;
    std::size_t j = 0;
    for (const auto& [a, b] : sentences(s)) {
        while (j < p.size() && p[j] + K <= a) ++j;
        const bool hit = j < p.size() && p[j] < b;
        if (hit) {
            ++cut_sent;
            continue;
        }
        out.append(s.substr(a, b - a));
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// Random lines with copied spans: the CPU's hits equal a byte-by-byte check; with --gpu built, the GPU's equal the CPU's.
int selftest() {
    std::uint64_t x = 20261105;
    const auto rnd = [&x] {
        x += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull, z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    const std::string abc = "abcde fgh. ij! kl? mn ";
    const auto text = [&](std::size_t n) {
        std::string s;
        for (std::size_t i = 0; i < n; ++i) s += abc[rnd() % abc.size()];
        return s;
    };
    std::vector<std::string> forbid;
    for (int i = 0; i < 300; ++i) forbid.push_back(text(rnd() % 400));
    std::vector<std::string> docs;
    for (int i = 0; i < 3000; ++i) {
        std::string d = text(rnd() % 600);
        for (int c = static_cast<int>(rnd() % 3); c-- > 0;) {  // plant copies of forbidden spans
            const auto& f = forbid[rnd() % forbid.size()];
            if (f.size() <= 40) continue;
            const std::size_t a = rnd() % (f.size() - 40), n = 32 + rnd() % 9;
            d.insert(d.empty() ? 0 : rnd() % d.size(), f.substr(a, n));
        }
        docs.push_back(d);
    }
    for (const std::size_t n : {0, 1, 31, 32, 33, 64}) docs.push_back(forbid[0].size() >= n ? forbid[0].substr(0, n) : text(n));
    docs.push_back(text(3u << 20));  // longer than a small chunk
    docs.back().insert(1000, forbid[1].substr(0, std::min<std::size_t>(forbid[1].size(), 100)));
    int bad = 0;
    const auto set = span_set_of(forbid);
    std::unordered_set<std::string> spans;
    for (const auto& f : forbid)
        for (std::size_t i = 0; i + K <= f.size(); ++i) spans.insert(f.substr(i, K));
    if (set.size() != spans.size()) std::fprintf(stderr, "FAIL set size %zu vs %zu spans\n", set.size(), spans.size()), ++bad;
    const Hits cpu = hit_all(docs, set);
    std::size_t nh = 0;
    for (std::size_t i = 0; i < docs.size(); ++i) {
        std::vector<std::size_t> want;
        for (std::size_t j = 0; j + K <= docs[i].size(); ++j)
            if (spans.count(docs[i].substr(j, K))) want.push_back(j);
        nh += want.size();
        if (cpu[i] != want) ++bad;
    }
    std::fprintf(stderr, "k5_overlap selftest: cpu, %zu docs, %zu hits, %zu spans: %s\n", docs.size(), nh, set.size(), bad ? "FAILED" : "ok");
    {
        const unsigned t0 = g_threads;
        g_threads = 4;
        const bool same = hit_all(docs, set) == cpu;
        g_threads = t0;
        std::fprintf(stderr, "k5_overlap selftest: cpu, 4 threads, %s\n", same ? "ok" : "FAILED");
        bad += !same;
    }
#ifdef CYPHA_CUDA
    for (const int run : {0, 1, 2}) {  // 1 MB chunks; 64 MB; 64 MB with the span sets sorted on the host
        const std::size_t chunk = run == 0 ? 1 : 64;
        try {
            g_gpu = std::make_unique<cypha::cuda_o4::Overlap>(chunk, run == 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "k5_overlap selftest: gpu skipped (%s)\n", e.what());
            break;
        }
        int gb = 0;
        if (span_set_of(forbid) != set) ++gb;
        const Hits gpu = hit_all(docs, set);
        for (std::size_t i = 0; i < docs.size(); ++i) gb += gpu[i] != cpu[i];
        std::vector<std::string_view> sub;  // a second batch with another set (re-upload) and back
        for (std::size_t i = 0; i < docs.size(); i += 7) sub.emplace_back(docs[i]);
        const auto set2 = span_set_of(std::vector<std::string>(forbid.begin(), forbid.begin() + 50));
        const Hits g2 = hit_all(sub, set2), g3 = hit_all(docs, set);
        for (std::size_t k = 0; k < sub.size(); ++k) gb += g2[k] != hit_pos(sub[k], set2);
        gb += g3 != cpu;
        if ((g_gpu->stats().host_sorts > 0) != (run == 2)) ++gb;
        std::fprintf(stderr, "k5_overlap selftest: gpu (%zu MB chunks%s), %s\n", chunk, run == 2 ? ", host sort" : "", gb ? "FAILED" : "ok");
        bad += gb;
        g_gpu.reset();
    }
#endif
    return bad ? 1 : 0;
}

void report_times(const std::chrono::steady_clock::time_point t0) {
    bool on_gpu = false;
#ifdef CYPHA_CUDA
    on_gpu = g_gpu != nullptr;
#endif
    std::fprintf(stderr, "k5_overlap: %s, total %.1f s: span sets %.1f s, hit passes %.1f s, reading files %.1f s, the rest (cutting, writing) %.1f s\n",
                 on_gpu ? "gpu" : (g_threads > 1 ? ("cpu, " + std::to_string(g_threads) + " threads").c_str() : "cpu, 1 thread"), secs(t0), g_hash_s, g_hit_s,
                 g_read_s, secs(t0) - g_hash_s - g_hit_s - g_read_s);
#ifdef CYPHA_CUDA
    if (g_gpu) {
        const auto& s = g_gpu->stats();
        std::fprintf(stderr, "k5_overlap: gpu %s, %.0f MB free at start, peak %.0f MB held, kernels and device sorts %.0f ms, copies %.0f ms, %zu set uploads, %zu host sorts, %.0f M windows\n",
                     s.device.c_str(), s.free_at_start / 1048576.0, s.peak_bytes / 1048576.0, s.kernel_ms, s.copy_ms, s.set_uploads, s.host_sorts, s.windows / 1e6);
    }
#endif
}
}  // namespace

int main(int argc, char** argv) {
    const auto t_start = std::chrono::steady_clock::now();
    {  // --gpu [--gpu-chunk MB], anywhere
        bool gpu = false;
        std::size_t chunk_mb = 64;
        int w = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--gpu") gpu = true;
            else if (a == "--gpu-chunk" && i + 1 < argc) chunk_mb = std::strtoull(argv[++i], nullptr, 10);
            else if (a == "--threads" && i + 1 < argc) g_threads = static_cast<unsigned>(std::max(1, std::atoi(argv[++i])));
            else argv[w++] = argv[i];
        }
        argc = w;
        if (argc >= 2 && std::string(argv[1]) == "selftest") return selftest();
        if (gpu) {
#ifdef CYPHA_CUDA
            try {
                g_gpu = std::make_unique<cypha::cuda_o4::Overlap>(chunk_mb);
            } catch (const std::exception& e) {
                std::cerr << "--gpu: " << e.what() << "\n";
                return 3;
            }
#else
            (void)chunk_mb;
            std::cerr << "--gpu: this k5_overlap was built without CUDA (cmake -DCYPHA_CUDA=ON)\n";
            return 3;
#endif
        }
    }
    if (argc < 4) {
        std::cerr << "usage: k5_overlap clean DOCS OUT REPORT FORBID... [--info INFO...] | k5_overlap check DOCS FORBID... [--gpu]\n";
        return 2;
    }
    try {
        const std::string mode = argv[1];
        if (mode == "check") {
            const auto docs = lines(argv[2]);
            const auto set = span_set(std::vector<std::string>(argv + 3, argv + argc));
            std::size_t bad = 0;
            for (const auto& p : hit_all(docs, set)) bad += !p.empty();
            std::printf("check %s: %zu documents, %zu hold a forbidden span (%zu spans in the set)\n", argv[2], docs.size(), bad, set.size());
            report_times(t_start);
            return bad ? 1 : 0;
        }
        if (mode != "clean" || argc < 6) return 2;
        const std::string docs_path = argv[2], out_path = argv[3], rep_path = argv[4];
        std::vector<std::string> forbid, info;
        bool in_info = false;
        for (int i = 5; i < argc; ++i) {
            const std::string s = argv[i];
            if (s == "--info") in_info = true;
            else (in_info ? info : forbid).push_back(s);
        }
        auto docs = lines(docs_path);
        std::ofstream rep(rep_path);
        std::vector<std::vector<std::uint64_t>> per;
        for (const auto& f : forbid) per.push_back(span_set({f}));
        const auto all = span_set(forbid);
        std::size_t in_bytes = 0;
        for (const auto& d : docs) in_bytes += d.size() + 1;
        rep << "docs_in " << docs.size() << "\nbytes_in " << in_bytes << "\n";
        for (std::size_t f = 0; f < forbid.size(); ++f) {
            std::size_t nd = 0, ns = 0;
            for (const auto& p : hit_all(docs, per[f])) nd += !p.empty(), ns += p.size();
            rep << "forbid " << forbid[f] << " spans_in_set " << per[f].size() << " docs_hit " << nd << " windows_hit " << ns << "\n";
            rep.flush();
        }
        // Each document's hits; then the cuts, round by round, every touched document of a round in one batch.
        std::size_t dropped = 0, dropped_bytes = 0, touched = 0, cut_sent = 0, cut_bytes = 0, rounds_max = 0, emptied = 0;
        std::vector<char> gone(docs.size(), 0);
        {
            Hits p = hit_all(docs, all);
            std::vector<std::size_t> act, before(docs.size(), 0), rounds(docs.size(), 0);
            for (std::size_t i = 0; i < docs.size(); ++i) {
                if (p[i].empty()) continue;
                if (covered(p[i]) * 10 > docs[i].size()) {
                    ++dropped, dropped_bytes += docs[i].size() + 1, gone[i] = 1;
                    continue;
                }
                ++touched, before[i] = docs[i].size(), act.push_back(i);
            }
            std::vector<std::size_t> touched_ids = act;
            while (!act.empty()) {
                std::vector<std::string_view> v;
                for (const auto i : act) {
                    docs[i] = cut(docs[i], p[i], cut_sent);
                    ++rounds[i];
                }
                for (const auto i : act) v.emplace_back(docs[i]);
                Hits q = hit_all(v, all);
                std::vector<std::size_t> next;
                for (std::size_t k = 0; k < act.size(); ++k)
                    if (!q[k].empty()) p[act[k]] = std::move(q[k]), next.push_back(act[k]);
                act.swap(next);
            }
            for (const auto i : touched_ids) {
                rounds_max = std::max(rounds_max, rounds[i]);
                cut_bytes += before[i] - docs[i].size();
                if (docs[i].empty()) ++emptied, gone[i] = 1;
            }
        }
        std::ofstream out(out_path, std::ios::binary);
        std::size_t kept = 0, out_bytes = 0;
        for (std::size_t i = 0; i < docs.size(); ++i) {
            if (gone[i]) continue;
            out << docs[i] << "\n";
            ++kept, out_bytes += docs[i].size() + 1;
        }
        out.close();
        rep << "docs_dropped_over_10pct " << dropped << "\nbytes_dropped " << dropped_bytes << "\ndocs_with_sentences_cut " << touched
            << "\nsentences_cut " << cut_sent << "\nbytes_cut " << cut_bytes << "\ndocs_emptied_by_cuts " << emptied
            << "\nmax_rounds " << rounds_max << "\ndocs_out " << kept << "\nbytes_out " << out_bytes << "\n";
        // INFO files: counted on the kept documents.
        const auto outd = lines(out_path);
        for (const auto& f : info) {
            const auto set = span_set({f});
            std::size_t nd = 0, ns = 0;
            for (const auto& p : hit_all(outd, set)) nd += !p.empty(), ns += p.size();
            rep << "info " << f << " spans_in_set " << set.size() << " kept_docs_hit " << nd << " windows_hit " << ns << "\n";
        }
        std::size_t bad = 0;
        for (const auto& p : hit_all(outd, all)) bad += !p.empty();
        rep << "recheck_kept_docs_with_forbidden_span " << bad << "\n";
        std::printf("clean: %zu docs in, %zu out; recheck %zu\n", docs.size(), kept, bad);
        report_times(t_start);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "k5_overlap: " << e.what() << "\n";
        return 3;
    }
}
