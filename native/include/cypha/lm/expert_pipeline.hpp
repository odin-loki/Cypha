#pragma once

/// Expert pipelining for context8k (docs/GENERATION_PLAN.md §13.41 O2; docs/reports/OPT_PIPELINE.md).
///
/// In context8k the holdout is known in advance, and every default expert learns only from the true bytes: none reads
/// the mixer's output or state. So each expert's predictions can be computed in its own thread, ahead of the mixer,
/// and handed over. The mixer then reads them in the same order and does the same arithmetic, so the output is the
/// same to the last bit. This header holds the plumbing only; no expert or mixer arithmetic lives here.
///
/// - Task: one job on its own thread (or run in place, when pipelining is off), joined before its results are read.
/// - Progress: a count of rows a producer has finished, which a reader can wait on.
/// - SparseRows: whole 256-byte distributions stored without their +0.0 entries (the similar-context expert's are
///   mostly zero). Every stored value is the computed double, so a row reads back bit for bit.
/// - BlockRing: a bounded ring of blocks of rows. Several producers each fill their own part of a block; one
///   consumer reads the blocks in order. A producer may run at most `slots` blocks ahead of the consumer, so memory
///   stays bounded.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace cypha::cyphalm::pipeline {

/// One job, run on its own thread by start(), or in place by start() when threaded is false (pipelining off: the
/// job then runs at once, on the caller's thread, so the order of work is the old one). wait() joins it and returns
/// its code (0: done); an exception in the job is caught and becomes code 1.
class Task {
 public:
    Task() = default;
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task() { (void)wait(); }

    void start(std::function<int()> f, bool threaded) {
        started_ = true;
        if (!threaded) {
            rc_ = guarded(f);
            return;
        }
        th_ = std::thread([this, f = std::move(f)] { rc_ = guarded(f); });
    }
    /// Joins the job (once); returns its code. A task never started returns 0.
    int wait() {
        if (th_.joinable()) th_.join();
        return rc_;
    }
    bool started() const { return started_; }

 private:
    static int guarded(const std::function<int()>& f) {
        try {
            return f();
        } catch (const std::exception&) {
            return 1;
        }
    }
    std::thread th_;
    int rc_ = 0;
    bool started_ = false;
};

/// Rows finished by one producer, in order. The producer calls advance(n) after row n - 1 is written (a few hundred
/// rows at a time: each call takes a lock); a reader calls wait(n) before it reads rows below n.
class Progress {
 public:
    void advance(std::size_t n) {
        std::lock_guard<std::mutex> g(m_);
        done_.store(n, std::memory_order_release);
        cv_.notify_all();
    }
    /// The producer stopped early (an error): every wait returns false from now on.
    void fail() {
        std::lock_guard<std::mutex> g(m_);
        failed_ = true;
        cv_.notify_all();
    }
    std::size_t done() const { return done_.load(std::memory_order_acquire); }
    /// Waits until at least n rows are done (true) or the producer failed (false); adds the seconds waited to waited.
    bool wait(std::size_t n, double* waited = nullptr) {
        if (done() >= n) return true;
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return failed_ || done() >= n; });
        if (waited) *waited += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return done() >= n;
    }

 private:
    std::atomic<std::size_t> done_{0};
    bool failed_ = false;
    std::mutex m_;
    std::condition_variable cv_;
};

/// A ring of n records between one writer thread and one reader thread, without locks. The writer fills write_slot()
/// and calls commit(); the reader takes read_slot() (nullptr once the writer has closed and the ring is empty) and calls
/// release(). Both wait by spinning, then yielding, then sleeping briefly.
template <class T>
class SpscRing {
 public:
    explicit SpscRing(std::size_t n) : buf_(std::max<std::size_t>(n, 2)) {}
    T& write_slot() {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        for (int spins = 0; h - tail_.load(std::memory_order_acquire) >= buf_.size(); ++spins) backoff(spins, wait_w_);
        return buf_[h % buf_.size()];
    }
    void commit() { head_.store(head_.load(std::memory_order_relaxed) + 1, std::memory_order_release); }
    void close() { closed_.store(true, std::memory_order_release); }
    T* read_slot() {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        for (int spins = 0; head_.load(std::memory_order_acquire) == t; ++spins) {
            if (closed_.load(std::memory_order_acquire) && head_.load(std::memory_order_acquire) == t) return nullptr;
            backoff(spins, wait_r_);
        }
        return &buf_[t % buf_.size()];
    }
    void release() { tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release); }
    double writer_waited() const { return wait_w_; }
    double reader_waited() const { return wait_r_; }

 private:
    static void backoff(int spins, double& waited) {
        if (spins < 2048) return;
        if (spins < 4096) {
            std::this_thread::yield();
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
        waited += 20e-6;
    }
    std::vector<T> buf_;
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
    std::atomic<bool> closed_{false};
    double wait_w_ = 0, wait_r_ = 0;
};

/// Rows of 256 doubles kept without their +0.0 entries (bit pattern 0; a -0.0 or any other value is kept). One
/// writer appends rows in order; readers read any row below the writer's Progress. Storage comes in chunks that never
/// move, and the tables are sized up front, so a reader never sees a reallocation.
class SparseRows {
 public:
    explicit SparseRows(std::size_t rows = 0) { reset(rows); }
    void reset(std::size_t rows) {
        rows_ = rows;
        off_.assign(rows + 1, 0);
        vals_.assign(rows * 256 / kChunk + 2, nullptr);
        idx_.assign(rows * 256 / kChunk + 2, nullptr);
        owned_v_.clear();
        owned_i_.clear();
        n_ = 0;
        k_ = 0;
    }
    /// Appends row k_ (rows come in order 0, 1, ...).
    void push(const double* P) {
        for (int s = 0; s < 256; ++s) {
            std::uint64_t b;
            std::memcpy(&b, &P[s], sizeof b);
            if (b == 0) continue;
            const std::size_t c = n_ / kChunk;
            if (!vals_[c]) {
                owned_v_.push_back(std::make_unique<double[]>(kChunk));
                owned_i_.push_back(std::make_unique<std::uint8_t[]>(kChunk));
                vals_[c] = owned_v_.back().get();
                idx_[c] = owned_i_.back().get();
            }
            vals_[c][n_ % kChunk] = P[s];
            idx_[c][n_ % kChunk] = static_cast<std::uint8_t>(s);
            ++n_;
        }
        off_[++k_] = n_;
    }
    /// Row k into P[256], every entry as it was pushed.
    void get(std::size_t k, double* P) const {
        std::fill(P, P + 256, 0.0);
        for (std::size_t e = off_[k]; e < off_[k + 1]; ++e) P[idx_[e / kChunk][e % kChunk]] = vals_[e / kChunk][e % kChunk];
    }
    std::size_t rows() const { return k_; }
    std::size_t entries() const { return n_; }
    std::size_t bytes() const {
        return off_.size() * sizeof(std::size_t) + owned_v_.size() * kChunk * (sizeof(double) + 1);
    }

 private:
    static constexpr std::size_t kChunk = std::size_t{1} << 16;
    std::size_t rows_ = 0, n_ = 0, k_ = 0;
    std::vector<std::size_t> off_;
    std::vector<double*> vals_;
    std::vector<std::uint8_t*> idx_;
    std::vector<std::unique_ptr<double[]>> owned_v_;
    std::vector<std::unique_ptr<std::uint8_t[]>> owned_i_;
};

/// A bounded ring of `slots` blocks for a stream of blocks 0, 1, 2, ... Each of `roles` producers fills its own part
/// of every block, in block order: acquire(role, b) waits until block b's slot is free (the consumer has released
/// block b - slots), then the producer writes, then publish(role, b). The consumer reads block b after await(b) (every
/// role has published it) and frees it with release(b). cancel() wakes everyone and makes every wait return false.
class BlockRing {
 public:
    BlockRing(std::size_t slots, int roles) : slots_(std::max<std::size_t>(slots, 1)), done_(static_cast<std::size_t>(roles), 0) {}
    std::size_t slots() const { return slots_; }
    std::size_t slot(std::size_t b) const { return b % slots_; }

    bool acquire(int role, std::size_t b) {
        (void)role;
        std::unique_lock<std::mutex> l(m_);
        if (b < released_ + slots_ || stop_) return !stop_;
        const auto t0 = std::chrono::steady_clock::now();
        free_.wait(l, [&] { return stop_ || b < released_ + slots_; });
        wait_prod_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return !stop_;
    }
    void publish(int role, std::size_t b) {
        std::lock_guard<std::mutex> g(m_);
        done_[static_cast<std::size_t>(role)] = b + 1;
        ready_.notify_all();
    }
    bool await(std::size_t b) {
        std::unique_lock<std::mutex> l(m_);
        auto all = [&] {
            for (std::size_t d : done_)
                if (d <= b) return false;
            return true;
        };
        if (all() || stop_) return !stop_;
        const auto t0 = std::chrono::steady_clock::now();
        ready_.wait(l, [&] { return stop_ || all(); });
        wait_cons_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return !stop_;
    }
    void release(std::size_t b) {
        std::lock_guard<std::mutex> g(m_);
        released_ = std::max(released_, b + 1);
        free_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> g(m_);
        stop_ = true;
        free_.notify_all();
        ready_.notify_all();
    }
    double consumer_waited() const { return wait_cons_; }
    double producers_waited() const { return wait_prod_; }

 private:
    std::size_t slots_, released_ = 0;
    std::vector<std::size_t> done_;
    bool stop_ = false;
    double wait_cons_ = 0, wait_prod_ = 0;
    std::mutex m_;
    std::condition_variable free_, ready_;
};

/// f(r0, r1) over [0, n) split into `parts` contiguous ranges, each on its own thread (parts <= 1: in place).
/// Used for passes whose rows are independent, so the split changes nothing but the time.
inline void parallel_rows(std::size_t n, int parts, const std::function<void(std::size_t, std::size_t)>& f) {
    if (parts <= 1 || n < 2) {
        f(0, n);
        return;
    }
    std::vector<std::thread> th;
    for (int t = 0; t < parts; ++t) {
        const std::size_t r0 = n * static_cast<std::size_t>(t) / static_cast<std::size_t>(parts);
        const std::size_t r1 = n * static_cast<std::size_t>(t + 1) / static_cast<std::size_t>(parts);
        th.emplace_back([&f, r0, r1] { f(r0, r1); });
    }
    for (auto& t : th) t.join();
}

/// Independent jobs on up to `threads` threads, each taking the next job not yet taken (threads <= 1: in place, in
/// order). Each job writes only its own results.
inline void parallel_jobs(const std::vector<std::function<void()>>& jobs, int threads) {
    if (threads <= 1 || jobs.size() < 2) {
        for (const auto& j : jobs) j();
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> th;
    const std::size_t n = std::min(jobs.size(), static_cast<std::size_t>(threads));
    for (std::size_t t = 0; t < n; ++t)
        th.emplace_back([&] {
            for (std::size_t j; (j = next.fetch_add(1)) < jobs.size();) jobs[j]();
        });
    for (auto& t : th) t.join();
}

}  // namespace cypha::cyphalm::pipeline
