#include "cypha/cyphalm/cyphalm_batch.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cypha {
namespace cyphalm {

namespace {

int thread_workers() {
  unsigned n = std::thread::hardware_concurrency();
  return n ? static_cast<int>(n) : 4;
}

template <class F>
void parallel_batch(int batch_size, F&& f) {
  if (batch_size <= 0) {
    return;
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
  for (int b = 0; b < batch_size; ++b) {
    f(b);
  }
#else
  const int nt = std::min(thread_workers(), batch_size);
  if (nt <= 1) {
    for (int b = 0; b < batch_size; ++b) {
      f(b);
    }
    return;
  }
  const int chunk = (batch_size + nt - 1) / nt;
  std::vector<std::thread> th;
  th.reserve(static_cast<std::size_t>(nt));
  for (int t = 0; t < nt; ++t) {
    const int lo = t * chunk;
    const int hi = std::min(batch_size, lo + chunk);
    if (lo >= hi) {
      break;
    }
    th.emplace_back([lo, hi, &f]() {
      for (int b = lo; b < hi; ++b) {
        f(b);
      }
    });
  }
  for (auto& x : th) {
    x.join();
  }
#endif
}

}  // namespace

CyphaLMBatch::CyphaLMBatch(CyphaLMBatchConfig config) : cfg(std::move(config)) {
  lstm = CharLSTMHead(cfg.vocab_size, cfg.lstm_hidden, cfg.seed);
  gria = GRIALowRank(cfg.field_dim, cfg.vocab_size, cfg.gria_rank, 0.5, true, cfg.seed + 1);
}

double CyphaLMBatch::train_sequence_batch(const int* token_ids, const int* next_ids, const double* field_vecs,
                                          int batch_size, int bptt_len) {
  if (batch_size <= 0 || bptt_len <= 0) {
    return 0.0;
  }
  h_states_.assign(static_cast<std::size_t>(batch_size), std::vector<double>(static_cast<std::size_t>(cfg.lstm_hidden), 0.0));
  c_states_.assign(static_cast<std::size_t>(batch_size), std::vector<double>(static_cast<std::size_t>(cfg.lstm_hidden), 0.0));

  std::vector<double> batch_loss(static_cast<std::size_t>(batch_size), 0.0);
  const int vocab = cfg.vocab_size;
  const int fd = cfg.field_dim;

  parallel_batch(batch_size, [&](int b) {
    std::vector<double> log_l(static_cast<std::size_t>(vocab));
    std::vector<double> log_g(static_cast<std::size_t>(vocab));
    std::vector<double> log_blend(static_cast<std::size_t>(vocab));
    CharLSTMCache cache;
    double local = 0.0;

    for (int t = 0; t < bptt_len; ++t) {
      const std::size_t seq_idx =
          static_cast<std::size_t>(b) * static_cast<std::size_t>(bptt_len) + static_cast<std::size_t>(t);
      const int tok = token_ids[seq_idx];
      const int tgt = next_ids[seq_idx];
      const double* v = field_vecs + seq_idx * static_cast<std::size_t>(fd);

      std::vector<double> h_new;
      std::vector<double> c_new;
      lstm.forward_step(tok, h_states_[static_cast<std::size_t>(b)].data(), c_states_[static_cast<std::size_t>(b)].data(),
                        log_l.data(), h_new, c_new, &cache);
      gria.forward(v, log_g.data());
      blend_log_probs(log_g.data(), log_l.data(), vocab, cfg.blend_logit, log_blend.data());

      local += -log_blend[static_cast<std::size_t>(tgt)];

      CharLSTMGrad lg = lstm.backward_step(cache, tgt);
      lstm.apply_grads(lg, cfg.lstm_lr);
      GRIALowRankGrad gg = gria.cross_entropy_gradients(v, tgt);
      gria.update_weights(gg, cfg.gria_lr);
      gria.update_alpha(gg, cfg.gria_lr);
      gria.update_bias(gg, cfg.gria_lr);

      h_states_[static_cast<std::size_t>(b)] = std::move(h_new);
      c_states_[static_cast<std::size_t>(b)] = std::move(c_new);
    }
    batch_loss[static_cast<std::size_t>(b)] = local;
  });

  double loss_sum = 0.0;
  for (double x : batch_loss) {
    loss_sum += x;
  }
  return loss_sum / static_cast<double>(batch_size * bptt_len);
}

}  // namespace cyphalm
}  // namespace cypha
