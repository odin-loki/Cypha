/// FCAST: the forecast's event models on GDELT event windows (docs/reports/FORECAST_EVENT_VOCAB.md).
///
///   forecast_vocab_bench --kind K [--order N] [--hp-bits B] [--theater-bits B] --train A.csv,B.csv,...
///                        --dev C.csv,... [--test D.csv,... --read-test] --out PREFIX [--every 20] [--samples 10000]
///
/// Trains on --train, then reads --dev with learning on: scored, or (with --read-test) unscored before scoring
/// --test. Writes PREFIX.loss (-log2 P per scored event), PREFIX.sub.tsv (every --every'th scored event: P(true),
/// top-1 id and probability, Brier over all ids, raw draw validity) and one summary line on stdout.
#include "cypha/forecast/dispute_data.hpp"
#include "cypha/forecast/event_model.hpp"
#include "cypha/forecast/event_vocabulary.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace cypha::forecast;

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (!part.empty()) out.push_back(part);
  }
  return out;
}

std::vector<std::uint32_t> load(const std::vector<std::string>& files, const EventVocabulary& vocab) {
  std::vector<GdeltEvent> all;
  for (const auto& f : files) {
    auto ev = load_gdelt_csv(f);
    all.insert(all.end(), ev.begin(), ev.end());
  }
  return gdelt_to_token_sequence(all, vocab);
}

long peak_rss_kb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmHWM:", 0) == 0) return std::atol(line.c_str() + 6);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  EventModelConfig cfg;
  std::vector<std::string> train_f, dev_f, test_f;
  std::string out = "fcast";
  bool read_test = false;
  int every = 20, n_samples = 10000;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--kind") cfg.kind = next();
    else if (a == "--order") cfg.ngram_order = std::atoi(next().c_str());
    else if (a == "--hp-bits") cfg.hp_table_bits = std::atoi(next().c_str());
    else if (a == "--theater-bits") cfg.theater_table_bits = std::atoi(next().c_str());
    else if (a == "--train") train_f = split(next());
    else if (a == "--dev") dev_f = split(next());
    else if (a == "--test") test_f = split(next());
    else if (a == "--read-test") read_test = true;
    else if (a == "--out") out = next();
    else if (a == "--every") every = std::max(1, std::atoi(next().c_str()));
    else if (a == "--samples") n_samples = std::atoi(next().c_str());
    else {
      std::cerr << "forecast_vocab_bench: unknown argument " << a << "\n";
      return 2;
    }
  }
  if (train_f.empty() || dev_f.empty() || (read_test && test_f.empty())) {
    std::cerr << "forecast_vocab_bench: --train and --dev are needed (and --test with --read-test)\n";
    return 2;
  }
  const EventVocabulary vocab;
  const auto train = load(train_f, vocab), dev = load(dev_f, vocab);
  const auto test = read_test ? load(test_f, vocab) : std::vector<std::uint32_t>{};
  auto model = make_event_model(vocab, cfg);
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  model->train(train, 1);
  const auto t1 = clock::now();
  model->set_learning(true);
  if (read_test) {
    for (const auto id : dev) (void)model->observe(id);
  }
  const auto& scored = read_test ? test : dev;
  std::ofstream loss(out + ".loss"), sub(out + ".sub.tsv");
  sub << "pos\tp_true\ttop1\tp_top1\tcorrect\tbrier\traw_valid\n";
  std::mt19937 rng(20261029);
  const auto t2 = clock::now();
  double bits = 0.0, full_s = 0.0;
  for (std::size_t i = 0; i < scored.size(); ++i) {
    const auto y = scored[i];
    if (i % static_cast<std::size_t>(every) == 0) {
      const auto f0 = clock::now();
      const auto p = model->distribution();
      std::size_t top = 0;
      double brier = 0.0;
      for (std::size_t k = 0; k < p.size(); ++k) {
        if (p[k] > p[top]) top = k;
        brier += p[k] * p[k];
      }
      brier += 1.0 - 2.0 * p[y];
      const bool raw = model->raw_sample_valid(rng);
      sub << i << '\t' << p[y] << '\t' << top << '\t' << p[top] << '\t' << (top == y ? 1 : 0) << '\t' << brier << '\t'
          << (raw ? 1 : 0) << '\n';
      full_s += std::chrono::duration<double>(clock::now() - f0).count();
    }
    const double b = model->observe(y);
    bits += b;
    loss << b << '\n';
  }
  const auto t3 = clock::now();
  model->set_learning(false);
  int invalid = 0;
  for (int i = 0; i < n_samples; ++i) {
    if (model->sample(1.0, rng) >= vocab.size()) ++invalid;
  }
  const double train_s = std::chrono::duration<double>(t1 - t0).count();
  const double score_s = std::chrono::duration<double>(t3 - t2).count() - full_s;
  std::printf(
      "FCAST %s kind=%s order=%d hp_bits=%d n_train=%zu n_dev=%zu n_test=%zu scored=%s n_scored=%zu "
      "bits_per_event=%.5f train_s=%.2f us_per_event_train=%.1f score_s=%.2f us_per_event_score=%.1f full_s=%.2f "
      "masked_invalid=%d/%d peak_rss_mb=%.1f\n",
      model->name().c_str(), cfg.kind.c_str(), cfg.ngram_order, cfg.hp_table_bits, train.size(), dev.size(),
      test.size(), read_test ? "test" : "dev", scored.size(), scored.empty() ? 0.0 : bits / scored.size(), train_s,
      train.empty() ? 0.0 : 1e6 * train_s / train.size(), score_s,
      scored.empty() ? 0.0 : 1e6 * score_s / scored.size(), full_s, invalid, n_samples, peak_rss_kb() / 1024.0);
  return 0;
}
