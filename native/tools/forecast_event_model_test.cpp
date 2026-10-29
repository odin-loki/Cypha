/// Event models over all 842 forecast event ids (FORECAST_EVENT_VOCAB.md): codes, normalisation, scoring, sampling,
/// marks, and Kneser-Ney against a direct count.
#include "cypha/forecast/dispute_data.hpp"
#include "cypha/forecast/event_model.hpp"
#include "cypha/forecast/event_vocabulary.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

int g_checks = 0, g_failed = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failed;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

using namespace cypha::forecast;

void test_codes(const EventVocabulary& vocab) {
  check(vocab.size() == 842, "the vocabulary has 842 ids");
  std::set<std::pair<int, int>> seen;
  bool in_range = true;
  for (std::uint32_t id = 0; id < vocab.size(); ++id) {
    const auto e = event_pair_code(vocab, id);
    seen.insert({e.hi, e.lo});
    if (id >= 2) {
      const auto t = vocab.decode(id);
      in_range = in_range && e.hi >= 2 && e.hi < 44 && e.lo >= 64 && e.lo < 84 &&
                 e.hi == 2 + static_cast<int>(t.theater) * 6 + static_cast<int>(t.level) && e.lo == 63 + t.cameo_root;
    }
  }
  check(seen.size() == vocab.size(), "hp_pair codes are distinct");
  check(in_range, "hp_pair codes carry theater, level and root");
}

void test_utf16() {
  const std::string le("y\0e\0a\0r\0,\0x\0\n\0", 14), be("\0y\0e\0a\0r\0,\0x\0\n", 14);
  check(decode_text_utf8(le) == "year,x\n", "UTF-16 little-endian without a mark decodes");
  check(decode_text_utf8(be) == "year,x\n", "UTF-16 big-endian without a mark decodes");
  check(decode_text_utf8(std::string("\xFF\xFEy\0", 4)) == "y", "UTF-16 with a mark decodes");
  check(decode_text_utf8("\xEF\xBB\xBFyear") == "year", "a UTF-8 mark is dropped");
  check(decode_text_utf8("year,month") == "year,month", "plain text is unchanged");
}

std::vector<std::uint32_t> some_events(const EventVocabulary& vocab, int n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const std::vector<std::uint32_t> pool{22, 23, 213, 300, 360, 450, 481, 575, 700, 716};
  std::vector<std::uint32_t> out;
  for (int i = 0; i < n; ++i) out.push_back(pool[(rng() % 3 == 0) ? (i * 7) % pool.size() : rng() % pool.size()]);
  (void)vocab;
  return out;
}

void test_model(const EventVocabulary& vocab, const std::string& kind) {
  EventModelConfig cfg;
  cfg.kind = kind;
  cfg.hp_table_bits = 16;
  cfg.theater_table_bits = 16;
  auto m = make_event_model(vocab, cfg);
  m->train(some_events(vocab, 300, 3), 1);
  const auto stream = some_events(vocab, 40, 9);
  for (std::size_t i = 0; i < 10; ++i) (void)m->observe(stream[i]);

  const auto p = m->distribution();
  double sum = 0.0;
  for (const auto v : p) sum += v;
  check(p.size() == vocab.size() && std::fabs(sum - 1.0) < 1e-9, kind + ": the distribution sums to 1");
  bool agree = true;
  for (const std::uint32_t id : {22u, 300u, 481u, 716u, 841u, 0u}) {
    agree = agree && std::fabs(m->log2_prob(id) - std::log2(p[id])) < 1e-9;
  }
  check(agree, kind + ": log2_prob agrees with the distribution");
  const double lp = m->log2_prob(450);
  check(std::fabs(m->observe(450) + lp) < 1e-9, kind + ": observe returns -log2 P before reading");

  m->set_learning(false);
  const std::uint64_t digest = m->learned_digest();
  m->mark();
  const auto before = m->distribution();
  std::mt19937 rng(5);
  bool valid = true;
  for (int i = 0; i < 50; ++i) {
    const auto id = m->sample(1.0, rng);
    valid = valid && id < vocab.size();
    (void)m->observe(id);
  }
  m->rewind();
  const auto after = m->distribution();
  double diff = 0.0;
  for (std::size_t i = 0; i < after.size(); ++i) diff = std::max(diff, std::fabs(after[i] - before[i]));
  check(valid, kind + ": samples are valid ids");
  check(diff < 1e-12, kind + ": rewind restores the distribution");
  check(m->learned_digest() == digest, kind + ": reading with learning off teaches nothing");
  m->unmark();

  bool raw_ok = true;
  for (int i = 0; i < 5; ++i) raw_ok = raw_ok && (m->raw_sample_valid(rng) || kind != "ngram");
  check(raw_ok, kind + ": an id model's raw draws always decode");
}

// Kneser-Ney order 2 after training equals the same formula on counts taken directly from the sequence.
void test_kneser_ney(const EventVocabulary& vocab) {
  EventModelConfig cfg;
  cfg.kind = "ngram";
  cfg.ngram_order = 2;
  auto m = make_event_model(vocab, cfg);
  const std::vector<std::uint32_t> seq{5, 6, 5, 6, 5, 7, 8, 5, 6};
  m->train(seq, 1);
  // Bigram counts with a PAD start; continuation counts: distinct left contexts per word.
  std::map<std::pair<std::uint32_t, std::uint32_t>, int> c2;
  std::uint32_t prev = 0;
  for (const auto w : seq) {
    ++c2[{prev, w}];
    prev = w;
  }
  std::map<std::uint32_t, int> c1;
  for (const auto& [k, v] : c2) {
    (void)v;
    ++c1[k.second];
  }
  const double d = 0.75, V = static_cast<double>(vocab.size());
  double t1 = 0.0;
  for (const auto& [w, v] : c1) t1 += v;
  auto p1 = [&](std::uint32_t w) {
    const double cw = c1.count(w) ? c1[w] : 0.0;
    return (std::max(cw - d, 0.0) + d * static_cast<double>(c1.size()) / V) / t1;
  };
  // A fresh stream: the context is PAD.
  double t2 = 0.0, types2 = 0.0;
  for (const auto& [k, v] : c2) {
    if (k.first == 0) {
      t2 += v;
      types2 += 1.0;
    }
  }
  bool ok = true;
  for (const std::uint32_t w : {5u, 6u, 7u, 8u, 9u, 841u}) {
    const double cw = c2.count({0u, w}) ? c2[{0u, w}] : 0.0;
    const double want = (std::max(cw - d, 0.0) + d * types2 * p1(w)) / t2;
    ok = ok && std::fabs(std::exp2(m->log2_prob(w)) - want) < 1e-12;
  }
  check(ok, "ngram: Kneser-Ney matches the direct count");
}

}  // namespace

int main() {
  const EventVocabulary vocab;
  test_codes(vocab);
  test_utf16();
  for (const char* kind : {"hp_pair", "hp_base256", "hp_theater", "ngram"}) test_model(vocab, kind);
  test_kneser_ney(vocab);
  std::printf("forecast_event_model_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
