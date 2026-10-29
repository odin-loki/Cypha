#pragma once

#include "cypha/forecast/event_vocabulary.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cypha::forecast {

/// A next-event model over the whole EventVocabulary (842 ids). Every distribution is over valid ids only
/// (renormalised), so a sampled event always decodes. The forecast's sequence model and its rollouts use this
/// interface (docs/reports/FORECAST_EVENT_VOCAB.md: hp reads bytes, so ids of 256 or more threw before).
class EventModel {
 public:
  virtual ~EventModel() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  /// Train from scratch on ``seq`` (``epochs`` passes, learning on), then start a fresh stream on what was learned.
  virtual void train(const std::vector<std::uint32_t>& seq, int epochs) = 0;
  /// A fresh stream (empty history) on what was learned.
  virtual void begin_stream() = 0;
  /// log2 P(next = id | stream so far). Does not advance.
  [[nodiscard]] virtual double log2_prob(std::uint32_t id) = 0;
  /// P(next = ·) over all ids (0 for an invalid id), summing to 1. Does not advance.
  [[nodiscard]] virtual std::vector<double> distribution() = 0;
  /// A sample from P(next)^(1 / temperature), over valid ids. Does not advance.
  [[nodiscard]] virtual std::uint32_t sample(double temperature, std::mt19937& rng) = 0;
  /// Read ``id`` into the stream, learning from it while learning is on. Returns -log2 P(id) before reading.
  virtual double observe(std::uint32_t id) = 0;
  virtual void set_learning(bool on) = 0;
  [[nodiscard]] virtual bool learning() const = 0;
  /// Remember the current state; ``rewind`` goes back to it as often as asked; ``unmark`` forgets it. Learning
  /// must be off from ``mark`` to the last ``rewind``.
  virtual void mark() = 0;
  virtual void rewind() = 0;
  virtual void unmark() = 0;
  /// A digest of what was learned (tables or counts), to check that reading with learning off taught nothing.
  [[nodiscard]] virtual std::uint64_t learned_digest() const = 0;
  /// Raw draws for the validity read: a draw from the model's own symbol distributions without the mask; true if
  /// it decodes to a valid id. Models over ids always decode. Does not advance.
  [[nodiscard]] virtual bool raw_sample_valid(std::mt19937& rng) {
    (void)rng;
    return true;
  }
};

struct EventModelConfig {
  /// hp_pair | hp_base256 | hp_theater | ngram (FORECAST_EVENT_VOCAB.md, "Candidates").
  std::string kind{"hp_pair"};
  /// hp models: table bits (0 = the production recipe's own) and the per-theater models' bits.
  int hp_table_bits{0};
  int theater_table_bits{16};
  /// ngram: order and absolute discount (interpolated Kneser-Ney, uniform base over the vocabulary).
  int ngram_order{3};
  double ngram_discount{0.75};
  std::uint64_t seed{42};
};

/// The model ``cfg.kind`` names over ``vocab``. Throws on an unknown kind.
[[nodiscard]] std::unique_ptr<EventModel> make_event_model(const EventVocabulary& vocab, const EventModelConfig& cfg);

/// hp_pair's two bytes for an id: hi = 2 + theater * 6 + level (0, 1 for PAD, UNK), lo = 64 + CAMEO root - 1
/// (64 for PAD and UNK).
struct EventBytePair {
  std::uint8_t hi{0};
  std::uint8_t lo{0};
};
[[nodiscard]] EventBytePair event_pair_code(const EventVocabulary& vocab, std::uint32_t id);

}  // namespace cypha::forecast
