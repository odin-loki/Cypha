#include "cypha/forecast/event_model.hpp"

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace cypha::forecast {
namespace {

constexpr int kLevels = static_cast<int>(EscalationLevel::Count);
constexpr int kTheaters = static_cast<int>(Theater::Count);

// Sample an index from weights w^(1/T) over ``valid`` (temperature <= 0: the largest).
std::size_t sample_index(const std::vector<double>& w, const std::vector<std::uint32_t>& valid, double temperature,
                         std::mt19937& rng) {
  if (valid.empty()) {
    throw std::logic_error("event model: nothing valid to sample");
  }
  if (temperature <= 0.0) {
    std::size_t best = valid[0];
    for (const auto v : valid) {
      if (w[v] > w[best]) best = v;
    }
    return best;
  }
  double sum = 0.0;
  std::vector<double> x(valid.size());
  for (std::size_t i = 0; i < valid.size(); ++i) {
    x[i] = w[valid[i]] > 0.0 ? std::pow(w[valid[i]], 1.0 / temperature) : 0.0;
    sum += x[i];
  }
  if (!(sum > 0.0)) {
    return valid[std::uniform_int_distribution<std::size_t>(0, valid.size() - 1)(rng)];
  }
  double draw = std::uniform_real_distribution<double>(0.0, sum)(rng);
  for (std::size_t i = 0; i < valid.size(); ++i) {
    draw -= x[i];
    if (draw <= 0.0) return valid[i];
  }
  return valid.back();
}

double masked_prob(const std::vector<double>& p, const std::vector<std::uint32_t>& valid, std::uint32_t b) {
  double sum = 0.0;
  for (const auto v : valid) sum += p[v];
  return sum > 0.0 ? p[b] / sum : 1.0 / static_cast<double>(valid.size());
}

std::uint64_t mix64(std::uint64_t h, std::uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
  return h;
}

// ---------------------------------------------------------------------------------------------------------------
// One hp model reading bytes (CyphaLM's production recipe at the given table bits).
class HpUnit {
 public:
  HpUnit(int table_bits, std::uint64_t seed) {
    cyphalm::CyphaLMConfig c;
    c.vocab_size = 256;
    c.seed = seed;
    if (table_bits >= 16) c.hp_table_bits = table_bits;
    cyphalm::apply_hybrid_production_recipe(c);
    model_ = std::make_unique<cyphalm::CyphaLMModel>(c);
  }
  cyphalm::HpSequenceBackend& hp() { return model_->hp_backend(); }
  const cyphalm::HpSequenceBackend& hp() const { return model_->hp_backend(); }
  void cold() { model_->reset_context(); }
  void begin_stream() { model_->reset_stream(false); }
  // P(next byte) for bytes 0 .. n - 1. hp's bit-tree fan-out skips subtrees past n, and gives the same value for
  // every byte below n as the full fan-out (exactly; a composite model mixes only full distributions, so it always
  // scores all 256).
  std::vector<double> probs(int n = 256) {
    auto lp = hp().next_byte_log_probs(hp().is_composite() ? 256 : n);
    for (auto& v : lp) v = std::exp(v);
    return lp;
  }
  void observe(std::uint8_t b) { (void)hp().observe_next_byte(b); }
  std::uint64_t digest() const { return model_->hp_backend().predictor().learned_digest(); }

 private:
  std::unique_ptr<cyphalm::CyphaLMModel> model_;
};

// A point in one or more hp streams to come back to: hp's undo frame (as beam decoding and the old rollouts use)
// plus the session cache and neural states each backend keeps beside its predictors.
class HpPoint {
 public:
  explicit HpPoint(const std::vector<HpUnit*>& units) {
    std::vector<hp::Predictor*> ps;
    for (HpUnit* u : units) {
      auto& b = u->hp();
      backends_.push_back(&b);
      sessions_.push_back(b.session_size());
      neural_.push_back(b.has_neural() ? b.neural_states() : std::vector<cyphalm::ByteNeuralExpert::State>{});
      for (hp::Predictor* p : b.all_predictors()) ps.push_back(p);
    }
    rewind_.emplace(ps);
  }
  void rewind() {
    rewind_->rewind();
    for (std::size_t i = 0; i < backends_.size(); ++i) {
      if (backends_[i]->has_neural()) backends_[i]->set_neural_states(neural_[i]);
      backends_[i]->truncate_session(sessions_[i]);
      backends_[i]->invalidate_scoring_cache();
    }
  }

 private:
  std::vector<cyphalm::HpSequenceBackend*> backends_;
  std::vector<std::size_t> sessions_;
  std::vector<std::vector<cyphalm::ByteNeuralExpert::State>> neural_;
  std::optional<hp::StreamRewind> rewind_;
};

// ---------------------------------------------------------------------------------------------------------------
// Two bytes per event: hp_pair (by meaning) and hp_base256 (by number).
struct PairCode {
  std::vector<EventBytePair> of_id;
  std::vector<std::uint32_t> valid_hi;
  std::vector<std::vector<std::uint32_t>> valid_lo;  // by hi
  std::vector<std::vector<std::int32_t>> id_of;      // [hi][lo] -> id or -1
};

PairCode make_pair_code(const EventVocabulary& vocab, bool by_meaning) {
  PairCode c;
  const std::size_t n = vocab.size();
  c.of_id.resize(n);
  c.valid_lo.assign(256, {});
  c.id_of.assign(256, std::vector<std::int32_t>(256, -1));
  for (std::uint32_t id = 0; id < n; ++id) {
    EventBytePair e;
    if (by_meaning) {
      e = event_pair_code(vocab, id);
    } else {
      e.hi = static_cast<std::uint8_t>(id >> 8);
      e.lo = static_cast<std::uint8_t>(id & 0xFF);
    }
    c.of_id[id] = e;
    if (c.valid_lo[e.hi].empty()) c.valid_hi.push_back(e.hi);
    c.valid_lo[e.hi].push_back(e.lo);
    if (c.id_of[e.hi][e.lo] != -1) throw std::logic_error("event pair code: two ids share a code");
    c.id_of[e.hi][e.lo] = static_cast<std::int32_t>(id);
  }
  std::sort(c.valid_hi.begin(), c.valid_hi.end());
  for (auto& v : c.valid_lo) std::sort(v.begin(), v.end());
  return c;
}

class HpPairModel final : public EventModel {
 public:
  HpPairModel(const EventVocabulary& vocab, const EventModelConfig& cfg, bool by_meaning)
      : name_(by_meaning ? "hp_pair" : "hp_base256"), code_(make_pair_code(vocab, by_meaning)),
        unit_(cfg.hp_table_bits, cfg.seed), n_(vocab.size()) {
    hi_n_ = static_cast<int>(code_.valid_hi.back()) + 1;
    lo_n_.assign(256, 0);
    for (const auto h : code_.valid_hi) lo_n_[h] = static_cast<int>(code_.valid_lo[h].back()) + 1;
  }

  std::string name() const override { return name_; }

  void train(const std::vector<std::uint32_t>& seq, int epochs) override {
    unmark();
    unit_.cold();
    unit_.hp().set_learning(true);
    for (int ep = 0; ep < std::max(1, epochs); ++ep) {
      if (ep > 0) unit_.begin_stream();
      for (const auto id : seq) {
        const auto& e = code_.of_id.at(id);
        unit_.observe(e.hi);
        unit_.observe(e.lo);
      }
    }
    unit_.begin_stream();
  }
  void begin_stream() override {
    unmark();
    unit_.begin_stream();
  }

  double log2_prob(std::uint32_t id) override {
    const auto& e = code_.of_id.at(id);
    const double ph = masked_prob(unit_.probs(hi_n_), code_.valid_hi, e.hi);
    HpPoint here({&unit_});
    unit_.observe(e.hi);
    const double pl = masked_prob(unit_.probs(lo_n_[e.hi]), code_.valid_lo[e.hi], e.lo);
    here.rewind();
    return std::log2(ph) + std::log2(pl);
  }

  std::vector<double> distribution() override {
    std::vector<double> out(n_, 0.0);
    const auto p = unit_.probs(hi_n_);
    double sum = 0.0;
    for (const auto h : code_.valid_hi) sum += p[h];
    HpPoint here({&unit_});
    for (const auto h : code_.valid_hi) {
      const double ph = sum > 0.0 ? p[h] / sum : 1.0 / static_cast<double>(code_.valid_hi.size());
      if (code_.valid_lo[h].size() == 1) {
        out[static_cast<std::size_t>(code_.id_of[h][code_.valid_lo[h][0]])] = ph;
        continue;
      }
      unit_.observe(static_cast<std::uint8_t>(h));
      const auto q = unit_.probs(lo_n_[h]);
      here.rewind();
      double qs = 0.0;
      for (const auto l : code_.valid_lo[h]) qs += q[l];
      for (const auto l : code_.valid_lo[h]) {
        const double pl = qs > 0.0 ? q[l] / qs : 1.0 / static_cast<double>(code_.valid_lo[h].size());
        out[static_cast<std::size_t>(code_.id_of[h][l])] = ph * pl;
      }
    }
    return out;
  }

  std::uint32_t sample(double temperature, std::mt19937& rng) override {
    const auto h = static_cast<std::uint32_t>(sample_index(unit_.probs(hi_n_), code_.valid_hi, temperature, rng));
    if (code_.valid_lo[h].size() == 1) {
      return static_cast<std::uint32_t>(code_.id_of[h][code_.valid_lo[h][0]]);
    }
    HpPoint here({&unit_});
    unit_.observe(static_cast<std::uint8_t>(h));
    const auto l = sample_index(unit_.probs(lo_n_[h]), code_.valid_lo[h], temperature, rng);
    here.rewind();
    return static_cast<std::uint32_t>(code_.id_of[h][l]);
  }

  bool raw_sample_valid(std::mt19937& rng) override {
    static const std::vector<std::uint32_t> all = [] {
      std::vector<std::uint32_t> v(256);
      for (std::uint32_t i = 0; i < 256; ++i) v[i] = i;
      return v;
    }();
    const auto h = sample_index(unit_.probs(), all, 1.0, rng);
    if (code_.valid_lo[h].empty()) return false;
    HpPoint here({&unit_});
    unit_.observe(static_cast<std::uint8_t>(h));
    const auto l = sample_index(unit_.probs(), all, 1.0, rng);
    here.rewind();
    return std::binary_search(code_.valid_lo[h].begin(), code_.valid_lo[h].end(), static_cast<std::uint32_t>(l));
  }

  double observe(std::uint32_t id) override {
    const auto& e = code_.of_id.at(id);
    const double ph = masked_prob(unit_.probs(hi_n_), code_.valid_hi, e.hi);
    unit_.observe(e.hi);
    const double pl = masked_prob(unit_.probs(lo_n_[e.hi]), code_.valid_lo[e.hi], e.lo);
    unit_.observe(e.lo);
    return -(std::log2(ph) + std::log2(pl));
  }

  void set_learning(bool on) override { unit_.hp().set_learning(on); }
  bool learning() const override { return unit_.hp().learning(); }
  void mark() override { mark_.emplace(std::vector<HpUnit*>{&unit_}); }
  void rewind() override {
    if (mark_) mark_->rewind();
  }
  void unmark() override { mark_.reset(); }
  std::uint64_t learned_digest() const override { return unit_.digest(); }

 private:
  std::string name_;
  PairCode code_;
  HpUnit unit_;
  std::size_t n_;
  int hi_n_{256};
  std::vector<int> lo_n_;
  std::optional<HpPoint> mark_;
};

// ---------------------------------------------------------------------------------------------------------------
// hp_theater: one hp model reads each event's theater (7 theaters and a symbol for PAD / UNK), one hp model per
// theater reads that theater's own events as its local index.
class HpTheaterModel final : public EventModel {
 public:
  HpTheaterModel(const EventVocabulary& vocab, const EventModelConfig& cfg) : n_(vocab.size()) {
    theater_ = std::make_unique<HpUnit>(cfg.hp_table_bits, cfg.seed);
    for (int t = 0; t < kTheaters; ++t) {
      local_.push_back(std::make_unique<HpUnit>(cfg.theater_table_bits, cfg.seed + 1 + static_cast<std::uint64_t>(t)));
      ids_.push_back(vocab.theater_token_ids(static_cast<Theater>(t)));
      std::vector<std::uint32_t> v(ids_.back().size());
      for (std::uint32_t i = 0; i < v.size(); ++i) v[i] = i;
      valid_local_.push_back(std::move(v));
    }
    sym_.assign(n_, kTheaters);
    local_of_.assign(n_, 0);
    for (std::uint32_t id = 0; id < 2; ++id) local_of_[id] = id;
    for (int t = 0; t < kTheaters; ++t) {
      for (std::uint32_t i = 0; i < ids_[t].size(); ++i) {
        sym_[ids_[t][i]] = static_cast<std::uint32_t>(t);
        local_of_[ids_[t][i]] = i;
      }
    }
    for (std::uint32_t s = 0; s <= static_cast<std::uint32_t>(kTheaters); ++s) valid_sym_.push_back(s);
  }

  std::string name() const override { return "hp_theater"; }

  void train(const std::vector<std::uint32_t>& seq, int epochs) override {
    unmark();
    theater_->cold();
    theater_->hp().set_learning(true);
    for (auto& u : local_) {
      u->cold();
      u->hp().set_learning(true);
    }
    for (int ep = 0; ep < std::max(1, epochs); ++ep) {
      if (ep > 0) streams();
      for (const auto id : seq) {
        const auto s = sym_.at(id);
        theater_->observe(static_cast<std::uint8_t>(s));
        if (s < static_cast<std::uint32_t>(kTheaters)) local_[s]->observe(static_cast<std::uint8_t>(local_of_[id]));
      }
    }
    streams();
  }
  void begin_stream() override {
    unmark();
    streams();
  }

  double log2_prob(std::uint32_t id) override {
    const auto s = sym_.at(id);
    double lp = std::log2(masked_prob(theater_->probs(kTheaters + 1), valid_sym_, s));
    if (s < static_cast<std::uint32_t>(kTheaters)) {
      lp += std::log2(masked_prob(local_[s]->probs(static_cast<int>(valid_local_[s].size())), valid_local_[s], local_of_[id]));
    } else {
      lp -= 1.0;  // PAD or UNK, uniform
    }
    return lp;
  }

  std::vector<double> distribution() override {
    std::vector<double> out(n_, 0.0);
    const auto pt = theater_->probs(kTheaters + 1);
    double sum = 0.0;
    for (const auto s : valid_sym_) sum += pt[s];
    for (const auto s : valid_sym_) {
      const double ps = sum > 0.0 ? pt[s] / sum : 1.0 / static_cast<double>(valid_sym_.size());
      if (s == static_cast<std::uint32_t>(kTheaters)) {
        out[0] += ps * 0.5;
        out[1] += ps * 0.5;
        continue;
      }
      const auto pl = local_[s]->probs(static_cast<int>(valid_local_[s].size()));
      double ls = 0.0;
      for (const auto i : valid_local_[s]) ls += pl[i];
      for (const auto i : valid_local_[s]) {
        out[ids_[s][i]] = ps * (ls > 0.0 ? pl[i] / ls : 1.0 / static_cast<double>(valid_local_[s].size()));
      }
    }
    return out;
  }

  std::uint32_t sample(double temperature, std::mt19937& rng) override {
    const auto s = sample_index(theater_->probs(kTheaters + 1), valid_sym_, temperature, rng);
    if (s == static_cast<std::size_t>(kTheaters)) {
      return static_cast<std::uint32_t>(std::uniform_int_distribution<int>(0, 1)(rng));
    }
    const auto i = sample_index(local_[s]->probs(static_cast<int>(valid_local_[s].size())), valid_local_[s], temperature, rng);
    return ids_[s][i];
  }

  bool raw_sample_valid(std::mt19937& rng) override {
    static const std::vector<std::uint32_t> all = [] {
      std::vector<std::uint32_t> v(256);
      for (std::uint32_t i = 0; i < 256; ++i) v[i] = i;
      return v;
    }();
    const auto s = sample_index(theater_->probs(), all, 1.0, rng);
    if (s > static_cast<std::size_t>(kTheaters)) return false;
    if (s == static_cast<std::size_t>(kTheaters)) return true;
    return sample_index(local_[s]->probs(), all, 1.0, rng) < valid_local_[s].size();
  }

  double observe(std::uint32_t id) override {
    const double lp = log2_prob(id);
    const auto s = sym_.at(id);
    theater_->observe(static_cast<std::uint8_t>(s));
    if (s < static_cast<std::uint32_t>(kTheaters)) local_[s]->observe(static_cast<std::uint8_t>(local_of_[id]));
    return -lp;
  }

  void set_learning(bool on) override {
    theater_->hp().set_learning(on);
    for (auto& u : local_) u->hp().set_learning(on);
  }
  bool learning() const override { return theater_->hp().learning(); }
  void mark() override {
    std::vector<HpUnit*> all{theater_.get()};
    for (auto& u : local_) all.push_back(u.get());
    mark_.emplace(all);
  }
  void rewind() override {
    if (mark_) mark_->rewind();
  }
  void unmark() override { mark_.reset(); }
  std::uint64_t learned_digest() const override {
    std::uint64_t h = theater_->digest();
    for (const auto& u : local_) h = mix64(h, u->digest());
    return h;
  }

 private:
  void streams() {
    theater_->begin_stream();
    for (auto& u : local_) u->begin_stream();
  }

  std::size_t n_;
  std::unique_ptr<HpUnit> theater_;
  std::vector<std::unique_ptr<HpUnit>> local_;
  std::vector<std::vector<std::uint32_t>> ids_;
  std::vector<std::vector<std::uint32_t>> valid_local_;
  std::vector<std::uint32_t> valid_sym_;
  std::vector<std::uint32_t> sym_;
  std::vector<std::uint32_t> local_of_;
  std::optional<HpPoint> mark_;
};

// ---------------------------------------------------------------------------------------------------------------
// ngram: interpolated Kneser-Ney over event ids, counts updated online. The highest order holds raw counts, lower
// orders continuation counts (a new n-gram type adds one continuation to its suffix); the base is uniform over the
// vocabulary. A stream starts with order - 1 PAD ids of history. Order 0 is the uniform baseline.
class NgramModel final : public EventModel {
 public:
  NgramModel(const EventVocabulary& vocab, const EventModelConfig& cfg)
      : n_(vocab.size()), order_(std::max(0, cfg.ngram_order)), d_(cfg.ngram_discount) {
    if (order_ > 6) throw std::invalid_argument("ngram event model: order above 6");
    if (!(d_ > 0.0 && d_ < 1.0)) throw std::invalid_argument("ngram event model: discount outside (0, 1)");
    tables_.resize(static_cast<std::size_t>(order_) + 1);
    begin_stream();
  }

  std::string name() const override { return order_ == 0 ? "uniform" : "ngram" + std::to_string(order_); }

  void train(const std::vector<std::uint32_t>& seq, int epochs) override {
    unmark();
    for (auto& t : tables_) t.clear();
    updates_ = 0;
    hash_ = 0;
    learning_ = true;
    for (int ep = 0; ep < std::max(1, epochs); ++ep) {
      begin_stream();
      for (const auto id : seq) (void)observe(id);
    }
    begin_stream();
  }
  void begin_stream() override {
    unmark();
    hist_.assign(static_cast<std::size_t>(std::max(0, order_ - 1)), EventVocabulary::kPad);
  }

  double log2_prob(std::uint32_t id) override { return std::log2(prob(id)); }

  std::vector<double> distribution() override {
    std::vector<double> p(n_, 1.0 / static_cast<double>(n_));
    for (int k = 1; k <= order_; ++k) {
      const Ctx* c = find(k);
      if (c == nullptr || c->total == 0) continue;
      const double t = static_cast<double>(c->total);
      const double back = d_ * static_cast<double>(c->counts.size()) / t;
      for (auto& v : p) v *= back;
      for (const auto& [w, cnt] : c->counts) p[w] += (static_cast<double>(cnt) - d_) / t;
    }
    return p;
  }

  std::uint32_t sample(double temperature, std::mt19937& rng) override {
    const auto p = distribution();
    static thread_local std::vector<std::uint32_t> all;
    if (all.size() != n_) {
      all.resize(n_);
      for (std::uint32_t i = 0; i < n_; ++i) all[i] = i;
    }
    return static_cast<std::uint32_t>(sample_index(p, all, temperature, rng));
  }

  double observe(std::uint32_t id) override {
    if (id >= n_) throw std::out_of_range("ngram event model: id out of the vocabulary");
    const double lp = log2_prob(id);
    if (learning_) update(id);
    if (!hist_.empty()) {
      hist_.erase(hist_.begin());
      hist_.push_back(id);
    }
    return -lp;
  }

  void set_learning(bool on) override { learning_ = on; }
  bool learning() const override { return learning_; }
  void mark() override { marked_ = hist_; }
  void rewind() override {
    if (marked_) hist_ = *marked_;
  }
  void unmark() override { marked_.reset(); }
  std::uint64_t learned_digest() const override { return mix64(hash_, updates_); }

 private:
  struct Ctx {
    std::uint64_t total{0};
    std::unordered_map<std::uint32_t, std::uint32_t> counts;
  };

  // The context of order k: the last k - 1 ids of the history, 10 bits each.
  std::uint64_t key(int k) const {
    std::uint64_t h = static_cast<std::uint64_t>(k);
    for (int j = 0; j < k - 1; ++j) h = (h << 10) | hist_[hist_.size() - static_cast<std::size_t>(k - 1) + static_cast<std::size_t>(j)];
    return h;
  }
  const Ctx* find(int k) const {
    const auto& t = tables_[static_cast<std::size_t>(k)];
    const auto it = t.find(key(k));
    return it == t.end() ? nullptr : &it->second;
  }
  double prob(std::uint32_t w) const {
    double p = 1.0 / static_cast<double>(n_);
    for (int k = 1; k <= order_; ++k) {
      const Ctx* c = find(k);
      if (c == nullptr || c->total == 0) continue;
      const auto it = c->counts.find(w);
      const double cw = it == c->counts.end() ? 0.0 : static_cast<double>(it->second);
      const double t = static_cast<double>(c->total);
      p = (std::max(cw - d_, 0.0) + d_ * static_cast<double>(c->counts.size()) * p) / t;
    }
    return p;
  }
  void update(std::uint32_t w) {
    for (int k = order_; k >= 1; --k) {
      Ctx& c = tables_[static_cast<std::size_t>(k)][key(k)];
      const bool fresh = c.counts.find(w) == c.counts.end();
      ++c.counts[w];
      ++c.total;
      ++updates_;
      hash_ = mix64(hash_, (key(k) << 10) ^ w);
      if (!fresh) break;  // a seen n-gram adds no continuation below it
    }
  }

  std::size_t n_;
  int order_;
  double d_;
  bool learning_{true};
  std::vector<std::unordered_map<std::uint64_t, Ctx>> tables_;
  std::vector<std::uint32_t> hist_;
  std::optional<std::vector<std::uint32_t>> marked_;
  std::uint64_t updates_{0};
  std::uint64_t hash_{0};
};

}  // namespace

EventBytePair event_pair_code(const EventVocabulary& vocab, std::uint32_t id) {
  if (id >= vocab.size()) throw std::out_of_range("event_pair_code: id out of the vocabulary");
  if (id < 2) return {static_cast<std::uint8_t>(id), 64};
  const EventToken t = vocab.decode(id);
  EventBytePair e;
  e.hi = static_cast<std::uint8_t>(2 + static_cast<int>(t.theater) * kLevels + static_cast<int>(t.level));
  e.lo = static_cast<std::uint8_t>(64 + t.cameo_root - 1);
  return e;
}

std::unique_ptr<EventModel> make_event_model(const EventVocabulary& vocab, const EventModelConfig& cfg) {
  if (cfg.kind == "hp_pair") return std::make_unique<HpPairModel>(vocab, cfg, true);
  if (cfg.kind == "hp_base256") return std::make_unique<HpPairModel>(vocab, cfg, false);
  if (cfg.kind == "hp_theater") return std::make_unique<HpTheaterModel>(vocab, cfg);
  if (cfg.kind == "ngram") return std::make_unique<NgramModel>(vocab, cfg);
  throw std::invalid_argument("make_event_model: unknown kind '" + cfg.kind + "'");
}

}  // namespace cypha::forecast
