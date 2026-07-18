#include "cypha/cyphalm/pgm_cell.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace cypha::cyphalm {

namespace {

constexpr int kMaxSlots = 262144;

int ipow(int base, int exp) {
  int r = 1;
  for (int i = 0; i < exp; ++i) {
    if (r > kMaxSlots / std::max(1, base)) {
      return kMaxSlots + 1;
    }
    r *= base;
  }
  return r;
}

}  // namespace

PGMCell::PGMCell(PGMCellConfig cfg) : cfg_(std::move(cfg)) {
  if (cfg_.d_input < 1) {
    throw std::invalid_argument("PGMCell: d_input must be >= 1");
  }
  if (cfg_.n_sub < 2) {
    throw std::invalid_argument("PGMCell: n_sub (branching factor) must be >= 2");
  }
  if (cfg_.levels < 1) {
    throw std::invalid_argument("PGMCell: levels must be >= 1");
  }
  if (cfg_.hidden < 1) {
    cfg_.hidden = cfg_.d_input;
  }
  if (cfg_.chunk_len < 2) {
    cfg_.chunk_len = 2;
  }
  if (cfg_.topk < 1) {
    cfg_.topk = 4;
  }
  if (cfg_.beam < 1) {
    cfg_.beam = 2;
  }
  if (cfg_.rehash_t < 1) {
    cfg_.rehash_t = 16;
  }
  if (cfg_.hops < 0) {
    cfg_.hops = 0;
  }

  if (cfg_.n_slots > 0) {
    n_slots_ = std::min(cfg_.n_slots, kMaxSlots);
  } else {
    n_slots_ = ipow(cfg_.n_sub, cfg_.levels);
    if (n_slots_ > kMaxSlots) {
      n_slots_ = kMaxSlots;
    }
  }
  if (n_slots_ < 2) {
    throw std::invalid_argument("PGMCell: n_slots must be >= 2");
  }

  init_codebooks();
  h_.assign(static_cast<std::size_t>(cfg_.hidden), 0.0);
  t1_.clear();
  t1_.reserve(static_cast<std::size_t>(cfg_.chunk_len));
}

void PGMCell::init_codebooks() {
  codebooks_.assign(static_cast<std::size_t>(cfg_.levels), {});
  std::mt19937_64 rng(cfg_.seed + 101);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int L = 0; L < cfg_.levels; ++L) {
    auto& book = codebooks_[static_cast<std::size_t>(L)];
    book.assign(static_cast<std::size_t>(cfg_.n_sub),
                std::vector<double>(static_cast<std::size_t>(cfg_.d_input), 0.0));
    for (int j = 0; j < cfg_.n_sub; ++j) {
      for (int d = 0; d < cfg_.d_input; ++d) {
        book[static_cast<std::size_t>(j)][static_cast<std::size_t>(d)] = nd(rng);
      }
      l2_normalize(book[static_cast<std::size_t>(j)]);
    }
  }
}

void PGMCell::reset() {
  V_.clear();
  occupied_.clear();
  edges_.clear();
  t1_.clear();
  std::fill(h_.begin(), h_.end(), 0.0);
}

std::size_t PGMCell::edge_count() const {
  std::size_t n = 0;
  for (const auto& kv : edges_) {
    n += kv.second.size();
  }
  return n;
}

double PGMCell::dot(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

void PGMCell::l2_normalize(std::vector<double>& v) {
  double n2 = 0.0;
  for (double x : v) {
    n2 += x * x;
  }
  if (n2 <= 0.0) {
    return;
  }
  const double inv = 1.0 / std::sqrt(n2);
  for (double& x : v) {
    x *= inv;
  }
}

std::vector<double> PGMCell::project_input(const std::vector<double>& x) const {
  std::vector<double> out(static_cast<std::size_t>(cfg_.d_input), 0.0);
  const int n = std::min(cfg_.d_input, static_cast<int>(x.size()));
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)];
  }
  l2_normalize(out);
  return out;
}

int PGMCell::primary_slot(const std::vector<double>& k) const {
  int slot = 0;
  for (int L = 0; L < cfg_.levels; ++L) {
    const auto& book = codebooks_[static_cast<std::size_t>(L)];
    int best_j = 0;
    double best = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < cfg_.n_sub; ++j) {
      const double s = dot(book[static_cast<std::size_t>(j)], k);
      if (s > best) {
        best = s;
        best_j = j;
      }
    }
    slot = slot * cfg_.n_sub + best_j;
  }
  if (slot < 0) {
    slot = 0;
  }
  if (slot >= n_slots_) {
    slot %= n_slots_;
  }
  return slot;
}

std::vector<int> PGMCell::candidate_slots(const std::vector<double>& k, int t) const {
  // Hierarchical beam expand: O(levels · beam · n_sub · d) — log-N addressing.
  struct Partial {
    int prefix = 0;
    double score = 0.0;
  };
  const int beam_w = std::max(t, cfg_.beam);
  std::vector<Partial> beam{{0, 0.0}};
  for (int L = 0; L < cfg_.levels; ++L) {
    const auto& book = codebooks_[static_cast<std::size_t>(L)];
    std::vector<Partial> nxt;
    nxt.reserve(beam.size() * static_cast<std::size_t>(cfg_.n_sub));
    for (const auto& p : beam) {
      for (int j = 0; j < cfg_.n_sub; ++j) {
        Partial q;
        q.prefix = p.prefix * cfg_.n_sub + j;
        q.score = p.score + dot(book[static_cast<std::size_t>(j)], k);
        nxt.push_back(q);
      }
    }
    const int keep = std::min(beam_w, static_cast<int>(nxt.size()));
    std::partial_sort(nxt.begin(), nxt.begin() + keep, nxt.end(),
                      [](const Partial& a, const Partial& b) { return a.score > b.score; });
    nxt.resize(static_cast<std::size_t>(keep));
    beam.swap(nxt);
  }

  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(t));
  for (const auto& p : beam) {
    int s = p.prefix;
    if (s < 0) {
      s = 0;
    }
    if (s >= n_slots_) {
      s %= n_slots_;
    }
    if (std::find(out.begin(), out.end(), s) == out.end()) {
      out.push_back(s);
    }
    if (static_cast<int>(out.size()) >= t) {
      break;
    }
  }
  if (out.empty()) {
    out.push_back(primary_slot(k));
  }
  return out;
}

int PGMCell::assign_rehash(const std::vector<double>& k) {
  const auto cands = candidate_slots(k, cfg_.rehash_t);
  for (int s : cands) {
    const auto it = occupied_.find(s);
    if (it == occupied_.end() || !it->second) {
      return s;
    }
    const auto vit = V_.find(s);
    if (vit != V_.end() && dot(vit->second, k) > sim_thresh_) {
      return s;
    }
  }
  return cands.empty() ? primary_slot(k) : cands[0];
}

int PGMCell::locate_rehash(const std::vector<double>& q) const {
  const auto cands = candidate_slots(q, cfg_.rehash_t);
  int best_s = cands.empty() ? primary_slot(q) : cands[0];
  double best_sim = -1.0;
  for (int s : cands) {
    const auto oit = occupied_.find(s);
    if (oit == occupied_.end() || !oit->second) {
      continue;
    }
    const auto vit = V_.find(s);
    if (vit == V_.end()) {
      continue;
    }
    const double sim = dot(vit->second, q);
    if (sim > best_sim) {
      best_sim = sim;
      best_s = s;
    }
  }
  return best_s;
}

void PGMCell::store_content(int slot, const std::vector<double>& x) {
  V_[slot] = x;
  occupied_[slot] = true;
}

void PGMCell::sparsify_row(int src) {
  auto it = edges_.find(src);
  if (it == edges_.end()) {
    return;
  }
  auto& nbrs = it->second;
  if (static_cast<int>(nbrs.size()) <= cfg_.topk) {
    // Row-normalize when all weights present.
    double sum = 0.0;
    for (const auto& e : nbrs) {
      sum += e.w;
    }
    if (sum > 0.0) {
      for (auto& e : nbrs) {
        e.w /= sum;
      }
    }
    return;
  }
  std::partial_sort(nbrs.begin(), nbrs.begin() + cfg_.topk, nbrs.end(),
                    [](const EdgeNbr& a, const EdgeNbr& b) { return a.w > b.w; });
  nbrs.resize(static_cast<std::size_t>(cfg_.topk));
  double sum = 0.0;
  for (const auto& e : nbrs) {
    sum += e.w;
  }
  if (sum > 0.0) {
    for (auto& e : nbrs) {
      e.w /= sum;
    }
  }
}

void PGMCell::write_link(const std::vector<double>& a, const std::vector<double>& b) {
  const int sa = assign_rehash(a);
  const int sb = assign_rehash(b);
  store_content(sa, a);
  store_content(sb, b);
  auto& nbrs = edges_[sa];
  bool found = false;
  for (auto& e : nbrs) {
    if (e.dst == sb) {
      e.w += eta_;
      found = true;
      break;
    }
  }
  if (!found) {
    nbrs.push_back(EdgeNbr{sb, eta_});
  }
  sparsify_row(sa);
}

void PGMCell::consolidate_adjacent() {
  if (t1_.size() < 2) {
    return;
  }
  for (std::size_t i = 0; i + 1 < t1_.size(); ++i) {
    write_link(t1_[i], t1_[i + 1]);
  }
}

std::vector<double> PGMCell::retrieve(const std::vector<double>& q) const {
  const int start = locate_rehash(q);
  struct BeamItem {
    int slot = 0;
    double score = 0.0;
  };
  std::vector<BeamItem> beam{{start, 0.0}};
  int best_tip = start;
  double best_score = 0.0;

  for (int hop = 0; hop < cfg_.hops; ++hop) {
    std::vector<BeamItem> nxt;
    for (const auto& b : beam) {
      const auto eit = edges_.find(b.slot);
      if (eit == edges_.end() || eit->second.empty()) {
        nxt.push_back(b);
        continue;
      }
      for (const auto& e : eit->second) {
        nxt.push_back(BeamItem{e.dst, b.score + e.w});
      }
    }
    if (nxt.empty()) {
      break;
    }
    const int keep = std::min(cfg_.beam, static_cast<int>(nxt.size()));
    std::partial_sort(nxt.begin(), nxt.begin() + keep, nxt.end(),
                      [](const BeamItem& a, const BeamItem& b) { return a.score > b.score; });
    nxt.resize(static_cast<std::size_t>(keep));
    beam.swap(nxt);
    if (beam[0].score > best_score) {
      best_score = beam[0].score;
      best_tip = beam[0].slot;
    }
  }

  std::vector<double> out(static_cast<std::size_t>(cfg_.d_input), 0.0);
  const auto vit = V_.find(best_tip);
  if (vit != V_.end()) {
    out = vit->second;
  } else {
    // Fall back to start content or query itself.
    const auto sit = V_.find(start);
    if (sit != V_.end()) {
      out = sit->second;
    } else {
      out = q;
    }
  }
  return out;
}

std::vector<double> PGMCell::step(const std::vector<double>& x) {
  const auto xin = project_input(x);
  t1_.push_back(xin);

  const auto tip = retrieve(xin);

  // Fast within-chunk mix: blend query and retrieved tip into hidden (graph frozen until consolidate).
  const int hdim = cfg_.hidden;
  const int d = cfg_.d_input;
  constexpr double kAlpha = 0.55;
  for (int i = 0; i < hdim; ++i) {
    const double qx = (i < d) ? xin[static_cast<std::size_t>(i)] : 0.0;
    const double tx = (i < d) ? tip[static_cast<std::size_t>(i)] : 0.0;
    h_[static_cast<std::size_t>(i)] =
        kAlpha * h_[static_cast<std::size_t>(i)] + (1.0 - kAlpha) * (0.5 * qx + 0.5 * tx);
  }

  if (static_cast<int>(t1_.size()) >= cfg_.chunk_len) {
    consolidate_adjacent();
    t1_.clear();
  }

  return h_;
}

nlohmann::json PGMCell::get_state() const {
  nlohmann::json j;
  j["format"] = "pgm_cell_v1";
  j["n_slots"] = n_slots_;
  j["eta"] = eta_;
  j["sim_thresh"] = sim_thresh_;
  j["h"] = h_;
  j["t1"] = t1_;
  j["codebooks"] = codebooks_;
  nlohmann::json v_obj = nlohmann::json::object();
  for (const auto& [slot, vec] : V_) {
    v_obj[std::to_string(slot)] = vec;
  }
  j["V"] = std::move(v_obj);
  nlohmann::json occ = nlohmann::json::array();
  for (const auto& [slot, on] : occupied_) {
    if (on) {
      occ.push_back(slot);
    }
  }
  j["occupied"] = std::move(occ);
  nlohmann::json e_obj = nlohmann::json::object();
  for (const auto& [src, nbrs] : edges_) {
    nlohmann::json row = nlohmann::json::array();
    for (const auto& e : nbrs) {
      row.push_back({{"dst", e.dst}, {"w", e.w}});
    }
    e_obj[std::to_string(src)] = std::move(row);
  }
  j["edges"] = std::move(e_obj);
  return j;
}

void PGMCell::set_state(const nlohmann::json& state) {
  if (state.contains("n_slots")) {
    n_slots_ = state.at("n_slots").get<int>();
  }
  if (state.contains("eta")) {
    eta_ = state.at("eta").get<double>();
  }
  if (state.contains("sim_thresh")) {
    sim_thresh_ = state.at("sim_thresh").get<double>();
  }
  if (state.contains("h")) {
    h_ = state.at("h").get<std::vector<double>>();
  }
  if (state.contains("t1")) {
    t1_ = state.at("t1").get<std::vector<std::vector<double>>>();
  }
  if (state.contains("codebooks")) {
    codebooks_ = state.at("codebooks").get<std::vector<std::vector<std::vector<double>>>>();
  }
  V_.clear();
  occupied_.clear();
  edges_.clear();
  if (state.contains("V") && state.at("V").is_object()) {
    for (const auto& [k, vec] : state.at("V").items()) {
      V_[std::stoi(k)] = vec.get<std::vector<double>>();
    }
  }
  if (state.contains("occupied") && state.at("occupied").is_array()) {
    for (const auto& s : state.at("occupied")) {
      occupied_[s.get<int>()] = true;
    }
  }
  if (state.contains("edges") && state.at("edges").is_object()) {
    for (const auto& [k, row] : state.at("edges").items()) {
      std::vector<EdgeNbr> nbrs;
      for (const auto& e : row) {
        EdgeNbr n;
        n.dst = e.at("dst").get<int>();
        n.w = e.at("w").get<double>();
        nbrs.push_back(n);
      }
      edges_[std::stoi(k)] = std::move(nbrs);
    }
  }
}

}  // namespace cypha::cyphalm
