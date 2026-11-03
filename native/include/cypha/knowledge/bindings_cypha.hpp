// K4/K5: the Cypha DIF classifier bound to the thought interpreter: the `classify` primitive's classifier and the
// applicability model of THOUGHT 3.1 (one classifier per process on situation features, its OOD flag routing to
// IMPASSE). Links cypha_core. docs/reports/K4_K5_THOUGHT_BINDINGS.md. K6: a calibration map over the classifier's
// P (CalibratedApplicability; docs/reports/K6_K7_CHUNKING_AND_MINING.md).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/thought/bindings.hpp"

namespace cypha::thought::bind {

/// A fresh, empty Cypha classifier with an identity encoder of `input_dim` (as cypha_word_classes builds one).
inline std::unique_ptr<cypha::Cypha> fresh_cypha(int input_dim, int field_dim = 16) {
  cypha::FreshModelParams fp;
  fp.input_dim = input_dim;
  fp.field_dim = field_dim;
  const cypha::CNode root = cypha::create_fresh_model_root(fp);
  static int serial = 0;
  const auto path = (std::filesystem::temp_directory_path() /
                     ("thought_dif_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                      std::to_string(serial++) + ".cypha"))
                        .string();
  cypha::save_cypha_file(path.c_str(), root);
  auto cy = std::make_unique<cypha::Cypha>();
  const bool ok = cy->load(path);
  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (!ok) throw std::runtime_error("thought: cannot load a fresh Cypha model");
  return cy;
}

/// The `classify` primitive's classifier hook over a Cypha instance (label, confidence of that label, OOD flag).
inline std::function<ClassifierOut(const std::vector<double>&)> cypha_classifier(cypha::Cypha& cy) {
  return [&cy](const std::vector<double>& x) {
    ClassifierOut o;
    const cypha::PredictOut p = cy.predict(x.data(), static_cast<int>(x.size()));
    o.label = p.label;
    o.confidence = p.confidence;
    o.ood = p.is_ood;
    return o;
  };
}

/// THOUGHT 3.1: P(applicable | x) from a Cypha classifier per process, trained online with labels "yes" (a run with
/// R >= the useful threshold) and "no" (a run that ended in IMPASSE). x is wm "situation" (an array of numbers). Until a
/// process has `warm` labels, P is the Beta mean of its labels so far and nothing is OOD.
class DifApplicability : public ApplicabilityModel {
 public:
  explicit DifApplicability(int dim, std::size_t warm = 20) : dim_(dim), warm_(warm) {}

  ApplicabilityQuery query(const std::string& pid, const std::string&, const WorkingMemory& wm) override {
    Entry& e = entry(pid);
    ApplicabilityQuery q;
    const std::size_t n = e.yes + e.no;
    if (n < warm_) {
      q.p = (1.0 + static_cast<double>(e.yes)) / (2.0 + static_cast<double>(n));
      return q;
    }
    const std::vector<double> x = features(wm);
    const cypha::PredictOut p = e.cy->predict(x.data(), dim_);
    if (!p.detail.empty() || p.label.empty()) {
      q.p = (1.0 + static_cast<double>(e.yes)) / (2.0 + static_cast<double>(n));
      return q;
    }
    q.p = p.label == "yes" ? p.confidence : 1.0 - p.confidence;
    q.ood = p.is_ood;
    return q;
  }

  void label(const std::string& pid, const std::string&, const WorkingMemory& wm, bool positive) override {
    Entry& e = entry(pid);
    const std::vector<double> x = features(wm);
    const std::string lb = positive ? "yes" : "no";
    e.cy->update(x.data(), dim_, &lb, nullptr);
    (positive ? e.yes : e.no)++;
  }

  std::size_t labels(const std::string& pid) {
    Entry& e = entry(pid);
    return e.yes + e.no;
  }

 private:
  struct Entry {
    std::unique_ptr<cypha::Cypha> cy;
    std::size_t yes = 0, no = 0;
  };

  Entry& entry(const std::string& pid) {
    auto it = by_.find(pid);
    if (it == by_.end()) {
      Entry e;
      e.cy = fresh_cypha(dim_);
      it = by_.emplace(pid, std::move(e)).first;
    }
    return it->second;
  }

  std::vector<double> features(const WorkingMemory& wm) const {
    std::vector<double> x(static_cast<std::size_t>(dim_), 0.0);
    if (const Json* s = wm.get("situation"); s && s->is_array())
      for (std::size_t i = 0; i < x.size() && i < s->size(); ++i) x[i] = s->at(i).get<double>();
    return x;
  }

  int dim_;
  std::size_t warm_;
  std::map<std::string, Entry> by_;
};

/// K6 (HANDOFF 5.23; docs/reports/K6_K7_CHUNKING_AND_MINING.md): DIF applicability (or any base model) through a
/// calibration map of its P.
/// At each label the classifier's raw P for that process is recorded before the classifier learns from the label
/// (prequential). One pooled map over all processes is fitted on these (raw P, label) pairs from `warm` pairs on,
/// refitted every `refit` new pairs; before that the raw P passes through. The OOD flag passes through unchanged.
/// Isotonic: ECS's IsotonicCalibrator. Platt: P = sigmoid(a logit(raw) + b), Newton on the log loss with a ridge of
/// 1e-3 towards the identity map (a = 1, b = 0).
class CalibratedApplicability : public ApplicabilityModel {
 public:
  enum class Map { Isotonic, Platt };

  CalibratedApplicability(std::shared_ptr<ApplicabilityModel> base, Map map, std::size_t warm = 50, std::size_t refit = 25)
      : base_(std::move(base)), map_(map), warm_(warm), refit_(refit) {}

  ApplicabilityQuery query(const std::string& pid, const std::string& cls, const WorkingMemory& wm) override {
    ApplicabilityQuery q = base_->query(pid, cls, wm);
    q.p = calibrate(q.p);
    return q;
  }

  void label(const std::string& pid, const std::string& cls, const WorkingMemory& wm, bool positive) override {
    raw_.push_back(base_->query(pid, cls, wm).p);  // before the classifier learns from this label
    y_.push_back(positive ? 1 : 0);
    base_->label(pid, cls, wm, positive);
    const std::size_t n = raw_.size();
    if (n >= warm_ && (n - warm_) % refit_ == 0) fit();
  }

  double calibrate(double p) const {
    if (!fitted_) return p;
    if (map_ == Map::Isotonic) return iso_.predict(p);
    const double z = logit(p);
    return 1.0 / (1.0 + std::exp(-(a_ * z + b_)));
  }

  bool fitted() const { return fitted_; }
  std::size_t pairs() const { return raw_.size(); }
  const std::vector<double>& raw_seen() const { return raw_; }
  const std::vector<int>& labels_seen() const { return y_; }
  double platt_a() const { return a_; }
  double platt_b() const { return b_; }

  static double logit(double p) {
    const double q = std::min(1.0 - 1e-6, std::max(1e-6, p));
    return std::log(q / (1.0 - q));
  }

  /// Platt by Newton: minimise the summed log loss + lambda ((a - 1)^2 + b^2).
  static void platt_fit(const std::vector<double>& raw, const std::vector<int>& y, double& a, double& b,
                        double lambda = 1e-3, int iters = 50) {
    a = 1.0;
    b = 0.0;
    for (int it = 0; it < iters; ++it) {
      double ga = 2.0 * lambda * (a - 1.0), gb = 2.0 * lambda * b;
      double haa = 2.0 * lambda, hab = 0.0, hbb = 2.0 * lambda;
      for (std::size_t i = 0; i < raw.size(); ++i) {
        const double z = logit(raw[i]);
        const double p = 1.0 / (1.0 + std::exp(-(a * z + b)));
        const double r = p - static_cast<double>(y[i]);
        const double w = p * (1.0 - p);
        ga += r * z;
        gb += r;
        haa += w * z * z;
        hab += w * z;
        hbb += w;
      }
      const double det = haa * hbb - hab * hab;
      if (!(det > 1e-12)) break;
      const double da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det;
      a -= da;
      b -= db;
      if (std::abs(da) + std::abs(db) < 1e-10) break;
    }
  }

 private:
  void fit() {
    if (map_ == Map::Isotonic) iso_.fit(raw_, y_);
    else platt_fit(raw_, y_, a_, b_);
    fitted_ = true;
  }

  std::shared_ptr<ApplicabilityModel> base_;
  Map map_;
  std::size_t warm_, refit_;
  std::vector<double> raw_;
  std::vector<int> y_;
  ev::IsotonicCalibrator iso_;
  double a_ = 1.0, b_ = 0.0;
  bool fitted_ = false;
};

/// The K3 stand-in as a model: a Beta per (process, key), where the key is the task class (Beta-class) or one bucket
/// for every situation (Beta-pooled).
class BetaApplicability : public ApplicabilityModel {
 public:
  explicit BetaApplicability(bool use_class) : use_class_(use_class) {}
  ApplicabilityQuery query(const std::string& pid, const std::string& cls, const WorkingMemory&) override {
    ApplicabilityQuery q;
    q.p = b_[key(pid, cls)].mean();
    return q;
  }
  void label(const std::string& pid, const std::string& cls, const WorkingMemory&, bool positive) override {
    b_[key(pid, cls)].update(positive ? 1.0 : 0.0);
  }

 private:
  std::string key(const std::string& pid, const std::string& cls) const { return use_class_ ? pid + "|" + cls : pid; }
  bool use_class_;
  std::map<std::string, Beta> b_;
};

}  // namespace cypha::thought::bind
