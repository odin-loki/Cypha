// K4/K5: the Cypha DIF classifier bound to the thought interpreter: the `classify` primitive's classifier and the
// applicability model of THOUGHT 3.1 (one classifier per process on situation features, its OOD flag routing to
// IMPASSE). Links cypha_core. docs/reports/K4_K5_THOUGHT_BINDINGS.md.
//
// Author: Odin Loch
#pragma once

#include <chrono>
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
