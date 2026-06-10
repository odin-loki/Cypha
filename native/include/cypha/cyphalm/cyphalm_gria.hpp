#pragma once

#include <vector>

namespace cypha::cyphalm {

/// Online softmax output layer (GRIA stand-in until full low-rank port lands).
class GriaHead {
 public:
    explicit GriaHead(int in_dim, int vocab_size, double alpha_init, double laplace);

    void reset();
    std::vector<double> forward(const std::vector<double>& v);
    void train_step(const std::vector<double>& v, int target_id, double lr);

    int in_dim() const { return in_dim_; }

 private:
    int in_dim_;
    int vocab_size_;
    double alpha_;
    double laplace_;
    std::vector<double> W_;
    std::vector<double> b_;
    std::vector<double> token_counts_;

    static std::vector<double> softmax(const std::vector<double>& logits);
};

}  // namespace cypha::cyphalm
