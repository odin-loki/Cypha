#include <cstdio>
#include "hp/predictor.hpp"

int main() {
    hp::Config cfg;
    cfg.table_bits = 18;
    cfg.normalize();
    std::puts("creating predictor...");
    hp::Predictor pred(cfg);
    std::puts("predict...");
    const int p = pred.predict();
    std::printf("p=%d\n", p);
    pred.update(1);
    std::puts("OK");
    return 0;
}
