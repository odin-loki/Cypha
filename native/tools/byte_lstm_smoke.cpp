/// ByteLstm: learns a periodic byte pattern, beats a unigram on text, and
/// save/load reproduces predictions exactly.
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

#include "cypha/cyphalm/byte_lstm.hpp"

int main() {
    using cypha::cyphalm::ByteLstm;
    using cypha::cyphalm::ByteLstmOptions;
    ByteLstmOptions o;
    o.hidden = 32;
    o.bptt = 10;
    o.lr = 1e-2f;
    ByteLstm net(o);
    const std::string pat = "abcdefgh";
    double tail = 0.0;
    for (int i = 0; i < 6000; ++i) {
        const float l = net.observe(pat[i % pat.size()]);
        if (i >= 5000) tail += l;
    }
    tail /= 1000.0;
    if (!(tail < 0.05)) {
        std::printf("byte_lstm_smoke FAIL pattern loss %.4f nats\n", tail);
        return 1;
    }
    std::string text;
    const char* w[] = {"the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dogs. "};
    unsigned s = 1;
    while (text.size() < 40000) {
        s = s * 1103515245u + 12345u;
        text += w[(s >> 16) % 8];
    }
    ByteLstm lm(o);
    double late = 0.0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const float l = lm.observe(static_cast<unsigned char>(text[i]));
        if (i >= text.size() - 5000) late += l;
    }
    late = late / 5000.0 / std::log(2.0);
    if (!(late < 2.0)) {  // unigram over this text is ~3.7 bits
        std::printf("byte_lstm_smoke FAIL text loss %.3f bits/byte\n", late);
        return 1;
    }
    std::stringstream ss;
    lm.write(ss);
    ByteLstm copy(o);
    copy.read(ss);
    for (int k = 0; k < 256; ++k) {
        if (copy.log_probs()[k] != lm.log_probs()[k]) {
            std::printf("byte_lstm_smoke FAIL save/load mismatch at %d\n", k);
            return 1;
        }
    }
    std::printf("byte_lstm_smoke OK pattern %.4f nats, text %.3f bits/byte\n", tail, late);
    return 0;
}
