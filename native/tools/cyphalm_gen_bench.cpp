/// Generation benchmark for CyphaLM: continue N held-out prompts and score each
/// continuation with a fixed judge model (bits/byte, learning off) next to the
/// true continuation, plus distinct 4-grams and speed.
///
/// The judge reads each prompt with learning off and scores both texts from the
/// same state (hp::StreamRewind), so every decoder and model is judged alike.
///
///   cyphalm_gen_bench --load A.json [--member B.json]... --judge J.json \
///       --text enwik8 --offset 96000000 --prompts 8 [--word-candidates 8]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using Clock = std::chrono::steady_clock;

/// Bytes of ``path`` as token ids; bytes outside the model's vocabulary
/// (e.g. UTF-8 with vocab 128) become '?'.
std::vector<int> read_slice(const std::string& path, std::uint64_t offset, std::uint64_t n, int vocab) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<char> buf(n);
    in.read(buf.data(), static_cast<std::streamsize>(n));
    buf.resize(static_cast<std::size_t>(in.gcount()));
    std::vector<int> ids;
    for (char c : buf) {
        const int b = static_cast<unsigned char>(c);
        ids.push_back(b < vocab ? b : '?');
    }
    return ids;
}

double distinct_4gram(const std::vector<int>& ids) {
    if (ids.size() < 4) return 0.0;
    std::vector<std::uint32_t> g;
    for (std::size_t i = 0; i + 4 <= ids.size(); ++i) {
        std::uint32_t v = 0;
        for (int k = 0; k < 4; ++k) v = (v << 8) | static_cast<std::uint32_t>(ids[i + k] & 0xff);
        g.push_back(v);
    }
    const double total = static_cast<double>(g.size());
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return static_cast<double>(g.size()) / total;
}

std::string printable(const std::vector<int>& ids) {
    std::string s;
    for (int b : ids) s += (b == '\n') ? std::string("\\n") : (b >= 32 && b < 127 ? std::string(1, static_cast<char>(b)) : std::string("?"));
    return s;
}

/// Judge bits/byte of ``text`` after ``prompt``; the judge is returned exactly
/// to its state on entry.
double judge_bits(cypha::cyphalm::CyphaLMModel& judge, const std::vector<int>& prompt,
                  const std::vector<int>& text) {
    auto& h = judge.hp_backend();
    hp::StreamRewind rewind(h.all_predictors());
    for (int b : prompt) h.serve_advance_byte(static_cast<std::uint8_t>(b));
    double bits = 0.0;
    for (int b : text) bits += h.observe_next_byte(static_cast<std::uint8_t>(b)) / std::log(2.0);
    rewind.rewind();
    h.invalidate_scoring_cache();
    return bits / static_cast<double>(std::max<std::size_t>(1, text.size()));
}

}  // namespace

int main(int argc, char** argv) {
    std::string load, judge_path, text_path;
    std::vector<std::string> members;
    std::string infinigram_path;
    std::uint64_t offset = 96000000;
    int prompts = 8, prompt_bytes = 256, gen_bytes = 200, stride = 8192;
    cypha::cyphalm::DecodeParams params;
    params.seed = 1234;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--load") load = next();
        else if (a == "--member") members.push_back(next());
        else if (a == "--infinigram") infinigram_path = next();
        else if (a == "--judge") judge_path = next();
        else if (a == "--text") text_path = next();
        else if (a == "--offset") offset = std::stoull(next());
        else if (a == "--prompts") prompts = std::stoi(next());
        else if (a == "--prompt-bytes") prompt_bytes = std::stoi(next());
        else if (a == "--gen-bytes") gen_bytes = std::stoi(next());
        else if (a == "--stride") stride = std::stoi(next());
        else if (a == "--word-candidates") params.word_candidates = std::stoi(next());
        else if (a == "--temperature") params.temperature = std::stod(next());
        else if (a == "--min-p") params.min_p = std::stod(next());
        else if (a == "--seed") params.seed = std::stoull(next());
        else {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }
    if (load.empty() || judge_path.empty() || text_path.empty()) {
        std::cerr << "need --load, --judge and --text\n";
        return 2;
    }

    auto model = cypha::cyphalm::load_cyphalm_model(load);
    for (const auto& m : members) {
        model.add_ensemble_member(cypha::cyphalm::load_cyphalm_model(m),
                                  1.0 / static_cast<double>(members.size() + 1));
    }
    if (!infinigram_path.empty()) model.attach_infinigram(infinigram_path);
    auto judge = cypha::cyphalm::load_cyphalm_model(judge_path);
    judge.hp_backend().set_learning(false);
    judge.reset_stream(/*keep_history=*/true);

    nlohmann::json rows = nlohmann::json::array();
    double sum_gen = 0.0, sum_ref = 0.0, sum_d4 = 0.0, sum_d4_ref = 0.0, gen_secs = 0.0;
    std::size_t gen_total = 0;
    for (int p = 0; p < prompts; ++p) {
        const std::uint64_t off = offset + static_cast<std::uint64_t>(p) * static_cast<std::uint64_t>(stride);
        const int vocab = model.config().vocab_size;
        const auto prompt = read_slice(text_path, off, static_cast<std::uint64_t>(prompt_bytes), vocab);
        const auto truth = read_slice(text_path, off + static_cast<std::uint64_t>(prompt_bytes),
                                      static_cast<std::uint64_t>(gen_bytes), vocab);
        const auto t0 = Clock::now();
        const auto g = cypha::cyphalm::generate_decode(model, prompt, gen_bytes, params);
        gen_secs += std::chrono::duration<double>(Clock::now() - t0).count();
        gen_total += g.generated_ids.size();
        const double jg = judge_bits(judge, prompt, g.generated_ids);
        const double jr = judge_bits(judge, prompt, truth);
        const double d4 = distinct_4gram(g.generated_ids), d4r = distinct_4gram(truth);
        sum_gen += jg;
        sum_ref += jr;
        sum_d4 += d4;
        sum_d4_ref += d4r;
        rows.push_back({{"offset", off}, {"judge_bits", jg}, {"reference_judge_bits", jr},
                        {"distinct_4gram", d4}, {"text", printable(g.generated_ids)}});
    }
    const double n = static_cast<double>(std::max(1, prompts));
    nlohmann::json out = {{"harness", "cyphalm_gen_bench"},
                          {"load", load},
                          {"members", members},
                          {"judge", judge_path},
                          {"text", text_path},
                          {"word_candidates", params.word_candidates},
                          {"temperature", params.temperature},
                          {"min_p", params.min_p},
                          {"mean_judge_bits", sum_gen / n},
                          {"mean_reference_judge_bits", sum_ref / n},
                          {"mean_distinct_4gram", sum_d4 / n},
                          {"mean_reference_distinct_4gram", sum_d4_ref / n},
                          {"ms_per_byte", 1e3 * gen_secs / static_cast<double>(std::max<std::size_t>(1, gen_total))},
                          {"prompts", rows}};
    std::cout << out.dump(2) << "\n";
    return 0;
}
