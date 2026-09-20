/// CLI: byte-level CyphaLM generation via serve path (greedy / temperature / top-k).
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--prompt TEXT] [--max-tokens N] [--strategy greedy|temperature|top_k] "
                 "[--temperature T] [--top-k K] [--seed S] [--table-bits M]\n",
                 argv0);
}

std::vector<int> bytes_from_text(const std::string& text, int vocab_size) {
    std::vector<int> out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (static_cast<int>(c) < vocab_size) {
            out.push_back(static_cast<int>(c));
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string prompt = "The quick brown fox ";
    int max_tokens = 32;
    std::string strategy = "temperature";
    double temperature = 0.9;
    int top_k = 40;
    std::uint64_t seed = 42;
    int table_bits = 16;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--strategy" && i + 1 < argc) {
            strategy = argv[++i];
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--table-bits" && i + 1 < argc) {
            table_bits = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = table_bits;

    cypha::cyphalm::CyphaLMModel model(cfg);
    const std::vector<int> prompt_ids = bytes_from_text(prompt, cfg.vocab_size);

    cypha::cyphalm::DecodeParams params;
    params.strategy = cypha::cyphalm::decode_strategy_from_string(strategy);
    params.temperature = temperature;
    params.top_k = top_k;
    params.seed = seed;

    const auto out = cypha::cyphalm::generate_decode(model, prompt_ids, max_tokens, params);

    std::string completion;
    completion.reserve(out.generated_ids.size());
    for (int id : out.generated_ids) {
        if (id >= 0 && id < 256) {
            completion.push_back(static_cast<char>(id));
        }
    }

    std::printf("strategy=%s prompt_bytes=%zu generated_bytes=%zu\n", strategy.c_str(),
                prompt.size(), completion.size());
    std::printf("--- completion ---\n%s\n", completion.c_str());
    return out.generated_ids.empty() ? 1 : 0;
}
