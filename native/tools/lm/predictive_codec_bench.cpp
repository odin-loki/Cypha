/// Train hybrid production recipe briefly, then measure predictive arithmetic coding
/// (model BPC vs coded BPC) on a WikiText slice or synthetic fallback.
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/predictive_codec.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string load_text(const std::filesystem::path& path, std::size_t max_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (s.size() > max_bytes) s.resize(max_bytes);
    return s;
}

std::vector<std::uint32_t> bytes_to_tokens(const std::string& s) {
    std::vector<std::uint32_t> t;
    t.reserve(s.size());
    for (unsigned char c : s) t.push_back(c);
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t n_train = 8000;
    std::size_t n_eval = 2000;
    int epochs = 1;
    int lstm_layers = 2;
    int lstm_hidden = 128;
    bool use_memory_attn = false;
    bool use_wave2_bptt = false;
    bool use_wave2_sched = false;
    int bptt_lstm = -1;
    int ngram_max_order = -1;
    double neural_floor = -1.0;
    double neural_prior = -1.0;
    int min_context_count = -1;
    std::filesystem::path corpus;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n-train" && i + 1 < argc) n_train = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (a == "--n-eval" && i + 1 < argc) n_eval = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (a == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (a == "--lstm-layers" && i + 1 < argc) lstm_layers = std::stoi(argv[++i]);
        else if (a == "--lstm-hidden" && i + 1 < argc) lstm_hidden = std::stoi(argv[++i]);
        else if (a == "--lstm-memory-attn") use_memory_attn = true;
        else if (a == "--wave2-bptt") use_wave2_bptt = true;
        else if (a == "--wave2-sched") use_wave2_sched = true;
        else if (a == "--bptt-lstm" && i + 1 < argc) bptt_lstm = std::stoi(argv[++i]);
        else if (a == "--mixer-order" && i + 1 < argc) ngram_max_order = std::stoi(argv[++i]);
        else if (a == "--neural-floor" && i + 1 < argc) neural_floor = std::stod(argv[++i]);
        else if (a == "--neural-prior" && i + 1 < argc) neural_prior = std::stod(argv[++i]);
        else if (a == "--min-context-count" && i + 1 < argc) min_context_count = std::stoi(argv[++i]);
        else if (a == "--corpus" && i + 1 < argc) corpus = argv[++i];
    }

    if (corpus.empty()) {
        const char* candidates[] = {
            "bench/data/wikitext2/wikitext-2/wiki.train.tokens",
            "../bench/data/wikitext2/wikitext-2/wiki.train.tokens",
            "../../bench/data/wikitext2/wikitext-2/wiki.train.tokens",
            "C:/Users/odinl/OneDrive/Desktop/Cypha/bench/data/wikitext2/wikitext-2/wiki.train.tokens",
        };
        for (const char* c : candidates) {
            if (std::filesystem::exists(c)) {
                corpus = c;
                break;
            }
        }
    }

    std::string text;
    if (!corpus.empty()) {
        text = load_text(corpus, n_train + n_eval + 8);
        std::cout << "corpus=" << corpus << " loaded=" << text.size() << "\n";
    }
    if (text.size() < n_train + n_eval) {
        text.clear();
        text.reserve(n_train + n_eval);
        for (std::size_t i = 0; i < n_train + n_eval; ++i) {
            text.push_back(static_cast<char>(32 + (i * 7 + 13) % 95));
        }
        std::cout << "corpus=synthetic printable ASCII\n";
    }

    const std::string train_s = text.substr(0, n_train);
    const std::string eval_s = text.substr(n_train, n_eval);
    auto train_tok = bytes_to_tokens(train_s);
    auto eval_tok = bytes_to_tokens(eval_s);

    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = 256;
    cfg.lstm_hidden = std::max(64, lstm_hidden);
    cfg.lstm_layers = std::max(1, lstm_layers);
    cfg.d_embed = 64;
    cfg.d_state = cfg.lstm_hidden;
    cfg.field_dim = 160;
    cfg.seed = 42;
    cfg.use_lstm_memory_attn = use_memory_attn;
    cypha::cyphalm::apply_hybrid_production_recipe(cfg);
    cfg.lstm_layers = std::max(1, lstm_layers);
    cfg.lstm_hidden = std::max(64, lstm_hidden);
    cfg.use_lstm_memory_attn = use_memory_attn;
    if (use_wave2_bptt || use_wave2_sched || bptt_lstm > 0) {
        cypha::cyphalm::Wave2BpttOptions w2;
        w2.bptt_steps = bptt_lstm > 0 ? bptt_lstm : 8;
        w2.with_schedule = use_wave2_sched;
        // Approx token updates for schedule scaling (train_sequence steps ≈ n_train * epochs).
        w2.train_steps = static_cast<int>(n_train) * std::max(1, epochs);
        cypha::cyphalm::apply_wave2_bptt_recipe(cfg, w2);
    }

    cypha::cyphalm::CyphaLMModel model(cfg);
    std::cout << "mode=" << cypha::cyphalm::context_mode_name(model.config().context_mode)
              << " ngram_fuse_split=" << model.config().ngram_fuse_split
              << " lstm_hidden=" << model.config().lstm_hidden
              << " lstm_layers=" << model.config().lstm_layers
              << " mem_attn=" << (model.config().use_lstm_memory_attn ? 1 : 0)
              << " bptt=" << model.config().lstm_bptt_steps
              << " optim=" << model.config().lstm_optim
              << " lr=" << model.config().lstm_lr
              << " warm=" << model.config().lstm_lr_warmup_steps
              << " cos=" << model.config().lstm_lr_cosine_steps << "\n";

    std::vector<int> train_ids(train_tok.begin(), train_tok.end());
    model.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, epochs);

    std::vector<int> eval_ids(eval_tok.begin(), eval_tok.end());
    const double eval_bpc = model.eval_bpc(eval_ids, static_cast<int>(eval_ids.size()));
    const auto ckpt = (std::filesystem::temp_directory_path() / "cypha_codec_bench_seq").string();
    model.save(ckpt);
    const std::string ckpt_json = ckpt + ".json";

    // Encode from the saved checkpoint (same start state as decode), not the warm train handle.
    cypha::cyphalm::CyphaLMModel enc = cypha::cyphalm::CyphaLMModel::from_json_npz(ckpt_json);
    cypha::cyphalm::PredictiveCodecOptions codec_opt;
    if (ngram_max_order >= 0) codec_opt.ngram_max_order = ngram_max_order;
    if (neural_floor >= 0.0) codec_opt.neural_weight_floor = neural_floor;
    if (neural_prior >= 0.0) codec_opt.neural_prior = neural_prior;
    if (min_context_count >= 0) codec_opt.min_context_count = min_context_count;

    auto packed = cypha::cyphalm::compress_tokens(enc, eval_tok, codec_opt);
    if (!packed.detail.empty()) {
        std::cerr << "compress failed: " << packed.detail << "\n";
        return 2;
    }
    {
        cypha::cyphalm::CyphaLMModel dec = cypha::cyphalm::CyphaLMModel::from_json_npz(ckpt_json);
        std::string detail;
        auto round = cypha::cyphalm::decompress_tokens(dec, packed.bytes, eval_tok.front(),
                                                       eval_tok.size(), &detail, codec_opt);
        if (!detail.empty() || round != eval_tok) {
            std::cerr << "roundtrip failed: " << detail;
            if (detail.empty()) {
                std::cerr << "size got=" << round.size() << " want=" << eval_tok.size();
                const std::size_t n = std::min(round.size(), eval_tok.size());
                for (std::size_t i = 0; i < n; ++i) {
                    if (round[i] != eval_tok[i]) {
                        std::cerr << " first_mismatch_i=" << i << " got=" << round[i]
                                  << " want=" << eval_tok[i];
                        break;
                    }
                }
            }
            std::cerr << "\n";
            return 3;
        }
    }
    cypha::cyphalm::CyphaLMModel model2 = cypha::cyphalm::CyphaLMModel::from_json_npz(ckpt_json);

    cypha::cyphalm::PredictiveCodecOptions no_mix;
    no_mix.use_mixer = false;
    no_mix.online_adapt = false;
    no_mix.use_hidden_knn = false;
    auto packed_n = cypha::cyphalm::compress_tokens(model2, eval_tok, no_mix);

    std::cout << "n_train=" << n_train << " n_eval=" << n_eval << " epochs=" << epochs << "\n";
    std::cout << "eval_bpc=" << eval_bpc << "\n";
    std::cout << "neural_bpc=" << packed.neural_bpc << " (no_mix=" << packed_n.model_bpc << ")\n";
    std::cout << "mix_bpc=" << packed.model_bpc << "\n";
    std::cout << "coded_bpc=" << packed.coded_bpc << "\n";
    std::cout << "mix_gain=" << (packed.neural_bpc - packed.model_bpc) << "\n";
    std::cout << "mixer_order=" << codec_opt.ngram_max_order
              << " neural_floor=" << codec_opt.neural_weight_floor
              << " neural_prior=" << codec_opt.neural_prior
              << " min_ctx=" << codec_opt.min_context_count << "\n";
    std::cout << "coded_bytes=" << packed.bytes.size() << " raw_bytes=" << eval_s.size() << "\n";
    std::cout << "uniform_byte_bpc=8.0\n";
    std::cout << "predictive_codec_bench OK\n";
    return 0;
}
