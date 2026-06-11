#pragma once

#include <memory>
#include <mutex>
#include <random>

#include "httplib.h"

namespace cypha {

struct CyphaInferModel;
struct PreprocessorState;

/// Bind classifier model / preprocessor / session RNG owned by ``cypha_rest`` main.
void dif_rest_configure(std::mutex* mu, std::unique_ptr<CyphaInferModel>* model,
                        std::unique_ptr<PreprocessorState>* pre, std::mt19937* rng);

/// Register ``POST /dif/generate`` and ``POST /dif/retrieve`` (CyphaDIF latent generation).
void register_dif_rest_routes(httplib::Server& svr);

}  // namespace cypha
