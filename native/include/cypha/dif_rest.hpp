#pragma once

#include <mutex>

#include "httplib.h"

namespace cypha {

class Cypha;

/// Bind Cypha instance owned by ``cypha_rest`` main.
void dif_rest_configure(std::mutex* mu, Cypha* cypha);

/// Register ``POST /sample`` and ``POST /retrieve`` (latent generation; former ``/dif/*``).
void register_dif_rest_routes(httplib::Server& svr);

}  // namespace cypha
