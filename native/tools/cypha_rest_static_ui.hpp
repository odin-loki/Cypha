#pragma once

#include "httplib.h"

#include <string>

namespace cypha_rest_ui {

/// Mount Studio SPA: GET / → index.html; GET /ui/* → static assets (disk or embedded).
void configure_static_ui(httplib::Server& svr, const char* argv0);

}  // namespace cypha_rest_ui
