#pragma once

/// MSVC exposes ``_popen`` / ``_pclose`` instead of POSIX ``popen`` / ``pclose``.

#include <cstdio>

#if defined(_WIN32) && !defined(popen)
#define popen _popen
#define pclose _pclose
#endif
