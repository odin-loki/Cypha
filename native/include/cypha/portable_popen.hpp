#pragma once

/// MSVC exposes ``_popen`` / ``_pclose`` instead of POSIX ``popen`` / ``pclose``.

#include <cstdio>

#ifndef popen
#define popen _popen
#define pclose _pclose
#endif
