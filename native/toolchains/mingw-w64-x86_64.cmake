# Cross-compile Cypha native for Windows x86_64 from Linux/WSL (MinGW-w64).
#
# Usage (from repo root; toolchain path must be absolute if CMake cwd differs):
#   cmake -S native -B native/build-mingw-w64 \
#     -DCMAKE_TOOLCHAIN_FILE="$PWD/native/toolchains/mingw-w64-x86_64.cmake" \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build native/build-mingw-w64 -j$(nproc)
#   cmake --test-dir native/build-mingw-w64 --output-on-failure
#
# Preset (see native/CMakePresets.json):
#   cmake --preset mingw-w64-cross
#
# Optional cache variables (first configure only unless -U):
#   -DCYPHA_MINGW_TOOLCHAIN_PREFIX=x86_64-w64-mingw32   # triplet prefix for gcc/g++/windres

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

if(NOT DEFINED CYPHA_MINGW_TOOLCHAIN_PREFIX)
  set(CYPHA_MINGW_TOOLCHAIN_PREFIX "x86_64-w64-mingw32" CACHE STRING "MinGW compiler triplet prefix (e.g. x86_64-w64-mingw32)")
endif()

set(_cypha_mgw_pfx "${CYPHA_MINGW_TOOLCHAIN_PREFIX}")

set(CMAKE_C_COMPILER "${_cypha_mgw_pfx}-gcc" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${_cypha_mgw_pfx}-g++" CACHE FILEPATH "")
set(CMAKE_RC_COMPILER "${_cypha_mgw_pfx}-windres" CACHE FILEPATH "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

set(CYPHA_MINGW_TOOLCHAIN_INCLUDED TRUE CACHE INTERNAL "Set when native/toolchains/mingw-w64-x86_64.cmake was used")
