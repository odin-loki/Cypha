# CyphaMinGW.cmake — MinGW-w64-specific link flags and helpers.
# Include after project(): include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CyphaMinGW.cmake")
#
# CMake sets MINGW when the target ABI is MinGW (native MSYS2/standalone or Linux→Windows cross).

if(MINGW)
  option(
    CYPHA_MINGW_STATIC_CXX_RUNTIME
    "MinGW: link -static-libgcc and -static-libstdc++ so Windows hosts do not need those DLLs on PATH"
    ON
  )
  option(
    CYPHA_MINGW_FULLY_STATIC_EXECUTABLES
    "MinGW: add -static to executables (fully static where supported; larger binaries, may need extra libs)"
    OFF
  )

  set(_cypha_mingw_link_opts "")
  if(CYPHA_MINGW_STATIC_CXX_RUNTIME)
    list(APPEND _cypha_mingw_link_opts -static-libgcc -static-libstdc++)
  endif()
  if(CYPHA_MINGW_FULLY_STATIC_EXECUTABLES)
    list(APPEND _cypha_mingw_link_opts -static)
  endif()

  message(
    STATUS
    "Cypha MinGW: CYPHA_MINGW_STATIC_CXX_RUNTIME=${CYPHA_MINGW_STATIC_CXX_RUNTIME} "
    "CYPHA_MINGW_FULLY_STATIC_EXECUTABLES=${CYPHA_MINGW_FULLY_STATIC_EXECUTABLES}"
  )
else()
  set(_cypha_mingw_link_opts "")
endif()

function(cypha_mingw_apply_exe_link_options target)
  if(NOT MINGW)
    return()
  endif()
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "cypha_mingw_apply_exe_link_options: not a target: ${target}")
  endif()
  if(_cypha_mingw_link_opts)
    target_link_options("${target}" PRIVATE ${_cypha_mingw_link_opts})
  endif()
endfunction()

# Parity fixtures live on the Windows filesystem at /mnt/c/... when configuring from WSL; Windows .exe
# cannot open those paths — rewrite to C:/... for add_test() arguments.
function(cypha_mingw_fix_golden_path_for_cross_host path_var)
  if(NOT (MINGW AND CMAKE_HOST_UNIX))
    return()
  endif()
  set(p "${${path_var}}")
  if(p MATCHES "^/mnt/c/")
    string(REGEX REPLACE "^/mnt/c/" "C:/" p "${p}")
  elseif(p MATCHES "^/mnt/([a-z])/")
    string(REGEX REPLACE "^/mnt/([a-z])/" "\\1:/" p "${p}")
  endif()
  set("${path_var}" "${p}" PARENT_SCOPE)
endfunction()
