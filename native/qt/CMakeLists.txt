# Optional Qt 6 — included only when parent sets CYPHA_BUILD_QT=ON.
# cypha_qt_stub: Core-only link check + buffer load.
# cypha_qt_shell: Widgets desktop shell (load .cypha, native classify row).

# Ubuntu 24.04 ships Qt 6.4.x; 6.5+ is nice-to-have for future APIs.
find_package(Qt6 6.4 QUIET COMPONENTS Core Widgets Network)
if(NOT Qt6_FOUND)
  message(WARNING "CYPHA_BUILD_QT is ON but Qt6 Core/Widgets/Network were not found — skipping Qt targets")
  return()
endif()

add_executable(cypha_qt_stub
  src/main.cpp
)
target_link_libraries(cypha_qt_stub PRIVATE Qt6::Core cypha_core)

add_executable(cypha_qt_shell
  src/shell_main.cpp
)
target_link_libraries(cypha_qt_shell PRIVATE Qt6::Widgets Qt6::Network cypha_core)

# M6: Optionally embed the experiment DB directly into the Qt shell.
# Requires SQLite (same detection logic as native/CMakeLists.txt, which already ran).
# _cypha_exp_sqlite_link is set by the parent before add_subdirectory(qt).
if(_cypha_exp_sqlite_link)
  target_sources(cypha_qt_shell PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/experiment_db.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/experiment_db_crud.cpp"
  )
  target_link_libraries(cypha_qt_shell PRIVATE ${_cypha_exp_sqlite_link})
  target_compile_definitions(cypha_qt_shell PRIVATE CYPHA_SHELL_EXPERIMENT_DB=1)
  message(STATUS "cypha_qt_shell: experiment DB enabled (SQLite via ${_cypha_exp_sqlite_link})")
else()
  message(STATUS "cypha_qt_shell: experiment DB disabled (no SQLite)")
endif()

if(CYPHA_QT_CHARTS)
  find_package(Qt6 6.4 QUIET COMPONENTS Charts)
  if(Qt6Charts_FOUND)
    target_link_libraries(cypha_qt_shell PRIVATE Qt6::Charts)
    target_compile_definitions(cypha_qt_shell PRIVATE CYPHA_SHELL_QT_CHARTS=1)
  else()
    message(WARNING "CYPHA_QT_CHARTS is ON but Qt6 Charts was not found — cypha_qt_shell uses the painted loss widget")
  endif()
endif()

if(MSVC)
  target_compile_options(cypha_qt_stub PRIVATE /W4 /permissive-)
  target_compile_options(cypha_qt_shell PRIVATE /W4 /permissive-)
else()
  target_compile_options(cypha_qt_stub PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(cypha_qt_shell PRIVATE -Wall -Wextra -Wpedantic)
endif()
