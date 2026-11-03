cmake_minimum_required(VERSION 3.18)

project(superlu_mt_backend LANGUAGES C)

include(FetchContent)

# ---------------------------------------------------------------------------
# These are cache variables because SuperLU_MT reads them from its own
# CMakeLists.txt when FetchContent_MakeAvailable() adds it as a subproject.
# ---------------------------------------------------------------------------
set(THREAD_API PTHREAD CACHE STRING "" FORCE)

# PLAT no longer needs to be set here: since SuperLU_MT v4.1.0 the upstream
# CMakeLists.txt derives it from THREAD_API for its EXAMPLE/ and TESTING/
# subdirectories.

set(enable_fortran OFF CACHE BOOL "" FORCE)
set(enable_examples OFF CACHE BOOL "" FORCE)
set(enable_tests ON CACHE BOOL "" FORCE)
set(enable_doc OFF CACHE BOOL "" FORCE)

# We want SuperLU_MT itself statically linked into our shared bridge.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Python bridge currently assumes SuperLU_MT int_t == int32.
set(LONGINT OFF CACHE BOOL "" FORCE)

# IMPORTANT:
# Do NOT set enable_internal_blaslib here.
#
# SuperLU_MT should:
#   1. discover an installed BLAS automatically;
#   2. fall back to its bundled CBLAS if none is found.
#

# ---------------------------------------------------------------------------
# Fetch upstream SuperLU_MT, pinned to the commit tagged v4.1.0.
#
# The pin is the full commit hash rather than the tag name: Git tags can be
# moved or deleted, a hash cannot. GIT_SHALLOW is deliberately not set, since
# CMake's shallow clone can only check out a hash that is still the tip of a
# remote branch, which stops being true as soon as upstream commits again.
#
# v4.1.0 is the first release that contains both fixes this project used to
# take from a fork (syadegari/superlu_mt, branch fe-integration):
#   * PR #33: p?gstrf_init() initializes options->fact and options->trans.
#   * PR #34: CMake build support for the bundled CBLAS
#             (enable_internal_blaslib=ON).
# It additionally contains:
#   * PR #35: PLAT is derived from THREAD_API, so EXAMPLE/ and TESTING/ link
#             the built superlu_mt_PTHREAD target.
#   * PR #36: per-thread workspace overlap fix for user-supplied work[]
#             (lwork > 0); the bridge passes lwork = 0 but the fix removes
#             a race in the library's own test suite.
#   * PR #37: L/U storage estimates (fill_lusup, fill_ucol, fill_lsub) are
#             settable through superlumt_options_t instead of only via
#             sp_ienv(); p?gstrf_init() sets them to 0 (sp_ienv defaults).
# ---------------------------------------------------------------------------
FetchContent_Declare(
    superlu_mt
    GIT_REPOSITORY
        https://github.com/xiaoyeli/superlu_mt.git
    GIT_TAG
        f2e50bfcbb7b20ef304c84cf92c038c283019863  # v4.1.0
)
FetchContent_MakeAvailable(superlu_mt)

# ---------------------------------------------------------------------------
# Build our bridge.
#
# Give the CMake target a project-specific name so it cannot collide with any
# similarly named target that might exist in the SuperLU_MT repository.
# The produced file is still called libsuperlu_mt_bridge.so.
# ---------------------------------------------------------------------------
include(CTest)
add_library(
    project_superlu_mt_bridge
    SHARED
    native/superlu_mt.c
)
target_include_directories(
    project_superlu_mt_bridge
    PRIVATE
        ${superlu_mt_SOURCE_DIR}/SRC
)
target_link_libraries(
    project_superlu_mt_bridge
    PRIVATE
        superlu_mt_PTHREAD
)
set_target_properties(
    project_superlu_mt_bridge
    PROPERTIES
        OUTPUT_NAME superlu_mt_bridge
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native"
)

# ---------------------------------------------------------------------------
# Optional standalone reproducer for the sporadic wrong solutions that
# multithreaded SuperLU_MT returns on small systems; see
# native/superlu_mt_race/README.md.  Not built by default and not a test,
# because the failure is probabilistic.
# ---------------------------------------------------------------------------
add_executable(
    superlu_mt_race_repro
    EXCLUDE_FROM_ALL
    native/superlu_mt_race/superlu_mt_race_repro.c
)
target_include_directories(
    superlu_mt_race_repro
    PRIVATE
        ${superlu_mt_SOURCE_DIR}/SRC
)
target_link_libraries(
    superlu_mt_race_repro
    PRIVATE
        superlu_mt_PTHREAD
        m
)
set_target_properties(
    superlu_mt_race_repro
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native"
)
