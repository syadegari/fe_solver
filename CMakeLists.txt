cmake_minimum_required(VERSION 3.18)

project(superlu_mt_backend LANGUAGES C)

include(FetchContent)

# ---------------------------------------------------------------------------
# These are cache variables because SuperLU_MT reads them from its own
# CMakeLists.txt when FetchContent_MakeAvailable() adds it as a subproject.
# ---------------------------------------------------------------------------

set(THREAD_API PTHREAD CACHE STRING "" FORCE)

# Still needed by the current SuperLU_MT EXAMPLE/CMakeLists.txt, even though
# the bridge does not use PLAT.
set(PLAT _PTHREAD CACHE STRING "" FORCE)

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
# Your fork now provides CBLAS/CMakeLists.txt, so both paths should work.


# ---------------------------------------------------------------------------
# TODO: Change that if PR is accepted
# Fetch forked SuperLU_MT for now. 
# ---------------------------------------------------------------------------

FetchContent_Declare(
    superlu_mt

    GIT_REPOSITORY
        https://github.com/syadegari/superlu_mt.git

    GIT_TAG
        CMake-Support-for-CBLAS

    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(superlu_mt)


# ---------------------------------------------------------------------------
# Build our bridge.
#
# Give the CMake target a project-specific name so it cannot collide with any
# similarly named target that might exist in the SuperLU_MT repository.
# The produced file is still called libsuperlu_mt_bridge.so.
# ---------------------------------------------------------------------------

include (CTest)

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
