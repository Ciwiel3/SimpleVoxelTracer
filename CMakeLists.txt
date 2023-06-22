cmake_minimum_required(VERSION 3.21)
project (SimpleVoxelTracer C)

set(CMAKE_C_STANDARD 17)

set(CMAKE_C_FLAGS "-std=c17 -mavx2 -mfma")

set(CMAKE_C_FLAGS_RELEASE "-O3 -static")
set(CMAKE_C_FLAGS_DEBUG " \
    -g3 \
    -O0 \
    \
    -Wall \
    -Wshadow \
    -Warray-bounds \
    -Wcast-align=strict \
    -Wstrict-prototypes \
    -Wmissing-prototypes \
    -Wwrite-strings \
    -Winit-self \
    -Wold-style-definition \
    \
    -Wno-missing-braces \
    -Wno-unused-function \
    -Wno-incompatible-pointer-types \
    -Wno-uninitialized \
    -Wno-unused-but-set-variable \
    -Wno-unused-variable \
    -Wno-format-zero-length \
    -Wno-unused-parameter \
    -Wno-comment \
    ")

# IPO / LTO
#include(CheckIPOSupported)
#check_ipo_supported(RESULT supported OUTPUT error)
#cmake_policy(SET CMP0069 NEW)
#set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
#if( supported )
#    message(STATUS "IPO / LTO enabled")
#    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
#else()
#    message(STATUS "IPO / LTO not supported: <${error}>")
#endif()

include_directories(inc)

add_executable(SimpleVoxelTracer
        src/main.c
        src/graphics.c
        src/terrain.c
        src/gllib.c)
target_precompile_headers(SimpleVoxelTracer PUBLIC inc/pch.h)

# fast noise
include_directories(ext/FastNoise)
set_source_files_properties(/ext/FastNoise/FastNoiseLite.h PROPERTIES COMPILE_FLAGS -w)

# stb include
include_directories(ext/stb)
set_source_files_properties(/ext/stb/stb_include.h PROPERTIES COMPILE_FLAGS -w)

# glad
include_directories(ext/glad/include)
add_library(glad STATIC ext/glad/src/glad.c)
target_link_libraries(SimpleVoxelTracer glad)

# GLFW
include_directories("ext/glfw-3.3.2/include")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/glfw-3.3.2)
target_link_libraries(SimpleVoxelTracer glfw)

# cpmath
include_directories("ext/cplib/")
set_source_files_properties(/ext/cplib/cpmath.h PROPERTIES COMPILE_FLAGS -w)
