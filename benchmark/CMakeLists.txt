############################################################################
# Copyright (c) xsimd-algorithm contributors                               #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.8)

project(xsimd-algorithm-benchmark)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    find_package(xsimd-algorithm REQUIRED CONFIG)
endif ()

find_package(benchmark REQUIRED)

set(XSIMD_ALGORITHM_BENCHMARKS
    main.cpp
    bench_math.cpp
    bench_map.cpp
)

add_executable(benchmark_xsimd_algorithm ${XSIMD_ALGORITHM_BENCHMARKS})
target_link_libraries(benchmark_xsimd_algorithm
    PRIVATE xsimd-algorithm xsimd::test-utils benchmark::benchmark)
