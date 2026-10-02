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

# Regular benchmark of algorithm implemented here
add_executable(
    benchmark_xsimd_algorithm
    main.cpp
    bench_map.cpp
)
target_link_libraries(
    benchmark_xsimd_algorithm
    PRIVATE xsimd-algorithm xsimd::test-utils benchmark::benchmark
)

# Benchmark actually targeting xsimd functionalities (hot loop in L1 cache,
# no header and trailers). Should probably be moved to xsimd itself.
add_executable(
    benchmark_xsimd
    main.cpp
    bench_math.cpp
)
target_link_libraries(
    benchmark_xsimd
    PRIVATE xsimd-algorithm xsimd::test-utils benchmark::benchmark
)
