############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         #
# Martin Renou                                                             #
# Copyright (c) QuantStack                                                 #
# Copyright (c) Serge Guelton                                              #
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

add_executable(bench_algorithms bench_algorithms.cpp)

target_link_libraries(bench_algorithms PRIVATE xsimd-algorithm benchmark::benchmark)

target_compile_options(bench_algorithms PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>)

set_target_properties(bench_algorithms PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark)
