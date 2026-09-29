############################################################################
# Copyright (c) xsimd-algorithm contributors                               #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.8)

project(xsimd-algorithm-test-utils)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    find_package(xsimd-algorithm REQUIRED CONFIG)
endif ()

add_library(xsimd-algorithm-test-utils INTERFACE)
add_library(xsimd::test-utils ALIAS xsimd-algorithm-test-utils)

target_include_directories(xsimd-algorithm-test-utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(xsimd-algorithm-test-utils INTERFACE xsimd-algorithm)
