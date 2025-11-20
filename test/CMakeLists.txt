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


project(xsimd-algorithm-test)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    enable_testing()
    find_package(xsimd-algorithm REQUIRED CONFIG)
    message(STATUS "${xsimd-algorithm_INCLUDE_DIRS}")
endif ()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

OPTION(XSIMD_ENABLE_WERROR "Turn on -Werror" OFF)


if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4005 /wd4146 /wd4800")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang AND MSVC AND WIN32) # We are using clang-cl
    add_compile_options(/EHsc /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

set(XSIMD_ALGORITHM_TESTS
    main.cpp
    test_algorithms.cpp
)

add_executable(test_xsimd_algorithm ${XSIMD_ALGORITHM_TESTS})# ${XSIMD_ALGORITHM_HEADERS})
target_link_libraries(test_xsimd_algorithm PRIVATE xsimd-algorithm)

option(DOWNLOAD_DOCTEST OFF)
find_package(doctest QUIET)
if (doctest_FOUND)
    set(DOCTEST_MINIMAL_VERSION 2.4.9)
    if (doctest_VERSION VERSION_LESS DOCTEST_MINIMAL_VERSION)
        message(FATAL_ERROR "Requires doctest >= ${DOCTEST_MINIMAL_VERSION}")
    endif()
    target_link_libraries(test_xsimd_algorithm PRIVATE doctest::doctest)
elseif(DOWNLOAD_DOCTEST)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doctest")
    file(DOWNLOAD
        "https://github.com/doctest/doctest/releases/download/v2.4.9/doctest.h"
        "${CMAKE_CURRENT_BINARY_DIR}/doctest/doctest.h"
        STATUS DOWNLOAD_DOCTEST_STATUS)
    list(GET DOWNLOAD_DOCTEST_STATUS 0 DOWNLOAD_DOCTEST_STATUS_CODE)
    list(GET DOWNLOAD_DOCTEST_STATUS 1 DOWNLOAD_DOCTEST_ERROR_MESSAGE)
    if(${DOWNLOAD_DOCTEST_STATUS_CODE} EQUAL 0)
        message(STATUS "Successfully downloaded doctest.h")
    else()
        message(FATAL_ERROR "Error occurred during download of doctest: ${DOWNLOAD_DOCTEST_ERROR_MESSAGE}")
    endif()
    target_include_directories(test_xsimd_algorithm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
else()
    message(FATAL_ERROR "
    Cannot find required doctest component.
    Please either set CMAKE_PREFIX_PATH to the location of doctestConfig.cmake,
    or set DOWNLOAD_DOCTEST=ON")
endif()

add_test(NAME test_xsimd_algorithm COMMAND test_xsimd_algorithm)

if (CROSS_COMPILE_ARM)
    add_custom_target(xtest COMMAND qemu-arm -L /usr/arm-linux-gnueabi/ test_xsimd_algorithm DEPENDS test_xsimd_algorithm)
else()
    add_custom_target(xtest COMMAND test_xsimd_algorithm DEPENDS test_xsimd_algorithm)
endif()

if (XSIMD_ENABLE_WERROR)
    target_compile_options(test_xsimd_algorithm PRIVATE -Werror -Wall -DXSIMD_SKIP_ON_WERROR)
endif()
