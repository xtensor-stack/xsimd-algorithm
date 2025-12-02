cmake_minimum_required(VERSION 3.15..3.29)
project(xsimd-algorithm CXX)

set(XSIMDALGO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Dependencies
# ============

set(xsimd_REQUIRED_VERSION 13.2.0)
if(TARGET xsimd)
    set(xsimd_VERSION ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
    if(${xsimd_VERSION} VERSION_LESS ${xsimd_REQUIRED_VERSION})
        message(ERROR "Mismatch xsimd versions. Found '${xsimd_VERSION}' but requires: '${xsimd_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xsimd v${xsimd_VERSION}")
    endif()
else()
    find_package(xsimd ${xsimd_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
endif()

# Build
# =====

set(XSIMDALGO_HEADERS
    ${XSIMDALGO_INCLUDE_DIR}/xsimd_algo/algorithms.hpp
)

add_library(xsimd-algorithm INTERFACE)

target_include_directories(xsimd-algorithm INTERFACE
    $<BUILD_INTERFACE:${XSIMDALGO_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(xsimd-algorithm INTERFACE cxx_std_20)
target_link_libraries(xsimd-algorithm INTERFACE xsimd)

OPTION(BUILD_TESTS "xsimd-algorithm test suite" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xsimd-algorithm
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(DIRECTORY ${XSIMDALGO_INCLUDE_DIR}/xsimd_algo
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(XSIMDALGO_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xsimd-algorithmConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XSIMDALGO_CMAKECONFIG_INSTALL_DIR})
