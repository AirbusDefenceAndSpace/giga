cmake_minimum_required(VERSION 3.15)

###############################################################################
# GIGA API
# (C) 2024 Airbus copyright all rights reserved
#
# Maintainers:
# Roland Brochard <roland.brochard@airbus.com>
# Lucas Marti <lucas.marti@airbus.com>
#
###############################################################################
# Software License:
#
#   Apache 2
#
###############################################################################
# Description:
#
#   CMake build and install script for the GIGA API
#
###############################################################################

project(GIGA VERSION "1.0.0" LANGUAGES CXX C)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-write-strings")
set(CMAKE_C_FLAGS "-Wno-narrowing")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

## Default build type is Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build mode" FORCE)
    MESSAGE("CMAKE_BUILD_TYPE set to " ${CMAKE_BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build mode" FORCE)
    MESSAGE("CMAKE_BUILD_TYPE is set to " ${CMAKE_BUILD_TYPE})
endif()


option(BUILD_TESTS "Compile tests" ON)
option(BUILD_BENCHMARKS "Compile benchmarks" ON)

option(INSTALL_TESTS "Intall tests" ON)
option(INSTALL_BENCHMARKS "Intall benchmarks" ON)

add_library(GIGA SHARED giga/giga.cpp)  # Target for the STUB library

configure_file(cmake/GIGAConfig.cmake.in
    "${PROJECT_BINARY_DIR}/GIGAConfig.cmake" @ONLY)
    
configure_file(cmake/GIGAConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/GIGAConfigVersion.cmake" @ONLY)

install(FILES giga/giga.h giga/float16.h
    DESTINATION include/giga COMPONENT dev)

install(PROGRAMS python/src/GIGA_export/nnef_to_giga.py python/src/GIGA_export/onnx_to_giga.py
    DESTINATION bin COMPONENT dev)

install(TARGETS GIGA
    LIBRARY DESTINATION lib COMPONENT dev)

install(FILES
"${PROJECT_BINARY_DIR}/GIGAConfig.cmake"
"${PROJECT_BINARY_DIR}/GIGAConfigVersion.cmake"
DESTINATION share/GIGA COMPONENT dev)

include_directories(${CMAKE_SOURCE_DIR})
if(BUILD_TESTS)
    enable_testing()

    macro(gen_test NAME)
        add_executable(giga_test_${NAME} "tests/${NAME}.cpp")
        target_link_libraries(giga_test_${NAME} GIGA)
        add_test(NAME giga_test_${NAME} COMMAND giga_test_${NAME})
        if(INSTALL_TESTS)
            install(TARGETS giga_test_${NAME} LIBRARY DESTINATION lib COMPONENT dev)
        endif(INSTALL_TESTS)
    endmacro()

    # Tests
    gen_test(float16)
    gen_test(comparison)
    gen_test(initialization)
    gen_test(allocation)
    gen_test(map_and_fill)
    gen_test(add)
    gen_test(conv2d)
    gen_test(dense)
    gen_test(softmax)
    gen_test(callback)
    gen_test(view)
    gen_test(reshape)
    gen_test(upsample)
    gen_test(avg_pooling)
endif(BUILD_TESTS)

if(BUILD_BENCHMARKS)

    macro(gen_benchmark NAME)
        add_executable(giga_benchmark_${NAME} "benchmarks/${NAME}.cpp")
        target_link_libraries(giga_benchmark_${NAME} GIGA)
        if(INSTALL_BENCHMARKS)
            install(TARGETS giga_benchmark_${NAME} LIBRARY DESTINATION lib COMPONENT dev)
        endif(INSTALL_BENCHMARKS)
    endmacro()

    # Benchmarks
    gen_benchmark(add)
    gen_benchmark(conv2d)
    gen_benchmark(dense)
    gen_benchmark(softmax)
    gen_benchmark(upsample)
endif(BUILD_BENCHMARKS)
