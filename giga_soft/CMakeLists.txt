cmake_minimum_required(VERSION 3.12)

project(GIGA_soft VERSION "1.0.0" LANGUAGES CXX C)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-write-strings")
set(CMAKE_C_FLAGS "-Wno-narrowing")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(GNUInstallDirs)

## Default build type is Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build mode" FORCE)
    MESSAGE("CMAKE_BUILD_TYPE set to " ${CMAKE_BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build mode" FORCE)
    MESSAGE("CMAKE_BUILD_TYPE is set to " ${CMAKE_BUILD_TYPE})
endif()

ADD_DEFINITIONS(-D__BUILD_TYPE__="${CMAKE_BUILD_TYPE}")

add_subdirectory(cpu)

# Use unit tests from GIGA
enable_testing()

macro(gen_test NAME)
    add_test(NAME giga_test_${NAME} COMMAND giga_test_${NAME})
    set_property(TEST giga_test_${NAME} PROPERTY ENVIRONMENT LD_PRELOAD=$<TARGET_FILE:GIGA_cpu> LD_LIBRARY_PATH=${GIGA_LIBRARY_DIR})
endmacro()

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
