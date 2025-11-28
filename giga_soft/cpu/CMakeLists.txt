configure_file("giga_cpu_version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/giga_cpu_version.cpp" @ONLY)

file(COPY "giga_cpu_version.h" DESTINATION "include")

option(ENABLE_OPTIMIZATION "Build an optimized (and multithreaded) CPU backend instead of the reference one (single threaded)." OFF)

find_package(GIGA REQUIRED)

set(GIGA_CPU_HEADER_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/include/giga_cpu_version.h
        giga_cpu.h
        )


set(GIGA_CPU_SOURCE_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/giga_cpu_version.cpp
        giga_cpu.cpp
        giga_cpu_add.cpp
        giga_cpu_conv2d.cpp
        giga_cpu_dense.cpp
        giga_cpu_memory.cpp
        giga_cpu_softmax.cpp
        giga_cpu_upsample.cpp
        )

add_library(GIGA_cpu SHARED ${GIGA_CPU_SOURCE_FILES} ${GIGA_CPU_HEADER_FILES})

if(ENABLE_OPTIMIZATION)
    find_package(OpenMP REQUIRED)
    target_compile_definitions(GIGA_cpu PRIVATE -DENABLE_OPTIMIZATION)
    target_include_directories(GIGA_cpu PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(GIGA_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -march=native")
endif(ENABLE_OPTIMIZATION)

include_directories(${GIGA_INCLUDE_DIR})
install(TARGETS GIGA_cpu LIBRARY DESTINATION lib COMPONENT bin)
