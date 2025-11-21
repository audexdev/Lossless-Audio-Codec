cmake_minimum_required(VERSION 3.16)
project(LAC VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LAC_BUILD_TESTS "Build LAC tests" OFF)
option(LAC_ENABLE_ZERORUN "Enable zero-run residual coder" ON)
option(LAC_ENABLE_PARTITIONING "Enable residual partitioning" ON)

# Source files
file(GLOB_RECURSE LAC_SOURCES
    src/*.cpp
)

file(GLOB_RECURSE LAC_HEADERS
    src/*.hpp
    include/lac/*.hpp
)

add_library(lac STATIC
    ${LAC_SOURCES}
    ${LAC_HEADERS}
)

target_include_directories(lac PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(lac_cli
    src/main.cpp
)

target_link_libraries(lac_cli PRIVATE lac)

if(LAC_ENABLE_ZERORUN)
    target_compile_definitions(lac PUBLIC LAC_ENABLE_ZERORUN=1)
    target_compile_definitions(lac_cli PUBLIC LAC_ENABLE_ZERORUN=1)
endif()
if(LAC_ENABLE_PARTITIONING)
    target_compile_definitions(lac PUBLIC LAC_ENABLE_PARTITIONING=1)
    target_compile_definitions(lac_cli PUBLIC LAC_ENABLE_PARTITIONING=1)
endif()

if(LAC_BUILD_TESTS)
    enable_testing()
    file(GLOB TEST_SOURCES
        tests/*.cpp
    )
    add_executable(lac_tests
        ${TEST_SOURCES}
    )
    target_link_libraries(lac_tests PRIVATE lac)
    if(LAC_ENABLE_ZERORUN)
        target_compile_definitions(lac_tests PUBLIC LAC_ENABLE_ZERORUN=1)
    endif()
    if(LAC_ENABLE_PARTITIONING)
        target_compile_definitions(lac_tests PUBLIC LAC_ENABLE_PARTITIONING=1)
    endif()
    add_test(NAME lpc_tests COMMAND lac_tests)
endif()
