cmake_minimum_required(VERSION 3.16)
project(LAC VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LAC_BUILD_TESTS "Build LAC tests" OFF)

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