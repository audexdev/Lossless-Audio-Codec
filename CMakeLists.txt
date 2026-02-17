cmake_minimum_required(VERSION 3.16)
project(LAC VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LAC_BUILD_TESTS "Build LAC tests" OFF)

add_library(lac STATIC
    src/codec/bitstream/bit_reader.cpp
    src/codec/bitstream/bit_writer.cpp
    src/codec/block/decoder.cpp
    src/codec/block/encoder.cpp
    src/codec/lac/decoder.cpp
    src/codec/lac/encoder.cpp
    src/codec/lpc/lpc.cpp
    src/codec/rice/rice.cpp
    src/codec/simd/neon.cpp
    src/io/wav_io.cpp
    src/utils/logger.cpp

    src/codec/bitstream/bit_reader.hpp
    src/codec/bitstream/bit_writer.hpp
    src/codec/block/constants.hpp
    src/codec/block/decoder.hpp
    src/codec/block/encoder.hpp
    src/codec/frame/frame_header.hpp
    src/codec/lac/decoder.hpp
    src/codec/lac/encoder.hpp
    src/codec/lac/thread_collector.hpp
    src/codec/lpc/lpc.hpp
    src/codec/rice/rice.hpp
    src/codec/simd/neon.hpp
    src/io/wav_io.hpp
    src/utils/endian.hpp
    src/utils/logger.hpp
    include/lac/lac.hpp
    include/lac/version.hpp
)

target_include_directories(lac PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(lac_cli
    src/main.cpp
)

target_link_libraries(lac_cli PRIVATE lac)

if(LAC_BUILD_TESTS)
    enable_testing()
    add_executable(lac_tests
        tests/test_lpc.cpp
        tests/test_e2e.cpp
        tests/test_partitioning.cpp
        tests/test_predictors.cpp
        tests/test_zerorun.cpp
    )
    target_link_libraries(lac_tests PRIVATE lac)
    target_compile_definitions(lac_tests PRIVATE LAC_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")
    add_test(NAME lac_tests COMMAND lac_tests)
endif()
