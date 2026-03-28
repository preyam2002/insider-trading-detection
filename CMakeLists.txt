cmake_minimum_required(VERSION 3.10)
project(InsiderTradingDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(detector 
    cpp/main.cpp
)

target_include_directories(detector PRIVATE ${CMAKE_SOURCE_DIR})

if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(detector PRIVATE -O3 -march=native)
endif()
