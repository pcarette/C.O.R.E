cmake_minimum_required(VERSION 3.31)

project(core CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(BEFORE SYSTEM "${CMAKE_SOURCE_DIR}/cuda_fix")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options(-O3 -Wall -Wextra -Werror -march=native -mtune=native -fopenmp)
endif ()

set(CMAKE_CUDA_FLAGS "-O3 --cuda-gpu-arch=sm_86 --cuda-gpu-arch=sm_89 -Wno-unknown-cuda-version -ffast-math")
set(CMAKE_CUDA_ARCHITECTURES 86 89)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(cuda_core STATIC src/kernel.cu)
target_include_directories(cuda_core PUBLIC include)
set_target_properties(cuda_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(core_engine
		src/main.cpp
		src/loader.cpp
		src/encoder.cpp
)

target_link_libraries(core_engine PRIVATE cuda_core OpenMP::OpenMP_CXX)
target_include_directories(core_engine PRIVATE include)

message(STATUS "--- C.O.R.E CONFIGURATION ---")
message(STATUS "Mode: LOCAL SYSTEM (No Internet Fetch)")
message(STATUS "Target GPU: Ampere (sm_86) & Ada Lovelace (sm_89)")
message(STATUS "-------------------------------")
