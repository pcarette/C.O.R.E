cmake_minimum_required(VERSION 3.31)

project(CORE_ENGINE
		VERSION 1.0.0
		DESCRIPTION "High Performance Genomics Engine"
		LANGUAGES CXX C CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if (result)
	message(STATUS "[INFO] Link Time Optimization (LTO) enabled for Release builds")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else ()
	message(WARNING "[WARN] LTO not supported by this compiler: ${output}")
endif ()

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(include)

add_library(cuda_core STATIC src/kernel.cu)

set_target_properties(cuda_core PROPERTIES
		CUDA_ARCHITECTURES "86;89"
)

target_compile_options(cuda_core PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:
		-O3
		-ffast-math
		-Wno-unknown-cuda-version
		-Wall -Wextra
		>
)

add_library(core_logic STATIC
		src/loader.cpp
		src/encoder.cpp
)

target_link_libraries(core_logic PUBLIC cuda_core OpenMP::OpenMP_CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	target_compile_options(core_logic PRIVATE
			-O3
			-march=native
			-mtune=native
			-mavx2
			-mbmi2
			-funroll-loops
			-Wall -Wextra -Werror
			-Wno-unused-parameter
	)
endif ()

add_executable(core_runner src/main.cpp)
target_link_libraries(core_runner PRIVATE cuda_core core_logic)

pybind11_add_module(core_engine src/python_bindings.cpp)
target_link_libraries(core_engine PRIVATE cuda_core core_logic)

message(STATUS "--------------------------------------------------")
message(STATUS "  C.O.R.E ENGINE - BUILD CONFIGURATION")
message(STATUS "--------------------------------------------------")
message(STATUS "  * System      : ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})")
message(STATUS "  * CXX Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  * CUDA Target : sm_86, sm_89 (Clang Native Mode)")
message(STATUS "  * Build Type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  * Components  : Core Logic, Runner, Python Module")
message(STATUS "--------------------------------------------------")
