cmake_minimum_required(VERSION 3.25)

project(wick_qc VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
include(cmake/EinsumBackend.cmake)

add_library(wickqc_domains src/symbolic/index_domain.cpp)
target_include_directories(wickqc_domains PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(wickqc_domains PUBLIC cxx_std_20)

# Runtime data contracts, without the symbolic engine.
add_library(wickqc_numeric src/runtime/tensor_binding.cpp src/runtime/numeric_kernel.cpp)
target_link_libraries(wickqc_numeric PUBLIC wickqc_domains wickqc_ndarray)

add_library(wickqc_symbolic src/symbolic/wick.cpp src/symbolic/serialization.cpp
                            src/equation/equation.cpp src/equation/graph.cpp
                            src/einsum/einsum.cpp src/einsum/numpy_graph.cpp
                            src/method/ic_nevpt2.cpp src/method/sc_nevpt2.cpp
                            src/method/ccsd.cpp src/method/ghf.cpp
                            src/method/uga_ccsd.cpp src/method/spatial_cc.cpp
                            src/method/spatial_mp.cpp)
target_include_directories(
  wickqc_symbolic PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(wickqc_symbolic PUBLIC wickqc_domains)

add_library(wickqc_runtime src/runtime/ndarray_executor.cpp)
target_link_libraries(wickqc_runtime PUBLIC wickqc_symbolic wickqc_numeric)

add_library(wickqc_codegen src/codegen/cpp_emitter.cpp)
target_link_libraries(wickqc_codegen PUBLIC wickqc_runtime)
add_executable(generate_cpp tools/generate_cpp.cpp)
target_link_libraries(generate_cpp PRIVATE wickqc_codegen)
include(cmake/PrecompileMethods.cmake)

add_library(wickqc_methods src/runtime/spatial_evaluator.cpp)
target_link_libraries(wickqc_methods PUBLIC wickqc_runtime wickqc_precompiled)

add_library(wickqc_rhf src/method/rhf_data.cpp)
target_link_libraries(wickqc_rhf PUBLIC wickqc_numeric)

add_executable(evaluate_mp2 tools/evaluate_mp2.cpp)
target_link_libraries(evaluate_mp2 PRIVATE wickqc_runtime)

add_executable(wick_qc src/main.cpp)
target_link_libraries(wick_qc PRIVATE wickqc_symbolic)

add_executable(generate_ic_nevpt2 tools/generate_ic_nevpt2.cpp)
target_link_libraries(generate_ic_nevpt2 PRIVATE wickqc_symbolic)

add_executable(generate_ccsd tools/generate_ccsd.cpp)
target_link_libraries(generate_ccsd PRIVATE wickqc_symbolic)

add_executable(generate_ghf tools/generate_ghf.cpp)
target_link_libraries(generate_ghf PRIVATE wickqc_symbolic)
