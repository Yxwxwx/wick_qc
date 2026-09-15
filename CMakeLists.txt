cmake_minimum_required(VERSION 3.25)

project(wick_qc VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
include(cmake/EinsumBackend.cmake)

add_library(wickqc_symbolic src/symbolic/wick.cpp src/symbolic/serialization.cpp
                            src/equation/equation.cpp src/equation/graph.cpp
                            src/einsum/einsum.cpp src/einsum/numpy_graph.cpp
                            src/method/ic_nevpt2.cpp src/method/sc_nevpt2.cpp
                            src/method/ccsd.cpp src/method/ghf.cpp
                            src/method/uga_ccsd.cpp src/method/spatial_cc.cpp
                            src/method/spatial_mp.cpp)
target_include_directories(
  wickqc_symbolic PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

add_library(wickqc_runtime src/runtime/ndarray_executor.cpp)
target_link_libraries(wickqc_runtime PUBLIC wickqc_symbolic wickqc_ndarray)

add_executable(evaluate_mp2 tools/evaluate_mp2.cpp)
target_link_libraries(evaluate_mp2 PRIVATE wickqc_runtime)

add_executable(test_h2o_pyscf unit_test/test_h2o_pyscf.cpp)
target_link_libraries(test_h2o_pyscf PRIVATE wickqc_runtime)

add_executable(wick_qc src/main.cpp)
target_link_libraries(wick_qc PRIVATE wickqc_symbolic)

add_executable(generate_ic_nevpt2 tools/generate_ic_nevpt2.cpp)
target_link_libraries(generate_ic_nevpt2 PRIVATE wickqc_symbolic)

add_executable(generate_ccsd tools/generate_ccsd.cpp)
target_link_libraries(generate_ccsd PRIVATE wickqc_symbolic)

add_executable(generate_ghf tools/generate_ghf.cpp)
target_link_libraries(generate_ghf PRIVATE wickqc_symbolic)

# Complete method examples depend only on the project library.
add_executable(test_spatial_mp unit_test/test_spatial_mp.cpp)
target_link_libraries(test_spatial_mp PRIVATE wickqc_symbolic)

add_executable(test_spatial_cc unit_test/test_spatial_cc.cpp)
target_link_libraries(test_spatial_cc PRIVATE wickqc_symbolic)

add_executable(test_nevpt2_methods unit_test/test_spatial_nevpt2.cpp)
target_link_libraries(test_nevpt2_methods PRIVATE wickqc_symbolic)

include(CTest)
option(WICKQC_BUILD_NUMERICAL_TESTS "Build the optional local NDArray integration tests" OFF)
if(BUILD_TESTING AND WICKQC_BUILD_NUMERICAL_TESTS)
  add_subdirectory(tests/numerical)
endif()
option(WICKQC_BUILD_REGRESSION_TESTS "Build the optional local regression suite" OFF)
if(BUILD_TESTING AND WICKQC_BUILD_REGRESSION_TESTS)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    message(FATAL_ERROR "WICKQC_BUILD_REGRESSION_TESTS requires the local tests/ directory")
  endif()
  add_subdirectory(tests)
endif()
