cmake_minimum_required(VERSION 3.25)

project(wick_qc VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
include(cmake/EinsumBackend.cmake)
include(cmake/LapackBackend.cmake)
include(cmake/TransposeBackend.cmake)

option(WICKQC_ENABLE_AO2MO "Enable the libcint/HDF5 AO2MO module" OFF)
if(WICKQC_ENABLE_AO2MO)
  include(cmake/AO2MO.cmake)
endif()

# All core modules are header-only. These interface targets express optional
# backend dependencies without compiling project implementation libraries.
add_library(wickqc INTERFACE)
add_library(wickqc::wickqc ALIAS wickqc)
target_link_libraries(wickqc INTERFACE wickqc_ndarray wickqc_lapack)
if(WICKQC_ENABLE_AO2MO)
  target_link_libraries(wickqc INTERFACE wickqc_ao2mo)
  target_compile_definitions(wickqc INTERFACE WICKQC_ENABLE_AO2MO=1)
endif()
# Symbolic consumers do not acquire numerical backend dependencies.
add_library(wickqc_domains INTERFACE)
target_include_directories(wickqc_domains INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(wickqc_domains INTERFACE cxx_std_20)
add_library(wickqc_symbolic INTERFACE)
target_link_libraries(wickqc_symbolic INTERFACE wickqc_domains)
foreach(module IN ITEMS numeric runtime codegen methods rhf)
  add_library(wickqc_${module} INTERFACE)
  target_link_libraries(wickqc_${module} INTERFACE wickqc_ndarray)
endforeach()

add_executable(generate_cpp src/codegen/main.cpp)
target_link_libraries(generate_cpp PRIVATE wickqc::wickqc)
include(cmake/PrecompileMethods.cmake)

include(CTest)
option(WICKQC_BUILD_EXAMPLES "Build API and code-generation examples" ON)
# Generated kernels are shared by examples and tests. Tests also work with
# WICKQC_BUILD_EXAMPLES=OFF and never invoke a reference Python package.
if(WICKQC_BUILD_EXAMPLES OR BUILD_TESTING)
  include(cmake/MethodKernels.cmake)
endif()
if(WICKQC_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()
if(BUILD_TESTING)
  add_subdirectory(unit_test)
  if(WICKQC_ENABLE_AO2MO)
    add_subdirectory(test)
  endif()
endif()
