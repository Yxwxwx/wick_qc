set(EINSUM_BACKEND "BLAS" CACHE STRING "Tensor contraction backend: BLAS or TBLIS")
set_property(CACHE EINSUM_BACKEND PROPERTY STRINGS BLAS TBLIS)
string(TOUPPER "${EINSUM_BACKEND}" EINSUM_BACKEND)
if(NOT EINSUM_BACKEND MATCHES "^(BLAS|TBLIS)$")
  message(FATAL_ERROR "Invalid EINSUM_BACKEND='${EINSUM_BACKEND}'; choose BLAS or TBLIS")
endif()

add_library(wickqc_ndarray INTERFACE)
target_include_directories(wickqc_ndarray INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_compile_features(wickqc_ndarray INTERFACE cxx_std_20)
target_link_libraries(wickqc_ndarray INTERFACE OpenMP::OpenMP_CXX)

if(EINSUM_BACKEND STREQUAL "TBLIS")
  # Use the installed package and its transitive dependencies (TCI/MArray).
  # CMAKE_PREFIX_PATH or TBLIS_ROOT can select a non-system installation.
  find_package(TBLIS CONFIG REQUIRED)
  target_link_libraries(wickqc_ndarray INTERFACE TBLIS::tblis)
  target_compile_definitions(wickqc_ndarray INTERFACE WICKQC_USE_TBLIS)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/BlasBackend.cmake)
endif()
message(STATUS "WickQC NDArray einsum backend: ${EINSUM_BACKEND}")
