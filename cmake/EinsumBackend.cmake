include(${CMAKE_CURRENT_LIST_DIR}/NumericalDependencies.cmake)
set(EINSUM_BACKEND "NATIVE" CACHE STRING "Einsum backend: TBLIS, BLIS, MKL, OPENBLAS, NETLIB, EIGEN, NATIVE, SIMPLE")
set_property(CACHE EINSUM_BACKEND PROPERTY STRINGS TBLIS BLIS MKL OPENBLAS NETLIB EIGEN NATIVE SIMPLE)
string(TOUPPER "${EINSUM_BACKEND}" EINSUM_BACKEND)
set(_wickqc_einsum_backends TBLIS BLIS MKL OPENBLAS NETLIB EIGEN NATIVE SIMPLE)
if(NOT EINSUM_BACKEND IN_LIST _wickqc_einsum_backends)
  message(FATAL_ERROR "Invalid EINSUM_BACKEND='${EINSUM_BACKEND}'; choose ${_wickqc_einsum_backends}")
endif()

add_library(wickqc_ndarray INTERFACE)
target_include_directories(wickqc_ndarray INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(wickqc_ndarray INTERFACE cxx_std_20)
target_link_libraries(wickqc_ndarray INTERFACE OpenMP::OpenMP_CXX)
target_compile_definitions(wickqc_ndarray INTERFACE WICKQC_USE_${EINSUM_BACKEND})

if(EINSUM_BACKEND STREQUAL "TBLIS")
  # Use the installed package and its transitive dependencies (TCI/MArray).
  # CMAKE_PREFIX_PATH or TBLIS_ROOT can select a non-system installation.
  find_package(TBLIS CONFIG REQUIRED)
  target_link_libraries(wickqc_ndarray INTERFACE TBLIS::tblis)
elseif(EINSUM_BACKEND MATCHES "^(MKL|NETLIB|OPENBLAS|BLIS)$")
  wickqc_link_cblas(wickqc_ndarray ${EINSUM_BACKEND})
elseif(EINSUM_BACKEND STREQUAL "EIGEN")
  wickqc_link_eigen(wickqc_ndarray)
endif()
message(STATUS "WickQC NDArray einsum backend: ${EINSUM_BACKEND}")
