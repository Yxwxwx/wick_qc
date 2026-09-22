# AO2MO retains its bounded two-pass GEMM kernels and shares the selected
# CBLAS provider. No libcint/HDF5 dependency reaches consumers of wick.hpp.
if(EINSUM_BACKEND MATCHES "^(MKL|OPENBLAS|NETLIB|BLIS)$")
  set(AO2MO_BLAS_BACKEND "${EINSUM_BACKEND}")
elseif(LAPACK_BACKEND MATCHES "^(MKL|OPENBLAS|NETLIB)$")
  set(AO2MO_BLAS_BACKEND "${LAPACK_BACKEND}")
else()
  message(FATAL_ERROR "AO2MO requires CBLAS: select MKL/OPENBLAS/NETLIB/BLIS for einsum or MKL/OPENBLAS/NETLIB for LAPACK")
endif()
if(NOT LAPACK_BACKEND STREQUAL "EIGEN" AND NOT LAPACK_BACKEND STREQUAL AO2MO_BLAS_BACKEND)
  message(FATAL_ERROR "AO2MO requires a single BLAS provider: einsum uses ${AO2MO_BLAS_BACKEND}, LAPACK uses ${LAPACK_BACKEND}")
endif()

add_library(wickqc_ao2mo INTERFACE)
add_library(wickqc::ao2mo ALIAS wickqc_ao2mo)
add_library(ao2mo::ao2mo ALIAS wickqc_ao2mo)
target_compile_features(wickqc_ao2mo INTERFACE cxx_std_20)
target_include_directories(wickqc_ao2mo INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
wickqc_link_cblas(wickqc_ao2mo ${AO2MO_BLAS_BACKEND})
if(AO2MO_BLAS_BACKEND STREQUAL "OPENBLAS" AND LAPACK_BACKEND STREQUAL "OPENBLAS")
  file(REAL_PATH "${OPENBLAS_LIBRARY}" ao2mo_blas_path)
  file(REAL_PATH "${LAPACK_OPENBLAS_LIBRARY}" ao2mo_lapack_path)
  if(NOT ao2mo_blas_path STREQUAL ao2mo_lapack_path)
    message(FATAL_ERROR "AO2MO and LAPACK must use the same OpenBLAS library")
  endif()
endif()

include(CheckCXXSourceRuns)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES wickqc_cblas_${AO2MO_BLAS_BACKEND})
# Recheck on reconfigure: the selected provider or its ABI may have changed.
unset(AO2MO_BLAS_ABI_OK CACHE)
check_cxx_source_runs("
#ifdef WICKQC_CBLAS_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <complex>
#if defined(OPENBLAS_USE64BITINT) || defined(MKL_ILP64)
#error LP64 CBLAS required
#endif
#ifdef WICKQC_CBLAS_MKL
static_assert(sizeof(MKL_INT) == 4);
#elif defined(WICKQC_CBLAS_OPENBLAS)
static_assert(sizeof(blasint) == 4);
#elif defined(WICKQC_CBLAS_BLIS)
static_assert(sizeof(f77_int) == 4);
#elif defined(WICKQC_CBLAS_NETLIB)
static_assert(sizeof(CBLAS_INT) == 4);
#endif
int main() {
  double a=3,b=4,c=0;
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,1,1,&a,1,&b,1,0,&c,1);
  std::complex<double> x(1,2),y(3,4),z,one(1),zero(0);
  cblas_zgemm(CblasRowMajor,CblasConjTrans,CblasNoTrans,1,1,1,&one,&x,1,&y,1,&zero,&z,1);
  return c != 12 || std::abs(z-std::conj(x)*y)>1e-14;
}" AO2MO_BLAS_ABI_OK)
cmake_pop_check_state()
if(NOT AO2MO_BLAS_ABI_OK)
  message(FATAL_ERROR "AO2MO requires working LP64 real/complex CBLAS; for MKL configure with -DMKL_INTERFACE=lp64")
endif()

find_package(HDF5 REQUIRED COMPONENTS C)
option(AO2MO_FETCH_LIBCINT "Build pinned libcint when no local library is supplied" ON)
if(TARGET cint::cint)
  set(AO2MO_CINT_LIBRARY cint::cint)
elseif(TARGET cint)
  set(AO2MO_CINT_LIBRARY cint)
elseif(CINT_INCLUDE_DIR OR CINT_LIBRARY OR NOT AO2MO_FETCH_LIBCINT)
  find_path(CINT_INCLUDE_DIR cint.h REQUIRED)
  find_library(CINT_LIBRARY NAMES cint REQUIRED)
  set(AO2MO_CINT_LIBRARY "${CINT_LIBRARY}")
else()
  include(FetchContent)
  FetchContent_Declare(libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG c72d30662ba08048940c9000cc352a220c867712 # v6.1.3
    UPDATE_DISCONNECTED TRUE)
  # libcint defaults BUILD_SHARED_LIBS to ON. Keep that default local instead
  # of changing the host's subsequently declared libraries via its cache.
  block()
    if(NOT DEFINED BUILD_SHARED_LIBS)
      set(BUILD_SHARED_LIBS ON)
    endif()
    FetchContent_MakeAvailable(libcint)
    target_include_directories(cint INTERFACE "$<BUILD_INTERFACE:${libcint_SOURCE_DIR}/include>")
  endblock()
  set(AO2MO_CINT_LIBRARY cint)
endif()
target_include_directories(wickqc_ao2mo SYSTEM INTERFACE ${CINT_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
target_link_libraries(wickqc_ao2mo INTERFACE OpenMP::OpenMP_CXX
  ${HDF5_LIBRARIES} ${AO2MO_CINT_LIBRARY} ${CMAKE_DL_LIBS})

# Hash the actual migrated source, including local edits. Historical benchmark
# fingerprints retain their original manifest and are never rewritten.
file(GLOB AO2MO_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ao2mo/*.hpp")
list(APPEND AO2MO_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/ao2mo.hpp")
list(SORT AO2MO_HEADERS)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${AO2MO_HEADERS})
set(AO2MO_SOURCE_DIGESTS "")
foreach(header IN LISTS AO2MO_HEADERS)
  file(SHA256 "${header}" digest)
  file(RELATIVE_PATH name "${CMAKE_CURRENT_SOURCE_DIR}/src" "${header}")
  string(APPEND AO2MO_SOURCE_DIGESTS "${name}:${digest}\n")
endforeach()
string(SHA256 AO2MO_SOURCE_FINGERPRINT "${AO2MO_SOURCE_DIGESTS}")
target_compile_definitions(wickqc_ao2mo INTERFACE AO2MO_SOURCE_FINGERPRINT="sha256:${AO2MO_SOURCE_FINGERPRINT}")
message(STATUS "WickQC AO2MO CBLAS provider: ${AO2MO_BLAS_BACKEND} (LP64)")

option(WICKQC_BUILD_AO2MO_CLI "Build the AO2MO command-line tool" ON)
if(WICKQC_BUILD_AO2MO_CLI)
  add_executable(ao2mo_cli src/ao2mo/main.cpp)
  set_target_properties(ao2mo_cli PROPERTIES OUTPUT_NAME wickqc_integrals
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  target_link_libraries(ao2mo_cli PRIVATE wickqc::ao2mo)
endif()
