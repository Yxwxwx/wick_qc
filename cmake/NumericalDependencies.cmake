include_guard(GLOBAL)

# Share one provider target between tensor GEMM and AO2MO's ordinary/adjoint
# GEMM. Their multiplication contracts differ; their CBLAS dependency does not.
function(wickqc_link_cblas target backend)
  set(provider wickqc_cblas_${backend})
  if(NOT TARGET ${provider})
    add_library(${provider} INTERFACE IMPORTED GLOBAL)
    if(backend STREQUAL "MKL")
      set(MKL_THREADING sequential CACHE STRING "MKL threading layer")
      find_package(MKL CONFIG REQUIRED HINTS "$ENV{MKLROOT}/lib/cmake/mkl")
      target_link_libraries(${provider} INTERFACE MKL::MKL)
    elseif(backend STREQUAL "NETLIB")
      wickqc_link_netlib(${provider} CBLAS)
    elseif(backend STREQUAL "OPENBLAS")
      set(OPENBLAS_ROOT "$ENV{HOME}/Applications/OpenBLAS/install" CACHE PATH "OpenBLAS prefix")
      find_path(OPENBLAS_INCLUDE_DIR cblas.h HINTS "${OPENBLAS_ROOT}/include" REQUIRED)
      find_library(OPENBLAS_LIBRARY openblas HINTS "${OPENBLAS_ROOT}/lib" "${OPENBLAS_ROOT}/lib64" REQUIRED)
      target_include_directories(${provider} SYSTEM INTERFACE "${OPENBLAS_INCLUDE_DIR}")
      target_link_libraries(${provider} INTERFACE "${OPENBLAS_LIBRARY}")
    elseif(backend STREQUAL "BLIS")
      set(BLIS_ROOT "$ENV{HOME}/Applications/blis/install" CACHE PATH "BLIS prefix (with CBLAS enabled)")
      find_path(BLIS_INCLUDE_DIR cblas.h HINTS "${BLIS_ROOT}/include" PATH_SUFFIXES blis REQUIRED)
      find_library(BLIS_LIBRARY blis HINTS "${BLIS_ROOT}/lib" "${BLIS_ROOT}/lib64" REQUIRED)
      target_include_directories(${provider} SYSTEM INTERFACE "${BLIS_INCLUDE_DIR}")
      target_link_libraries(${provider} INTERFACE "${BLIS_LIBRARY}")
    else()
      message(FATAL_ERROR "${backend} does not provide CBLAS")
    endif()
    target_compile_definitions(${provider} INTERFACE WICKQC_CBLAS_${backend})
  endif()
  target_link_libraries(${target} INTERFACE ${provider})
endfunction()

# Keep Netlib's dependency DSOs as direct dependencies: reference installations
# often have no RUNPATH of their own, and executable RUNPATH is not transitive.
# A link-library feature scopes --no-as-needed without changing other providers.
# Cache visibility also covers consumers in a parent add_subdirectory project.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_LINK_LIBRARY_USING_wickqc_netlib
    "LINKER:--push-state,--no-as-needed" "<LINK_ITEM>" "LINKER:--pop-state"
    CACHE INTERNAL "Retain Netlib's explicitly selected shared dependencies")
  set(CMAKE_LINK_LIBRARY_USING_wickqc_netlib_SUPPORTED TRUE CACHE INTERNAL "")
  list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    CMAKE_LINK_LIBRARY_USING_wickqc_netlib
    CMAKE_LINK_LIBRARY_USING_wickqc_netlib_SUPPORTED)
endif()

# Both selectors can request these dependencies independently.
function(wickqc_link_netlib target component)
  set(NETLIB_ROOT "$ENV{HOME}/Applications/netlib/install" CACHE PATH "Netlib LAPACK installation prefix")
  # Netlib exports vary across releases; locate the C interfaces and their
  # underlying libraries explicitly, as for the other CBLAS providers.
  find_library(WICKQC_NETLIB_BLAS_LIBRARY blas
    HINTS "${NETLIB_ROOT}/lib" "${NETLIB_ROOT}/lib64" REQUIRED)
  if(component STREQUAL "CBLAS")
    find_path(WICKQC_NETLIB_CBLAS_INCLUDE_DIR cblas.h HINTS "${NETLIB_ROOT}/include" REQUIRED)
    find_library(WICKQC_NETLIB_CBLAS_LIBRARY cblas
      HINTS "${NETLIB_ROOT}/lib" "${NETLIB_ROOT}/lib64" REQUIRED)
    target_include_directories(${target} SYSTEM INTERFACE "${WICKQC_NETLIB_CBLAS_INCLUDE_DIR}")
    set(libraries "${WICKQC_NETLIB_CBLAS_LIBRARY}" "${WICKQC_NETLIB_BLAS_LIBRARY}")
  elseif(component STREQUAL "LAPACKE")
    find_path(WICKQC_NETLIB_LAPACKE_INCLUDE_DIR lapacke.h HINTS "${NETLIB_ROOT}/include" REQUIRED)
    find_library(WICKQC_NETLIB_LAPACKE_LIBRARY lapacke
      HINTS "${NETLIB_ROOT}/lib" "${NETLIB_ROOT}/lib64" REQUIRED)
    find_library(WICKQC_NETLIB_LAPACK_LIBRARY lapack
      HINTS "${NETLIB_ROOT}/lib" "${NETLIB_ROOT}/lib64" REQUIRED)
    target_include_directories(${target} SYSTEM INTERFACE "${WICKQC_NETLIB_LAPACKE_INCLUDE_DIR}")
    set(libraries "${WICKQC_NETLIB_LAPACKE_LIBRARY}"
      "${WICKQC_NETLIB_LAPACK_LIBRARY}" "${WICKQC_NETLIB_BLAS_LIBRARY}")
  else()
    message(FATAL_ERROR "Unknown Netlib component '${component}'")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(JOIN libraries "," library_arguments)
    target_link_libraries(${target} INTERFACE "$<LINK_LIBRARY:wickqc_netlib,${library_arguments}>")
  else()
    target_link_libraries(${target} INTERFACE ${libraries})
  endif()
endfunction()

function(wickqc_link_eigen target)
  set(EIGEN_ROOT "$ENV{HOME}/Applications/eigen" CACHE PATH "Eigen source tree or installation prefix")
  if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 CONFIG QUIET HINTS "${EIGEN_ROOT}")
    if(NOT TARGET Eigen3::Eigen)
      # A downloaded Eigen source tree needs neither configuration nor building.
      find_path(WICKQC_EIGEN_INCLUDE_DIR Eigen/Core HINTS "${EIGEN_ROOT}"
        PATH_SUFFIXES include/eigen3 eigen3 REQUIRED)
      add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
      set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${WICKQC_EIGEN_INCLUDE_DIR}")
    endif()
  endif()
  target_link_libraries(${target} INTERFACE Eigen3::Eigen)
endfunction()
