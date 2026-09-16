include_guard(GLOBAL)

# Keep Netlib's dependency DSOs as direct dependencies: reference installations
# often have no RUNPATH of their own, and executable RUNPATH is not transitive.
# A link-library feature scopes --no-as-needed without changing other providers.
# Cache visibility also covers consumers in a parent add_subdirectory project.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_LINK_LIBRARY_USING_wickqc_netlib
    "LINKER:--push-state,--no-as-needed" "<LINK_ITEM>" "LINKER:--pop-state"
    CACHE INTERNAL "Retain Netlib's explicitly selected shared dependencies")
  set(CMAKE_LINK_LIBRARY_USING_wickqc_netlib_SUPPORTED TRUE CACHE INTERNAL "")
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
