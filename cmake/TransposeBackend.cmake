option(WICKQC_ENABLE_HPTT "Enable the optional HPTT materialized-transpose backend" OFF)
set(WICKQC_HPTT_MIN_BYTES "" CACHE STRING
  "Measured minimum tensor payload bytes for HPTT; empty keeps automatic copies native")

add_library(wickqc_transpose INTERFACE)
target_include_directories(wickqc_transpose INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(wickqc_transpose INTERFACE cxx_std_20)

if(WICKQC_ENABLE_HPTT)
  set(HPTT_ROOT "$ENV{HOME}/Applications/hptt/install" CACHE PATH "HPTT installation prefix")
  find_path(WICKQC_HPTT_INCLUDE_DIR hptt.h HINTS "${HPTT_ROOT}/include" REQUIRED)
  find_library(WICKQC_HPTT_LIBRARY hptt HINTS "${HPTT_ROOT}/lib" "${HPTT_ROOT}/lib64" REQUIRED)
  target_include_directories(wickqc_transpose SYSTEM INTERFACE "${WICKQC_HPTT_INCLUDE_DIR}")
  target_link_libraries(wickqc_transpose INTERFACE "${WICKQC_HPTT_LIBRARY}" OpenMP::OpenMP_CXX)
  target_compile_definitions(wickqc_transpose INTERFACE WICKQC_TRANSPOSE_HPTT)
  if(NOT WICKQC_HPTT_MIN_BYTES STREQUAL "")
    if(NOT WICKQC_HPTT_MIN_BYTES MATCHES "^(0|[1-9][0-9]*)$")
      message(FATAL_ERROR "WICKQC_HPTT_MIN_BYTES must be empty or a nonnegative integer")
    endif()
    target_compile_definitions(wickqc_transpose INTERFACE
      WICKQC_HPTT_MIN_BYTES=${WICKQC_HPTT_MIN_BYTES})
  endif()
endif()

target_link_libraries(wickqc_ndarray INTERFACE wickqc_transpose)
