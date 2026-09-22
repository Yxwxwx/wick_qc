find_package(GTest 1.15 CONFIG REQUIRED)

function(wickqc_dimension_tests target prefix)
  foreach(shape IN ITEMS small alternate)
    add_test(NAME ${prefix}_${shape} COMMAND ${target})
    if(shape STREQUAL "small")
      set(dimensions "WICKQC_TEST_OCCUPIED=2;WICKQC_TEST_ACTIVE=2;WICKQC_TEST_VIRTUAL=3")
    else()
      set(dimensions "WICKQC_TEST_OCCUPIED=3;WICKQC_TEST_ACTIVE=3;WICKQC_TEST_VIRTUAL=2")
    endif()
    set_tests_properties(${prefix}_${shape} PROPERTIES
      ENVIRONMENT "${dimensions};OMP_NUM_THREADS=1;TBLIS_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
      LABELS "unit;${prefix}")
  endforeach()
endfunction()

foreach(module IN ITEMS wick ndarray backend rhf lapack nevpt2 transpose codegen header_only)
  add_executable(wickqc_${module}_tests test_${module}.cpp)
  target_link_libraries(wickqc_${module}_tests PRIVATE wickqc::wickqc GTest::gtest_main)
  wickqc_dimension_tests(wickqc_${module}_tests ${module})
endforeach()
target_link_libraries(wickqc_codegen_tests PRIVATE wickqc_example_kernels)
# Exactly two consumers catch ODR/link errors, including optional modules.
target_sources(wickqc_header_only_tests PRIVATE header_only_peer.cpp)

if(WICKQC_ENABLE_AO2MO)
  foreach(module IN ITEMS integrals hdf5)
    if(NOT (UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
      message(STATUS "${module} fault injection requires GNU-compatible linker wrapping")
      continue()
    endif()
    add_executable(wickqc_${module}_tests test_${module}.cpp)
    target_link_libraries(wickqc_${module}_tests PRIVATE wickqc::wickqc GTest::gtest_main)
    add_test(NAME ${module} COMMAND wickqc_${module}_tests)
    set_tests_properties(${module} PROPERTIES LABELS "unit;${module}"
      ENVIRONMENT "OMP_NUM_THREADS=1;TBLIS_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1")
  endforeach()
  if(TARGET wickqc_integrals_tests)
    target_link_options(wickqc_integrals_tests PRIVATE -Wl,--wrap=int2e_sph -Wl,--wrap=int2e_optimizer)
    target_link_options(wickqc_hdf5_tests PRIVATE -Wl,--wrap=H5Dwrite -Wl,--wrap=H5Awrite
      -Wl,--wrap=H5Dclose -Wl,--wrap=H5Aclose -Wl,--wrap=H5Gclose -Wl,--wrap=H5Fflush -Wl,--wrap=H5Fclose)
  endif()
endif()
