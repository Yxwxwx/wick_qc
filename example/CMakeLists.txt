add_library(wickqc_example_io tensor_io.cpp)
target_include_directories(wickqc_example_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wickqc_example_io PUBLIC wickqc_numeric)

add_executable(example_rhf evaluate_rhf.cpp)
target_link_libraries(example_rhf PRIVATE wickqc_methods wickqc_rhf wickqc_precompiled wickqc_example_io)

add_subdirectory(build_time)

if(WICKQC_ENABLE_AO2MO)
  foreach(example IN ITEMS wick_mp2 wick_spinor_mp2)
    add_executable(example_${example} ${example}.cpp)
    target_link_libraries(example_${example} PRIVATE wickqc::wickqc)
  endforeach()
endif()
