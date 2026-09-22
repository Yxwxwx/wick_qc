# The executable's link dependencies contain numeric code only. Wick runs in
# the custom commands above; integral values and dimensions are runtime inputs.
add_executable(example_precompiled evaluate.cpp)
target_link_libraries(example_precompiled PRIVATE wickqc_example_kernels wickqc_example_io)
