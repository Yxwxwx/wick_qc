set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ccsd_source "${generated_dir}/ccsd.generated.cpp")
add_custom_command(OUTPUT "${ccsd_source}"
  COMMAND generate_cpp cc 2 chemist example_ccsd "${ccsd_source}"
  DEPENDS generate_cpp
  COMMENT "Generating the spatial CCSD example kernel"
  JOB_POOL wickqc_codegen_pool VERBATIM)

add_executable(example_generate_nevpt2 generate_nevpt2.cpp)
target_link_libraries(example_generate_nevpt2 PRIVATE wickqc_codegen)
set(ic_blocks ijrs_plus ijrs_minus rsiap_plus rsiap_minus ijrap_plus ijrap_minus
  rsabpq_plus rsabpq_minus ijabpq_plus ijabpq_minus irabpq rabcpqg iabcpqg)
set(ic_sources "${generated_dir}/ic_registry.generated.cpp")
foreach(block IN LISTS ic_blocks)
  list(APPEND ic_sources "${generated_dir}/ic_${block}.generated.cpp")
endforeach()
add_custom_command(OUTPUT ${ic_sources}
  COMMAND example_generate_nevpt2 "${generated_dir}"
  DEPENDS example_generate_nevpt2
  COMMENT "Generating all 13 FIC/IC-NEVPT2 block kernels"
  JOB_POOL wickqc_codegen_pool VERBATIM)

add_library(wickqc_example_kernels ${ccsd_source} ${ic_sources})
target_include_directories(wickqc_example_kernels PUBLIC ${PROJECT_SOURCE_DIR}/example)
target_link_libraries(wickqc_example_kernels PUBLIC wickqc_numeric)

# The executable's link dependencies contain numeric code only. Wick runs in
# the custom commands above; integral values and dimensions are runtime inputs.
add_executable(example_precompiled evaluate.cpp)
target_link_libraries(example_precompiled PRIVATE wickqc_example_kernels wickqc_example_io)
