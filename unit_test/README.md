# C++ unit tests

This directory contains focused GTest tests. It does not contain method drivers,
reference source copies, generated formulas, or molecular output files.

| File | Checks |
| --- | --- |
| `test_wick.cpp` | Fermion algebra, spin-free metric, chemist symmetry, substitution, serialization, graph coefficient regression |
| `test_ndarray.cpp` | Views/ownership, reductions, ellipsis, broadcasting, integer labels, invalid and empty inputs |
| `test_backend.cpp` | Real/complex backend contractions against explicit loops, strides, batches, alpha/beta, padded GEMM output |
| `test_mp2.cpp` | Spatial MP2 energy and residuals against the denominator formula |
| `test_lapack.cpp` | Over-/underdetermined and rank-deficient minimum-norm solves, cutoff equality and DGELSD fallback, input preservation and invalid inputs |
| `test_transpose.cpp` | All 4D permutations with unequal runtime extents, real/complex HPTT vs native, views, materialization, heuristic gates, arbitrary strides, alpha/beta, empty/scalar and overlap fallbacks |
| `test_header_only.cpp`, `header_only_peer.cpp` | Umbrella-only consumers in two translation units, serialization and runtime tensor dimensions without a core object library |
| `test_codegen.cpp` | Generated CCSD and all 13 IC-NEVPT2 kernels against unoptimized equations, real/complex arithmetic |

`wickqc_unit_tests` builds the first four files independently of examples.
`wickqc_codegen_tests` separately checks the generated-example kernels.
`wickqc_lapack_tests` checks the selected LAPACK provider independently.
MKL, OpenBLAS, Netlib, and Eigen use the same tests. Eigen also runs the
real/complex contraction tests with dynamic dimensions.
`wickqc_transpose_tests` independently checks materialized transpose selection.
Run it with HPTT both enabled (`-DWICKQC_ENABLE_HPTT=ON`) and disabled. The
tests explicitly lower the byte gate to exercise actual HPTT calls on small
fixtures; that setting is not a calibrated performance threshold.

The dimensions come from `WICKQC_TEST_OCCUPIED`, `WICKQC_TEST_ACTIVE` and
`WICKQC_TEST_VIRTUAL` at process launch. CTest runs the same executable twice
with different dimensions. Fixed seeds and explicit absolute/relative tolerances
make the comparisons reproducible. Complex checks validate tensor algebra under
the existing symmetry conventions, not a general complex-orbital QC method.

```sh
module load googletest/1.15.0
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Configure another build with `-DEINSUM_BACKEND=TBLIS` (and `TBLIS_ROOT`) to run
the same tests on TBLIS. `-DWICKQC_BUILD_EXAMPLES=OFF` omits the generated-example
integration tests but keeps the Wick/NDArray/backend/MP2 unit tests.
`-DBUILD_TESTING=OFF` removes the GTest requirement from library/example builds.
