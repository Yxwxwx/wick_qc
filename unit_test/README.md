# Component tests

One main GTest source per component. Inputs are small analytic/synthetic tensors;
no test here needs molecular fixtures, PySCF, block2 or the functional tests.

| File | Component and checks |
| --- | --- |
| `test_wick.cpp` | Wick algebra, symmetry, substitution, serialization, graph coefficient regression |
| `test_ndarray.cpp` | Views/ownership, reductions, broadcasting, labels, empty/invalid inputs |
| `test_backend.cpp` | Real/complex contractions vs explicit loops, strides, batches, alpha/beta |
| `test_rhf.cpp` | RHF bindings, analytic MP2, CCSD convergence/residuals and memory preflight |
| `test_nevpt2.cpp` | SC/IC assembly, restrictions, coupled/empty spaces, least-squares and budgets |
| `test_lapack.cpp` | Minimum-norm solves, rank deficiency, cutoff semantics and input preservation |
| `test_transpose.cpp` | Unequal runtime dimensions, views, HPTT/native semantics and fallback gates |
| `test_codegen.cpp` | Generated CCSD/all IC kernels vs unoptimized equations, real/complex |
| `test_header_only.cpp` | Umbrella consumer, ODR and optional integral module |
| `test_integrals.cpp` | Analytic real/spinor AO→MO, layouts, storage, getters, prepared reuse, corruption/budget rejection |
| `test_hdf5.cpp` | Write/close/flush failures, handle cleanup and preservation of caller-owned data |

`header_only_peer.cpp` is the required second translation unit of the ODR test.
The integral and HDF5 fault tests use GNU-compatible linker wrapping; production
has no fault-injection hooks. Integral tests are enabled with the optional
libcint/HDF5 module. They remain part of the unified wick_qc test suite.

Each target is named `wickqc_<component>_tests`. CTest labels component tests
`unit`. Runtime dimensions come from `WICKQC_TEST_OCCUPIED`,
`WICKQC_TEST_ACTIVE` and `WICKQC_TEST_VIRTUAL`; most executables run twice with
different shapes. Integral tests use unequal coefficient-family dimensions.
All comparisons have explicit tolerances and deterministic inputs.

```sh
module load googletest/1.15.0
cmake -S . -B build -DWICKQC_PRECOMPILE_MP=2 -DWICKQC_PRECOMPILE_CC=2
cmake --build build -j 1
ctest --test-dir build --output-on-failure -L unit -j 1
```

`WICKQC_BUILD_EXAMPLES=OFF` does not disable component/code-generation tests.
`BUILD_TESTING=OFF` removes the GTest requirement. The same component targets
exercise the selected einsum/LAPACK/transpose backends; HPTT tests deliberately
lower the gate to cover actual calls, not to recommend a performance threshold.
