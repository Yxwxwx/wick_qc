# wick_qc

`wick_qc` is a C++20 symbolic Wick engine and tensor-equation compiler for
quantum-chemistry methods. The current implementation covers fermionic and
spin-free normal ordering, contractions, symmetry-aware simplification,
Tensor Equation IR, Einsum IR, contraction-graph optimization, NumPy rendering,
and C++ numerical execution through NDArray and selectable GEMM backends.

The independent implementation is built as `wickqc_symbolic` from
`src/symbolic`, `src/equation`, `src/einsum`, and `src/method`. The supplied
`docs/wick.hpp` is a local, Git-ignored block2 reference. It is used only by
the optional regression suite and is not required by the default build.

Declaration headers use `.h`, with definitions in `.cpp`. Headers containing
implementations retain `.hpp`, including the block2 oracle and inline comparison
helpers.

## Build

The default build requires CMake 3.25+, a C++20 compiler, and OpenMP.
It builds the library, CLI, and standalone method examples without GoogleTest,
`tests/`, or `docs/wick.hpp`.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

`BLAS_BACKEND` selects `NATIVE` (default), `SIMPLE`, `MKL`, `OPENBLAS`, or
`BLIS`. NATIVE and SIMPLE need no external BLAS installation. The numerical
library is `wickqc_runtime`; generic NDArray users can link only the
`wickqc_ndarray` interface target. `wickqc_symbolic` has no BLAS dependency.

```bash
cmake -S . -B build/openblas -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBLAS_BACKEND=OPENBLAS -DOPENBLAS_ROOT=/path/to/openblas
cmake -S . -B build/blis -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBLAS_BACKEND=BLIS -DBLIS_ROOT=/path/to/blis
module load mkl-2024.2.1
cmake -S . -B build/mkl -G Ninja -DCMAKE_BUILD_TYPE=Release -DBLAS_BACKEND=MKL
```

BLIS requires its CBLAS interface. MKL is discovered through its CMake package
or `MKLROOT` and defaults to sequential GEMM; `MKL_THREADING` can select a
threading layer compatible with the application. OpenMP is linked explicitly
for NDArray reductions. Control BLAS/OpenMP thread counts in the calling job.

## Execute Wick equations with NDArray

The NDArray and GEMM implementations supplied in `tmp_backend` now live in
`src/backend/ndarray.hpp` and `src/backend/blas.hpp`, in namespace `wickqc`.
The original directory is not a build dependency. They remain template headers;
the new adapter uses `.h` declarations and a `.cpp` implementation.

The execution path is:

```text
method.Equations() → optional graph.Simplify() → NdArrayExecutor::Compile()
                  → Evaluate(tensors, dimensions) → NDArray einsum → GEMM
```

```cpp
#include "method/spatial_cc.h"
#include "runtime/ndarray_executor.h"

using namespace wickqc;
const runtime::Dimensions dimensions{
    {{1, 0}, nocc}, {{2, 0}, nact}, {{8, 0}, nvir}};
const auto graph = method::SpatialCcGenerator(2).Equations().Simplify();
const auto executor = runtime::NdArrayExecutor::Compile(graph);
runtime::TensorMap<double> tensors;
for (const auto& binding : executor.Inputs()) {
  // Load the integral/amplitude/RDM block named binding.name into an NDArray
  // with shape binding.Shape(dimensions), respecting its declared symmetry.
  tensors.emplace(binding.name, load_tensor(binding.name, binding.Shape(dimensions)));
}
const auto result = executor.Evaluate(tensors, dimensions);
// result contains energy, residual1, residual2. Reuse executor for new amplitudes.
```

The domain keys are `{orbital_mask, spin_mask}`; orbital masks `1`, `2`, `4`,
`8` mean inactive, active, single, external. Spin masks `0`, `1`, `2` mean
spin-free, alpha, beta. Supply each used domain's extent explicitly. Tensor
names and axis order match the existing NumPy emitter, e.g. `vIIEE`, `u1EEII`,
`E1`. The executor validates names/shapes, constructs Kronecker deltas and
broadcast ones, applies coefficients and output permutations, schedules graph
dependencies, and releases intermediates after their last use. Results start
at zero and own their storage; input views are never modified. Graph assignment
names must be unique. `double` and `std::complex<double>` are supported by all
five backends; einsum multiplication does not implicitly conjugate operands.

MP2--MP4 and CCSD/T/Q use the same adapter. SC-/IC-NEVPT2 return named graphs;
compile/evaluate each entry of `ScNevpt2Generator().Equations()` or
`IcNevpt2Generator().Equations()`. These evaluate the SC norms/Hamiltonian
expectations and the IC Hamiltonian/RHS blocks. Orbital restrictions, NEVPT2
energy assembly/linear solves, and iterative MP/CC amplitude solvers remain at
the method/application layer; this adapter executes the tensor equations.

Graph compilation and optimization are explicit setup operations. During
evaluation, NDArray consumes integer index lists and prepares contractions and
packing using the supplied shapes. This is a reusable runtime executor, not
yet the project's planned constexpr contraction/workspace compiler.

A complete MP2 model supplies integrals and denominator-divided amplitudes,
evaluates the optimized Wick graph, and checks its energy/residuals:

```bash
./build/evaluate_mp2
# E2 = -0.0744044593545318; residual norms approximately 0 and 1e-16
```

Generic einsum is also available without the symbolic engine:

```cpp
auto c = wickqc::NDArray<double>::Einsum("ik,jk->ij", {a, b});
auto d = wickqc::NDArray<double>::Einsum({{0, 1}, {2, 1}}, {0, 2}, {a, b});
```

## Optional numerical validation

The Git-ignored `tests/numerical` suite is independent of `docs/wick.hpp` and
does not enable the large block2 regression suite. It checks scalar-loop GEMM
references, real/complex tensors, views, diagonals, reductions, broadcasts,
permutations, empty spaces, invalid inputs, and method results against the
original NumPy equations. Tests use two orbital shapes and tensors satisfying
the declared symmetries.

```bash
module load googletest/1.15.0
cmake -S . -B build/numerical -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DWICKQC_BUILD_NUMERICAL_TESTS=ON -DBLAS_BACKEND=NATIVE
cmake --build build/numerical -j 4
ctest --test-dir build/numerical --output-on-failure
# Longer high-rank CC validation (default CTest covers CCSD):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tests/numerical/check_methods.py \
  "$PWD/build/numerical/tests/numerical/check_ndarray_methods" \
  "$PWD/build/numerical/cc_through_quadruples" 4
```

## Standalone method examples

`unit_test/` contains three runnable examples using only `wickqc_symbolic`.
They generate all equations for the selected method and write NumPy code to
stdout. The complete method implementations live in `src/method/`.

```bash
./build/test_spatial_mp 2 > build/mp2.generated.py
./build/test_spatial_mp 3 > build/mp3.generated.py
./build/test_spatial_mp 4 --optimize > build/mp4.generated.py
./build/test_spatial_cc 2 > build/ccsd.generated.py
./build/test_spatial_cc 3 > build/ccsdt.generated.py
./build/test_spatial_cc 4 --optimize > build/ccsdtq.generated.py
./build/test_nevpt2_methods sc --optimize > build/sc_nevpt2.generated.py
./build/test_nevpt2_methods ic > build/ic_nevpt2.generated.py
```

MP takes perturbation order (2--4); CC takes maximum excitation rank
(2=CCSD, 3=CCSDT, 4=CCSDTQ). MP and CC output energies and amplitude residuals;
SC-/IC-NEVPT2 output compute functions for every subspace. These programs
generate equations; integrals, RDMs, and amplitudes are supplied by the caller.
High-rank CC generation takes substantially longer than CCSD.

## Optional local regression suite

The full regression suite remains in the Git-ignored `tests/` directory.
It is disabled by default. When that directory and `docs/wick.hpp` are present,
enable it explicitly with GoogleTest 1.15 or newer:

```bash
module load googletest/1.15.0
cmake -S . -B build/regression -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DWICKQC_BUILD_REGRESSION_TESTS=ON
cmake --build build/regression -j 4
ctest --test-dir build/regression --output-on-failure
```

The suite includes the supplied GHF, CCSD, UGA-CCSD, IC-NEVPT2, SC-NEVPT2,
and IC-MRCI fixtures, plus symbolic, equation, einsum, and method comparisons.
The original fixtures remain unchanged in `tests/reference/block2/`.
The MP, CC, and NEVPT2 comparisons now live in `tests/reference/test_spatial_*.cpp`;
their executables are `test_spatial_mp_reference`, `test_spatial_cc_reference`,
and `test_nevpt2_methods_reference`. Regression artifacts are written under
`build/regression/tests/`. Python and NumPy are optional test dependencies.
The `reference_wick_probe` target is available only with this suite enabled.

## Generate equations

```bash
./build/wick_qc ghf > build/ghf.generated.py
./build/wick_qc ccsd > build/ccsd.generated.py
./build/wick_qc uga-ccsd > build/uga_ccsd.generated.py
./build/wick_qc ic-nevpt2 > build/ic_nevpt2.generated.py
./build/wick_qc sc-nevpt2 --optimize > build/sc_nevpt2.generated.py
./build/wick_qc spatial-mp 4 --optimize > build/mp4.generated.py
./build/wick_qc spatial-cc 3 --optimize > build/ccsdt.generated.py
```

The standalone `generate_ghf`, `generate_ccsd`, and `generate_ic_nevpt2`
executables produce the same respective outputs. These are generated equation
fragments: initialize the named arrays and accumulators in the calling program.
GHF outputs coefficient tensors grouped by spin block and number of remaining
normal-ordered operators; their axes follow that operator sequence. CCSD outputs
the correlation energy and singles/doubles residuals. IC-NEVPT2 emits the supplied
fixture's compute functions.
Spatial CC takes maximum excitation rank (2=CCSD, 3=CCSDT, 4=CCSDTQ);
spatial MP takes perturbation order. Both emit covariant spin-free equations,
with amplitudes supplied as inputs. See the [spatial validation record](docs/spatial_methods_validation.md)
for the precise projection, normalization and energy conventions and the
completed acceptance results.

The generated reference fixture contains 631 NumPy einsum statements. Its
current byte-for-byte regression fingerprint is:

```text
SHA-256 7e827e84cf24e5f48de3256eb4f6b36756b49d950ff2d2dc1fe69690766f1cbe
60202 bytes, 869 lines
```

`live_reference_equivalence` compares the current C++ implementations directly:
all five GHF Hamiltonian blocks and their NumPy coefficients, CCSD energy/singles
and doubles through fourth order, both CCSD integral conventions, intermediate
BCH expressions, index-type compatibility, and the entire
IC-NEVPT2 NumPy output. Comparison artifacts are written to the build directory.
When Python and NumPy are available, `numpy_reference_samples` additionally
checks 1698 contractions using two seeds/shapes and explicit scalar loops.
Python and NumPy are test-only dependencies.

`multireference_equivalence` compares SC-NEVPT2 norms and commutators in all eight
subspaces, both with free external indices and with all indices summed. It also
compares all 225 IC-MRCI Hamiltonian component pairs and 16 overlap blocks,
including the two-component nonorthogonal subspace. Each equation passes through
both local IRs before its NumPy text is compared. Additional checks examine raw
operation stages, tensor symmetry metadata, nested SC substitutions, and all
1629 I/A/E partitions of the tested E1/E2 operator patterns. The supplied
fixtures' `assert()` checks remain enabled in Release builds.

See [`docs/block2_multireference_comparison.md`](docs/block2_multireference_comparison.md)
for the implementation differences found and the scope of this comparison.
The [code review](docs/code_review.md) records the original gaps and tooling
audit. The [continuing alignment report](docs/block2_alignment.md) records
single-occupancy and tagged-operator fixes, all three expansion controls,
binary serialization interoperability, and UGA-CCSD comparisons through
fourth order, including optimized CCSD and UGA-CCSD graphs. The
[operation inventory](docs/block2_api_coverage.md) maps the reference workflow
to the independent implementation and states its validation limits. Verified reference errors are
corrected and documented in the [difference register](docs/block2_differences.md).

The symbolic API includes simultaneous `RenameIndices`, tensor-definition
`ParseDefinition`, capture-avoiding `Substitute`, and `FullySummedProduct`.
Parsing accepts inline sums, parenthesized numeric coefficients, bracketed and
LaTeX tensor indices, and explicit summed-index orbital types such as
`SUM <pq|IE>`. `SplitIndexDomains` and `NormalOrder` expose the two expansion
stages; `SimplifyDeltas`, `RemoveZeros` and `MergeTerms` expose the simplification
pipeline. `SortFactors` sorts commuting tensor factors while retaining the
operator sequence. `TensorEquation::FromExpression` records output,
coefficient, inputs, and reductions; `einsum::Program::Lower` assigns integer
index IDs, keeping textual labels as rendering metadata. Neither IR requires a
numerical backend or an optimizer.

`equation::ContractionGraph` provides contraction ordering, binary splitting,
intermediate reuse, common-factor and output-permutation extraction,
topological scheduling, and expansion back to the original equations. Each
stage is callable independently. `einsum::RenderNumpy(graph.Simplify(), 17)`
emits an optimized program with double-precision coefficient text; omitting
the second argument selects the reference's six-significant-digit formatting.
Initialize named output arrays to zero before executing a graph fragment.
Intermediate arrays are created and released by the fragment. Broadcast
operands such as `ident2EI` must be supplied as all-ones arrays with the
indicated orbital-domain dimensions; domain suffixes prevent shape collisions
inside a graph. Graph numerical
tests compare execution with direct scalar evaluation of the original equations
and check that input tensors are unchanged.

The validated scope, conventions, and v0.0.1 acceptance mapping are recorded in
[`docs/v0.0.1_validation.md`](docs/v0.0.1_validation.md).

The design and release roadmap are documented in
[`docs/wick_qc_project_plan.md`](docs/wick_qc_project_plan.md).
