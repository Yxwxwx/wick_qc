# wick_qc

`wick_qc` is a C++20 symbolic Wick engine and tensor-equation compiler for
quantum-chemistry methods. The current implementation covers fermionic and
spin-free normal ordering, contractions, symmetry-aware simplification,
Tensor Equation IR, Einsum IR, contraction-graph optimization, NumPy rendering,
and build-time C++ code generation and numerical execution through NDArray
with TBLIS or BLAS backends.

The independent implementation is built as `wickqc_symbolic` from
`src/symbolic`, `src/equation`, `src/einsum`, and `src/method`. The supplied
`docs/wick.hpp` is a local, Git-ignored block2 reference. It is retained for
algorithm and convention comparisons and is not required by the build.

Declaration headers use `.h`, with definitions in `.cpp`. Headers containing
implementations retain `.hpp`, including the NDArray backends and the block2
reference.

## Build

The default build requires CMake 3.25+, C and C++20 compilers, and OpenMP.
It builds the libraries and command-line tools. The supplied `docs/wick.hpp`
reference is not a build dependency.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

`EINSUM_BACKEND` selects `BLAS` (default) or `TBLIS`. For the BLAS path,
`BLAS_BACKEND` selects `NATIVE` (default), `SIMPLE`, `MKL`, `OPENBLAS`, or
`BLIS`. NATIVE and SIMPLE need no external BLAS installation. The numerical
library is `wickqc_runtime`; generic NDArray users can link only the
`wickqc_ndarray` interface target. `wickqc_symbolic` has no numerical-backend
dependency. Selection is per build, and all consumers of NDArray should link
the same interface target to inherit consistent definitions and libraries.

```bash
cmake -S . -B build/openblas -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBLAS_BACKEND=OPENBLAS -DOPENBLAS_ROOT=/path/to/openblas
cmake -S . -B build/blis -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBLAS_BACKEND=BLIS -DBLIS_ROOT=/path/to/blis
module load mkl-2024.2.1
cmake -S . -B build/mkl -G Ninja -DCMAKE_BUILD_TYPE=Release -DBLAS_BACKEND=MKL
cmake -S . -B build/tblis -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DEINSUM_BACKEND=TBLIS -DTBLIS_ROOT=/path/to/tblis/install
cmake --build build/tblis -j 4
```

BLIS requires its CBLAS interface. MKL is discovered through its CMake package
or `MKLROOT` and defaults to sequential GEMM; `MKL_THREADING` can select a
threading layer compatible with the application. OpenMP is linked explicitly
for NDArray reductions. Control BLAS/OpenMP thread counts in the calling job.

TBLIS is discovered with `find_package(TBLIS CONFIG REQUIRED)` and linked
through `TBLIS::tblis`; `CMAKE_PREFIX_PATH` also works in place of `TBLIS_ROOT`.
The project enables both C and CXX for that package's dependencies. This path
does not select a separate GEMM backend: `BLAS_BACKEND` is unused when
`EINSUM_BACKEND=TBLIS`. Set `TBLIS_NUM_THREADS` for TBLIS and `OMP_NUM_THREADS`
for NDArray reductions. The installed TBLIS 2.0 also falls back to
`BLIS_NUM_THREADS`, then `OMP_NUM_THREADS`, if its own variable is unset.

The repository `.clangd` reads `build/compile_commands.json`. To edit the
optional TBLIS adapter while the main build uses BLAS, configure `build/tblis`
as above; that header uses the TBLIS compilation database for its dependency
paths. Reconfigure the corresponding build directory after changing targets
or backend settings.

## Execute Wick equations with NDArray

The NDArray and GEMM implementations supplied in `tmp_backend` now live in
`src/backend/ndarray.hpp` and `src/backend/blas.hpp`, in namespace `wickqc`.
The original directory is not a build dependency. They remain template headers;
the new adapter uses `.h` declarations and a `.cpp` implementation.

The execution path is:

```text
method.Equations() → optional graph.Simplify() → NdArrayExecutor::Compile()
                  → Evaluate(tensors, dimensions) → NDArray::Einsum()
                  → einsum planning → binary ContractionPlan
                     ├─ TBLIS: tensor views → tblis_tensor_mult
                     └─ BLAS: tensor → matrix transpose/pack → Gemm
                              ├─ MKL / BLIS / OpenBLAS
                              └─ NATIVE / SIMPLE
```

Both executors consume the same axis pairing and output order from
`src/backend/contraction_plan.hpp`. `src/backend/tblis.hpp` passes tensor
lengths, strides, labels, and alpha/beta directly to `tblis_tensor_mult`.
NDArray performs no matrix packing on that path; TBLIS handles its own
internal execution. Diagonals, unary reductions, copies, and permutations
remain NDArray operations, so no TBLIS level-1 entry points are needed.

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
BLAS backends and TBLIS; einsum multiplication does not implicitly conjugate
operands. TBLIS accepts strided input views, including negative and zero
strides. `TensordotInto` requires a C-contiguous output; overlapping inputs
are snapshotted before execution. Empty contractions and alpha/beta scaling
are handled consistently across executors.

MP2--MP4 and CCSD/T/Q use the same adapter. SC-/IC-NEVPT2 return named graphs;
compile/evaluate each entry of `ScNevpt2Generator().Equations()` or
`IcNevpt2Generator().Equations()`. These evaluate the SC norms/Hamiltonian
expectations and the IC Hamiltonian/RHS blocks. Orbital restrictions, NEVPT2
energy assembly/linear solves, and iterative MP/CC amplitude solvers remain at
the method/application layer; this adapter executes the tensor equations.

Graph compilation and optimization are explicit setup operations. During
evaluation, NDArray consumes integer index lists and prepares contractions
using the supplied shapes, with packing for BLAS when needed. Integer indices
are remapped locally for TBLIS, without restricting the caller's label values.
The installed TBLIS label type limits the number of distinct indices in one
binary contraction (256 for its default `char`); exceeding that limit throws
instead of silently colliding labels. Orbital dimensions and strides remain runtime data, including in the
precompiled method kernels described below.

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

## Build-time MP/CC code generation

CMake/Ninja builds `generate_cpp`, runs Wick expansion and graph optimization,
and compiles the emitted `build/.../generated/*.generated.cpp` into
`wickqc_precompiled`. Generated functions call integer-index
`NDArray::Einsum()` directly. They retain the runtime executor's input checks,
output permutations, intermediate scheduling, and last-use releases.

**Only the method equations are fixed at build time.** Integral values,
amplitudes, orbital dimensions, and NDArray shapes/strides are supplied at
runtime. The generator has no molecular-data argument and never reads PySCF
files. The same compiled function can evaluate different molecular sizes.
Einsum planning, tensor allocation, and the selected TBLIS/BLAS execution still
happen at runtime; Wick expansion and equation lowering do not.

```bash
# Defaults: MP2--MP4, CCSD and CCSDT, native chemist notation.
cmake -S . -B build/aot -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DWICKQC_PRECOMPILE_MP="2;3;4" -DWICKQC_PRECOMPILE_CC="2;3" \
  -DWICKQC_PRECOMPILE_CONVENTIONS=chemist
cmake --build build/aot -j 3

# Optionally compile CCSDTQ as well, or support both integral conventions.
cmake -S . -B build/aot -DWICKQC_PRECOMPILE_CC="2;3;4" \
  -DWICKQC_PRECOMPILE_CONVENTIONS="chemist;physicist"
cmake --build build/aot -j 3

# Empty selections give a runtime-only method registry.
cmake -S . -B build/runtime_only -G Ninja \
  -DWICKQC_PRECOMPILE_MP="" -DWICKQC_PRECOMPILE_CC=""
```

CC numbers are maximum excitation ranks: 2=CCSD, 3=CCSDT, 4=CCSDTQ.
The choices control which kernels enter the library/binary. High-rank
expansion can take minutes; Ninja limits concurrent code-generation jobs to
one. Generated files are build artifacts, marked as generated and covered by
`build/` in `.gitignore`. No molecular dimensions are CMake options. Ninja
tracks the generator executable and its source dependencies; changing the
method selection updates the registry and the selected generated sources.

`runtime::SpatialEvaluator` selects a precompiled kernel when present. An
unselected method is expanded and lowered once in its constructor; keep the
object for all subsequent evaluations. `GenerationPolicy::kPrecompiledOnly`
rejects missing kernels, and `kRuntimeOnly` explicitly exercises the fallback.
An unselected high-order method therefore has a **one-time runtime Wick cost**;
its evaluations perform no symbolic work. Avoiding that first cost requires
selecting the method during the build.

```cpp
#include "method/rhf_data.h"
#include "runtime/spatial_evaluator.h"

// All counts and arrays come from the host program or files at runtime.
wickqc::method::RhfData data{nocc, mo_energies, fock_mo, eri_chemist};
wickqc::runtime::SpatialEvaluator cc({
    wickqc::method::SpatialFamily::kCc, requested_rank,
    wickqc::method::IntegralConvention::kChemist});
// amplitudes contains tEI, tEEII, and higher-rank tensors as required.
auto inputs = data.Bind(cc.Inputs(), amplitudes);
auto outputs = cc.Evaluate(inputs, data.Dimensions());
// Repeat with updated amplitudes using the same cc object.
```

Link this host to `wickqc_methods` and `wickqc_rhf`. For a host that uses only
precompiled methods, link `wickqc_precompiled` and `wickqc_rhf`, include
`runtime/precompiled.h`, and obtain the `NumericKernel` through
`FindPrecompiled(SpatialMethod)`. That path has no link dependency on the Wick
engine. `PrecompiledMethods()` lists the configured selection. The
`wickqc_numeric` library provides just the shared runtime tensor contracts.
Generated kernels support `double` and `complex<double>`; the RHF data adapter
uses real integrals/amplitudes. The spin-free
method equations retain their existing real-orbital symmetry conventions.

`RhfData` expects canonical closed-shell spatial MOs, occupied first,
`eri[p,q,r,s]=(pq|rs)`, and the **MO Fock matrix** (not the core Hamiltonian).
Its blocks are strided views. For physicist kernels it supplies the matching
view `v[p,q,r,s]=(pr|qs)`; chemist kernels use the original ordering directly.
The kernel's `Inputs()` supplies required amplitude names and runtime shapes.
MP2--MP4 use `u1`/`u2` wavefunction coefficients; CC uses `t` amplitudes, with
virtual axes followed by occupied axes. Energies are correlation contributions;
residuals retain the covariant spatial spin metric. This interface evaluates
these equations at supplied amplitudes; high-order amplitude solvers are a
separate host responsibility.

### Higher MP orders

`SpatialMpGenerator(order, convention, maximum_excitation_rank=0)` extends the
existing MP2--MP4 hierarchy to MPn. The unbounded default retains excitation
ranks through twice the required wavefunction order. A positive optional MP
bound can use the runtime electron/hole limit, or represent an explicit
excitation truncation. A bound below the physical limit changes the method.
`SpatialMethod::maximum_excitation_rank` selects this runtime specialization;
precompiled kernels use zero (the full equations).

The recursion uses intermediate normalization, `E1=0`,
`F_N |m> + Q V |m-1> - sum(k=2..m-1) E_k |m-k> = 0`, and Wigner's
[2m+1 rule](https://doi.org/10.1080/00268978000100121). Writing
`S_ab=<a|b>`, the energies are

```text
E_(2m)   = -<m|F_N|m> - sum(a,b=1..m-1) E_(2m-a-b) S_ab
E_(2m+1) =  <m|V|m>   - sum(a,b=1..m; a+b<=2m-1) E_(2m+1-a-b) S_ab
```

MP2--MP4 retain their previously validated expressions. MP5 agrees with the
formula in the [block2 MP driver](https://github.com/block-hczhai/block2-preview/blob/master/pyblock2/mp.py);
subsequent orders continue the same perturbation hierarchy. Lower energies
are internal graph dependencies. High-order expansion and dense amplitudes
can be expensive; no claim is made that arbitrary orders are practical.

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
inside a graph.

The design and release roadmap are documented in
[`docs/wick_qc_project_plan.md`](docs/wick_qc_project_plan.md).
