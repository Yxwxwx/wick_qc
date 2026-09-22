# Examples

`einsum_text.cpp` shows formula generation. `evaluate_rhf.cpp` shows how a C++
host supplies real RHF integrals and amplitudes and evaluates MP2/CCSD.
`build_time/` demonstrates generating and compiling numeric kernels through
CMake. The backend choice is inherited from the project build.

## Obtain einsum text

```sh
./build/example/example_einsum ccsd > build/ccsd.generated.py
./build/example/example_einsum ic-nevpt2 > build/ic_nevpt2.generated.py
./build/example/example_einsum fic-nevpt2 > build/fic_nevpt2.generated.py
# Optional contraction-graph optimization and higher excitation ranks:
./build/example/example_einsum spatial-cc 3 --optimize --chemist > build/ccsdt.generated.py
```

The `ccsd` command uses **spin-free spatial CCSD with chemist integrals**:

```cpp
#include "method/spatial.hpp"

const auto generator = wickqc::method::SpatialCCGenerator(
    2, wickqc::method::IntegralConvention::kChemist);
const auto text = generator.GenerateNumpy();
```

`CCSDGenerator` is the existing spin-orbital generator; the corresponding
command is `spin-orbital-ccsd`. `ICNEVPT2Generator().GenerateNumpy()` produces the
existing internally contracted functions. Here `fic-nevpt2` aliases `ic-nevpt2`;
they refer to the same fully internally contracted equations, not two methods.
The text expects NumPy as `np`, named input arrays and output initialization as
shown in the emitted fragments/functions.

## Generate numeric C++ during the build

`../cmake/MethodKernels.cmake` contains the shared generation commands:

1. `generate_cpp cc 2 chemist example_ccsd OUTPUT.generated.cpp` derives and
   emits spatial CCSD through `CPPEmitter::Render`.
2. `example_generate_nevpt2 OUTPUT_DIRECTORY` visits every
   `ICNEVPT2Generator::Equations()` block and emits 13 IC and 8 SC `.generated.cpp` files plus
   a small kernel registry. All RHS and Hamiltonian components of the coupled
   `irabpq` block are included.

CMake compiles these files into `wickqc_example_kernels`. `example_precompiled`
links that library, the numeric runtime and the example tensor I/O library;
it does not link Wick or the method generators. This can be checked with:

```sh
cmake --build build --target example_precompiled -j 4
./build/example/build_time/example_precompiled --list
nm -C build/example/build_time/example_precompiled | grep 'symbolic::Expression'
# No symbolic::Expression symbols should be present.
```

Generated kernels call `NDArray::Einsum()`. Orbital counts, shapes, integrals,
RDMs and amplitudes are all supplied at runtime. No molecular data is needed by
CMake or the C++ generator.

For an embedding C++ host, the essential calls are:

```cpp
#include "build_time/kernels.hpp"
#include <string>

// dimensions and tensors are filled by the host after compilation.
const auto cc = wickqc::generated::Kernel_example_ccsd().Evaluate(tensors, dimensions);
for (const auto& entry : wickqc::example::ICNEVPT2Kernels()) {
  const auto result = entry.kernel->Evaluate(block_inputs.at(std::string(entry.name)), dimensions);
  // Consume the RHS and effective-Hamiltonian arrays for this block.
}
```

`kernels.hpp` is the example's declaration header. `Inputs()` and `Outputs()` on
each kernel provide tensor names and domains; `binding.Shape(dimensions)` gives
the required runtime shape.

The file-based example uses `dimensions.txt` containing three integers:
`inactive active external`. Each `<tensor-name>.bin` is little-endian IEEE
float64 in C order, without a header. Inspect the required tensors, supply their
values, then run the kernel:

```sh
./build/example/build_time/example_precompiled --describe ccsd INPUT/dimensions.txt
./build/example/build_time/example_precompiled ccsd INPUT OUTPUT
./build/example/build_time/example_precompiled --describe ic-nevpt2/irabpq INPUT/dimensions.txt
./build/example/build_time/example_precompiled ic-nevpt2/irabpq INPUT OUTPUT
```

CCSD uses `v[p,q,r,s]=(pq|rs)` and amplitudes ordered virtual axes then occupied
axes. The IC generator retains its physicist `w[p,q,r,s]=(pr|qs)` convention,
its inactive-core effective one-body `h`, inactive/external energies `orbe`,
and spin-free active RDMs `E1` through `E4`. IC kernels return unrestricted
arrays. Apply the same orbital-pair restrictions, coupled-block assembly and
linear solves shown in the NumPy functions to obtain the final NEVPT2 energy.
The build-time example covers equation evaluation, not a new NEVPT2 solver.

## Supply a real molecular RHF calculation

`RHFData` in `src/method/rhf.hpp` binds occupied/virtual views of runtime MO
arrays. `evaluate_rhf.cpp` uses that API, derives MP2 amplitudes, and performs one
CCSD Jacobi update after inverting the covariant spin metric. It accepts:

```sh
./build/example/example_rhf INPUT OUTPUT compiled chemist
./build/example/example_rhf INPUT OUTPUT runtime physicist
```

`compiled` requires the selected method/convention to be precompiled; it never
falls back silently. `runtime` constructs the Wick equations once per method.
The molecular functional tests in [`../test/`](../test/README.md) use fixed
HDF5 data and the complete method solvers; they do not import PySCF.

## Integral input and host adapters

With `WICKQC_ENABLE_AO2MO=ON`, `wick_mp2.cpp` and `wick_spinor_mp2.cpp` show
how transformed blocks become NDArray inputs. Their targets are
`example_wick_mp2` and `example_wick_spinor_mp2`. The complex example evaluates
the energy equation; it does not claim general complex CCSD support.

`scalar.py`, `spinor.py`, `export_input.py` and `export_reference.py` are
optional upstream data-export examples. They may use PySCF/block2 (and socutils
for the spinor example); the C++ project does not require these packages.
`read_helper.py` reads exported integral blocks with NumPy/h5py.

Complete MP2/CCSD and SC/IC-NEVPT2 file-based drivers live in `test/rhf.cpp`
and `test/nevpt2.cpp`, alongside their fixed molecular fixtures. Both call the
same public C++ solvers that a host program would use. See the
[functional test guide](../test/README.md) for invocation and data conventions.
