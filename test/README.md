# Fixed-data functional tests

Normal tests read `data/*.h5`; they do not run or import PySCF/block2.
C++ performs AO→MO, Wick/einsum evaluation and the method solve. Python with
NumPy/h5py only launches the executables and compares their output with the
independent results saved during offline fixture generation.

| Files | Checks |
| --- | --- |
| `rhf.cpp`, `mp2_ccsd.py` | MP2; identical initial CCSD amplitudes; first energy, amplitudes and residuals; converged CCSD; both ERI conventions; memory and iteration limits |
| `nevpt2.cpp`, `nevpt2.py` | SC/IC energy and every subspace, norms/H/RHS, ranks, singular values and minimum-norm solves; selected roots and frozen cores |
| `reference.cpp`, `reference.py` | Reference I/O/metadata, selective bindings, storage ownership, malformed files and read budgets |
| `stored_integrals.hpp` | Test-only dense AO→MO preparation using runtime NDArray/einsum |
| `generate_data.py` | Explicit offline reference generation; never invoked by CMake or CTest |

The stored-AO path runs with compiled and runtime Wick kernels. The same
fixtures also cross-check the production selective incore/outcore integral
paths, which evaluate libcint AO integrals from the saved basis. The dense
stored-AO path is a test preparation path, not a replacement for that bounded
two-pass production algorithm. Shapes and values are never baked into C++.

```sh
module load googletest/1.15.0
cmake -S . -B build/full -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DWICKQC_ENABLE_AO2MO=ON \
  -DEINSUM_BACKEND=OPENBLAS -DLAPACK_BACKEND=OPENBLAS \
  -DWICKQC_PRECOMPILE_MP=2 -DWICKQC_PRECOMPILE_CC=2 \
  -DPython3_EXECUTABLE=/path/to/python-with-numpy-and-h5py
cmake --build build/full -j 1
ctest --test-dir build/full --output-on-failure -L functional -j 1
```

These tests also work with `WICKQC_BUILD_EXAMPLES=OFF`. Set
`WICKQC_BUILD_FUNCTIONAL_TESTS=OFF` for a C++-only test build. Project/library
builds with `BUILD_TESTING=OFF` need neither Python nor GTest.

Tolerances retain the numerical acceptance contract: MP2/common-guess energies
`1e-11`, first amplitudes `2e-11 + 2e-10*abs(reference)`, first residuals `2e-10`,
converged CCSD energy `1e-10`, amplitudes `2e-9 + 1e-8*abs(reference)`, SC/IC
energies `1e-8`, intermediate tensors `1e-10 + 1e-10*abs(reference)`.
IC checks NumPy `lstsq(rcond=None)` on the **same native matrices**. Separately
rounded null-space amplitudes are compared by matrix action and Weyl rank
bounds; the algorithm/cutoff is never changed to force agreement.

## Data and regeneration

See [data/README.md](data/README.md) for the cases and conventions. Reference
results are produced by PySCF and installed block2 on exactly the exported
orbitals. No external implementation source is copied into the fixtures.
To explicitly regenerate (requires PySCF, block2, NumPy and h5py):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /path/to/reference-python test/generate_data.py
# Or regenerate one named case:
/path/to/reference-python test/generate_data.py water-cc-pvdz
```

Review changed fixture data and rerun all functional tests before committing.
Files contain producer versions, geometry, basis and generator fingerprint.
Replacement is atomic per successfully generated fixture. Generation is serial.

## Known full-IC conditioning limit

`lih-triplet-root1-frozen1.h5` retains the unconstrained state-average LiH
triplet example. A mathematically zero IC subspace leaves a matrix of order
`1e-29`; a relative SVD cutoff can amplify that noise into a large spurious
energy. block2's full algorithm is also unstable there. This remains a
**diagnostic, not a passing energy-equivalence test**. No regularization or
alternative solver is substituted. Run the saved reproducer without PySCF:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 test/nevpt2.py build/full/test/wickqc_test_nevpt2 --conditioning
```

It checks intermediate tensors and same-input solves, and prints IC energy
differences without declaring energy acceptance. The regular excited-root
fixture constrains the singlet sector and retains the usual energy gate.
