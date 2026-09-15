# Molecular comparisons

`compare_pyscf.py` runs RHF for H2O/cc-pVDZ (spherical AOs, no frozen core),
exports MO integrals after the C++ executable has been built, and compares:

- MP2 correlation energy, denominator amplitudes, and zero residuals;
- CCSD energy at a shared initial guess;
- every singles/doubles residual and amplitude after one Jacobi update;
- CCSD correlation energy after that update.

The default checks both chemist/physicist conventions and both precompiled/runtime
Wick paths. The C++ program executes the contractions and amplitude update;
Python supplies the independent PySCF reference. The covariant spin metric is
inverted in the C++ RHF example before updating amplitudes. DIIS, damping and
level shifts are disabled in this first-step comparison.

```sh
module load googletest/1.15.0
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  '-DWICKQC_PRECOMPILE_CONVENTIONS=chemist;physicist'
cmake --build build -j 4
python3 test/compare_pyscf.py --executable build/example/example_rhf
```

For a build with only chemist kernels, pass `--convention chemist`. Several
backend executables may be supplied to `--executable`. A second runtime shape
and nonzero initial singles can be checked using `--basis sto-3g
--initial-guess perturbed --output build/pyscf_sto3g`. Neither option rebuilds
anything. Missing executables, missing precompiled methods, or numerical
mismatches fail the run. Results, input fingerprints, executable fingerprints,
and tolerances are saved in `comparison.json` under `--output`.

This is a molecular integration test, separate from the small GTest suite in
`unit_test/`. It needs NumPy and PySCF; normal C++ builds do not import Python.
