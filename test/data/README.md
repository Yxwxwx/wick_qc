# Molecular fixtures

These files are inputs and independent expected results, not build products.
They are intended for version control. Normal tests never regenerate them.

| File | Reference state |
| --- | --- |
| `water-cc-pvdz.h5` | H2O RHF/cc-pVDZ; no frozen core; MP2 and CCSD |
| `water-6-31g-frozen1.h5` | H2O RHF/6-31G; one frozen orbital; MP2 and CCSD |
| `lih.h5` | LiH CASSCF(2e,2o)/STO-3G ground singlet; SC/IC-NEVPT2 |
| `lih-singlet-root1-frozen1.h5` | Equal-weight two-root singlet state average; root 1, one frozen orbital |
| `water-frozen1.h5` | H2O CASSCF(4e,4o)/6-31G; one frozen orbital; SC/IC-NEVPT2 |
| `lih-triplet-root1-frozen1.h5` | Unconstrained two-root state average, triplet root 1; diagnostic only |

Geometry (angstrom): water `O 0 0 0; H 0 -.757 .587; H 0 .757 .587`;
LiH `Li 0 0 0; H 0 0 1.6`. AOs are real spherical functions. AO ERIs are
full chemist tensors `(uv|wx)`; `coefficients/mo/real` maps AO rows to MO columns.
`ao/h1e` is the bare core Hamiltonian and `ao/fock` is the SCF Fock matrix.
All float arrays are IEEE float64. Dimensions come from the HDF5 datasets.

The files retain the `ao2mo.input.v1` and `wickqc.reference.v1` input schemas.
`reference` records orbital partition, energies, occupations, identity and root.
`reference/rdms/E1`–`E4` use block2's spin-free normal-ordered convention in the
exported active basis. Core/virtual orbitals are canonical; active orbitals are
not naturalized. Frozen orbitals are included in inactive-core dressing.
Reference total energies include nuclear and core constants exactly once.

`expected` contains PySCF MP2/RCCSD and block2 WickRCCSD results, or block2
SC/IC subspace energies, raw tensors, assembled systems and SVD results.
CCSD uses the stored `expected/initial/t1,t2` as the common guess, with PySCF's
occupied-first order; the C++ driver permutes to its virtual-first convention.
The test-only `ao` group is always consumed by the stored-AO execution path.

Regeneration commands and acceptance tolerances are in [../README.md](../README.md).
Producer versions and generator SHA-256 are recorded in each file. A different
SCF/CASSCF build may rotate orbitals or round null spaces differently; regenerated
files require review and fresh method comparisons, not bytewise equality.
