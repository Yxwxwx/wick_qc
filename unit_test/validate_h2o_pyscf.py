"""H2O/cc-pVDZ: PySCF RHF/MP2 and one common-initial-guess CCSD update.

Python supplies molecular data and the independent reference. The executable
computes MP2 amplitudes/energy and CCSD residuals/update/energy using Wick+NDArray.
Both chemist and physicist integral conventions must pass, including amplitudes.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyscf
from pyscf import ao2mo, cc, gto, lib, mp, scf


def write_array(directory, name, value):
    array = np.asarray(value, dtype="<f8", order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Nonfinite input {name}")
    array.tofile(directory / name)


def read_array(directory, name, shape):
    return np.fromfile(directory / name, dtype="<f8").reshape(shape)


def compare(actual, expected, name, atol, rtol=0.0):
    if not np.all(np.isfinite(actual)):
        raise AssertionError(f"{name}: nonfinite result")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    return float(np.max(np.abs(np.asarray(actual) - expected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, nargs="+", default=[Path("build/test_h2o_pyscf")])
    parser.add_argument("--output", type=Path, default=Path("build/h2o_ccpvdz"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    lib.num_threads(args.threads)
    output = args.output.resolve()
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    geometry = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    mol = gto.M(atom=geometry, basis="cc-pvdz", unit="Angstrom", charge=0, spin=0,
                cart=False, symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    reference_mp = mp.MP2(mf, frozen=0)
    mp_energy, mp_t2 = reference_mp.kernel()
    reference_cc = cc.CCSD(mf, frozen=0)
    reference_cc.diis = False
    reference_cc.level_shift = 0
    reference_cc.iterative_damping = 1.0
    reference_cc.incore_complete = True
    eris = reference_cc.ao2mo()
    _, initial_t1, initial_t2 = reference_cc.init_amps(eris)
    initial_energy = reference_cc.energy(initial_t1, initial_t2, eris)
    first_t1, first_t2 = reference_cc.update_amps(initial_t1.copy(), initial_t2.copy(), eris)
    first_energy = reference_cc.energy(first_t1, first_t2, eris)
    nocc, nvir = initial_t1.shape
    nmo = nocc + nvir
    integrals = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape((nmo,) * 4)
    (inputs / "dimensions.txt").write_text(f"{nocc} {nvir}\n")
    for name, array in {
        "eps_mp.bin": mf.mo_energy,
        "eps_cc.bin": eris.mo_energy,
        "fock.bin": eris.fock,
        "eri_chemist.bin": integrals,
        "cc_t1_initial.bin": initial_t1.T,
        "cc_t2_initial.bin": initial_t2.transpose(2, 3, 0, 1),
    }.items():
        write_array(inputs, name, array)
    gap = eris.mo_energy[:nocc, None] - eris.mo_energy[None, nocc:]
    residual1 = (first_t1 - initial_t1) * gap
    residual2 = (first_t2 - initial_t2) * (gap[:, None, :, None] + gap[None, :, None, :])
    covariant1 = 2 * residual1.T
    covariant2 = (4 * residual2 - 2 * residual2.transpose(1, 0, 2, 3)).transpose(2, 3, 0, 1)
    summary = {
        "pyscf_version": pyscf.__version__, "geometry_angstrom": geometry,
        "basis": "cc-pvdz", "spherical": True, "frozen_orbitals": 0,
        "nocc": nocc, "nvir": nvir, "threads": args.threads,
        "cc_initial_guess": "PySCF init_amps (MP2 doubles and f_ov/denominator singles)",
        "cc_step": "one Jacobi update; no DIIS, damping, or level shift",
        "reference": {"rhf_total": float(mf.e_tot), "mp2_correlation": float(mp_energy),
                      "mp2_total": float(mf.e_tot + mp_energy),
                      "cc_initial_correlation": float(initial_energy),
                      "cc_first_correlation": float(first_energy),
                      "cc_first_total": float(mf.e_tot + first_energy)},
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs.iterdir())},
        "tolerances": {"energy_absolute": 1e-11, "amplitude_absolute": 2e-11,
                       "amplitude_relative": 2e-10, "residual_absolute": 2e-10},
        "runs": [], "passed": False,
    }
    # A failed rerun must not leave a success report from an older calculation.
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PySCF {pyscf.__version__}: H2O/cc-pVDZ, {nocc} occupied + {nvir} virtual", flush=True)
    print(json.dumps(summary["reference"], indent=2), flush=True)
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "TBLIS_NUM_THREADS"):
        environment[variable] = str(args.threads)
    for number, executable in enumerate(args.executable):
        directory = output / f"run_{number}"
        subprocess.run([str(executable.resolve()), str(inputs), str(directory)], env=environment, check=True)
        for convention in ("chemist", "physicist"):
            result_dir = directory / convention
            values = json.loads((result_dir / "energies.json").read_text())
            errors = {}
            for name, expected in (("mp2_correlation", mp_energy), ("cc_initial_correlation", initial_energy),
                                   ("cc_first_correlation", first_energy)):
                errors[name] = compare(values[name], expected, name, 1e-11)
            for name, shape, expected in (
                ("mp2_t2", (nvir, nvir, nocc, nocc), mp_t2.transpose(2, 3, 0, 1)),
                ("cc_t1_first", (nvir, nocc), first_t1.T),
                ("cc_t2_first", (nvir, nvir, nocc, nocc), first_t2.transpose(2, 3, 0, 1)),
            ):
                errors[name] = compare(read_array(result_dir, name + ".bin", shape), expected, name, 2e-11, 2e-10)
            for name, expected in (("cc_r1_initial", covariant1), ("cc_r2_initial", covariant2)):
                errors[name] = compare(read_array(result_dir, name + ".bin", expected.shape), expected, name, 2e-10, 2e-10)
            for name in ("mp2_residual1_norm", "mp2_residual2_norm"):
                compare(values[name], 0, name, 2e-10)
            summary["runs"].append({"executable": str(executable.resolve()), "convention": convention,
                                    "values": values, "max_absolute_errors": errors})
            print(f"PASS {executable.parent.name}/{convention}: energy error {errors['cc_first_correlation']:.3e}, "
                  f"t1 {errors['cc_t1_first']:.3e}, t2 {errors['cc_t2_first']:.3e}", flush=True)
    summary["passed"] = True
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"All molecular comparisons passed. Report: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
