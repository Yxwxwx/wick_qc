"""MP2 and complete CCSD against fixed independent references (no PySCF)."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np


def compare(actual, expected, name, atol=1e-11, rtol=0):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)


def run(executable, source, output, convention="chemist", workspace="stored", mode="compiled", iterations=100, dense=False):
    command = [str(executable), str(source), str(output), convention, workspace, str(iterations), mode]
    if dense:
        command.append("dense")
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == (2 if iterations == 1 else 0), proc.stdout + proc.stderr
    assert not list(output.parent.glob("ao2mo-half-*"))


def check(executable, source, directory, runtime_only=False):
    with h5py.File(source) as fixture:
        expected = fixture["expected"]
        reference_energy = fixture["reference"].attrs["reference_energy"]
        baseline = None
        for convention, workspace, mode, dense in (
            ("chemist", "stored", "compiled", False),
            ("physicist", "stored", "runtime", False),
            ("chemist", "incore", "compiled", False),
            ("chemist", "outcore", "runtime", False),
            ("chemist", "incore", "compiled", True),
        ):
            if runtime_only:
                mode = "runtime"
            output = directory / f"{source.stem}-{convention}-{workspace}-{mode}-{dense}.h5"
            output.unlink(missing_ok=True)
            run(executable, source, output, convention, workspace, mode, dense=dense)
            with h5py.File(output) as f:
                assert f.attrs["complete"] == 1
                peaks = [int(f[name].attrs["estimated_peak_bytes"]) for name in ("mp2", "initial", "first", "ccsd")]
                assert 0 < max(peaks) <= f.attrs["memory_limit_bytes"]
                for name in ("mp2", "initial", "first", "ccsd"):
                    atol = 1e-10 if name == "ccsd" else 1e-11
                    compare(f[name].attrs["correlation_energy"], expected[name+"/energy"][()], name, atol)
                    compare(f[name].attrs["total_energy"], reference_energy + expected[name+"/energy"][()], name+" total", atol)
                    for amplitude in ("t2",) if name == "mp2" else ("t1", "t2"):
                        actual = f[name+"/"+amplitude][:]
                        actual = actual.T if amplitude == "t1" else actual.transpose(2,3,0,1)
                        compare(actual, expected[name+"/"+amplitude][:], name+"/"+amplitude,
                                2e-9 if name == "ccsd" else 2e-11, 1e-8 if name == "ccsd" else 2e-10)
                        if name in ("first", "ccsd"):
                            compare(actual, expected["block2_"+name+"/"+amplitude][:], "block2 "+name+"/"+amplitude,
                                    2e-9 if name == "ccsd" else 2e-11, 1e-8 if name == "ccsd" else 2e-10)
                for name in ("residual1", "residual2"):
                    actual = f["initial/"+name][:]
                    actual = actual.T if name == "residual1" else actual.transpose(2,3,0,1)
                    compare(actual,expected["initial/"+name][:],name,2e-10)
                assert f["mp2"].attrs["residual_norm"] < 1e-10
                final = f["ccsd"]
                assert final.attrs["converged"] == 1 and final.attrs["residual_norm"] < 1e-10
                history = final["history"][:]
                assert len(history) == final.attrs["iterations"] + 1 and abs(history[-1,2]) < 1e-12
                compare(final.attrs["correlation_energy"],expected["block2_ccsd/energy"][()],"block2 CCSD",1e-10)
                current = (final.attrs["correlation_energy"],final["t1"][:],final["t2"][:])
                if baseline is not None:
                    for value, previous in zip(current,baseline,strict=True):
                        compare(value,previous,"execution paths",2e-11)
                baseline = current
                if workspace != "stored" and not dense:
                    n = int(fixture["reference"].attrs["nmo"]-fixture["reference"].attrs["frozen"])
                    assert f.attrs["integral_blocks"] > 1 and f.attrs["integral_bytes"] < n**4*8
        output = directory / (source.stem+"-limited.h5")
        run(executable,source,output,iterations=1,mode="runtime" if runtime_only else "compiled")
        with h5py.File(output) as f:
            assert f["ccsd"].attrs["converged"] == 0 and f["ccsd"].attrs["iterations"] == 1
        output = directory / (source.stem+"-rejected.h5")
        process = subprocess.run([str(executable),str(source),str(output),"--memory","1"],capture_output=True,text=True)
        assert process.returncode == 1 and "memory budget" in process.stderr and not output.exists()
        print(source.stem+": MP2, common-guess CCSD update and converged CCSD passed",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable",type=Path)
    parser.add_argument("--runtime-only",action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="wickqc-rhf-") as temporary:
        for name in ("water-cc-pvdz", "water-6-31g-frozen1"):
            check(args.executable.resolve(),Path(__file__).with_name("data")/(name+".h5"),Path(temporary),args.runtime_only)

if __name__ == "__main__":
    main()
