"""Reference I/O and binding validation using fixed data; no reference package imports."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import h5py
import numpy as np

EXE = Path(sys.argv[1]).resolve()

def run(source, destination, valid=True, outcore=False):
    command = [str(EXE), str(source), str(destination)] + (["outcore"] if outcore else [])
    result = subprocess.run(command, capture_output=True, text=True)
    if valid:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, f"accepted {source.name}"
        assert not destination.exists(), f"published output for {source.name}"
    return result


def compare(name, actual, expected):
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    error = float(np.max(np.abs(actual - expected), initial=0))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, err_msg=name)
    return error


def check(source, directory):
    output = directory/(source.stem+".out.h5")
    run(source,output)
    other = output.with_suffix(".outcore.h5")
    run(source,other,outcore=True)
    assert not list(directory.glob("ao2mo-half-*"))
    with h5py.File(source) as data, h5py.File(output) as result, h5py.File(other) as disk:
        r = data["reference"]
        for name in ("orbital_energies","occupations","fock_mo"):
            compare(name,result[name][:],r[name][:])
        assert result.attrs["root"] == r.attrs["root"]
        assert result.attrs["reference_energy"] == r.attrs["reference_energy"]
        assert result.attrs["loading_budget_bytes"] <= 64<<20
        for name in result:
            compare(name,disk[name][:],result[name][:])
        if r.attrs["kind"] == "casscf":
            for order in range(1,5):
                compare(f"E{order}",result[f"E{order}"][:],r[f"rdms/E{order}"][:])
            for name in result:
                if name.startswith("sc_"):
                    compare(name,result[name][:],data["expected/inputs/"+name[3:]][:])
        else:
            c = data["coefficients/mo/real"][:]
            eri = np.einsum("up,uvwx->pvwx",c,data["ao/eri"][:],optimize=True)
            eri = np.einsum("vq,pvwx->pqwx",c,eri,optimize=True)
            eri = np.einsum("wr,pqwx->pqrx",c,eri,optimize=True)
            eri = np.einsum("xs,pqrx->pqrs",c,eri,optimize=True)
            spaces = {"I":slice(r.attrs["frozen"],r.attrs["ncore"]),"E":slice(r.attrs["ncore"],r.attrs["nmo"])}
            for name in result:
                if name.startswith(("chem_","phys_")):
                    key = name.split("_",1)[1]
                    if key.startswith("v"):
                        tensor = eri if name.startswith("chem_") else eri.transpose(0,2,1,3)
                        expected = tensor[tuple(spaces[c] for c in key[1:])]
                    elif key.startswith("f"):
                        expected = r["fock_mo"][:][tuple(spaces[c] for c in key[1:])]
                    else:
                        expected = r["orbital_energies"][:][spaces[key[-1]]]
                    compare(name,result[name][:],expected)
    print(source.stem+": reference I/O and selective bindings passed",flush=True)

def malformed(source, directory):
    def dtype(f):
        path = "reference/rdms/E4"
        value = f[path][:].astype("<f4")
        del f[path]
        f[path] = value

    changes = {
        "schema": lambda f: f["reference"].attrs.__setitem__("schema", "unknown"),
        "incomplete": lambda f: f["reference"].attrs.__setitem__("complete", 0),
        "kind": lambda f: f["reference"].attrs.__setitem__("kind", "uhf"),
        "representation": lambda f: f["reference"].attrs.__setitem__("representation", "spinor"),
        "energy-convention": lambda f: f["reference"].attrs.__setitem__("energy_convention", "electronic_only"),
        "partition": lambda f: f["reference"].attrs.__setitem__("ncore", 0),
        "noninteger": lambda f: f["reference"].attrs.__setitem__("nalpha", 1.5),
        "root": lambda f: f["reference/rdms"].attrs.__setitem__("root", 99),
        "identity": lambda f: f["reference/fock_mo"].attrs.__setitem__("orbital_identity", "other"),
        "mo-values": lambda f: f["coefficients/mo/real"].__setitem__((0, 0), 123.),
        "ao-values": lambda f: f["basis/env"].__setitem__(-1, 123.),
        "orbital-order": lambda f: f["reference/orbital_ids"].__setitem__(slice(None), f["reference/orbital_ids"][:][::-1]),
        "active-order": lambda f: f["reference/rdms/active_orbital_ids"].__setitem__(slice(None), f["reference/rdms/active_orbital_ids"][:][::-1]),
        "rdm-convention": lambda f: f["reference/rdms"].attrs.__setitem__("convention", "pyscf_raw"),
        "missing-E4": lambda f: f["reference/rdms"].__delitem__("E4"),
        "dtype": dtype,
        "nan": lambda f: f["reference/rdms/E1"].__setitem__((0, 0), np.nan),
        "occupation": lambda f: f["reference/occupations"].__setitem__(0, 0.),
        "dressing": lambda f: f["one_electron"].attrs.__setitem__("dressing_state", "all-inactive-folded"),
    }
    for name, edit in changes.items():
        bad = directory / f"bad-{name}.h5"
        shutil.copyfile(source, bad)
        with h5py.File(bad, "r+") as f:
            edit(f)
        result = run(bad, bad.with_suffix(".out.h5"), valid=False)
        assert result.stderr.strip()


if __name__ == "__main__":
    data = Path(__file__).with_name("data")
    with tempfile.TemporaryDirectory(prefix="wickqc-reference-") as temporary:
        directory=Path(temporary)
        for name in ("water-6-31g-frozen1","lih","lih-singlet-root1-frozen1","water-frozen1"):
            check(data/(name+".h5"),directory)
        malformed(data/"water-frozen1.h5",directory)
    print("Malformed reference inputs rejected")
