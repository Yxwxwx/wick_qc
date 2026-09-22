"""SC/IC-NEVPT2 from fixed AO data; independent block2 results are stored offline."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np


def compare(actual, expected, name, atol=1e-10, rtol=1e-10):
    assert np.shape(actual) == np.shape(expected), (name,np.shape(actual),np.shape(expected))
    np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol,err_msg=name)


def run(executable, source, output, workspace, mode):
    process = subprocess.run([str(executable),str(source),str(output),workspace,mode,"diagnostics"],capture_output=True,text=True)
    assert process.returncode == 0, process.stdout+process.stderr
    assert not list(output.parent.glob("ao2mo-half-*"))


def check(executable, source, directory, conditioning=False, numpy_source=None):
    with h5py.File(source) as fixture:
        expected = fixture["expected"]
        assert bool(fixture.attrs["energy_acceptance"]) != conditioning
        if numpy_source is not None and not conditioning:
            values = {name:tensor[()] for name,tensor in expected["inputs"].items()}
            r = fixture["reference"].attrs
            values.update(np=np,ncore=int(r["ncore"]-r["frozen"]),ncas=int(r["nactive"]),nvirt=int(r["nmo"]-r["ncore"]-r["nactive"]))
            exec(numpy_source,values)
            for name,system in expected["ic/solves"].items():
                energy = -np.sum(system["rhs"][:]*system["amplitudes"][:])
                compare(values["compute_"+name](),energy,"NumPy generator "+name,1e-8,0)
            numpy_empty_pairs(numpy_source)

        for workspace,mode in (("stored","compiled"),("stored","runtime"),("incore","compiled"),("outcore","runtime")):
            output = directory/f"{source.stem}-{workspace}-{mode}.h5"
            run(executable,source,output,workspace,mode)
            with h5py.File(output) as f:
                assert f.attrs["complete"] == 1 and f.attrs["root"] == fixture["reference"].attrs["root"]
                for method in ("sc","ic"):
                    assert 0 < f[method].attrs["estimated_peak_bytes"] <= f.attrs["memory_limit_bytes"]
                    if not conditioning:
                        energy = expected[method+"/energy"][()]
                        compare(f[method].attrs["correlation_energy"],energy,method,1e-8,0)
                        compare(f[method].attrs["total_energy"],fixture["reference"].attrs["reference_energy"]+energy,method+" total",1e-8,0)
                        for name,value in expected[method+"/contributions"].items():
                            compare(f[method+"/contributions"].attrs[name],value[()],name,1e-8,0)
                    for name,tensors in expected[method+"/raw"].items():
                        for key,tensor in tensors.items():
                            compare(f[method+"/"+name+"/"+key][:],tensor[:],method+"/"+name+"/"+key)
                for name,tensor in f["inputs"].items():
                    compare(tensor[:],expected["inputs/"+name][:],name)
                for name,value in expected["sc/norms"].items():
                    compare(f["sc/"+name].attrs["norm"],value[()],name+" norm")
                for name,data in expected["ic/solves"].items():
                    native = f["ic/"+name]
                    for key in ("matrix","rhs","singular_values"):
                        compare(native[key][:],data[key][:],name+"/"+key)
                    matrix,rhs,amplitudes = (native[k][:] for k in ("matrix","rhs","amplitudes"))
                    # Same floating matrices, independent NumPy LAPACK solve.
                    same = [np.linalg.lstsq(h,r,rcond=None) for h,r in zip(matrix,rhs)]
                    compare(amplitudes,np.asarray([s[0] for s in same]).reshape(rhs.shape),name+" minimum-norm solve")
                    np.testing.assert_array_equal(native["ranks"][:],[s[2] for s in same])
                    if conditioning:
                        continue
                    # Separately rounded null spaces have ill-defined coefficients.
                    # Check their action and Weyl rank bounds, never raise cutoffs.
                    action = np.einsum("bij,bj->bi",data["matrix"][:],amplitudes-data["amplitudes"][:])
                    compare(action,np.zeros_like(rhs),name+" amplitude action")
                    for h,href,singular,rank in zip(matrix,data["matrix"][:],data["singular_values"][:],native["ranks"][:]):
                        if not len(singular):
                            continue
                        delta = np.linalg.norm(h-href)
                        relative = np.finfo(float).eps*len(singular)
                        cutoff,bound = relative*singular[0],(1+relative)*delta
                        lo,hi = int(np.sum(singular>cutoff+bound)),int(np.sum(singular>cutoff-bound))
                        assert lo <= rank <= hi,(name,rank,lo,hi)
                if conditioning:
                    difference = f["ic"].attrs["correlation_energy"]-expected["ic/energy"][()]
                    print(f"{source.stem} {workspace}/{mode}: IC energy difference {difference:.6g}; diagnostic only, no energy acceptance",flush=True)
        output = directory/(source.stem+"-rejected.h5")
        process = subprocess.run([str(executable),str(source),str(output),"--memory","1"],capture_output=True,text=True)
        assert process.returncode == 1 and "memory budget" in process.stderr and not output.exists()
        if not conditioning:
            print(source.stem+": SC/IC energies, intermediate tensors and solves passed",flush=True)


def numpy_empty_pairs(source):
    # One doubly occupied active orbital has no antisymmetric active pair.
    values = dict(np=np,ncore=2,ncas=1,nvirt=2,orbeI=np.full(2,-1.),orbeE=np.full(2,1.),
                  deltaII=np.eye(2),deltaAA=np.eye(1),deltaEE=np.eye(2))
    shapes = {"I":2,"A":1,"E":2}
    # All block inputs used by the generated functions, including zero coupling.
    import re
    for name,axes in re.findall(r"\b([whf]([IAE]{2,4}))\b",source):
        values[name] = np.zeros(tuple(shapes[c] for c in axes))
    for order in range(1,5):
        values[f"E{order}"] = np.full((1,)*(2*order),2. if order<=2 else 0.)
    for rank in range(1,4):
        values[f"ident{rank}"] = np.ones((1,)*rank)
    exec(source,values)
    for name,compute in values.items():
        if name.startswith("compute_"):
            assert compute() == 0,name
    # The reference's reshape(-1,0) is ambiguous for its empty p<q selection.
    rhs = np.zeros((2,2,1,1))
    grid = np.indices(rhs.shape)
    restricted = rhs[(grid[0]<grid[1]) & (grid[2]<grid[3])]
    try:
        restricted.reshape(-1,0)
    except ValueError:
        return
    raise AssertionError("Expected the reference empty reshape to fail")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable",type=Path)
    parser.add_argument("--conditioning",action="store_true",help="Run the documented triplet sensitivity reproducer, not an energy acceptance test")
    parser.add_argument("--generator",type=Path)
    args = parser.parse_args()
    numpy_source = subprocess.run([str(args.generator),"ic-nevpt2"],capture_output=True,text=True,check=True).stdout if args.generator else None
    names = ("lih-triplet-root1-frozen1",) if args.conditioning else ("lih","lih-singlet-root1-frozen1","water-frozen1")
    with tempfile.TemporaryDirectory(prefix="wickqc-nevpt2-") as temporary:
        for name in names:
            check(args.executable.resolve(),Path(__file__).with_name("data")/(name+".h5"),Path(temporary),args.conditioning,numpy_source)

if __name__ == "__main__":
    main()
