"""Offline fixture generation only: requires PySCF, block2, NumPy and h5py.

Normal builds and tests never execute this script or import PySCF/block2.
Existing fixtures are replaced atomically after each reference calculation.
"""
import argparse
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import h5py
import numpy as np
import pyscf
import block2
from pyscf import gto, mp, scf, mcscf
from pyscf.cc import rccsd
from pyblock2.cc.rccsd import WickRCCSD
from pyblock2.icmr import eri_helper, icnevpt2_full as icref, scnevpt2 as scref

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
from export_reference import export_rhf, export_casscf

def local_name(key):
    return key[:-1] + {"+": "_plus", "-": "_minus", "1": "", "*": ""}[key[-1]]


def reference(mc, frozen, root):
    sc = scref.WickSCNEVPT2(mc, frozen=frozen)
    ic = icref.WickICNEVPT2(mc, frozen=frozen)
    sc.canonicalized = ic.canonicalized = True
    sc.run(root=root)
    solves = []
    original = icref._linear_solve

    def capture(matrix, rhs):
        # Observe the actual reference assembly; preserve its solver unchanged.
        solution = original(matrix, rhs)
        svd = [np.linalg.lstsq(h, r, rcond=None) for h, r in zip(matrix, rhs)]
        solves.append(dict(matrix=matrix.copy(), rhs=rhs.copy(), amplitudes=solution.copy(),
                           ranks=np.asarray([s[2] for s in svd], dtype=int),
                           singular_values=np.asarray([s[3] for s in svd]).reshape(rhs.shape)))
        return solution

    with patch.object(icref, "_linear_solve", capture):
        ic.run(root=root)
    keys = [key for key in icref.sub_spaces if not key.endswith("2")]
    solves = dict(zip(map(local_name, keys), solves, strict=True))
    nc, na = mc.ncore - frozen, mc.ncas
    ne = mc.mo_coeff.shape[1] - mc.ncore - na
    pdms = eri_helper.init_pdms(mc, scref.pdm_eqs, root=root)
    eris = sc.eris
    spaces = {"I": slice(nc), "A": slice(nc, nc + na), "E": slice(nc + na, None)}
    values = dict(np=np, ncore=nc, ncas=na, nvirt=ne,
                  orbeI=mc.mo_energy[frozen:mc.ncore], orbeE=mc.mo_energy[mc.ncore + na:],
                  deltaII=np.eye(nc), deltaAA=np.eye(na), deltaEE=np.eye(ne),
                  ident1=np.ones((1,)), ident2=np.ones((1, 1)), ident3=np.ones((1, 1, 1)))
    values.update({f"E{k}": d for k, d in enumerate(pdms, 1)})
    for axes in ("AA", "AI", "EI", "EA"):
        values["h" + axes] = values["f" + axes] = eris.h1eff[tuple(spaces[c] for c in axes)]
    for axes in ("AAAA", "EAAA", "EAIA", "EAAI", "AAIA", "EEIA", "EAII", "EEAA", "AAII", "EEII",
                 "AAAI", "AEAI", "EEAI"):
        values["w" + axes] = eris.get_phys(axes)
    raw_sc, raw_ic = {}, {}
    for key in scref.sub_spaces:
        shape = tuple(nc if c in "ij" else ne for c in key)
        arrays = dict(norm=np.zeros(shape), ener=np.zeros(shape))
        exec(scref.norm_eqs[key] + scref.ener_eqs[key], {**values, **arrays})
        raw_sc[key] = dict(norm=arrays["norm"], hexp=arrays["ener"])
    for key in keys:
        rank = len(key) - 5
        outer = key[:4 - rank]
        shape = tuple(nc if c in "ij" else ne for c in outer)
        rhs_axes = outer + key[4:-1]
        rhs_shape, h_shape = shape + (na,) * rank, shape + (na,) * (2 * rank)
        raw = {}
        components = ("1", "2") if key.endswith("1") else ("",)
        for row in components:
            row_key = key[:-1] + row if row else key
            rhs_name = "rheq" + row
            raw[rhs_name] = np.zeros(rhs_shape)
            exec(icref.rhhk_eqs[row_key].to_einsum(icref.PT(f"{rhs_name}[{rhs_axes}]")), {**values, **raw})
            for col in components:
                h_name = "hexp" + row + col
                raw[h_name] = np.zeros(h_shape)
                equation = icref.ener_eqs[row_key] if row == col else icref.ener2_eqs[row_key]
                exec(equation.to_einsum(icref.PT(f"{h_name}[{key[:-1]}]")), {**values, **raw})
        raw_ic[local_name(key)] = raw
    return sc, ic, solves, raw_sc, raw_ic, values


def save(group, values):
    for name, value in values.items():
        if isinstance(value, dict):
            save(group.create_group(name), value)
        else:
            a = np.asarray(value)
            group.create_dataset(name, data=a, **(dict(compression="gzip", shuffle=True) if a.ndim and a.size else {}))


def add_ao(file, calculation):
    mf = calculation if isinstance(calculation, scf.hf.SCF) else calculation._scf
    save(file.create_group("ao"), dict(
        eri=calculation.mol.intor("int2e", aosym="s1"),
        h1e=calculation.get_hcore(), fock=mf.get_fock(dm=mf.make_rdm1())))
    file.attrs.update(fixture_schema="wickqc.test.v1", geometry=str(calculation.mol.atom),
                      basis=str(calculation.mol.basis), pyscf_version=pyscf.__version__,
                      block2_version=str(getattr(block2, "__version__", "unknown")),
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def rhf(path, basis, frozen):
    mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
    export_rhf(path, mf, frozen=frozen)
    emp, tmp = mp.MP2(mf, frozen=frozen).kernel()
    ref, oracle = rccsd.RCCSD(mf, frozen=frozen), WickRCCSD(mf, frozen=frozen)
    eris = ref.ao2mo()
    _, t1, t2 = ref.init_amps(eris)
    t1next, t2next = ref.update_amps(t1.copy(), t2.copy(), eris)
    ni = t1.shape[0]
    gap = eris.mo_energy[:ni, None] - eris.mo_energy[None, ni:]
    values = dict(mp2=dict(energy=emp, t2=tmp),
                  initial=dict(energy=ref.energy(t1, t2, eris), t1=t1, t2=t2,
                               residual1=(t1next-t1)*gap,
                               residual2=(t2next-t2)*(gap[:,None,:,None]+gap[None,:,None,:])),
                  first=dict(energy=ref.energy(t1next,t2next,eris), t1=t1next,t2=t2next))
    bt1, bt2 = oracle.update_amps(t1.copy(), t2.copy(), eris)
    values["block2_first"] = dict(t1=bt1, t2=bt2)
    for key, method in (("ccsd",ref),("block2_ccsd",oracle)):
        method.conv_tol, method.conv_tol_normt, method.max_cycle = 1e-13, 1e-11, 150
        ec, tc1, tc2 = method.kernel(t1.copy(), t2.copy(), eris=eris)
        assert method.converged
        values[key] = dict(energy=ec, t1=tc1, t2=tc2)
    with h5py.File(path,"r+") as f:
        add_ao(f,mf)
        save(f.create_group("expected"),values)


def nevpt2(path, case):
    water = case == "water-frozen1"
    mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587" if water else "Li 0 0 0; H 0 0 1.6",
                basis="6-31g" if water else "sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    mc = mcscf.CASSCF(mf,4 if water else 2,4 if water else 2)
    root = int("root1" in case)
    frozen = int("frozen1" in case)
    if "singlet" in case:
        mc = mc.fix_spin_(ss=0)
    if root:
        mc = mc.state_average_([0.5,0.5])
    mc.run(conv_tol=1e-10)
    final = export_casscf(path,mc,frozen=frozen,root=root)
    sc,ic,solves,raw_sc,raw_ic,values = reference(final,frozen,root)
    values = {k:v for k,v in values.items() if isinstance(v,np.ndarray)}
    with h5py.File(path,"r+") as f:
        add_ao(f,final)
        f.attrs["energy_acceptance"] = "triplet" not in case
        save(f.create_group("expected"),dict(inputs=values,sc=dict(energy=sc.e_corr,
             contributions=sc.sub_eners,norms=sc.sub_norms,raw=raw_sc),
             ic=dict(energy=ic.e_corr,contributions=ic.sub_eners,raw=raw_ic,solves=solves)))
        if root:
            f.attrs["spin_square"] = float(mc.fcisolver.states_spin_square(mc.ci,2,mc.nelecas)[0][root])


def main():
    cases = {"water-cc-pvdz": lambda p: rhf(p,"cc-pvdz",0),
             "water-6-31g-frozen1": lambda p: rhf(p,"6-31g",1)}
    for name in ("lih", "lih-singlet-root1-frozen1", "water-frozen1", "lih-triplet-root1-frozen1"):
        cases[name] = lambda p, name=name: nevpt2(p,name)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="Default: regenerate all fixtures")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name in args.cases or cases:
        if name not in cases:
            parser.error("Unknown fixture: " + name)
        target = args.output/(name+".h5")
        temporary = target.with_suffix(".tmp.h5")
        temporary.unlink(missing_ok=True)
        print("Generating " + name, flush=True)
        try:
            cases[name](temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved {target} ({target.stat().st_size} bytes)",flush=True)

if __name__ == "__main__":
    main()
