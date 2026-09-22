"""Export converged real spatial PySCF references, without transforming ERIs.

The returned shallow copy contains the exact final orbitals/CI used for export
and can be passed to an independent oracle. The original calculation is intact.
RDM conversion uses the installed block2 helper once, during export only.
"""

import copy
import hashlib
import operator
from pathlib import Path
import struct

import h5py
import numpy as np

from export_input import export


def _fingerprint(domain, words=(), arrays=()):
    # Matches ao2mo::Fingerprint, including domain/lengths and scalar LE bits.
    value = 14695981039346656037

    def feed(data):
        nonlocal value
        for byte in data:
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)

    feed(struct.pack("<Q", len(domain)))
    feed(domain.encode("ascii"))
    for word in words:
        feed(struct.pack("<Q", word))
    for array in arrays:
        array = np.asarray(array)
        feed(struct.pack("<Q", array.size))
        feed(np.ascontiguousarray(array, dtype="<f8" if array.dtype.kind == "f" else "<i8").tobytes())
    return f"fnv1a64-le-v1:{value:016x}"


def _real(array, shape, name):
    value = np.asarray(array)
    if value.shape != shape or np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError(f"invalid real spatial {name}")
    return np.asarray(value, dtype="<f8")


def _write(path, calculation, *, kind, energies, occupations, ncore, ncas,
           frozen, root, root_count, reference_energy, rdms=(), provenance=None):
    mol = calculation.mol
    c = np.asarray(calculation.mo_coeff)
    if c.ndim != 2:
        raise ValueError("expected restricted spatial MO coefficients")
    c = _real(c, (mol.nao_nr(), c.shape[1]), "coefficients")
    nmo = c.shape[1]
    if not 0 <= frozen <= ncore <= nmo - ncas or not 0 <= root < root_count:
        raise ValueError("invalid frozen/partition/root")
    mf = calculation if kind == "rhf" else calculation._scf
    if getattr(mf, "with_df", None) is not None or getattr(mf, "with_x2c", None) is not None:
        raise ValueError("this method exporter requires nonrelativistic full-Coulomb references")
    h1e = _real(c.T @ calculation.get_hcore() @ c, (nmo, nmo), "h1e")
    fock = _real(c.T @ mf.get_fock(dm=mf.make_rdm1()) @ c, (nmo, nmo), "Fock")
    energies = _real(energies, (nmo,), "energies")
    if kind == "rhf" and np.max(np.abs(fock - np.diag(energies)), initial=0) > 1e-8:
        raise ValueError("RHF Fock/energies are not canonical within 1e-8; tighten SCF convergence")
    occupations = _real(occupations, (nmo,), "occupations")
    rdms = tuple(_real(rdm, (ncas,) * (2 * order), f"E{order}")
                 for order, rdm in enumerate(rdms, 1))
    if kind == "casscf" and len(rdms) != 4:
        raise ValueError("native NEVPT2 requires E1--E4; missing E4 is not a zero tensor")
    if not np.isfinite(reference_energy):
        raise ValueError("non-finite reference energy")
    orbital_ids = np.arange(nmo, dtype="<i8")
    provenance = dict(provenance or {})
    provenance["reference_exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if kind == "casscf":
        import block2
        from pyblock2.icmr import eri_helper, scnevpt2
        provenance["block2_version"] = str(getattr(block2, "__version__", "unknown"))
        for module in (eri_helper, scnevpt2):
            provenance[Path(module.__file__).stem + "_sha256"] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    # Exclusive creation preserves an existing destination on failure. A failed
    # attachment leaves the new file incomplete, which ReadReference rejects.
    export(path, mol, {"mo": c}, partition=dict(ncore=ncore, ncas=ncas, nmo=nmo, frozen=frozen),
           h1e=h1e, operator_label="nonrelativistic_full_coulomb_hcore",
           orbital_ids=orbital_ids, provenance=provenance)
    with h5py.File(path, "r+") as f:
        r = f.create_group("reference")
        r.attrs.update(schema="wickqc.reference.v1", complete=0, kind=kind,
                       representation="real_spatial", coefficient_family="mo",
                       energy_convention="total_including_nuclear_and_core",
                       reference_energy=float(reference_energy), root=root, root_count=root_count,
                       nalpha=mol.nelec[0], nbeta=mol.nelec[1], nmo=nmo, ncore=ncore,
                       nactive=ncas, frozen=frozen,
                       orbital_state="canonical" if kind == "rhf" else "core_virtual_canonical",
                       energy_source="pyscf.scf.mo_energy" if kind == "rhf" else "pyscf.casscf.canonicalize.cas_natorb_false",
                       fock_source="pyscf.scf.get_fock.final_scf_density")
        identity = str(f["coefficients/mo"].attrs["identity"])
        r.attrs["orbital_identity"] = identity
        r.attrs["basis_fingerprint"] = _fingerprint("ao2mo.basis.real-spherical.v1", arrays=(mol._atm, mol._bas, mol._env))
        r.attrs["coefficient_fingerprint"] = _fingerprint("ao2mo.coefficients.float64.v1", c.shape, (c, np.empty(0)))
        r["orbital_ids"] = orbital_ids
        for name, value in dict(orbital_energies=energies, occupations=occupations, fock_mo=fock).items():
            r[name] = value
            r[name].attrs["orbital_identity"] = identity
        f["one_electron/h1e_mo"].attrs["orbital_identity"] = identity
        if rdms:
            d = r.create_group("rdms")
            d.attrs.update(convention="block2.spin_free.normal_ordered.v1", root=root,
                           orbital_identity=identity)
            d["active_orbital_ids"] = orbital_ids[ncore:ncore + ncas]
            for order, rdm in enumerate(rdms, 1):
                d[f"E{order}"] = rdm
        r.attrs.modify("complete", 1)


def export_rhf(path, mf, *, frozen=0, provenance=None):
    """Closed-shell converged canonical RHF; includes frozen orbitals in C."""
    from pyscf import scf
    if not isinstance(mf, scf.hf.RHF) or isinstance(mf, scf.rohf.ROHF) or not mf.converged:
        raise ValueError("export_rhf requires a converged closed-shell RHF calculation")
    if mf.mol.nelec[0] != mf.mol.nelec[1]:
        raise ValueError("export_rhf requires a closed shell")
    frozen = operator.index(frozen)
    _write(path, mf, kind="rhf", energies=mf.mo_energy, occupations=mf.mo_occ,
           ncore=mf.mol.nelec[0], ncas=0, frozen=frozen, root=0, root_count=1,
           reference_energy=mf.e_tot, provenance=provenance)
    return copy.copy(mf)


def export_casscf(path, mc, *, frozen=0, root=0, provenance=None):
    """Canonicalize one selected root; export its final E1--E4 and total energy."""
    from pyscf import mcscf
    from pyblock2.icmr import eri_helper, scnevpt2
    frozen, root = operator.index(frozen), operator.index(root)
    if not isinstance(mc, mcscf.mc1step.CASSCF) or not mc.converged:
        raise ValueError("export_casscf requires a converged spatial CASSCF calculation")
    multi = isinstance(mc.ci, (list, tuple))
    count = len(mc.ci) if multi else 1
    if not 0 <= root < count:
        raise ValueError("invalid CASSCF root")
    ci = mc.ci[root] if multi else mc.ci
    if isinstance(mc.fcisolver, mcscf.addons.StateAverageFCISolver):
        dm1 = mc.fcisolver.states_make_rdm1(mc.ci, mc.ncas, mc.nelecas)[root]
    else:
        dm1 = mc.fcisolver.make_rdm1(ci, mc.ncas, mc.nelecas)
    final = copy.copy(mc)
    final.mo_coeff, final.ci, final.mo_energy = mc.canonicalize(
        mc.mo_coeff, ci=ci, cas_natorb=False, casdm1=dm1, verbose=0)
    rdms = eri_helper.init_pdms(final, scnevpt2.pdm_eqs)
    if any(rdm is None for rdm in rdms):
        raise ValueError("native NEVPT2 requires E1--E4; DMRG response is unsupported")
    nmo = final.mo_coeff.shape[1]
    occupations = np.zeros(nmo)
    occupations[:mc.ncore] = 2
    occupations[mc.ncore:mc.ncore + mc.ncas] = np.diag(rdms[0])
    energies = mc.e_states if hasattr(mc, "e_states") else mc.e_tot
    energy = np.asarray(energies).reshape(-1)[root]
    final.e_tot = float(energy)
    _write(path, final, kind="casscf", energies=final.mo_energy, occupations=occupations,
           ncore=mc.ncore, ncas=mc.ncas, frozen=frozen, root=root, root_count=count,
           reference_energy=energy, rdms=rdms, provenance=provenance)
    return final
