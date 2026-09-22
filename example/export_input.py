"""Export existing PySCF/socutils orbitals; no AO2MO or SCF in this module."""

from pathlib import Path
import hashlib
import sys

import h5py
import numpy as np


def complex_dataset(group, name, values):
    # Explicit HDF5 1.x r/i compound, even with h5py linked to HDF5 2.x.
    values = np.ascontiguousarray(values, dtype="<c16")
    group.create_dataset(name, data=values.view([("r", "<f8"), ("i", "<f8")]))


def spinor_components(mol, coefficients):
    """Coefficients must already refer to mol's uncontracted j-spinor AO set.

    If upstream used so_contr, first apply that documented mapping to C and
    pass its corresponding mol. Shape alone cannot establish representation.
    """
    if getattr(mol, "so_contr", None) is not None:
        raise ValueError("resolve so_contr upstream and supply its matching AO mol")
    c = np.asarray(coefficients)
    if c.ndim != 2 or c.shape[0] != mol.nao_2c():
        raise ValueError("expected coefficients in this mol's j-spinor AO basis")
    ua, ub = mol.sph2spinor_coeff()
    return ua @ c, ub @ c


def export(path, mol, families, *, requests=None, spaces=None, partition=None,
           h1e=None, dressing_state="bare", operator_label=None,
           upstream_frozen=(), frozen_electronic=0.0, orbital_ids=None, provenance=None):
    """families maps IDs to real matrices or (C_alpha,C_beta) tuples.

    Each request is {dataset, coefficients: four IDs, indices: four index
    lists, ordering='chemist', layout='dense'}. Omit for full first family.
    """
    if mol.cart:
        raise ValueError("only real spherical AOs are supported")
    if not families:
        raise ValueError("at least one coefficient family is required")
    if dressing_state == "frozen-folded" and orbital_ids is None:
        raise ValueError("folded input requires explicit original orbital_ids")
    spinor = isinstance(next(iter(families.values())), tuple)
    basis_hash = hashlib.sha256()
    for array in (mol._atm, mol._bas, mol._env):
        basis_hash.update(np.ascontiguousarray(array).tobytes())
    with h5py.File(Path(path), "x", libver="earliest") as f:
        f.attrs["schema"] = "ao2mo.input.v1"
        b = f.create_group("basis")
        b.attrs["identity"] = basis_hash.hexdigest()
        b["atm"] = np.asarray(mol._atm, dtype="<i4")
        b["bas"] = np.asarray(mol._bas, dtype="<i4")
        b["env"] = np.asarray(mol._env, dtype="<f8")
        b["ao_loc_sph"] = np.asarray(mol.ao_loc_nr(cart=False), dtype="<i8")
        for key, value in families.items():
            if isinstance(value, tuple) != spinor:
                raise ValueError("mixed real and spinor families are unsupported")
            arrays = value if spinor else (value,)
            arrays = tuple(np.asarray(a) for a in arrays)
            if any(a.ndim != 2 or a.shape[0] != mol.nao_nr() or
                   not np.isfinite(a).all() for a in arrays):
                raise ValueError("invalid coefficient array")
            if spinor and arrays[0].shape != arrays[1].shape:
                raise ValueError("alpha/beta shape mismatch")
            group = f.create_group("coefficients/" + key)
            group.attrs["basis_identity"] = basis_hash.hexdigest()
            identity = hashlib.sha256()
            for array in arrays:
                dtype = "<c16" if spinor else "<f8"
                identity.update(np.asarray(array.shape, dtype="<u8").tobytes())
                identity.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
            group.attrs["identity"] = identity.hexdigest()
            if spinor:
                complex_dataset(group, "alpha", arrays[0])
                complex_dataset(group, "beta", arrays[1])
            else:
                if np.iscomplexobj(arrays[0]):
                    raise ValueError("complex coefficients need both spin components")
                group["real"] = np.asarray(arrays[0], dtype="<f8")
        for name, indices in (spaces or {}).items():
            f[f"spaces/{name}/indices"] = np.asarray(indices, dtype="<i8")
        if partition is not None:
            group = f.create_group("partition")
            for key, value in partition.items():
                group.attrs[key] = value
            group.attrs["counting_unit"] = "spinor" if spinor else "spatial_orbital"
        if h1e is not None:
            if operator_label is None:
                raise ValueError("h1e requires the actual upstream operator label")
            group = f.create_group("one_electron")
            group.attrs["dressing_state"] = dressing_state
            group.attrs["operator_label"] = operator_label
            group.attrs["core_set"] = np.array(upstream_frozen, dtype="<i8")
            group.attrs["frozen_electronic"] = float(frozen_electronic)
            group.attrs["nuclear_repulsion"] = mol.energy_nuc()
            if orbital_ids is not None:
                group["orbital_ids"] = np.asarray(orbital_ids, dtype="<i8")
            if spinor:
                complex_dataset(group, "h1e_mo", h1e)
            else:
                if np.iscomplexobj(h1e):
                    raise ValueError("scalar h1e must be real")
                group["h1e_mo"] = np.asarray(h1e, dtype="<f8")
        for i, request in enumerate(requests or []):
            group = f.create_group(f"requests/{i:04d}")
            group.attrs["dataset"] = request["dataset"]
            group.attrs["ordering"] = request.get("ordering", "chemist")
            group.attrs["layout"] = request.get("layout", "dense")
            for axis, (key, indices) in enumerate(zip(request["coefficients"], request["indices"], strict=True)):
                group.attrs[f"coefficient{axis}"] = key
                group[f"indices{axis}"] = np.asarray(indices, dtype="<i8")
        import pyscf
        group = f.create_group("provenance")
        for name, value in (provenance or {}).items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValueError("source provenance requires string names and values")
            group.attrs[name] = value
        group.attrs["pyscf_version"] = pyscf.__version__
        group.attrs["numpy_version"] = np.__version__
        group.attrs["h5py_version"] = h5py.__version__
        group.attrs["hdf5_version"] = h5py.version.hdf5_version
        group.attrs["python_version"] = sys.version
        group.attrs["exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        group.attrs["representation"] = "real-spherical-alpha-beta" if spinor else "real-spherical"
        group.attrs["two_electron_operator"] = "full_coulomb_int2e_sph"
