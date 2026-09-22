"""Explicit, budgeted HDF5 loading into the installed reference helper APIs."""
import h5py
import numpy as np


def load_helper(path, memory_bytes):
    with h5py.File(path, "r", rdcc_nbytes=0) as f:
        schema = f.attrs["schema"]
        if isinstance(schema, bytes): schema = schema.decode()
        if schema != "ao2mo.helper.v1" or f.attrs["complete"] != 1:
            raise ValueError("not a completed ao2mo helper file")
        p = f["partition"].attrs
        nc, na, n = (int(p[key]) for key in ("ncore", "ncas", "nmo"))
        if min(nc, na, n) < 0 or nc + na > n:
            raise ValueError("invalid helper partition")
        unit = p["counting_unit"]
        if isinstance(unit, bytes): unit = unit.decode()
        if unit not in ("spinor", "spatial_orbital"):
            raise ValueError("unknown orbital counting unit")
        physical = "blocks/phys" in f
        if physical and unit != "spinor":
            raise ValueError("physical ten-block helper requires spinor counting")
        if physical:
            paths = [f"blocks/phys/{key}" for key in f["blocks/phys"]]
            paths += [f"one_electron/h1eff_blocks/{key}" for key in ("AA", "AI", "EI", "EA")]
        else:
            paths = [f"blocks/chem/{key}" for key in f["blocks/chem"]]
            paths += ["one_electron/h1e", "one_electron/h1eff"]
        sizes = dict(c=nc, a=na, v=n-nc-na, p=n, I=nc, A=na, E=n-nc-na, P=n)
        kind, width = ("c", 16) if unit == "spinor" else ("f", 8)
        for name in paths:
            dataset = f[name]
            key = name.rsplit("/", 1)[1]
            shape = (n, n) if key in ("h1e", "h1eff") else tuple(sizes.get(x, -1) for x in key)
            rank = 4 if name.startswith("blocks/") else 2
            if len(shape) != rank or dataset.shape != shape:
                raise ValueError(f"wrong helper block shape: {name}")
            if dataset.dtype.kind != kind or dataset.dtype.itemsize != width:
                raise ValueError(f"wrong helper block dtype: {name}")
        # Payload plus a documented metadata/cache allowance. Loading happens
        # only after the whole selected helper fits; these are owned arrays.
        required = sum(f[path].size * f[path].dtype.itemsize for path in paths) + (4 << 20)
        # The actual dense spinor initializer checks Hermiticity using NumPy
        # allclose/conjugation temporaries. Account for those before loading.
        if unit == "spinor" and not physical:
            required += 3 * n * n * 16
        if required > memory_bytes:
            raise MemoryError(f"helper load requires at least {required} bytes")
        arrays = {path: f[path][:] for path in paths}
        for name, array in arrays.items():
            flat = array.reshape(-1)
            for start in range(0, flat.size, 65536):
                if not np.isfinite(flat[start:start+65536]).all():
                    raise ValueError(f"non-finite helper block: {name}")
        diagnostics = {"source": str(path), "mode": f.attrs["mode"],
                       "constants": {key: f["constants/" + key][0, 0] for key in f["constants"]},
                       "retained_indices": f["partition/retained_indices"][:],
                       "frozen_indices": f["partition/frozen_indices"][:]}
        if not all(np.isfinite(x) for x in diagnostics["constants"].values()):
            raise ValueError("non-finite helper constant")
        if physical:
            from socutils.mrpt.nevpt2_utils import _WickERIBlocks
            return _WickERIBlocks(nc, na, n - nc - na,
                                 {key: arrays[f"one_electron/h1eff_blocks/{key}"] for key in ("AA", "AI", "EI", "EA")},
                                 {key: arrays[f"blocks/phys/{key}"] for key in f["blocks/phys"]}, diagnostics)
        h1e, h1eff = (arrays["one_electron/" + key] for key in ("h1e", "h1eff"))
        if unit == "spinor":
            from socutils.mrpt.spinor_helper import init_eris
            result = init_eris(h1e, arrays["blocks/chem/pppp"], nc, na, h1eff=h1eff)
            result.symmetry_diagnostics = diagnostics
            return result
        from pyblock2.icmr.eri_helper import _ChemistsERIs
        result = _ChemistsERIs()
        result.ncore, result.ncas, result.nocc = nc, na, nc + na
        result.nmo, result.nvirt = n, n - nc - na
        result.h1e, result.h1eff = h1e, h1eff
        result.known = list(f["blocks/chem"])
        result.symmetry_diagnostics = diagnostics
        for key in result.known: setattr(result, key, arrays["blocks/chem/" + key])
        spaces = dict(c=slice(nc), a=slice(nc,nc+na), v=slice(nc+na,n))
        for left in spaces:
            for right in spaces:
                sl = (spaces[left], spaces[right])
                setattr(result, "h" + left + right, h1e[sl])
                setattr(result, "heff" + left + right, h1eff[sl])
        return result
