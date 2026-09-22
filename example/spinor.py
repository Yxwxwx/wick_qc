"""Actual local socutils X2CAMF orbitals, with upstream hcore preserved."""
from pathlib import Path
import subprocess
from pyscf import gto
from socutils.scf import spinor_hf
from export_input import export, spinor_components

mol = gto.M(atom="H 0 0 0; F 0 0 .917", basis="sto-3g")
mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
mf.kernel()
c = mf.mo_coeff
source = Path(spinor_hf.__file__).resolve().parent
try:
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    provenance = {"socutils_commit": revision, "socutils_tracked_dirty": str(bool(dirty))}
except (OSError, subprocess.CalledProcessError):
    provenance = {"socutils_commit": "unavailable (installed without accessible Git metadata)"}
export("spinor.h5", mol, {"C": spinor_components(mol, c)},
       partition=dict(ncore=2, ncas=3, nmo=c.shape[1]),
       h1e=c.conj().T @ mf.get_hcore() @ c,
       operator_label="socutils X2CAMF hcore; Gaunt=false Breit=false", provenance=provenance)
