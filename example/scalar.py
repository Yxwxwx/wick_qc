"""Run in a writable directory, then: build/full/bin/wickqc_integrals transform scalar.h5 eri.h5."""
from pyscf import gto, scf
from export_input import export

mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = scf.RHF(mol).run()
c = mf.mo_coeff
export("scalar.h5", mol, {"C": c},
       partition=dict(ncore=2, ncas=2, nmo=c.shape[1]),
       h1e=c.T @ mf.get_hcore() @ c, operator_label="PySCF RHF hcore")
