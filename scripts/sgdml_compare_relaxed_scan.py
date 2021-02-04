#!/usr/bin/env python
"""
compute energies along the relaxed scan and plot comparison with DFT energies.
The geometries are taken from the formatted checkpoint file.
"""
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('axes', labelsize=16)
import matplotlib.pyplot as plt

import numpy as np
import torch
import sys
import os.path

# # Local Imports
from semiclassical.readers import FormattedCheckpointFile
from semiclassical.gdml_predictor import GDMLPredict

if len(sys.argv) < 3:
    usage = """
 Usage: {script}  scan.fchk  model.npz

      compute energies with sGDML for geometries in relaxed scan and
      compare with DFT energies.

    Input Files:
      scan.fchk   -  formatted checkpoint file from relaxed scan with ModRedundant
      model.npz   -  sGDML model

    """.format(script=os.path.basename(sys.argv[0]))
    print(usage)
    exit(-1)

fchk_file = sys.argv[1]
model_file = sys.argv[2]

model = np.load(model_file, allow_pickle=True)
gdml = GDMLPredict(model)

with open(fchk_file) as f:
    fchk = FormattedCheckpointFile(f)

nat = fchk['Number of atoms']

# check that order of atoms agrees
assert np.array_equal(model['z'], fchk['Atomic numbers']), "Order of atoms in sGDML model and checkpoint file differs"

geometries = []
energies_dft = []
for key in fchk.keys():
    if ("Opt point" in key) and ("Geometries" in key):
        geometries.append( fchk[key][-3*nat:] )
    if ("Opt point" in key) and ("Results" in key):
        energies_dft.append( fchk[key][-2] )

geometries = torch.from_numpy(np.vstack(geometries))
energies_dft = np.array(energies_dft)

energies_gdml = gdml.forward(geometries, order=0)

plt.plot(energies_dft, lw=2, label="DFT")
plt.plot(energies_gdml, lw=2, label="sGDML")
plt.xlabel("Scan point")
plt.ylabel("Energy / Hartree")

plt.legend()
plt.tight_layout()

name = fchk_file.replace(".fchk", "")
plt.savefig(f"{name}.svg")
plt.savefig(f"{name}.png", dpi=300)
plt.show()

