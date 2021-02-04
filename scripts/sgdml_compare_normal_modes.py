#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Hessians from two chemical models (QM and sGDML) are diagonalized
and the frequencies and normal modes are compared graphically.
"""
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('legend', fontsize=12)
matplotlib.rc('axes', labelsize=12)
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as la
import torch

# # Local Imports
from semiclassical import units
from semiclassical.readers import FormattedCheckpointFile
from semiclassical.gdml_predictor import GDMLPredict

def vibrational_analysis(hess, masses, zero_threshold=1.0e-9):
    """
    compute vibrational frequencies and modes from Hessian

    Parameters:
    -----------
    hess:  ndarray (shape (3*nat,3*nat))
       Hessian in cartesian coordinates
    masses: ndarray (shape (3*nat))
       masses for each cartesian coordinate (in atomic units)
    zero_threshold: float
       modes with frequencies^2 below that threshold are treated as zero

    Returns:
    --------
    freq:  ndarray (shape (3*nat-6)))
      vibrational frequencies in atomic units
    modes: ndarray (shape (3*nat, 3*nat-6)))
      displacement vectors in mass-weighted coordinates
    """
    # convert Hessian to mass-weighted coordinates
    hess_mwc = hess / np.sqrt(np.outer(masses, masses))
    # mass weighted coordinates are now qi = sqrt(mi) dxi
    # compute eigen values of hess_mwc
    omega2,modes = la.eigh(hess_mwc)
    # modes that are zero within numerical accuracy
    zero_modes = np.where((omega2 < zero_threshold))[0]
    vib_modes = np.where((omega2 >= zero_threshold))[0]

    # number of zeros modes, #(zero-modes)
    Nzero = len(zero_modes)

    freqs = np.sqrt(omega2*(1.0+0.0j))
    # period T = 2*pi*hbar/E
    periods = 2.0*np.pi/freqs
    print("Frequencies")
    print("===========")
    print("- Zero modes (should be close to zero)")
    for fr in freqs[zero_modes]:
        print( "   %5.7f Hartree     %15.7f cm-1   " % (fr, fr*units.hartree_to_wavenumbers))
    print("- Vibrations")
    for fr,T in zip(freqs[vib_modes].real, periods[vib_modes].real):
        print("   %5.7f Hartree      %15.7f cm-1      %15.7f fs" % (fr, fr*units.hartree_to_wavenumbers, T*units.autime_to_fs))
    print("")
    en_zp = np.sum(freqs[6:].real)/2.0
    print("zero-point energy:   %5.7f Hartree      %5.7f cm-1" % (en_zp, en_zp*units.hartree_to_wavenumbers))
    # remove 6 translational and rotational modes
    return freqs[6:].real, modes[:,6:].real

def compare_frequencies(freqs_qm, freqs_sgdml):
    mode_nums = range(1, len(freqs_qm)+1)
    plt.bar(mode_nums, freqs_sgdml*units.hartree_to_wavenumbers, label="sGDML")
    plt.bar(mode_nums, freqs_qm*units.hartree_to_wavenumbers, label="QM", alpha=0.8)

    plt.ylabel("Frequency / cm$^{-1}$")
    plt.xlabel("Normal mode")
    
    plt.legend()
    plt.savefig("frequencies_sgdml_qm.svg")

    plt.show()

def compare_modes(modes_qm, modes_sgdml):
    mode_nums = range(1, modes_qm.shape[1]+1)
    # overlap between normal modes
    overlap = np.dot(modes_qm.T, modes_sgdml)
    #
    plt.bar(mode_nums, abs(np.diag(overlap)), label="mode overlap")
    
    plt.ylabel("Overlap")
    plt.xlabel("Normal mode")

    plt.legend()
    plt.savefig("modes_sgdml_qm.svg")

    plt.show()
    
if __name__ == "__main__":
    import sys
    import os.path

    usage = """
 Usage: {script}  freq.fchk  model.npz

      diagonalize mass-weighted Hessians of QM and sGDML model 
      and compare frequencies and normal modes.

    Input Files:
      freq.fchk               :   fchk-file from frequency calculation
      model.npz               :   sGDML model

    """.format(script=os.path.basename(sys.argv[0]))
    args = sys.argv[1:]
    if len(args) < 2:
        print(usage)
        exit(-1)

    fchk_file = args[0]
    model_file = args[1]

    # QM frequencies
    with open(fchk_file) as f:
        fchk = FormattedCheckpointFile(f)        

    pos, _, _, hessian_qm = fchk.harmonic_approximation()
    masses = fchk.masses()

    # sGDML frequencies
    model = np.load(model_file, allow_pickle=True)
    gdml = GDMLPredict(model)

    _, _, hess = gdml.forward( torch.from_numpy(pos).unsqueeze(0) )
    print(hess.shape)
    hessian_sgdml = hess[0,:,:]
    
    freqs_qm, modes_qm = vibrational_analysis(hessian_qm, masses)
    freqs_sgdml, modes_sgdml = vibrational_analysis(hessian_sgdml, masses)

    compare_frequencies(freqs_qm, freqs_sgdml)
    compare_modes(modes_qm, modes_sgdml)
    
