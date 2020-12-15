#!/usr/bin/env python
# coding: utf-8
"""
unit tests for reading different file formats in DATA/<program>/
"""
import unittest

import numpy as np
import numpy.linalg as la
import logging

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[testing] %(message)s", level=logging.ERROR)

# # Local Imports
from semiclassical.readers import FormattedCheckpointFile
from semiclassical import units

class TestFormattedCheckpointFile(unittest.TestCase):
    def test_frequencies(self):
        """
        compute frequencies and normal modes for S0 and S1 minimum 
        from force constants in formatted checkpoint
        """
        fchk_files = ["DATA/Gaussian16/coumarin_s0.fchk", "DATA/Gaussian16/coumarin_s1.fchk"]
        for fchk_file in fchk_files:
            logger.info(f"reading {fchk_file}")
            with open(fchk_file) as f:
                fchk = FormattedCheckpointFile(f)
                nmodes = fchk["Number of Normal Modes"]
                # normal mode frequencies in cm-1 from checkpoint file
                frequencies_fchk = fchk["Vib-E2"][:nmodes]
                # Now we compute the normal frequencies by diagonalizing the mass-weighted
                # Hessian. We should get the same numbers.
                masses = fchk.masses()
                pos, energy, grad, hess = fchk.harmonic_approximation()
                # diagonalize mass weighted hessian  H^{mw} = M^{-1/2} d^2V/dxdx M^{-1/2}
                isqM = np.diag(1.0/np.sqrt(masses))
                mwH = np.dot(isqM, np.dot(hess, isqM))
                w2, evec = la.eigh(mwH)
                # 6 frequencies for translation and rotation should be 0
                self.assertTrue(np.isclose(w2[:6], np.zeros(6)).all())
                # internal vibrations
                frequencies = np.sqrt(w2[6:]) * units.hartree_to_wavenumbers
                self.assertTrue(np.isclose(frequencies, frequencies_fchk).all())
        
    def test_nonadiabatic_coupling(self):
        with open("DATA/Gaussian16/coumarin_s1.fchk") as f:
            fchk = FormattedCheckpointFile(f)
        nac = fchk.nonadiabatic_coupling()
        
if __name__ == "__main__":
    unittest.main()
