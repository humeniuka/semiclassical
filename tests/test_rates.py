#!/usr/bin/env python
# coding: utf-8
import unittest
import numpy as np
import logging

from semiclassical import units
from semiclassical import broadening
from semiclassical import rates

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[testing] %(message)s", level=logging.INFO)

class TestRates(unittest.TestCase):
    def test_fourier_transform(self):
        """
        check that the Fourier transform of the lineshape function g(t) is a probability
        distribution G(w) over energies w, which is normalized to 1, i.e.
              /
              | dw G(w) = 1
              /
        """
        # Gaussian lineshape function
        hwhmG_ev = 0.5
        sigma = hwhmG_ev / np.sqrt(2.0 * np.log(2.0)) / units.hartree_to_ev
        lineshape = broadening.gaussian(sigma)
        # 
        times = np.linspace(0.0, 10.0, 500) / units.autime_to_fs
        # set correlation to 1, so that we effectively only compute the Fourier
        # transform of the lineshape function.
        correlation = np.ones_like(times)

        # compute Fourier transform of lineshape function, G(w)
        w, G = rates.rate_from_correlation(times, correlation, lineshape)

        # convert from seconds^{-1} back to atomic units
        G /= 1.0e15 / units.autime_to_fs

        # integrate G(w)
        dw = w[1]-w[0]
        integ_G = np.sum(G * dw)
        
        # check normalization of G(w)
        logger.info(f"integral of G(w) dw = {integ_G}")
        self.assertAlmostEqual(integ_G, 1.0)
        
        
if __name__ == "__main__":
    unittest.main()

    
