# coding: utf-8
"""rate constants by Fourier transform of correlation functions"""

__all__ = ['internal_conversion_rate']

# # Imports
import numpy as np
from numpy import fft
from scipy import interpolate
import logging

from semiclassical import units


# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(module)-12s] %(message)s", level=logging.INFO)


def internal_conversion_rate(times, ic_correlation, lineshape):
    """
    compute the IC rate k_ic(E) as the Fourier transform of $f(t) \tilde{k}_{ic}(t)$

      $k_{ic}(\Delta E) = \frac{1}{2\pi hbar} \int_{-\infty}^{\infty} dt ~ e^{\imath/hbar \Delta E t} \tilde{f}(t) \tilde{k}_{ic}(t)$

    
    Parameters
    ----------
    times           :  real ndarray (nt,)
       equidistant time grid covering [0, t_max]
    ic_correlation  :  complex ndarray (nt,)
       correlation function kic(t) on the time grid
    lineshape       :  callable
       lineshape function f(t), is called as `lineshape(time)` for a time array

    Returns
    -------
    energies    :  real ndarray (2*nt-1,)
       energy gap (in Hartree)
    ic_rate     :  real ndarray (2*nt-1,)
       rate constants for internal conversion k_ic(E) as a function of the energy gap E (in s-1)
    """
    assert times.min() == 0.0, "time grid `times` should start at 0.0"
    assert times.shape == ic_correlation.shape, "arrays `times` and `ic_correlation` should have the same length"
    nt = times.shape[0]
    t_max = times.max()
    # k_ic(t) was calculated on the time interval [0, t_max] with nt equidistant samples
    # For the Fourier transform we need the symmetric interval [-t_max, t_max], the nt-1
    # points with t < 0 are filled in using the symmetry of the correlation function:
    #   k_ic(-t) = k_ic(t)^*

    # [-t_max, +t_max] grid contains 2*nt-1 sampling points.
    times_ = np.linspace(-t_max, t_max, 2*nt-1)
    # energy sample points for \Delta E
    energies = fft.fftfreq(2*nt-1) * (2*nt-1)/(2*t_max) * 2.0*np.pi

    # k_ic(t) for positive and negative times
    ic_correlation_ = np.zeros(2*nt-1, dtype=complex)
    ic_correlation_[0.0 <= times_] = ic_correlation
    ic_correlation_[times_  < 0.0] = ic_correlation[1:].conj()

    # Fourier transform of broadening function is the lineshape
    lineshape_t = lineshape(times_)

    # discrete Fourier transform
    ic_rate = 2*t_max * fft.ifft( fft.ifftshift(lineshape_t * ic_correlation_) )
    
    # convert IC rate from atomic units to seconds^-1
    ic_rate *= 1.0e15 / units.autime_to_fs

    return energies, ic_rate

