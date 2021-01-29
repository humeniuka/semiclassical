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


def rate_from_correlation(times, correlation, lineshape):
    """
    compute the rate constant k(E) as the Fourier transform of the correlation function \tilde{k}(t),
    the environment is included by damping the C(t) by the lineshape function f(t)

                           /+oo    i/hbar E t  ~    ~
      k(E) = 1/(2 pi hbar) | dt   e            f(t) k(t)
                           /-oo
    
    Parameters
    ----------
    times           :  real ndarray (nt,)
       equidistant time grid covering [0, t_max]
    correlation  :  complex ndarray (nt,)
       correlation function \tilde{k}(t) on the time grid
    lineshape       :  callable
       lineshape function \tilde{f}(t), is called as `lineshape(time)` for a time array

    Returns
    -------
    energies    :  real ndarray (2*nt-1,)
       energy gap (in Hartree)
    rate        :  real ndarray (2*nt-1,)
       rate constants for k(E) as a function of the energy gap E (in s-1)
    """
    assert times.min() == 0.0, "time grid `times` should start at 0.0"
    assert times.shape == correlation.shape, "arrays `times` and `correlation` should have the same length"
    nt = times.shape[0]
    t_max = times.max()
    # k(t) was calculated on the time interval [0, t_max] with nt equidistant samples
    # For the Fourier transform we need the symmetric interval [-t_max, t_max], the nt-1
    # points with t < 0 are filled in using the symmetry of the correlation function:
    #   k(-t) = k(t)^*

    # [-t_max, +t_max] grid contains 2*nt-1 sampling points.
    times_ = np.linspace(-t_max, t_max, 2*nt-1)
    # energy sample points for \Delta E
    energies = fft.fftfreq(2*nt-1) * (2*nt-1)/(2*t_max) * 2.0*np.pi

    # k(t) for positive and negative times
    correlation_ = np.zeros(2*nt-1, dtype=complex)
    correlation_[0.0 <= times_] = correlation
    correlation_[times_  < 0.0] = correlation[1:].conj()

    # Fourier transform of broadening function is the lineshape
    lineshape_t = lineshape(times_)

    # discrete Fourier transform
    rate = 2*t_max * fft.ifft( fft.ifftshift(lineshape_t * correlation_) )
    
    # convert rate from atomic units to seconds^-1
    rate *= 1.0e15 / units.autime_to_fs
    
    return energies, rate

