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


def internal_conversion_rate(t, kic_t, lineshape, rate_file=""):
    """
    compute the IC rate k_ic(E) as the Fourier transform of $f(t) \tilde{k}_{ic}(t)$

      $k_{ic}(\Delta E) = \frac{1}{2\pi} \int_{-\infty}^{\infty} dt ~ e^{\imath \Delta E t} \tilde{f}(t) \tilde{k}_{ic}(t)$

    
    Parameters
    ----------
    t           :  real ndarray (nt,)
       equidistant time grid covering [0, t_max]
    kic_t       :  complex ndarray (nt,)
       correlation function kic(t) on the time grid
    lineshape   :  callable
       lineshape function f(t), is called as `lineshape(time)` for a time array

    Optional
    --------
    rate_file   :  str
       table with kic(E) vs. E is saved to this file, unless the path is empty

    Returns
    -------
    kic_e       :  callable
       interpolated Fourier transform for evaluating kic(E) (in s-1)
    """
    assert t.min() == 0.0, "time grid `t` should start at 0.0"
    assert t.shape == kic_t.shape, "arrays `t` and `kic_t` should have the same length"
    nt = t.shape[0]
    t_max = t.max()
    # k_ic(t) was calculated on the time interval [0, t_max] with nt equidistant samples
    # For the Fourier transform we need the symmetric interval [-t_max, t_max], the nt-1
    # points with t < 0 are filled in using the symmetry of the correlation function:
    #   k_ic(-t) = k_ic(t)^*

    # [-t_max, +t_max] grid contains 2*nt-1 sampling points.
    times_ = np.linspace(-t_max, t_max, 2*nt-1)
    # energy sample points for \Delta E
    energies = fft.fftfreq(2*nt-1) * (2*nt-1)/(2*t_max) * 2.0*np.pi

    # k_ic(t) for positive and negative times
    kic_t_ = np.zeros(2*nt-1, dtype=complex)
    kic_t_[0.0 <= times_] = kic_t
    kic_t_[times_  < 0.0] = kic_t[1:].conj()

    # Fourier transform of broadening function is the lineshape
    lineshape_t = lineshape(times_)

    # discrete Fourier transform
    kic_e = 2*t_max * fft.ifft( fft.ifftshift(lineshape_t * kic_t_) )
    
    # convert IC rate from atomic units to seconds^-1
    kic_e *= 1.0e15 / units.autime_to_fs

    if rate_file != "":
        # save IC rate vs. energy curves
        data = np.vstack((energies * units.hartree_to_ev, kic_e.real)).transpose()
        with open(rate_file, "w") as f:
            f.write("# Energy / eV             IC rate / s^-1\n")
            np.savetxt(f, data)
        logger.info("wrote table with kIC(E) to '%s'" % rate_file)
    
    # Since k_ic(-t) = k_ic(t)^*, the Fourier transform k_ic(E) has to be real.
    kic_e = interpolate.interp1d(energies, kic_e.real, fill_value='extrapolate')
    return kic_e
