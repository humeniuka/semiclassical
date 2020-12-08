# coding: utf-8
"""
Broadening Functions

Broadening functions are the Fourier transforms of the corresponding lineshape functions.
The functions defined in this module are factories that create lineshape functions.

Example:

    # generate Gaussian-type lineshape function of certain width
    >>> sigma = 1.0
    >>> lineshape = gaussian(sigma)
    # Now evaluate line shape function g(t)
    >>> time = np.linspace(0.0, 100.0, 1000)
    >>> lineshape(time)
"""

__all__ = ['gaussian', 'lorentzian', 'voigtian']

# # Imports
import numpy as np



def gaussian(sigma):
    """ 
    The Fourier transform of a Gaussian

       G(w) = 1/(sqrt(2pi) sigma) * exp(- 1/2 (w/sigma)^2 )

    is again a Gaussian

       g(t) = 1/(2pi) exp(-1/2 sigma^2 * t^2)

    Parameters
    ----------
    sigma : float
      standard deviation of Gaussian in energy domain

    Returns
    -------
    lineshape  :  function
      lineshape(t) evaluates the g(t) on a time grid
    """
    def lineshape(t):
        """
        Gaussian lineshape function

        Parameters
        ----------
        t     : array (nt,)
          time grid

        Returns
        -------
        g     : array (nt,)
          broadening function on time time grid, g(t)
        """
        g = np.exp(-0.5 * sigma**2 * t**2)
        g /= 2.0*np.pi
        return g
    return lineshape

def lorentzian(gamma):
    """
    The Fourier transform of a Lorentzian

                   gamma
        L(w) = -------------------
               pi (gamma^2 + w^2)

    is
                       
        l(t) = 1/(2pi) ( exp(gamma t) Heaviside(-t) + exp(-gamma t) Heaviside[t] )

    Parameters
    ----------
    gamma : float
      width parameter of Lorentzian line shape function in energy domain

    Returns
    -------
    lineshape  :  function
      lineshape(t) evaluates the l(t) on a time grid
    """
    def lineshape(t):
        """
        Lorentzian lineshape function

        Parameters
        ----------
        t     : array (nt,)
          time grid

        Returns
        -------
        l     : array (nt,)
          broadening function on time time grid, l(t)
        """
        l = 0.0*t
        l[t > 0]  = np.exp(-gamma*t[t > 0])
        l[t < 0] += np.exp(+gamma*t[t < 0])
        l /= 2.0*np.pi
        return l
    return lineshape

def voigtian(sigma, gamma):
    """
    The Voigt lineshape function is a convolution of a Gaussian with a Lorentzian.
    By the convolution theorem, 

          Fourier[conv(f,g)] = Fourier[f]*Fourier[g]
    
    its Fourier transform is the product of the Gaussian and Lorentzian Fourier transforms.

         v(t) = g(t) * l(t)

    Parameters
    ----------
    sigma : float
      standard deviation of Gaussian in energy domain
    gamma : float
      width parameter of Lorentzian line shape function in energy domain

    Returns
    -------
    lineshape  :  function
      lineshape(t) evaluates the v(t) on a time grid
    """
    def lineshape(t):
        """
        Voigtian lineshape function

        Parameters
        ----------
        t     : array (nt,)
          time grid

        Returns
        -------
        v     : array (nt,)
          broadening function on time time grid, v(t)
        """
        v = gaussian(t, sigma) * lorentzian(t, gamma)
        return v
    return lineshape

