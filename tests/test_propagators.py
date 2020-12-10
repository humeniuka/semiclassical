#!/usr/bin/env python
# coding: utf-8
"""
unit tests for semiclassical propagators
"""
import unittest

import numpy as np
from numpy import fft
import scipy.linalg as sla
import logging

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[testing] %(message)s", level=logging.ERROR)

import torch
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    logger.info("CUDA available")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# # Local Imports
from semiclassical.propagators import _sym_sqrtm
from semiclassical.propagators import HermanKlukPropagator, WaltonManolopoulosPropagator
from semiclassical.potentials import NonHarmonicPotential, MorsePotential
from semiclassical.propagators import hbar
from semiclassical.propagators import CoherentStatesOverlap

# make random numbers reproducible
torch.manual_seed(0)

class TestLinearAlgebra(unittest.TestCase):
    def test_sym_sqrtm(self):
        """ tests the implementation of sqrtm(A) based on the eigenvalue decomposition"""        
        # create random symmetric n x n  matrix
        n = 5
        A = 5.0 * 2.0*(torch.rand(n,n) - 0.5)
        A = A + A.T

        # reference implementation of scipy
        sqA_scipy = sla.sqrtm(A.numpy())
        # my own implementation using pure torch functions
        sqA = _sym_sqrtm(A).numpy()
        
        self.assertTrue(np.isclose(sqA, sqA_scipy).all())

class TestCoherentStates(unittest.TestCase):
    """
    check overlap integrals between multidimension coherent states
    """
    def test_normalization(self):
        n = 5
        # draw random numbers for positive definite, symmetric n x n matrix of width parameters
        Gi = 5.0 * 2.0*(torch.rand(n,n) - 0.5)
        # symmetrize
        Gi = 0.5*(Gi + Gi.T)
        # random numbers for position and momentum
        qi,pi = torch.rand(n,1), torch.rand(n,1)
        # check <qi,pi,Gi|qi,pi,Gi> = 1        
        cso = CoherentStatesOverlap(Gi,Gi)
        olap = cso(qi,pi, qi,pi)
        self.assertEqual(olap.squeeze().item(), 1.0)
        
class TestSemiclassicalPropagators1D(unittest.TestCase):
    """
    run dynamics on anharmonic 1D potential described in Herman & Kluk (1986)
    and compare with exact QM dynamics
    """
    def setUp(self):

        # # Grids for Time Propagation
        
        # time grid
        nt = 4000 // 40
        # I believe in the HK paper time is measured in units of oscilla
        tau_max = 12.0 / 40
        # frequency of oscillator
        omega = 1.0
        t_max = tau_max * 2.0*np.pi/omega
        times = np.linspace(0.0, t_max, nt)
        dt = times[1]-times[0]
        dtau = dt * omega/(2.0*np.pi)
        
        # spatial grid
        nx = 10000
        x_ = np.linspace(-10.0, 40.0, nx)
        dx = x_[1]-x_[0]
        # reshape spatial grid as (1,nx)
        x = torch.tensor(x_).unsqueeze(0)

        # # Propagation with Split Operator Method on a Grid
        mass = 1.0

        eps = 0.975
        b = (12.0)**(-0.5)
        # potential, see eqn. (7) in HK paper
        v = eps/(2*b**2) * (1.0 - np.exp(-b*x_))**2 + (1.0-eps)*0.5*omega*x_**2
        
        # I believe in the paper time is measured in units of oscillations, tau=t/T
        tau_max = 12.0
        dtau = dt * omega/(2.0*np.pi)
        logger.info(f"time step dt= {dt}  dtau= {dtau} T")
        
        # number of propagation steps per time step
        m = 20

        k = 2.0*np.pi*fft.fftfreq(nx,d=dx)
        # exponential of kinetic energy operator in momentum space
        # -1/(2*m) d^2/dx^2 exp(i*k*x) -->  k^2/(2*m) exp(i*k*x)
        expT = np.exp(-(1.0j/hbar) * (hbar*k)**2/(2.0*mass) * (dt/m) )
        # exponential of potential energy in coordinate space
        expV = np.exp(-(1.0j/hbar) * v * (dt/m))
        
        # initial Gaussian wavefunction, see eqn. (8)
        # I think there is a mistake in Kluk & Herman's paper, the exponent of the Gaussian
        # should be alpha=1/2 so that the wavefunction is the HO ground state wavefunction 
        # for the potential V(x) = 1/2 x^2
        alpha = 0.5*omega #1.0
        x0 = 7.3
        p0 = 0.0
        psi0 = (2*alpha/np.pi)**0.25 * np.exp(-alpha*(x_-x0)**2 + 1.0j*p0*(x_-x0)/hbar)
        
        # propagate wavefunction for t=0.0,...,12.0 using the split operator propagator
        psi = psi0
        
        # autocorrelation function, overlap between initial wavefunction and wavefunction at time t
        # Cauto(t) = <phi(0)|phi(t)>
        autocorrelation_qm = np.zeros((nt,), dtype=complex)

        logger.info("running split operator propagator (exact QM):")
        logger.info("... propagate wavepacket psi(0) in time")
        for t in range(0, nt):
            autocorrelation_qm[t] = np.sum(psi0.conjugate() * psi *dx)
            for i in range(0, m):
                # split operator step
                # |psi(t+dt)> = U(dt) |psi(t)> 
                #             = exp(-i/hbar (V+T) dt) |psi(t)> 
                #             ~ exp(-i/hbar V dt) exp(-i/hbar T dt) |psi(t)>
                psi = expV * fft.ifft( expT * fft.fft(psi) )
                #  
            if t % (nt//10+1) == 0:
                # kinetic energy T|psi(t)>
                Tpsi = fft.ifft(k**2/(2*mass) * fft.fft(psi))
                # potential energy V|psi(t)>
                Vpsi = v*psi
                # energy expectation value <psi(t)|T+V|psi(t)>
                energy = np.sum(psi.conjugate() * (Tpsi+Vpsi) * dx)
                logger.info("time= %12.5f   |psi(t)|= %e  <E>= %e" % (times[t], np.sqrt(np.sum(abs(psi)**2*dx)), energy.real))

        # # IC Correlation function

        # # Propagate Interaction-Prepared Wavefunction
        # To compute the exact IC correlation function we propagator the interaction-prepared wavefunction
        # $\vert \psi(0) \rangle = \hat{V}^{\dagger} \vert \phi(0) \rangle$
        # in time

        # This is the vibrational ground state <x|phi(0)> on the excited state potential.
        phi0 = (2*alpha/np.pi)**0.25 * np.exp(-alpha*(x-x0)**2 + 1.0j*p0*(x-x0)/hbar)

        # The non-adiabatic coupling vector is assumed to be constant
        def nac(x):
            return 1.0
        # The interaction operator is 
        #   Veg = hbar^2/m * nac * d/dx
        # The interaction prepared wavefunction becomes
        #   |psi(0)> = V^+|phi(0)>
        psi0 = hbar**2/mass * nac(x) * fft.ifft((1.0j*k) * fft.fft(phi0))

        # propagate wavefunction for t=0.0,...,12.0 using the split operator propagator
        psi = psi0

        # save the correlation function corr(t) = <psi(0)|psi(t)> = <phi(0)|V e^{-i*t*H} V^+|phi(0)>
        ic_correlation_qm = np.zeros(nt, dtype=complex)

        # zero-point energy of the excited state potential
        en0 = hbar/2.0 * omega

        logger.info("... propagate interaction-prepared wavepacket V^+\phi(0) in time")
        for t in range(0, nt):
            ic_correlation_qm[t] = 2*np.pi/hbar * np.exp(1j*times[t]*en0) * np.sum(psi0.conjugate() * psi * dx)
            for i in range(0, m):
                # split operator step
                # |psi(t+dt)> = U(dt) |psi(t)> 
                #             = exp(-i/hbar (V+T) dt) |psi(t)> 
                #             ~ exp(-i/hbar V dt) exp(-i/hbar T dt) |psi(t)>
                psi = expV * fft.ifft( expT * fft.fft(psi) )
            if t % (nt//10+1) == 0:
                # kinetic energy T|psi(t)>
                Tpsi = fft.ifft(k**2/(2*mass) * fft.fft(psi))
                # potential energy V|psi(t)>
                Vpsi = v*psi
                # energy expectation value <psi(t)|T+V|psi(t)>
                energy = np.sum(psi.conjugate() * (Tpsi+Vpsi) * dx)
                logger.info("time= %12.5f   |psi(t)|= %e  <E>= %e" % (times[t], np.sqrt(np.sum(abs(psi)**2*dx)), energy.real))


        #
        self.times = times
        self.nt = nt
        self.dt = dt
        
        self.omega = omega
                
        # save autocorrelation functions and IC correlation functions
        # from exact QM propagation on a grid
        self.autocorrelation_qm = autocorrelation_qm
        self.ic_correlation_qm = ic_correlation_qm

        # potential and initial conditions for semiclassical propagators
        self.potential = NonHarmonicPotential()

        # center of initial wavepacket
        self.q0 = torch.tensor([7.3])
        # initial momentum
        self.p0 = torch.tensor([0.0])
        
        self.Gamma_i = torch.tensor([[2*2.5]])
        self.Gamma_t = self.Gamma_i

        self.Gamma_0 = torch.tensor([[self.omega]])        

        # zero-point energy of the excited state potential
        self.en0 = hbar/2.0 * omega
        
    def test_HermanKlukPropagator(self):
        # create HK propagator
        propagator = HermanKlukPropagator(self.Gamma_i, self.Gamma_t, device=device)
        propagator.initial_conditions(self.q0, self.p0, self.Gamma_0, ntraj=50000)
        
        # save autocorrelation functions for each time step
        autocorrelation = np.zeros((self.nt,), dtype=complex)
        ic_correlation = np.zeros((self.nt,), dtype=complex)

        logger.info("running Herman-Kluk propagator:")
        for t in range(0, self.nt):
            autocorrelation[t] = propagator.autocorrelation()
            ic_correlation[t] = propagator.ic_correlation(self.potential, energy0_es=self.en0)
            if t % (self.nt//10+1) == 0:
                logger.info("time= %8.5f   time/T= %5.3f" % (self.times[t], self.times[t]*self.omega/(2*np.pi)))
            propagator.step(self.potential, self.dt)

        # compare semiclassical correlation functions with QM results
        self.assertTrue(np.isclose(autocorrelation, self.autocorrelation_qm, rtol=0.05).all())
        self.assertTrue(np.isclose(ic_correlation, self.ic_correlation_qm, rtol=0.1).all())

        # check norm of wavefunction is ~ 1 at last time step
        norm = propagator.norm()
        self.assertAlmostEqual(norm, 1, delta=0.05)
        
    def test_WaltonManolopoulosPropagator(self):
        # create WM propagator
        beta = 100.0
        propagator = WaltonManolopoulosPropagator(self.Gamma_i, self.Gamma_t, beta, device=device)
        propagator.initial_conditions(self.q0, self.p0, self.Gamma_0, ntraj=50000)
        
        # save autocorrelation functions for each time step
        autocorrelation = np.zeros((self.nt,), dtype=complex)
        ic_correlation = np.zeros((self.nt,), dtype=complex)

        logger.info("running Walton-Manolopoulos propagator:")
        for t in range(0, self.nt):
            autocorrelation[t] = propagator.autocorrelation()
            ic_correlation[t] = propagator.ic_correlation(self.potential, energy0_es=self.en0)
            if t % (self.nt//10+1) == 0:
                logger.info("time= %8.5f   time/T= %5.3f" % (self.times[t], self.times[t]*self.omega/(2*np.pi)))
            propagator.step(self.potential, self.dt)

        # compare semiclassical correlation functions with QM results
        self.assertTrue(np.isclose(autocorrelation, self.autocorrelation_qm, rtol=0.05).all())
        self.assertTrue(np.isclose(ic_correlation, self.ic_correlation_qm, rtol=0.1).all())

        # check norm of wavefunction is ~ 1 at last time step
        norm = propagator.norm()
        self.assertAlmostEqual(norm, 1, delta=0.05)
    
if __name__ == "__main__":
    unittest.main()

    
