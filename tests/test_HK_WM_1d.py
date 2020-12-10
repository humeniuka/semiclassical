#!/usr/bin/env python
# coding: utf-8

# # Imports

import numpy as np
from numpy import fft
import torch

import torch
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    print("CUDA available")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

from semiclassical.propagators import HermanKlukPropagator, WaltonManolopoulosPropagator
from semiclassical.potentials import NonHarmonicPotential, MorsePotential
from semiclassical.propagators import hbar

# # Grids for Time Propagation

# time grid
nt = 4000
# I believe in the HK paper time is measured in units of oscilla
tau_max = 12.0
# frequency of oscillator
omega = 1.0
t_max = tau_max * 2.0*np.pi/omega
times = np.linspace(0.0, t_max, nt)
dt = times[1]-times[0]
dtau = dt * omega/(2.0*np.pi)
print(f"time step dt= {dt}  dtau= {dtau} T")

# spatial grid
nx = 10000
x_ = np.linspace(-10.0, 40.0, nx)
dx = x_[1]-x_[0]
# reshape spatial grid as (1,nx)
x = torch.tensor(x_).unsqueeze(0)



# # Propagate Wavefunction

potential = NonHarmonicPotential()

# center of initial wavepacket
q0 = torch.tensor([7.3])
# initial momentum
p0 = torch.tensor([0.0])

Gamma_i = torch.tensor([[2*2.5]])
Gamma_t = Gamma_i
propagator_hk = HermanKlukPropagator(Gamma_i, Gamma_t)
propagator_wm = WaltonManolopoulosPropagator(Gamma_i, Gamma_t, 100.0)

Gamma_0 = torch.tensor([[omega]])
propagator_hk.initial_conditions(q0, p0, Gamma_0, ntraj=500)
propagator_wm.initial_conditions(q0, p0, Gamma_0, ntraj=500)

# save Herman-Kluk and Walton-Manolopoulos wavefunctions for some time steps
wavefunctions_hk = np.zeros((nx,nt), dtype=complex)
wavefunctions_wm = np.zeros((nx,nt), dtype=complex)

# save autocorrelation function for each time step
autocorrelation_hk = np.zeros((nt,), dtype=complex)
autocorrelation_wm = np.zeros((nt,), dtype=complex)

# correlation function for internal conversion
ic_correlation_hk = np.zeros((nt,), dtype=complex)
ic_correlation_wm = np.zeros((nt,), dtype=complex)

# Evaluating the frozen gaussian wavefunction for all time steps is too time-consuming.
# Therefore we do this only for the time steps `ts`, which we want to plot later.
# tau = t/T
taus_save = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                      0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 
                      0.8, 1.2, 1.6, 2.0,2.4, 4.0, 6.0, 8.0, 10.0, 12.0])
# frequency of oscillator
omega = 1.0
ts_save = [int((tau/omega*2*np.pi)/dt) for tau in taus_save]

# zero-point energy of the excited state potential
en0 = hbar/2.0 * omega

for t in range(0, nt):
    autocorrelation_hk[t] = propagator_hk.autocorrelation()
    autocorrelation_wm[t] = propagator_wm.autocorrelation()
    ic_correlation_hk[t] = propagator_hk.ic_correlation(potential, energy0_es=en0)
    ic_correlation_wm[t] = propagator_wm.ic_correlation(potential, energy0_es=en0)
    if t in ts_save:
        wavefunctions_hk[:,t] = propagator_hk.wavefunction(x)
        wavefunctions_wm[:,t] = propagator_wm.wavefunction(x)
        
        print("time= %8.5f   time/T= %5.3f" % (times[t], times[t]*omega/(2*np.pi)))
        print("    HK: |psi(t)|= %e" % propagator_hk.norm())
        print("    WM: |psi(t)|= %e" % propagator_wm.norm())
    propagator_hk.step(potential, dt)
    propagator_wm.step(potential, dt)


# # Propagation with Split Operator Method on a Grid
mass = 1.0

eps = 0.975
b = (12.0)**(-0.5)
# potential, see eqn. (7) in HK paper
v = eps/(2*b**2) * (1.0 - np.exp(-b*x_))**2 + (1.0-eps)*0.5*omega*x_**2

# I believe in the paper time is measured in units of oscillations, tau=t/T
tau_max = 12.0
dtau = dt * omega/(2.0*np.pi)
print(f"time step dt= {dt}  dtau= {dtau} T")

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

# save wavefunctions for all time steps
wavefunctions_qm = np.zeros((nx,nt), dtype=complex)
# autocorrelation function, overlap between initial wavefunction and wavefunction at time t
# Cauto(t) = <phi(0)|phi(t)>
autocorrelation_qm = np.zeros((nt,), dtype=complex)

for t in range(0, nt):
    wavefunctions_qm[:,t] = psi
    autocorrelation_qm[t] = np.sum(psi0.conjugate() * psi *dx)
    for i in range(0, m):
        # split operator step
        # |psi(t+dt)> = U(dt) |psi(t)> 
        #             = exp(-i/hbar (V+T) dt) |psi(t)> 
        #             ~ exp(-i/hbar V dt) exp(-i/hbar T dt) |psi(t)>
        psi = expV * fft.ifft( expT * fft.fft(psi) )
        #  
    if t % 100 == 0:
        # kinetic energy T|psi(t)>
        Tpsi = fft.ifft(k**2/(2*mass) * fft.fft(psi))
        # potential energy V|psi(t)>
        Vpsi = v*psi
        # energy expectation value <psi(t)|T+V|psi(t)>
        energy = np.sum(psi.conjugate() * (Tpsi+Vpsi) * dx)
        print("time= %12.5f   |psi(t)|= %e  <E>= %e" % (times[t], np.sqrt(np.sum(abs(psi)**2*dx)), energy.real))

#np.savez("wfn_quantum.npz",
#    times=times, x=x, wavefunctions=wavefunctions_qm)


# # Compare QM and Semiclassical Wavefunctions

# In[56]:


import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('axes', labelsize=16)
matplotlib.rc('image', cmap='viridis')

import matplotlib.pyplot as plt

#%matplotlib widget
#get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")


# In[57]:


# Time is measured in units of the period T = 2*pi/omega  with omega=1.0
# so tau=t/T = t*omega/(2*pi)
taus = np.array([0.4, 0.8, 1.2, 1.6, 2.0,2.4])
#taus = np.array([0.0, 6.0, 7.0, 8.0, 9.0, 12.0])

fix,axes = plt.subplots(int(np.ceil(len(taus)/3)),3, figsize=(10,5))
for ij,tau in enumerate(taus):
    time = tau/omega * 2.0*np.pi
    t = int(time/dt)
    jax = ij % 3
    iax = ij // 3
    ax = axes[iax,jax]
    
    ax.set_title(f"t/T={tau:3.2f}")
    ax.set_xlim((-4.0,24.0))
    ax.set_ylim((0.0,18.0))
    # potential V(x)
    ax.plot(x_,v,lw=2)
    # wavefunction  |psi(x;t)|
    ax.plot(x_, abs(wavefunctions_qm[:,t])**2 *15+5, label="QM")
    ax.plot(x_, abs(wavefunctions_hk[:,t])**2 *15+5, ls="-.", label="Herman-Kluk")
    ax.plot(x_, abs(wavefunctions_wm[:,t])**2 *15+5, ls="--", label="Walton-Monolopoulos")
    
    # compute norm from grid representation
    norm_qm = np.sqrt(np.sum(abs(wavefunctions_qm[:,t])**2*dx))
    norm_hk = np.sqrt(np.sum(abs(wavefunctions_hk[:,t])**2*dx))
    norm_wm = np.sqrt(np.sum(abs(wavefunctions_wm[:,t])**2*dx))
    print(f"t/T= {tau:5.2f}   norms |psi|    QM= {norm_qm:e}  HK= {norm_hk:e}  WM= {norm_wm:e}")
    
    if ij == 0:
        ax.legend()
    
plt.tight_layout()
plt.show()


# # Complex Autocorrelation Functions $C_{\text{auto}}(t) = \langle \phi(0) \vert \phi(t) \rangle$

# In[58]:


fig = plt.gcf()
fig.set_size_inches(10.0, 5.0)

plt.xlabel(r"Time $t/T$")
plt.ylabel(r"$C_{auto}(t) = \langle \phi(0) \vert \phi(t) \rangle$")

taus = times * (2*np.pi)/omega
plt.plot(taus, autocorrelation_qm.real, label="Re[$C(t)$] QM")
plt.plot(taus, autocorrelation_qm.imag, label="Im[$C(t)$] QM")

plt.plot(taus, autocorrelation_hk.real, label="Re[$C(t)$] HK", ls="-.")
plt.plot(taus, autocorrelation_hk.imag, label="Im[$C(t)$] HK", ls="-.")

plt.plot(taus, autocorrelation_wm.real, label="Re[$C(t)$] WM", ls="--")
plt.plot(taus, autocorrelation_wm.imag, label="Im[$C(t)$] WM", ls="--")

plt.xlim((0,100))
plt.legend(ncol=3)
plt.tight_layout()
plt.show()


# # Error of Autocorrelation Function

# In[59]:


plt.xlabel(r"Time $t/T$")
plt.ylabel(r"$\vert C^{semi}_{auto}(t) - C^{QM}_{auto}(t) \vert$")

taus = times * (2*np.pi)/omega
error_hk = abs(autocorrelation_hk - autocorrelation_qm)
error_wm = abs(autocorrelation_wm - autocorrelation_qm)
plt.plot(taus, error_hk, label="error |HK-QM|")
plt.plot(taus, error_wm, label="error |WM-QM|")

plt.xlim((0,400))
plt.legend()
plt.tight_layout()
plt.show()


# # IC Correlation function

# # Propagate Interaction-Prepared Wavefunction
# To compute the exact IC correlation function we propagator the interaction-prepared wavefunction
# $\vert \psi(0) \rangle = \hat{V}^{\dagger} \vert \phi(0) \rangle$
# in time

# In[46]:


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

# save interaction-prepared wavefunctions for all time steps
ic_wavefunctions_qm = np.zeros((nx,nt), dtype=complex)
# save the correlation function corr(t) = <psi(0)|psi(t)> = <Theta_es,0|V e^{-i*t*H} V^+|Theta_es,0>
ic_correlation_qm = np.zeros(nt, dtype=complex)

# zero-point energy of the excited state potential
en0 = hbar/2.0 * omega

for t in range(0, nt):
    ic_wavefunctions_qm[:,t] = psi
    ic_correlation_qm[t] = 2*np.pi/hbar * np.exp(1j*times[t]*en0) * np.sum(psi0.conjugate() * psi * dx)
    for i in range(0, m):
        # split operator step
        # |psi(t+dt)> = U(dt) |psi(t)> 
        #             = exp(-i/hbar (V+T) dt) |psi(t)> 
        #             ~ exp(-i/hbar V dt) exp(-i/hbar T dt) |psi(t)>
        psi = expV * fft.ifft( expT * fft.fft(psi) )
    if t % 100 == 0:
        # kinetic energy T|psi(t)>
        Tpsi = fft.ifft(k**2/(2*mass) * fft.fft(psi))
        # potential energy V|psi(t)>
        Vpsi = v*psi
        # energy expectation value <psi(t)|T+V|psi(t)>
        energy = np.sum(psi.conjugate() * (Tpsi+Vpsi) * dx)
        print("time= %12.5f   |psi(t)|= %e  <E>= %e" % (times[t], np.sqrt(np.sum(abs(psi)**2*dx)), energy.real))
        


# ## Compare QM and semiclassical IC correlation functions

# In[60]:


fig = plt.gcf()
fig.set_size_inches(10.0, 5.0)

plt.xlabel(r"Time $t/T$")
plt.ylabel(r"$\tilde{k}_{ic}(t)$")

taus = times * (2*np.pi)/omega
plt.plot(taus, ic_correlation_qm.real, label=r"Re[$\tilde{k}_{ic}(t)$] QM")
plt.plot(taus, ic_correlation_qm.imag, label=r"Im[$\tilde{k}_{ic}(t)$] QM")

plt.plot(taus, ic_correlation_hk.real, label=r"Re[$\tilde{k}_{ic}(t)$] HK", ls="-.")
plt.plot(taus, ic_correlation_hk.imag, label=r"Im[$\tilde{k}_{ic}(t)$] HK", ls="-.")

plt.plot(taus, ic_correlation_wm.real, label=r"Re[$\tilde{k}_{ic}(t)$] WM", ls="--")
plt.plot(taus, ic_correlation_wm.imag, label=r"Im[$\tilde{k}_{ic}(t)$] WM", ls="--")

plt.xlim((300,400))
plt.legend(ncol=3)
plt.tight_layout()
plt.show()


# # Error of IC-Correlation Function

# In[61]:


plt.xlabel(r"Time $t/T$")
plt.ylabel(r"$\vert \tilde{k}^{semi}_{ic}(t) - \tilde{k}^{QM}_{ic}(t) \vert$")

taus = times * (2*np.pi)/omega
error_hk = abs(ic_correlation_hk - ic_correlation_qm)
error_wm = abs(ic_correlation_wm - ic_correlation_qm)
plt.plot(taus, error_hk, label="error |HK-QM|")
plt.plot(taus, error_wm, label="error |WM-QM|")

plt.xlim((0,400))
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




