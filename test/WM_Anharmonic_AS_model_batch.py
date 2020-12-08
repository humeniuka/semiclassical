#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import numpy.linalg as la
from numpy import fft
import torch

from semiclassical.propagators import HermanKlukPropagator, WaltonManolopoulosPropagator
from semiclassical.potentials import MorsePotential
from semiclassical.propagators import hbar
from semiclassical.broadening import gaussian
from semiclassical.rates import internal_conversion_rate
from semiclassical import units

if len(sys.argv) < 3:
    print("Expected argument: <number of samples> <anharmonicity>")
    exit(-1)

num_samples = int(sys.argv[1])
anharmonicity = float(sys.argv[2])

print(f"number of samples : {num_samples}")
print(f"anharmonicity chi : {anharmonicity}")


# # GPU or CPU ?
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    print("CUDA available")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# # Adiabatic Shift Model

dat_file = "DATA/huang_rhys_nacs_AS.dat"

# load frequencies, Huang-Rhys factors and NACs
data = torch.from_numpy(np.loadtxt(dat_file))

print("selected vibrational modes (cm^-1):")
print(data[:,0])

# frequencies in Hartree
omega = data[:,0] / units.hartree_to_wavenumbers
# Huang-Rhys factors
S = data[:,1]
# NACs
nac = data[:,2]
    
# The horizontal shift dQ of the excited state PES is related to the Huang-Rhys factor
#  S = 1/2 dQ^2 omega
#  dQ = sqrt(2*|S|/omega) * sign(S)
dQ = torch.sqrt(2.0*abs(S)/omega) * torch.sign(S)
# The sign of S is not needed anymore, Huang-Rhys factors are always positive
S = abs(S)

# anharmonicity
chi = torch.tensor([anharmonicity]).expand_as(omega)

# ground state potential energy surface 
potential = MorsePotential(omega, chi, nac)

# width of initial wavepacket psi(x,t=0) on the excited state
Gamma_0 = torch.diag(omega)
# center of initial wavepacket
q0 = dQ
# momentum of initial wavepacket
p0 = 0.0*q0

# zero-point energy of the excited state potential
en0 = torch.sum(hbar/2.0 * omega).item()


# # Grid for Time Propagation

# time grid
nt = 4000                                // 40
# propagate for 150 fs 
t_max = 150.0 / units.autime_to_fs       // 40
times = torch.linspace(0.0, t_max, nt)
dt = times[1]-times[0]
print(f"time step dt= {dt*units.autime_to_fs} fs")


# # Semiclassical Time Evolution

# choose width parameters of the frozen Gaussians equal to the normal mode frequencies
Gamma_i = torch.diag(omega)
Gamma_t = Gamma_i
beta = 100.0

# make random numbers reproducible
#torch.manual_seed(0)

#propagator = HermanKlukPropagator(Gamma_i, Gamma_t, beta, device=device)
propagator = WaltonManolopoulosPropagator(Gamma_i, Gamma_t, beta, device=device)

# initial conditions
propagator.initial_conditions(q0, p0, Gamma_0, ntraj=num_samples)

# save autocorrelation function for each time step
autocorrelation = np.zeros((nt,), dtype=complex)

# correlation function for internal conversion
ic_correlation = np.zeros((nt,), dtype=complex)

for t in range(0, nt):
    autocorrelation[t] = propagator.autocorrelation()
    ic_correlation[t] = propagator.ic_correlation(potential, energy0_es=en0)
    if t % 1 == 0:
        print(f"{t+1:6}/{nt:6}   time= {times[t]:10.4f}   time/fs= {times[t]*units.autime_to_fs:10.4f}")

    propagator.step(potential, dt)


corr_file = "/tmp/correlation_chi%s_T0.dat" % anharmonicity
# save correlation function k_ic(t)
data = np.vstack( (times*units.autime_to_fs, ic_correlation.real, ic_correlation.imag) ).transpose()
with open(corr_file, "w") as f:
    f.write("# Time / fs         Re[k_IC(t)]         Im[k_IC(t)]\n")
    np.savetxt(f, data)
print("wrote table with correlation function to '%s'" % corr_file)

# lineshape function
# The width is taken to be proportional to the spacing of the vibrational energy levels
sigma = 0.5*torch.mean(omega)

# width of the broadening function in the time domain
tau = hbar/sigma
print(f"sigma= {sigma} Hartree --> tau = {tau*units.autime_to_fs} fs")

# `gaussian(...)` is a factory that creates another function of t
lineshape = gaussian(sigma)

# # Fourier transform

kic_e = internal_conversion_rate(times, ic_correlation, lineshape, rate_file="/tmp/kICvsE_chi%s_T0.dat" % anharmonicity)



