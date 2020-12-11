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

if len(sys.argv) < 4:
    print("Expected argument: <number of samples> <anharmonicity> <propagator>")
    exit(-1)

num_samples = int(sys.argv[1])
anharmonicity = float(sys.argv[2])
propagator_name = sys.argv[3]
assert propagator_name in ["HK", "WM"]

print(f"number of samples : {num_samples}")
print(f"anharmonicity chi : {anharmonicity}")
print(f"propagator        : {propagator_name}")

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

if len(data.shape) == 1:
    # When only singly mode is read, data has the wrong shape
    #  (ncol,) -> (1,ncol)
    data = np.reshape(data, (1, len(data)))

print("selected vibrational modes (cm^-1):")
print(data[:,0])

# number of nuclear degrees of freedom
dim = data.shape[1]

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
nt = 4000
# propagate for 150 fs 
t_max = 150.0 / units.autime_to_fs
times = torch.linspace(0.0, t_max, nt)
dt = times[1]-times[0]
print(f"time step dt= {dt*units.autime_to_fs} fs")


# # Semiclassical Time Evolution

# choose width parameters of the frozen Gaussians equal to the normal mode frequencies
Gamma_i = torch.diag(omega)
Gamma_t = Gamma_i

# What is a reasonable value for beta?
e, V = torch.symeig(Gamma_0, eigenvectors=True)
alpha = 10.0 * e.max().item()
beta = 10.0 * 1.0/e.min().item()

print(f"alpha= {alpha}  beta = {beta}")
print("volume of phase space cell V= ", np.sqrt(alpha*beta)**dim)

# make random numbers reproducible
#torch.manual_seed(0)

if propagator_name == "WM":
    propagator = WaltonManolopoulosPropagator(Gamma_i, Gamma_t, alpha, beta, device=device)
else:
    propagator = HermanKlukPropagator(Gamma_i, Gamma_t, device=device)
    
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
        norm = propagator.norm()
        print(f"|psi|= {norm}")
    propagator.step(potential, dt)


corr_file = "/tmp/correlation_chi%s_T0_%s.dat" % (anharmonicity, propagator_name)
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

kic_e = internal_conversion_rate(times, ic_correlation, lineshape, rate_file="/tmp/kICvsE_chi%s_T0_%s.dat" % (anharmonicity, propagator_name))



