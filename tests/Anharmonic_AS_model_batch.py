#!/usr/bin/env python
# coding: utf-8
# # Imports
import sys
import numpy as np
import numpy.linalg as la
from numpy import fft
import torch
import logging
import os.path

# # Local Imports
from semiclassical.propagators import HermanKlukPropagator, WaltonManolopoulosPropagator
from semiclassical.potentials import MorsePotential
from semiclassical.propagators import hbar
from semiclassical.broadening import gaussian
from semiclassical.rates import internal_conversion_rate
from semiclassical import units

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[testing] %(message)s", level=logging.INFO)


if len(sys.argv) < 6:
    print("Expected argument: <AS model file> <number of trajectories> <anharmonicity> <propagator> <output dir>")
    exit(-1)

model_file = sys.argv[1]
num_trajectories = int(sys.argv[2])
anharmonicity = float(sys.argv[3])
propagator_name = sys.argv[4]
outdir = sys.argv[5]

assert propagator_name in ["HK", "WM"]

# The trajectories are run in parallel in batches.
batch_size = 5000 #50000
num_repetitions = max(num_trajectories // batch_size, 1)
num_samples = min(batch_size, num_trajectories)

logger.info(f"  propagator                                : {propagator_name}")
logger.info(f"  number of repetitions                     : {num_repetitions}")
logger.info(f"  number of trajectories per repetition     : {num_samples}")
logger.info(f"  total number of trajectories              : {num_samples * num_repetitions}")
logger.info(f"  adiabatic shift model                     : {model_file}")
logger.info(f"  anharmonicity chi                         : {anharmonicity}")

# # GPU or CPU ?
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    logger.info("CUDA available")
    # If there are several GPU's available, we use the last one,
    # i.e. "cuda:1" on a workstation with 2 GPUs.
    device = torch.device("cuda:%d" % (torch.cuda.device_count()-1))
else:
    device = torch.device('cpu')
    
# # Adiabatic Shift Model

# load frequencies, Huang-Rhys factors and NACs
data = torch.from_numpy(np.loadtxt(model_file))

if len(data.shape) == 1:
    # When only singly mode is read, data has the wrong shape
    #  (ncol,) -> (1,ncol)
    data = np.reshape(data, (1, len(data)))

logger.info("selected vibrational modes (cm^-1):")
logger.info(data[:,0])

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
logger.info(f"time step dt= {dt*units.autime_to_fs} fs")


# # Semiclassical Time Evolution

# choose width parameters of the frozen Gaussians equal to the normal mode frequencies
Gamma_i = torch.diag(omega)
Gamma_t = Gamma_i

# Choose cell dimensions (volume proportional to 1/(a*b)^(dim/2))
alpha = 100.0
beta = 100.0

logger.info(f"alpha= {alpha}  beta = {beta}")
logger.info(f"volume of phase space cell V= {(1.0/(2*np.sqrt(alpha*beta)))**dim}")

# make random numbers reproducible
#torch.manual_seed(0)

# The dynamics simulation is repeated multiple times with different randomly chosen
# initial conditions, since only a limited number of trajectories can run in parallel
# due to memory constraints. At the end the correlation functions from different runs
# are averaged.

# The following arrays accumulate the autocorrelation and IC correlation functions
# from different repetitions.
autocorrelation_acc = np.zeros((nt,), dtype=complex)
ic_correlation_acc = np.zeros((nt,), dtype=complex)

for repetition in range(0, num_repetitions):
    logger.info(f"*** Repetition {repetition+1} ***")
    if propagator_name == "WM":
        propagator = WaltonManolopoulosPropagator(Gamma_i, Gamma_t, alpha, beta, device=device)
    else:
        propagator = HermanKlukPropagator(Gamma_i, Gamma_t, device=device)

    # save autocorrelation function for each time step
    autocorrelation = np.zeros((nt,), dtype=complex)

    # correlation function for internal conversion
    ic_correlation = np.zeros((nt,), dtype=complex)

    # initial conditions
    propagator.initial_conditions(q0, p0, Gamma_0, ntraj=num_samples)

    for t in range(0, nt):
        autocorrelation[t] += propagator.autocorrelation()
        ic_correlation[t] += propagator.ic_correlation(potential, energy0_es=en0)
        if t % 1 == 0:
            logger.info(f" ({repetition+1:6}/{num_repetitions:6}) {t+1:6}/{nt:6}   time= {times[t]:10.4f}   time/fs= {times[t]*units.autime_to_fs:10.4f}")
            #norm = propagator.norm()
            #logger.info(f"|psi|= {norm}")
        propagator.step(potential, dt)
        
    # save autocorrelation function <phi(0)|phi(t)>
    autocorr_file = os.path.join(outdir, "autocorrelation_%dtrajs_repetition-%d.dat" % (num_samples, repetition))
    data = np.vstack( (times*units.autime_to_fs, autocorrelation.real, autocorrelation.imag) ).transpose()
    with open(autocorr_file, "w") as f:
        f.write("# Time / fs         Re[<phi(0)|phi(t)>]   Im[<phi(0)|phi(t)>]\n")
        np.savetxt(f, data)
    logger.info(f"wrote table with autocorrelation function to '{autocorr_file}'")

    # save IC correlation function k_ic(t)
    ic_corr_file = os.path.join(outdir, "ic_correlation_%dtrajs_repetition-%d.dat" % (num_samples, repetition))
    data = np.vstack( (times*units.autime_to_fs, ic_correlation.real, ic_correlation.imag) ).transpose()
    with open(ic_corr_file, "w") as f:
        f.write("# Time / fs         Re[k_IC(t)]         Im[k_IC(t)]\n")
        np.savetxt(f, data)
    logger.info(f"wrote table with correlation function to '{ic_corr_file}'")

    autocorrelation_acc += autocorrelation
    ic_correlation_acc += ic_correlation
    
# average correlation functions accumulated from different runs
autocorrelation_acc /= num_repetitions
ic_correlation_acc /= num_repetitions

# save autocorrelation function <phi(0)|phi(t)>
autocorr_file = os.path.join(outdir, "autocorrelation_acc_%dtrajs.dat" % num_trajectories)
data = np.vstack( (times*units.autime_to_fs, autocorrelation_acc.real, autocorrelation_acc.imag) ).transpose()
with open(autocorr_file, "w") as f:
    f.write("# Time / fs         Re[<phi(0)|phi(t)>]   Im[<phi(0)|phi(t)>]\n")
    np.savetxt(f, data)
logger.info(f"wrote table with autocorrelation function to '{autocorr_file}'")

# save IC correlation function k_ic(t)
ic_corr_file = os.path.join(outdir, "ic_correlation_acc_%dtrajs.dat" % num_trajectories)
data = np.vstack( (times*units.autime_to_fs, ic_correlation_acc.real, ic_correlation_acc.imag) ).transpose()
with open(ic_corr_file, "w") as f:
    f.write("# Time / fs         Re[k_IC(t)]         Im[k_IC(t)]\n")
    np.savetxt(f, data)
logger.info(f"wrote table with correlation function to '{ic_corr_file}'")

    
# lineshape function
# The width is taken to be proportional to the spacing of the vibrational energy levels
sigma = 0.5*torch.mean(omega)

# width of the broadening function in the time domain
tau = hbar/sigma
logger.info(f"sigma= {sigma} Hartree --> tau = {tau*units.autime_to_fs} fs")

# `gaussian(...)` is a factory that creates another function of t
lineshape = gaussian(sigma)

# # Fourier transform

kic_e = internal_conversion_rate(times, ic_correlation, lineshape, rate_file=os.path.join(outdir, "kICvsE_chi%s_T0_%s.dat" % (anharmonicity, propagator_name)))



