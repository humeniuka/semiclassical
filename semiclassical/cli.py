#!/usr/bin/python
# coding: utf-8

# MIT License
#
# Copyright (c) 2020 Alexander Humeniuk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import argparse
import sys
import json
import os.path

import numpy as np
import torch
import ase
import tqdm

# Local imports
import semiclassical
from semiclassical import potentials
from semiclassical import propagators
from semiclassical import readers
from semiclassical import units
from semiclassical.units import hbar


# # Logging
logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s '
        + semiclassical.__version__
        + ' [Python {}, NumPy {}'.format(
            '.'.join(map(str, sys.version_info[:3])), np.__version__
        )
        + ', PyTorch {}'.format(torch.__version__)
        + ', ASE {}'.format(ase.__version__)
        + ']',
    )

        
    # sub commands
    subparsers = parser.add_subparsers(help='commands', dest='command')
    # - dynamics
    parser_dynamics = subparsers.add_parser(
        'dynamics',
        help="run semiempirical dynamics")
    parser_dynamics.add_argument('json_input', type=str, metavar='input.json', help='input file in JSON format')
    parser_dynamics.add_argument(
        '--cuda',
        type=int, dest='cuda', default=0, metavar='id', 
        help="select cuda device if more than one is available")
    # - plotting
    parser_plot = subparsers.add_parser(
        'plot',
        help="plot correlation functions from .npz files")
    parser_plot.add_argument('correlation_files', type=str, metavar='correlation.npz', help='plot correlation functions from one or more npz-files', nargs='+')

    args = parser.parse_args()

    try:
        
        if args.command == 'dynamics':
            with open(args.json_input) as f:
                config = json.load(f)

            for task in config['semi']:
                _run_semiclassical_dynamics(task)

        elif args.command == 'plot':
            _plot_correlation_functions(args.correlation_files)
            
    except:
        logging.exception("An error occurred, see traceback below")
        
class ConfigurationError(Exception):
    pass
        
def _run_semiclassical_dynamics(task):
    assert task['title'] == "internal conversion"
    
    # ground state potential and non-adiabatic couplings
    p = task['potential']
    if p['type'] == "harmonic":
        # ground state potential and non-adiabatic coupling
        with open(p['ground']) as f:
            freq_fchk = readers.FormattedCheckpointFile(f)
        with open(p['coupling']) as f:
            nacs_fchk = readers.FormattedCheckpointFile(f)
        potential = potentials.MolecularHarmonicPotential(freq_fchk, nacs_fchk)
        
        # initial wavepacket on excited state
        with open(p['excited']) as f:
            excited_fchk = readers.FormattedCheckpointFile(f)
        x0, Gamma_0, en_zpt = excited_fchk.vibrational_groundstate()
        # center of initial wavepacket
        q0 = torch.from_numpy(x0)
        # momentum of initial wavepacket
        p0 = torch.zeros_like(q0)

        ### DEBUG
        en_zpt = 0.0
        ###
        
        # frozen Gaussian are equal to vibrational ground state
        Gamma_0 = torch.from_numpy(Gamma_0)

    elif p['type'] == "gdml":
        # ground state potential and non-adiabatic coupling
        model_pot = np.load(p['ground'], allow_pickle=True)
        model_nac = np.load(p['coupling'], allow_pickle=True)

        potential = potentials.MolecularGDMLPotential(model_pot, model_nac)

        # initial wavepacket on excited state
        with open(p['excited']) as f:
            minimumI = readers.FormattedCheckpointFile(f)
        x0, Gamma_0, en_zpt = minimumI.vibrational_groundstate()
        # center of initial wavepacket
        q0 = torch.from_numpy(x0)
        # momentum of initial wavepacket
        p0 = torch.zeros_like(q0)

        # frozen Gaussian are equal to vibrational ground state
        Gamma_0 = torch.from_numpy(Gamma_0)

    elif p['type'] == "anharmonic AS":
        model_file = p['model_file']
        anharmonicity = p['anharmonicity']

        # load frequencies, Huang-Rhys factors and NACs
        data = torch.from_numpy( np.loadtxt(model_file) )
        
        if len(data.shape) == 1:
            # When only a single mode is read, data has the wrong shape
            #  (ncol,) -> (1,ncol)
            data = torch.reshape(data, (1, -1))

        logger.info("vibrational modes (cm^-1):")
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
        # zero modes are not displaced
        dQ[omega == 0.0] = 0.0
        # The sign of S is not needed anymore, Huang-Rhys factors are always positive
        S = abs(S)

        # anharmonicity
        chi = torch.tensor([anharmonicity]).expand_as(omega)

        # ground state potential energy surface 
        potential = potentials.MorsePotential(omega, chi, nac)

        # width of initial wavepacket psi(x,t=0) on the excited state
        Gamma_0 = torch.diag(omega)
        
        # center of initial wavepacket
        q0 = dQ
        # momentum of initial wavepacket
        p0 = 0.0*q0
        
        # zero-point energy of the excited state potential
        en_zpt = torch.sum(hbar/2.0 * omega).item()

        
    else:
        raise ConfigurationError(f"Unknown potential type in {task['potential']}")


    Gamma_i = Gamma_0
    Gamma_t = Gamma_0

    # # Grid for Time Propagation

    dt = task['time_step_fs'] / units.autime_to_fs
    nt = task['num_steps']
    t_max = nt*dt
    times = torch.linspace(0.0, t_max, nt)

    logger.info(f"  time step                                 : {dt*units.autime_to_fs} fs")
    logger.info(f"  number of time steps                      : {nt}")
    logger.info(f"  propagation time                          : {t_max*units.autime_to_fs} fs")
    
    
    # The trajectories are run in parallel in batches.
    batch_size = task.get('batch_size', 10000)
    num_trajectories = task.get('num_trajectories', 50000)
    num_repetitions = max(num_trajectories // batch_size, 1)
    num_samples = min(batch_size, num_trajectories)

    logger.info(f"  number of repetitions                     : {num_repetitions}")
    logger.info(f"  number of trajectories per repetition     : {num_samples}")
    logger.info(f"  total number of trajectories              : {num_samples * num_repetitions}")

    propagator_name = task.get('propagator', 'HK')
    logger.info(f"  propagator                                : {propagator_name}")

    # # GPU or CPU ?
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        logger.info("CUDA available")
        # If there are several GPU's available, we use the last one,
        # i.e. "cuda:1" on a workstation with 2 GPUs.
        device = torch.device("cuda:%d" % (torch.cuda.device_count()-1))
    else:
        device = torch.device('cpu')

    # The dynamics simulation is repeated multiple times with different randomly chosen
    # initial conditions, since only a limited number of trajectories can run in parallel
    # due to memory constraints. At the end the correlation functions from different runs
    # are averaged.

    # The following arrays accumulate the autocorrelation and IC correlation functions
    # from different repetitions.
    autocorrelation = np.zeros((nt,), dtype=complex)
    ic_correlation = np.zeros((nt,), dtype=complex)        

    filename = task['results'].get('correlations', 'correlations.npz')
    if task['results'].get('overwrite', True) == True or (not os.path.exists(filename)):
        # overwrite old file with empty data
        np.savez(filename,
                 propagator=propagator_name,
                 times=times,
                 autocorrelation=autocorrelation,
                 ic_correlation=ic_correlation,
                 trajectories=0)
    else:
        # check that existing data is compatible compatible with this dynamics run
        data = np.load(filename)
        assert np.array_equal(data['times'], times.numpy()), \
            f"Time steps in {filename} differ. Delete the old file or change the grid for time propagation."
        assert data['propagator'] == propagator_name, \
            f"Data produced with different propagators cannot be added."

        
    # make random numbers reproducible if desired
    seed = task.get('manual_seed', None)
    if not seed is None:
        torch.manual_seed(seed)

    # run semiclassical dynamics 
    for repetition in range(0, num_repetitions):
        logger.info(f"*** Repetition {repetition+1} ***")
        if propagator_name == "WM":
            alpha = task.get('cell_width', 10000.0)
            beta = alpha
            propagator = propagators.WaltonManolopoulosPropagator(Gamma_i, Gamma_t,
                                                                  alpha, beta, device=device)
        else:
            propagator = propagators.HermanKlukPropagator(Gamma_i, Gamma_t,
                                                          device=device)

        # save autocorrelation function for each time step
        autocorrelation_ = np.zeros((nt,), dtype=complex)

        # correlation function for internal conversion
        ic_correlation_ = np.zeros((nt,), dtype=complex)

        # initial conditions
        propagator.initial_conditions(q0, p0, Gamma_0,
                                      ntraj=num_samples)

        with tqdm.tqdm(total=nt) as progress_bar:
            for t in range(0, nt):
                autocorrelation_[t] += propagator.autocorrelation()
                ic_correlation_[t] += propagator.ic_correlation(potential, energy0_es=en_zpt)

                progress_bar.set_description(f" ({repetition+1:6}/{num_repetitions:6}) {t+1:6}/{nt:6}   time= {times[t]:10.4f}   time/fs= {times[t]*units.autime_to_fs:10.4f}")
                progress_bar.update(1)

                propagator.step(potential, dt)

        # running average
        #
        # F(n) = 1/n sum_i=1^n f(i)
        #
        # F(n+1) = 1/(n+1) sum_i=1^{n+1} f(i)
        #
        #        = 1/(n+1) (n*F(n) + f(n+1))
        #
        autocorrelation = ( repetition*autocorrelation + autocorrelation_ ) / (repetition + 1.0)
        ic_correlation = ic_correlation_
        
        data = np.load(filename)        
        ntraj_old = data['trajectories']
        ntraj_new = num_samples
        ntraj_tot = ntraj_old + ntraj_new
        autocorrelation = (ntraj_new * autocorrelation + ntraj_old * data['autocorrelation'])/ntraj_tot
        ic_correlation  = (ntraj_new * ic_correlation  + ntraj_old * data['ic_correlation'] )/ntraj_tot

        # <phi(0)|phi(0> should be equal to 1.0 for a converged autocorrelation function
        logger.info(f"<phi(0)|phi(0)>= {autocorrelation[0]}")
        
        np.savez(filename,
                 propagator=propagator_name,
                 times=times,
                 autocorrelation=autocorrelation,
                 ic_correlation=ic_correlation,
                 trajectories=ntraj_tot)

def _plot_correlation_functions(filenames):
    """
    plot autocorrelation and IC correlation loaded from .npz file
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12.0,6.0))
               
    ax1 = plt.subplot(1,2,1)
    ax1.set_xlabel("Time / fs")
    ax1.set_ylabel("Autocorrelation")

    ax2 = plt.subplot(1,2,2)
    ax2.set_xlabel("Time / fs")
    ax2.set_ylabel("IC correlation")

    # numbers of trajectories and names of propagators used in each file
    trajectories = []
    propagators = []

    linestyles = ["-", "--", "-.", ".."]
    
    for ifile, filename in enumerate(filenames):
        data = np.load(filename)

        linestyle = linestyles[ifile % len(linestyles)]
        trajectories.append( int(data['trajectories']) )
        propagators.append( str(data['propagator']) )

        lre, = ax1.plot(data['times'] * units.autime_to_fs, data['autocorrelation'].real,
                        ls=linestyle)
        lim, = ax1.plot(data['times'] * units.autime_to_fs, data['autocorrelation'].imag,
                        ls=linestyle)

        ax2.plot(data['times'] * units.autime_to_fs, data['ic_correlation'].real,
                 ls=linestyle,
                 color=lre.get_color(),
                 label=f"Re[{filename}]")
        ax2.plot(data['times'] * units.autime_to_fs, data['ic_correlation'].imag,
                 ls=linestyle,
                 color=lim.get_color(),
                 label=f"Im[{filename}]")

    plt.suptitle(f"trajectories: {trajectories}, propagators: {propagators}")
    ax2.legend(bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
