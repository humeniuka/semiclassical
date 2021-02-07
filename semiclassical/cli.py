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
import ase.io.extxyz
import tqdm

# Local imports
import semiclassical
from semiclassical import potentials
from semiclassical import propagators
from semiclassical import readers
from semiclassical import units
from semiclassical.units import hbar
from semiclassical import broadening
from semiclassical import rates

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
    parser_dynamics.add_argument('json_input',
                                 type=str, metavar='input.json', help='input file in JSON format')
    parser_dynamics.add_argument(
        '--cuda',
        type=int, dest='cuda', default=0, metavar='id', 
        help="select id of cuda device if more than one is available, i.e. 0 for 'cuda:0'")
    
    # - rates
    parser_rates = subparsers.add_parser(
        'rates',
        help="compute Fermi's Golden Rule transition rates by Fourier transforming correlation functions")
    parser_rates.add_argument('json_input',
                              type=str, metavar='input.json', help='input file in JSON format')

    
    # - plotting
    parser_plot = subparsers.add_parser(
        'plot',
        help="plot correlation and rate functions from .npz files")
    parser_plot.add_argument('correlation_files',
                             type=str, metavar='correlation.npz',
                             help='plot correlation functions from one or more npz-files', nargs='+')
    
    # - export
    parser_export = subparsers.add_parser(
        'export',
        help="export correlation functions and rates from .npz file to .dat files for plotting with external program, produces the tables 'autocorrelation.dat', 'ic_correlation.dat' (and 'ic_rate.dat' if available).")
    parser_export.add_argument('correlation_file',
                               type=str, metavar='correlation.npz',
                               help='export data from this .npz file')
    
    # - show
    parser_show = subparsers.add_parser(
        'show',
        help="show information about .npz file")
    parser_show.add_argument('correlation_file',
                             type=str, metavar='correlation.npz',
                             help='show information about this .npz file')

    
    args = parser.parse_args()
    
    try:
        
        if args.command == 'dynamics':
            # run on GPU or CPU ?
            torch.set_default_dtype(torch.float64)
            if torch.cuda.is_available():
                args.cuda = min(args.cuda, torch.cuda.device_count())
                device = torch.device(f"cuda:{args.cuda}")
            else:
                device = torch.device('cpu')

            with open(args.json_input) as f:
                config = json.load(f)

            logger.info(f"run all 'dynamics' tasks in {args.json_input}")
            for task in config['semi']:
                if task['task'] == 'dynamics':
                    run_semiclassical_dynamics(task, device=device)

        elif args.command == 'rates':
            
            with open(args.json_input) as f:
                config = json.load(f)

            logger.info(f"run all 'rates' tasks in {args.json_input}")
            for task in config['semi']:
                if task['task'] == 'rates':
                    calculate_rates(task)
                    
        elif args.command == 'plot':
            _plot_correlation_functions(args.correlation_files)

        elif args.command == 'export':
            _export_tables(args.correlation_file)
                
        elif args.command == 'show':
            _show_information(args.correlation_file)
                
    except:
        logging.exception("""
        An error occurred, see traceback below

        Suggestions:
         * Check the JSON input file for mistakes.
         * If there is insufficient memory, decrease 'batch_size'.
        """)
        
class ConfigurationError(Exception):
    pass
        
def run_semiclassical_dynamics(task, device='cpu'):
    """
    Parameters
    ----------
    task  :   JSON structure
    """
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

        # frozen Gaussian are equal to vibrational ground state
        Gamma_0 = torch.from_numpy(Gamma_0)

        # molecule corresponding to this potential (only needed for visualization)
        atoms = ase.atoms.Atoms(numbers=excited_fchk['Atomic numbers'])
        atoms.set_positions(q0.numpy().reshape(-1,3) * units.bohr_to_angs)
        atoms.set_momenta(p0.numpy().reshape(-1,3))
        
    elif p['type'] == "gdml":
        # ground state potential and non-adiabatic coupling
        model_pot = np.load(p['ground'], allow_pickle=True)
        with open(p['coupling']) as f:
            nacs_fchk = readers.FormattedCheckpointFile(f)

        potential = potentials.MolecularGDMLPotential(model_pot, nacs_fchk)

        # initial wavepacket on excited state
        with open(p['excited']) as f:
            excited_fchk = readers.FormattedCheckpointFile(f)
        x0, Gamma_0, en_zpt = excited_fchk.vibrational_groundstate()
        # center of initial wavepacket
        q0 = torch.from_numpy(x0)
        # momentum of initial wavepacket
        p0 = torch.zeros_like(q0)

        # frozen Gaussian are equal to vibrational ground state
        Gamma_0 = torch.from_numpy(Gamma_0)

        # molecule corresponding to this potential (only needed for visualization)
        atoms = ase.atoms.Atoms(numbers=excited_fchk['Atomic numbers'])
        atoms.set_positions(q0.numpy().reshape(-1,3) * units.bohr_to_angs)
        atoms.set_momenta(p0.numpy().reshape(-1,3))
        
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

        # not a molecular potential
        atoms = None
    else:
        raise ConfigurationError(f"Unknown potential type in {task['potential']}")

    # Find minimum of final potential energy surface and set it as the origin of the energy axis.
    # We start at the minimum of the initial potential and follow the gradient on
    # the final potential until the minimum is found. All energies are measured with
    # respect to the minimum energy.
    if hasattr(potential, "minimize"):
        logger.info("find minimum on final potential energy surface")
        potential.minimize(q0)

    if p['type'] in ["harmonic", "gdml"]:
        # For molecular potentials compute adiabatic excitation energy
        adiabatic_gap = excited_fchk.total_energy() - potential.total_energy()
        logger.info(f"  adiabatic excitation energy               : {adiabatic_gap*units.hartree_to_ev:.4f} eV")
    else:
        # for model potentials the energy gap is not defined
        adiabatic_gap = np.nan
        
    Gamma_i = Gamma_0
    Gamma_t = Gamma_0

    # # Grid for Time Propagation

    dt = task['time_step_fs'] / units.autime_to_fs
    nt = task['num_steps']
    t_max = nt*dt
    times = torch.linspace(0.0, t_max, nt)

    logger.info(f"  time step                                 : {dt*units.autime_to_fs:.5f} fs")
    logger.info(f"  number of time steps                      : {nt}")
    logger.info(f"  propagation time                          : {t_max*units.autime_to_fs:.5f} fs")
    
    
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
                 adiabatic_gap=adiabatic_gap,
                 trajectories=0)
    else:
        assert task.get('manual_seed', None) is None, \
            "Multiple runs with the same sequence of random numbers make no sense! Do not use `manual_seed` and `overwrite=False` at the same time"
        
        # check that existing data is compatible compatible with this dynamics run
        data = np.load(filename)
        assert np.array_equal(data['times'], times.numpy()), \
            f"Time steps in {filename} differ. Delete the old file or change the grid for time propagation."
        assert data['propagator'] == propagator_name, \
            f"Data produced with different propagators cannot be added."

        
    # make random numbers reproducible if desired
    seed = task.get('manual_seed', None)
    if not seed is None:
        logger.warning("The random number generator should not be seeded manually unless for debugging!")
        logger.warning("Sequences of random numbers will be identical between different runs.")
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
        
        # sample initial conditions
        propagator.initial_conditions(q0, p0, Gamma_0,
                                      ntraj=num_samples)

        # for molecular potentials, export initial coordinates and momenta for visualization
        _export_trajectories_extxyz(task.get('export_initial', ''), atoms, propagator,
                                    append=(repetition > 0))
                
        # run semiclassical dynamics
        with tqdm.tqdm(total=nt) as progress_bar:
            for t in range(0, nt):
                autocorrelation_[t] += propagator.autocorrelation()
                ic_correlation_[t] += propagator.ic_correlation(potential, energy0_es=en_zpt)

                # If any NaN's are detected the simulation is aborted.
                assert not np.isnan(autocorrelation_).any(), f"encountered NaN's in autocorrelation : {autocorrelation_}"
                assert not np.isnan(ic_correlation_).any(), f"encountered NaN's in IC correlation : {ic_autocorrelation_}"

                # Monitoring the norm tells us if the calculation is converged with respect to
                # the number of trajectories (i.e. if the basis of coherent states is complete).
                # Calculating the norm is extremely costly (scales like Ntraj^2) and should be
                # avoided except for debugging or finding the optimal number of trajectories.
                # The norm can only be computed for the trajectories in a single batch. Multiple
                # batches improve the statistics, but this not reflected in the norm of a single batch.
                calc_norm_every = task.get('calc_norm_every', 0)
                if (calc_norm_every > 0) and (t % calc_norm_every == 0):
                    # compute norm every `calc_norm_every` time step
                    logger.info(f" computing norm of wavefunction ...")
                    norm = propagator.norm()
                    logger.info(f" time/fs= {times[t]*units.autime_to_fs}  norm= {norm:9.6f}")

                # show progress
                progress_bar.set_description(f" ({repetition+1:6}/{num_repetitions:6}) {t+1:6}/{nt:6}   time= {times[t]:10.4f}   time/fs= {times[t]*units.autime_to_fs:10.4f}")
                progress_bar.update(1)

                # advance to next time step,  t -> t+dt
                propagator.step(potential, dt)

        # for molecular potentials, export final coordinates and momenta for visualization
        _export_trajectories_extxyz(task.get('export_final', ''), atoms, propagator,
                                    append=(repetition > 0))
                
        # add averages from different repetitions
        #
        #                    i=n+m
        #  F(n:n+m) = 1/m sum     f(i)
        #                    i=n
        #
        #                        i=m+n
        #  F(1:n+m) = 1/(m+n) sum      f(i) = ( n * F(1:n) + m * F(n:n+m) ) / (m+n)
        #                        i=1
        #
        
        data = dict(np.load(filename))
        ntraj_old = data['trajectories']
        ntraj_new = num_samples
        ntraj_tot = ntraj_old + ntraj_new
        autocorrelation = (ntraj_new * autocorrelation_ + ntraj_old * data['autocorrelation'])/ntraj_tot
        ic_correlation  = (ntraj_new * ic_correlation_  + ntraj_old * data['ic_correlation'] )/ntraj_tot

        # C(t=0) = <phi(0)|phi(0> should be be exactly 1 because the initial conditions are sampled
        # from the normalized distribution function P(qi,pi) ~ |<qi,pi|q0,p0>|^2
        #
        #                    /                                          /
        #  <phi(0)|phi(0)> = | dqi dpi   <q0,p0|qi,pi><qi,pi|q0,p0>   = | P(qi,pi) dqi dpi = 1/n sum P(x)/P(x) = 1
        #                    /                                          /                        x~P
        logger.info(f"<phi(0)|phi(0)>= {autocorrelation[0]}")
        assert abs(autocorrelation[0] - 1.0) < 1.0e-5

        # update data in npz-file
        data['trajectories'] = ntraj_tot
        data['autocorrelation'] = autocorrelation
        data['ic_correlation'] = ic_correlation
        # The IC rates are not up-to-date anymore, so remove them
        data.pop('ic_rate', None)
        
        np.savez(filename, **data)

def _export_trajectories_extxyz(filename, atoms, propagator, append=False):
    """
    save current positions and momenta in extended XYZ format

    Parameters
    ----------
    filename   :  str
      xyz output file
    propagator :  HermanKlukPropagator or WaltonManolopoulosPropagator
      provides coordinates and momenta 
    atoms      :  instance of ase.atoms.Atoms
      molecule with equilibrium structure

    Optional
    --------
    append     :  bool
      If append is set to True, the file is for append (mode 'a'), 
      otherwise it is overwritten (mode 'w')
    """
    if filename == '':
        return
    if atoms is None:
        return
    atoms_list = []
    if not append:
        # first geometry is the equilibrium structure
        atoms_list.append(atoms)
    # remaining geometries are the trajectories
    q,p = (qp.cpu() for qp in
           propagator.current_positions_and_momenta())
    _, ntraj = q.size()
    for i in range(0, ntraj):
        atoms_ = atoms.copy()
        atoms_.set_positions(q[:,i].reshape(-1,3) * units.bohr_to_angs)
        atoms_.set_momenta(p[:,i].reshape(-1,3))
        atoms_list.append(atoms_)
    ase.io.extxyz.write_extxyz(filename, atoms_list,
                               columns=['symbols', 'positions', 'momenta'],
                               write_results=False, append=append)
    logger.info(f"positions and momenta saved to '{filename}'")
        
def calculate_rates(task):
    """
    compute rates by Fourier transforming correlation functions
    """
    # widths of broadening functions
    hwhmG = task.get('hwhmG_ev', 0.01) 
    hwhmL = task.get('hwhmL_ev', 1.0e-6)

    # convert HWHM to parameters of lineshape functions
    sigma = hwhmG / np.sqrt(2.0 * np.log(2.0)) / units.hartree_to_ev
    gamma = hwhmL / units.hartree_to_ev
    
    # type of broadening functions
    broad = task.get('broadening', 'gaussian')
    if broad == "gaussian":
        lineshape = broadening.gaussian(sigma)
    elif broad == "lorentzian":
        lineshape = broadening.lorentzian(gamma)
    elif broad == "voigtian":
        lineshape = broadening.voigtian(sigma, gamma)
    else:
        raise ValueError("'broadening' should be one of 'gaussian', 'lorentzian' or 'voigtian'")
    
    # read correlation function from this file
    corr_file = task.get('correlations', 'correlations.npz')
    # write rates vs. energy to this file
    rate_file = task.get('rates', 'correlations.npz')

    logger.info(f"compute rates from correlation functions in '{corr_file}'")

    data = dict(np.load(corr_file))

    logger.info(f"trajectories : {data['trajectories']}")
    logger.info(f"time grid    : tmin= {data['times'].min():.4f} tmax= {data['times'].max():.4f} steps= {len(data['times'])}")
    
    data['broadening'] = broad
    data['hwhmG'] = hwhmG
    data['hwhmL'] = hwhmL

    energies, ic_rate = rates.rate_from_correlation(data['times'], data['ic_correlation'], lineshape)

    # TODO: 
    # For some reason, we have to multiply the rate by a factor of 4*pi
    # to get agreement with FCclasses3.
    # Well, anyway this must be the correct expression, but it would be nice to know why.
    ic_rate *= 4.0*np.pi
    
    data['energies'] = energies[energies >= 0.0]
    data['ic_rate'] = ic_rate[energies >= 0.0].real

    logger.info(f"rates are saved to '{rate_file}'")
    np.savez(rate_file, **data)
    
    
def _export_tables(filename):
    """
    save correlation functions to .dat files for plotting with external programmes

    The output filenames are 
     *  autocorrelation.dat
     *  ic_correlation.dat
     *  ic_rate.dat         (if the rates have been computed)

    Parameters
    ----------
    filename  :  name of .npz file
      contains correlation functions
    """
    data = np.load(filename)
    trajectories = int(data['trajectories'])
    propagator = str(data['propagator'])
        
    datfile = os.path.splitext(filename)[0]+".dat"
    logger.info(f"exporting correlation functions from '{filename}' to tables 'autocorrelation.dat' and 'ic_correlation.dat'")
    # write table with correlation functions to file
    with open("autocorrelation.dat", "w") as f:
        f.write('# autocorrelation function\n')
        f.write(f"# propagator: {propagator}   trajectories: {trajectories}\n")
        f.write('#\n')
        f.write('# Time/fs                  Re[C(t)]                  Im[C(t)]\n')
        np.savetxt(f, np.vstack(
            (data['times'] * units.autime_to_fs,
             data['autocorrelation'].real,
             data['autocorrelation'].imag)
        ).T)
    with open("ic_correlation.dat", "w") as f:
        f.write('# IC-correlation function\n')
        f.write(f"# propagator: {propagator}   trajectories: {trajectories}\n")
        f.write('#\n')
        f.write('# Time/fs                  Re[kIC(t)]                Im[kIC(t)]\n')
        np.savetxt(f, np.vstack(
            (data['times'] * units.autime_to_fs,
             data['ic_correlation'].real,
             data['ic_correlation'].imag)
        ).T)
    if 'ic_rate' in data:
        logger.info(f"exporting IC rates from '{filename}' to tables 'ic_rate.dat'")
        with open("ic_rate.dat", "w") as f:
            f.write('# internal conversion rate\n')
            f.write(f"# propagator: {propagator}   trajectories: {trajectories}\n")
            f.write(f"# broadening: {data['broadening']}   HWHM_G: {data['hwhmG']} eV   HWHM_L: {data['hwhmL']} eV\n")
            f.write('#\n')
            f.write('# Energy/eV                kIC(E)/s^-1\n')
            np.savetxt(f, np.vstack(
                (data['energies'] * units.hartree_to_ev,
                 data['ic_rate'].real)
            ).T)
            
    
def _plot_correlation_functions(filenames):
    """
    plot autocorrelation, IC correlation and (if available) transition rates 
    loaded from .npz file
    """
    import matplotlib
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    matplotlib.rc('legend', fontsize=12)
    matplotlib.rc('axes', labelsize=12)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12.0,6.0))
               
    ax1 = plt.subplot(1,3,1)
    ax1.set_xlabel("Time / fs")
    ax1.set_ylabel("Autocorrelation")

    ax2 = plt.subplot(1,3,2)
    ax2.set_xlabel("Time / fs")
    ax2.set_ylabel("IC correlation")

    ax3 = plt.subplot(1,3,3)
    ax3.set_xlabel("Energy / eV")
    ax3.set_ylabel("IC rate (log) / s$^{-1}$")
    ax3.set_yscale('log')
    ax3.set_xlim((0.0,10.0))
    
    # numbers of trajectories and names of propagators used in each file
    trajectories = []
    propagators = []

    linestyles = ["-", "--", "-.", ".."]
    
    for ifile, filename in enumerate(filenames):
        data = np.load(filename)

        linestyle = linestyles[ifile % len(linestyles)]
        trajectories.append( int(data['trajectories']) )
        propagators.append( str(data['propagator']) )

        # correlation (functions of time)
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

        # rates (functions of energy)
        if 'ic_rate' in data:
            ax3.plot(data['energies'] * units.hartree_to_ev, data['ic_rate'], ls=linestyle)
        
    plt.suptitle(f"trajectories: {trajectories}, propagators: {propagators}")
    ax2.legend(bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    plt.show()

def _show_information(filename):
    """
    Show what is in the .npz file
    """
    data = np.load(filename)
    print(f"""
    filename               : {filename}
    propagator             : {data['propagator']}
    trajectories           : {data['trajectories']:10}
    time step (fs)         : {(data['times'][1]-data['times'][0])*units.autime_to_fs:10.4f}
    propagation time (fs)  : {max(data['times'])*units.autime_to_fs:10.4f}
    """)
    if 'ic_rate' in data:
        if not np.isnan(data['adiabatic_gap']):
            # read off IC rate at point closest to adiabatic excitation energy
            iclosest = np.argmin(abs(data['energies'] - data['adiabatic_gap']))
            kic = data['ic_rate'][iclosest]
        else:
            kic = np.nan
        print(f"""
    adiabatic gap Ead (eV) : {data['adiabatic_gap']*units.hartree_to_ev:6.3f}
    IC rate kic(Ead) (s-1) : {kic:6.3e}
        """)
    else:
        print("  No rates found in file, you have to compute them first with the command 'semi rates'.")

    
if __name__ == "__main__":
    main()
