
Internal Conversion Rates from Semiclassical Dynamics
-----------------------------------------------------

The long term goal of this project is to compute rates for internal conversion in medium sized molecules
from semiclassical molecular dynamics.

Milestones
----------

 * implement Herman-Kluk and Walton-Manolopoulos semiclassical propagators (*done*)
 * compare with exact QM rates for _anharmonic_ adiabatic shift (AS) model with 5 (*done*), 10,...,60 modes
 * compare with exact QM rates for _harmonic_ model: AH (*done* for methylium with 6 modes, 12 cartesian coordinates)
 * include Herzberg-Teller effect (coordinate dependence of non-adiabatic coupling vectors)
 * implement Hessians for sGDML force field (*done*)

   
Requirements
------------

Required python packages:

 * pytorch
 * numpy, scipy, matplotlib
 * ase (Atomic Simulation Environment)
 * tqdm

Complex numbers are not supported in older releases of pytorch, 
this code requires pytorch=1.8.0 or higher.
pytorch=1.8.0 is available from the channel pytorch-nightly

.. code-block:: bash

   $ conda install pytorch --channel pytorch-nightly


Getting Started
---------------
The package is installed by running

.. code-block:: bash

   $ pip install -e .
   
in the top directory. Calculations are run via the command line interface `semi`:

- running semiclassical dynamics:

.. code-block:: bash

   $ semi dynamics input.json

- computing rates from correlation functions

.. code-block::
   
   $ semi rates input.json

- plotting correlation functions and rates:

.. code-block::

   $ semi plot correlations.npz
   
- exporting correlation functions and rates to .dat files

.. code-block::
   
   $ semi export correlations.npz
  
- show content of .npz file and IC rates at the adiabatic excitation energy

.. code-block::

   $ semi show correlations.npz

   

-------------
Documentation
-------------

All calculations are controlled by the settings in a command file which is
structured using the JSON format.

=======================
Structure of Input File
=======================

.. code-block:: javascript

   { "semi" : [
      {
      	"task" : "dynamics",
	... keywords ...
      },		
      {		
	"task" : "rates",
	... keywords ...
      },
      ...
   ]}


``task`` determines the type of calculation:
 * **"dynamics"** runs semiclassical dynamics and computes correlation functions.
 * **"rates"** Fourier transforms these correlation functions to obtain transition rates.

Each type of calculation requires different keywords listed below. Tasks can be run separately, since the computational effort associated with obtaining the correlation function is much larger than computing the rates once the correlation functions are ready.

For instance, one could repeatedly run batches of semiclassical trajectories, until the correlation function is converged (to accumulate results from different runs use `overwrite = false`):

.. code-block::
   
   $ semi dynamics input.json

After checking visually that the autocorrelation function has converged,

.. code-block::
   
   $ semi plot correlations.npz

the rates can be computed with

.. code-block::
   
   $ semi rates input.json

The resulting correlation functions and rates are stored in .npz files.
They can be converted to .dat files with

.. code-block::

   $ semi export correlations.npz

which should create the files called `autocorrelation.dat`, `ic_correlation.dat` and `ic_rate.dat`.

See also the examples for input files at the end.

============================
Keywords for "dynamics" task
============================
    
.. topic:: ``potential``

   | **Description:** Block defining the potential.
   | **Datatype:** JSON
   | **Keywords:**
   |
   |          ``type``
   |
   |             **Description:** Chooses type of potential. Different potentials have different keywords.
   |             **Datatype:** string
   |                - *'anharmonic AS'* :  anharmonic adiabatic shift model ( [anharmAS]_ )
   |                - *'harmonic'*      :  The molecular potential of the ground and excited state
   |                                       are expanded harmonically around equilibrium geometries.
   |                - *'gdml'*          :  gradient-domain machine-learned potentials ( [sGDML]_ )
   |
   | Keywords for the 'anharmonic AS' potential:
   | 
   |          ``model_file``
   |
   |		 **Description:** Path to file with vibrational frequencies (in cm-1), Huang-Rhys
   |                 factors, non-adiabatic couplings and anharmonicities for each vibrational mode.
   |             **Datatype:** string
   |
   | Keywords for the 'harmonic' potential:
   |
   |          ``ground``
   |
   |             **Description:** Path to formatted checkpoint file from frequency calculation
   |                 at the ground state minimum.
   |             **Datatype:** string (path to fchk-file)
   |
   |          ``excited``
   |
   |             **Description:** Path to formatted checkpoint file from frequency calculation
   |                 at the excited state minimum.
   |             **Datatype:** string (path to fchk-file)
   |
   |          ``coupling``
   |
   |             **Description:** Path to formatted checkpoint file with the non-adiabatic coupling
   |                 vector between the ground and excited state.
   |             **Datatype:** string (path to fchk-file)
   |
   | Keywords for the 'gdml' potential:
   |
   |          ``ground``
   |
   |             **Description:** Path to sGDML model trained to reproduce ground state energies,
   |                 gradients and Hessians.
   |             **Datatype:** string (path to npz-file)
   |
   |          ``excited``
   |
   |             **Description:** Path to formatted checkpoint file from frequency calculation
   |                 at the excited state minimum.
   |             **Datatype:** string (path to fchk-file)
   |
   |          ``coupling``
   |
   |             **Description:** Path to formatted checkpoint file with the non-adiabatic coupling
   |                 vector between the ground and excited state. The atomic masses are taken from
   |                 this file.
   |             **Datatype:** string (path to fchk-file)


.. topic:: ``propagator``

   | **Description:** Name of the semiclassical propagator
   | **Datatype:** string
   |    - 'HK' :  Herman & Kluk propagator (see [HK]_)
   |    - 'WM' :  Walton & Manolopoulos propagator (see [WM]_)
   | **Default:** ``HK``

   
.. topic:: ``num_steps``

   | **Description:** Number of time steps for dynamics.
   | **Datatype:** integer

   
.. topic:: ``time_step_fs``

   | **Description:** Duration of a single time step in fs.
   | **Datatype:** float

   
.. topic:: ``num_trajectories``

   | **Description:** Total number of trajectories. ``batch_size`` trajectories are run in parallel.
   | **Datatype:** integer
   | **Default:** 50000

   
.. topic:: ``batch_size``

   | **Description:** ``batch_size`` trajectories are run in parallel.
   |     If memory is limited, the batch size should be reduced. 
   | **Datatype:** integer
   | **Default:** 10000

   
.. topic:: ``results``

   | **Description:** Controls how results of the dynamics calculation are stored on file.
   | **Datatype:** JSON
   | **Keywords:**
   |
   |          ``correlations``
   |
   |             **Description:** Name of file where results will be written to in npz-format.
   |                 This binary file can be read with numpy. It contains the autocorrelation and correlation function
   |                 for internal conversion on the equidistant grid specified by `num_steps` and `time_step_fs`.
   |             **Datatype:** string
   |             **Default:** 'correlations.npz'
   |
   |          ``overwrite``
   |
   |		 **Description:** If set to true an existing npz-file is overwritten.
   |                 Otherwise correlation functions from different runs are accumulated.
   |             **Datatype:** boolean
   |             **Default:** true

.. topic:: ``manual_seed``

   | **Description:** Initial values for positions and momenta are drawn randomly.
   |     To make the random numbers reproducible between runs,
   |     a manual seed for the random number generator can be provided.
   | **Datatype:** integer
   | **Default:** None
   | **Recommendation:** Avoid seeding the RNG manually.

   
=========================
Keywords for "rates" task
=========================

.. topic:: ``correlations``

   | **Description:** Converged correlation functions are read from this npz-file.
   | **Datatype:** string
   | **Default:** 'correlations.npz'

.. topic:: ``rates``

   | **Description:** Transition rates are written to this npz-file.
   | **Datatype:** string
   | **Default:** 'correlations.npz'

.. topic:: ``broadening``

   | **Description:** Lineshape function (*'gaussian'*, *'lorentzian'* or *'voigtian'*)
   | **Datatype:** string
   | **Default:** 'gaussian'

.. topic:: ``hwhmG_ev``

   | **Description:** Gaussian width of lineshape function in energy domain (in eV)
   | **Datatype:** float
   | **Default:** 0.01

.. topic:: ``hwhmL_ev``

   | **Description:** Lorentzian width of lineshape function in energy domain (in eV)
   | **Datatype:** float
   | **Default:** 1.0e-6
	   


--------
Examples
--------

==============================
with 'anharmonic AS' potential
==============================

.. code-block:: javascript

  { "semi" : [
    {
	"task" : "dynamics",
	"potential" : {
	    "type"          : "anharmonic AS",
	    "model_file"    : "AS_model.dat"
	},
	"propagator" : "HK",
	"batch_size"            : 10000,
	"num_trajectories"      : 50000,
	"num_steps"             : 2000,
	"time_step_fs"          : 0.005,
	"results" : {
	    "correlations"      : "correlations.npz",
	    "overwrite"         : false
	},
	"manual_seed"           : 0
    },
    {
        "task"  : "rates",
	"broadening"   : "gaussian",
	"hwhmG_ev"      : 0.001,
	"correlations" : "correlations.npz",
	"rates"        : "correlations.npz"
    }
  ]}

  
The file `AS_model.dat` contains the normal mode frequencies, Huang-Rhys factors, NACs
and anharmonicities for each mode. The initial potential differs from the final one by
a rigid displacement. 

The sign of the Huang-Rhys factor encodes the sign of the displacement:

.. math::
   
   \Delta Q = sign(S) \sqrt{2 |S|/\omega}


The initial potential is harmonic, the final one is a Morse potential with anharmonicity
CHI. The anharmonicities can be obtained in principle from the ratio of the
anharmonic to the harmonic frequencies as:

.. math::
   
   \chi = \frac{1}{2} (1 - \frac{\omega(anharmonic)}{\omega(harmonic)} )


.. raw:: html
	 
	 <details>
	 <summary>Example <b>AS_model.dat</b> file for 5 modes:</summary>
	 <pre>
	 # normal frequency            Huang-Rhys factor     Non-adiabatic coupling     Anharmonicity
	 # OMEGA_j / cm^-1                  |S_j|                (S0|d/dQj|S1)             CHI
	     +500.8809000000            +0.3474950080            -0.0000460805            0.02
	     +827.3282000000            +0.3824004553            +0.0000595520            0.02
	     +990.0261000000            -0.4168571687            -0.0000150425            0.02
	    +1351.1072000000            -0.0935664944            +0.0002054889            0.02
	    +3256.3099000000            +0.0033317953            +0.0000665122            0.02
	 </pre>
	 </details>

	 
=========================
with 'harmonic' potential
=========================

.. code-block:: javascript

  { "semi" : [
    {
	"task" : "dynamics",
	"potential" : {
	    "type"      : "harmonic",
	    "ground"    : "opt_freq_s0.fchk",
	    "excited"   : "opt_freq_s1.fchk",
	    "coupling"  : "opt_freq_s1.fchk"
	},
	"propagator" : "WM",
	"batch_size"            : 10000,
	"num_trajectories"      : 50000,
	"num_steps"             : 2000,
	"time_step_fs"          : 0.005,
	"results" : {
	    "correlations"      : "correlations.npz"
	}
    },
    {
        "task"  : "rates",
	"broadening"   : "gaussian",
	"hwhmG_ev"      : 0.001,
	"correlations" : "correlations.npz",
	"rates"        : "correlations.npz"
    }
  ]}

`opt_freq_s0.fchk` and `opt_freq_s1.fchk` are formatted checkpoint files from optimizations on S0 and S1, respectively, followed frequency and NAC (only for S1) calculations. You can use the script `trim_formatted_checkpoint_file.awk` to extracty the required fields from a formatted checkpoint file.
  
.. raw:: html

    <details>
    <summary>Example <b>opt_freq_s0.fchk</b>, harmonic S0 potential of methylium:</summary>
    <pre>
    optimize on S0 potential, frequencies                                   
    Freq      RwB97XD                                                     def2SVP             
    Number of atoms                            I                4
    Atomic numbers                             I   N=           4
	       1           6           1           1
    Current cartesian coordinates              R   N=          12
      4.45491930E-01  4.15739191E+00  9.45322479E-02 -1.63708697E+00  4.15740227E+00
      9.44766147E-02 -2.67842103E+00  5.96093482E+00  9.45328735E-02 -2.67841408E+00
      2.35386090E+00  9.44034888E-02
    Real atomic weights                        R   N=           4
      1.00782504E+00  1.20000000E+01  1.00782504E+00  1.00782504E+00
    Total Energy                               R     -3.943503734612899E+01
    Cartesian Gradient                         R   N=          12
      1.92317021E-06 -6.76080549E-07  7.69437388E-07  4.13362538E-06  2.24121132E-06
     -2.30920968E-06 -2.96661554E-06  3.90776059E-06  7.65128225E-07 -3.09018006E-06
     -5.47289136E-06  7.74644072E-07
    Cartesian Force Constants                  R   N=          78
      3.59316423E-01 -1.12159121E-06  4.41541878E-02  8.61581304E-06  8.77307199E-07
      2.00138511E-02 -3.57492724E-01  1.51668664E-06 -7.97868310E-06  6.28382302E-01
      1.13533618E-06 -6.14899388E-02 -7.24467215E-08 -4.42352665E-06  6.28129653E-01
     -7.33503579E-06 -5.84551234E-08 -5.96120727E-02  9.22299826E-06  1.59525818E-05
      1.78546614E-01 -9.11640326E-04  2.05014641E-02  4.12357010E-07 -1.35445741E-01
      1.28142779E-01  3.62971294E-06  1.22927984E-01 -3.92045005E-03  8.66793935E-03
     -6.56297014E-07  1.28131788E-01 -2.83321055E-01 -6.38897366E-06 -1.36427813E-01
      2.80357150E-01 -7.68196960E-07  2.02503349E-08  1.97990308E-02  3.95922771E-06
     -6.93829406E-06 -5.94669691E-02 -3.58695227E-06  7.89328491E-06  1.98211778E-02
     -9.12058917E-04 -2.05018592E-02 -1.04948695E-06 -1.35443837E-01 -1.28139490E-01
     -5.51767541E-06  1.34293977E-02  1.22164750E-02  3.95921521E-07  1.22926498E-01
      3.92043630E-03  8.66781166E-03 -1.48563447E-07 -1.28128881E-01 -2.83318659E-01
     -9.50515302E-06 -1.22164297E-02 -5.70403375E-03 -9.75241191E-07  1.36424875E-01
      2.80354881E-01 -5.12580499E-07 -8.39102571E-07  1.97991907E-02 -5.20354252E-06
     -8.94183992E-06 -5.94675720E-02 -4.55117569E-07 -8.48014663E-07  1.98467605E-02
      6.17124058E-06  1.06289571E-05  1.98216208E-02
    Gaussian Version                           C   N=           2
    ES64L-G16RevA.03        
    </pre>
    </details>

.. raw:: html

    <details>
    <summary>Example <b>opt_freq_s1.fchk</b>, harmonic S1 potential of methylium:</summary>
    <pre>
    optimize on S1 potential, frequencies, S0/S1 NAC vector                 
    Freq      RwB97XD TD-FC                                               def2SVP             
    Number of atoms                            I                4
    Atomic numbers                             I   N=           4
	       1           6           1           1
    Current cartesian coordinates              R   N=          12
      7.22778314E-01  4.15611802E+00 -3.21792826E-02 -1.27934073E+00  4.15849904E+00
      5.25786668E-01 -2.99598583E+00  5.49400875E+00 -5.88472233E-02 -2.99588189E+00
      2.82096408E+00 -5.68149376E-02
    Real atomic weights                        R   N=           4
      1.00782504E+00  1.20000000E+01  1.00782504E+00  1.00782504E+00
    Total Energy                               R     -3.925836216784926E+01
    Cartesian Gradient                         R   N=          12
     -3.43691605E-05 -3.20700523E-05 -2.80997555E-05 -1.88930942E-05  5.05119020E-05
      7.83269774E-06  1.47113919E-05 -5.98730725E-05  2.04582756E-05  3.85508629E-05
      4.14312228E-05 -1.91217821E-07
    Cartesian Force Constants                  R   N=          78
      3.39884947E-01 -3.61764181E-04  1.93783224E-02 -8.74056705E-02  8.68860085E-05
      3.48168470E-02 -3.42850585E-01  3.55000406E-04  1.03096412E-01  6.75939164E-01
      3.63253137E-04 -5.55944879E-03 -9.49841351E-05 -3.37830920E-04  8.58489739E-02
      7.80896289E-02 -1.04048086E-04 -4.07886771E-02  1.18874915E-03  2.95085211E-05
      8.86823153E-02  1.50791922E-03  2.36160154E-02 -7.86646682E-03 -1.66676520E-01
      4.66733904E-02 -3.97129280E-02  1.45219551E-01 -2.68895345E-03 -6.88588851E-03
     -3.57156844E-03  8.43077698E-02 -4.02250454E-02  3.47377446E-02 -7.60115523E-02
      8.40917333E-02  4.66580855E-03 -1.68616973E-03  2.98968710E-03 -5.22208363E-02
      1.80225651E-02 -2.39935585E-02  3.37589811E-02 -2.38132057E-02  1.52223647E-02
      1.45771850E-03 -2.36092516E-02 -7.82427432E-03 -1.66412060E-01 -4.66988127E-02
     -3.95654501E-02  1.99490493E-02 -5.60726413E-03  1.37960467E-02  1.45005292E-01
      2.68746449E-03 -6.93298512E-03  3.57966657E-03 -8.43249393E-02 -4.00644797E-02
     -3.46632050E-02  5.72214640E-03 -3.69807994E-02  7.47681033E-03  7.59153284E-02
      8.39782642E-02  4.65023305E-03  1.70333181E-03  2.98214303E-03 -5.20643245E-02
     -1.79570895E-02 -2.39000797E-02  1.38204137E-02 -7.35297046E-03  5.78150669E-03
      3.35936777E-02  2.36067281E-02  1.51364300E-02
    Nonadiabatic coupling                      R   N=          12
      7.69794074E-05  1.35949721E-01 -2.47294349E-04 -4.57705419E-04 -1.68634768E-02
     -5.29196540E-04  1.73570747E-01 -5.58179958E-02 -5.22702665E-01 -1.73191088E-01
     -5.55348660E-02  5.23407463E-01
    Gaussian Version                           C   N=           2
    ES64L-G16RevA.03        
    </pre>
    </details>


=====================
with 'gdml' potential
=====================

.. code-block:: javascript

  { "semi" : [
    {
	"task" : "dynamics",
	"potential" : {
	    "type"      : "gdml",
	    "ground"    : "pot_s0.npz",
	    "excited"   : "opt_freq_s1.fchk",
	    "coupling"  : "opt_freq_s1.fchk"
	},
	"propagator" : "WM",
	"batch_size"            : 10000,
	"num_trajectories"      : 50000,
	"num_steps"             : 2000,
	"time_step_fs"          : 0.005,
	"results" : {
	    "correlations"      : "correlations.npz"
	}
    },
    {
        "task"  : "rates",
	"broadening"   : "gaussian",
	"hwhmG_ev"      : 0.001,
	"correlations" : "correlations.npz",
	"rates"        : "correlations.npz"
    }
  ]}

`pot_s0.npz` is an sGDML model trained to reproduce the ground state potential.
To verify that the sGDML model fits the ground state potential accurately,
the harmonic normal mode frequencies and displacement vectors can be compared
visually with those stored in a Gaussian 16 formatted checkpoint file:

.. code-block:: bash

   $ sgdml_compare_normal_modes.py  opt_freq_s0.fchk  pot_s0.npz

   
If `scan.fchk` contains the results of a relaxed scan (using `Opt=ModRedundant` in Gaussian 16)
the ab initio and sGDML energies can be compared with:
   
.. code-block:: bash

   $ sgdml_compare_relaxed_scan.py  scan.fchk  pot_s0.npz


----------
References
----------

.. [HK] E. Kluk, M. Herman, H. Davis,
   *Comparison of the propagation of semiclassical frozen Gaussian wave functions with quantum propagation for a highly excited anharmonic oscillator*
   J. Chem. Phys. 84, 326, (1986)
   https://doi.org/10.1063/1.450142
       
.. [WM] A. Walton, D. Manolopoulos,
   *A new semiclassical initial value method for Franck-Condon spectra*
   Mol. Phys. 87, 961-978, (1996)
   https://doi.org/10.1080/00268979600100651
     
.. [sGDML] S. Chmiela et al.
   *sGDML: Constructing Accurate and Data Efficient Molecular Force Fields Using Machine Learning.*
   Comput. Phys. Commun. 240, 38-45 (2019)
   https://doi.org/10.1016/j.cpc.2019.02.007 
   https://github.com/stefanch/sGDML
   
.. [anharmAS] A. Humeniuk et al.
   *Predicting fluorescence quantum yields for molecules in solution: A critical assessment of the harmonic approximation and the choice of the lineshape function*
   J. Chem. Phys. 152, 054107 (2020)
   https://doi.org/10.1063/1.5143212


   
