
Internal Conversion Rates from Semiclassical Dynamics
-----------------------------------------------------

The long term goal of this project is to compute rates for internal conversion in medium sized molecules
from semiclassical molecular dynamics.

Milestones
----------

 * implement Herman-Kluk and Walton-Manolopoulos semiclassical propagators (*done*)
 * compare with exact QM rates for _anharmonic_ adiabatic shift (AS) model with 5,10,...,60 modes
 * compare with exact QM rates for different _harmonic_ models (AH, VH)
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


Installation
------------
The package is installed by running

.. code-block:: bash

   $ pip install -e .
   
in the top directory. Calculations are run via the command line interface `semi`.

- running semiclassical dynamics:

.. code-block:: bash

   $ semi dynamics input.json

- plotting correlation functions:

.. code-block::

   $ semi plot correlations.npz



---------------
Getting Started
---------------

All calculations are controlled by the settings in a command file which is
structured using the JSON format.

=======================
Structure of Input File
=======================

.. code-block:: javascript

   { "semi" : [
      {
      	"title" : "internal conversion",
	... keywords ...
      },		
      {		
	"title" : "internal conversion",
	... keywords ...
      },
      ...
   ]}


`title` determines the type of calculation. So far only correlation functions for
"internal conversion" are implemented. This type of calculation requires the keywords listed below.
See also the examples for input files at the end.

========
Keywords
========
    
.. topic:: ``potential``

   | **Description:** Block defining the potential.
   | **Datatype:** JSON
   | **Keywords:**
   |
   |          ``type``
   |
   |             **Description:** Chooses type of potential. Different potentials have different keywords.
   |             **Datatype:** string
   |                - *'anharmonic AS'* :  anharmonic adiabatic shift model
   |                - *'harmonic'*      :  The molecular potential of the ground and excited state
   |                                       are expanded harmonically around equilibrium geometries.
   |                - *'gdml'*          :  gradient-domain machine-learned potentials ( [sGDML]_ )
   |
   | Keywords for the 'anharmonic AS' potential:
   | 
   |          ``model_file``
   |
   |		 **Description:** Path to file with vibrational frequencies (in cm-1), Huang-Rhys
   |                 factors and non-adiabatic couplings for each vibrational mode.
   |             **Datatype:** string
   |
   |          ``anharmonicity``
   |
   |             **Description:** Degree of anharmonicity chi (0 : harmonic, > 0.0 : anharmonic)
   |             **Datatype:** float
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
   |                 vector between the ground and excited state.
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
  

--------
Examples
--------

==============================
with 'anharmonic AS' potential
==============================

.. code-block:: javascript

  { "semi" : [
    {
	"title" : "internal conversion",
	"potential" : {
	    "type"          : "anharmonic AS",
	    "model_file"    : "AS_model.dat",
	    "anharmonicity" : 0.02
	},
	"propagator" : "HK",
	"batch_size"            : 10000,
	"num_trajectories"      : 50000,
	"num_steps"             : 10000,
	"time_step_fs"          : 0.001,
	"results" : {
	    "correlations"      : "correlations.npz",
	    "overwrite"         : false
	},
	"manual_seed"           : 0
    }
  ]}


=========================
with 'harmonic' potential
=========================

.. code-block:: javascript

  { "semi" : [
    {
	"title" : "internal conversion",
	"potential" : {
	    "type"      : "harmonic",
	    "ground"    : "opt_freq_s0.fchk",
	    "excited"   : "opt_freq_s1.fchk",
	    "coupling"  : "opt_freq_s1.fchk"
	},
	"propagator" : "HK",
	"batch_size"            : 1000,
	"num_trajectories"      : 2000,
	"num_steps"             : 10,
	"time_step_fs"          : 0.001,
	"results" : {
	    "correlations"      : "correlations.npz"
	}
    }
  ]}

  
=====================
with 'gdml' potential
=====================

.. code-block:: javascript

  { "semi" : [
    {
	"title" : "internal conversion",
	"potential" : {
	    "type"      : "gdml",
	    "ground"    : "pot_s0.npz",
	    "excited"   : "opt_freq_s1.fchk",
	    "coupling"  : "opt_freq_s1.fchk"
	},
	"propagator" : "HK",
	"batch_size"            : 1000,
	"num_trajectories"      : 2000,
	"num_steps"             : 10,
	"time_step_fs"          : 0.001,
	"results" : {
	    "correlations"      : "correlations.npz"
	}
    }
  ]}


----------
References
----------

.. [HK] E. Kluk, M. Herman, H. Davis,
    "Comparison of the propagation of semiclassical frozen Gaussian wave functions with quantum propagation for a highly excited anharmonic oscillator",
    J. Chem. Phys. 84, 326, (1986)

.. [WM] A. Walton, D. Manolopoulos,
     "A new semiclassical initial value method for Franck-Condon spectra",
     Mol. Phys. 87, 961-978, (1996)

.. [sGDML]  https://github.com/stefanch/sGDML
