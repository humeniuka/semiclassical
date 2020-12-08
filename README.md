
The long term goal of this project is to compute rates for internal conversion in medium sized molecules
from semiclassical molecular dynamics.

Milestones
----------
 * implement Herman-Kluk and Walton-Manolopoulos semiclassical propagators (*done*)
 * compare with exact QM rates for _anharmonic_ adiabatic shift (AS) model with 5,10,...,60 modes
 * compare with exact QM rates for different _harmonic_ models (AH, VH)
 * include Herzberg-Teller effect (coordinate dependence of non-adiabatic coupling vectors)
 * use TorchANI potential energies in ground state propagation



Complex numbers are not supported in older releases of pytorch, 
this code requires pytorch=1.7.0 or higher

References
----------
[2] E. Kluk, M. Herman, H. Davis,
    "Comparison of the propagation of semiclassical frozen Gaussian wave functions with quantum propagation for a highly excited anharmonic oscillator",
    J. Chem. Phys. 84, 326, (1986)
    
[3] J. Brickmann and P. Russegger,
    "Quasidecay of coherent states",
    J. Chem. Phys. 75, 12, (1981)
    
[4] Kenneth G. Kay, 
    "Semiclassical propagation for multidimensional systems by an initial value method",
    J. Chem. Phys. 101, 2250 (1994)
