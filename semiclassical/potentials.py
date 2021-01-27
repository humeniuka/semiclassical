# coding: utf-8
"""potential energy surfaces"""

__all__ = ['NonHarmonicPotential', 'MorsePotential',
           'MolecularHarmonicPotential', 'MolecularGDMLPotential']


# # Imports
import torch
import logging
from ase.data import atomic_masses

# # Local Imports
from semiclassical import units
from semiclassical.gdml_predictor import GDMLPredict

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(module)-12s] %(message)s", level=logging.INFO)


# # Potential Energy Surfaces


class NonHarmonicPotential(object):
    """
    non-harmonic potential, eps*Morse + (1-eps)*(harmonic oscillator), see eqn. (7)
    
      V(x) = eps  * 1/(2 b^2) (1 - exp(-b*x))^2 + (1-eps) 1/2 x^2
    
    with eps = 0.975 and b = (12)^{-1/2}

    see eqn. (7) in Herman-Kluk paper
    """
    def __init__(self, 
                 eps=torch.tensor([0.975]), 
                 b=torch.tensor([(12.0)**(-0.5)])):
        self.eps = eps
        self.b = b

    def dimensions(self):
        """
        number of nuclear degrees of freedom
        
        Returns
        -------
        dim     :   int
        """
        dim = self.eps.size()[0]
        return dim

    def masses(self):
        """
        masses (in atomic units) for each degree of freedom
        
        Returns
        -------
        masses  :  real Tensor (dim,)
        """
        dim = self.eps.size()[0]
        return torch.ones(dim)
        
    def _energy(self, r):
        """
        evaluate potential energy V(r)
        
        Parameters
        ----------
        r  :  real Tensor (dim,n)
          batch of position vectors
        
        Returns
        -------
        v  :  real Tensor (n,)
          potential energies
        """
        eps = self.eps.to(r.device).unsqueeze(0).expand_as(r)
        b   = self.b.to(r.device).unsqueeze(0).expand_as(r)
        
        v = eps/(2*b**2) * (1.0 - torch.exp(- b*r))**2 + (1-eps)*0.5*r**2
        vpot = torch.sum(v, 0)
        return vpot
    
    def _gradient(self, r):
        """
        evaluate gradient of potential energy dV/dr
        
        Parameters
        ----------
        r  :  real Tensor (dim,n)
          batch of position vectors
        
        Returns
        -------
        grad  :  real Tensor (dim,n)
          batch of gradient vectors
        """
        eps = self.eps.to(r.device).unsqueeze(0).expand_as(r)
        b   = self.b.to(r.device).unsqueeze(0).expand_as(r)
        
        grad = eps/b * (torch.exp(-b*r) - torch.exp(-2*b*r)) + (1-eps)*r
        
        return grad
    
    def _hessian(self, r):
        """
        evaluate Hessian of potential energy surface
                   d^2 V
          H   = -----------
           ij   dr(i) dr(j)
           
        Parameters
        ----------
        r  :  real Tensor (dim,n)
          batch of position vectors
        
        Returns
        -------
        hess  :  real Tensor (dim,dim,n)
          batch of Hessian matrices
        """
        eps = self.eps.to(r.device).unsqueeze(0).expand_as(r)
        b   = self.b.to(r.device).unsqueeze(0).expand_as(r)

        dim,n = r.size()
        
        hess = torch.zeros((dim,dim,n)).to(r.device)
        hess_diag = torch.diagonal(hess, dim1=0, dim2=1)

        hess_diag[...] = torch.transpose(
            eps*(2*torch.exp(-2*b*r) - torch.exp(-b*r)) + (1-eps),
                             0,1 )
            
        return hess

    def harmonic_approximation(self, r):
        """
        local harmonic approximation to the potential energy surface around a point r:

           V(r') = V(r) + [grad V(r)]^T . (r'-r) + 1/2 (r'-r)^T . [hess V(r)] . (r'-r)

        This function computes the energies, gradients and Hessians for a batch of geometries.

        Parameters
        ----------
        r  :  real Tensor (dim,n)
          batch of position vectors

        Returns
        -------
        vpot  :  real Tensor (n,)
          potential energies
        grad  :  real Tensor (dim,n)
          batch of gradient vectors
        hess  :  real Tensor (dim,dim,n)
          batch of Hessian matrices
        """
        vpot = self._energy(r)
        grad = self._gradient(r)
        hess = self._hessian(r)
        
        return vpot, grad, hess
    
    def derivative_coupling_1st(self, r):
        """
        first order derivative non-adiabatic coupling
        
               (1)           d
            tau   = <ground|----- excited>        k=1,...,d
               k            dx(k)
               
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau1  :  real Tensor (d,*)
          derivative coupling vector tau1(r)
        """
        tau1 = torch.ones_like(r)
        return tau1
    
    def derivative_coupling_2nd(self, r):
        """
        second order derivative non-adiabatic coupling
          
               (2)           d^2
            tau   = <ground|------- excited>       k=1,...,d
               k            dx(k)^2
        
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau2  :  real Tensor (d,*)
          2nd order derivative coupling tau2(r)   
        """
        tau2 = torch.zeros_like(r)
        return tau2



class MorsePotential(object):
    """
    Morse potential with anharmonicity \chi as in eqn. (6) of 
    https://doi.org/10.1063/1.5143212
                                                                                                                                                                                                          
    Potential of ground state                                                                                                                                                                                
    The Morse potential                                                                                                                                                                                      
                                     2                                                                                                                                                                         
       V(r) = D ( 1 - exp(-a*r) )                                                                                                                                                                          
       
    has the eigenenergies                                                                                                                                                                                    
                                                                                                                                                                                                             
       E_n = [(n+1/2) - chi (n+1/2)^2] omega                                                                                                                                                                   
                                                                                                                                                                                                             
    The parameters `D` and `a` can be expressed in terms of the frequency                                                                                                                                    
    omega and the anharmonicity `chi`                                                                                                                                                                          
                                                                                                                                                                                                              
       a = sqrt(2 * omega * chi)                                                                                                                                                                           
       
       D = 1/4 * omega/chi                                                                                                                                                                                     
    """
    def __init__(self, omega, chi, nac):
        """
        Parameters
        ----------
        omega  :   float Tensor (dim,)
          vibrational frequencies of each mode in Hartree
        chi    :   float Tensor (dim,)
          anharmonicity of each mode
        nac    :   float Tensor (dim,)
          non-adiabatic coupling vector
        """
        self.omega = omega
        self.chi = chi
        self.nac = nac
        
        if (self.chi == 0.0).all():
            logger.info("Potential is harmonic.")
        
        self.a = torch.sqrt(2*omega*chi)
        self.D = 0.25 * omega/chi

    def dimensions(self):
        dim = self.a.size()[0]
        return dim

    def masses(self):
        dim = self.a.size()[0]
        return torch.ones(dim)
        
    def _energy(self, r):
        """V(r)"""
        if (self.chi == 0.0).all():
            # potential is harmonic
            omega = self.omega.to(r.device).unsqueeze(1).expand_as(r)
            v = 0.5 * omega**2 * r**2
            vpot = torch.sum(v, 0)
            return vpot
        else:
            # anharmonic Morse potential
            a = self.a.to(r.device).unsqueeze(1).expand_as(r)
            D = self.D.to(r.device).unsqueeze(1).expand_as(r)
        
            v = D * (1.0 - torch.exp(-a*r))**2
            vpot = torch.sum(v, 0)
            return vpot
    
    def _gradient(self, r):
        """dV/dr"""
        if (self.chi == 0.0).all():
            # harmonic potential
            omega = self.omega.to(r.device).unsqueeze(1).expand_as(r)
            grad = omega**2 *r
            return grad
        else:
            # anharmonic Morse potential
            a = self.a.to(r.device).unsqueeze(1).expand_as(r)
            D = self.D.to(r.device).unsqueeze(1).expand_as(r)
        
            grad = 2*a*D*torch.exp(-a*r)*(1.0-torch.exp(-a*r))
        
            return grad
    
    def _hessian(self, r):
        """d^2(V)/dr(i)dr(j)"""
        if (self.chi == 0.0).all():
            # harmonic potential
            omega = self.omega.to(r.device).unsqueeze(1).expand_as(r)
            
            dim,n = r.size()
            
            hess = torch.zeros((dim,dim,n)).to(r.device)
            hess_diag = torch.diagonal(hess, dim1=0, dim2=1)
            
            hess_diag[...] = torch.transpose(
                omega**2,
                                 0,1 )
            return hess
        else:
            # anharmonic Morse potential
            a = self.a.to(r.device).unsqueeze(1).expand_as(r)
            D = self.D.to(r.device).unsqueeze(1).expand_as(r)

            dim,n = r.size()

            hess = torch.zeros((dim,dim,n)).to(r.device)
            hess_diag = torch.diagonal(hess, dim1=0, dim2=1)

            hess_diag[...] = torch.transpose(
                2*a**2*D*torch.exp(-a*r)*(2*torch.exp(-a*r)-1.0),
                                 0,1 )
            
            return hess

    def harmonic_approximation(self, r):
        """
        local harmonic approximation to the potential energy surface around a point r:

           V(r') = V(r) + [grad V(r)]^T . (r'-r) + 1/2 (r'-r)^T . [hess V(r)] . (r'-r)

        This function computes the energies, gradients and Hessians for a batch of geometries.

        Parameters
        ----------
        r  :  real Tensor (dim,n)
          batch of position vectors

        Returns
        -------
        vpot  :  real Tensor (n,)
          potential energies
        grad  :  real Tensor (dim,n)
          batch of gradient vectors
        hess  :  real Tensor (dim,dim,n)
          batch of Hessian matrices
        """
        vpot = self._energy(r)
        grad = self._gradient(r)
        hess = self._hessian(r)
        
        return vpot, grad, hess
        
    def derivative_coupling_1st(self, r):
        """
        first order derivative non-adiabatic coupling
        
               (1)           d
            tau   = <ground|----- excited>        k=1,...,d
               k            dx(k)
               
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau1  :  real Tensor (d,*)
          derivative coupling vector tau1(r)
        """
        tau1 = self.nac.to(r.device).unsqueeze(1).expand_as(r)
        return tau1
    
    def derivative_coupling_2nd(self, r):
        """
        second order derivative non-adiabatic coupling
          
               (2)           d^2
            tau   = <ground|------- excited>       k=1,...,d
               k            dx(k)^2
        
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau2  :  real Tensor (d,*)
          2nd order derivative coupling tau2(r)   
        """
        tau2 = torch.zeros_like(r)
        return tau2

# for mixing in some methods specifically for molecules 
class _MolecularPotentialBase(object):
    _masses, _dim = None, None
    # Energies are measured relative to this origin.
    _origin = 0.0
    def dimensions(self):
        """
        number of nuclear degrees of freedom
        
        Returns
        -------
        dim     :   int
        """
        return self._dim

    def masses(self):
        """
        masses (in multiples of electron mass) for each degree of freedom
        
        Returns
        -------
        masses  :  real Tensor (dim,)
        """
        return self._masses

    def minimize(self, r_guess, maxiter=10000, rtol=1.0e-5, gtol=1.0e-8):
        """
        find the local minimum of the potential energy surface in the vicinity
        of `r_guess`. 

        The origin of the energy axis is shifted so that at the minimum E=0,
        henceforth all energies are measured relative to the bottom of the potential.

        Parameters
        ----------
        r_guess  :  real Tensor (dim,)
          minimization starts at this position

        Optional
        --------
        maxiter  :  int
          maximum number of optimization steps
        rtol     :  float (bohr)
          optimization stops when geometry changes by < ftol
        gtol     :  float (Hartree bohr^-1)
          optimization stops when gradient norm is < gtol

        Returns
        -------
        nothing, but sets `self._origin`
        """
        # unset origin of energy axis
        self._origin = 0.0
        # find minimum of potential energy surface
        r = r_guess.unsqueeze(1)
        for i in range(0, maxiter):
            energy, grad, hess = self.harmonic_approximation(r)

            # The next geometry r' is obtained by minimizing the harmonic model potential
            # at the current geometry r:
            #  V(r') = V(r) + [grad V(r)](r'-r) + 1/2 (r'-r)^T [hess V(r)] (r'-r)
            # The condition for a minimum
            #  dV/dr(r') = 0
            # leads to
            #  r' = r + [hess V(r)]^{-1} (- [grad V(r)])
            # solve AX = B
            dr, _ = torch.solve(-grad, hess.squeeze())

            # directional derivative   dE = dE/dr * dr
            delta_energy = torch.sum(grad*dr)

            if delta_energy > 0.0:
                # dr is not a descent direction, take a steepest descent step
                dr = -grad
                delta_energy = torch.sum(grad*dr)
            
            # stop if gradient d(energy)/dr is small enough or position does not change anymore
            grad_norm = torch.norm(grad)
            disp_norm = torch.norm(dr)
            logger.info(f"  iteration= {i:5}  energy= {energy.item():f} Hartree  |gradient|= {grad_norm:e} (threshold= {gtol} )  |geometry change|= {disp_norm:e} (threshold= {rtol} )")
            if (grad_norm < gtol) or (disp_norm < rtol):
                logger.info("  converged")
                break

            # perform a line search along the search direction `dr` using the
            # Armijo backtracking algorithm, see Algorithm 3.1 in
            # J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006
            
            # parameters for line search
            rho = 0.3
            c = 0.0001
            lmax = 100

            # find optimal step size a*dr
            a = 1.0
            for l in range(0, lmax):
                r_interp = r + a*dr
                energy_interp, _, _ = self.harmonic_approximation(r_interp)
                if energy_interp <= energy + c*a*delta_energy:
                    # found step size that leads to a sufficient decrease in energy
                    break
                else:
                    # reduce step size
                    a *= rho
            else:
                raise RuntimeError("Linesearch failed! Could not find a step length that satisfies the sufficient decrease condition.")
                    
            # move to next geometry
            r = r_interp
            
        else:
            raise RuntimeError(f"Could not find minimum within {maxiter} iterations.")

        # energy at minimum
        emin, _, _ = self.harmonic_approximation(r)
        self._origin = emin.item()
        logger.info(f"shift origin of energy axis to minimum energy = {self._origin} Hartree ")

        
class MolecularHarmonicPotential(_MolecularPotentialBase, object):
    """
    harmonic expansion around a reference geometry (usually minimum):

      V(r') = V0(r) + [grad V0]^T . (r'-r) + 1/2 (r'-r)^T [hess V0] (r'-r)

    Parameters
    ----------
    freq_fchk  :  FormattedCheckpointFile
      formatted checkpoint file object with cartesian geometry, gradient
      and cartesian force constants (from a frequency calculation at the
      reference geometry).
    nac_fchk   :  FormattedCheckpointFile
      formatted checkpoint file object with cartesian non-adiabatic coupling vector
    """
    def __init__(self, freq_fchk, nac_fchk):
        self.pos0, self.energy0, self.grad0, self.hess0 = (torch.from_numpy(x)
                                                           for x in
                                                           freq_fchk.harmonic_approximation())
        self.nac0 = torch.from_numpy(nac_fchk.nonadiabatic_coupling())
        self._masses = torch.from_numpy(freq_fchk.masses())
        self._dim = len(self._masses)
        logger.info(f"atomic masses (multiples of electron mass) : {self._masses}")
        
    def harmonic_approximation(self, r):
        """
        local harmonic approximation to the potential energy surface around a point r:

           V(r') = V(r) + [grad V(r)]^T . (r'-r) + 1/2 (r'-r)^T . [hess V(r)] . (r'-r)

        This function computes the energies, gradients and Hessians for a batch of geometries.

        Parameters
        ----------
        r  :  real Tensor (dim,n)
          batch of position vectors

        Returns
        -------
        vpot  :  real Tensor (n,)
          potential energies
        grad  :  real Tensor (dim,n)
          batch of gradient vectors
        hess  :  real Tensor (dim,dim,n)
          batch of Hessian matrices
        """
        if (self.energy0.device != r.device):
            self.energy0 = self.energy0.to(r.device)
            self.pos0 = self.pos0.to(r.device)
            self.grad0 = self.grad0.to(r.device)
            self.hess0 = self.hess0.to(r.device)

        dim,n = r.size()
        # r'-r
        dr = r - self.pos0.unsqueeze(1).expand_as(r)

        # expand around new point r'
        vpot = (  self.energy0
                + torch.einsum('in,i->n', dr, self.grad0)
                + 0.5 * torch.einsum('in,ij,jn->n', dr, self.hess0, dr) )
        grad = (  self.grad0.unsqueeze(1).expand_as(r)
                + torch.einsum('ij,jn->in', self.hess0, dr))
        hess = self.hess0.unsqueeze(2).expand(-1,-1,n)
        
        return vpot-self._origin, grad, hess

    def derivative_coupling_1st(self, r):
        """
        first order derivative non-adiabatic coupling
        
               (1)           d
            tau   = <ground|----- excited>        k=1,...,d
               k            dx(k)
               
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau1  :  real Tensor (d,*)
          derivative coupling vector tau1(r)
        """
        if (self.nac0.device != r.device):
            self.nac0 = self.nac0.to(r.device)

        tau1 = self.nac0.unsqueeze(1).expand_as(r)
        return tau1
    
    def derivative_coupling_2nd(self, r):
        """
        second order derivative non-adiabatic coupling
          
               (2)           d^2
            tau   = <ground|------- excited>       k=1,...,d
               k            dx(k)^2
        
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau2  :  real Tensor (d,*)
          2nd order derivative coupling tau2(r)   
        """
        tau2 = torch.zeros_like(r)
        return tau2
        
    
class MolecularGDMLPotential(_MolecularPotentialBase, object):
    """
    machine-learned molecular potential (symmetric gradient-domain ML, sGDML)

    Parameters
    ----------
    model_pot : NpzFile or mapping
      Data for sGDML model fitted to reproduce the ground state potential energy surface.
      It is assumed that the model uses atomic units (bohr for lengths and Hartree for energies).
    nac_fchk  :  FormattedCheckpointFile
      formatted checkpoint file object with cartesian non-adiabatic coupling vector
    """
    def __init__(self, model_pot, nac_fchk):
        # predict energy, gradient and Hessian of ground state potential
        self.gdml_pot = GDMLPredict(model_pot)
        # constant non-adiabatic coupling vector (Condon approximation)
        self.nac0 = torch.from_numpy(nac_fchk.nonadiabatic_coupling())

        assert (model_pot['z'] == nac_fchk.atomic_numbers()).all(), "GDML models for potential energy and NAC vector should be for the same molecule."
        # mass in atomic units for each cartesian coordinate
        self._masses = (torch.tensor([atomic_masses[z]*units.amu_to_aumass for z in model_pot['z']])
                        .repeat(3))
        self._dim = len(self._masses)
        logger.info(f"atomic masses (multiples of electron mass) : {self._masses}")
        
    def harmonic_approximation(self, r):
        """
        local harmonic approximation to the potential energy surface around a point r:

           V(r') = V(r) + [grad V(r)]^T . (r'-r) + 1/2 (r'-r)^T . [hess V(r)] . (r'-r)

        This function computes the energies, gradients and Hessians for a batch of geometries.

        Parameters
        ----------
        r  :  real Tensor (dim,n)
          batch of position vectors

        Returns
        -------
        vpot  :  real Tensor (n,)
          potential energies
        grad  :  real Tensor (dim,n)
          batch of gradient vectors
        hess  :  real Tensor (dim,dim,n)
          batch of Hessian matrices
        """
        if (self.gdml_pot.device != r.device):
            self.gdml_pot.to(r.device)

        # GDMLPredict expects the first axis to be the batch dimension,
        # while in the propagators the last dimension is the batch dimension,
        # so we have to change the order of the axes.
        vpot, grad, hess = self.gdml_pot.forward(r.permute(1,0))
        
        return vpot-self._origin, grad.permute(1,0), hess.permute(1,2,0)

    def derivative_coupling_1st(self, r):
        """
        first order derivative non-adiabatic coupling
        
               (1)           d
            tau   = <ground|----- excited>        k=1,...,d
               k            dx(k)
               
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau1  :  real Tensor (d,*)
          derivative coupling vector tau1(r)
        """
        if (self.nac0.device != r.device):
            self.nac0 = self.nac0.to(r.device)

        tau1 = self.nac0.unsqueeze(1).expand_as(r)
        return tau1
    
    def derivative_coupling_2nd(self, r):
        """
        second order derivative non-adiabatic coupling
          
               (2)           d^2
            tau   = <ground|------- excited>       k=1,...,d
               k            dx(k)^2
        
        Parameters
        ----------
        r  :  real Tensor (d,*)
          nuclear coordinates
        
        Returns
        -------
        tau2  :  real Tensor (d,*)
          2nd order derivative coupling tau2(r)   
        """
        tau2 = torch.zeros_like(r)
        return tau2
