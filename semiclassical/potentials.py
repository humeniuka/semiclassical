# coding: utf-8
"""potential energy surfaces"""

__all__ = ['NonHarmonicPotential', 'MorsePotential']


# # Imports
import torch
import logging

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
        
    def energy(self, r):
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
    
    def gradient(self, r):
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
    
    def hessian(self, r):
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
        
    def energy(self, r):
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
    
    def gradient(self, r):
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
    
    def hessian(self, r):
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

