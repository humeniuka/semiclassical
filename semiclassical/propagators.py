# coding: utf-8
"""semiclassical propagators"""

__all__ = ['HermanKlukPropagator', 'WaltonManolopoulosPropagator', 'hbar']


# # Imports
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import logging

# Atomic units are used throughout
#  hbar = h/(2pi) = 1
#  mass_electron = 1
hbar = 1.0


# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(module)-12s] %(message)s", level=logging.INFO)


# # Linear Algebra

def _triangular_det(T):
    """
    compute the determinant of a triangular matrix as the product
    of the diagonal elements
    
    Parameters
    ----------
    T    :   Tensor (m,n,n)
       batch of `m` upper or lower triangular `n` x `n` matrices
       
    Returns
    -------
    det  :   Tensor (m,)
       determinants, det[m] = det(T[m,:,:])
    """
    det = torch.prod(torch.diagonal(T, dim1=1, dim2=2), 1)
    return(det)
    
def _complex_det(A):
    """
    determinants of a batch of complex matrices
    
    Parameters
    ----------
    A    :  Tensor (m,n,n)
       batch of `m` complex n x n matrices
    
    Returns
    -------
    det  :  Tensor (m,)
       determinants, det[m] = det(A[m,:,:])
    """
    # 
    n,n = A.size()[-2:]
    if n == 1:
        # The determinant of a scalar is the identity
        return A.squeeze()
    
    # LU decomposition of matrix A
    LU, pivots = A.lu()
    P, L, U = torch.lu_unpack(LU, pivots)
    # L and U are lower and upper triangular matrices. The determinant of a triangular
    # matrix is just the product of the diagonal elements. 
    # det(A) = det(L) * det(U) * det(P)
    det = _triangular_det(L) * _triangular_det(U) * torch.det(P.real)
    #Could probably calculate det(P) [which is +-1] efficiently using Sylvester's determinant identity
    return det


# # Fourth-Order Runge-Kutta Integrator

def _rk4_step(f, y, t, h, *args):
    """

    Propagate a solution of the differential equation

      y' = f(t,y, args)

    for one time step from t to t+h with the 4th order Runge-Kutta method.

    Parameters
    ----------
    f  :  callable
      function giving derivative y'
    y  :  Tensor
      solution at initial time, y(t)
    t  :  float
      time
    h  :  float
      time steps
    args : additional arguments passed to `f`

    Returns
    -------
    y1 : Tensor
      solution at end of time step, y(t+h)

    """
    # displacements
    k1 = f(t,         y,            *args)
    k2 = f(t + 0.5*h, y + 0.5*h*k1, *args)
    k3 = f(t + 0.5*h, y + 0.5*h*k2, *args)
    k4 = f(t +     h, y +     h*k3, *args)
    #
    return y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)


# # Coherent States

class CoherentStatesOverlap(object):
    def __init__(self, Gi, Gj):
        """
        class for evaluating overlap integrals 
        
          O  = <qi,pi,Gi|qj,pj,Gj> 
           ij
         
        between batches of coherent states with position representation
        
                       det(G) 1/4               T                  T
          <x|q,p,G> = (------)    exp(-1/2 (x-q) G (x-q) + i/hbar p (x-q))
                        pi^N
                      
        Parameters
        ----------
        Gi   :  real Tensor (dim,dim)
          width matrix of bra coherent states
        Gj   :  real Tensor (dim,dim)
          width matrix of ket coherent states
        """
        assert Gi.size() == Gj.size(), "width matrices Gi and Gj have to have the same shape"
        self.Gi = Gi
        self.Gj = Gj
        
        self.detGi = torch.det(Gi)
        self.detGj = torch.det(Gj)
        
        self.Gij = Gi+Gj
        self.iGij = torch.inverse(self.Gij)
        self.detGij = torch.det(self.Gij)
        
        #             -1
        # Gi . [Gi+Gj]  . Gj
        self.Gi_iGij_Gj = self.Gi @ self.iGij @ self.Gj
        #             -1
        # Gj . [Gi+Gj]
        self.Gj_iGij = self.Gj @ self.iGij
        
        self.dim = self.Gi.size()[0]

    def __call__(self,qi,pi, qj,pj):
        """
        overlap matrix between two sets of coherent states
        
         O  = <qi(i),pi(i),Gi|qj(j),pj(j),Gj> 
          ij
          
        The bra and ket coherent states have width parameter matrices `Gi` and `Gj`,
        respectively.
        
        Parameters
        ----------
        qi,pi  :  real Tensor (dim,ni)
          positions and momenta of bra coherent states
        qj,pj  :  real Tensor (dim,nj)
          positions and momenta of ket coherent states
          
        Returns
        -------
        olap   :  complex Tensor (ni,nj)
          overlap matrix
        """
        assert qi.size()[0] == pi.size()[0] == self.dim, "dimension of phase space points (qi, pi) is wrong"
        assert qj.size()[0] == pj.size()[0] == self.dim, "dimension of phase space points (qj, pj) is wrong"
        if len(qi.size()) == 1:
            # If qi,pi contain only a single coherent state, 
            # the shape of the tensor is changed to (dim,1)
            qi = qi.unsqueeze(1)
            pi = pi.unsqueeze(1)
        if len(qj.size()) == 1:
            # If qj,pj contain only a single coherent state, 
            # the shape of the tensor is changed to (dim,1)
            qj = qj.unsqueeze(1)
            pj = pj.unsqueeze(1)
        
        # Reshape qi,pi,qj and pj so that the tensor have the shape (dim,ni,nj)
        
        # ni is the number of bra coherent states <qi,pi|
        d,ni = qi.size()
        # nj is the number of ket coherent states |qj,pj>
        d,nj = qj.size()
        #           i     ->          i         j
        # (dim, ntraj_i)  ->  (dim, ntraj_i, ntraj_j)
        qi, pi = qi.unsqueeze(2).expand(-1,-1,nj), pi.unsqueeze(2).expand(-1,-1,nj)
        #         j       ->          i      j
        # (dim, ntraj_j)  ->  (dim, ntraj_i, ntraj_j)
        qj, pj = qj.unsqueeze(1).expand(-1,ni,-1), pj.unsqueeze(1).expand(-1,ni,-1)
        
        # prefactor from normalization
        fac = torch.sqrt(2**d * torch.sqrt(self.detGi*self.detGj) / self.detGij)
        
        olap = fac * torch.exp(
            -0.5         * torch.einsum('aij,ab,bij->ij', qj-qi, self.Gi_iGij_Gj, qj-qi)
            -0.5/hbar**2 * torch.einsum('aij,ab,bij->ij', pj-pi, self.iGij      , pj-pi)
            -1j/hbar     * torch.einsum('aij,aij->ij'   , pj                    , qj-qi)
            +1j/hbar     * torch.einsum('aij,ab,bij->ij', qj-qi, self.Gj_iGij   , pj-pi)
                              )
        assert olap.size() == (ni,nj)
        
        return olap
    
    
class CoherentStatesWavefunction(object):
    def __init__(self, G):
        self.G = G
        self.detG = torch.det(G)
    def __call__(self, q,p,v, x):
        """
        evaluates a wavefunction consisting of a linear combination of coherent
        states on a spatial grid
        
          phi(x) = sum  v(i) <x|q(i),p(i)>
                      i
        
        Parameters
        ----------
        q,p  :  real Tensor (dim,ntraj)
          positions and momenta of bra coherent states
        v    :  complex Tensor (ntraj,)
          coefficients of coherent states
        x    :  real Tensor (dim,nx)
          spatial grid
          
        Returns
        -------
        phi   :  Tensor (nx,)
          wavefunction `phi(x)` on the spatial grid
        """
        d,nx = x.size()
        dim,ntraj = q.size()
        assert d == dim, "dimensions of spatial grid and coherent states differ"
        
        x = x.unsqueeze(1).expand(-1,ntraj,-1)
        q = q.unsqueeze(2).expand_as(x)
        p = p.unsqueeze(2).expand_as(x)
        # v has shape (ntraj,)   --> (ntraj,nx_)
        v = v.unsqueeze(1).expand(-1,nx)
        # normalization factor
        fac = (self.detG/np.pi**d)**0.25
        # evaluate coherent state wavefunctions of all trajectories <x|q(i),p(i)>
        # on the spatial grid --> shape(gaussians) = (ntraj,nx_)
        gaussians = fac * torch.exp(-0.5     * torch.einsum('inx,ij,jnx->nx',x-q,self.G,x-q)
                                    +1j/hbar * torch.einsum('inx,inx->nx',     p,       x-q))
        # sum_i v(i) <x|q(i),p(i)>
        phi = torch.sum( v * gaussians, 0 )
        
        return phi

    
# # Newton's equations for classical trajectories and monodromy matrices
    
def _equations_of_motion(t, y, potential):
    """
    Newton's equations of motion for propagating the
        * positions q,
        * momenta p,
        * monodromy matrices Mqq = dq(t)/dq(0), Mqp = dq(t)/dp(0), Mpq = dp(t)/dq(0), Mpp = dp(t)/dp(0)
        * and classical action S
    are combined into a single first-order differential equation
    
        dy/dt = f(y)
        
    for the vector y = (q,p,Mqq,Mqp,Mpq,Mpp,action).
    
    This function evaluates the derivative f(y).
    
    Parameters
    ----------
    t   :  float
      current time (in a.u.)
    y   :  Tensor (4*d**2+2*d+1,n)
      current solution vector
    potential : object
      potential energy surface implementing energy(q), gradient(q) and hessian(q) methods
    """
    d = potential.dimensions()
    masses = potential.masses()
    
    q,p, Mqq,Mqp,Mpq,Mpp, action = torch.split(y, [d,d,d**2,d**2,d**2,d**2,1])
    
    Mqq = Mqq.view(d,d,-1)
    Mqp = Mqp.view(d,d,-1)
    Mpq = Mpq.view(d,d,-1)
    Mpp = Mpp.view(d,d,-1)

    vpot = potential.energy(q)
    grad = potential.gradient(q)
    hess = potential.hessian(q)

    #
    # d   dq_a(t)          dp_a(t)
    # -- (-------) = 1/m_a -------
    # dt  dq_b(i)          dq_b(t)
    DMqq = Mpq / masses.unsqueeze(1).unsqueeze(2).expand_as(Mpq)
    #
    # d   dp_a(t)              d^2 V     dq_g(t)
    # -- (-------) = sum  -  --------- * -------
    # dt  dq_b(i)       g    dq_a dq_g   dq_b(i)
    DMpq = - torch.einsum('ag...,gb...->ab...', hess, Mqq)
    #
    # d   dq_a(t)          dp_a(t)
    # -- (-------) = 1/m_a -------
    # dt  dp_b(i)          dp_b(t)
    DMqp = Mpp / masses.unsqueeze(1).unsqueeze(2).expand_as(Mpp)
    #
    # d   dp_a(t)              d^2 V     dq_g(t)
    # -- (-------) = sum  -  --------- * -------
    # dt  dp_b(i)       g    dq_a dq_g   dp_b(i)
    DMpp = - torch.einsum('ag...,gb...->ab...', hess, Mqp)

    #
    # dq_a/dt = 1/m_a p_a
    Dq = p / masses.unsqueeze(1).expand_as(p)
    #
    # dp_a/dt = - dV/dq_a
    Dp = -grad

    # dS = p^2/(2m) - V
    tkin = 0.5 * torch.sum(p**2 / masses.unsqueeze(1).expand_as(p), 0)
    Daction = tkin - vpot
    
    # combine all derivatives
    dydt = torch.cat((Dq,Dp,
                      DMqq.reshape(d**2,-1),
                      DMqp.reshape(d**2,-1),
                      DMpq.reshape(d**2,-1),
                      DMpp.reshape(d**2,-1),
                      Daction.reshape(1,-1)), 0)
    
    # Average of total energies (kinetic + potential) of all classical trajectories 
    # should be conservd.
    en_mean = torch.mean(tkin + vpot)
    logger.debug(f"average energy of classical trajectories <T+V>= {en_mean}")
    
    return dydt


# # Herman-Kluk Propagator
# [HK] E. Kluk, M. Herman, H. Davis,
#     "Comparison of the propagation of semiclassical frozen Gaussian wave functions with quantum propagation  for a highly excited anharmonic oscillator",
#     J. Chem. Phys. 84, 326, (1986)


class HermanKlukPropagator(object):
    def __init__(self, Gamma_i, Gamma_t):
        """
        semiclassical Herman-Kluk propagator
        
        Parameters
        ----------
        Gamma_i  :  real Tensor (dim,dim)
          width parameter matrix of initial coherent states at t=0
        Gamma_t  :  real Tensor (dim,dim)
          width parameter matrix of coherent states at later times t
        """
        # width parameters of coherent states
        self.Gamma_i = Gamma_i
        self.Gamma_t = Gamma_t
        # \Gamma_i^{1/2}
        self.sqGi = torch.tensor(sla.sqrtm(Gamma_i))
        # \Gamma_t^{1/2}
        self.sqGt = torch.tensor(sla.sqrtm(Gamma_t))
        # \Gamma_i^{-1/2}
        self.isqGi = torch.inverse(self.sqGi)
        # \Gamma_t^{-1/2}
        self.isqGt = torch.inverse(self.sqGt)
        
        # time in a.u.
        self.t = 0.0
    def initial_conditions(self, q0, p0, Gamma_0, ntraj=5000):
        """
        sample initial positions qi and momenta pi from the normalized probability distribution
        
          P(qi,pi) ~ |<qi,pi,Gamma_i|q0,p0,Gamma_0>|
          
        The initial Gaussian wavefunction centered at `q0` with initial momentum `p0`is
          
          phi(x,t=0) = <x|q0,p0,Gamma_0>
          
                        det(Gamma_0) 1/4                 T                        T
                     = (------------)    exp( -1/2 (x-q0) Gamma (x-q0) + i/hbar p0 (x-q0))
                           pi^N
                           
        Parameters
        ----------
        q0    :  real Tensor (dim,)
          center of initial wavefunction
        p0    :  real Tensor (dim,)
          momentum of initial wavefunction
        Gamma_0 :  real Tensor (dim,dim)
          width parameter matrix
        """
        assert Gamma_0.size() == self.Gamma_i.size(), "width parameter matrix Gamma_0 has wrong dimensions"
        # abbreviations
        n = ntraj
        d = q0.size()[0]
        
        G0 = Gamma_0
        Gi = self.Gamma_i
        Gi0 = Gi+G0
        iGi0 = torch.inverse(Gi0)
        # covariance matrices for q and p
        #              -1      -1
        # ( G . [Gi+G0]  . G  )
        #   i              0
        cov_q = torch.inverse( Gi @ iGi0 @ G0 )
        #
        # hbar^2 [Gi+G0]
        cov_p = hbar**2 * Gi0
        
        # covariance matrix
        cov = torch.block_diag(cov_q, cov_p)
        # center in phase space loc = (q0,p0)
        loc = torch.cat((q0, p0))
        distribution = MultivariateNormal(loc, cov)
        
        # sample initial positions and momenta from normal distribution
        zi = distribution.sample((ntraj,)).T
        
        # compare expected center and covariance of normal distribution with
        # mean and standard deviation of samples
        logger.info("== Initial Conditions ==")
        qi,pi = torch.split(zi,[d,d])
        logger.info(f"number of trajectories= {n}")
        logger.info(f"cov(q)= {torch.diag(cov_q)} \t std(q)^2= {torch.std(qi)**2}")
        logger.info(f"cov(p)= {torch.diag(cov_p)} \t std(p)^2= {torch.std(pi)**2}")
        logger.info(f"q0= {q0} \t <q>= {torch.mean(qi)}  ")
        logger.info(f"p0= {p0} \t <p>= {torch.mean(pi)}  ")
        logger.info("")
        
        # normalization constant, the wavefunction obtained as a superposition of frozen
        # Gaussians has to be divided by the normalization constant.
        # The wavefunction is assembled by Monte Carlo integration over the initial values.
        # The normalization constants are the probabilities for sampling each of the initial
        # values.
        norm_fac = 1.0/((2*np.pi)**d * torch.sqrt(torch.det(cov)))
        # mean (q0,p0)
        mu = loc.unsqueeze(1).expand_as(zi)
        # P(qi,pi) probability of sampling (qi,pi)
        probi = norm_fac * torch.exp(-0.5 * torch.einsum('in,ij,jn->n', zi-mu, torch.inverse(cov), zi-mu))
     
        # Initialize solution vector y (contains positions, momenta, monodromy matrix and classical action)
        yi = torch.zeros((2*d+4*d**2+1, n))
        z,Mqq,Mqp,Mpq,Mpp,action = torch.split(yi, [2*d,d**2,d**2,d**2,d**2,1])
        
        Mqq = Mqq.view(d,d,-1)
        Mqp = Mqp.view(d,d,-1)
        Mpq = Mpq.view(d,d,-1)
        Mpp = Mpp.view(d,d,-1)

        # Initial conditions
        #
        # dr_a(i)/dr_b(i) = delta_ab
        #
        Mqq[...] = torch.eye(d).unsqueeze(2).expand(-1,-1,n)
        #
        # dr_a(i)/dp_b(i) = 0
        # dp_a(i)/dr_b(i) = 0
        #
        # dp_a(i)/dp_b(i) = delta_ab
        #
        Mpp[...] = torch.eye(d).unsqueeze(2).expand(-1,-1,n)

        # set initial phase space points (q0,p0)
        z[...] = zi
        
        # save some data for use in other methods
        # dimensions
        self.dim = d
        # save characteristics of initial Gaussian wavepacket
        self.q0 = q0
        self.p0 = p0
        self.Gamma_0 = Gamma_0
        #
        self.detGi0 = torch.det(Gi0)
        self.iGi0 = torch.inverse(Gi0)
        # number of trajectories
        self.ntraj = ntraj
        # initial positions and momenta zi = (qi,pi)
        self.zi = zi
        # probabilities for sampling initial conditions P(qi,pi)
        self.probi = probi
        # initialize solution vector y
        self.y = yi
        # semiclassical prefactor
        self.c = torch.ones(n, dtype=torch.complex128)
        
        # Initialize variables for t=0
        self._prefactor()
        
    def step(self, potential, dt):
        """
        propagates solution for one time step (t -> t+dt) under the influence of `potential`
        """
        assert self.dim == potential.dimensions(), "potential has wrong dimensions"
        
        self.y = _rk4_step(_equations_of_motion, self.y, self.t, dt, potential)
        self._prefactor()
        
        self.t += dt
        
    def coefficients(self): 
        """
        expansion coefficients of Herman-Kluk wavefunction in the basis of the coherent
        state trajectories
        
                      i=ntraj
          phi(x,t) = sum      v  <x|q(t),p(t)>
                      i=1      i     i    i
            
        Returns
        -------
        v  :  complex Tensor (ntraj,)
          expansion coefficients
        """
        d = self.dim
        action = self.classical_action()
        c = self.semiclassical_prefactor()
        
        # overlap of the initial wavefunction with initial coherent states 
        # <qi,pi|phi(0)>
        csoi0 = CoherentStatesOverlap(self.Gamma_i, self.Gamma_0)
        qi,pi = self.initial_positions_and_momenta()
        v0 = csoi0(qi,pi, self.q0,self.p0).squeeze()
        
        # expansion coefficients of wavefunction in the basis of gaussian wavepackets
        # v_i = C(qi,pi,t)*e^{i*S/hbar} / (2 pi hbar)^d * <qi,pi|phi(0)>
        v = c * torch.exp(1j/hbar*action) / (2*np.pi*hbar)**d * v0
        # We also have to divide by the probability to sample (qi,pi), i.e. n*P(qi,pi)
        v = v/(self.ntraj * self.probi)
    
        return v
    
    def wavefunction(self, x):
        """
        evaluate the frozen Gaussian approximation of the wavefunction psi(x,t)

        The wavefunction is represented as a superposition of frozen Gaussians
        
                        det(Gamma_t) 1/4                 T                          T
          g(qi,pi;x) = (------------)    exp{ -1/2 (x-q )  Gamma_t (x-q ) + i/hbar p (x-q ) }
                           pi^dim                      i               i            i    i
                           
        with width parameter matrix Gamma_t centered at the phase space points (q_i,p_i). The expansion
        coefficients in the semiclassical approximation are given by

          vi = C(qi,pi,t) * exp(I*(Si/hbar)) / (2 pi)^dim  <qi,pi|phi(0)> / (ntraj*P(qi,pi))

        so that

        psi(x,t) = sum  v  g (x,t)
                    i    i  i


        Parameters
        ----------
        x     :  real Tensor (dim,nx)
          spatial grid

        Returns
        -------
        phi   :  complex numpy.ndarray (nx,)
          wavefunction phi(x,t) on the spatial grid
        """
        d,nx = x.shape
        assert d == self.dim, "spatial grid has wrong dimensions"
        
        q,p = self.current_positions_and_momenta()
        v = self.coefficients()
        
        csw = CoherentStatesWavefunction(self.Gamma_t)
            
        phi = torch.zeros(nx, dtype=torch.complex128)
        nchunk = nx // 100 + 1
        # Split the spatial grid into chunks and compute the wavefunction on each chunk.
        for x_,phi_ in zip(torch.chunk(x,nchunk,dim=1), 
                           torch.chunk(phi,nchunk,dim=0)):
            phi_[:] = csw(q,p,v,x_)
            
        return phi.detach().numpy()
    
    def norm(self):
        """
        The frozen Gaussian approximation conserves the normalization of the wavefunction
        only if the number of trajectories is sufficiently large.
        Monitoring the norm of the wavefunction tells us if the calculation is converged 
        with respect to the number of trajectories.
        
        To check if the wavefunction 
        
              psi(x,t) = sum  v  g (x,t)
                          i    i  i

        is normalized, we have to compute the normalization constant as
                                             1/2
              |psi| = ( sum  v^* v  <gi|gj> )
                        i,j   i   j

        Computing the norm is very expensive since it scales like N_{traj}^2.
        
        Returns
        -------
        norm   :  float
          norm |psi| of the wavefunction
        """
        v = self.coefficients()
        # The computation of the overlap matrix O[i,j] = <qi,pi|qj,pj> is split into blocks to avoid running
        # out of memory. The vector of coefficients is also split into chunks and the norm is obtained
        # by accumulating the terms v^T.O.v from all combinations of blocks.
        q,p = self.current_positions_and_momenta()
        cso = CoherentStatesOverlap(self.Gamma_t, self.Gamma_t)
        norm2 = torch.tensor([0.0j])
        # slip array of trajectories into `nchunk` chunks
        nchunk = self.ntraj // 1000 + 1
        for qi,pi,vi in zip(torch.chunk(q,nchunk,dim=1), 
                            torch.chunk(p,nchunk,dim=1), 
                            torch.chunk(v,nchunk,dim=0)):
            for qj,pj,vj in zip(torch.chunk(q,nchunk,dim=1), 
                                torch.chunk(p,nchunk,dim=1),
                                torch.chunk(v,nchunk,dim=0)):
                # overlap between Gaussian wavepackets at different points in phase space
                # olap[i,j] = <qi,pi|qj,pj>
                olap_ij = cso(qi,pi, qj,pj)
                # norm = sqrt( sum_{i,j} v[i]^* olap[i,j] * v[j] )
                # contribution from blocks i and j to norm
                norm2 += torch.einsum('i,ij,j', vi.conj(), olap_ij, vj)
        norm = torch.sqrt( norm2.real )

        #logger.info(f"|psi|= {norm}")

        return norm
        
    def autocorrelation_qp(self):
        """
        contribution of each trajectory to autocorrelation function
        """
        d = self.dim
        action = self.classical_action()
        c = self.semiclassical_prefactor()
        
        # overlap of the initial wavefunction with initial coherent states 
        # <qi,pi|phi(0)>
        csoi0 = CoherentStatesOverlap(self.Gamma_i, self.Gamma_0)
        qi,pi = self.initial_positions_and_momenta()
        vi = csoi0(qi,pi, self.q0,self.p0).squeeze()
        
        # overlap of the initial wavefunction with coherent states at current time 
        # <qt,pt|phi(0)>
        csot0 = CoherentStatesOverlap(self.Gamma_t, self.Gamma_0)
        qt,pt = self.current_positions_and_momenta()
        vt = csot0(qt,pt, self.q0,self.p0).squeeze()
        
        # contribution from individual trajectories to autocorrelation function
        #    (qp)
        #  C      = <phi(0)|qt,pt> C(qi,pi;t) * e^{i/hbar * S(qi,pi;t)} <qi,pi|phi(0)
        #   auto
        cauto_qp = vt.conj() * vi * c * torch.exp(1j/hbar*action)
        return cauto_qp
        
    def autocorrelation(self):
        """
        autocorrelation function for current time step
        
          C    (t) = <phi(0)|phi(t)>
           auto
           
        Returns
        -------
        cauto  :   float
           current value of autocorrelation function
        """
        cauto_qp = self.autocorrelation_qp()
        
        # Since the integral over initial values is done by Monte Carlo integration
        # with importance sampling we also have to divide by the probability 
        # to sample (qi,pi), i.e. by n*P(qi,pi)
        #
        #           /    dq dp      (qp)                                (qi,pi)
        # C   (t) = |------------- C   (t)  =  1/(2 pi hbar)^d *  sum  C (t)    / (n * P(qi,pi))
        #  auto     /(2 pi hbar)^d  auto                             i  auto
        #
        cauto = torch.sum(cauto_qp/(self.ntraj * self.probi)) /  (2*np.pi*hbar)**self.dim
    
        return cauto
    
    def ic_correlation(self, potential, energy0_es=0.0):
        """
        correlation function for internal conversion rate
        
          ~                       i/hbar t E_0^(es)              -i/har t H^(gs) 
          k  (t) = (2 pi)/hbar * e                  <phi(0)|T   e                T   |phi(0)>
           ic                                                eg                   ge   
          
        see equations in `notes/Herman_Kluk_ic_correlation.pdf`
          
        Parameters
        ----------
        potential  :  object
          potential energy surface, which provides non-adiabatic couplings
          between initial and final electronic states
        energy0_es :  float
          zero-point energy of vibrational ground on initial surface
        
        Returns
        -------
        kic  :  complex
          correlation function \tilde{k}_ic(t) at current time step
        """
        # trajectory-wise autocorrelation function
        cauto_qp = self.autocorrelation_qp()
        # phase space positions of trajectories at t=0 and t
        # tensors of dimensions (dim,ntraj)
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
        # phase space center of initial wavefunction phi(0)
        q0 = self.q0.unsqueeze(1).expand_as(q)
        p0 = self.p0.unsqueeze(1).expand_as(p)
        
        # non-adiabatic couplings
        # for initial geometries q = qi
        tau1q = potential.derivative_coupling_1st(q)
        tau2q = potential.derivative_coupling_2nd(q)
        # at current geometries Q = qt
        tau1Q = potential.derivative_coupling_1st(Q)
        tau2Q = potential.derivative_coupling_2nd(Q)
        
        masses = potential.masses()
        # eqn. (89)
        n1q = - hbar**2 * torch.einsum('k,kn->kn', 1.0/masses, tau1q)
        n1Q = - hbar**2 * torch.einsum('k,kn->kn', 1.0/masses, tau1Q)
        # eqn. (90)
        n2q = - hbar**2 * 0.5 * torch.einsum('k,kn->n', 1.0/masses, tau2q)
        n2Q = - hbar**2 * 0.5 * torch.einsum('k,kn->n', 1.0/masses, tau2Q)
        
        PI = p0 + torch.einsum('ij,jk,kn->in', self.Gamma_0, self.iGi0, P-p0)
        pi = p0 + torch.einsum('ij,jk,kn->in', self.Gamma_0, self.iGi0, p-p0)
        
        # constant matrix
        R = torch.einsum('ij,jk,kl->il', self.Gamma_0, self.iGi0, self.Gamma_i)
        
        nacQ = n2Q + (           torch.einsum('in,ij,jn->n', q0-Q, R, n1Q)
                      -1j/hbar * torch.einsum('in,in->n',         PI, n1Q))
        nacq = n2q + (           torch.einsum('in,ij,jn->n', q0-q, R, n1q)
                      +1j/hbar * torch.einsum('in,in->n',         pi, n1q))
        
        kic_t_qp =   2*np.pi / hbar \
                   * torch.exp(1j/hbar * self.t * torch.tensor(energy0_es)) \
                   * nacQ * nacq * cauto_qp
            
        kic_t = torch.sum(kic_t_qp/(self.ntraj * self.probi)) /  (2*np.pi*hbar)**self.dim

        return kic_t
    
    # methods for data access 
    def initial_positions_and_momenta(self):
        d = self.dim
        qi,pi = torch.split(self.zi, [d,d])
        return qi,pi
    def current_positions_and_momenta(self):
        d = self.dim
        q,p = torch.split(self.y, [d,d,d**2,d**2,d**2,d**2,1])[:2]
        return q,p
    def classical_action(self):
        d = self.dim
        action = torch.split(self.y, [d,d,d**2,d**2,d**2,d**2,1])[-1]
        action = action.squeeze()
        return action
    def monodromy_matrices(self):
        d = self.dim
        q,p,Mqq,Mqp,Mpq,Mpp,action = torch.split(self.y, [d,d,d**2,d**2,d**2,d**2,1])
        
        Mqq = Mqq.view(d,d,-1)
        Mqp = Mqp.view(d,d,-1)
        Mpq = Mpq.view(d,d,-1)
        Mpp = Mpp.view(d,d,-1)
        
        return Mqq,Mqp,Mpq,Mpp
    def semiclassical_prefactor(self):
        """
        Semiclassical prefactor C(t) without alignment of signs.
        To get a smooth curve in time C has to be multiplied by `self.signs`.
        
        Returns
        -------
        c    :   Tensor (ntraj,)
          Herman-Kluk semiclassical prefactor C(t)
        """
        signs = self._get_signs_of_sqrt("prefactorC")
        return signs * self.c
    
    # private methods (should only be called from within this class or derived classes)
    def _prefactor(self):
        """
        computes the semiclassical prefactor
        
         C(q,p;t) = det(...)^{1/2}
         
        according to eqn. (29)
        """
        Mqq,Mqp,Mpq,Mpp = self.monodromy_matrices()
        
        sqGi = self.sqGi
        sqGt = self.sqGt
        isqGi = self.isqGi
        isqGt = self.isqGt

        mat = 0.5 * (
                    #    1/2         -1/2
                    #   G   . M   . G    
                    #    t     qq    i
                     torch.einsum('ai,ijn,jb->abn',  sqGt, Mqq, isqGi)
                    #    -1/2         1/2
                    #   G   . M   . G    
                    #    t     pp    i
               +     torch.einsum('ai,ijn,jb->abn', isqGt, Mpp,  sqGi)
                    #           1/2         1/2
                    # -i*hbar  G   . M   . G    
                    #           t     qp    i
            -1j*hbar*torch.einsum('ai,ijn,jb->abn',  sqGt, Mqp,  sqGi)
                    #           -1/2         -1/2
                    # +i/hbar  G   . M   . G    
                    #           t     pq    i
            +1j/hbar*torch.einsum('ai,ijn,jb->abn', isqGt, Mpq, isqGi)
        )
        
        # To compute the determinants for a batch of matrices, the matrices have to be stacked
        # along the 0-th dimension, i.e. the array has to have the shape (ntraj,d,d), not (d,d,ntraj)
        mat = mat.permute(2,0,1)
        # Determinants of complex matrices are not yet implemented in pytorch, so that we have
        # to fall back to ...
        # ... our own implementation
        c2 = _complex_det(mat)
        """
        # ... or numpy
        c2 = torch.tensor(la.det(mat))
        # 
        """
        self.c = torch.sqrt(c2)
           
        # store C^2(t) for aligning signs relative to C^2(t-dt) stored in self.sqrt2_last
        self._track_signs_of_sqrt("prefactorC", c2)
        
    def _track_signs_of_sqrt(self, key, z):
        """
        track the signs of
        
            f(z(t)) = sqrt(z)
            
        Since the function f(z)=sqrt(z) has a branch cut at the
        negative half of the real axis, f(z) is not a continuous function, if z crosses the
        branch cut. To make f(t) a continuous function we have keep track of the sign changes
    
          f(f(z)) -> sign(t)*f(z(t))
      
        Parameters
        ----------
        key   :  string
          name of the quantity being tracked
        z     :  complex Tensor
          current value of z, the sign is determined by comparing
          with the previous value that was registered under the 
          same key
        """
        if not hasattr(self, "sign_trackers"):
            # stores values of z at previous time step
            #  sign_trackers[key] ->  z(t-dt)
            self.sign_trackers = {}
        
        # z(t) = f^2(t)
        
        # retrieve previous value and current signs or create a new tracker
        tracker = self.sign_trackers.get(key, {"signs": torch.ones_like(z), "previous": z})
        signs = tracker["signs"]
        # complex number z1 = f^2(t-dt)
        z1 = tracker["previous"]
        
        assert signs.size() == z.size(), "Size of tensor whose square is being tracked has changed."
        
        # complex number z2 = f^2(t)
        z2 = z
        # z(t) crosses the negative real axis
        cond = (z1.real < 0) & (z2.real < 0) & (z1.imag * z2.imag < 0)
        # keep track of sign changes
        signs[cond] *= -1.0
        
        # prepare for next step t -> t+dt
        tracker["signs"] = signs
        tracker["previous"] = z
        self.sign_trackers[key] = tracker
        
    def _get_signs_of_sqrt(self, key):
        """
        get signs of square roots of complex numbers which are being tracked
        
        This requires a call to `_track_signs_of_sqrt(key, z)` in every propagation
        step.
        """
        try:
            tracker = self.sign_trackers[key]
        except KeyError as err:
            logger.error(f"Apparently the sign of the square root of the quantity '{key}' is not being tracked.")
            raise err
        return tracker["signs"]
        


# # Walton-Manolopoulos Propagator
# 
# [WM] A. Walton, D. Manolopoulos,
#     "A new semiclassical initial value method for Franck-Condon spectra",
#     Mol. Phys. 87, 961-978, (1996)


class WaltonManolopoulosPropagator(HermanKlukPropagator):
    def __init__(self, Gamma_i, Gamma_t, beta):
        """
        semiclassical Herman-Kluk propagator
        
        Parameters
        ----------
        Gamma_i  :  real Tensor (dim,dim)
          width parameter matrix of initial coherent states at t=0
        Gamma_t  :  real Tensor (dim,dim)
          width parameter matrix of coherent states at later times t
        beta     :  float > 0
          inverse width of phase space cell over which the HK propagator
          integrated out
        """
        super().__init__(Gamma_i, Gamma_t)
        self.beta = beta
    def _expand_L(self):
        """
        The function 
        
          L(q,p;t) = log(C(q,p;t)) + i/hbar S(q,p;t)
          
        is expanded to quadratic order around the initial phase space point z=(qi,pi)
        
                                  dL T           T  d^2 L
          L(q,p;t) = L(qi,pi;t) +(--)  z + 1/2  z   ----- z
                                  dz                dzdz
                                  
        In Walton & Manolopoulos (1996) it is assumed that log(C(q,p;t)) is approximately
        constant, so that only the gradient and Hessian of the action needs to be computed.
        Some terms are neglected in the Hessian of S (see appendix A in WM eqns. A6-A9)
        
        Returns
        -------
        gradL  :  complex Tensor (2*dim,ntraj)
          gradient of L(q,p;t) with respect to z=(q,p) at zi=(qi,pi)
        hessL  :  complex Tensor (2*dim,2*dim,ntraj)
          Hessian of L(q,p;t) with respect to z=(q,p) at zi(qi,pi)
        """
        Mqq,Mqp,Mpq,Mpp = self.monodromy_matrices()
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
        
        # We assume that C is constant and only expand S
        # 
        #  L = log(C) + 1j/hbar * S
        #
        
        # approximate grad(L) = i/hbar grad(S)
        # eqn. (A4)    dS/dq = Mqq^T . P - p
        dSdq = torch.einsum('ijn,in->jn', Mqq, P) - p
        # eqn. (A5)    dS/dp = Mqp^T . P
        dSdp = torch.einsum('ijn,in->jn', Mqp, P)
        # combine derivatives 
        #
        #  dL     i   (dS/dq)
        #  -- = ----  (     )
        #  dz   hbar  (dS/qp)
        #
        gradL = 1j/hbar * torch.cat((dSdq, dSdp), dim=0)
        
        # approximate hess(L) = i/hbar hess(S)
        # eqns. A6-A9, only terms with first order derivatives of q and p are retained.
        Sqq = torch.einsum('ijn,ikn->jkn', Mpq, Mqq)
        Sqp = torch.einsum('ijn,ikn->jkn', Mpq, Mqp)
        Spq = torch.einsum('ijn,ikn->jkn', Mqp, Mpq)
        Spp = torch.einsum('ijn,ikn->jkn', Mqp, Mpp)
        # combine derivatives
        #
        #  d^2 L     i   (Sqq  Sqp)
        #  ----- = ----  (        )
        #  dz dz   hbar  (Spq  Spp)
        #
        hessL = 1j/hbar * torch.cat((
            torch.cat((Sqq,Sqp), dim=1),
            torch.cat((Spq,Spp), dim=1)), dim=0)
        
        return gradL, hessL
    def _prefactor(self):
        # compute Herman-Kluk prefactor C
        super()._prefactor()
        # compute Walton-Manolopoulos prefactor
        #
        # In the Einstein summations the indices have the following meanings:
        #  -  i,j,k,l = 1,...,dim denote nuclear degrees of freedom
        #  -  n = 1,...,ntraj enumertes the trajectories
        #
        d,n = self.dim, self.ntraj
        Mqq,Mqp,Mpq,Mpp = self.monodromy_matrices()
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
        gradL, hessL = self._expand_L()
        # eqn. (39)
        Mqz = torch.cat((Mqq,Mqp), dim=1)
        Mpz = torch.cat((Mpq,Mpp), dim=1)
        # eqn. (40)
        Eqz = torch.cat((torch.eye(d), torch.zeros(d,d)), dim=1).unsqueeze(2).expand_as(Mqz)
        Epz = torch.cat((torch.zeros(d,d), torch.eye(d)), dim=1).unsqueeze(2).expand_as(Mpz)
        # (2 dim) x (2 dim) identity matrix for n trajectories
        Id = torch.eye(2*d).unsqueeze(2).expand(-1,-1,n)
        # eqn. (50)
        A = 2*self.beta * Id - hessL + (
             torch.einsum('jin,jk,kln->iln', Mqz, self.Gamma_t, Mqz)
            +torch.einsum('jin,jk,kln->iln', Eqz, self.Gamma_i, Eqz)
            +2j/hbar * (
                 torch.einsum('jin,jkn->ikn', Mpz, Mqz)
                -torch.einsum('jin,jkn->ikn', Epz, Eqz)
            ))
        # To invert a batch of matrices, the axes have to be ordered as (ntraj,dim,dim)
        # not (dim,dim,ntraj). After invering the matrices, the original order has to
        # be restored again.
        iA = torch.inverse(A.permute(2,0,1)).permute(1,2,0)
        # eqn. (53)
        BQ = torch.einsum('ij,jkn->ikn', self.Gamma_t, Mqz) + 1j/hbar * Mpz
        # eqn. (54)
        Bq = torch.einsum('ij,jkn->ikn', self.Gamma_i, Eqz) - 1j/hbar * Epz
        # eqn. (55)
        b0 = gradL - 1j/hbar * (
             torch.einsum('jin,jn->in', Mqz, P)
            -torch.einsum('jin,jn->in', Eqz, p))
        # eqn. (57)
        Gt = self.Gamma_t.unsqueeze(2).expand(-1,-1,n) - torch.einsum('ijn,jkn,lkn->iln', BQ, iA, BQ)
        # eqn. (58)
        Gi = self.Gamma_i.unsqueeze(2).expand(-1,-1,n) - torch.einsum('ijn,jkn,lkn->iln', Bq, iA, Bq)
        # eqn. (59)
        Gti = torch.einsum('ijn,jkn,lkn->iln', BQ, iA, Bq)
        # eqn. (60)
        pi_t = P - 1j*hbar * torch.einsum('ijn,jkn,kn->in', BQ, iA, b0)
        pi_i = p + 1j*hbar * torch.einsum('ijn,jkn,kn->in', Bq, iA, b0)
        
        # eqn. (68)
        Gi0 = self.Gamma_0 + self.Gamma_i
        iGi0, detGi0 = self.iGi0, self.detGi0
        
        # vectors with dimensions (dim,ntraj)
        q0 = self.q0.unsqueeze(1).expand_as(q)
        p0 = self.p0.unsqueeze(1).expand_as(p)
        
        # einsum() only works with tensors of the same type, so we have to cast
        # the real tensors to complex tensors
        Gamma_0 = self.Gamma_0.type(torch.complex128)
        iGi0 = iGi0.type(torch.complex128)
        q,p = q.type(torch.complex128), p.type(torch.complex128)
        Q,P = Q.type(torch.complex128), P.type(torch.complex128)
        
        # eqn. (69)
        Cqq = Gamma_0 - torch.einsum('ij,jk,kl->il', Gamma_0, iGi0, Gamma_0)
        Cqq = Cqq.unsqueeze(2).expand(-1,-1,n)
        # eqn. (70)
        CQQ = Gt - torch.einsum('ijn,jk,lkn->iln', Gti, iGi0, Gti)
        # eqn. (71)
        CqQ = torch.einsum('ij,jk,lkn->iln', Gamma_0, iGi0, Gti)
        
        # eqn. (72)
        PIq = p0   - torch.einsum('ij,jk,kn->in', Gamma_0, iGi0, p0-pi_i)
        # eqn. (73)
        PIQ = pi_t + torch.einsum('ijn,jk,kn->in', Gti, iGi0, p0-pi_i)
        # eqn. (74)
        eps =   0.5         * torch.einsum('in,ijn,jn->n', b0, iA, b0) \
              - 0.5/hbar**2 * torch.einsum('in,ij,jn->n', p0-pi_i, iGi0, p0-pi_i)
        
        c = self.semiclassical_prefactor()

        # To compute the determinant of a batch of matrices, the axes have to be 
        # ordered as (ntraj,2*dim,2*dim) not (2*dim,2*dim,ntraj).
        detA = _complex_det(A.permute(2,0,1))
            
        # We have to choose the signs of sqrt(det(A)) such that a continuous
        # function of time results
        self._track_signs_of_sqrt("detA", detA)
        
        # save variables needed for constructing wavefunction
        self.Cqq, self.CQQ, self.CqQ = Cqq, CQQ, CqQ
        self.PIq, self.PIQ = PIq, PIQ
        self.detA = detA
        self.eps = eps
        
        # quantities needed for autocorrelation function
        G0 = Gamma_0.unsqueeze(2).expand(-1,-1,n)
        # eqn. (78)
        M = G0 + CQQ
        
        # inverse of M
        iM = torch.inverse(M.permute(2,0,1)).permute(1,2,0)
        # determinant of M
        detM = _complex_det(M.permute(2,0,1))
        
        # batches of dim x dim matrices
        # eqn. (79)
        Rqq = Cqq - torch.einsum('ijn,jkn,lkn->iln', CqQ, iM, CqQ)
        # eqn. (80)
        RQQ = G0 - torch.einsum('ij,jkn,kl->iln', Gamma_0, iM, Gamma_0)
        # eqn. (81)
        RqQ = torch.einsum('ijn,jkn,kl->iln', CqQ, iM, Gamma_0)
        
        # vectors
        # eqn. (82)
        Pq = PIq - torch.einsum('ijn,jkn,kn->in', CqQ, iM, PIQ-p0)
        # eqn. (83)
        PQ = p0 + torch.einsum('ij,jkn,kn->in', Gamma_0, iM, PIQ-p0)
        
        # scalars
        # eqn. (84)
        gamma = eps - 0.5/hbar**2 * torch.einsum('in,ijn,jn->n', PIQ-p0, iM, PIQ-p0)
        
        # save variables needed for constructing autocorrelation function
        self.Rqq, self.RQQ, self.RqQ = Rqq, RQQ, RqQ
        self.Pq, self.PQ = Pq, PQ
        self.gamma = gamma
        self.detM = detM
        
        # track the sign of sqrt(det(M))
        self._track_signs_of_sqrt("detM", detM)
            
    def coefficients(self):
        d,n = self.dim, self.ntraj
        C = self.semiclassical_prefactor()
        S = self.classical_action()
        
        signs_detA = self._get_signs_of_sqrt("detA")
        
        # coefficients of Gaussians, eqn. (75)
        v =   (torch.det(self.Gamma_0)/np.pi**d)**(1/4) \
            * (torch.det(self.Gamma_t)/np.pi**d)**(1/4) \
            * (torch.det(self.Gamma_i)/np.pi**d)**(1/4) \
            * torch.sqrt((2*np.pi)**d/self.detGi0) \
            * 1/(2*np.pi)**d \
            * C * torch.exp(1j/hbar * S) \
            * (2*self.beta)**d / torch.sqrt(self.detA) * signs_detA \
            * torch.exp(self.eps)
        
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
            
        # vectors with dimensions (dim,ntraj)
        q0 = self.q0.unsqueeze(1).expand_as(q)
        p0 = self.p0.unsqueeze(1).expand_as(p)    
            
        q,p = q.type(torch.complex128), p.type(torch.complex128)
        Q,P = Q.type(torch.complex128), P.type(torch.complex128)    
        
        # x-independent part of exponent in eqn. (75)
        v = v * torch.exp(
                -0.5 * torch.einsum('in,ij,jn->n', q0-q, self.Cqq, q0-q)
            -1j/hbar * torch.einsum('in,in->n', self.PIq, q0-q) )
        
        # We also have to divide by the probability to sample (qi,pi), i.e. n*P(qi,pi)
        v = v/(n * self.probi)
        
        return v
        
    def wavefunction(self, x):
        """
        evaluate the WM wavefunction psi(x,t) on a spatial grid

        Parameters
        ----------
        x     :  real Tensor (dim,nx)
          spatial grid

        Returns
        -------
        phi   :  complex numpy.ndarray (nx,)
          wavefunction phi(x,t) on the spatial grid
        """
        d,nx = x.shape
        assert d == self.dim, "spatial grid has wrong dimensions"
        d,n = self.dim, self.ntraj
        
        v = self.coefficients()
        
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
            
        # vectors with dimensions (dim,ntraj)
        q0 = self.q0.unsqueeze(1).expand_as(q)
        p0 = self.p0.unsqueeze(1).expand_as(p)    
            
        q,p = q.type(torch.complex128), p.type(torch.complex128)
        Q,P = Q.type(torch.complex128), P.type(torch.complex128)    
            
        phi = torch.zeros(nx, dtype=torch.complex128)
        nchunk = nx // 100 + 1
        # Split the spatial grid into chunks and compute the wavefunction on each chunk.
        for x_,phi_ in zip(torch.chunk(x,nchunk,dim=1), 
                           torch.chunk(phi,nchunk,dim=0)):
            d_,nx_ = x_.size()
            # (dim,nx_) -> (dim,ntraj,nx_)
            x_ = x_.unsqueeze(1).expand(-1,n,-1)
            Q_ = Q.unsqueeze(2).expand_as(x_)
            # eqn. (75), the exponent contains only the x-dependent parts, the x-independent
            # parts are absorbed into the coefficient v
            phi_[:] = torch.sum(
                v.unsqueeze(1).expand(-1,nx_) * torch.exp(
                    -0.5 * torch.einsum('inx,ijn,jnx->nx', x_-Q_, self.CQQ, x_-Q_)
                    +      torch.einsum('in,ijn,jnx->nx', q0-q, self.CqQ, x_-Q_)
                    +1j/hbar * torch.einsum('in,inx->nx', self.PIQ, x_-Q_)),
                dim=0)
            
        return phi.detach().numpy()
    
    def norm(self):
        """
        norm of Walton & Manolopoulos' semiclassical wavefunction 
        (see notes/Walton_Manolopoulos_wavefunction_norm.pdf)

        Computing the norm is very expensive since it scales like N_{traj}^2.
        
        Returns
        -------
        norm   :  float
          norm |psi| of the wavefunction
        """
        d,n = self.dim, self.ntraj
        v = self.coefficients()
        
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
        
        # vectors with dimensions (dim,ntraj)
        q0 = self.q0.unsqueeze(1).expand_as(q).type(torch.complex128)
        
        # In the Einstein notation the indices have the following meanings:
        #   i,j enumerate trajectories, i=1,...,ni, j=1,...,nj
        #   a,b enumerate dimensions, a,b = 1,...,dim
        dvec = torch.einsum('ban,bn->an', self.CqQ, q0 - q) + 1j/hbar * self.PIQ
        
        # The computation of the overlap matrix O[i,j] = <qi,pi|qj,pj> is split into blocks to avoid running
        # out of memory. The vector of coefficients is also split into chunks and the norm is obtained
        # by accumulating the terms v^T.O.v from all combinations of blocks.
        norm2 = torch.tensor([0.0j])
        # slip array of trajectories into `nchunk` chunks
        nchunk = self.ntraj // 1000 + 1
        #
        for Qi,di,CQQi,vi in zip(torch.chunk(Q,nchunk,dim=1), 
                                 torch.chunk(dvec,nchunk,dim=1),
                                 torch.chunk(self.CQQ,nchunk,dim=2),
                                 torch.chunk(v,nchunk,dim=0)):
            ni = vi.size()[0]
            for Qj,dj,CQQj,vj in zip(torch.chunk(Q,nchunk,dim=1), 
                                     torch.chunk(dvec,nchunk,dim=1),
                                     torch.chunk(self.CQQ,nchunk,dim=2),
                                     torch.chunk(v,nchunk,dim=0)):
                nj = vj.size()[0]
                # dQij has shape (dim,ni,nj)
                dQij = Qj.unsqueeze(1).expand(-1,ni,-1) - Qi.unsqueeze(2).expand(-1,-1,nj)
                dQij = dQij.type(torch.complex128)
                # 
                di_ = di.unsqueeze(2).expand(-1,-1,nj)
                dj_ = dj.unsqueeze(1).expand(-1,ni,-1)
                #
                CQQi_ = CQQi.unsqueeze(3).expand(-1,-1,-1,nj)
                CQQj_ = CQQj.unsqueeze(2).expand(-1,-1,ni,-1)
                # D^{(ij)} = C_QQ^{(i)*} + C_QQ^{(j)}
                Dij = CQQi_.conj() + CQQj_
                # permute axis in `Dij` so that batch dimensions ni,nj come first, 
                #  (d,d,ni,nj) -> (ni,nj,d,d)
                Dij = Dij.permute(2,3,0,1)
                # inverse of D^(ij) = C_QQ^(i)^* + C_QQ^(j) and restore original order of axes
                iDij = torch.inverse(Dij).permute(2,3,0,1)
                # determinant of D^(ij)
                detDij = _complex_det(Dij)
                #
                bij = torch.einsum('abij,bij->aij', CQQj_, dQij) + di_.conj() + dj_
                
                # What about the sign of sqrt(det(Dij))? Do we have to track the sign
                # between consecutive time steps, as done for the semiclassical prefactor?
                
                # This is not really an overlap but we call it like this because the code
                # looks similar to the HK norm.
                olap_ij = torch.sqrt((2*np.pi)**d / detDij) * torch.exp(
                    -0.5 * torch.einsum('aij,abij,bij->ij', dQij, CQQj_, dQij)
                         - torch.einsum('aij,aij->ij', dj_, dQij)
                    +0.5 * torch.einsum('aij,abij,bij->ij', bij, iDij, bij))
                
                # norm = sqrt( sum_{i,j} v[i]^* olap[i,j] * v[j] )
                # contribution from blocks i and j to norm
                norm2 += torch.einsum('i,ij,j', vi.conj(), olap_ij, vj)

        norm = torch.sqrt( norm2.real )

        #logger.info(f"|psi|= {norm}")

        return norm
    
    def autocorrelation_qp(self):
        """
        contribution of each trajectory to autocorrelation function
        """
        d = self.dim
        C = self.semiclassical_prefactor()
        S = self.classical_action()
        
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
        
        # vectors with dimensions (dim,ntraj)
        q0 = self.q0.unsqueeze(1).expand_as(q).type(torch.complex128)
        
        # sign changes of sqrt(det(M)) and sqrt(det(A))
        signs_detM = self._get_signs_of_sqrt("detM")
        signs_detA = self._get_signs_of_sqrt("detA")
        
        # contribution from individual trajectories to autocorrelation function
        # prefactor part in eqn. (85)
        cauto_qp =    (torch.det(self.Gamma_0)/np.pi**d)**(1/2) \
                    * (torch.det(self.Gamma_t)/np.pi**d)**(1/4) \
                    * (torch.det(self.Gamma_i)/np.pi**d)**(1/4) \
                    * torch.sqrt((2*np.pi)**d/self.detGi0) \
                    * C * torch.exp(1j/hbar * S) \
                    * (2*self.beta)**d / torch.sqrt(self.detA) * signs_detA \
                    * torch.sqrt((2*np.pi)**d/self.detM) * signs_detM \
                    * torch.exp(self.gamma)
        
        # exponential part in eqn. (85)
        cauto_qp = cauto_qp * torch.exp(
                -0.5 * torch.einsum('in,ijn,jn->n', q0-q, self.Rqq, q0-q)
                -0.5 * torch.einsum('in,ijn,jn->n', q0-Q, self.RQQ, q0-Q)
                +      torch.einsum('in,ijn,jn->n', q0-q, self.RqQ, q0-Q)
            -1j/hbar * torch.einsum('in,jn->n', self.Pq, q0-q)
            +1j/hbar * torch.einsum('in,jn->n', self.PQ, q0-Q)
        )
        return cauto_qp
    
    def autocorrelation(self):
        """
        autocorrelation function for current time step
        
          C    (t) = <phi(0)|phi(t)>
           auto
           
        Returns
        -------
        cauto  :   float
           current value of autocorrelation function
        """
        cauto_qp = self.autocorrelation_qp()
        
        # Since the integral over initial values is done by Monte Carlo integration
        # with importance sampling we also have to divide by the probability 
        # to sample (qi,pi), i.e. by n*P(qi,pi)
        #
        #           /    dq dp      (qp)                                (qi,pi)
        # C   (t) = |------------- C   (t)  =  1/(2 pi hbar)^d *  sum  C (t)    / (n * P(qi,pi))
        #  auto     /(2 pi hbar)^d  auto                             i  auto
        #
        cauto = torch.sum(cauto_qp/(self.ntraj * self.probi)) /  (2*np.pi*hbar)**self.dim
    
        return cauto
    
    def ic_correlation(self, potential, energy0_es=0.0):
        """
        correlation function for internal conversion
        
          ~                       i/hbar t E_0^(es)              -i/har t H^(gs) 
          k  (t) = (2 pi)/hbar * e                  <phi(0)|T   e                T   |phi(0)>
           ic                                                eg                   ge   
          
        Parameters
        ----------
        potential  :  object
          potential energy surface, which provides non-adiabatic couplings
          between initial and final electronic states
        energy0_es :  float
          zero-point energy of vibrational ground on initial surface
        
        Returns
        -------
        kic  :  complex
          correlation function \tilde{k}_ic(t) at current time step
        """
        # trajectory-wise autocorrelation function
        cauto_qp = self.autocorrelation_qp()
        # phase space positions of trajectories at t=0 and t
        # tensors of dimensions (dim,ntraj)
        q,p = self.initial_positions_and_momenta()
        Q,P = self.current_positions_and_momenta()
        # phase space center of initial wavefunction phi(0)
        q0 = self.q0.unsqueeze(1).expand_as(q)
        p0 = self.p0.unsqueeze(1).expand_as(p)
        
        # non-adiabatic couplings
        # for initial geometries q = qi
        tau1q = potential.derivative_coupling_1st(q)
        tau2q = potential.derivative_coupling_2nd(q)
        # at current geometries Q = qt
        tau1Q = potential.derivative_coupling_1st(Q)
        tau2Q = potential.derivative_coupling_2nd(Q)
        
        masses = potential.masses()
        # eqn. (89)
        n1q = - hbar**2 * torch.einsum('k,kn->kn', 1.0/masses, tau1q).type(torch.complex128)
        n1Q = - hbar**2 * torch.einsum('k,kn->kn', 1.0/masses, tau1Q).type(torch.complex128)
        # eqn. (90)
        n2q = - hbar**2 * 0.5 * torch.einsum('k,kn->n', 1.0/masses, tau2q).type(torch.complex128)
        n2Q = - hbar**2 * 0.5 * torch.einsum('k,kn->n', 1.0/masses, tau2Q).type(torch.complex128)
        
        # cast to complex tensors
        q0 = q0.type(torch.complex128)
        q = q.type(torch.complex128)
        Q = Q.type(torch.complex128)
    
        nacqQ = torch.einsum('in,ijn,jn->n', n1q, self.RqQ, n1Q)
        nacQ = n2Q + (           torch.einsum('in,ijn,jn->n', q0-Q, self.RQQ, n1Q)
                               - torch.einsum('in,ijn,jn->n', q0-q, self.RqQ, n1Q)
                      -1j/hbar * torch.einsum('in,in->n',           self.PQ,  n1Q))
        nacq = n2q + (           torch.einsum('in,ijn,jn->n', q0-q, self.Rqq, n1q)
                               - torch.einsum('in,jin,jn->n', q0-Q, self.RqQ, n1q)
                      +1j/hbar * torch.einsum('in,in->n',           self.Pq,  n1q))
        # eqn. (100)
        kic_t_qp =   2*np.pi / hbar \
                   * torch.exp(1j/hbar * self.t * torch.tensor(energy0_es)) \
                   * (nacqQ + nacQ * nacq) * cauto_qp

        # integrate over initial conditions with appropriate weights from Monte-Carlo integration
        kic_t = torch.sum(kic_t_qp/(self.ntraj * self.probi)) /  (2*np.pi*hbar)**self.dim

        return kic_t

