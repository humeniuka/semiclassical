#!/usr/bin/env python
# coding: utf-8
"""probability distributions for sampling initial conditions"""
import torch
import numpy as np
from scipy import special

class UniformOverlapDistribution(object):
    def __init__(self, dim, device='cpu'):
        """
        draw vectors x from R^dim such that the radial parts
        |x| are distributed according to

              o = exp(-1/2 |x|^2)
           P(o) = uniform

        while the orientations x/|x| are uniformly distributed points
        on the unit sphere embedded in the dim-dimensional space,

           P(x/|x|) = 1/(area of unit sphere)

        Parameters
        ----------
        dim   :   int
          dimension of space

        Optional
        --------
        device :  str
          device ('cpu' or 'cuda:#') where random numbers should be sampled
        """
        self.dim = dim
        # surface are of unit sphere in dim dimensions
        #  S_{dim-1}(r=1)
        self.unit_sphere_area = torch.tensor(
            2.0 * np.pi**(dim/2) / special.gamma(dim/2)  ).to(device)

        # normal distributions for `dim` random variables with mean = 0, std = 1
        self.normal = torch.distributions.Normal(torch.zeros(dim).to(device),
                                                 torch.ones(dim).to(device))
        # uniform distribution from interval [0,1)
        self.uniform = torch.distributions.Uniform(torch.zeros(1).to(device),
                                                   torch.ones(1).to(device))
    def sample(self, n):
        """
        draw samples from distribution 

        Parameters
        ----------
        n     :   int
          number of samples

        Returns
        -------
        x     :   Tensor (dim, n)
          randomly sampled points
        """
        dim = self.dim
        # The method for generating uniformly distributed random points on a sphere
        # is explained in 
        #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # or
        #  Donald Knuth's "The art of computer programming II",
        #  3.4.1  Excercise 6) "Random point on an n-dimensional sphere with radius one."
        y = self.normal.sample((n,)).T
        # The points u are uniformly distributed on the unit sphere in dim dimensions
        u = y / torch.norm(y, dim=0)
        # overlap o = exp(-1/2 r^2) should be distributed uniformly in [0,1)
        o = self.uniform.sample((n,)).T
        # invert expression for r
        r = torch.sqrt(-2.0*torch.log(o))
        # Scale the unit vector by r
        x = r*u

        return x
        
    def probability(self, x):
        """
        compute the probability P(x) for sampling x

        Parameters
        ----------
        x     :   Tensor (dim, n)
          sampled points
        
        Returns
        -------
        prob  :   Tensor (n,)
          probabilities for sampling points, prob[i] = P(x[i])
        """
        dim, n = x.size()
        assert self.dim == dim, f"Points `x` have wrong dimension, expected {self.dim}, got {dim}."
        # r = |x|
        r = torch.norm(x, dim=0)
        # P(o) do = P(r) dr
        # Since `o = exp(-1/2 r^2)` is uniformly distributed
        # P(r) ~ |do/dr| ~ o r = r * exp(-1/2 r^2)
        # radial distribution P(|x|)
        Pr = r * torch.exp(-1/2 * r**2)
        # The probability of a point on the sphere with radius r
        # is inversely proportional to the surface area of that sphere,
        # which scales like r^(dim-1):
        #
        #  S_{dim-1}(r) = S_{dim-1}(r=1) r^{dim-1}
        #
        #     x                      1
        #  P(--- given |x|=r) = -----------
        #    |x|                S_{dim-1}(r)
        Pu = 1.0/(self.unit_sphere_area * r**(dim-1))
        # 
        prob = Pr*Pu

        return prob

