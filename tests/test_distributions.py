#!/usr/bin/env python
# coding: utf-8
"""test distribution functions and sampling probabilities"""

import unittest
import logging

import torch
import numpy as np
from scipy import special

from semiclassical.distributions import UniformOverlapDistribution, MultivariateNormalDistribution, UnitBallFillingDistribution

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[testing] %(message)s", level=logging.INFO)


class TestUniformOverlapDistribution(unittest.TestCase):
    def test_overlap_distribution(self):
        """
        check that the random number 

           o = exp(-1/2 |x|^2)

        is distributed uniformly in the interval [0,1) 
        when x is drawn from the `UniformOverlapDistribution(dim)`
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        dim = 5
        distribution = UniformOverlapDistribution(dim)
        x = distribution.sample(100000)

        # distance from origin r=|x|
        r = torch.norm(x, dim=0)
        # overlap with other Gaussian centered at the origin
        o = torch.exp(-1/2 * r**2)

        # put overlaps into bins to get approximate probability distribution
        prob_o, _ = np.histogram(o.numpy(), bins=30, range=(0.0, 1.0), density=True)

        # overlap distribution should be uniform, P(o) = 1
        self.assertTrue( np.allclose(prob_o, 1.0, atol=0.05) )
        
    def test_probabilities_2d(self):
        """
        compare the histogram of samples in 2 dimensions with the expected
        probability distribution P(x)
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        distribution = UniformOverlapDistribution(2)
        x_random = distribution.sample(100000)
        # histogram in [-5,5] y [-5,5]
        bins = 30
        rmax = 5.0
        hist, x1_, x2_ = np.histogram2d(x_random[0,:].numpy(), x_random[1,:].numpy(),
                                        density=True, range=[[-rmax, rmax],
                                                             [-rmax,rmax]],
                                        bins=bins)
        
        # compute probabilities P(x1,x2) at the center of the bins
        x_ = torch.zeros(2,bins*bins)
        # x1-coordinates of grid
        x_[0,:] = torch.tensor( 0.5*(x1_[1:] + x1_[:-1]) ).unsqueeze(1).expand(-1,bins).reshape(-1)
        # x2-cordinates
        x_[1,:] = torch.tensor( 0.5*(x2_[1:] + x2_[:-1]) ).unsqueeze(0).expand(bins,-1).reshape(-1)
        # evaluate probabilities on grid
        prob = distribution.probability(x_).reshape(bins,bins)

        """
        # compare distributions visually
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        
        ax1.imshow(hist)
        ax2.imshow(prob.numpy())

        plt.show()
        """

        # compare histogram and P(x)
        self.assertTrue( np.allclose(hist, prob.numpy(), atol=0.01) )

    def test_monte_carlo_integration(self):
        """
        compute the volume of the unit sphere in dim=1,2,3,... dimensions
        by Monte Carlo integration

          /           / f(x)            1             f(xi)
          | f(x) dx = | ---- P(x) dx =  -     sum      -----
          /           / P(x)            n   xi ~ P(x)  P(xi)

        n is the number of samples from probability distribution P(x)

                 1   |x| <= 1
        f(x) = { 
                 0   else
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        logger.info(f"Volume of unit sphere in d dimensions")

        for dim in [1,2,3,4,5]:
            with self.subTest(dimension=dim):
                n = 100000
                distribution = UniformOverlapDistribution(dim)
                x = distribution.sample(n)
                # P(x)
                p = distribution.probability(x)
                # |x|
                r = torch.norm(x, dim=0)
                # 1 if |x| < 1, else 0
                f = 1.0 - torch.heaviside(r-1.0, torch.tensor(1.0))
                # volume from MC integration
                volume = torch.sum(f/p) / n
                # expected volume of unit sphere Vn
                volume_exact = (np.pi)**(dim/2) / special.gamma(dim/2+1)
                logger.info(f"d= {dim}  V_dim= {volume:.5f} (Monte-Carlo)   {volume_exact:.5f} (exact)")

                self.assertTrue( abs(volume - volume_exact) < 0.05 )


class TestMultivariateNormalDistribution(unittest.TestCase):
    def test_probabilities_2d(self):
        """
        compare the histogram of samples in 2 dimensions with the expected
        probability distribution P(x)
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        distribution = MultivariateNormalDistribution(2)
        x_random = distribution.sample(100000)
        # histogram in [-5,5] y [-5,5]
        bins = 30
        rmax = 5.0
        hist, x1_, x2_ = np.histogram2d(x_random[0,:].numpy(), x_random[1,:].numpy(),
                                        density=True, range=[[-rmax, rmax],
                                                             [-rmax,rmax]],
                                        bins=bins)
        
        # compute probabilities P(x1,x2) at the center of the bins
        x_ = torch.zeros(2,bins*bins)
        # x1-coordinates of grid
        x_[0,:] = torch.tensor( 0.5*(x1_[1:] + x1_[:-1]) ).unsqueeze(1).expand(-1,bins).reshape(-1)
        # x2-cordinates
        x_[1,:] = torch.tensor( 0.5*(x2_[1:] + x2_[:-1]) ).unsqueeze(0).expand(bins,-1).reshape(-1)
        # evaluate probabilities on grid
        prob = distribution.probability(x_).reshape(bins,bins)

        """
        # compare distributions visually
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        
        ax1.imshow(hist)
        ax2.imshow(prob.numpy())

        plt.show()
        """

        # compare histogram and P(x)
        self.assertTrue( np.allclose(hist, prob.numpy(), atol=0.01) )

    def test_monte_carlo_integration(self):
        """
        compute the volume of the unit sphere in dim=1,2,3,... dimensions
        by Monte Carlo integration

          /           / f(x)            1             f(xi)
          | f(x) dx = | ---- P(x) dx =  -     sum      -----
          /           / P(x)            n   xi ~ P(x)  P(xi)

        n is the number of samples from probability distribution P(x)

                 1   |x| <= 1
        f(x) = { 
                 0   else
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        logger.info(f"Volume of unit sphere in d dimensions")

        for dim in [1,2,3,4,5]:
            with self.subTest(dimension=dim):
                n = 100000
                distribution = MultivariateNormalDistribution(dim)
                x = distribution.sample(n)
                # P(x)
                p = distribution.probability(x)
                # |x|
                r = torch.norm(x, dim=0)
                # 1 if |x| < 1, else 0
                f = 1.0 - torch.heaviside(r-1.0, torch.tensor(1.0))
                # volume from MC integration
                volume = torch.sum(f/p) / n
                # expected volume of unit sphere Vn
                volume_exact = (np.pi)**(dim/2) / special.gamma(dim/2+1)
                logger.info(f"d= {dim}  V_dim= {volume:.5f} (Monte-Carlo)   {volume_exact:.5f} (exact)")

                self.assertTrue( abs(volume - volume_exact) < 0.05 )

                
class TestUnitBallFillingDistribution(unittest.TestCase):
    def test_probabilities_2d(self):
        """
        compare the histogram of samples in 2 dimensions with the expected
        probability distribution P(x)
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        distribution = UnitBallFillingDistribution(2)
        x_random = distribution.sample(100000)
        # histogram in [-5,5] y [-5,5]
        bins = 30
        rmax = 5.0
        hist, x1_, x2_ = np.histogram2d(x_random[0,:].numpy(), x_random[1,:].numpy(),
                                        density=True, range=[[-rmax, rmax],
                                                             [-rmax,rmax]],
                                        bins=bins)
        
        # compute probabilities P(x1,x2) at the center of the bins
        x_ = torch.zeros(2,bins*bins)
        # x1-coordinates of grid
        x_[0,:] = torch.tensor( 0.5*(x1_[1:] + x1_[:-1]) ).unsqueeze(1).expand(-1,bins).reshape(-1)
        # x2-cordinates
        x_[1,:] = torch.tensor( 0.5*(x2_[1:] + x2_[:-1]) ).unsqueeze(0).expand(bins,-1).reshape(-1)
        # evaluate probabilities on grid
        prob = distribution.probability(x_).reshape(bins,bins)

        """
        # compare distributions visually
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        
        ax1.imshow(hist)
        ax2.imshow(prob.numpy())

        plt.show()
        """

        # compare histogram and P(x)
        self.assertTrue( np.allclose(hist, prob.numpy(), atol=0.02) )

    def test_monte_carlo_integration(self):
        """
        compute the volume of the unit sphere in dim=1,2,3,... dimensions
        by Monte Carlo integration

          /           / f(x)            1             f(xi)
          | f(x) dx = | ---- P(x) dx =  -     sum      -----
          /           / P(x)            n   xi ~ P(x)  P(xi)

        n is the number of samples from probability distribution P(x)

                 1   |x| <= 1
        f(x) = { 
                 0   else
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        logger.info(f"Volume of unit sphere in d dimensions")

        for dim in [1,2,3,4,5]:
            with self.subTest(dimension=dim):
                n = 100000
                distribution = UnitBallFillingDistribution(dim)
                x = distribution.sample(n)
                # P(x)
                p = distribution.probability(x)
                # |x|
                r = torch.norm(x, dim=0)
                # 1 if |x| < 1, else 0
                f = 1.0 - torch.heaviside(r-1.0, torch.tensor(1.0))
                # volume from MC integration
                volume = torch.sum(f/p) / n
                # expected volume of unit sphere Vn
                volume_exact = (np.pi)**(dim/2) / special.gamma(dim/2+1)
                logger.info(f"d= {dim}  V_dim= {volume:.5f} (Monte-Carlo)   {volume_exact:.5f} (exact)")

                self.assertTrue( abs(volume - volume_exact) < 0.05 )

    def test_fills_unit_ball(self):
        """
        check that approximately half the samples lie inside the unit ball
        """
        # make random numbers reproducible
        torch.manual_seed(0)

        logger.info(f"Volume of unit sphere in d dimensions")

        for dim in [1,2,3,4,5,10,20,30]:
            with self.subTest(dimension=dim):
                n = 100000
                distribution = UnitBallFillingDistribution(dim)
                x = distribution.sample(n)
                r = torch.norm(x, dim=0)
                # count samples with |x| <= 1
                n_inside = torch.count_nonzero(r <= 1.0)

                logger.info(f"dimension= {dim}  ratio of samples inside unit ball= {n_inside/n:.3f}")
                self.assertTrue(n_inside / n > 0.5) 
                self.assertTrue(n_inside / n < 0.8)
                
if __name__ == "__main__":
    unittest.main()
