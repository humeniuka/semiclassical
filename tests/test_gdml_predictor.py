#!/usr/bin/env python
# coding: utf-8
"""
unit tests for machine-learned sGDML model
"""
import unittest

import numpy as np
import numpy.linalg as la
import torch
import logging
import time

from semiclassical.gdml_predictor import  GDMLPredict

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[testing] %(message)s", level=logging.INFO)

import torch
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    logger.info("CUDA available")
    # If there are several GPU's available, we use the last one,
    # i.e. "cuda:1" on a workstation with 2 GPUs.
    device = torch.device("cuda:%d" % (torch.cuda.device_count()-1))
else:
    device = torch.device('cpu')

class TestGDMLPredict(unittest.TestCase):
    def setUp(self):
        try:
            from sgdml.torchtools import GDMLTorchPredict
            from sgdml.utils import io
            from ase.units import Bohr
        except ImportError as err:
            logger.error("This test case requires the sgdml package (https://github.com/stefanch/sGDML) and the ase (Atomic Simulation Environment) package")
            raise err
        # model fitted to ground state forces of coumarin
        model = np.load('DATA/GDML/coumarin_forces_au-wB97XD_def2SVP-train200-sym1.npz', allow_pickle=True)
        # reference implementation
        self.gdml_ref = GDMLTorchPredict(model)
        # new implementation with analytical Hessians
        self.gdml = GDMLPredict(model).to(device)
        # load geometry
        r,_ = io.read_xyz('DATA/GDML/coumarin.xyz')
        # convert from Angstrom to bohr
        r /= Bohr
        self.coords = torch.from_numpy(r).to(device)
        
    def test_energies_gradients(self):
        """check that energies and gradients agree with reference implementation"""
        # make random numbers reproducible
        torch.manual_seed(0)

        natom = self.coords.size()[1]//3
        # timing for different batch sizes
        for batch_size in [1,10,100,1000,5000,10000]:
            logger.info(f"batch size {batch_size}")
            # batch (B,3*N)
            rs = self.coords.repeat(batch_size, 1) + 0.1 * torch.rand(batch_size,3*natom).to(device)
            # (B, N, 3)
            rs_3N = rs.reshape(batch_size, -1, 3)

            t_start = time.time()
            # compute energy and Hessian with reference implementation
            en_ref, force_ref = self.gdml_ref.forward(rs_3N)
            grad_ref = -force_ref.reshape(rs.size())
            
            t_end = time.time()
            logger.info(f"timing reference implementation, energy+gradient    : {t_end-t_start} seconds")

            t_start = time.time()
            # and compare with this implementation
            en, grad, hessian = self.gdml.forward(rs)

            t_end = time.time()
            logger.info(f"timing new implementation, energy+gradient+hessian  : {t_end-t_start} seconds")
            
            # error per sample
            err_en = torch.norm(en_ref - en)/batch_size
            err_grad = torch.norm(grad_ref - grad)/batch_size

            logger.info(f"   error of energy   : {err_en}")
            logger.info(f"   error of gradient : {err_grad}")
            
            self.assertTrue(err_en < 1.0e-6)
            self.assertTrue(err_grad < 1.0e-6)

    def test_hessian(self):
        """compare numerical and analytic Hessians of sGDML potential"""
        from sgdml.intf.ase_calc import SGDMLCalculator
        from ase.io.xyz import read_xyz
        from ase.optimize import BFGS
        from ase.vibrations import Vibrations
        from ase.units import Bohr, Hartree

        # compute Hessian numerically using ASE
        with open('DATA/GDML/coumarin.xyz') as f:
            mol = next(read_xyz(f))
        sgdml_calc = SGDMLCalculator('DATA/GDML/coumarin_forces_au-wB97XD_def2SVP-train200-sym1.npz',
                                     E_to_eV=Hartree,
                                     F_to_eV_Ang=Hartree/Bohr)
        mol.calc = sgdml_calc
        vib = Vibrations(mol, name="/tmp/vib_sgdml")
        vib.run()
        vib.get_energies()
        vib.clean()
        
        hessian_numerical = vib.H * Bohr**2 / Hartree

        # compute analytic Hessian directly from sGDML model
        hessian_analytical = self.gdml.forward(self.coords)[2][0,:,:].cpu().numpy()

        # check that Hessian is symmetric
        err_sym = la.norm(hessian_analytical - hessian_analytical.T)
        logger.info(f"|Hessian-Hessian^T|= {err_sym}")
        self.assertTrue(err_sym < 1.0e-10)

        err = la.norm(hessian_numerical - hessian_analytical)
        logger.info(f"|Hessian(num)-Hessian(ana)|= {err}")
        self.assertTrue(err < 1.0e-3)
        
        
            
if __name__ == "__main__":
    unittest.main()
