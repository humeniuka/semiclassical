#!/usr/bin/env python
# coding: utf-8
"""
runs input scripts from folders in DATA/examples 
and compare with the expected results
"""
import unittest
import glob
import logging
import torch
import shutil, tempfile
import os.path
import json
import numpy as np

from semiclassical import cli

# For each example calculation there should be a separate folder
# in EXAMPLES_DIR.
EXAMPLES_DIR = "DATA/examples"

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[testing] %(message)s", level=logging.INFO)

class cd:
    """
    context manager for temporarily changing the working directory

    stolen from: https://stackoverflow.com/questions/431684/how-do-i-change-the-working-directory-in-python
    """
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)
    def __enter__(self):
        self.old_path = os.getcwd()
        os.chdir(self.new_path)
    def __exit__(self):
        os.chdir(self.old_path)
        

class TestExamples(unittest.TestCase):
    
    def setUp(self):
        # run on GPU or CPU ?
        global device
        torch.set_default_dtype(torch.float64)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # create temporary folder
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Tests are run in temporary folder {self.temp_dir}")

    def tearDown(self):
        # remove temporary folder after tests have finished
        shutil.rmtree(self.temp_dir)
        
    def _run_example(self, directory):
        """
        runs the input script `semi.json` in the folder `directory` and
        checks that files marked with .CHECK are reproduced correctly.
        """
        logger.info(f"Testing {directory}")
        # copy input to temporary folder
        test_dir = os.path.join(self.temp_dir, directory)
        shutil.copytree(directory, test_dir)
        
        # enter temporary directory and run example
        with cd(test_dir):
            
            # run calculations in temporary folder, the JSON input should
            # always be called 'semi.json'
            with open("semi.json") as f:
                config = json.load(f)

            for task in config['semi']:
                if task['task'] == 'dynamics':
                    cli.run_semiclassical_dynamics(task, device=device)

            for task in config['semi']:
                if task['task'] == 'rates':
                    cli.calculate_rates(task)

            # compare output files with expected outputs. For each file that
            # should be compared there should be a file with the same name and
            # the extension *.CHECK .
            # For example,
            #  correlations.npz  should match with  correlations.npz.CHECK
            for filename_check in glob.glob("*.CHECK"):
                filename, _ = os.path.splitext(filename_check)
                if filename.endswith(".npz"):
                    # compare content of .npz file
                    data_check = np.load(filename_check)
                    data = np.load(filename)

                    # compare correlation functions
                    self.assertTrue( np.array_equal(data['autocorrelation'], data_check['autocorrelation']) )
                    self.assertTrue( np.array_equal(data['ic_correlation'], data_check['ic_correlation']) )
                    # compare rates
                    self.assertTrue( np.array_equal(data['ic_rates'], data_check['ic_rates']) )
                    
    def test_all_examples(self):
        logger.info(f"Test all examples in {EXAMPLES_DIR}")
        for directory in glob.glob(f"{EXAMPLES_DIR}/*"):
            with self.subTest(example=directory):
                self._run_example(directory)
                
if __name__ == "__main__":
    unittest.main()

    
