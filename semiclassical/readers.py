#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
interfaces for reading output of different quantum chemistry codes

The following file formats are supported:

 * formatted checkpoint files (*.fchk)

"""
__all__ = ["FormattedCheckpointFile"]

# # Imports
import numpy as np
from collections import OrderedDict
import re
import logging

# # Local Imports
from semiclassical import units

# # Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(module)-12s] %(message)s", level=logging.INFO)

class FormattedCheckpointFile(object):
    """
    reads all fields from formatted checkpoint files produced by the quantum chemistry
    programs Gaussian 16 and QChem.

    Parameters
    ----------
    f    :   File
      file handle opened for reading a formatted checkpoint file
      The user has to ensure the file handle is opened and closed at the end.

    The fields of the checkpoint file can be accessed by their names (see example below).
    Array fields are stored as 1D numpy arrays of float (R) or integer (I) type.

    Example
    -------

      >>> with open("freq.fchk") as f:
      >>>   fchk = FormattedCheckpointFile(f)
      >>>   print(fchk["Number of atoms"])

    """
    def __init__(self, f):
        self.filename = f.name
        self.data = OrderedDict()
        # accumulate all lines belonging to the same field (whithout newlines)
        acc = ""
        dtype = None
        for line_number, line in enumerate(f.readlines()):
            # count lines starting from 1
            line_number += 1
            # The name of a field starts in the first column and with a capital letter
            if re.match(r"^[A-Z].*", line):
                if len(acc) > 0 and not dtype is None:
                    # All lines belonging to the previous field must have been read,
                    # so we convert it to a numpy array.
                    try:
                        if dtype == str:
                            self.data[field] = acc
                        else:
                            # numerical types
                            array = np.fromstring(acc, dtype=dtype, sep=" ")
                            assert len(array) == count
                            self.data[field] = array
                    except (ValueError,AssertionError) as err:
                        logger.warning(f"A problem occurred reading field `{field}` in line {line_number:10} in {f.name} .")
                        logger.warning(err)
                        self.data[field] = np.zeros(count, dtype=dtype)
                # reset accumulator
                acc = ""

                try:
                    if len(line) < 43:
                        # skip title and method
                        logger.debug(f"skipping line {line_number:10} in {f.name}: `{line.strip()}`")
                        continue
                    # First 43 columns are reserved for the field name
                    field = line[0:43].strip()
                    logger.debug(f"field `{field}` encountered")
                    # Colum 43 contains a character indicating the data type:
                    #   I -> integer
                    #   R -> real
                    type_char = line[43]
                    if type_char == "I":
                        dtype = int
                    elif type_char == "R":
                        dtype = float
                    elif type_char == "C":
                        dtype = str
                    else:
                        dtype = None
                        # skip lines without I or R data type markers
                        logger.debug(f"skipping line {line_number:10} in {f.name}: `{line.strip()}` .")
                        continue
                    # If column 47-48 contain the string "N=",  we are dealing with an array
                    # and the last integer indicates the number of elements
                    if line[47:49] == "N=":
                        count = int(line[49:])
                    else:
                        # scalar value
                        self.data[field] = dtype(line[49:])
                except Exception as err:
                    logger.error(f"An error occurred while reading line {line_number:10} in {f.name} .")
                    raise err
            else:
                acc += " " + line
        # read last field
        if len(acc)  > 0:
            self.data[field] = np.fromstring(acc, dtype=dtype, sep=" ")
            assert len(self.data[field]) == count    
    def __getitem__(self, key):
        """
        access data fields by their names

        Parameters
        ----------
        key     :   str
          name of field that should be retrieved (e.g. 'Number of atoms')

        Returns
        -------
        field   :  float, int or ndarray
          a KeyError is raised if the field is not present in the formatted checkpoint file
        """
        return self.data[key]
    def keys(self):
        """
        list names of all fields present in the formatted checkpoint file

        Returns
        -------
        keys  :  list of str
          field names
        """
        return self.data.keys()
    def harmonic_approximation(self):
        """
        extract the position, gradient and Hessian of the potential energy in cartesian coordinates 
        
        The potential is expanded to second order around the current position x0:

             E(x) = E(x0) + grad(E)^T.(x-x0) + 1/2 (x-x0)^T . hess(E) . (x-x0)

        A frequency calculation has to be present in the formatted checkpoint file.
        The frequency calculation should be performed in a separate Gaussian 16 job using the
        following route line for the ground state calculation:

             #P functional/basis  Freq NoSymm IOp(7/32=5)

        and the following route line for an excited state frequency calculation:

             #P functional/basis TD=(Nstates=2, Root=1, NAC) Freq NoSymm IOp(7/32=5)

        Returns
        -------
        pos    :  ndarray (3*nat,)
          cartesian coordinates x0
        energy :  float
          total energy E(x0) of state of interest (in Hartree) 
        grad   :  ndarray (3*nat,)
          cartesian gradient dE/dx(x0)  (in Hartree/bohr)
        hess   :  ndarray (3*nat,3*nat)
          cartesian force constants d^2E/(dxdx)(x0)  (in Hartree/bohr^2)
        """
        try:
            nat = self.data["Number of atoms"]
            # total energy of state of interest
            energy = self.data["Total Energy"]
            # geometry
            pos = self.data["Current cartesian coordinates"]
            # cartesian gradient
            grad = self.data["Cartesian Gradient"]
            # Only the lower triangular part of the Hessian is stored.
            hess = np.zeros((3*nat,3*nat))
            row, col = np.tril_indices(3*nat)
            hess[row,col] = self.data["Cartesian Force Constants"]
        except KeyError as err:
            logger.error(f"A required field could not be found in formatted checkpoint file {self.filename} .")
            raise err
        return pos, energy, grad, hess
    def nonadiabatic_coupling(self):
        """
        extract non-adiabatic coupling vector between ground and excited state (Root=I), if present.

        Only Gaussian 16 saves the NAC vector in the checkpoint file, while QChem writes it to the output file.

        Returns
        -------
        nac  :  ndarray (3*nat,)
          1st order derivative coupling <0|d/dx|I>
        """
        try:
            nac = self.data["Nonadiabatic coupling"]
        except KeyError as err:
            logger.error(f"The field `Nonadiabatic coupling` could not be found in the formatted checkpoint file {self.filename} .")
            raise err
        if (nac == 0.0).all():
            logger.warning(f"All components of non-adiabatic coupling vector in {self.filename} are zero.")
        return nac
    def masses(self):
        """
        atomic masses in a.u.

        Returns
        -------
        masses  :  ndarray (3*nat,)
          masses for each cartesian coordinate in multiples of electron mass
        """
        mass = self.data["Real atomic weights"] * units.amu_to_aumass
        mass = np.repeat(mass, 3)
        return mass

