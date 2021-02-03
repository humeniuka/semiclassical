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
import scipy.linalg as sla
from collections import OrderedDict
import re
import logging
import ase.atoms

# # Local Imports
from semiclassical import units
from semiclassical.units import hbar

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
        energy :  ndarray (1,)
          total energy E(x0) of state of interest (in Hartree) 
        grad   :  ndarray (3*nat,)
          cartesian gradient dE/dx(x0)  (in Hartree/bohr)
        hess   :  ndarray (3*nat,3*nat)
          cartesian force constants d^2E/(dxdx)(x0)  (in Hartree/bohr^2)
        """
        try:
            nat = self.data["Number of atoms"]
            # total energy of state of interest
            energy = np.array(self.data["Total Energy"])
            # geometry
            pos = self.data["Current cartesian coordinates"]
            # cartesian gradient
            grad = self.data["Cartesian Gradient"]
            # Only the lower triangular part of the Hessian is stored.
            hess = np.zeros((3*nat,3*nat))
            row, col = np.tril_indices(3*nat)
            hess[row,col] = self.data["Cartesian Force Constants"]
            # Hessian is symmetric, H^T = H
            hess[col,row] = hess[row,col]
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
    def vibrational_groundstate(self):
        """
        The vibrational ground state belonging to the harmonic potential is given by

                                          1/4                 T
           psi (x) = (det(Gamma ) / pi^N)    exp{ -1/2 (x-x )  Gamma  (x-x ) }
              0                0                           0        0     0

        provided that x0 is the minimum. This function computes the width parameter matrix
        Gamma_0 from the Hessian at the minimum. Rotational and translational modes are 
        projected out.

        Returns
        -------
        x0      : ndarray (3*nat,)
          center of Gaussian, in cartesian coordinates (bohr)
        Gamma0  : ndarray (3*nat,3*nat)
          symmetric, positive semi-definite matrix of width parameters (bohr^{-2})
        en_zpt  : float
          zero-point energy (Hartree)
        """
        x0, energy, grad, hess = self.harmonic_approximation()
        mass = self.masses()
        # diagonals of M^{1/2} and M^{-1/2}
        msq = np.sqrt(mass)
        imsq = 1.0/msq
        # mass-weighted Hessian H
        hess_mwc = np.einsum('i,ij,j->ij', imsq, hess, imsq)

        # diagonalize symmetric  H = V.diag(w).V^T
        w2,V = sla.eigh(hess_mwc)

        # vibrational energies
        w = np.sqrt(w2+0j)

        logger.info("Normal mode frequencies (cm-1) before eliminating translation/rotation")
        logger.info(w*units.hartree_to_wavenumbers)

        # The molecular potential only depends on the internal degrees of freedom and does not
        # change if the molecule is rotated or translated. Therefore 5 (in linear molecules)
        # or 6 (in non-linear molecules) normal modes have zero frequency. Because of numerical
        # errors the eigenvalues of these rotational and translational modes might differ from 0
        # so that they can be confused with internal vibrational modes with low frequencies.
        # To separate the rotations/translations from internal vibrations, the normal modes for
        # rotation and translation are constructed and the Hessian is transformed into a basis
        # where the first 5 or 6 basis vectors correspond to translations/rotations.
        
        # project out rotation and translation
        mol = ase.atoms.Atoms(numbers=self.atomic_numbers())
        mol.set_positions(x0.reshape(-1,3))
        # shift molecule to center of mass
        mol.set_masses(mass[::3])
        center = mol.get_center_of_mass()
        logger.info(f"center of mass (Ang) : {center * units.bohr_to_angs}")
        mol.translate(-center)
        r = mol.get_positions()
        
        # 1) find principle axes of inertia
        # eigenvalues and eigenvectors of tensor of inertia
        principal_moments, Xrot = mol.get_moments_of_inertia(vectors=True)
        # Xrot[i,:] is the eigenvector belonging to the i-th principal moment

        # 2) generate matrix D for transformation from mass-weighted cartesian coordinates
        # to internal coordinates, the first 3 vectors are for translation, the next 2 or 3 for rotation
        # and the remaining 3*nat-5 or 3*nat-6 vectors are for internal vibrations.

        D = np.zeros_like(hess_mwc)
        # mass-weighted coordinates
        mwc = msq.reshape((-1,3)) * r

        # translation
        for i in [0,1,2]:
            # rigid shift along x-, y- and z-axis
            #            [1 0 0]
            # sqrt(mi) * [0 1 0]  for each atom i
            #            [0 0 1]
            D[i::3,i] = msq[i::3]
            
        nat, _ = r.shape
        # rotations, `nz` counts numbers of zero modes
        nz = 3
        for i in [0,1,2]:
            if principal_moments[i] > 0.0:
                # rigid rotation around principal axes
                #  dr = omega x (sqrt(m) r)
                D[:,nz] = np.cross(Xrot[i,:], mwc).reshape(-1)
                nz += 1
                
        # normalize
        for i in range(0,nz):
            D[:,i] /= sla.norm(D[:,i])

        # 3) Gram-Schmidt orthogonalization with respect to the remaining vectors
        dim,_ = D.shape
        for n in range(nz, dim):
            D[:,n] = V[:,n]
            for m in range(0, n):
                # numerically stable modified Gram-Schmidt
                D[:,n] -= np.dot(D[:,m],D[:,n])*D[:,m] 
            D[:,n] /= sla.norm(D[:,n])

        # check that D is really an orthonormal basis
        err = sla.norm(np.dot(D.T, D) - np.eye(dim))
        assert err < 1.0e-10, f"Gram-Schmidt orthogonalization failed, |D^T.D-Id|= {err}"

        logger.info(f"rotational/translational modes   : {nz}")
        if (nz == 3):
            logger.error("All principal moments of inertia are zero. Is this a single atom?")
        elif (nz == 5):
            logger.info("found a linear molecule")
        elif (nz == 6):
            logger.info("found a non-linear molecule")
        else:
            logger.error(f"Strange number of rotational/translation modes, expected to find 3 (atom), 5 (linear) of 6 (general molecule), but got {nz}")

        # 4) Transform mass-weighted Hessian matrix to internal coordinates and diagonalize
        #  Hi = D^T . Hmwc . D
        hess_internal = np.einsum('ij,jk,kl->il', D.T, hess_mwc, D)

        # diagonalize symmetric  Hi = Vi.diag(w).Vi^T
        # The subblocks for external zero modes and the subblock belonging to
        # internal vibrations are diagonalized separately.
        
        # external zero modes
        wz2,Vz = sla.eigh(hess_internal[:nz,:nz])
        wz = np.sqrt(wz2+0j)
        # If coupling between rotations and vibrations is significant, 3 of these
        # frequencies may differ from 0.0.
        logger.info("Frequencies (cm-1) of translations and rotations")
        logger.info(wz*units.hartree_to_wavenumbers)
        
        # internal vibrational modes
        wi2,Vi = sla.eigh(hess_internal[nz:,nz:])
        wi = np.sqrt(wi2)

        logger.info("Vibrational frequencies (cm-1) after eliminating translation/rotation")
        logger.info(wi*units.hartree_to_wavenumbers)

        if not (wi * units.hartree_to_wavenumbers > 0.0).all():
            logger.error("At a minimum all vibrational frequencies should be positive, found imaginary ones.")

        # zero-point energy
        en_zpt = 0.5 * hbar * np.sum(wi)
        logger.info(f"Zero point energy (cm-1) : {en_zpt * units.hartree_to_wavenumbers}")

        # transform normal modes back from internal coordinates to mass-weighted cartesian ones
        V = np.dot(D[:,nz:], Vi)
        # L = hbar^{-1/2} M^{1/2} D Vi w^{1/2}
        L = hbar**(-1/2) * np.einsum('i,ij,j->ij', msq, V, np.sqrt(wi))

        # Gamma_0 = L . L^T
        Gamma_0 = np.einsum('ij,kj->ik', L, L)
        
        return x0, Gamma_0, en_zpt
        
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
    
    def atomic_numbers(self):
        """
        atomic numbers 

        Returns
        -------
        numbers :  ndarray(nat,)
          atomic number for each atom
        """
        return self.data["Atomic numbers"]

    def total_energy(self):
        """
        Returns
        -------
        energy :  float
          total energy of state of interest in Hartree
        """
        return self.data["Total Energy"]
