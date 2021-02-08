#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import numpy.linalg as la
import argparse
import tqdm

import ase.io
from ase.io import extxyz

parser = argparse.ArgumentParser(
    description="""Postprocesses forces or non-adiabatic coupling vectors in extended XYZ file. 

 * Rigid rotation/translation can be removed from the vector field so that the forces depend only on internal coordinates.
 * To make the vector field continuous, the overall sign of the vectors at each geometry can be chosen so as to minimize 
   the angle with vectors at neighbouring geometries.
 * Ensure correct dissociation limit by augmenting the dataset with a set of scaled geometries where all bonds are broken.
""",
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('input_xyz',
                    metavar='input.xyz',
                    type=str,
                    help='Forces or NAC vectors are read from this extended xyz file.'
                    )

parser.add_argument('output_xyz',
                    metavar='output.xyz',
                    type=str,
                    help='The postprocessed forces or NAC vectors are written to this extended xyz file.',
                    )

parser.add_argument(
    '-r',
    '--remove_rotation',
    dest='remove_rotation',
    action='store_true',
    help='remove rigid rotation from vector field (make total angular momentum vanish)')

parser.add_argument(
    '-t',
    '--remove_translation',
    dest='remove_translation',
    action='store_true',
    help='remove rigid translation (make total linear momentum vanish)')

parser.add_argument(
    '-s',
    '--align_signs',
    dest='align_signs',
    action='store_true',
    help='align signs of NAC vectors so as to make the vector field continuous')

parser.add_argument(
    '-d',
    '--fix_dissociation',
    dest='fix_dissociation',
    action='store_true',
    help='add dissociated geometries with zero forces and energies equal to sum of atomic energies so as to enforce correct dissociation limit')
parser.add_argument(
    '--dissoc_limit',
    dest='dissoc_limit',
    type=float,
    default=None,
    metavar='ENERGY',
    help='set dissociation limit explicity (in Hartree), if not set the dissociation limit is computed from atomic wB97XD/def2SVP energies')

args = parser.parse_args()

assert args.input_xyz != args.output_xyz, "Input and output files should be different."
assert (   args.remove_rotation
        or args.remove_translation
        or args.align_signs
        or args.fix_dissociation), "At least one operation (-s and/or -r, -t, -d) has to be performed."

def eliminate_translation(mol):
    """
    remove rigid translation so that the total linear momentum vanishes

    Parameters
    ----------
    mol    :  ase.atoms.Atoms
      molecule with momenta
    
    Returns
    -------
    nothing, momenta of `mol` are changed in place
    """
    # shift molecule to center of mass
    center = mol.get_center_of_mass()
    mol.translate(-center)

    # total linear momentum
    linmom = np.sum(mol.get_momenta(), axis=0)

    masses = mol.get_masses()[:,np.newaxis]
    totmass = np.sum(masses)
    velocities = mol.get_momenta() / masses

    velocities = velocities - linmom[np.newaxis,:] / totmass
    mol.set_momenta(velocities * masses)

    # Total angular momentum should be zero now.
    linmom = np.sum(mol.get_momenta(), axis=0)
    assert abs(linmom).max() < 1.0e-10, "Rigid translation could not be eliminated."

    
def eliminate_rotation(mol):
    """
    remove rigid rotation so that the total angular momentum vanishes

    Parameters
    ----------
    mol    :  ase.atoms.Atoms
      molecule with momenta
    
    Returns
    -------
    nothing, momenta of `mol` are changed in place
    """
    # shift molecule to center of mass
    center = mol.get_center_of_mass()
    mol.translate(-center)
    
    # total angular momentum
    angmom = mol.get_angular_momentum()
    # eigenvalues and eigenvectors of tensor of inertia
    e,v = mol.get_moments_of_inertia(vectors=True)
    # solve L = I*w for angular velocity
    omega = np.einsum('ij,j,jk,k->i', v.T, 1.0/e, v, angmom)

    masses = mol.get_masses()[:,np.newaxis]
    velocities = mol.get_momenta() / masses
    # v = v - omega x r
    velocities -= np.cross(omega, mol.get_positions())
    mol.set_momenta(velocities * masses)

    # Total angular momentum should be zero now.
    angmom = mol.get_angular_momentum()
    assert abs(angmom).max() < 1.0e-10, "Rigid rotation could not be eliminated."

    
def align_phases(molecules):
    """
    turn momentum vectors with random signs into continuous vector field by choosing signs
    so as to minimize the angle with vectors at neighbouring points.

    The momenta of the molecules are samples from a vector field that should be continuous.
    However, the global phases (+1,-1) of the vectors are random. This happens because the
    non-adiabatic coupling vectors (stored as momenta in the Atoms object) come out with
    random phases. To make the vector field continuous, momentum vectors of some molecules 
    have to be flipped. 

    Parameters
    ----------
    molecules   :   list of ase.atoms.Atoms
      list of molecules with momenta, the input list is destroyed
    Returns
    -------
    assigned    :   list of ase.atoms.Atoms
      list of molecules with aligned momenta, order of molecules is changed
    """
    # We start with the molecule with the largest momentum vector. 
    momlen = [la.norm(mol.get_momenta()) for mol in molecules]
    iL = np.argmax(momlen)
    assert momlen[iL] > 0.0
    # `assigned` is the list of molecules whose phases have been determined
    assigned = [molecules[iL]]
    del molecules[iL]

    # Molecules are divided into two sets, those for which the signs of the momenta have
    # been determined (set A) and those with still undetermined signs (set M). In each iteration
    # one molecule is selected from set M and moved to set A.
    with tqdm.tqdm(total=len(molecules)) as progress_bar:
        while len(molecules) > 0:
            # We need to choose the next molecule (molM) whose vectors should be aligned.
            # It should be the molecule in set M that is closest to
            dists_M = [la.norm(assigned[-1].get_positions() - mol.get_positions()) for mol in molecules]
            iM = np.argmin(dists_M)
            molM = molecules[iM]
            # Find the closest molecule to molM for which we have determined the phase already.
            dists_A = [la.norm(molM.get_positions() - molA.get_positions()) for molA in assigned]
            iA = np.argmin(dists_A)
            molA = assigned[iA]

            # Align vectors of molM to molA
            dotprod = np.sum(molM.get_momenta() * molA.get_momenta())
            normM = la.norm(molM.get_momenta())
            normA = la.norm(molA.get_momenta())
            #                 v(A) . v(M)
            # cos(alpha) = -----------------
            #                |v(A)| |v(M)|
            cos = dotprod / (normA * normM)
            if cos < 0.0:
                # vectors are anti-parallel => switch sign
                signum = -1.0
            else:
                # vectors are parallel
                signum = 1.0
            molM.set_momenta( molM.get_momenta() * signum )

            if abs(cos) > 0.8:
                # move molM from set M to set A
                assigned.append(molM)
            del molecules[iM]

            progress_bar.update(1)
            progress_bar.set_description(f"   cos(angle)= {cos:+f}   assigned= {len(assigned)}   remaining= {len(molecules)}")
            if (abs(cos) < 0.8):
                print("   Warning: cos(angle) < 0.8, NAC vectors differ too much. Alignment is probably unreliable. Skip this geometry.")
            
    return assigned



def fix_dissociation_limit(molecules, dissoc_limit=None, scale=3.0, n=1000):
    """
    expand the data set by adding fully dissociated molecules with forces=0 and energies
    equal to the dissociation energy

    This is supposed to ensure that the fitted potential energy surface has the correct
    dissociation limit.

    Parameters
    ----------
    molecules    :   list of ase.atoms.Atoms
      list of molecules from original data set
    dissoc_limit :   float or None
      dissociation limit (in Hartree), if None compute limit from atomic energies
    scale        :   float > 2.0
      All coordinates are scaled by this factor, it should be large enough
      so that all bonds in the scaled geometry are broken
    n            :   int
      Only the first `n` geometries are scaled

    Returns
    -------
    scaled    :   list of ase.atoms.Atoms (n,)
      list of dissociated molecules
    """
    # Energies of isolated atoms in Singlet or Double ground state
    # at the wB97XD/def2SVP level of theory (in Hartree)
    _atomic_energies = {
        'H' :   -0.5018819259,
        'C' :  -37.7281684180,
        'N' :  -54.4087068942,
        'O' :  -74.8713971303,
        'F' :  -99.6122174159,
        'S' : -397.9008049257,
        'Cl': -459.9762645185
    }

    # If all bonds in the molecule are broken, the energy should be equal
    # to the sum of the atomic energies (assuming all atoms are in the ground state)
    # This is the dissociation energy to which the ground state energy
    # of a molecule should converge as the bonds are elongated.
    assert len(molecules) > 0, "list of molecules is empty"
    atoms = molecules[0]
    if dissoc_limit is None:
        print(f"computing dissociation energy as sum of atomic wB97XD/def2SVP energies")
        assert set(atoms.get_chemical_symbols()) - set(['H', 'C', 'N', 'O', 'F', 'S', 'Cl']) == set(), "Atomic energies are only available for H,C,N,O,F,S,Cl"
        dissoc_limit = np.sum([_atomic_energies[elem] for elem in atoms.get_chemical_symbols()])
    print(f"dissociation energy        : {dissoc_limit:10.5f} Hartree")
    # energies of samples
    energies = [atoms.info["Energy"] for atoms in molecules]
    print(f"average energy of samples :  {np.mean(energies):10.5f} Hartree")
    print(f"minimum energy of samples :  {np.min(energies):10.5f} Hartree")
    print(f"maximum energy of samples :  {np.max(energies):10.5f} Hartree")
    
    scaled = []
    for i in range(0, min(len(molecules), n)):
        mol = molecules[i].copy()
        mol.set_positions(scale * mol.get_positions())
        # set energy to dissociation energy
        mol.info["Energy"] = dissoc_limit
        # set gradient of energy to 0
        mol.set_momenta(0.0 * mol.get_momenta())
        scaled.append(mol)

    return scaled
            
def _properties_parser(comment_line):
    dic = extxyz.key_val_str_to_dict(comment_line)
    # If the extended xyz file lacks a property string,
    # we assume that there are 7 columns with symbols, coordinates and forces
    if not 'Properties' in dic:
        dic['Properties'] = 'species:S:1:pos:R:3:forces:R:3'
    return dic

# read all frames from trajectory with forces in extended XYZ format
molecules = ase.io.read(args.input_xyz, index=":",
                        format="extxyz",
                        properties_parser=_properties_parser)
assert len(molecules) > 0, f"Could not read any molecule from input file {args.input_xyz}."

# ASE does not allow assigning forces, therefore the forces are stored as momenta.
for i,mol in enumerate(molecules):
    try:
        # Read forces and store them as momenta.
        #  Properties=species:S:1:pos:R:3:forces:R:3
        mol.set_momenta(mol.get_forces())
    except RuntimeError:
        #  Properties=species:S:1:pos:R:3:momenta:R:3
        pass

if args.align_signs:
    print(" * aligning signs to make vector field continuous")
    molecules = align_phases(molecules)

if args.remove_translation:
    print(" * eliminating rigid translation")
    for i,mol in enumerate(molecules):
        eliminate_translation(mol)
        
if args.remove_rotation:
    print(" * eliminating rigid rotation to make vector field curl-free")
    for i,mol in enumerate(molecules):
        eliminate_rotation(mol)

if args.fix_dissociation:
    print(" * add fully dissociated molecules to dataset to fix dissociation limit")
    molecules += fix_dissociation_limit(molecules)
        
# save processed molecules
with open(args.output_xyz, "w") as f:
    for i,mol in enumerate(molecules):
        ase.io.extxyz.write_extxyz(f, [mol], columns=['symbols', 'positions', 'momenta'], write_results=False, append=True)

