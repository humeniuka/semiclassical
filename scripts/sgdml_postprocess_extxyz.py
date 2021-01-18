#!/usr/bin/env python
"""
The global sign of the NAC vectors is arbitrary. To obtain a continuous vector field
the sign is chosen so as to maximize the overlap with the NAC vector at the initial
geometry.
"""
import ase
from ase import io
import sys
import numpy as np
import numpy.linalg as la
import argparse
import tqdm

parser = argparse.ArgumentParser(
    description="""Postprocesses forces or non-adiabatic coupling vectors in extended XYZ file. 

 * Rigid rotation/translation can be removed from the vector field so that the forces depend only on internal coordinates.
 * To make the vector field continuous, the overall sign of the vectors at each geometry can be chosen so as to minimize 
   the angle with vectors at neighbouring geometries.""",
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

args = parser.parse_args()

assert args.input_xyz != args.output_xyz, "Input and output files should be different."
assert args.remove_rotation or args.remove_translation or args.align_signs, "At least one operation (-s and/or -r, -t) has to be performed."

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
    iL = np.argmin(momlen)
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

molecules = io.read(args.input_xyz, format="extxyz", index=":")

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
    
# save processed molecules
with open(args.output_xyz, "w") as f:
    for i,mol in enumerate(molecules):
        io.extxyz.write_extxyz(f, [mol], columns=['symbols', 'positions', 'momenta'], write_results=False, append=True)

