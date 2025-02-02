# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sphecerix import Molecule, BasisFunction, SymmetryOperations,\
                      visualize_matrices, CharacterTable, ProjectionOperator

def main():
    mol = Molecule()
    mol.add_atom('C', 0.0,  0.0, 0.0, unit='angstrom')
    mol.add_atom('H', 1,1,1, unit='angstrom')
    mol.add_atom('H', 1,-1,-1, unit='angstrom')
    mol.add_atom('H', -1,1,-1, unit='angstrom')
    mol.add_atom('H', -1,-1,1, unit='angstrom')
    
    molset = {
        'C': [BasisFunction(1,0,0),
              BasisFunction(2,0,0),
              BasisFunction(2,1,1),
              BasisFunction(2,1,-1),
              BasisFunction(2,1,0)],
        'H': [BasisFunction(1,0,0)]
    }
    mol.build_basis(molset)
    
    symops = SymmetryOperations(mol)
    symops.add('identity')
    
    # add C3 rotations
    for i in range(0,4):
        axis = mol.atoms[i+1][1]
        axis /= np.linalg.norm(axis) # normalize axis
        for j in range(0,2):
            symops.add('rotation', '3,%i' % (i*2+j+1), axis, (-1)**j * 2.0 * np.pi / 3)
    
    # C2 rotations
    for i in range(0,3):
        axis = np.zeros(3)
        axis[i] = 1.0
        symops.add('rotation', '2,%i' % (i+1), axis, np.pi)
        
    # S4 rotations
    for i in range(0,3):
        axis = np.zeros(3)
        axis[i] = 1.0
        for j in range(0,2):
            symops.add('improper', '4,%i' % (i+1), axis, (-1)**j * np.pi/2)

    # sigma_d mirror planes
    ctr = 0
    for i in range(0,4):
        axis1 = mol.atoms[i+1][1]
        for j in range(i+1,4):
            axis2 = mol.atoms[j+1][1]
            normal = np.cross(axis1, axis2)
            normal /= np.linalg.norm(normal)
            ctr += 1
            symops.add('mirror', ',d(%i)' % (ctr), normal)

    symops.run()
    
    visualize_matrices(symops.operation_matrices, 
                        [op.name for op in  symops.operations],
                        [bf.name for bf in symops.mol.basis], 
                        xlabelrot=90, figsize=(18,12), numcols=6)
    
    # print result LOT
    ct = CharacterTable('td')
    print(ct.lot(np.trace(symops.operation_matrices, axis1=1, axis2=2)))
    
    # # apply projection operator
    po = ProjectionOperator(ct, symops)
    mos = po.build_mos(verbose=True)
    newmats = [mos @ m @ mos.transpose() for m in symops.operation_matrices]
    
    visualize_matrices(newmats, 
                        [op.name for op in  symops.operations],
                        ['$\phi_{%i}$' % (i+1) for i in range(len(symops.mol.basis))],
                        xlabelrot=90, figsize=(18,12), numcols=6,
                        highlight_groups=po.get_blocks())

if __name__ == '__main__':
    main()