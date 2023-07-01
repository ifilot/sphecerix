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
    mol.add_atom('N', 0.00000000, 0.00000000, -0.06931370, unit='angstrom')
    mol.add_atom('H', 0.00000000, 0.94311105,  0.32106944, unit='angstrom')
    mol.add_atom('H', -0.81675813, -0.47155553, 0.32106944, unit='angstrom')
    mol.add_atom('H', 0.81675813, -0.47155553, 0.32106944, unit='angstrom')
    
    molset = {
        'N': [BasisFunction(1,0,0),
              BasisFunction(2,0,0),
              BasisFunction(2,1,1),
              BasisFunction(2,1,-1),
              BasisFunction(2,1,0)],
        'H': [BasisFunction(1,0,0)]
    }
    mol.build_basis(molset)
    
    symops = SymmetryOperations(mol)
    symops.add('identity')
    symops.add('rotation', '3+', np.array([0,0,1]), 2.0 * np.pi / 3)
    symops.add('rotation', '3-', -np.array([0,0,1]), 2.0 * np.pi / 3)
    
    for i in range(0,3):
        symops.add('mirror', 'v1', np.array([np.cos(i * 2.0 * np.pi / 3),
                                             np.sin(i * 2.0 * np.pi / 3),
                                             0.0]))

    symops.run()
    
    visualize_matrices(symops.operation_matrices, 
                       [op.name for op in  symops.operations],
                       [bf.name for bf in symops.mol.basis], 
                       xlabelrot=90, figsize=(9,6))
    
    # print result LOT
    ct = CharacterTable('c3v')
    print(ct.lot(np.trace(symops.operation_matrices, axis1=1, axis2=2)))
    
    # apply projection operator
    po = ProjectionOperator(ct, symops)
    mos = po.build_mos()
    newmats = [mos @ m @ mos.transpose() for m in symops.operation_matrices]
    
    visualize_matrices(newmats, 
                       [op.name for op in  symops.operations],
                       ['$\phi_{%i}$' % (i+1) for i in range(len(symops.mol.basis))],
                       figsize=(9,6),
                       highlight_groups=po.get_blocks())

if __name__ == '__main__':
    main()