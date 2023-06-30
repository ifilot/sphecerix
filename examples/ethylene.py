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
    mol.add_atom('C', -0.6530176758,  0.0000000000 ,0.0000000000, unit='angstrom')
    mol.add_atom('C',  0.6530176758,  0.0000000000 ,0.0000000000, unit='angstrom')
    mol.add_atom('H', -1.2288875372, -0.9156191261 ,0.0000000000, unit='angstrom')
    mol.add_atom('H', -1.2288875372,  0.9156191261 ,0.0000000000, unit='angstrom')
    mol.add_atom('H',  1.2288875372,  0.9156191261 ,0.0000000000, unit='angstrom')
    mol.add_atom('H',  1.2288875372, -0.9156191261 ,0.0000000000, unit='angstrom')
    
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
    symops.add('rotation', '2(z)', np.array([0,0,1]), np.pi)
    symops.add('rotation', '2(y)', np.array([0,1,0]), np.pi)
    symops.add('rotation', '2(x)', np.array([1,0,0]), np.pi)
    symops.add('inversion')
    symops.add('mirror', 'v(xy)', np.array([0,0,1]))
    symops.add('mirror', 'v(xz)', np.array([0,1,0]))
    symops.add('mirror', 'v(yz)', np.array([1,0,0]))

    symops.run()
    
    visualize_matrices(symops.operation_matrices, 
                       [op.name for op in  symops.operations],
                       [bf.name for bf in symops.mol.basis], 
                       xlabelrot=90, figsize=(18,10), numcols=4)

    # print result LOT
    ct = CharacterTable('d2h')
    print(ct.lot(np.trace(symops.operation_matrices, axis1=1, axis2=2)))
    
    # apply projection operator
    po = ProjectionOperator(ct, symops)
    mos = po.build_mos()
    newmats = [mos @ m @ mos.transpose() for m in symops.operation_matrices]
    
    visualize_matrices(newmats, 
                       [op.name for op in  symops.operations],
                       ['$\phi_{%i}$' % (i+1) for i in range(len(symops.mol.basis))],
                       figsize=(18,10), numcols=4)

if __name__ == '__main__':
    main()