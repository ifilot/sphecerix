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
    mol.add_atom('C',  0.0000000015, -1.3868467444, 0.0000000000, unit='angstrom')
    mol.add_atom('C',  1.2010445126, -0.6934233709, 0.0000000000, unit='angstrom')
    mol.add_atom('C',  1.2010445111,  0.6934233735, 0.0000000000, unit='angstrom')
    mol.add_atom('C', -0.0000000015,  1.3868467444, 0.0000000000, unit='angstrom')
    mol.add_atom('C', -1.2010445126,  0.6934233709, 0.0000000000, unit='angstrom')
    mol.add_atom('C', -1.2010445111, -0.6934233735, 0.0000000000, unit='angstrom')
    mol.add_atom('H',  0.0000000027, -2.4694205285, 0.0000000000, unit='angstrom')
    mol.add_atom('H',  2.1385809117, -1.2347102619, 0.0000000000, unit='angstrom')
    mol.add_atom('H',  2.1385809090,  1.2347102666, 0.0000000000, unit='angstrom')
    mol.add_atom('H', -0.0000000027,  2.4694205285, 0.0000000000, unit='angstrom')
    mol.add_atom('H', -2.1385809117,  1.2347102619, 0.0000000000, unit='angstrom')
    mol.add_atom('H', -2.1385809090, -1.2347102666, 0.0000000000, unit='angstrom')
    
    molset = {
        'C': [BasisFunction(2,0,0),
              # BasisFunction(2,1,1),  # x
              # BasisFunction(2,1,-1), # y
              BasisFunction(2,1,0)], # z
        'H': [BasisFunction(1,0,0)]
    }
    mol.build_basis(molset)
    
    symops = SymmetryOperations(mol)
    
    # E
    symops.add('identity')
    
    # 2C6
    symops.add('rotation', '6+', np.array([0,0,1]), 2.0 * np.pi / 6)
    symops.add('rotation', '6-', np.array([0,0,1]), -2.0 * np.pi / 6)
    
    # 2C3
    symops.add('rotation', '3+', np.array([0,0,1]), 2.0 * np.pi / 3)
    symops.add('rotation', '3-', np.array([0,0,1]), -2.0 * np.pi / 3)
    
    # C2
    symops.add('rotation', '2', np.array([0,0,1]), np.pi)
    
    # 3C2'
    for i in range(0,3):
        symops.add('rotation', '2,%i' % i, np.array([np.sin(2.0 * np.pi * i/6.),
                                                      -np.cos(2.0 * np.pi * i/6.),
                                                      0.0]), np.pi)
    
    # 3C2''
    for i in range(0,3):
        symops.add('rotation', '2,%i' % i, np.array([np.sin(2.0 * np.pi * (i/6. + 1./12)),
                                                      -np.cos(2.0 * np.pi * (i/6. + 1./12)),
                                                      0.0]), np.pi)
    
    
    # inversion
    symops.add('inversion')
    
    # 2S3
    symops.add('improper', '3+', np.array([0,0,1]), 2.0 * np.pi / 3)
    symops.add('improper', '3-', np.array([0,0,1]), -2.0 * np.pi / 3)
    
    # 2S6
    symops.add('improper', '6+', np.array([0,0,1]), 2.0 * np.pi / 6)
    symops.add('improper', '6-', np.array([0,0,1]), -2.0 * np.pi / 6)
    
    # sigma_h
    symops.add('mirror', 'h(xy)', np.array([0,0,1]))
    
    # sigma_d
    for i in range(0,3):
        symops.add('mirror', 'd,%i' % i, np.array([np.cos(2.0 * np.pi * (i/6. + 1./12)),
                                                   np.sin(2.0 * np.pi * (i/6. + 1./12)),
                                                   0.0]))
    
    # sigma_v
    for i in range(0,3):
        symops.add('mirror', 'v,%i' % i, np.array([np.cos(2.0 * np.pi * i/6),
                                                   np.sin(2.0 * np.pi * i/6),
                                                   0.0]))

    symops.run()
    
    visualize_matrices(symops.operation_matrices, 
                        [op.name for op in  symops.operations],
                        [bf.name for bf in symops.mol.basis], 
                        xlabelrot=90, figsize=(24,32), numcols=4)

    # print result LOT
    ct = CharacterTable('d6h')
    print(ct.lot(np.trace(symops.operation_matrices, axis1=1, axis2=2)))
    
    # apply projection operator
    po = ProjectionOperator(ct, symops)
    
    mos = po.build_mos(verbose=True)
    newmats = [mos @ m @ mos.transpose() for m in symops.operation_matrices]
    
    visualize_matrices(newmats, 
                        [op.name for op in  symops.operations],
                        ['$\phi_{%i}$' % (i+1) for i in range(len(symops.mol.basis))],
                        figsize=(24,32), numcols=4)

if __name__ == '__main__':
    main()