# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sphecerix import Molecule, BasisFunction, SymmetryOperations,\
                      visualize_matrices, CharacterTable, ProjectionOperator,\
                      plot_matrix

def main():
    mol = Molecule()
    mol.from_file('molecules/dodecahedrane.xyz')
    
    molset = {
        'C': [#BasisFunction(1,0,0),
              #BasisFunction(2,0,0),
              BasisFunction(2,1,1),
              BasisFunction(2,1,-1),
              BasisFunction(2,1,0)],
        'H': [BasisFunction(1,0,0)]
    }
    mol.build_basis(molset)
    
    symops = SymmetryOperations(mol)
    symops.add('identity')
    
    # Find C5 axes; for this specific implementation of the molecule, it is
    # given that indices [0-4] form a pentagon. The center of all pentagons lie
    # at the same distance with respec to the origin
    c5_axes = []
    d = np.linalg.norm(np.average(np.take([at[1] for at in mol.atoms], range(5), axis=0), axis=0))
    for k in combinations(range(20), 5):
        dd = np.linalg.norm(np.average(np.take([at[1] for at in mol.atoms], k, axis=0), axis=0))
        if np.abs(dd - d) < 1e-5:
            c5_axes.append(np.average(np.take([at[1] for at in mol.atoms], k, axis=0), axis=0))
            c5_axes[-1] /= np.linalg.norm(c5_axes[-1])
    
    for i,ax in enumerate(c5_axes):
        symops.add('rotation', '5,%i' % (i+1), ax, 2.0 * np.pi / 5)
    
    # create c5^2 
    for i,ax in enumerate(c5_axes):
        symops.add('rotation', '5^2,%i' % (i+1), ax, 4.0 * np.pi / 5)
    
    # create c3 axes
    for i,at in enumerate(mol.atoms[0:20]):
        axis = at[1] / np.linalg.norm(at[1])
        symops.add('rotation', '3,%i' % (i+1), axis, 2.0 * np.pi / 3)
        
    # Find C2 axes; all C2 axes lie at the vertices of two adjacent atoms. We can
    # adopt the same strategy as for the C5 axes, though it will yield exactly
    # double duplicates as similar rotational axes lie at opposite sites of the
    # icosphere. Note that we can also immediately use this procedure to find the
    # mirror planes
    d = np.linalg.norm(np.average(np.take([at[1] for at in mol.atoms], range(2), axis=0), axis=0))
    c2_axes = []
    mirror_normals = []
    for k in combinations(range(20), 2):
        dd = np.linalg.norm(np.average(np.take([at[1] for at in mol.atoms], k, axis=0), axis=0))
        if np.abs(dd - d) < 1e-5:
            axis = np.average(np.take([at[1] for at in mol.atoms], k, axis=0), axis=0)
            if axis[0] < 0: # prune duplicates
                continue
            c2_axes.append(axis)
            c2_axes[-1] /= np.linalg.norm(c2_axes[-1])
            
            mirror_normals.append(np.cross(mol.atoms[k[0]][1], mol.atoms[k[1]][1]))
            mirror_normals[-1] /= np.linalg.norm(mirror_normals[-1])
    
    # C2 axes
    for i,ax in enumerate(c2_axes):
        symops.add('rotation', '2,%i' % (i+1), ax, np.pi)
        
    symops.add('inversion')

    # S10 improper rotations
    for i,ax in enumerate(c5_axes):
        symops.add('improper', '10,%i' % (i+1), ax, 2.0 * np.pi / 10)
        
    # S10^3 improper rotations
    for i,ax in enumerate(c5_axes):
        symops.add('improper', '10^3,%i' % (i+1), ax, 6.0 * np.pi / 10)
    
    # S6 operations
    for i,at in enumerate(mol.atoms[0:20]):
        axis = at[1] / np.linalg.norm(at[1])
        symops.add('improper', '6,%i' % (i+1), axis, 2.0 * np.pi / 6)

    # sigma mirror planes
    for i,n in enumerate(mirror_normals):
        symops.add('mirror', 'd,%i' % (i+1), n)

    symops.run()
    
    # print result LOT
    ct = CharacterTable('ih')
    print(ct.lot(np.trace(symops.operation_matrices, axis1=1, axis2=2)))
    
    # apply projection operator
    po = ProjectionOperator(ct, symops)
    mos = po.build_mos(verbose=True)
    newmats = [mos @ m @ mos.transpose() for m in symops.operation_matrices]
    
    # construct some fancy labels to put above the plots
    plotlabels = []
    plotlabels.append(r'$E$')
    for i in range(0,12):
        plotlabels.append(r'$C_{5},(%i)$' % (i+1))
    for i in range(0,12):
        plotlabels.append(r'$C_{5}^{2},(%i)$' % (i+1))
    for i in range(0,20):
        plotlabels.append(r'$C_{3},(%i)$' % (i+1))
    for i in range(0,15):
        plotlabels.append(r'$C_{2},(%i)$' % (i+1))
    plotlabels.append(r'$i$')
    for i in range(0,12):
        plotlabels.append(r'$S_{10},(%i)$' % (i+1))
    for i in range(0,12):
        plotlabels.append(r'$S_{10}^{3},(%i)$' % (i+1))
    for i in range(0,20):
        plotlabels.append(r'$S_{6},(%i)$' % (i+1))
    for i in range(0,15):
        plotlabels.append(r'$\sigma,(%i)$' % (i+1))
    
    for op_idx in tqdm(range(len(plotlabels))):
        #print('Plotting %s' % plotlabels[op_idx])
        fig, ax = plt.subplots(1,1,dpi=144,figsize=(25,25))
        plot_matrix(ax, 
                    symops.operation_matrices[op_idx],
                    [bf.name for bf in symops.mol.basis],
                    plotlabels[op_idx],
                    titlefontsize=40)
        plt.tight_layout()
        plt.savefig('figures/matR_%03i.png' % op_idx)
        plt.close()
        
        fig, ax = plt.subplots(1,1,dpi=144,figsize=(25,25))
        plot_matrix(ax, newmats[op_idx],
                    ['$\phi_{%i}$' % (i+1) for i in range(len(symops.mol.basis))],
                    plotlabels[op_idx],
                    titlefontsize=40,
                    highlight_groups=po.get_blocks())
        plt.tight_layout()
        plt.savefig('figures/matTRTt_%03i.png' % op_idx)
        plt.close()

if __name__ == '__main__':
    main()