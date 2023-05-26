# -*- coding: utf-8 -*-

import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sphecerix import tesseral_wigner_D_axis_angle,wffield_l
import numpy as np
from pytessel import PyTessel

def main():
    # build rotation axis and set angle
    axis = np.ones(3) / np.sqrt(3)
    angle = np.pi
    
    # construct tesseral Wigner D matrix
    D = tesseral_wigner_D_axis_angle(3, axis, angle)
    Y = np.zeros(7)
    Y[3] = 1
    
    # calculate linear combination of spherical harmonics after rotation
    Yp = D @ Y 
    
    # construct scalar field for all spherical harmonics of l=2
    dim = 25
    dfields = wffield_l(4,3,dim,200)
    
    # construct isosurface construction library and set unit cell
    pytessel = PyTessel()
    unitcell = np.diag(np.ones(3) * dim * 2)
    isovalue = 0.005
    
    # build isosurface of dz2 orbital before transformation
    vertices, normals, indices = pytessel.marching_cubes(dfields[3].flatten(), dfields[3].shape, unitcell.flatten(), isovalue)
    pytessel.write_ply('fz3p.ply', vertices, normals, indices)
    vertices, normals, indices = pytessel.marching_cubes(dfields[3].flatten(), dfields[3].shape, unitcell.flatten(), -isovalue)
    pytessel.write_ply('fz3n.ply', vertices, normals, indices)
    
    # build isosurface of dz2 orbital after transformation
    ffield = np.einsum('i,ijkl->jkl', Yp, dfields)
    vertices, normals, indices = pytessel.marching_cubes(ffield.flatten(), ffield.shape, unitcell.flatten(), isovalue)
    pytessel.write_ply('rfz3p.ply', vertices, normals, indices)
    vertices, normals, indices = pytessel.marching_cubes(ffield.flatten(), ffield.shape, unitcell.flatten(), -isovalue)
    pytessel.write_ply('rfz3n.ply', vertices, normals, indices)
    
if __name__ == '__main__':
    main()