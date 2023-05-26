# -*- coding: utf-8 -*-

import numpy as np

def tesseral_transformation(l):
    """
    Produce tesseral transformation matrix for order l
    """
    T = np.zeros((2*l+1,2*l+1), dtype=np.complex128)
    invsq2r = 1/np.sqrt(2)
    invsq2i = 1j * invsq2r
    
    for i,m1 in enumerate(range(-l,l+1)):
        for j,m2 in enumerate(range(-l,l+1)):
            if np.abs(m1) != np.abs(m2):
                continue
            
            if m1 < 0:
                if m2 < 0:
                    T[i,j] = invsq2i
                elif m2 > 0:
                    T[i,j] = -(-1)**(m1) * invsq2i
            if m1 > 0:
                if m2 < 0:
                    T[i,j] = invsq2r
                elif m2 > 0:
                    T[i,j] = (-1)**(m1) * invsq2r
    
    T[l,l] = 1
    
    return T

def permutation_sh_car():
    """
    Grab the permutation matrix that transforms a 3D space from
    yzx ordering to xyz ordering. This is basically the permutation
    needed to convert the default orientation of the spherical harmonics
    of order l=1 to the conventional Cartesian axes.
    """
    return np.array([
        [0,1,0],
        [0,0,1],
        [1,0,0]
    ])