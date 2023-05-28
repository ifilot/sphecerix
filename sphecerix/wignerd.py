# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import factorial
from scipy.spatial.transform import Rotation as R
from .tesseral import tesseral_transformation

def tesseral_wigner_D(l, Robj):
    """
    Produce the Wigner D-matrix for tesseral spherical harmonics
    """
    # verify that Robj is a rotation object
    if not isinstance(Robj, R):
        raise TypeError('Second argument Robj should be of type scipy.spatial.transform.R')
    
    T = tesseral_transformation(l)
    alpha, beta, gamma = Robj.as_euler('zyz', degrees=False)
    D = wigner_D(l, Robj)
    
    return np.real(T @ D @ T.conjugate().transpose())

def wigner_D(l, Robj):
    """
    Produce Wigner D-matrix for order l of spherical harmonics and
    given axis angle rotation
    """
    # verify that Robj is a rotation object
    if not isinstance(Robj, R):
        raise TypeError('Second argument Robj should be of type scipy.spatial.transform.R')

    alpha, beta, gamma = Robj.as_euler('zyz', degrees=False)
    d = wigner_d(l, beta)
    m = np.arange(-l, l+1)
    diag_alpha = np.diag(np.exp(1j * m * alpha))
    diag_gamma = np.diag(np.exp(1j * m * gamma))
    
    return diag_gamma @ d @ diag_alpha
    
def wigner_d(l, beta):
    """
    Produce Wigner (small) d-matrix for order l of spherical harmonics and
    euler angles
    """
    d = np.zeros((2*l+1,2*l+1))
    for i,m1 in enumerate(range(-l,l+1)):
        for j,m2 in enumerate(range(-l,l+1)):
            d[i,j] = wigner_d_element_euler_angles(l, m1, m2, beta)
    
    return d

def wigner_d_element_euler_angles(l, m1, m2, beta):
    """
    Calculate single element in Wigner (small) d-matrix
    """
    smin = max(0, m2-m1)
    smax = min(l+m2, l-m1)
    
    pre = np.sqrt(factorial(l+m1) * \
                  factorial(l-m1) * \
                  factorial(l+m2) * \
                  factorial(l-m2))
    
    v = 0
    for s in range(smin, smax+1):
        num = (-1)**(m1-m2+s) * np.cos(beta/2)**(2*l+m2-m1-2*s) * \
                                np.sin(beta/2)**(m1-m2+2*s)
        denom = factorial(l + m2 - s) * factorial(s) * \
                factorial(m1 - m2 + s) * factorial(l - m1 - s)
            
        v += num / denom
        
    return pre * v
