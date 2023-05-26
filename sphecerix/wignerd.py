# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import factorial
from .rotations import axis_angle_to_euler
from .tesseral import tesseral_transformation

def tesseral_wigner_D_axis_angle(l, axis, angle):
    """
    Produce the Wigner D-matrix for tesseral spherical harmonics
    """
    T = tesseral_transformation(l)
    alpha, beta, gamma = axis_angle_to_euler(axis, angle)
    D = wigner_D(l, alpha, beta, gamma)
    
    return np.real(T @ D @ T.conjugate().transpose())

def wigner_D_axis_angle(l, axis, angle):
    """
    Produce Wigner D-matrix for order l of spherical harmonics and
    given axis angle rotation
    """
    alpha, beta, gamma = axis_angle_to_euler(axis, angle)
    return wigner_D(l, alpha, beta, gamma)

def wigner_D(l, alpha, beta, gamma):
    """
    Produce Wigner D-matrix for order l of spherical harmonics and
    euler angles
    """
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
