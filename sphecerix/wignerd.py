# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import factorial
from scipy.spatial.transform import Rotation as R
from .tesseral import tesseral_transformation
import warnings

def tesseral_wigner_D(l, Robj):
    """
    Produce the Wigner D-matrix for tesseral spherical harmonics

    Parameters
    ----------
    l : int
        Order of the spherical harmonics
    Robj : scipy.spatial.transform.Rotation
        Rotation in :math:`\mathbb{R}^{3}`

    Returns
    -------
    D : numpy.ndarray
        Real-valued Wigner-D matrix with dimensions :math:`(2l+1) \\times (2l+1)`

    Raises
    ------
    TypeError
        If the Robj object is not of type scipy.spatial.transform.R.

    Examples
    --------
    >>> from sphecerix import tesseral_wigner_D
    ... from scipy.spatial.transform import Rotation as R
    ... import numpy as np
    ...
    ... # build rotation axis and set angle
    ... axis = np.ones(3) / np.sqrt(3)
    ... angle = np.pi
    ... Robj = R.from_rotvec(axis * angle)
    ...
    ... # construct tesseral Wigner D matrix
    ... D = tesseral_wigner_D(2, Robj)
    ... print(D)
    [[ 5.55555556e-01  2.22222222e-01  7.69800359e-01  2.22222222e-01
       1.89744731e-16]
     [ 2.22222222e-01  5.55555556e-01 -3.84900179e-01  2.22222222e-01
       6.66666667e-01]
     [ 7.69800359e-01 -3.84900179e-01 -3.33333333e-01 -3.84900179e-01
       5.42310034e-16]
     [ 2.22222222e-01  2.22222222e-01 -3.84900179e-01  5.55555556e-01
      -6.66666667e-01]
     [-1.01229242e-16  6.66666667e-01 -4.65653372e-16 -6.66666667e-01
      -3.33333333e-01]]

    Construct the Wigner-D matrix for the tesseral p-orbitals for a rotation around
    the :math:`\\frac{1}{\\sqrt{3}}(1,1,1)` axis by an angle :math:`\\pi`.

    """
    # verify that Robj is a rotation object
    if not isinstance(Robj, R):
        raise TypeError('Second argument Robj should be of type scipy.spatial.transform.R')
    
    T = tesseral_transformation(l)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.')
        alpha, beta, gamma = Robj.as_euler('zyz', degrees=False)
        
    D = wigner_D(l, Robj)
    
    return np.real(T @ D @ T.conjugate().transpose())

def tesseral_wigner_D_mirror(l, normal):
    """
    Produce the Wigner D-matrix for tesseral spherical harmonics for a mirror operation

    Parameters
    ----------
    l : int
        Order of the spherical harmonics
    normal : np.array
        Normal vector

    Returns
    -------
    D : numpy.ndarray
        Real-valued Wigner-D matrix with dimensions :math:`(2l+1) \\times (2l+1)`

    Examples
    --------
    >>> from sphecerix import tesseral_wigner_D_mirror
    ... import numpy as np
    ... 
    ... # construct mirror normal vector
    ... normal = np.array([-1,1,0]) / np.sqrt(2)
    ... 
    ... # construct wigner D matrix
    ... D = tesseral_wigner_D_mirror(1, normal)
    ... 
    ... print(D)
    [[-2.83276945e-16 -0.00000000e+00  1.00000000e+00]
     [ 1.22464680e-16  1.00000000e+00  3.46914204e-32]
     [ 1.00000000e+00 -1.22464680e-16  2.83276945e-16]]

    Construct the Wigner-D matrix for the tesseral p-orbitals for a mirror
    operation with the mirror plane corresponding to the xz plane rotated
    around the z-axis by 45 degrees.

    """    
    # construct mirror matrix
    normal /= np.linalg.norm(normal) # ensure normalization
    M = np.identity(3) - 2 * np.outer(normal, normal)
    
    # decompose mirror operation into a rotation and an inversion
    Robj = R.from_matrix(-M)
    inv = (-1)**l
    
    T = tesseral_transformation(l)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.')
        alpha, beta, gamma = Robj.as_euler('zyz', degrees=False)
        
    D = wigner_D(l, Robj)
    
    return inv * np.real(T @ D @ T.conjugate().transpose())

def wigner_D(l, Robj):
    """
    Produce Wigner D-matrix for canonical spherical harmonics

    Parameters
    ----------
    l : int
        Order of the spherical harmonics
    Robj : scipy.spatial.transform.Rotation
        Rotation in :math:`\mathbb{R}^{3}`

    Returns
    -------
    D : numpy.ndarray
        Complex-valued Wigner-D matrix with dimensions :math:`(2l+1) \\times (2l+1)`

    Raises
    ------
    TypeError
        If the Robj object is not of type scipy.spatial.transform.R. 

    Examples
    --------
    >>> from sphecerix import wigner_D
    ... from scipy.spatial.transform import Rotation as R
    ... import numpy as np
    ...
    ... # build rotation axis and set angle
    ... axis = np.ones(3) / np.sqrt(3)
    ... angle = np.pi
    ... Robj = R.from_rotvec(axis * angle)
    ...
    ... # construct tesseral Wigner D matrix
    ... D = wigner_D(2, Robj)
    ... print(D)
    [[ 1.11111111e-01-1.45486986e-16j -2.22222222e-01+2.22222222e-01j
      -3.29266657e-16-5.44331054e-01j  4.44444444e-01+4.44444444e-01j
      -4.44444444e-01-4.42577444e-17j]
     [-2.22222222e-01-2.22222222e-01j  5.55555556e-01-3.33066907e-16j
      -2.72165527e-01+2.72165527e-01j -5.55111512e-17+2.22222222e-01j
      -4.44444444e-01-4.44444444e-01j]
     [ 3.83471103e-16+5.44331054e-01j -2.72165527e-01-2.72165527e-01j
      -3.33333333e-01+0.00000000e+00j  2.72165527e-01-2.72165527e-01j
       3.83471103e-16-5.44331054e-01j]
     [ 4.44444444e-01-4.44444444e-01j -5.55111512e-17-2.22222222e-01j
       2.72165527e-01+2.72165527e-01j  5.55555556e-01+3.33066907e-16j
       2.22222222e-01-2.22222222e-01j]
     [-4.44444444e-01+4.42577444e-17j -4.44444444e-01+4.44444444e-01j
      -3.29266657e-16+5.44331054e-01j  2.22222222e-01+2.22222222e-01j
       1.11111111e-01+1.45486986e-16j]]

    Construct the Wigner-D matrix for the canonical p-orbitals for a rotation around
    the :math:`\\frac{1}{\\sqrt{3}}(1,1,1)` axis by an angle :math:`\\pi`.

    """
    # verify that Robj is a rotation object
    if not isinstance(Robj, R):
        raise TypeError('Second argument Robj should be of type scipy.spatial.transform.R')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.')
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
