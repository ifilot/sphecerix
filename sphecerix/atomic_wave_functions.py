# -*- coding: utf-8 -*-
import numpy as np
from math import factorial
from scipy.special import assoc_laguerre
from scipy.special import sph_harm

def wffield(n,l,m,d,npts):
    """
    Create discrete scalar field for wave function
    
    n : pritimive quantum number
    l : azimuthal quantum number
    m : magnetic quantum number
    d : half the edge length of the unit cell
    npts : number of data points in each cartesian direction
    
    The scalar field is constructed such that x is the fastest moving index
    and z the slowest moving index. The return object is a 3D-array.
    """
    x = np.linspace(-d,d,npts)
    zz,yy,xx = np.meshgrid(x,x,x, indexing='ij')
    
    field = wfcart(n,l,m,xx,yy,zz)
    
    return field.reshape(npts, npts, npts)

def wffield_l(n,l,d,npts):
    """
    Create discrete scalar field for all possible values of m for given
    set of n and l
    
    n : pritimive quantum number
    l : azimuthal quantum number
    d : half the edge length of the unit cell
    npts : number of data points in each cartesian direction
    
    The scalar field is constructed such that x is the fastest moving index
    and z the slowest moving index. The return object is a 3D-array.
    """
    fields = np.zeros((2*l+1, npts, npts, npts))
    
    for i,m in enumerate(range(-l, l+1)):
        fields[i,:,:,:] = wffield(n,l,m,d,npts)
    
    return fields

def wfcart(n,l,m,x,y,z):
    """
    Construct the wave function using Cartesian coordinates
    
    n : pritimive quantum number
    l : azimuthal quantum number
    m : magnetic quantum number
    """
    r = np.linalg.norm([x,y,z], axis=0)
    theta = np.arctan2(y,x)
    phi = np.arccos(z/r)

    return wf(n,l,m,r,theta,phi)

def wf(n,l,m,r,theta,phi):
    """
    Construct the wave function using spherical coordinates
    
    n : pritimive quantum number
    l : azimuthal quantum number
    m : magnetic quantum number
    r : radius
    theta : azimuthal angle
    phi : polar angle
    """
    return radial(n,l,r) * angular(l,m,theta,phi)
    
def angular(l,m,theta,phi):
    """
    Construct the angular part of the wave function
    
    l : azimuthal quantum number
    m : magnetic quantum number
    theta : azimuthal angle
    phi : polar angle
    """
    # see: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    #
    # this create so-called Tesseral spherical harmonics
    #
    if m == 0:
        return np.real(sph_harm(m,l,theta,phi))
    elif m < 0:
        return np.real(1j / np.sqrt(2) * (sph_harm(m,l,theta,phi) - (-1)**m * sph_harm(-m,l,theta,phi)))
    elif m > 0:
        return np.real(1 / np.sqrt(2) * (sph_harm(-m,l,theta,phi) + (-1)**m * sph_harm(m,l,theta,phi)))

def radial(n,l,r):
    """
    This is the formulation for the radial wave function as encountered in
    Griffiths "Introduction to Quantum Mechanics 3rd edition"
    
    n : pritimive quantum number
    l : azimuthal quantum number
    r : radius
    """
    n = int(n)
    l = int(l)
    a = 1.0
    rho = 2.0 * r / (n * a)
    val =  np.sqrt((2.0 / (n * a))**3) * \
           np.sqrt(factorial(n - l - 1) / (2 * n * factorial(n + l))) * \
           np.exp(-0.5 * rho) * \
           rho**l * \
           assoc_laguerre(rho, n-l-1, 2*l+1)

    return val
