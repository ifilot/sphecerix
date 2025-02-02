.. _usage:
.. index:: Usage

Usage
=====

:program:`Sphecerix` offers two functions for building Wigner-d matrices. One
can either use :func:`sphecerix.wignerd.wigner_D` or :func:`sphecerix.wignerd.tesseral_wigner_D` for the canonical and
tesseral spherical harmonics, respectively. These functions take two
arguments, the order ``l`` of the spherical harmonics and a `Scipy rotation
object <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html>`_.

The Wigner-D matrices are built such that the rows and columns run with
increasing value of :math:`m`. For example, for the :math:`d`-orbitals the
:math:`m` values run from :math:`m \in (-2,-1,0,1,2)`.

To calculate how a spherical harmonic or a linear combination of spherical
harmonics transforms under a given rotation, one simply multiplies the vector
representing the spherical harmonics by the Wigner-D matrix.

.. note::
	:program:`Sphecerix` uses the 
	`rotation object from Scipy to decribe <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html>`_
	the rotations.

Below an example is given for the rotation among the tesseral p-orbitals::

	from sphecerix import tesseral_wigner_D
	from scipy.spatial.transform import Rotation as R
	import numpy as np

	def main():
	    # build rotation axis and set angle
	    axis = np.ones(3) / np.sqrt(3)
	    angle = np.pi
	    Robj = R.from_rotvec(axis * angle)
	    
	    # construct tesseral Wigner D matrix
	    D = tesseral_wigner_D(2, Robj)
	    Y = np.zeros(5)
	    Y[2] = 1
	    
	    # calculate linear combination of the spherical harmonics after rotation
	    Yp = D @ Y
	    print(Yp)
	    
	if __name__ == '__main__':
	    main()