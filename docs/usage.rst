.. _usage:
.. index:: Usage

Usage
*****

.. note::
	:program:`sphecerix` uses the 
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