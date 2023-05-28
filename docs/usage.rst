.. _usage:
.. index:: Usage

Usage
*****

Below an example is given for the rotation among the tesseral p-orbitals::

	from sphecerix import tesseral_wigner_D_axis_angle
	import numpy as np

	def main():
	    # build rotation axis and set angle
	    axis = np.ones(3) / np.sqrt(3)
	    angle = np.pi
	    
	    # construct tesseral Wigner D matrix
	    D = tesseral_wigner_D_axis_angle(2, axis, angle)
	    Y = np.zeros(5)
	    Y[2] = 1
	    
	    # calculate linear combination of the spherical harmonics after rotation
	    Yp = D @ Y
	    print(Yp)
	    
	if __name__ == '__main__':
	    main()