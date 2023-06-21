import unittest
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import tesseral_wigner_D_improper

class TestImproperRotation(unittest.TestCase):
    """
    Test improper rotation matrices
    """

    def test_improper_s4(self):
        """
        Test the result under an improper rotation
        """
        # construct rotation vector
        axis = np.array([1,0,0])
        Robj = R.from_rotvec(axis * np.pi / 2)
        
        # construct wigner D matrix
        D = tesseral_wigner_D_improper(1, Robj)
        
        # assert that determinant is -1
        np.testing.assert_almost_equal(np.linalg.det(D), -1)
        
        # rotate a point at +x,+y,+z under S4 to -x,-y,+z
        # note that the ordering in the vector (using increasing value of m)
        # is [y,z,x]
        np.testing.assert_almost_equal(D @ np.array([1,1,1]), np.array([-1,1,-1]))
        
if __name__ == '__main__':
    unittest.main()
