import unittest
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R
from sphecerix import tesseral_wigner_D_mirror

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import tesseral_transformation

class TestMirror(unittest.TestCase):
    """
    Test transformation among tesseral spherical harmonics for mirroring
    operations
    """

    def test_mirror_45_l1(self):
        """
        Test tesseral transformation matrix for l=1 given a reflection through
        a mirror at a 45 degree angle with respect to the xz plane
        """
        # construct mirror normal vector
        normal = np.array([-1,1,0]) / np.sqrt(2)
        
        # construct wigner D matrix
        D = tesseral_wigner_D_mirror(1, normal)
        
        np.testing.assert_almost_equal(D @ np.array([0,0,1]), np.array([1,0,0]))
        np.testing.assert_almost_equal(D @ np.array([0,1,0]), np.array([0,1,0]))
        
    def test_mirror_120_l1(self):
        """
        Test tesseral transformation matrix for l=1 given a reflection through
        a mirror at a 120 degree angle with respect to the xz plane
        """
        # construct mirror normal vector
        angle = np.radians(120)
        x = np.cos(angle)
        y = np.sin(angle)
        normal = np.array([x,y,0])
        
        # construct wigner D matrix
        D = tesseral_wigner_D_mirror(1, normal)
        
        np.testing.assert_almost_equal(D @ np.array([0,0,1]), np.array([np.sqrt(3)/2,0,0.5]))
        np.testing.assert_almost_equal(D @ np.array([0,1,0]), np.array([0,1,0]))
        
    def test_mirror_45_l2(self):
        """
        Test tesseral transformation matrix for l=2 given a reflection through
        a mirror at a 45 degree angle with respect to the xz plane
        """
        # construct mirror normal vector
        normal = np.array([0,1,0]) / np.sqrt(2)
        
        # construct wigner D matrix
        D = tesseral_wigner_D_mirror(2, normal)
    
        np.testing.assert_almost_equal(D @ np.array([0,0,1,0,0]), np.array([0,0,1,0,0]))
        np.testing.assert_almost_equal(D @ np.array([1,0,0,0,0]), np.array([-1,0,0,0,0]))

if __name__ == '__main__':
    unittest.main()
