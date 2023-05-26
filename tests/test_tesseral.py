import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import tesseral_transformation

class TestTesseral(unittest.TestCase):
    """
    Test the construction of tesseral transformation matrices
    """

    def test_tesseral_l1(self):
        """
        Test tesseral transformation matrix for l=1
        """
        T = tesseral_transformation(1)
        invsq2r = 1/np.sqrt(2)
        invsq2i = 1j * invsq2r
        Tref = np.array([
            [invsq2i,0,invsq2i],
            [0,1,0],
            [invsq2r,0,-invsq2r]
        ])
        
        np.testing.assert_array_almost_equal_nulp(T, Tref)
        
    def test_tesseral_l2(self):
        """
        Test tesseral transformation matrix for l=2
        """
        T = tesseral_transformation(2)
        invsq2r = 1/np.sqrt(2)
        invsq2i = 1j * invsq2r
        Tref = np.array([
            [invsq2i,0,0,0,-invsq2i],
            [0,invsq2i,0,invsq2i,0],
            [0,0,1,0,0],
            [0,invsq2r,0,-invsq2r,0],
            [invsq2r,0,0,0,invsq2r]
        ])
        
        np.testing.assert_array_almost_equal_nulp(T, Tref)

if __name__ == '__main__':
    unittest.main()
