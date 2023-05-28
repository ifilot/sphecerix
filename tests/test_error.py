import unittest
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import wigner_D, tesseral_wigner_D

class TestError(unittest.TestCase):
    """
    Test whether parsing wrong class raises an error
    """

    def test_wigner_D_error(self):
        Robj = 'this is not a rotation object'
        with self.assertRaisesRegex(TypeError, expected_regex='Second argument Robj should be of type scipy.spatial.transform.R'):
            wigner_D(1, Robj)
        
    def test_tesseral_wigner_D_error(self):
        Robj = 'this is not a rotation object'
        with self.assertRaisesRegex(TypeError, expected_regex='Second argument Robj should be of type scipy.spatial.transform.R'):
            tesseral_wigner_D(1, Robj)
        
if __name__ == '__main__':
    unittest.main()
