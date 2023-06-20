import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import tesseral_wigner_D, tesseral_wigner_D_mirror
from scipy.spatial.transform import Rotation as R

class TestMethane(unittest.TestCase):
    """
    Test the construction of tesseral transformation matrices
    """

    def test_rotation_c3(self):
        coords = self.build_coordinates_methane()
        angle = np.radians(120)  # C3 rotation
        
        for axis in coords[1:]:
            axis /= np.linalg.norm(axis)
            Robj = R.from_rotvec(np.array(axis) * angle)
            T = tesseral_wigner_D(1, Robj)
            
            # test that determinant of the transformation matrix is 1.0
            # because the operation contains *no* reflection
            np.testing.assert_almost_equal(np.linalg.det(T), 1.0)
            
            # test the the matrix multiplied with its complex conjugate
            # equals the identity matrix
            np.testing.assert_almost_equal(T @ T.transpose(), np.identity(3))
            
            # test that trace equals the character under C3
            np.testing.assert_almost_equal(np.trace(T), 0)
            
    def test_rotation_c2(self):
        axes = [
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ]
        angle = np.radians(180) # C2 rotation
        
        for axis in axes:
            axis /= np.linalg.norm(axis)
            Robj = R.from_rotvec(np.array(axis) * angle)
            T = tesseral_wigner_D(1, Robj)
            
            # test that determinant of the transformation matrix is 1.0
            # because the operation contains *no* reflection
            np.testing.assert_almost_equal(np.linalg.det(T), 1.0)
            
            # test the the matrix multiplied with its complex conjugate
            # equals the identity matrix
            np.testing.assert_almost_equal(T @ T.transpose(), np.identity(3))
            
            # test that trace equals the character under C2
            np.testing.assert_almost_equal(np.trace(T), -1)
            
    def test_rotation_s4(self):
        coords = self.build_coordinates_methane()
        angle = np.radians(90)  # S4 rotation
        
        for axis in coords[1:]:
            axis /= np.linalg.norm(axis)
            Robj = R.from_rotvec(np.array(axis) * angle)
            T = tesseral_wigner_D(1, Robj)
            M = tesseral_wigner_D_mirror(1, axis)
            
            # test that determinant of the transformation matrix is -1.0
            # because the operation contains a reflection
            np.testing.assert_almost_equal(np.linalg.det(M @ T), -1.0)
            
            # test the the matrix multiplied with its complex conjugate
            # equals the identity matrix
            np.testing.assert_almost_equal((M @ T) @ (M @ T).transpose(), np.identity(3))
                
            # test that trace equals the character under C2
            np.testing.assert_almost_equal(np.trace(M @ T), -1)

    def build_coordinates_methane(self):
        """
        Construct fictitious methane coordinates
        """
        coords = [
            [0,0,0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
        
        return coords

if __name__ == '__main__':
    unittest.main()
