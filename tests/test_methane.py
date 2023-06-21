import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import tesseral_wigner_D, tesseral_wigner_D_mirror, \
                      tesseral_wigner_D_improper
from scipy.spatial.transform import Rotation as R

class TestMethane(unittest.TestCase):
    """
    Test the construction of tesseral transformation matrices for the
    p-orbitals on the C atom of methane (lying at the origin) 
    
    Methane has Td symmetry and the p-orbitals belong to the T2 symmetry group,
    thus we expect the traces for the transformation matrices under
    (C3, C2, S4, sigma_d) to correspond to (0,-1,-1,+1)
    """

    def test_rotation_c3(self):
        """
        Assert that the characters under C3 correspond to 0
        """
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
        """
        Assert that the characters under C2 correspond to -1
        """
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
            
    def test_improper_rotation_s4(self):
        """
        Assert that the characters under S4 correspond to -1
        """
        coords = self.build_coordinates_methane()
        angle = np.radians(90)  # S4 rotation
        
        for axis in coords[1:]:
            axis /= np.linalg.norm(axis)
            Robj = R.from_rotvec(np.array(axis) * angle)
            T = tesseral_wigner_D_improper(1, Robj)
            
            # test that determinant of the transformation matrix is -1.0
            # because the operation contains a reflection
            np.testing.assert_almost_equal(np.linalg.det(T), -1.0)
            
            # test the the matrix multiplied with its complex conjugate
            # equals the identity matrix
            np.testing.assert_almost_equal(T @ T.transpose(), np.identity(3))
                
            # test that trace equals the character under S4
            np.testing.assert_almost_equal(np.trace(T), -1)
            
    def test_mirror_sigma_d(self):
        """
        Assert that the characters under S4 correspond to 1
        """
        coords = self.build_coordinates_methane()
        
        for i in range(1,5):
            for j in range(i+1,5):
                axis = (coords[i] + coords[j]) / 2
                M = tesseral_wigner_D_mirror(1, axis)
                
                # test that determinant of the transformation matrix is -1.0
                # because the operation contains a reflection
                np.testing.assert_almost_equal(np.linalg.det(M), -1.0)
                
                # test the the matrix multiplied with its complex conjugate
                # equals the identity matrix
                np.testing.assert_almost_equal(M @ M.transpose(), np.identity(3))
                    
                # test that trace equals the character under sigma_d
                np.testing.assert_almost_equal(np.trace(M), 1)

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
        
        return np.array(coords)

if __name__ == '__main__':
    unittest.main()
